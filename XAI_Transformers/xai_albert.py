import math 
import torch 
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.albert import AlbertConfig
from transformers import AlbertForSequenceClassification
from transformers.pytorch_utils import prune_linear_layer, find_pruneable_heads_and_indices
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa
from transformers.utils import logging
from typing import Optional, Union
from .normalization import DetachableLayerNorm
from .attribution import compute_GAE_attr


logger = logging.get_logger(__name__)


class AlbertDetachableAttention(nn.Module):
    
    def __init__(self, config: AlbertConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads}"
            )

        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.attention_dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.output_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = DetachableLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pruned_heads = set()

        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)
        self.detach = False # added detach option, False by default 

    def prune_heads(self, heads: list[int]) -> None:
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.num_attention_heads, self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.query = prune_linear_layer(self.query, index)
        self.key = prune_linear_layer(self.key, index)
        self.value = prune_linear_layer(self.value, index)
        self.dense = prune_linear_layer(self.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.num_attention_heads = self.num_attention_heads - len(heads)
        self.all_head_size = self.attention_head_size * self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: bool = False,
    ) -> Union[tuple[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        batch_size, seq_length, _ = hidden_states.shape
        query_layer = self.query(hidden_states)
        key_layer = self.key(hidden_states)
        value_layer = self.value(hidden_states)
        query_layer = query_layer.view(batch_size, -1, self.num_attention_heads, self.attention_head_size).transpose(
            1, 2
        )
        key_layer = key_layer.view(batch_size, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        value_layer = value_layer.view(batch_size, -1, self.num_attention_heads, self.attention_head_size).transpose(
            1, 2
        )
        # print('using eager implementation')

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        if self.detach:
            attention_probs = attention_probs.detach()  # detach attention scores if specified

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attention_dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.transpose(2, 1).flatten(2)

        projected_context_layer = self.dense(context_layer)
        projected_context_layer_dropout = self.output_dropout(projected_context_layer)
        layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
        # print('using eager implementation')
        return (layernormed_context_layer, attention_probs) if output_attentions else (layernormed_context_layer,)
    

class AlbertDetachableSdpaAttention(AlbertDetachableAttention):
    # mostly copied from the OG version 
    # results are minimally different i.e. 6-8 decimal places the same 
    def __init__(self, config):
        super().__init__(config)
        self.dropout_prob = config.attention_probs_dropout_prob

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: bool = False,
    ) -> Union[tuple[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        # print('using sdpa implementation')
        if self.position_embedding_type != "absolute" or output_attentions: #or self.detach: # added 
            # logger.warning(
            #     "AlbertSdpaAttention is used but `torch.nn.functional.scaled_dot_product_attention` does not support "
            #     "non-absolute `position_embedding_type` or `output_attentions=True`  or detachment of the attention. Falling back to "
            #     "the eager attention implementation, but specifying the eager implementation will be required from "
            #     "Transformers version v5.0.0 onwards. This warning can be removed using the argument "
            #     '`attn_implementation="eager"` when loading the model.'
            # ) 
            return super().forward(hidden_states, attention_mask, output_attentions=output_attentions)

        batch_size, seq_len, _ = hidden_states.size()
        query_layer = (
            self.query(hidden_states)
            .view(batch_size, -1, self.num_attention_heads, self.attention_head_size)
            .transpose(1, 2)
        )
        key_layer = (
            self.key(hidden_states)
            .view(batch_size, -1, self.num_attention_heads, self.attention_head_size)
            .transpose(1, 2)
        )
        value_layer = (
            self.value(hidden_states)
            .view(batch_size, -1, self.num_attention_heads, self.attention_head_size)
            .transpose(1, 2)
        )

        if self.detach: # produces same results as eager impl. up to 5 decimal places
            query_layer = query_layer.detach()
            key_layer = key_layer.detach()

        attention_output = torch.nn.functional.scaled_dot_product_attention(
            query=query_layer,
            key=key_layer,
            value=value_layer,
            attn_mask=attention_mask,
            dropout_p=self.dropout_prob if self.training else 0.0,
            is_causal=False,
        )

        attention_output = attention_output.transpose(1, 2)
        attention_output = attention_output.reshape(batch_size, seq_len, self.all_head_size)

        projected_context_layer = self.dense(attention_output)
        projected_context_layer_dropout = self.output_dropout(projected_context_layer)
        layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
        return (layernormed_context_layer,)




class AlbertForSequenceClassificationXAI(AlbertForSequenceClassification):
    """
    HF ALBERT with a forward_and_explain pass that mimics the leaf/VJP flow you posted.
    Works with attn_implementation='sdpa' and custom detachable attention/LayerNorm.
    """
    def __init__(self, config):
        super().__init__(config)
        self.attention_probs = {i: [] for i in range(self.config.num_hidden_layers)}   
        self.R0 = None
        self.config.detach_layernorm = False 
        self.config.detach_kq = False
        self.config.detach_mean = False # not sure if its used 

    
    def forward_and_explain(
        self,
        input_ids=None,
        cl=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        past_key_values_length: int = 0,
        gammas=None,    # both unused for now
        method=None, 
        keep_graph_for_expl: bool = True,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = None,
    ):
        """
        Returns:
          {'loss': base CE/BCE loss (or None), 'logits': [B,C], 'R': [B,T]}
        """
        # taken from HF AlbertModel forward()
        assert (input_ids is not None) or (inputs_embeds is not None), "Provide input_ids or inputs_embeds."

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            if hasattr(self.albert.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.albert.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        # end taking 

        # embeddings like Bert 
        hidden_states = self.albert.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )  # [B, T, H]

        A = {}
        A["embeddings"] = hidden_states

        # back to copying from HF forward() (with minor edits)
        use_sdpa_attention_mask = ( # might want to add detach here too
            self.albert.attn_implementation == "sdpa"
            and self.albert.position_embedding_type == "absolute"
            and head_mask is None
            and not output_attentions
        )

        if use_sdpa_attention_mask:
            extended_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                attention_mask, hidden_states.dtype, tgt_len=seq_length
            )
        else:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(self.dtype).min

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        
        # implementation is in AlbertTransformer 
        # so far only copied from transformer forward()
        hidden_states = self.albert.encoder.embedding_hidden_mapping_in(hidden_states)

        all_hidden_states = (hidden_states,) if output_hidden_states else None
        all_attentions = () if output_attentions else None

        head_mask = [None] * self.albert.encoder.config.num_hidden_layers if head_mask is None else head_mask
        global_layer_step = 0
        leaf_ids = []
        for i in range(self.config.num_hidden_layers): # below can be simplified --> ther is only one group 
            # Number of layers in a hidden group
            layers_per_group = int(self.albert.encoder.config.num_hidden_layers / self.albert.encoder.config.num_hidden_groups)

            # Index of the hidden group
            group_idx = int(i / (self.albert.encoder.config.num_hidden_layers / self.albert.encoder.config.num_hidden_groups))
            
            group = self.albert.encoder.albert_layer_groups[group_idx]
            head_mask_group = head_mask[group_idx * layers_per_group : (group_idx + 1) * layers_per_group]
            # start copy from albertlayergroup forward()
            layer_hidden_states = ()
            layer_attentions = ()
            for layer_index, albert_layer in enumerate(group.albert_layers): # these are the equivalent of an AttentionBlock
                # hidden_states = attn_inp
                x_leaf = hidden_states.detach().requires_grad_(True)          # leaf (probe point)
                A[f"attn_in_leaf_{global_layer_step}"] = x_leaf
                if method == 'LRP' or global_layer_step == 0 or keep_graph_for_expl:
                    inp = hidden_states
                else:
                    inp = x_leaf
                # inp = hidden_states if global_layer_step == 0 or method == 'LRP' else x_leaf
                out = albert_layer(inp, 
                    extended_attention_mask, 
                    head_mask_group[layer_index], 
                    output_attentions=output_attentions,
                    output_hidden_states=False, #output_attentions
                    )
 
                y_live = out[0]                                               
                # hidden_states = y_live 
              
                A[f"out_live_{global_layer_step}"] = y_live    
                A[f"attn_in_live_{global_layer_step}"] = hidden_states
                hidden_states = y_live

                # could be deleted: if you want that, use forward() ? 
                if output_attentions:
                    layer_attention = out[1]
                    self.attention_probs[global_layer_step] = layer_attention
                    layer_attentions = layer_attentions + (layer_attention,)

                if output_hidden_states:
                    layer_hidden_states = layer_hidden_states + (hidden_states,)
                leaf_ids.append(global_layer_step) # there is only one group so this should be straightforward 
                global_layer_step += 1

            layer_group_output = (hidden_states,)
            # could be deleted: 
            if output_hidden_states:
                layer_group_output = layer_group_output + (layer_hidden_states,)
            if output_attentions:
                layer_group_output = layer_group_output + (layer_attentions,)
            # return outputs  # last-layer hidden state, (layer hidden states), (layer attentions)
            # end copy from albertlayergroup forward() --> then they repeat it in transformer and deal with type of output 

            # back to copying transformer block 
            hidden_states = layer_group_output[0]

            if output_attentions:
                all_attentions = all_attentions + layer_group_output[-1]

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            # end transformer block --> left out how it is being returned 
        # sequence_output = encoder_outputs[0] # this is the last hidden state 
        out_live = hidden_states  
        # below is based on AlbertModel forward() 
        pooled = self.albert.pooler_activation(self.albert.pooler(out_live[:, 0])) if self.albert.pooler is not None else None
  
        # omitting the dropout here as it should not be used for explain 
        logits     = self.classifier(pooled) # change to live 
        A["logits"] = logits

        if labels is not None: # simplifying the whole loss flow (see sequence classification forward())
            base_loss = F.cross_entropy(logits, labels)
        else:
            base_loss = None

        if torch.is_tensor(cl):
            Rout = logits.gather(1, cl.view(-1, 1)).sum()
        else:
            Rout = logits[:, cl].sum()
        self.R0 = Rout.detach().cpu().numpy()

        # ---- Step 1: get g_pooled = dRout/d pooled_leaf ----
        g_pooled = torch.autograd.grad(Rout, 
                                       pooled, 
                                       retain_graph=True, 
                                       create_graph=keep_graph_for_expl)[0]
        if not keep_graph_for_expl:
            g_pooled = g_pooled.detach()

        # VJP through pooler to out_leaf
        g_out = torch.autograd.grad(outputs=pooled, 
                                    inputs=out_live,
                                    grad_outputs=g_pooled, 
                                    retain_graph=True, 
                                    create_graph=keep_graph_for_expl)[0]

        # Grad×Input relevance at last block output
        R_   = g_out * out_live          # <-- keep attn_input NOT detached
        if not keep_graph_for_expl:
            g_out = g_out.detach()
        g_cur = g_out 
                          

        # Walk blocks backward
        self.attention_gradients = {}
        for i in reversed(leaf_ids):
            y_live = A[f"out_live_{i}"] 
            x_leaf = A[f"attn_in_leaf_{i}"]
            x_live = A[f"attn_in_live_{i}"]

            if method == 'GAE':
                A_i = self.attention_probs[i]
                inp = x_live if i == 0 or keep_graph_for_expl else x_leaf
                g_in, g_A = torch.autograd.grad(
                    outputs=y_live,
                    inputs=(inp, A_i),
                    grad_outputs=g_cur,
                    retain_graph=True,
                    create_graph=keep_graph_for_expl
                )

                if not keep_graph_for_expl:
                    g_in = g_in.detach()
                    g_A  = g_A.detach()
                self.attention_gradients[i] = g_A
            else:
                g_in = torch.autograd.grad(outputs=R_.sum(),
                                        inputs=x_live,
                                        retain_graph=True,
                                        create_graph=keep_graph_for_expl)[0]

                if not keep_graph_for_expl:
                    g_in = g_in.detach()


            R_   = g_in * x_live           # gradients flow via x_live only
            g_cur = g_in                   # keep pushing as a *detached* vector
        if method == 'GAE':
            attribution = compute_GAE_attr(self)
        else:
            attribution = R_.sum(dim=-1)

        
        return {'loss': base_loss, 'logits': logits, 'R': attribution}
    
    def explain(self):
        self.train(False) # important to not use _switch_train_mode here (?) 
        self.switch_detach(detach_kq=True, detach_lnorm_mean=False, detach_lnorm_std=True)

    def train(self, mode: bool = True):
        # copied 
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)

        if mode==True: # turning off explain if it's runnign 
            self.switch_detach(detach_kq=False, detach_lnorm_mean=False, detach_lnorm_std=False) 
        
        return self # standard return

    def switch_detach(self, detach_kq=True, detach_lnorm_mean=False, detach_lnorm_std=True):
        """
        Sets the .detach attribute of all instances of
        AlbertDetachableAttention, AlbertDetachableSdpaAttention, and DetachableLayerNorm
        in the model to the given values.
        """
        self._switch_detach_KQ(detach_kq)
        self._switch_detach_LayerNorm(detach_mean=detach_lnorm_mean, detach_std=detach_lnorm_std)

    def _switch_detach_KQ(self, detach_kq=True):
        """
        Sets the .detach attribute of all instances of AlbertDetachableAttention
        in the model to the given value.
        """
        self.config.detach_kq = detach_kq
        for module in self.modules():
            if isinstance(module, (AlbertDetachableAttention, AlbertDetachableSdpaAttention)):
                module.detach = detach_kq

    def _switch_detach_LayerNorm(self, detach_mean=False, detach_std=True):
        """
        Sets the .detach attribute of all instances of DetachableLayerNorm
        in the model to the given values.
        """
        self.config.detach_layernorm = detach_mean or detach_std
        self.config.detach_mean = detach_mean
        for module in self.modules():
            if isinstance(module, DetachableLayerNorm):
                module.detach_mean = detach_mean
                module.detach_std = detach_std
        


    
