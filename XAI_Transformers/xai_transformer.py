# the code below stems originally from the paper 
# XAI for Transformers: Better Explanations through Conservative Propagation 
# significant adjustments have been made to the original code (safer pproc implementation, switching modes, setup of forward and explain)

from torch import nn
import torch
import torch.nn.functional as F
import sys
import os 
import copy
root_dir = './../'
## if it breaks code below is necessary 

parent_dir = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(parent_dir)
# print(parent_dir)
sys.path.append(root_dir)
from .utils import LayerNorm
from .attribution import compute_GAE_attr


class LNargs(object):
    
    def __init__(self):
        self.lnv = 'nowb'
        self.sigma = None
        self.hidden = None
        self.adanorm_scale = 1.
        self.nowb_scale = None
        self.mean_detach = False
        self.std_detach = False

class LNargsDetach(object):
    
    def __init__(self):
        self.lnv = 'nowb'
        self.sigma = None
        self.hidden = None
        self.adanorm_scale = 1.
        self.nowb_scale = None
        self.mean_detach = True
        self.std_detach = True
        
class LNargsDetachNotMean(object):
    
    def __init__(self):
        self.lnv = 'nowb'
        self.sigma = None
        self.hidden = None
        self.adanorm_scale = 1.
        self.nowb_scale = None
        self.mean_detach = False
        self.std_detach = True
       
def make_p_layer(layer, gamma):
    player = copy.deepcopy(layer)
    player.weight = torch.nn.Parameter(layer.weight+gamma*layer.weight.clamp(min=0))
    player.bias   = torch.nn.Parameter(layer.bias +gamma*layer.bias.clamp(min=0))
    return player
        
    
    
class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        self.first_token_tensor = hidden_states[:, 0]
        self.pooled_output1 = self.dense(self.first_token_tensor)
        self.pooled_output2 = self.activation(self.pooled_output1)
        return self.pooled_output2
        
class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.config = config
        
        if self.config.train_mode == True:
            self.dropout =  torch.nn.Dropout(p=0.1, inplace=False)

        
        if config.detach_layernorm == True:
            assert config.train_mode==False

            if config.detach_mean==False:
                # print('Detach LayerNorm only Norm')
                largs = LNargsDetachNotMean()
            else:
                # print('Detach LayerNorm Mean+Norm')
                largs = LNargsDetach()
        else:
            largs =  LNargs()
        
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, args=largs)
                
        self.detach = config.detach_layernorm

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)                
        if self.config.train_mode == True:
            hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor) 

        return hidden_states
    
    def pforward(self, hidden_states, input_tensor, gamma):
        pdense =  make_p_layer(self.dense, gamma)
        hidden_states = pdense(hidden_states)                        
        #hidden_states = self.dense(hidden_states)                
        if self.config.train_mode == True: # pforward only in train mode False hence this does not make sense 
            hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor) 

        return hidden_states


class AttentionBlock(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.query = nn.Linear(config.hidden_size, config.all_head_size)
        self.key = nn.Linear(config.hidden_size, config.all_head_size)
        self.value = nn.Linear(config.hidden_size, config.all_head_size)
        self.output = BertSelfOutput(config)
        self.detach = config.detach_kq 
        if self.config.train_mode == True:
            self.dropout =  torch.nn.Dropout(p=0.1, inplace=False)

        if self.detach == True:
            assert self.config.train_mode==False
            # print('Detach K-Q-softmax branch')

        

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients
        # print('Saved attention gradients with shape:', attn_gradients.shape)

    def get_attn_gradients(self):
        return self.attn_gradients
        
    def transpose_for_scores(self, x):
        # x torch.Size([1, 10, 768])
        # xout torch.Size([1, 10, 12, 64])        
        new_x_shape = x.size()[:-1] + (self.config.num_attention_heads, self.config.attention_head_size)
        x = x.view(*new_x_shape)
        X= x.permute(0, 2, 1, 3)
        return X
    
    def un_transpose_for_scores(self, x, old_shape):
        x = x.permute(0,  1, 2, 3)
        return x.reshape(old_shape)
    
    @staticmethod
    def pproc(layer, player, x):
        eps = 1e-12
        z  = layer(x)
        zp = player(x)
        # make denom nonzero (and finite) everywhere
        # denom = torch.where(zp.abs() < eps, torch.full_like(zp, eps), zp)
        denom = torch.where(
            zp.abs() < eps,
            torch.copysign(torch.full_like(zp, eps), zp),
            zp
        )
        return zp * (z / denom).detach()

        
        
    def forward(self, hidden_states, gamma=0 , method=None, attention_mask=None):
        
      #  print('PKQ gamma', gamma)
        # gamma is always zero so these are just the normal layers 
        pquery = make_p_layer(self.query, gamma)
        pkey = make_p_layer(self.key, gamma)
        pvalue = make_p_layer(self.value, gamma)
        
        n_nodes= hidden_states.shape[1]         

        
        if self.config.train_mode or gamma == 0: # gamma added later 
            query_ = self.query(hidden_states) 
            key_ = self.key(hidden_states) 
            val_ = self.value(hidden_states)
        else:
            query_ = self.pproc(self.query, pquery, hidden_states) 
            key_ = self.pproc(self.key, pkey, hidden_states)
            val_ = self.pproc(self.value, pvalue, hidden_states)

        
        query_t = self.transpose_for_scores(query_)
        key_t = self.transpose_for_scores(key_)
        val_t = self.transpose_for_scores(val_)        
        
        attention_scores = torch.matmul(query_t, key_t.transpose(-1, -2))
        
        if attention_mask is not None:
            # avoid attention for padding tokens 
            mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2)
            attention_scores = attention_scores.masked_fill(mask, torch.finfo(attention_scores.dtype).min)
     
        

        if self.detach:    
            assert self.config.train_mode==False
            attention_probs = nn.Softmax(dim=-1)(attention_scores).detach()
        
        else:
            attention_probs = nn.Softmax(dim=-1)(attention_scores)
            

        
        if self.config.train_mode:
            attention_probs = self.dropout(attention_probs)
        # if method == 'GAE':
        #     # print('Registering hook for attention gradients')
        #     attention_probs.register_hook(self.save_attn_gradients)

        context_layer = torch.matmul(attention_probs, val_t)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        old_context_layer_shape = context_layer.shape
        new_context_layer_shape = context_layer.size()[:-2] + (self.config.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        if self.config.train_mode or gamma == 0: # this call seemed to break it, since gamma is always 0, better to avoid - does not fix it 
            output = self.output(context_layer, hidden_states)
        else:
            output = self.output.pforward(context_layer, hidden_states, gamma=gamma)


        return output, attention_probs #, (attention_scores, hidden_states) #, query_t, key_t, val_t)
    


class BertAttention(nn.Module):
    def __init__(self, config, embeddings):
        super().__init__()

        n_blocks = config.n_blocks
        self.n_blocks=n_blocks
        self.embeds = embeddings
        
        self.config = config
        self.attention_layers = torch.nn.Sequential(*[AttentionBlock(config) for i in range(n_blocks)])
        # self.output = BertSelfOutput(config) # this is not being used, only in attention block
        
        self.pooler = BertPooler(config)
        self.classifier = nn.Linear(in_features=config.hidden_size, out_features=config.n_classes, bias=True)
        self.device = config.device
        
        self.attention_probs = {i: [] for i in range(n_blocks)}
        self.attention_debug = {i: [] for i in range(n_blocks)}
        self.attention_gradients = {i: [] for i in range(n_blocks)}
        self.attention_cams      = {i: [] for i in range(n_blocks)}

        self.attention_lrp_gradients = {i: [] for i in range(n_blocks)}
    
    def eval(self):
        self.train(mode=False) # standard pytorch 

    def explain(self):
        self.train(False) # important to not use _switch_train_mode here (?) 
        self.switch_detach(detach_kq=True, detach_lnorm_mean=False, detach_lnorm_std=True)

    def train(self, mode=True):
        # standard train method 
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)

        # additional bit 
        self._switch_train_mode(mode)
        if mode==True:
            self.switch_detach(detach_kq=False, detach_lnorm_mean=False, detach_lnorm_std=False) 
        
        return self # standard return

    def switch_detach(self, detach_kq, detach_lnorm_mean, detach_lnorm_std):
        if (detach_kq or detach_lnorm_mean or detach_lnorm_std) and self.config.train_mode==True:
            # print('Switching to eval mode for detaching')
            self.train(False)

        if detach_kq != self.config.detach_kq:
            self._switch_KQ_detach(detach_kq)
        if (detach_lnorm_mean != self.config.detach_mean) or (detach_lnorm_std != self.config.detach_layernorm):
            self._switch_LN_detach(detach_lnorm_mean, detach_lnorm_std)
        

    def _switch_LN_detach(self, detach_lnorm_mean, detach_lnorm_std):
        for block in self.attention_layers:
            block.output.detach = detach_lnorm_mean or detach_lnorm_std
            ln = block.output.LayerNorm
            ln.set_mean_detach(detach_lnorm_mean)
            ln.set_std_detach(detach_lnorm_std)
            block.output.LayerNorm = ln # not sure if this is necessary, better safe than editing a copy 
            block.output.config.detach_layernorm = detach_lnorm_mean or detach_lnorm_std
        
        # print('Detach LayerNorm mean:', detach_lnorm_mean, ' std:', detach_lnorm_std, ' for all blocks')

    def _switch_KQ_detach(self, detach_kq):
        for block in self.attention_layers:
            block.detach = detach_kq
            block.config.detach_kq = detach_kq
        # print('Detach KQ-softmax branch:', detach_kq, ' for all blocks')

    def _switch_train_mode(self, train_mode):
        self.config.train_mode = train_mode
        for block in self.attention_layers:
            block.config.train_mode = train_mode
            block.output.config.train_mode = train_mode
            
        
    def forward(self, input_ids, 
                      attention_mask=None,
                      token_type_ids=None, 
                      position_ids=None, 
                      inputs_embeds=None, 
                      labels=None,
                      past_key_values_length=0, expl_method=None, **kwargs):
        
        
        hidden_states = self.embeds(input_ids=input_ids, 
                                          token_type_ids=token_type_ids, 
                                          position_ids=None, 
                                          inputs_embeds=None, 
                                          past_key_values_length=0).to(self.config.device)


        attn_input = hidden_states
        for i,block in enumerate(self.attention_layers):
            
            output, attention_probs = block(attn_input, 
                                            attention_mask=attention_mask, 
                                            method=expl_method)
            
            self.attention_probs[i] = attention_probs
          #  self.attention_debug[i] = debug_data +  (output,)
            attn_input = output
            
            
        pooled = self.pooler(output)
        logits = self.classifier(pooled)
        
        self.output_= output
        self.pooled_ = pooled
        self.logits_=logits
        
        if labels is not None:
            loss =  torch.nn.CrossEntropyLoss()(logits,labels)
        else:
            loss = None
            
        return {'loss': loss, 'logits': logits}
    
    
    def prep_lrp_legacy(self, x):
        x = x.data
        x.requires_grad_(True) 
        return x
    
    def prep_lrp(self, x):
        x = x.detach().requires_grad_(True)
        # x.requires_grad_(True) 
        return x
    

    def forward_and_explain(self,
                            input_ids,
                            cl,
                            attention_mask=None,
                            token_type_ids=None,
                            position_ids=None,
                            inputs_embeds=None,
                            labels=None,
                            past_key_values_length=0,
                            gammas=None,
                            method=None,
                            keep_graph_for_expl=False, **kwargs):
        """
        Returns:
        {'loss': base CE/BCE loss (or None), 'logits': [B,C], 'R': [B,T]}
        keep_graph_for_expl=True -> create_graph=True so the explanation MSE can
        backprop into model parameters.
        """
        
        
        A = {}

        # ---- Embeddings ----
        hidden_states = self.embeds(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length
        )
        A["hidden_states"] = hidden_states

        attn_input = hidden_states

        # ---- Transformer blocks (stash leaf inputs and live outputs) ----
        leaf_ids = []
        for i, block in enumerate(self.attention_layers):
            x_leaf = attn_input.detach().requires_grad_(True)         
            A[f"attn_in_leaf_{i}"] = x_leaf
            if method == 'LRP' or i == 0 or keep_graph_for_expl:
                # for LRP we need live input at first layer too
                inp = attn_input # attn_input = attn_input.detach().requires_grad_(True)
            else:
                inp = x_leaf  # later blocks use leaf inputs
            
            y_live, attention_probs = block(inp,
                                            gamma=(0.0 if gammas is None else gammas[i]),
                                            method=method, attention_mask=attention_mask)
            self.attention_probs[i] = attention_probs
            A[f"out_live_{i}"] = y_live                                
            A[f"attn_in_live_{i}"] = attn_input                        
            attn_input = y_live
            leaf_ids.append(i)

        # ---- Pool + classify (also keep a leaf before pool) ----
        # x_pooler_leaf   = attn_input.detach().requires_grad_(True)
        x_pooler_live = y_live  # live input to pooler
        pooled = self.pooler(x_pooler_live)
        # pooled_leaf = pooled.detach().requires_grad_(True)
        logits     = self.classifier(pooled)
        A["logits"] = logits

        # ---- Base supervised loss ----
        if labels is not None:
            base_loss = F.cross_entropy(logits, labels)
        else:
            base_loss = None

        # ---- Pick target logit (scalar) for explanations ----
        if torch.is_tensor(cl):
            Rout = logits.gather(1, cl.view(-1, 1)).sum()
        else:
            Rout = logits[:, cl].sum()
        self.R0 = Rout.detach().cpu().numpy()

        g_pooled = torch.autograd.grad(Rout, pooled, retain_graph=True, create_graph=keep_graph_for_expl)[0]
        
        if not keep_graph_for_expl:
            # double backprop trick to get gradients that flow via LN etc.
            g_pooled = g_pooled.detach()
 
        g_out = torch.autograd.grad(outputs=pooled, inputs=x_pooler_live,
                                    grad_outputs=g_pooled, retain_graph=True, create_graph=keep_graph_for_expl)[0]
        R_   = g_out * attn_input # actually its final attention 
        if not keep_graph_for_expl:
            g_out = g_out.detach()
       

        # Grad×Input relevance at last block output
       
        g_cur = g_out

        self.attention_gradients = {}

        for i in reversed(leaf_ids):
            y_live = A[f"out_live_{i}"]
            x_leaf = A[f"attn_in_leaf_{i}"]
            x_live = A[f"attn_in_live_{i}"]
            A_i    = self.attention_probs[i]

            # inp = x_live if i == 0 else x_leaf
            if method == 'GAE':
                
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

                # store attention gradient for this layer
                self.attention_gradients[i] = g_A
            else:
                g_in = torch.autograd.grad(outputs=R_.sum(), 
                                        inputs=x_live,
                                        retain_graph=True,
                                        create_graph=keep_graph_for_expl)[0]
                
                if not keep_graph_for_expl:
                    # print('detaching LRP grads')
                    g_in = g_in.detach()

            # Grad×Input relevance via x_live, **not** x_leaf - keep graph connected 
                R_   = g_in * x_live #x_live
                g_cur = g_in     # propagate upstream gradient

        if method == 'LRP':
            attribution = R_.sum(dim=-1)      # [B, T]
        elif method == 'GAE':
            # now self.attention_probs and self.attention_gradients
            # are filled for all layers using targeted gradients
            attribution = compute_GAE_attr(self)  
        
        
        return {'loss': base_loss, 'logits': logits, 'R': attribution}
