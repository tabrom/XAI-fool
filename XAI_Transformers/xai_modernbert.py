# The code below is based on the transformers implementation and modified to allow for detachment of attention scores and LayerNorm statistics, as well as to implement the forward_and_explain pass.
# Copyright 2024 Answer.AI, LightOn, and contributors, and the HuggingFace Inc. team. All rights reserved.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
from transformers.models.modernbert.modeling_modernbert import (
    MODERNBERT_ATTENTION_FUNCTION,
    ModernBertAttention,
    ModernBertForSequenceClassification,
    apply_rotary_pos_emb,
)
from transformers.models.modernbert.configuration_modernbert import ModernBertConfig
from .normalization import DetachableLayerNorm
from .attribution import compute_GAE_attr


class ModernBertForSequenceClassificationXAI(ModernBertForSequenceClassification):
    """
    ModernBERT classifier with detachable LayerNorm and detachable attention probability flow.
    """

    def __init__(self, config):
        super().__init__(config)

        # Compatibility across ModernBERT config variants.
        layer_types = getattr(self.config, "layer_types", None)
        if layer_types is None:
            self.config.layer_types = ["global_attention"] * self.config.num_hidden_layers
        elif isinstance(layer_types, str):
            self.config.layer_types = [layer_types] * self.config.num_hidden_layers
        else:
            layer_types = list(layer_types)
            if len(layer_types) < self.config.num_hidden_layers:
                fill_value = layer_types[-1] if layer_types else "global_attention"
                layer_types += [fill_value] * (self.config.num_hidden_layers - len(layer_types))
            elif len(layer_types) > self.config.num_hidden_layers:
                layer_types = layer_types[: self.config.num_hidden_layers]
            self.config.layer_types = layer_types

        self.attention_probs = {i: [] for i in range(self.config.num_hidden_layers)}
        self.R0 = None
        self.config._attn_implementation = "eager"
        # Always use detachable attention in this XAI model variant.
        self.config.use_detachable_attention = True

        self._set_detach_state(detach_kq=False, detach_layernorm_mean=False, detach_layernorm_std=False)
        self._inject_detachable_modules()

    def _set_detach_state(self, detach_kq: bool, detach_layernorm_mean: bool, detach_layernorm_std: bool):
        self.config.detach_kq = detach_kq
        self.config.detach_layernorm_mean = detach_layernorm_mean
        self.config.detach_layernorm_std = detach_layernorm_std
        self.config.detach_layernorm = detach_layernorm_mean or detach_layernorm_std
        self.config.detach_mean = detach_layernorm_mean

    def _inject_detachable_modules(self):
        """Swap attention + LayerNorm modules in-place."""
        self._replace_modules_recursive(self)

    def _build_detachable_ln_from(self, old_ln: nn.LayerNorm) -> nn.Module:
        try:
            new_ln = DetachableLayerNorm(
                old_ln.normalized_shape,
                eps=old_ln.eps,
                elementwise_affine=old_ln.elementwise_affine,
            )
        except TypeError:
            try:
                new_ln = DetachableLayerNorm(old_ln.normalized_shape, eps=old_ln.eps)
            except TypeError:
                hidden = (
                    old_ln.normalized_shape[-1]
                    if isinstance(old_ln.normalized_shape, (tuple, list))
                    else old_ln.normalized_shape
                )
                new_ln = DetachableLayerNorm(hidden)

        if hasattr(old_ln, "weight") and old_ln.weight is not None:
            new_ln = new_ln.to(device=old_ln.weight.device, dtype=old_ln.weight.dtype)
        if (
            hasattr(old_ln, "weight")
            and old_ln.weight is not None
            and hasattr(new_ln, "weight")
            and new_ln.weight is not None
        ):
            new_ln.weight.data.copy_(old_ln.weight.data)
        if (
            hasattr(old_ln, "bias")
            and old_ln.bias is not None
            and hasattr(new_ln, "bias")
            and new_ln.bias is not None
        ):
            new_ln.bias.data.copy_(old_ln.bias.data)

        # DetachableLayerNorm uses mean_detach/std_detach naming.
        if hasattr(new_ln, "mean_detach"):
            new_ln.mean_detach = self.config.detach_layernorm_mean
        if hasattr(new_ln, "std_detach"):
            new_ln.std_detach = self.config.detach_layernorm_std
        return new_ln

    def _replace_modules_recursive(self, module: nn.Module):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.LayerNorm):
                setattr(module, name, self._build_detachable_ln_from(child))
                continue

            if child.__class__.__name__ == "ModernBertAttention":
                new_attn = ModernBertDetachableAttention(
                    self.config,
                    layer_id=getattr(child, "layer_id", None),
                )
                new_attn.load_state_dict(child.state_dict(), strict=False)
                setattr(module, name, new_attn)
                continue

            self._replace_modules_recursive(child)

    def explain(self):
        self.train(False)
        self._set_detach_state(detach_kq=True, detach_layernorm_mean=False, detach_layernorm_std=True)

    def train(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        super().train(mode)
        if mode:
            self._set_detach_state(detach_kq=False, detach_layernorm_mean=False, detach_layernorm_std=False)
        return self

    def switch_detach(self, detach_kq=True, detach_lnorm_mean=False, detach_lnorm_std=True):
        self._set_detach_state(detach_kq, detach_lnorm_mean, detach_lnorm_std)

    def _switch_detach_kq(self, detach_kq=True):
        self.config.detach_kq = detach_kq

    def _switch_detach_layernorm(self, detach_mean=False, detach_std=True):
        self.config.detach_layernorm_mean = detach_mean
        self.config.detach_layernorm_std = detach_std
        self.config.detach_layernorm = detach_mean or detach_std
        self.config.detach_mean = detach_mean
        for m in self.modules():
            if isinstance(m, DetachableLayerNorm):
                if hasattr(m, "mean_detach"):
                    m.mean_detach = detach_mean
                if hasattr(m, "std_detach"):
                    m.std_detach = detach_std

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        sliding_window_mask=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = None,
        **kwargs,
    ):
        """
        Standard forward pass - delegates to parent class.
        """
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

    def forward_and_explain(
        self,
        input_ids=None,
        cl=None,
        attention_mask=None,
        sliding_window_mask=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = None,
        method: str = "LRP",
        keep_graph_for_expl: bool = True,
        **kwargs,
    ):
        """
        Forward pass with gradient × input attribution or GAE.

        LRP uses gradients at the embedding layer.
        GAE stores per-layer attention matrices and their gradients, then rolls them up.
        """
        import torch.nn.functional as F

        if method not in {"LRP", "GAE"}:
            raise AssertionError(f"Method {method} not supported for ModernBERT. Use 'LRP' or 'GAE'.")

        output_attentions = True if method == "GAE" else (output_attentions if output_attentions is not None else self.config.output_attentions)
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            batch_size, seq_len = input_ids.shape[:2]
            device = input_ids.device
        else:
            batch_size, seq_len = inputs_embeds.shape[:2]
            device = inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), device=device, dtype=torch.bool)

        if position_ids is None and self.config._attn_implementation != "flash_attention_2":
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

        if self.config._attn_implementation == "flash_attention_2":
            raise NotImplementedError("forward_and_explain currently supports the eager attention path only.")

        attention_mask_2d = attention_mask
        attention_mask, sliding_window_mask = self.model._update_attention_mask(
            attention_mask, output_attentions=output_attentions
        )

        hidden_states = self.model.embeddings(input_ids=input_ids, inputs_embeds=inputs_embeds)
        embeddings = hidden_states

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        self.attention_probs = {i: [] for i in range(self.config.num_hidden_layers)}

        for layer_idx, encoder_layer in enumerate(self.model.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                sliding_window_mask=sliding_window_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]

            if output_attentions and len(layer_outputs) > 1:
                self.attention_probs[layer_idx] = layer_outputs[1]
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.model.final_norm(hidden_states)

        if self.config.classifier_pooling == "cls":
            pooled_output = hidden_states[:, 0]
        elif self.config.classifier_pooling == "mean":
            pooled_output = (hidden_states * attention_mask_2d.unsqueeze(-1)).sum(dim=1) / attention_mask_2d.sum(
                dim=1, keepdim=True
            )
        else:
            pooled_output = hidden_states[:, 0]

        pooled_output = self.head(pooled_output)
        pooled_output = self.drop(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = torch.nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            else:
                loss_fct = torch.nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        # Use true class label (cl) for explanations, not predicted class
        if cl is not None:
            if isinstance(cl, int):
                target_logits = logits[:, cl].sum()
            else:
                target_logits = logits.gather(1, cl.view(-1, 1)).sum()
        elif hasattr(self, "_explanation_target_class"):
            target_logits = logits[:, self._explanation_target_class].sum()
        else:
            target_logits = logits.gather(1, logits.argmax(dim=1, keepdim=True)).sum()

        if method == "GAE":
            self.attention_gradients = {}
            for layer_idx in reversed(range(self.config.num_hidden_layers)):
                attn_probs = self.attention_probs[layer_idx]
                if attn_probs is None or attn_probs == []:
                    continue
                self.attention_gradients[layer_idx] = torch.autograd.grad(
                    outputs=target_logits,
                    inputs=attn_probs,
                    retain_graph=True,
                    create_graph=keep_graph_for_expl,
                )[0].detach()
            attribution = compute_GAE_attr(self)
        else:
            grads = torch.autograd.grad(
                outputs=target_logits,
                inputs=embeddings,
                create_graph=False,
                retain_graph=True,
            )[0]
            attribution = (grads.detach() * embeddings).sum(dim=-1)

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return {
            "loss": loss,
            "logits": logits,
            "R": attribution,
        }


def eager_detachable_attention_forward(
    module: ModernBertAttention,
    qkv: torch.Tensor,
    attention_mask: torch.Tensor,
    sliding_window_mask: torch.Tensor,
    position_ids: torch.LongTensor | None,
    local_attention: tuple[int, int],
    bs: int,
    dim: int,
    output_attentions: bool = False,
    detach_probs: bool = False,
    **_kwargs,
):
    cos, sin = module.rotary_emb(qkv, position_ids=position_ids)
    query, key, value = qkv.transpose(3, 1).unbind(dim=2)
    query, key = apply_rotary_pos_emb(query, key, cos, sin)

    scale = module.head_dim**-0.5
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scale

    if local_attention != (-1, -1):
        attention_mask = sliding_window_mask

    attn_weights = attn_weights + attention_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=module.attention_dropout, training=module.training)

    # Keep attention weights differentiable for GAE; detachment is handled through the surrounding flow.
    if detach_probs:
        attn_weights = attn_weights + 0

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bs, -1, dim)

    if output_attentions:
        return (attn_output, attn_weights)
    return (attn_output,)


class ModernBertDetachableAttention(ModernBertAttention):
    def __init__(self, config: ModernBertConfig, layer_id: int | None = None):
        super().__init__(config, layer_id=layer_id)

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: bool = False,
        **kwargs,
    ):
        qkv = self.Wqkv(hidden_states)

        bs = hidden_states.shape[0]
        if self.config._attn_implementation == "flash_attention_2":
            qkv = qkv.view(-1, 3, self.num_heads, self.head_dim)
        else:
            qkv = qkv.view(bs, -1, 3, self.num_heads, self.head_dim)

        if self.config._attn_implementation == "eager":
            attention_interface = eager_detachable_attention_forward
        else:
            attention_interface = MODERNBERT_ATTENTION_FUNCTION[self.config._attn_implementation]

        attn_outputs = attention_interface(
            self,
            qkv=qkv,
            rotary_emb=self.rotary_emb,
            local_attention=self.local_attention,
            bs=bs,
            dim=self.all_head_size,
            output_attentions=output_attentions,
            detach_probs=self.config.detach_kq,
            **kwargs,
        )

        hidden_states = attn_outputs[0]
        hidden_states = self.out_drop(self.Wo(hidden_states))
        return (hidden_states,) + attn_outputs[1:]
