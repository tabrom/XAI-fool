# The code below is based on the transformers implementation and modified to allow for detachment of attention scores and LayerNorm statistics, as well as to implement the forward_and_explain pass.
# Copyright 2019-present, the HuggingFace Inc. team, The Google AI Language Team and Facebook, Inc.
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

import math

import torch
import torch.nn as nn
from transformers.models.distilbert.modeling_distilbert import (
    DistilBertForSequenceClassification,
    MultiHeadSelfAttention,
)

from .attribution import compute_GAE_attr
from .normalization import DetachableLayerNorm


class DistilBertDetachableAttention(MultiHeadSelfAttention):
    """
    DistilBERT attention with optional K/Q gradient detachment.

    The detach mode preserves forward values but blocks gradient flow through K and Q.
    """

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
        head_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        bs, q_length, dim = query.size()
        k_length = key.size(1)
        dim_per_head = self.dim // self.n_heads
        mask_reshp = (bs, 1, 1, k_length)

        def shape(x: torch.Tensor) -> torch.Tensor:
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x: torch.Tensor) -> torch.Tensor:
            return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)

        q = shape(self.q_lin(query))
        k = shape(self.k_lin(key))
        v = shape(self.v_lin(value))

        # Preserve forward activations while zeroing gradient flow through K and Q.
        if getattr(self.config, "detach_kq", False):
            q = q.detach() #+ (q - q.detach()) * 0.0
            k = k.detach() #+ (k - k.detach()) * 0.0

        q = q / math.sqrt(dim_per_head)
        scores = torch.matmul(q, k.transpose(2, 3))

        mask = (mask == 0).view(mask_reshp).expand_as(scores)
        scores = scores.masked_fill(mask, torch.tensor(torch.finfo(scores.dtype).min, device=scores.device))

        weights = nn.functional.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        if head_mask is not None:
            weights = weights * head_mask

        context = torch.matmul(weights, v)
        context = unshape(context)
        context = self.out_lin(context)

        if output_attentions:
            return (context, weights)
        return (context,)


class DistilBertForSequenceClassificationXAI(DistilBertForSequenceClassification):
    """
    DistilBERT classifier with detachable LayerNorm and detachable attention K/Q flow.
    """

    def __init__(self, config):
        super().__init__(config)

        self.attention_probs = {i: [] for i in range(self.config.num_hidden_layers)}
        self.attention_gradients = {i: [] for i in range(self.config.num_hidden_layers)}
        self.R0 = None

        # Keep deterministic behavior with attention outputs available for GAE.
        self.config._attn_implementation = "eager"
        self.distilbert._use_flash_attention_2 = False
        self.distilbert._use_sdpa = False

        self._set_detach_state(detach_kq=False, detach_layernorm_mean=False, detach_layernorm_std=False)
        self._inject_detachable_modules()

    def _set_detach_state(self, detach_kq: bool, detach_layernorm_mean: bool, detach_layernorm_std: bool):
        self.config.detach_kq = detach_kq
        self.config.detach_layernorm_mean = detach_layernorm_mean
        self.config.detach_layernorm_std = detach_layernorm_std
        self.config.detach_layernorm = detach_layernorm_mean or detach_layernorm_std
        self.config.detach_mean = detach_layernorm_mean

    def _inject_detachable_modules(self):
        self._replace_modules_recursive(self)

    def _build_detachable_ln_from(self, old_ln: nn.LayerNorm) -> nn.Module:
        new_ln = DetachableLayerNorm(
            old_ln.normalized_shape,
            eps=old_ln.eps,
            elementwise_affine=old_ln.elementwise_affine,
        )

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

            if isinstance(child, MultiHeadSelfAttention):
                new_attn = DistilBertDetachableAttention(self.config)
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
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
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
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = None,
        method: str = "LRP",
        keep_graph_for_expl: bool = True,
        **kwargs,
    ):
        if method not in {"LRP", "GAE"}:
            raise AssertionError(f"Method {method} not supported for DistilBERT. Use 'LRP' or 'GAE'.")

        output_attentions = True if method == "GAE" else (output_attentions if output_attentions is not None else self.config.output_attentions)
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            device = input_ids.device
        else:
            input_shape = inputs_embeds.size()[:-1]
            device = inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)

        head_mask = self.distilbert.get_head_mask(head_mask, self.config.num_hidden_layers)

        embeddings = self.distilbert.embeddings(input_ids=input_ids, input_embeds=inputs_embeds)

        transformer_output = self.distilbert.transformer(
            x=embeddings,
            attn_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_state = transformer_output[0]
        pooled_output = hidden_state[:, 0]
        pooled_output = self.pre_classifier(pooled_output)
        pooled_output = nn.ReLU()(pooled_output)
        pooled_output = self.dropout(pooled_output)
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
            self.attention_probs = {i: [] for i in range(self.config.num_hidden_layers)}
            self.attention_gradients = {i: [] for i in range(self.config.num_hidden_layers)}

            all_attentions = transformer_output.attentions if return_dict else transformer_output[-1]
            for layer_idx, attn_probs in enumerate(all_attentions):
                self.attention_probs[layer_idx] = attn_probs

            for layer_idx in reversed(range(self.config.num_hidden_layers)):
                attn_probs = self.attention_probs[layer_idx]
                # if attn_probs is None or attn_probs == []:
                #     continue
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
