# the code below stems originally from the paper 
# XAI for Transformers: Better Explanations through Conservative Propagation 
# minor adjustments made to implement switching detachments 

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch
import torch.nn as nn
from torch.nn import functional as F
from .attribution import softmax
from .normalization import DetachableLayerNorm
import os
import numpy as np
from .xai_albert import AlbertDetachableSdpaAttention, AlbertDetachableAttention, AlbertForSequenceClassificationXAI
from transformers.models.albert.modeling_albert import AlbertSdpaAttention, AlbertAttention

def set_up_dir(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


class LayerNormImpl(nn.Module):
    __constants__ = ['weight', 'bias', 'eps']


    def __init__(self, args, hidden, eps=1e-5, elementwise_affine=True):
        super(LayerNormImpl, self).__init__()
        self.mode = args.lnv
        self.sigma = args.sigma
        self.hidden = hidden
        self.adanorm_scale = args.adanorm_scale
        self.nowb_scale = args.nowb_scale
        self.mean_detach = args.mean_detach
        self.std_detach = args.std_detach
        if self.mode == 'no_norm' or 'nowb':
            elementwise_affine = False
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(hidden))
            self.bias = nn.Parameter(torch.Tensor(hidden))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)
    
    def set_mean_detach(self, mean_detach):
        self.mean_detach = mean_detach
    
    def set_std_detach(self, std_detach):
        self.std_detach = std_detach
    

    def forward(self, input):
        if self.mode == 'no_norm':
            return input
        elif self.mode == 'topk':
            T, B, C = input.size()
            input = input.reshape(T*B, C)
            k = max(int(self.hidden * self.sigma), 1)
            input = input.view(1, -1, self.hidden)
            topk_value, topk_index = input.topk(k, dim=-1)
            topk_min_value, top_min_index = input.topk(k, dim=-1, largest=False)
            top_value = topk_value[:, :, -1:]
            top_min_value = topk_min_value[:, :, -1:]
            d0 = torch.arange(top_value.shape[0], dtype=torch.int64)[:, None, None]
            d1 = torch.arange(top_value.shape[1], dtype=torch.int64)[None, :, None]
            input[d0, d1, topk_index] = top_value
            input[d0, d1, top_min_index] = top_min_value
            input = input.reshape(T, B, self.hidden)
            return F.layer_norm(
                input, torch.Size([self.hidden]), self.weight, self.bias, self.eps)
        elif self.mode == 'adanorm':
            mean = input.mean(-1, keepdim=True)
            std = input.std(-1, keepdim=True)
            input = input - mean
            mean = input.mean(-1, keepdim=True)
            graNorm = (1 / 10 * (input - mean) / (std + self.eps)).detach()
            input_norm = (input - input * graNorm) / (std + self.eps)
            return input_norm*self.adanorm_scale
        elif self.mode == 'nowb':
            'using nowb'
            mean = input.mean(dim=-1, keepdim=True)
            std = input.std(dim=-1, keepdim=True)
            if self.mean_detach:
                mean = mean.detach()
            if self.std_detach:
                std = std.detach()
            input_norm = (input - mean) / (std + self.eps)
            return input_norm

        elif self.mode == 'distillnorm':
            mean = input.mean(dim=-1, keepdim=True)
            std = input.std(dim=-1, keepdim=True)
            if self.mean_detach:
                mean = mean.detach()
            if self.std_detach:
                std = std.detach()
            input_norm = (input - mean) / (std + self.eps)

            input_norm = input_norm*self.weight + self.bias

            return input_norm

        elif self.mode == 'gradnorm':
            mean = input.mean(dim=-1, keepdim=True)
            std = input.std(dim=-1, keepdim=True)
            input_norm = (input - mean) / (std + self.eps)
            output = input.detach() + input_norm
            return output


def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, export=False, args=None):
    if args is not None:
        if args.lnv != 'origin':
            return LayerNormImpl(args, normalized_shape, eps, elementwise_affine)
    if not export and torch.cuda.is_available():
        try:
            from apex.normalization import FusedLayerNorm
            return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
        except ImportError:
            pass
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)

def flip(model, x, token_ids, tokens, y_true,  fracs, flip_case,random_order = False, tokenizer=None, device='cpu'):

    x = np.array(x)

    UNK_IDX = tokenizer.unk_token_id
    inputs0 = torch.tensor(token_ids).to(device)

    y0 = model(inputs0, labels = None)['logits'].squeeze().detach().cpu().numpy()
    orig_token_ids = np.copy(token_ids.detach().cpu().numpy())

    if random_order==False:
        if  flip_case=='generate':
            inds_sorted = np.argsort(x)[::-1]
        elif flip_case=='pruning':
            inds_sorted =  np.argsort(np.abs(x))
        else:
            raise
    else:

        inds_ = np.array(list(range(x.shape[-1])))
        remain_inds = np.array(inds_)
        np.random.shuffle(remain_inds)

        inds_sorted = remain_inds

    inds_sorted = inds_sorted.copy()
    vals = x[inds_sorted]

    mse = []
    evidence = []
    model_outs = {'sentence': tokens, 'y_true':y_true.detach().cpu().numpy(), 'y0':y0}

    N=len(x)

    evolution = {}
    for frac in fracs:
        inds_generator = iter(inds_sorted)
        n_flip=int(np.ceil(frac*N))
        inds_flip = [next(inds_generator) for i in range(n_flip)]

        if flip_case == 'pruning':

            inputs = inputs0
            for i in inds_flip:
                inputs[:,i] = UNK_IDX

        elif flip_case == 'generate':
            inputs = UNK_IDX*torch.ones_like(inputs0)
            # Set pad tokens
            inputs[inputs0==0] = 0

            for i in inds_flip:
                inputs[:,i] = inputs0[:,i]

        y = model(inputs, labels =  torch.tensor([y_true]*len(token_ids)).long().to(device))['logits'].detach().cpu().numpy()
        y = y.squeeze()

        err = np.sum((y0-y)**2)
        mse.append(err)
        evidence.append(softmax(y)[int(y_true)])

      #  print('{:0.2f}'.format(frac), ' '.join(tokenizer.convert_ids_to_tokens(inputs.detach().cpu().numpy().squeeze())))
        evolution[frac] = (inputs.detach().cpu().numpy(), inds_flip, y)

    if flip_case == 'generate' and frac == 1.:
        assert (inputs0 == inputs).all()


    model_outs['flip_evolution']  = evolution
    return mse, evidence, model_outs


def _is_layernorm_like(m: nn.Module):
    # True for nn.LayerNorm
    if isinstance(m, nn.LayerNorm):
        return True
    # True for any class named "LayerNormImpl" (even if it's from a different module version)
    if type(m).__name__ == "LayerNormImpl":
        return True
    # Heuristic: looks like LN (has eps & either normalized_shape/hidden/weight shape)
    if hasattr(m, "eps"):
        if hasattr(m, "normalized_shape"):
            return True
        if hasattr(m, "hidden"):
            return True
        if getattr(m, "weight", None) is not None:
            return True
    return False

def _get_normalized_shape(m: nn.Module):
    if hasattr(m, "normalized_shape"):
        ns = m.normalized_shape
        # nn.LayerNorm stores as int/tuple; normalize to tuple
        return tuple(ns) if isinstance(ns, (list, tuple)) else (ns,)
    if hasattr(m, "hidden"):
        return (int(m.hidden),)
    w = getattr(m, "weight", None)
    b = getattr(m, "bias", None)
    if w is not None:
        return tuple(w.shape)
    if b is not None:
        return (b.numel(),)
    # Worst case: cannot infer; default to 1 (should basically never happen)
    return (1,)

def _get_eps(m: nn.Module, default=1e-5):
    return getattr(m, "eps", default)

def _get_elementwise_affine(m: nn.Module):
    if hasattr(m, "elementwise_affine"):
        return bool(m.elementwise_affine)
    # Fallback: if it has learnable weight/bias assume True
    w = getattr(m, "weight", None)
    b = getattr(m, "bias", None)
    return (w is not None) or (b is not None)

def _get_mode(m: nn.Module):
    return getattr(m, "mode", 'nowb')

def swap_layernorms_with_detachable(module):
    """
    Recursively swap LayerNorm-like modules with DetachableLayerNorm.
    Preserves weights/bias and basic hyperparams (normalized_shape, eps, affine).
    Detachments set to False by default; can be changed later via model.set_detach_layernorm.
    """
    for name, child in list(module.named_children()):
        # Recurse first
        if name == 'embeds' or name == 'embeddings': # to match XAI implementation
            continue
        swap_layernorms_with_detachable(child)

        if _is_layernorm_like(child):
            print(f"Swapping LayerNorm-like: {name} ({child.__class__.__name__}) in {module.__class__.__name__}")

            normalized_shape = _get_normalized_shape(child)
            eps = _get_eps(child)
            elementwise_affine = _get_elementwise_affine(child)
            layer_mode = getattr(child, "mode", None) # Only for legacy bertattention model 

            new_ln = DetachableLayerNorm(
                normalized_shape=normalized_shape,
                eps=eps,
                elementwise_affine=elementwise_affine,
                mean_detach=False,
                std_detach=False,
                mode=layer_mode,
            )

            # copy weights/bias if present
            with torch.no_grad():
                if elementwise_affine:
                    # child might have None weights; guard each
                    if getattr(child, "weight", None) is not None:
                        new_ln.weight.copy_(child.weight)
                    if getattr(child, "bias", None) is not None:
                        new_ln.bias.copy_(child.bias)

            setattr(module, name, new_ln)
        else:
            # Optional: keep quiet if too noisy
            # print(f"Not a LayerNorm: {child.__class__.__name__}")
            pass

    return module


def swap_attention_with_detachable(model: nn.Module):
    """
    Recursively swap all AlbertAttention and AlbertSdpaAttention modules with
    AlbertDetachableAttention and AlbertDetachableSdpaAttention modules.
    Takes over all original parameters and settings like affinity.
    Detachments set to False by default; can be changed later via model.set_detach_kq.
    Args:
        model (nn.Module): the model to modify in-place
    """
    # Walk ALBERT blocks and swap attention
    for lg in model.albert.encoder.albert_layer_groups:
        for layer in lg.albert_layers:
            old = layer.attention
            if isinstance(old, AlbertSdpaAttention):
                new = AlbertDetachableSdpaAttention(model.config)
                # copy all parameters/buffers
                new.load_state_dict(old.state_dict())
                layer.attention = new
            elif isinstance(old, AlbertAttention):
                new = AlbertDetachableAttention(model.config)
                # copy all parameters/buffers
                new.load_state_dict(old.state_dict())
                layer.attention = new

    return model


def load_xai_albert(model_name, device='cpu', mean_detach=False, std_detach=False, run_id=None):
    if run_id is not None:
        base_path = '/vol/csedu-nobackup/project/anonuser/results/'

        chkpt_path = os.path.join(base_path, run_id)
        if not os.path.isdir(chkpt_path):
            raise ValueError(f"Checkpoint path {chkpt_path} does not exist.")
        chkpt_dir = os.listdir(chkpt_path)[0]
        model_name = os.path.join(chkpt_path, chkpt_dir) # just use dir and pretrained - untested!
        # raise NotImplementedError("Loading from run_id is untested, double check before usage.")
    model = AlbertForSequenceClassificationXAI.from_pretrained(model_name, num_labels=2)

    model = swap_layernorms_with_detachable(model)
    model = swap_attention_with_detachable(model)

    return model
