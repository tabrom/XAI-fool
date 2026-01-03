import torch
import torch.nn as nn
import torch.nn.functional as F

class DetachableLayerNorm(nn.Module):
    __constants__ = ['normalized_shape', 'eps', 'elementwise_affine']
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, 
                 mean_detach=False, std_detach=False, mode='nowb'):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.mean_detach = mean_detach
        self.std_detach = std_detach

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(*self.normalized_shape))
            self.bias   = nn.Parameter(torch.zeros(*self.normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.mode = mode

    @torch.no_grad()
    def set_detach_stats(self, mean_detach: bool, std_detach: bool):
        self.mean_detach = mean_detach
        self.std_detach  = std_detach

    def set_mean_detach(self, mean_detach: bool):
        self.mean_detach = mean_detach

    def set_std_detach(self, std_detach: bool):
        self.std_detach = std_detach

    def forward(self, x):
        # Compute per-feature stats along the last normalized dims
        # Match PyTorch LayerNorm: population variance (unbiased=False)
        dims = tuple(range(-len(self.normalized_shape), 0))
        mean = x.mean(dim=dims, keepdim=True)
        var  = x.var(dim=dims, unbiased=False, keepdim=True)

        if self.mean_detach:
            mean = mean.detach()
        if self.std_detach:
            # detach std by detaching var before sqrt (either way is fine)
            var = var.detach()

        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        if self.elementwise_affine and self.weight is not None and self.mode != 'nowb':
            x_hat = x_hat * self.weight + self.bias
        return x_hat
