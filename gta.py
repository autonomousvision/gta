import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
import torch.nn.functional as F
from timm.layers import use_fused_attn
from torch.jit import Final
from einops import rearrange
from functools import partial
from itertools import repeat
import collections.abc
from rmsnorm import RMSNorm


def make_2dcoord(H, W, normalize=False):
    """
    Return(torch.Tensor): 2d coord values of shape [H, W, 2] 
    """
    x = np.arange(H, dtype=np.float32)   # [0, H)
    y = np.arange(W, dtype=np.float32)   # [0, W)
    if normalize:
        x = x / H
        y = y / W
    x_grid, y_grid = np.meshgrid(x, y, indexing='ij')
    return torch.Tensor(np.stack([x_grid.flatten(), y_grid.flatten()], -1).reshape(H, W, 2))


def make_SO2mats(coord, nfreqs):
    """
    Args:
      coord: [..., 2 or 3]
      freqs: [n_freqs, 2 or 3]
    Return:
      mats of shape [..., n_freqs, (2 or 3), 2, 2]
    """
    dim = coord.shape[-1]
    b = 10000.0
    freqs = torch.exp(torch.arange(0., 2*nfreqs, 2) *
                      -(math.log(b) / (2*nfreqs)))
    grid_ths = [torch.einsum(
        '...i,j->...ij', coord[..., d:d+1], freqs).flatten(-2, -1) for d in range(dim)]

    _mats = [[torch.cos(grid_ths[d]), -torch.sin(grid_ths[d]),
              torch.sin(grid_ths[d]), torch.cos(grid_ths[d])] for d in range(dim)]
    mats = [rearrange(torch.stack(_mats[d], -1),
                      '... (h w)->... h w', h=2, w=2) for d in range(dim)]
    mat = torch.stack(mats, -3)
    return mat

# GTA
@torch.jit.script
def rep_mul_x(rep, x):
    #  rep.shape=[T, F, 2, 2], x.shape=[B, H, T, F*2]
    shape = x.shape
    return (rep[None, None] * (x.unflatten(-1, (-1, 2))[..., None, :])).sum(-1).view(shape)


@torch.jit.script
def rep_mul_qkv(rep, q, k, v):
    return rep_mul_x(rep, q), rep_mul_x(rep, k), rep_mul_x(rep, v)


@torch.jit.script
def rep_mul_qk(rep, q, k):
    return rep_mul_x(rep, q), rep_mul_x(rep, k)


class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = RMSNorm,
            gta: bool = False,
            resolutions=[16, 16],
            v_transform: bool = True,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.gta = gta
        self.v_transform = v_transform

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if gta:
            F = self.head_dim // 4
            coord = make_2dcoord(resolutions[0], resolutions[1])
            self.so2rep = make_SO2mats(coord, F).flatten(
                2, 3).flatten(0, 1)  # [h*w, d, 2, 2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        # RoPE or GTA. Apply ρ^-1.
        if self.gta:
            rep = self.so2rep.to(x.device)  # [T, F, 2, 2] or [T, F, 2, 2, 2]
            if self.v_transform:  # GTA
                q, k, v = rep_mul_qkv(rep, q, k, v)
            else:  # RoPE
                q, k = rep_mul_qk(rep, q, k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        # GTA. Apply ρ.
        if self.gta and self.v_transform:
            x = rep_mul_x(rep.transpose(-2, -1), x)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
