import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
from source.utils.nerf import get_vertical_rays
from source.utils.gta import multihead_geometric_transform_attention, multihead_vecrep_attention
import numpy as np

import math
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
__USE_DEFAULT_INIT__ = False

class JaxLinear(nn.Linear):
    """ Linear layers with initialization matching the Jax defaults """

    def reset_parameters(self):
        if __USE_DEFAULT_INIT__:
            super().reset_parameters()
        else:
            input_size = self.weight.shape[-1]
            std = math.sqrt(1/input_size)
            init.trunc_normal_(self.weight, std=std, a=-2.*std, b=2.*std)
            if self.bias is not None:
                init.zeros_(self.bias)


class ViTLinear(nn.Linear):
    """ Initialization for linear layers used by ViT """

    def reset_parameters(self):
        if __USE_DEFAULT_INIT__:
            super().reset_parameters()
        else:
            init.xavier_uniform_(self.weight)
            if self.bias is not None:
                init.normal_(self.bias, std=1e-6)


class SRTLinear(nn.Linear):
    """ Initialization for linear layers used in the SRT decoder """

    def reset_parameters(self):
        if __USE_DEFAULT_INIT__:
            super().reset_parameters()
        else:
            init.xavier_uniform_(self.weight)
            if self.bias is not None:
                init.zeros_(self.bias)


class PositionalEncoding(nn.Module):

    def __init__(self, num_octaves=8, start_octave=0):
        super().__init__()
        self.num_octaves = num_octaves
        self.start_octave = start_octave

        octaves = torch.arange(
            self.start_octave, self.start_octave + self.num_octaves)
        octaves = octaves.float()
        self.multipliers = 2**octaves * math.pi

    def forward(self, coords):

        shape, dim = coords.shape[:-1], coords.shape[-1]

        multipliers = self.multipliers.to(coords)
        coords = coords.unsqueeze(-1)
        while len(multipliers.shape) < len(coords.shape):
            multipliers = multipliers.unsqueeze(0)

        scaled_coords = coords * multipliers

        sines = torch.sin(scaled_coords).reshape(
            *shape, dim * self.num_octaves)
        cosines = torch.cos(scaled_coords).reshape(
            *shape, dim * self.num_octaves)

        result = torch.cat((sines, cosines), -1)
        return result


class RayPosEncoder(nn.Module):
    def __init__(self, pos_octaves=8, pos_start_octave=0, ray_octaves=4, ray_start_octave=0):
        super().__init__()
        self.pos_encoding = PositionalEncoding(
            num_octaves=pos_octaves, start_octave=pos_start_octave)
        self.ray_encoding = PositionalEncoding(
            num_octaves=ray_octaves, start_octave=ray_start_octave)

    def forward(self, pos, rays):
        pos_enc = self.pos_encoding(pos)
        ray_enc = self.ray_encoding(rays)
        x = torch.cat((pos_enc, ray_enc), -1)
        return x


class RayOnlyEncoder(nn.Module):
    def __init__(self, ray_octaves=4, ray_start_octave=0):
        super().__init__()
        self.ray_encoding = PositionalEncoding(
            num_octaves=ray_octaves, start_octave=ray_start_octave)

    def forward(self, rays):
        if len(rays.shape) == 4:
            batchsize, height, width, dims = rays.shape
            rays = rays.flatten(1, 2)
            ray_enc = self.ray_encoding(rays)
            ray_enc = ray_enc.view(batchsize, height, width, ray_enc.shape[-1])
            ray_enc = ray_enc.permute((0, 3, 1, 2))
        else:
            ray_enc = self.ray_encoding(rays)

        return ray_enc


class LearnedRayEmbedding(nn.Module):
    def __init__(self, ray_octaves=60, ray_start_octave=-30, h=320, w=240):
        super().__init__()
        self.encoding = PositionalEncoding(
            num_octaves=ray_octaves, start_octave=ray_start_octave)
        initial_emb = self.encoding(get_vertical_rays(width=w, height=h)[None])
        self.emb = nn.Parameter(initial_emb)
        print(self.emb.shape)

    def forward(self, N):
        return self.emb[None].repeat(N, 1, 1, 1)


# Transformer implementation based on ViT
# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py


class TemperatureAdjsutableSoftmax(nn.Module):

    def __init__(self, init_tau=1.0, dim=-1):
        super().__init__()
        self.tau = nn.Parameter(torch.Tensor([init_tau]))
        self.softmax = nn.Softmax(dim=dim)

    def forward(self, x):
        return self.softmax(x/self.tau)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(
            dim) if dim is not None else lambda x: torch.nn.functional.normalize(x, dim=-1)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0., linear_module=ViTLinear):
        super().__init__()
        self.net = nn.Sequential(
            linear_module(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0. else nn.Identity(),
            linear_module(hidden_dim, dim),
            nn.Dropout(dropout) if dropout > 0. else nn.Identity(),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., kv_dim=None, attn_args={}, linear_module=JaxLinear, **kwargs):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        # kv_dim being None indicates this attention is used as self-attention
        self.selfatt = (kv_dim is None)
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.method = attn_args['method']['name']
        self.method_args = attn_args['method']['args']
        self.rpe = 'rpe' in self.method_args and self.method_args['rpe']
        self.use_bias = self.method_args.get('use_bias', False)

        if self.method == 'gta':
            if 'se3' in self.method_args['f_dims'] and self.method_args['f_dims']['se3'] > 0:
                if not self.method_args.get('elementwise_mul', False):
                    self.trans_coeff = nn.Parameter(torch.Tensor([0.01]))
            else:
                self.trans_coeff = None

        if 'softmax' in attn_args and attn_args['softmax'] == 'adjustable':
            self.attend = TemperatureAdjsutableSoftmax(init_tau=1.0, dim=-1)
            tau = self.attend.tau
        else:
            self.attend = nn.Softmax(-1)
            tau = 1.0

        class AttnFn(torch.nn.Module):
            def __init__(self, scale):
                self.scale = scale
                super().__init__()

            def forward(self, q, k, v):
                sim = q @ k.transpose(-1, -2)  # [B, H, Nq*Tq, Nk*Tk]
                attn = nn.Softmax(-1)(sim * self.scale / tau)
                out = (attn @ v)
                return out, attn

        class EuclidAttnFn(torch.nn.Module):
            def __init__(self, scale):
                self.scale = scale
                super().__init__()
                print("""Euclid Attention""")

            def forward(self, q, k, v):
                # sim(Q, K) = -0.5*||Q-K||^2 = Q'K - 0.5Q'Q - 0.5K'K
                sim = q @ k.transpose(-1, -2)  - 0.5 * q.sum(-1)[..., None] - 0.5 * k.sum(-1)[..., None, :] 
                attn = nn.Softmax(-1)(sim * self.scale / tau)
                out = (attn @ v)
                return out, attn

        self.euclid = self.method_args.get('euclid_sim', False)
        self.attn_fn = EuclidAttnFn(self.scale) if self.euclid else AttnFn(self.scale)

        # parse
        if self.method == 'repast':
            if kv_dim is None:
                kv_dim = dim
            self.v_bias = self.method_args['v_bias'] if 'v_bias' in self.method_args else False
            self.to_q = linear_module(
                dim+self.method_args['q_emb_dim'], inner_dim, bias=self.use_bias)
            self.to_k = linear_module(
                kv_dim+self.method_args['k_emb_dim'], inner_dim, bias=self.use_bias)
            self.to_v = linear_module(
                kv_dim+self.method_args['k_emb_dim'] if self.v_bias else kv_dim, inner_dim, bias=self.use_bias)
        else:
            q_inner_dim = inner_dim
            if kv_dim is not None:
                self.to_q = linear_module(dim, q_inner_dim, bias=self.use_bias)
                self.to_kv = linear_module(
                    kv_dim, 2*inner_dim, bias=self.use_bias)
            else:
                self.to_qkv = linear_module(
                    dim, inner_dim * 2 + q_inner_dim, bias=self.use_bias)
            
            if self.method == 'ape':
                if kv_dim is not None:
                    self.linear_q = nn.Linear(16+180, dim)
                    self.linear_k = nn.Linear(16+180, kv_dim)
                else:
                    self.linear = nn.Linear(16+180, dim)

            elif self.rpe:
                # Only support with SE(3) + SO(2)
                (self.q_bias, self.k_bias, self.v_bias) = [
                    nn.Parameter(torch.cat([
                        torch.eye(4)[None].repeat(
                            self.heads, 1, 1).flatten(-2, -1),
                        torch.eye(2)[None, None, :, 0].repeat(self.heads, self.method_args['so2']*2, 1).flatten(-2, -1)], -1))
                    for _ in range(3)]

        if self.method == 'gta':
            if self.method_args.get('elementwise_mul', False):
                # compute full dimension:
                so2dim = self.method_args['f_dims']['so2']
                freqs = so2dim // 4
                self.rep_to_vec = nn.Linear(16+2*freqs*2*2, inner_dim//heads)

        if self.method == 'gbt':
            self.geo_weights = nn.Parameter(data=torch.FloatTensor([1]), requires_grad=True)

        if self.method == 'mln':
            if kv_dim is not None:
                self.linear_q_g = nn.Linear(16+180, dim)
                self.linear_q_b = nn.Linear(16+180, dim)
                self.linear_k_g = nn.Linear(16+180, kv_dim)
                self.linear_k_b = nn.Linear(16+180, kv_dim)
            else:
                self.linear_g = nn.Linear(16+180, dim)
                self.linear_b = nn.Linear(16+180, dim)
        
        self.to_out = nn.Sequential(
            linear_module(inner_dim if not self.rpe else inner_dim +
                      self.heads*self.q_bias.shape[-1], dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, z=None, return_attmap=False, extras={}):

        if self.method == 'repast':
            q = x
            if len(q.shape) == 4:
                # query is already augmented
                q_is_already_augmented = True
                B, Tq, Nk = q.shape[0], q.shape[1], q.shape[2]
            else:
                q_is_already_augmented = False
                # ray expressed in Key tokens' canonical coordinates of shape [B, Tq, Nk, emb_dim]. This associates with query and key
                q_ray = extras['query_ray_emb']
                B, Tq, Nk = q_ray.shape[0], q_ray.shape[1], q_ray.shape[2]
                q = q.unsqueeze(2).expand(-1, -1, Nk, -1)  # [B, Tq, Nk, C]
                q = torch.cat([q, q_ray], -1)  # augmented query

            k = v = x if z is None else z

            if len(k.shape) == 4:
                # key and value are already augmented, but that only makes sense if k does not share its value with q
                assert z is not None
            else:
                # process key and value veoctrs 
                # Rays are expressed in key tokens' canonical coordinates of shape [B, Nk, Lk, emb_dim].
                k_ray = extras['key_ray_emb']

                k = k.reshape(*k_ray.shape[:-1], -1)  # [B, Nk, Lk, C]
                k = torch.cat([k, k_ray], -1)  # [B, Nk, Lk, C]

                if self.v_bias:
                    shape = v.shape
                    v = v.reshape(*k_ray.shape[:-1], -1)
                    v = torch.cat([v, k_ray], -1).reshape(shape[0],
                                                          shape[1], -1)  # [B, Nk, Lk, C]

            q = self.to_q(q)  
            k = self.to_k(k)  
            v = self.to_v(v) 
            q, k = map(lambda t: rearrange(
                t, 'b n m (h d) -> b h n m d', h=self.heads), (q, k))

            v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)

            sim = torch.einsum('bhtnc,bhnlc->bhtnl', q,
                               k)  # [B, H, Tq, Nk, Lk]
            sim = sim.reshape(B, self.heads, Tq, -1)  # [B, H, Tq, Nk*Lk]
            if 'enable_scale' in self.method_args and self.method_args['enable_scale']:
                attn = self.attend(sim * self.scale)
            else:
                attn = self.attend(sim)
            out = torch.matmul(attn, v)
            out = rearrange(out, 'b h n d -> b n (h d)')
            if q_is_already_augmented:
                out = out.unsqueeze(2).expand(-1, -1, Nk, -1)
            out = self.to_out(out)
        else:
            if self.method == 'ape':
                if z is not None:
                    coord_q = extras['target_coord_emb']  # [B, Nq, Tq, 180]
                    coord_k = extras['input_coord_emb']  # [B, Nk, Tk, 180]
                    Cq = extras['target_transforms'].flatten(-2, -1)[:, :, None].repeat(
                        1, 1, coord_q.shape[2], 1)  # [B, Nq, Tq, 16]
                    Ck = extras['input_transforms'].flatten(-2, -1)[:, :, None].repeat(
                        1, 1, coord_k.shape[2], 1)  # [B, Nk, Tk, 16]

                    emb_q = torch.cat([Cq, coord_q], -1).flatten(1, 2)
                    x = x + self.linear_q(emb_q)
                    emb_k = torch.cat([Ck, coord_k], -1).flatten(1, 2)
                    z = z + self.linear_k(emb_k)
                else:
                    coord_q = extras['input_coord_emb']  # [B, Nq, Tq, 180]
                    Cq = extras['input_transforms'].flatten(-2, -1)[:, :, None].repeat(
                        1, 1, coord_q.shape[2], 1)   # [B, Nq, Tq, 16]
                    emb = torch.cat([Cq, coord_q], -1).flatten(1, 2)
                    x = x + self.linear(emb)
            if self.method == 'mln':
                if z is not None:
                    coord_q = extras['target_coord_emb']  # [B, Nq, Tq, 180]
                    coord_k = extras['input_coord_emb']  # [B, Nk, Tk, 180]
                    Cq = torch.linalg.inv(extras['target_transforms']).flatten(-2, -1)[:, :, None].repeat(
                        1, 1, coord_q.shape[2], 1)  # [B, Nq, Tq, 16]
                    Ck = torch.linalg.inv(extras['input_transforms']).flatten(-2, -1)[:, :, None].repeat(
                        1, 1, coord_k.shape[2], 1)  # [B, Nk, Tk, 16]

                    emb_q = torch.cat([Cq, coord_q], -1).flatten(1, 2)
                    x = self.linear_q_g(emb_q)*x + self.linear_q_b(emb_q)
                    emb_k = torch.cat([Ck, coord_k], -1).flatten(1, 2)
                    z = self.linear_k_g(emb_k)*z + self.linear_k_b(emb_k)
                else:
                    coord_q = extras['input_coord_emb']  # [B, Nq, Tq, 180]
                    Cq = extras['input_transforms'].flatten(-2, -1)[:, :, None].repeat(
                        1, 1, coord_q.shape[2], 1)   # [B, Nq, Tq, 16]
                    emb = torch.cat([Cq, coord_q], -1).flatten(1, 2)
                    x = self.linear_g(emb)*x + self.linear_b(emb)


            if z is None:
                qkv = self.to_qkv(x).chunk(3, dim=-1)
            else:
                q = self.to_q(x)
                k, v = self.to_kv(z).chunk(2, dim=-1)
                qkv = (q, k, v)      
            q, k, v = map(lambda t: rearrange(
                t, 'b n (h d) -> b h n d', h=self.heads), qkv)

            if 'rpe' in self.method_args and self.method_args['rpe']:
                q_bias = self.q_bias[None, :, None].repeat(
                    q.shape[0], 1, q.shape[2], 1)
                k_bias = self.k_bias[None, :, None].repeat(
                    k.shape[0], 1, k.shape[2], 1)
                v_bias = self.v_bias[None, :, None].repeat(
                    v.shape[0], 1, v.shape[2], 1)
                q, k, v = map(
                    lambda x: torch.cat(x, -1),
                    ((q, q_bias), (k, k_bias), (v, v_bias))
                )

            if self.method == 'gta':
                if self.method_args.get('elementwise_mul', False):
                    (extras['vecrep_q'],
                     extras['vecrep_k'],
                     extras['vecinvrep_q']) = map(lambda frep: self.rep_to_vec(frep), 
                                                         (extras['flattened_rep_q'],
                                                          extras['flattened_rep_k'],
                                                          extras['flattened_invrep_q'])) 
                    fn = multihead_vecrep_attention
                else:
                    fn = multihead_geometric_transform_attention

                v_transform = self.method_args['v_transform'] if 'v_transform' in self.method_args else True
                out, attn = fn(
                    q, k, v, attn_fn=self.attn_fn,
                    f_dims=self.method_args['f_dims'],
                    reps=extras,
                    trans_coeff=self.trans_coeff if not self.method_args.get('elementwise_mul', False) else None,
                    v_transform=v_transform,
                    euclid=self.euclid)
                out = rearrange(out, 'b h n d -> b n (h d)')
                out = self.to_out(out)
            else:
                sim = torch.matmul(q, k.transpose(-1, -2)) * \
                    self.scale  # [B, nh, n_queries, n_inputs]
                if self.method == 'gbt':
                    sim = sim - ((self.geo_weights**2) * extras['plucker_dist'])[:, None]
                attn = self.attend(sim)
                out = torch.matmul(attn, v)
                out = rearrange(out, 'b h n d -> b n (h d)')
                out = self.to_out(out)

        if return_attmap:
            return out, attn
        else:
            return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim,
                 dropout=0.,
                 selfatt=True,
                 kv_dim=None,
                 return_last_attmap=False,
                 attn_args={}):

        super().__init__()
        self.heads = heads
        self.layers = nn.ModuleList([])

        linear_module_attn = lambda *args, **kwargs: JaxLinear(*args, **kwargs)
        linear_module_ff = lambda *args, **kwargs: ViTLinear(*args, **kwargs)

        prenorm_fn = lambda m: PreNorm(dim, m)
        for k in range(depth):
            attn = prenorm_fn(Attention(
                    dim, heads=heads, dim_head=dim_head,
                    dropout=dropout, selfatt=selfatt, kv_dim=kv_dim, attn_args=attn_args, 
                    linear_module=linear_module_attn))
            ff = prenorm_fn(FeedForward(
                dim, mlp_dim,
                dropout=dropout,
                linear_module=linear_module_ff))
            self.layers.append(nn.ModuleList([attn, ff]))
        self.return_last_attmap = return_last_attmap

    def forward(self, x, z=None, extras=None):
        
        for l, (attn, ff) in enumerate(self.layers):
            if l == len(self.layers)-1 and self.return_last_attmap:
                out, attmap = attn(x, z=z, return_attmap=True, extras=extras)
                x = out + x
            else:
                x = attn(x, z=z, extras=extras) + x
            x = ff(x) + x

        if self.return_last_attmap:
            return x, attmap
        else:
            return x
 