import numpy as np
import torch
import torch.nn as nn
from source.utils.nerf import get_vertical_rays, transform_points_torch
from source.utils.common import positionalencoding2d_given_coord
from source.layers import RayPosEncoder, Transformer, SRTLinear
from source.utils import nerf
from source.utils.common import rigid_transform,  rotmat_to_wigner_d_matrices, downsample
from source.utils.gta import make_SO2mats, ray2rotation, make_T2mats
from einops import rearrange
from source.utils.gbt import get_plucker_parameterization, plucker_dist, positional_encoding
from source.utils.frustum_posemb import generate_frustum_pixel_points


def get_act_module(act):
    if act == 'relu':
        _act = nn.ReLU()
    elif act == 'lrelu':
        _act = nn.LeakyReLU()
    elif act == 'gelu':
        _act = nn.GELU()
    else:
        raise NotImplementedError
    return _act


class RayPredictor(nn.Module):
    def __init__(self,
                 dim=180,
                 num_att_blocks=2,
                 pos_start_octave=0,
                 z_dim=768,
                 input_mlp=False,
                 heads=12,
                 dim_head=128,
                 mlp_dim=3072,
                 return_last_attmap=False,
                 emb='ray',
                 dropout=None,
                 H=128,
                 W=128,
                 attn_args={},
                 **kwargs):
        super().__init__()
        self.emb = emb
        self.H = H
        self.W = W

        if emb == 'ray':
            self.query_encoder = RayPosEncoder(
                pos_octaves=15, pos_start_octave=pos_start_octave,
                ray_octaves=15)
            q_dim = 180
        elif emb == 'camera_planar':
            self.height = kwargs['scale_h']
            self.width = kwargs['scale_w']
            q_dim = 180+12
        elif emb == 'planar':
            self.height = kwargs['scale_h']
            self.width = kwargs['scale_w']
            q_dim = 180
        elif emb == 'const':
            self.initial_emb = nn.Parameter(torch.randn(dim))
            q_dim = dim
        elif emb is None:
            pass
        else:
            raise NotImplementedError

        if emb == 'ray' or emb == 'camera_planar' or emb == 'planar':
            if input_mlp:  # Input MLP added with OSRT improvements
                self.input_mlp = nn.Sequential(
                    SRTLinear(q_dim, 360),
                    nn.ReLU(),
                    SRTLinear(360, dim))
            else:
                self.input_mlp = None

        self.dim = dim
        self.return_last_attmap = return_last_attmap

        self.transformer = Transformer(
            self.dim,
            depth=num_att_blocks,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            selfatt=False,
            kv_dim=z_dim,
            return_last_attmap=return_last_attmap,
            dropout=dropout,
            attn_args=attn_args,
        )

    def forward(self, z, x, rays, extras, queries=None):
        """
        Args:
            z: scene encoding [batch_size, num_patches, patch_dim]
            x: query camera positions [batch_size, num_rays, 3]
            rays: query ray directions [batch_size, num_rays, 3]
        """
        if queries is None:
            if self.emb == 'const':
                B, K = rays.shape[0], rays.shape[1]
                queries = self.initial_emb[None, None].expand(B, K, -1)

            else:
                if self.emb == 'ray':
                    queries = self.query_encoder(x, rays)
                elif self.emb == 'camera_planar':
                    cam = extras['target_transforms'][:, :,
                                                      :3].flatten(-2, -1)  # [B, Nk, 12]
                    coord = extras['target_coord']  # [B, Nk, T, 2]
                    emb = positionalencoding2d_given_coord(
                        180, coord, [self.height, self.width])  # [B, Nk, T, 180]
                    # [B, Nk, T, 180+12])])
                    queries = torch.cat(
                        [emb, cam[:, :, None].expand(-1, -1, coord.shape[2], -1)], dim=-1)
                    queries = queries.flatten(1, 2)
                elif self.emb == 'planar':
                    coord = extras['target_coord']  # [B, Nk, T, 2]
                    emb = positionalencoding2d_given_coord(
                        180, coord, [self.height, self.width])  # [B, Nk, T, 180]
                    queries = emb.flatten(1, 2)

                if self.input_mlp:
                    queries = self.input_mlp(queries)

                if queries.shape[0] != z.shape[0]:
                    # z's shape is [B*Nk, Tq, C]
                    queries = queries.reshape(
                        z.shape[0], -1, queries.shape[-1])

        out = self.transformer(queries, z, extras)

        return out, z, queries


class ImprovedSRTDecoder(nn.Module):
    """ Scene Representation Transformer Decoder with the improvements from Appendix A.4 in the OSRT paper."""

    def __init__(self,
                 dim=180,
                 num_att_blocks=2, pos_start_octave=0, z_dim=768, heads=12, return_last_attmap=False, rmlp_dim=1536,
                 act='lrelu', dim_in=None, dropout=None, dim_head=None, mlp_dim=None, emb='ray', prenorm=True, sigmoid=True,
                 attn_args={},
                 **kwargs):
        super().__init__()
        self.lin_in = nn.Linear(
            dim_in, z_dim) if dim_in is not None else lambda x: x

        dim_head = z_dim // heads if dim_head is None else dim_head
        mlp_dim = z_dim * 2 if mlp_dim is None else mlp_dim

        self.allocation_transformer = RayPredictor(
            dim=dim,
            num_att_blocks=num_att_blocks,
            pos_start_octave=pos_start_octave,
            z_dim=z_dim,
            input_mlp=True,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            return_last_attmap=return_last_attmap,
            dropout=dropout,
            emb=emb,
            prenorm=prenorm,
            attn_args=attn_args,
            **kwargs
        )
        self.method = attn_args['method']['name']
        self.is_gta = self.method == 'gta'
        self.attn_args = attn_args['method']['args']
        self.heads = heads

        if self.method == 'ape' or self.method == 'mln':
            self.height = kwargs['scale_h']
            self.width = kwargs['scale_w']

        # returning attmap as segmentation map makes sense only if heads==1
        assert (not return_last_attmap) or heads == 1
        _act = get_act_module(act)
        self.return_last_attmap = return_last_attmap

        self.render_mlp = nn.Sequential(
            SRTLinear(dim, rmlp_dim),
            _act,
            SRTLinear(rmlp_dim, rmlp_dim),
            _act,
            SRTLinear(rmlp_dim, rmlp_dim),
            _act,
            SRTLinear(rmlp_dim, rmlp_dim),
            _act,
            SRTLinear(rmlp_dim, 3),
            nn.Sigmoid() if sigmoid else nn.Identity(),
        )

        if self.method == 'frustum_posemb':
            D = self.attn_args['D']
            indim = D*4
            self.frustum_phi = nn.Sequential(
                nn.Linear(indim, dim*2),
                nn.ReLU(),
                nn.Linear(dim*2, dim))

    def _replicate_rays(self, x, rays, extras):
        """
        Args:
            x: Tensor of shape [B, T, 3]
            rays: Tensor of shape [B, T, 3]
            extras: dict includes input camera poses 
        Returns Tensor of shape [B, T, Nk, C]
        """
        input_transforms = extras['input_transforms']  # [B, Nk, 4, 4]
        Nk = input_transforms.shape[1]
        _x = rigid_transform(input_transforms, x.unsqueeze(
            1).expand(-1, Nk, -1, -1), trans_coeff=1.0).transpose(1, 2)
        _rays = rigid_transform(input_transforms, rays.unsqueeze(
            1).expand(-1, Nk, -1, -1),  trans_coeff=0.0).transpose(1, 2)
        return _x, _rays

    def compute_gbt_biases(self, x, rays, extras):
        # [B, Tq, 6].
        plucker_ray = get_plucker_parameterization(torch.cat((x, rays), -1))
        dist = plucker_dist(plucker_ray, extras['ray_input'])
        extras['plucker_dist'] = dist
        return torch.chunk(plucker_ray, 2, -1)  # [B, Tq, 3], [B, Tq, 3]

    def compute_frustum_posemb(self, extras):
        input_coord = extras['target_coord']
        rel_cam = extras['target_transforms']  # [B, N, 4, 4]
        p3d = generate_frustum_pixel_points(
            input_coord,
            torch.linalg.inv(rel_cam),
            self.attn_args['D'],
            dmin=self.attn_args.get('dmin', 0.1),
            dmax=self.attn_args.get('dmax', 10))  # [B, N, T, D*4]
        if self.attn_args.get('normalize', False):
            p3d = 0.01 * p3d
        if self.attn_args.get('fourier', False):
            p3d = positional_encoding(
                p3d, self.attn_args['freqs'], parameterize=None)
        emb = self.frustum_phi(p3d)  # [B, N, T, dim]
        emb = emb.flatten(1, 2)
        return emb

    def pre_compute_reps(self, attn_kwargs, extras):
        f_dims = attn_kwargs['f_dims']
        flattened_reps = []
        flattened_invreps = []
        if 'so2' in f_dims and f_dims['so2'] > 0:
            coord = extras['target_coord']
            coord = coord.reshape(coord.shape[0], -1, 2)  # [B, Nq*Tq, 2]
            so2rep = make_SO2mats(coord, nfreqs=attn_kwargs['so2'],
                                  max_freqs=[attn_kwargs['max_freq_h'],
                                             attn_kwargs['max_freq_w']],
                                  shared_freqs=attn_kwargs['shared_freqs'] if 'shared_freqs' in attn_kwargs else False)  # [B, Nq*Tq, deg, 2, 2, 2]
            so2rep = so2rep.flatten(-4, -3)
            extras['so2rep_q'] = so2rep
            extras['so2fn'] = lambda A, x: torch.einsum(
                'btcij,bhtcj->bhtci', A, x)

            if 'recompute_so2' in attn_kwargs and attn_kwargs['recompute_so2']:
                coord = extras['input_coord']
                coord = coord.reshape(coord.shape[0], -1, 2)  # [B, Nq*Tq, 2]
                so2rep = make_SO2mats(coord,
                                      nfreqs=attn_kwargs['so2'],
                                      max_freqs=[
                                          attn_kwargs['max_freq_h'], attn_kwargs['max_freq_w']],
                                      shared_freqs=attn_kwargs['shared_freqs'] if 'shared_freqs' in attn_kwargs else False)  # [B, Nq*Tq, deg, 2, 2, 2]
                so2rep = so2rep.flatten(-4, -3)
                extras['so2rep_k'] = so2rep

            NqTq = so2rep.shape[1]
            flattened = so2rep.reshape(
                so2rep.shape[0], so2rep.shape[1], -1)  # [B, T, C*2*2]
            flattened_reps.append(flattened)
            # [B, T, C*2*2]
            flattened_inv = so2rep.transpose(-2, -1).reshape(
                so2rep.shape[0], so2rep.shape[1], -1)
            flattened_invreps.append(flattened_inv)

        if 't2' in f_dims and f_dims['t2'] > 0:
            coord = extras['target_coord']
            coord = coord.reshape(coord.shape[0], -1, 2)  # [B, Nq*Tq, 2]
            t2rep = make_T2mats(coord)  # [B, Nq*Tq, 2] -> [B, Nq*Tq, 3, 3]
            extras['t2rep_q'] = t2rep
            extras['inv_t2rep_q'] = torch.linalg.inv(t2rep)
            extras['t2fn'] = lambda A, x: torch.einsum(
                'btij,bhtcj->bhtci', A, x)

        if 'se3' in f_dims and f_dims['se3'] > 0:
            extrinsic = extras['target_transforms']
            se3rep = torch.linalg.inv(extrinsic)  # [B, Nq, 4, 4]
            if 'ray_to_se3' in attn_kwargs and attn_kwargs['ray_to_se3']:
                B, Nq = se3rep.shape[0], se3rep.shape[1]
                target_rays = extras['target_rays'].reshape(B, Nq, -1, 3)
                # [B, Nq, T, 4, 4] R: ray direction-> base ray (extrinsic)
                R = ray2rotation(target_rays, return_4x4=True)
                se3rep = torch.einsum(
                    'bnij,bntjk->bntik', se3rep, R)  # mul from right
                extrinsic = torch.einsum(
                    'bntij,bnjk->bntik',  R.transpose(-2, -1), extrinsic)  # mul from left
                extras['se3fn'] = lambda A, x: torch.einsum(
                    'bntij,bhntcj->bhntci', A, x)
            else:
                extras['se3fn'] = lambda A, x: torch.einsum(
                    'bnij,bhntcj->bhntci', A, x)
            extras['se3rep_q'] = se3rep
            extras['inv_se3rep_q'] = extrinsic
            if not 'se3rep_k' in extras:
                extrinsic = extras['input_transforms']
                se3rep = torch.linalg.inv(extrinsic)  # [B, Nk, 4, 4]
                if 'ray_to_se3' in attn_kwargs and attn_kwargs['ray_to_se3']:
                    B, Nk = se3rep.shape[0], se3rep.shape[1]
                    input_rays = downsample(
                        extras['input_rays'], 3).reshape(B, Nk, -1, 3)
                    # [B, Nq, T, 4, 4]
                    R = ray2rotation(input_rays, return_4x4=True)
                    se3rep = torch.einsum(
                        'bnij,bntjk->bntik', se3rep, R)  # mul from right
                    extrinsic = torch.einsum(
                        'bntij,bnjk->bntik',  R.transpose(-2, -1), extrinsic)  # mul from left
                extras['se3rep_k'] = se3rep
            flattened = extrinsic.repeat_interleave(
                NqTq//extrinsic.shape[1], 1).transpose(-2, -1).reshape(se3rep.shape[0], -1, 16)  # [B, T, 4*4]
            flattened_reps.append(flattened)
            flattened_inv = extrinsic.repeat_interleave(
                NqTq//extrinsic.shape[1], 1).reshape(se3rep.shape[0], -1, 16)  # [B, T, 4*4]
            flattened_invreps.append(flattened_inv)

        if 'so3' in f_dims and f_dims['so3'] > 0:
            n_degs = attn_kwargs['so3']
            R_q = torch.linalg.inv(extras['target_transforms'])[..., :3, :3]
            B, Nq = R_q.shape[0], R_q.shape[1]
            D_q = rotmat_to_wigner_d_matrices(n_degs, R_q.flatten(0, 1))[1:]
            for i, D in enumerate(D_q):
                if 'zeroout_so3' in attn_kwargs and attn_kwargs['zeroout_so3']:
                    D_q[i] = torch.zeros_like(
                        D.reshape(B, Nq, D.shape[-2], D.shape[-1]))
                elif 'id_so3' in attn_kwargs and attn_kwargs['id_so3']:
                    D_q[i] = torch.stack([torch.eye(
                        D.shape[-1])]*B*Nq, 0).reshape(B, Nq, D.shape[-2], D.shape[-1]).to(D.device)
                else:
                    D_q[i] = D.reshape(B, Nq, D.shape[-2], D.shape[-1])
            extras['so3rep_q'] = D_q
            extras['so3fn'] = lambda A, x: torch.einsum(
                'bnij,bhnkj->bhnki', A, x)

        if 'sep' in f_dims and f_dims['sep'] > 0:
            se3rep = torch.linalg.inv(
                extras['target_transforms'])  # [B, Nq, 4, 4]
            n_degs = attn_kwargs['so3']
            R_q = se3rep[..., :3, :3]
            B, Nq = se3rep.shape[0], se3rep.shape[1]
            target_rays = extras['target_rays'].reshape(B, Nq, -1, 3)
            R = ray2rotation(target_rays)  # [B, Nq, T, 4, 4]
            R_q = torch.einsum('bnij,bntjk->bntik', R_q,
                               R)  # mul from right
            D_q = rotmat_to_wigner_d_matrices(n_degs, R_q.flatten(0, 2))[1:]
            for i, D in enumerate(D_q):
                D_q[i] = D.reshape(B, Nq, -1, D.shape[-2], D.shape[-1])
            extras['so3rep_q'] = D_q
            extras['so3fn'] = lambda A, x: torch.einsum(
                'bntij,bhntkjdm->bhntkidm', A, x)

            coord = se3rep[..., :3, 3]  # [B, Nq, 3]
            coord = extras['trans_coeff'] * coord
            so2rep = make_SO2mats(coord,
                                  nfreqs=attn_kwargs['so2'],
                                  max_freqs=attn_kwargs['max_freqs'])  # [B, Nq, deg, 3, 2, 2]
            so2rep = so2rep.flatten(-4, -3)  # [B, Nq, deg*3, 2, 2]
            extras['so2rep_q'] = so2rep
            extras['so2fn'] = lambda A, x: torch.einsum(
                'bndlm,bhntkjdm->bhntkjdl', A, x)

        extras['flattened_rep_q'] = torch.cat(
            flattened_reps, -1)  # 16 + 2*freqs*2*2
        extras['flattened_invrep_q'] = torch.cat(
            flattened_invreps, -1)  # 16 + 2*freqs*2*2

    def forward(self, z, x, rays, extras):
        z = self.lin_in(z)

        queries = None
        if self.method == 'repast':
            x, rays = self._replicate_rays(x, rays, extras)  # [B, T, Nk, C]
        if self.method == 'gbt':
            x, rays = self.compute_gbt_biases(x, rays, extras)
        if self.method == 'frustum_posemb':
            queries = self.compute_frustum_posemb(extras)
        if self.is_gta:
            self.pre_compute_reps(self.attn_args, extras)
        if self.method == 'ape' or self.method == 'mln':
            emb = positionalencoding2d_given_coord(180, extras['target_coord'], [
                                                   self.height, self.width])  # [B, Nk, T, 180]
            extras['target_coord_emb'] = emb

        x, _, _ = self.allocation_transformer(
            z, x, rays, extras, queries=queries)
        ret_dict = {}
        if self.return_last_attmap:
            x, attn = x
            attn = attn.squeeze(1)
            ret_dict['masks'] = attn

        if self.method == 'repast':
            x = torch.mean(x, 2)

        pixels = self.render_mlp(x)
        return pixels, ret_dict
