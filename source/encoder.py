import numpy as np
import torch
import torch.nn as nn
from source.layers import RayPosEncoder, Transformer
from source.utils.common import positionalencoding2d, downsample, rigid_transform, make_SO2mats, ray2rotation, make_T2mats
from source.utils.wigner_d import rotmat_to_wigner_d_matrices
from source.utils.nerf import transform_points_torch
from torch.nn import init
from einops import rearrange, repeat
from source.utils.gbt import get_plucker_parameterization, plucker_dist, positional_encoding
from source.utils.frustum_posemb import generate_frustum_pixel_points
import math


class SRTConvBlock(nn.Module):
    def __init__(self, idim, hdim=None, odim=None, downsample=True):
        super().__init__()
        if hdim is None:
            hdim = idim

        if odim is None:
            odim = 2 * hdim

        conv_kwargs = {'bias': False, 'kernel_size': 3, 'padding': 1}
        self.layers = nn.Sequential(
            nn.Conv2d(idim, hdim, stride=1, **conv_kwargs),
            nn.ReLU(),
            nn.Conv2d(hdim, odim, stride=2 if downsample else 1, **conv_kwargs),
            nn.ReLU())

    def forward(self, x):
        return self.layers(x)


class PatchfyConv(nn.Module):

    def __init__(self,
                 dim=768,
                 num_conv_blocks=3,
                 pos_start_octave=0,
                 downsample=True,
                 ):
        super().__init__()
        self.ray_encoder = RayPosEncoder(pos_octaves=15, pos_start_octave=pos_start_octave,
                                         ray_octaves=15)
        in_dims = []
        out_dims = []
        h_dims = []
        for i in range(num_conv_blocks):
            if i == 0:
                in_dims.append(183)
                h_dims.append(96)
                out_dims.append(dim // (2**(num_conv_blocks-1))
                                if downsample else dim)
            if i > 0:
                if downsample:
                    in_dims.append(dim // (2**(num_conv_blocks-i)))
                    h_dims.append(None)
                    out_dims.append(dim // (2**(num_conv_blocks-i-1)))
                else:
                    in_dims.append(dim)
                    h_dims.append(dim//2)
                    out_dims.append(dim)

        conv_blocks = []
        for i in range(num_conv_blocks):
            conv_blocks.append(SRTConvBlock(
                idim=in_dims[i], hdim=h_dims[i], odim=out_dims[i], downsample=downsample))

        self.conv_blocks = nn.Sequential(*conv_blocks)

        self.per_patch_linear = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, images, camera_pos, rays):
        batch_size, num_images = images.shape[:2]

        x = images.flatten(0, 1)
        camera_pos = camera_pos.flatten(0, 1)
        rays = rays.flatten(0, 1)

        ray_enc = self.ray_encoder(camera_pos, rays)
        x = torch.cat((x, ray_enc), 1)
        x = self.conv_blocks(x)
        x = self.per_patch_linear(x)
        x = x.flatten(2, 3).permute(0, 2, 1)

        patches_per_image, channels_per_patch = x.shape[1:]
        x = x.reshape(batch_size, num_images *
                      patches_per_image, channels_per_patch)
        return x


class ImprovedSRTEncoder(nn.Module):
    """
    Scene Representation Transformer Encoder with the improvements from Appendix A.4 in the OSRT paper.
    """

    def __init__(self,
                 dim=768,
                 attdim=768,
                 num_conv_blocks=3,
                 num_att_blocks=5,
                 pos_start_octave=0,
                 heads=12,
                 dim_out=None,
                 dropout=None,
                 output_scaler=False,
                 patch_method='conv',
                 emb='ray',
                 attn_args={},
                 **kwargs,
                 ):
        super().__init__()

        self.method = attn_args['method']['name']
        self.emb = emb
        self.is_gta = self.method == 'gta'
        self.attn_args = attn_args['method']['args']
        self.heads = heads

        if self.emb == 'ray' or self.method == 'repast':
            self.ray_encoder = RayPosEncoder(pos_octaves=15,
                                             pos_start_octave=pos_start_octave,
                                             ray_octaves=15)
            if self.method == 'repast':
                # no input embedding for the repast method.
                emb_dim = 0
            else:
                emb_dim = 180
        elif self.emb == 'planar':
            emb_dim = 180
        elif self.emb == 'camera_planar':
            self.lin_camera = nn.Linear(12, attdim)  # For camera
            self.lin_planar = nn.Linear(180, attdim) # For 2d positions
            emb_dim = 0
        elif self.emb is False:
            emb_dim = 0
        else:
            raise NotImplementedError

        self.patch_method = patch_method

        conv_blocks = [SRTConvBlock(idim=3+emb_dim, hdim=dim//8)]
        cur_hdim = dim//4
        for i in range(1, num_conv_blocks):
            conv_blocks.append(SRTConvBlock(idim=cur_hdim, odim=None))
            cur_hdim *= 2
        self.conv_blocks = nn.Sequential(*conv_blocks)
        self.per_patch_linear = nn.Conv2d(cur_hdim, attdim, kernel_size=1)

        self.transformer = Transformer(
            dim=attdim,
            depth=num_att_blocks,
            heads=heads,
            dim_head=attdim//heads,
            mlp_dim=attdim*2,
            selfatt=True,
            dropout=dropout,
            attn_args=attn_args)

        if self.is_gta:
            if 'sep' in self.attn_args['f_dims'] and self.attn_args['f_dims']['sep'] > 0:
                self.trans_coeff = nn.Parameter(torch.Tensor([0.01]))

        self.lin_out = nn.Linear(
            attdim, dim_out) if dim_out is not None else lambda x: x

        self.output_scaler = output_scaler
        if output_scaler:
            self.scaler = nn.Parameter(torch.Tensor((output_scaler,)))

        if self.method == 'gbt':
            self.lin_ray = nn.Linear(180, attdim)

        if self.method == 'frustum_posemb':
            D = self.attn_args['D']
            indim = D*4
            self.frustum_phi = nn.Sequential(
                nn.Linear(indim, attdim*2),
                nn.ReLU(),
                nn.Linear(attdim*2, attdim))

    def upsample_tokens(self, tokens, h, w, k, extras):
        """
        Args:
            tokens: Tensor of shape: [B, Nk*h*w, C]
        Return
            Tensor of shape: [B, (2**k*h)*(2**k*w), C]
        """
        input_coord = extras['input_coord']
        Nk = input_coord.shape[1]
        factor = 2**k
        tokens = rearrange(tokens, 'b (Nk h w) (f1 f2 c) ->b (Nk h f1 w f2) c',
                           Nk=Nk, h=h, w=w, f1=factor, f2=factor)
        pos_w = torch.arange(0., 2**k)/(2**k) / (h)
        pos_h = torch.arange(0., 2**k)/(2**k) / (w)
        local_coord = torch.meshgrid([pos_h, pos_w], indexing='ij')
        local_coord = torch.stack(local_coord, -1)
        local_coord = repeat(
            local_coord, 'f1 f2 c -> (h f1 w f2) c', h=h, w=w).to(tokens.device)
        upsampled_coord = repeat(
            input_coord, 'B Nk (h w) c -> B Nk (h f1 w f2) c', f1=2**k, f2=2**k, h=h, w=w)
        extras['input_coord'] = upsampled_coord + local_coord[None]
        return tokens


    def add_ray_embs_to_extras(self, x, rays, extras, downsample_factor=3):
        # For RePAST
        input_transforms = extras['input_transforms']
        B, Nq = input_transforms.shape[0], input_transforms.shape[1]

        # [B, Nq, (H//(2**dsfactor))*(W//(2**dsfactor)), 3]
        _rays = downsample(rays, downsample_factor).flatten(2, 3)
        # [B, Nq, (H//(2**dsfactor))*(W//(2**dsfactor)), 3]
        _x = x[:, :, None].expand(-1, -1, _rays.shape[2], -1)

        _x_key = rigid_transform(input_transforms, _x, trans_coeff=1.0)
        _rays_key = rigid_transform(input_transforms, _rays, trans_coeff=0.0)
        T = _x_key.shape[2]
        extras['key_ray_emb'] = self.ray_encoder(_x_key.flatten(
            1, 2), _rays_key.flatten(1, 2)).reshape(B, Nq, T, -1)

        # [B, Nq, (Nq*(H//(2**dsfactor))*(W//(2**dsfactor))), 3]
        _x_rep = _x.unsqueeze(1).expand(-1, Nq, -1, -1, -1).flatten(2, 3)
        _rays_rep = _rays.unsqueeze(1).expand(-1, Nq, -1, -1, -1).flatten(2, 3)
        _x_query = rigid_transform(input_transforms, _x_rep, trans_coeff=1.0)
        _rays_query = rigid_transform(
            input_transforms, _rays_rep, trans_coeff=0.0)
        T = _x_query.shape[2]
        extras['query_ray_emb'] = self.ray_encoder(_x_query.flatten(
            1, 2), _rays_query.flatten(1, 2)).reshape(B, Nq, T, -1).transpose(1, 2)

    def compute_gbt_biases(self, x, rays, extras, downsample_factor=3):
        # GBT
        # [B, Nq, (H//(2**dsfactor))*(W//(2**dsfactor)), 3]
        _rays = downsample(rays, downsample_factor).flatten(2, 3)
        # [B, Nq, (H//(2**dsfactor))*(W//(2**dsfactor)), 3]
        _x = x[:, :, None].expand(-1, -1, _rays.shape[2], -1)
        # [B, Nq, Tq, 6].
        plucker_ray = get_plucker_parameterization(torch.cat((_x, _rays), -1))
        plucker_ray = plucker_ray.reshape(plucker_ray.shape[0], -1, 6)
        extras['ray_input'] = plucker_ray  # [B, Nq, Tq, 6].
        dist = plucker_dist(plucker_ray, plucker_ray)  # [B, Nq*Tq, Nq*Tq]
        extras['plucker_dist'] = dist

        # return late-fusion pos emb;
        emb = positional_encoding(plucker_ray, parameterize=None)
        return emb

    def compute_frustum_posemb(self, extras):
        # already downsampled #[B, N, T, 2]
        input_coord = extras['input_coord']
        rel_cam = extras['input_transforms']  # [B, N, 4, 4]
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
        return emb

    def pre_compute_reps(self, attn_kwargs, extras):
        f_dims = attn_kwargs['f_dims']
        flattened_reps = []
        flattened_invreps = []
        if 'so2' in f_dims and f_dims['so2'] > 0:
            coord = extras['input_coord']
            coord = coord.reshape(coord.shape[0], -1, 2)  # [B, Nq*Tq, 2]
            so2rep = make_SO2mats(coord,
                                  nfreqs=attn_kwargs['so2'],
                                  max_freqs=[attn_kwargs['max_freq_h'],
                                             attn_kwargs['max_freq_w']],
                                  shared_freqs=attn_kwargs['shared_freqs'] if 'shared_freqs' in attn_kwargs else False)  # [B, Nq*Tq, deg, 2, 2, 2]
            so2rep = so2rep.flatten(-4, -3)
            NqTq = so2rep.shape[1]
            extras['so2rep_q'] = extras['so2rep_k'] = so2rep  # [B, T, C, 2, 2]
            extras['so2fn'] = lambda A, x: torch.einsum(
                'btcij,bhtcj->bhtci', A, x)
            flattened = so2rep.reshape(
                so2rep.shape[0], so2rep.shape[1], -1)  # [B, T, C*2*2]
            flattened_reps.append(flattened)
            # [B, T, C*2*2]
            flattened_inv = so2rep.transpose(-2, -1).reshape(
                so2rep.shape[0], so2rep.shape[1], -1)
            flattened_invreps.append(flattened_inv)

        if 't2' in f_dims and f_dims['t2'] > 0:
            coord = extras['input_coord']
            coord = coord.reshape(coord.shape[0], -1, 2)  # [B, Nq*Tq, 2]
            t2rep = make_T2mats(coord)  # [B, Nq*Tq, 2] -> [B, Nq*Tq, 3, 3]
            extras['t2rep_q'] = extras['t2rep_k'] = t2rep
            extras['inv_t2rep_q'] = torch.linalg.inv(t2rep)
            extras['t2fn'] = lambda A, x: torch.einsum(
                'btij,bhtcj->bhtci', A, x)

        if 'se3' in f_dims and f_dims['se3'] > 0:
            extrinsic = extras['input_transforms']
            se3rep = torch.linalg.inv(extrinsic)  # [B, Nq, 4, 4]
            if 'ray_to_se3' in attn_kwargs and attn_kwargs['ray_to_se3']:
                B, Nq = se3rep.shape[0], se3rep.shape[1]
                input_rays = downsample(
                    extras['input_rays'], 3).reshape(B, Nq, -1, 3)
                # [B, Nq, T, 4, 4]
                R = ray2rotation(input_rays, return_4x4=True)
                se3rep = torch.einsum(
                    'bnij,bntjk->bntik', se3rep, R.transpose(-2, -1))  # mul from right
                extrinsic = torch.einsum(
                    'bntij,bnjk->bntik',  R, extrinsic)  # mul from left
                extras['se3fn'] = lambda A, x: torch.einsum(
                    'bntij,bhntcj->bhntci', A, x)
            else:
                extras['se3fn'] = lambda A, x: torch.einsum(
                    'bnij,bhntcj->bhntci', A, x)
            extras['se3rep_q'] = extras['se3rep_k'] = se3rep
            extras['inv_se3rep_q'] = extrinsic

            flattened = extrinsic.repeat_interleave(
                NqTq//extrinsic.shape[1], 1).transpose(-2, -1).reshape(se3rep.shape[0], -1, 16)  # [B, T, 4*4]
            flattened_reps.append(flattened)
            flattened_inv = extrinsic.repeat_interleave(
                NqTq//extrinsic.shape[1], 1).reshape(se3rep.shape[0], -1, 16)  # [B, T, 4*4]
            flattened_invreps.append(flattened_inv)

        if 'so3' in f_dims and f_dims['so3'] > 0:
            n_degs = attn_kwargs['so3']
            R_q = torch.linalg.inv(extras['input_transforms'])[..., :3, :3]
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
            extras['so3rep_q'] = extras['so3rep_k'] = D_q
            extras['so3fn'] = lambda A, x: torch.einsum(
                'bnij,bhnkj->bhnki', A, x)

        if 'sep' in f_dims and f_dims['sep'] > 0:
            extras['trans_coeff'] = self.trans_coeff
            se3rep = torch.linalg.inv(
                extras['input_transforms'])  # [B, Nq, 4, 4]
            n_degs = attn_kwargs['so3']
            R_q = se3rep[..., :3, :3]
            B, Nq = se3rep.shape[0], se3rep.shape[1]
            input_rays = downsample(
                extras['input_rays'], 3).reshape(B, Nq, -1, 3)
            R = ray2rotation(input_rays)  # [B, Nq, T, 4, 4]
            R_q = torch.einsum('bnij,bntjk->bntik', R_q,
                               R.transpose(-2, -1))  # mul from right
            D_q = rotmat_to_wigner_d_matrices(n_degs, R_q.flatten(0, 2))[1:]
            for i, D in enumerate(D_q):
                D_q[i] = D.reshape(B, Nq, -1, D.shape[-2], D.shape[-1])
            extras['so3rep_q'] = extras['so3rep_k'] = D_q
            extras['so3fn'] = lambda A, x: torch.einsum(
                'bntij,bhntkjdm->bhntkidm', A, x)

            coord = se3rep[..., :3, 3]  # [B, Nq, 3]
            coord = extras['trans_coeff'] * coord
            so2rep = make_SO2mats(coord,
                                  nfreqs=attn_kwargs['so2'],
                                  max_freqs=attn_kwargs['max_freqs'])  # [B, Nq, deg, 3, 2, 2]
            so2rep = so2rep.flatten(-4, -3)  # [B, Nq, deg*3, 2, 2]
            extras['so2rep_q'] = extras['so2rep_k'] = so2rep
            extras['so2fn'] = lambda A, x: torch.einsum(
                'bndlm,bhntkjdm->bhntkjdl', A, x)

        extras['flattened_rep_q'] = extras['flattened_rep_k'] = torch.cat(
            flattened_reps, -1)  # 16 + 2*freqs*2*2
        extras['flattened_invrep_q'] = torch.cat(flattened_invreps, -1)


    def forward(self, images, camera_pos, rays, extras={}):
        """
        Args:
            images: [batch_size, num_images, 3, height, width]. Assume the first image is canonical.
            camera_pos: [batch_size, num_images, 3]
            rays: [batch_size, num_images, height, width, 3]
        Returns:
            scene representation: [batch_size, num_patches, channels_per_patch]
        """

        batch_size, num_images = images.shape[:2]

        if self.method == 'repast':
            self.add_ray_embs_to_extras(
                camera_pos, rays, extras, downsample_factor=3)
        if self.method == 'gbt':
            gbtemb = self.compute_gbt_biases(
                camera_pos, rays, extras, downsample_factor=3)  # Precompute ray distances
        if self.is_gta:
            self.pre_compute_reps(self.attn_args, extras)

        x = images.flatten(0, 1)
        camera_pos = camera_pos.flatten(0, 1)
        rays = rays.flatten(0, 1)

        if self.emb:
            if self.emb == 'ray':
                h, w = rays.shape[1], rays.shape[2]
                camera_pos = camera_pos[:, None, None].expand(-1, h, w, -1)
                emb = self.ray_encoder(camera_pos, rays).permute(0, 3, 1, 2)
                x = torch.cat((x, emb), 1)
            elif self.emb == 'planar':
                emb = positionalencoding2d(
                    180, x.shape[-2], x.shape[-1]).to(x.device)
                emb = emb[None].repeat(batch_size*num_images, 1, 1, 1)
                x = torch.cat((x, emb), 1)

        x = self.conv_blocks(x)
        x = self.per_patch_linear(x)

        H_attn, W_attn = x.shape[-2:]
        if self.method == 'ape' or self.method == 'mln':
            emb = positionalencoding2d(180, H_attn, W_attn).to(
                x.device).reshape(-1, 180)
            emb = emb[None, None].repeat(batch_size, num_images, 1, 1)
            extras['input_coord_emb'] = emb

        if self.emb == 'camera_planar':
            emb_2dpos = self.lin_planar(positionalencoding2d(
                180, H_attn, W_attn).to(x.device).permute(1, 2, 0)).permute(2, 0, 1)
            emb_2dpos = emb_2dpos[None].repeat(batch_size*num_images, 1, 1, 1)
            pose = extras['input_transforms']  # [B, Nq, 4, 4]
            emb_camera = self.lin_camera(
                pose[..., :3, :].reshape(-1, 12))[:, :, None, None].expand(-1, -1, H_attn, W_attn)
            x = x + emb_2dpos + emb_camera
        elif self.method == 'gbt':
            gbtemb = self.lin_ray(gbtemb)  # [B, N*H*W, C]
            gbtemb = gbtemb.reshape(x.shape[0], x.shape[2], x.shape[3], -1)
            gbtemb = gbtemb.permute(0, 3, 1, 2)
            x = x + gbtemb
        elif self.method == 'frustum_posemb':
            emb = self.compute_frustum_posemb(extras)  # [B, N, T, D*4]
            emb = emb.reshape(
                emb.shape[0]*emb.shape[1], H_attn, W_attn, -1).permute(0, 3, 1, 2)
            x = x + emb

        x = x.flatten(2, 3).permute(0, 2, 1)  # [B]

        patches_per_image, channels_per_patch = x.shape[1:]
        x = x.reshape(batch_size, num_images *
                      patches_per_image, channels_per_patch)
        x = self.transformer(x, None, extras)

        x = self.lin_out(x)

        if self.output_scaler:
            extras['scaler'] = self.scaler
        return x, extras