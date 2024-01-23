import torch
from torch import nn
from torch.distributions import Normal, Uniform
from torch.linalg import matrix_exp

from source.encoder import ImprovedSRTEncoder
from source.decoder import ImprovedSRTDecoder
from einops import rearrange, reduce, repeat
from source.utils.wigner_d import rotmat_to_wigner_d_matrices
from source.utils.nerf import get_vertical_rays, transform_points_torch
from source.utils.gta import scale_mask


class SRT(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        if 'encoder' in cfg and cfg['encoder'] == 'isrt':
            self.encoder = ImprovedSRTEncoder(**cfg['encoder_kwargs'])
        else:  # We leave the SRTEncoder as default for backwards compatibility
            raise ValueError('Unknown encoder type', cfg['encoder'])

        if cfg['decoder'] == 'isrt':
            self.decoder = ImprovedSRTDecoder(**cfg['decoder_kwargs'])
        else:
            raise ValueError('Unknown decoder type', cfg['decoder'])

    def forward(self, input_images, input_camera_pos, input_rays, target_camera_pos, target_rays, extras):
        z, extras = self.encoder(
            input_images, input_camera_pos, input_rays, extras)
        return self.decoder(z, target_camera_pos, target_rays, extras)

    def decode(self, z, x, rays, extras, **kwargs):
        return self.decoder(z, x, rays, extras)


class TransformingSRT(SRT):
    def __init__(self, cfg):
        self.ftl = cfg.get('ftl', False)
        super().__init__(cfg)

    def apply_batch_matmul(self, M, z):
        if len(M.shape) == 4:
            if len(z.shape) == 4:
                z = torch.einsum('nmij, nmkj->nmki', M, z)
            else:
                z = torch.einsum('nmij, nkj->nmki', M, z)
        else:
            z = torch.einsum('nij, nkj->nki', M, z)
        return z

    def decode(self, z, x, rays,  extras={}):
        """
        Args:
            z [n, k, c]: set structured latent variables, or dictionary that consits of part whole vecs
            camera_pos [n, nt, p, 3]: camera position
            rays [n, nt, p, 3]: ray directions
            transforms [n, nt, 4, 4]: 4x4 SE(3) matrices
            render_kwargs: kwargs passed on to decoder
        """
        if self.ftl:
            iT = extras['input_transforms']
            Ni = iT.shape[1]
            tT = extras['target_transforms']
            Nt = tT.shape[1]
            msk = scale_mask(self.trans_coeff, iT.device)
            iT = msk[None, None] * iT
            tT = msk[None, None] * tT
            B, T, C = z.shape
            z = z.reshape(B, Ni, -1, C // 4, 4)
            z = torch.einsum('bnij,bntcj->bntci', torch.linalg.inv(iT), z)
            target_coord = extras['target_coord']  # [B, N, T, 2]
            pixels_list = []
            for n in range(Nt):
                z_t = torch.einsum('bij,bntcj->bntci', tT[:, n], z)
                z_t = z_t.reshape(B, T, C)
                extras['target_coord'] = target_coord[:, n]
                pixels, _ = self.decoder(z_t, x[:, n], rays[:, n], extras)
                pixels_list.append(pixels)
            return torch.stack(pixels_list, 1).flatten(1, 2), extras
        else:
            if len(x.shape) == 4:
                x = x.flatten(1, 2)
                rays = rays.flatten(1, 2)

            return self.decoder(z, x, rays, extras)

    def forward(self, input_images, input_camera_pos, input_rays, target_camera_pos, target_rays, extras={}):
        z, extras = self.encoder(
            input_images, input_camera_pos, input_rays, extras)
        return self.decode(z, target_camera_pos, target_rays, extras=extras)
