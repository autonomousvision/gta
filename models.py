import torch.nn.functional as F
import time
import torch
import torch.nn as nn
import numpy as np
from midas import dpt_depth, midas_net, midas_net_custom

from utils import util

import geometry
from epipolar import project_rays
from encoder import SpatialEncoder, ImageEncoder, UNetEncoder
from resnet_block_fc import ResnetFC
import timm
from collections import OrderedDict
from copy import deepcopy
from midas.vit import make_SO2mats
from wigner_d import rotmat_to_wigner_d_matrices

def encode_relative_ray(ray, transform):
    s = ray.size()
    b, ncontext = transform.size()[:2]

    ray = ray.view(b, ncontext, *s[1:])
    ray = (ray[:, :, :, :, None, :] * transform[:, :, None, None, :3, :3]).sum(dim=-1)

    ray = ray.view(*s)
    return ray


def encode_relative_point(ray, transform):
    s = ray.size()
    b, ncontext = transform.size()[:2]

    ray = ray.view(b, ncontext, *s[1:])
    ray = torch.cat([ray, torch.ones_like(ray[..., :1])], dim=-1)
    ray = (ray[:, :, :, :, None, :] * transform[:, :, None, None, :4, :4]).sum(dim=-1)[..., :3]

    ray = ray.view(*s)
    return ray


class CrossAttentionRenderer(nn.Module):
    def __init__(self, 
                 no_sample=False, 
                 no_latent_concat=False,
                 no_multiview=False, 
                 no_high_freq=False, 
                 model="midas_vit", 
                 uv=None, 
                 repeat_attention=True, 
                 n_view=1, 
                 npoints=64, 
                 num_hidden_units_phi=128,
                 GTA=False,
                 kv_trnsfm=False):
        super().__init__()
        self.GTA = GTA
        self.n_view = n_view
        self.so3 = True
        self.kv_trnsfm = kv_trnsfm

        if self.n_view == 2 or self.n_view == 1:
            self.npoints = 64
        else:
            self.npoints = 48

        if npoints:
            self.npoints = npoints

        self.repeat_attention = repeat_attention

        self.no_sample = no_sample
        self.no_latent_concat = no_latent_concat
        self.no_multiview = no_multiview
        self.no_high_freq = no_high_freq

        if model == "resnet":
            self.encoder = SpatialEncoder(use_first_pool=False, num_layers=4)
            self.latent_dim = 512
        elif model == 'midas':
            self.encoder = midas_net_custom.MidasNet_small(
                path=None,
                features=64,
                backbone="efficientnet_lite3",
                exportable=True,
                non_negative=True,
                blocks={'expand': True}
            )
            checkpoint = (
                    "https://github.com/AlexeyAB/MiDaS/releases/download/midas_dpt/midas_v21_small-70d6b9c8.pt"
            )
            state_dict = torch.hub.load_state_dict_from_url(
                checkpoint, map_location=torch.device('cpu'), progress=True, check_hash=True
            )
            self.encoder.load_state_dict(state_dict)
            self.latent_dim = 512
        elif model == 'midas_vit':
            self.encoder = dpt_depth.DPTDepthModel(
                path=None,
                backbone="vitb_rn50_384",
                non_negative=True,
                GTA=GTA,
                so3=self.so3,
            )
            checkpoint = (
                "https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt"
            )

            self.encoder.pretrained.model.patch_embed.backbone.stem.conv = timm.models.layers.std_conv.StdConv2dSame(3, 64, kernel_size=(7, 7), stride=(2, 2), bias=False)
            self.latent_dim = 512 + 64

            self.conv_map = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3)
        else:
            self.encoder = UNetEncoder()
            self.latent_dim = 32

        if self.n_view > 1 and (not self.no_latent_concat):
            self.query_encode_latent = nn.Conv2d(self.latent_dim + 3, self.latent_dim, 1)
            self.query_encode_latent_2 = nn.Conv2d(self.latent_dim, self.latent_dim // 2 , 1)
            self.latent_dim = self.latent_dim // 2
            self.update_val_merge = nn.Conv2d(self.latent_dim * 2 + 6, self.latent_dim, 1)
        elif self.no_latent_concat:
            self.feature_map = nn.Conv2d(self.latent_dim, self.latent_dim // 2 , 1)
        else:
            self.update_val_merge = nn.Conv2d(self.latent_dim + 6, self.latent_dim, 1)

        self.model = model
        self.num_hidden_units_phi = num_hidden_units_phi

        hidden_dim = 128 

        if not self.no_latent_concat:
            self.latent_value = nn.Conv2d(self.latent_dim * self.n_view, self.latent_dim, 1)
            self.key_map = nn.Conv2d(self.latent_dim * self.n_view, hidden_dim, 1)
            self.key_map_2 = nn.Conv2d(hidden_dim, hidden_dim, 1)
        else:
            self.latent_value = nn.Conv2d(self.latent_dim, self.latent_dim, 1)
            self.key_map = nn.Conv2d(self.latent_dim, hidden_dim, 1)
            self.key_map_2 = nn.Conv2d(hidden_dim, hidden_dim, 1)

        if self.GTA:
            self.initial_emb = nn.Parameter(0.*torch.zeros(self.latent_dim))
            self.initial_query_embed = nn.Linear(self.latent_dim, hidden_dim)
            self.query_embed = nn.Conv2d(4, hidden_dim, 1)
            self.query_embed_2 = nn.Conv2d(hidden_dim, hidden_dim, 1)
        else:
            self.query_embed = nn.Conv2d(16, hidden_dim, 1)
            self.query_embed_2 = nn.Conv2d(hidden_dim, hidden_dim, 1)

        self.hidden_dim = hidden_dim

        self.latent_avg_query = nn.Conv2d(9+16, hidden_dim, 1)
        self.latent_avg_query_2 = nn.Conv2d(hidden_dim, hidden_dim, 1)

        self.latent_avg_key = nn.Conv2d(self.latent_dim, hidden_dim, 1)
        self.latent_avg_key_2 = nn.Conv2d(hidden_dim, hidden_dim, 1)

        self.query_repeat_embed = nn.Conv2d(16+128 if not self.GTA else 4+128, hidden_dim, 1)
        self.query_repeat_embed_2 = nn.Conv2d(hidden_dim, hidden_dim, 1)

        self.latent_avg_repeat_query = nn.Conv2d(9+16+128, hidden_dim, 1)
        self.latent_avg_repeat_query_2 = nn.Conv2d(hidden_dim, hidden_dim, 1)

        self.encode_latent = nn.Conv1d(self.latent_dim, 128, 1)

        self.phi = ResnetFC(self.n_view * 9 if not self.GTA else 0, n_blocks=3, d_out=3,
                            d_latent=self.latent_dim * self.n_view, d_hidden=self.num_hidden_units_phi)
        


    def get_z(self, input, val=False):
        # self.normalize_input(input)
        rgb = input['context']['rgb']
        intrinsics = input['context']['intrinsics']
        context = input['context']

        cam2world = context['cam2world']
        rel_cam2world = torch.matmul(torch.inverse(cam2world[:, :1]), cam2world) # C^{-1}[:, 0] C [:, 0:2]

        # Flatten first two dims (batch and number of context)
        rgb = torch.flatten(rgb, 0, 1)
        intrinsics = torch.flatten(intrinsics, 0, 1)
        intrinsics = intrinsics[:, None, :, :]
        rgb = rgb.permute(0, -1, 1, 2) # (b*n_ctxt, ch, H, W)
        self.H, self.W = rgb.shape[-2], rgb.shape[-1]

        if self.model == "resnet":
            rgb = (rgb + 1) / 2.
            rgb = util.normalize_imagenet(rgb)
            rgb = torch.cat([rgb], dim=1)
        elif self.model == "midas" or self.model == "midas_vit":
            rgb = (rgb + 1) / 2
            rgb = util.normalize_imagenet(rgb)

        if self.no_multiview:
            cam2world_encode = rel_cam2world.view(-1, 16)
            cam2world_encode = torch.zeros_like(cam2world_encode)
        else:
            cam2world_encode = rel_cam2world.view(-1, 16)

        z = self.encoder.forward(rgb, cam2world_encode, self.n_view) # (b*n_ctxt, self.latent_dim, H, W)

        if self.model == "midas" or self.model == "midas_vit":
            z_conv = self.conv_map(rgb)

            if self.no_high_freq:
                z_conv = torch.zeros_like(z_conv)

            z = z + [z_conv]

        return z

    def forward(self, input, z=None, val=False, debug=False):

        out_dict = {}
        input = deepcopy(input)

        query = input['query']
        context = input['context']
        b, n_context = input['context']["rgb"].shape[:2]

        # Query rayes 
        n_qry, n_qry_rays = query["uv"].shape[1:3]

        # Get img features
        if z is None:
            z = z_orig = self.get_z(input) 
        else:
            z_orig = z
        
        # Get relative coordinates of the query and context ray in each context camera coordinate system
        context_cam2world = torch.matmul(torch.inverse(context['cam2world']), context['cam2world'])
        query_cam2world = torch.matmul(torch.inverse(context['cam2world']), query['cam2world'])

        # Compute each context relative to the first view (not used)
        #context_rel_cam2world = torch.matmul(torch.inverse(context['cam2world'][:, :1]), context['cam2world'])

        # R3xS2 to Plucker coordinate
        lf_coords = geometry.plucker_embedding(
            torch.flatten(query_cam2world, 0, 1), 
            torch.flatten(query['uv'].expand(-1, query_cam2world.size(1), -1, -1).contiguous(), 0, 1), 
            torch.flatten(query['intrinsics'].expand(-1, query_cam2world.size(1), -1, -1).contiguous(), 0, 1))
        lf_coords = lf_coords.reshape(b, n_context, n_qry_rays, 6) # Typically [B, n_views=2, 192, 6] 

        lf_coords.requires_grad_(True)
        out_dict['coords'] = lf_coords.reshape(b*n_context, n_qry_rays, 6)
        out_dict['uv'] = query['uv']
        # Compute epi line
        if self.no_sample:
            start, end, diff, valid_mask, pixel_val = geometry.get_epipolar_lines_volumetric(lf_coords, query_cam2world, context['intrinsics'], self.H, self.W, self.npoints, debug=debug)
        else:

            # Prepare arguments for epipolar line computation
            intrinsics_norm = context['intrinsics'].clone()
            # Normalize intrinsics for a 0-1 image
            intrinsics_norm[:, :, :2, :] = intrinsics_norm[:, :, :2, :] / self.H

            camera_origin = geometry.get_ray_origin(query_cam2world) # just extracts translational elements of the SE(3) mats
            ray_dir = lf_coords[..., :3] # d of (d, x) [B, N=2, 192, 3]
            # Extrinsics (but all identites) [B, N=2, 4, 4]
            extrinsics = torch.eye(4).to(ray_dir.device)[None, None, :, :].expand(ray_dir.size(0), ray_dir.size(1), -1, -1) 
            camera_origin = camera_origin[:, :, None, :].expand(-1, -1, ray_dir.size(2), -1) # [B, N=2, 192, 3]

            s = camera_origin.size()

            # Compute 2D epipolar line samples for the image
            output = project_rays(torch.flatten(camera_origin, 0, 1), torch.flatten(ray_dir, 0, 1), torch.flatten(extrinsics, 0, 1), torch.flatten(intrinsics_norm, 0, 1))

            valid_mask = output['overlaps_image'] # 
            start, end = output['xy_min'], output['xy_max']

            start = start.view(*s[:2], *start.size()[1:])
            end = end.view(*s[:2], *end.size()[1:])
            valid_mask = valid_mask.view(*s[:2], valid_mask.size(1))
            start = (start - 0.5) * 2 #range: [-1, 1] is valid
            end = (end - 0.5) * 2 #range: [-1, 1]  is valid


            start[torch.isnan(start)] = 0
            start[torch.isinf(start)] = 0
            end[torch.isnan(end)] = 0
            end[torch.isinf(end)] = 0

            #diff = end - start

            valid_mask = valid_mask.float()
            start = start[..., :2]
            end = end[..., :2]

        #diff = end - start
        interval = torch.linspace(0, 1, self.npoints, device=lf_coords.device) # get uniform samples along epipolar line on each image

        if (not self.no_sample):
            pixel_val = None
        else:
            pixel_val = torch.flatten(pixel_val, 0, 1)

        latents_out = []
        at_wts = []

        diff = end[:, :, :, None, :] - start[:, :, :, None, :]

        if pixel_val is None and (not self.no_sample):
            pixel_val = start[:, :, :, None, :] + diff * interval[None, None, None, :, None]
            pixel_val = torch.flatten(pixel_val, 0, 1)

        # Gather corresponding features on line. Image features 
        interp_val_orig = interp_val = torch.cat([F.grid_sample(latent, pixel_val, mode='bilinear', padding_mode='border', align_corners=False) for latent in z], dim=1)

        # Find the 3D point correspondence in every other camera view
        if self.n_view == 2 and (not self.no_latent_concat):
            # Find the nearest neighbor latent in the other frame when given 2 views
            # z is a list of 3 different resolution features
            pt, _, _, _ = geometry.get_3d_point_epipolar(
                lf_coords.flatten(0, 1), 
                pixel_val, 
                context_cam2world.flatten(0, 1), 
                self.H, 
                self.W, 
                context['intrinsics'].flatten(0, 1))

            context_rel_cam2world_view1 = torch.matmul(torch.inverse(context['cam2world'][:, 0:1]), context['cam2world'])
            context_rel_cam2world_view2 = torch.matmul(torch.inverse(context['cam2world'][:, 1:2]), context['cam2world'])

            pt_view1 = encode_relative_point(pt, context_rel_cam2world_view1)
            pt_view2 = encode_relative_point(pt, context_rel_cam2world_view2)

            intrinsics_view1 = context['intrinsics'][:, 0]
            intrinsics_view2 = context['intrinsics'][:, 1]

            s = pt_view1.size()
            pt_view1 = pt_view1.view(b, n_context, *s[1:])
            pt_view2 = pt_view2.view(b, n_context, *s[1:])

            s = interp_val.size()
            interp_val = interp_val.view(b, n_context, *s[1:])

            interp_val_1 = interp_val[:, 0]
            interp_val_2 = interp_val[:, 1]

            pt_view1_context1 = pt_view1[:, 0]
            pt_view1_context2 = pt_view1[:, 1]

            pt_view2_context1 = pt_view2[:, 0]
            pt_view2_context2 = pt_view2[:, 1]

            pixel_val_view2_context1 = geometry.project(pt_view2_context1[..., 0], pt_view2_context1[..., 1], pt_view2_context1[..., 2], intrinsics_view2)
            pixel_val_view2_context1 = util.normalize_for_grid_sample(pixel_val_view2_context1[..., :2], self.H, self.W)

            pixel_val_view1_context2 = geometry.project(pt_view1_context2[..., 0], pt_view1_context2[..., 1], pt_view1_context2[..., 2], intrinsics_view1)
            pixel_val_view1_context2 = util.normalize_for_grid_sample(pixel_val_view1_context2[..., :2], self.H, self.W)

            pixel_val_stack = torch.stack([pixel_val_view1_context2, pixel_val_view2_context1], dim=1).flatten(0, 1)
            interp_val_nearest = torch.cat(
                [F.grid_sample(latent, pixel_val_stack, mode='bilinear', padding_mode='zeros', align_corners=False) 
                 for latent in z],
                dim=1)
            interp_val_nearest = interp_val_nearest.view(b, n_context, *s[1:])
            interp_val_nearest_1 = interp_val_nearest[:, 0]
            interp_val_nearest_2 = interp_val_nearest[:, 1]

            pt_view1_context1 = torch.nan_to_num(pt_view1_context1, 0)
            pt_view2_context2 = torch.nan_to_num(pt_view2_context2, 0)
            pt_view1_context2 = torch.nan_to_num(pt_view1_context2, 0)
            pt_view2_context1 = torch.nan_to_num(pt_view2_context1, 0)

            pt_view1_context1 = pt_view1_context1.detach()
            pt_view2_context2 = pt_view2_context2.detach()

            if self.GTA:
                interp_val_1_view_1 = torch.cat([interp_val_1, torch.tanh(pt_view1_context1 / 5.).permute(0, 3, 1, 2)], dim=1)
                interp_val_1_view_2 = torch.cat([interp_val_nearest_2, torch.tanh(pt_view2_context1 / 5.).permute(0, 3, 1, 2)], dim=1)
                interp_val_1_encode_1 = self.query_encode_latent_2(F.relu(self.query_encode_latent(interp_val_1_view_1)))
                interp_val_1_encode_2 = self.query_encode_latent_2(F.relu(self.query_encode_latent(interp_val_1_view_2)))

                interp_val_1_avg = torch.stack([interp_val_1_encode_1, interp_val_1_encode_2], dim=1).flatten(1, 2)
                
                interp_val_2_view_2 = torch.cat([interp_val_2, torch.tanh(pt_view2_context2 / 5.).permute(0, 3, 1, 2)], dim=1)
                interp_val_2_view_1 = torch.cat([interp_val_nearest_1, torch.tanh(pt_view1_context2 / 5.).permute(0, 3, 1, 2)], dim=1)
                interp_val_2_encode_1 = self.query_encode_latent_2(F.relu(self.query_encode_latent(interp_val_2_view_1)))
                interp_val_2_encode_2 = self.query_encode_latent_2(F.relu(self.query_encode_latent(interp_val_2_view_2)))
                # val2enc = [val2enc2, val2enc1]
                interp_val_2_avg = torch.stack([interp_val_2_encode_2, interp_val_2_encode_1], dim=1).flatten(1, 2)

                # shape: [B*N, 2*C, query, points]
                interp_val = torch.stack([interp_val_1_avg, interp_val_2_avg], dim=1).flatten(0, 1)


            else:
                interp_val_1_view_1 = torch.cat([interp_val_1, torch.tanh(pt_view1_context1 / 5.).permute(0, 3, 1, 2)], dim=1)
                interp_val_1_view_2 = torch.cat([interp_val_nearest_2, torch.tanh(pt_view2_context1 / 5.).permute(0, 3, 1, 2)], dim=1)

                interp_val_1_encode_1 = self.query_encode_latent_2(F.relu(self.query_encode_latent(interp_val_1_view_1)))
                interp_val_1_encode_2 = self.query_encode_latent_2(F.relu(self.query_encode_latent(interp_val_1_view_2)))
                interp_val_1_avg = torch.stack([interp_val_1_encode_1, interp_val_1_encode_2], dim=1).flatten(1, 2)

                interp_val_2_view_2 = torch.cat([interp_val_2, torch.tanh(pt_view2_context2 / 5.).permute(0, 3, 1, 2)], dim=1)
                interp_val_2_view_1 = torch.cat([interp_val_nearest_1, torch.tanh(pt_view1_context2 / 5.).permute(0, 3, 1, 2)], dim=1)

                interp_val_2_encode_1 = self.query_encode_latent_2(F.relu(self.query_encode_latent(interp_val_2_view_1)))
                interp_val_2_encode_2 = self.query_encode_latent_2(F.relu(self.query_encode_latent(interp_val_2_view_2)))

                interp_val_2_avg = torch.stack([interp_val_2_encode_1, interp_val_2_encode_2], dim=1).flatten(1, 2)

                interp_val = torch.stack([interp_val_1_avg, interp_val_2_avg], dim=1).flatten(0, 1)
        elif (self.n_view == 3) and not self.no_latent_concat:
            # Find the nearest neighbor latent in the other 2 frames when given 3 views
            pt, _, _, _ = geometry.get_3d_point_epipolar(lf_coords.flatten(0, 1), pixel_val, context_cam2world.flatten(0, 1), self.H, self.W, context['intrinsics'].flatten(0, 1))

            context_rel_cam2world_view1 = torch.matmul(torch.inverse(context['cam2world'][:, 0:1]), context['cam2world'])
            context_rel_cam2world_view2 = torch.matmul(torch.inverse(context['cam2world'][:, 1:2]), context['cam2world'])
            context_rel_cam2world_view3 = torch.matmul(torch.inverse(context['cam2world'][:, 2:3]), context['cam2world'])

            pt_view1 = encode_relative_point(pt, context_rel_cam2world_view1)
            pt_view2 = encode_relative_point(pt, context_rel_cam2world_view2)
            pt_view3 = encode_relative_point(pt, context_rel_cam2world_view3)

            intrinsics_view1 = context['intrinsics'][:, 0]
            intrinsics_view2 = context['intrinsics'][:, 1]
            intrinsics_view3 = context['intrinsics'][:, 2]

            s = pt_view1.size()
            pt_view1 = pt_view1.view(b, n_context, *s[1:])
            pt_view2 = pt_view2.view(b, n_context, *s[1:])
            pt_view3 = pt_view3.view(b, n_context, *s[1:])

            s = interp_val.size()
            interp_val = interp_val.view(b, n_context, *s[1:])
            interp_val_1 = interp_val[:, 0]
            interp_val_2 = interp_val[:, 1]
            interp_val_3 = interp_val[:, 2]

            pt_view1_context1 = pt_view1[:, 0]
            pt_view1_context2 = pt_view1[:, 1]
            pt_view1_context3 = pt_view1[:, 2]

            pt_view2_context1 = pt_view2[:, 0]
            pt_view2_context2 = pt_view2[:, 1]
            pt_view2_context3 = pt_view2[:, 2]

            pt_view3_context1 = pt_view3[:, 0]
            pt_view3_context2 = pt_view3[:, 1]
            pt_view3_context3 = pt_view3[:, 2]

            # Compute the coordinates to gather for view 2 and 3 on view 1
            pt_view1_context = torch.flatten(torch.stack([pt_view2_context1, pt_view3_context1], dim=1), 1, 2)
            pt_view2_context = torch.flatten(torch.stack([pt_view1_context2, pt_view3_context2], dim=1), 1, 2)
            pt_view3_context = torch.flatten(torch.stack([pt_view1_context3, pt_view2_context3], dim=1), 1, 2)


            pixel_val_view2_context = geometry.project(pt_view2_context[..., 0], pt_view2_context[..., 1], pt_view2_context[..., 2], intrinsics_view2)
            pixel_val_view2_context = util.normalize_for_grid_sample(pixel_val_view2_context[..., :2], self.H, self.W)

            pixel_val_view1_context = geometry.project(pt_view1_context[..., 0], pt_view1_context[..., 1], pt_view1_context[..., 2], intrinsics_view1)
            pixel_val_view1_context = util.normalize_for_grid_sample(pixel_val_view1_context[..., :2], self.H, self.W)

            pixel_val_view3_context = geometry.project(pt_view3_context[..., 0], pt_view3_context[..., 1], pt_view3_context[..., 2], intrinsics_view3)
            pixel_val_view3_context = util.normalize_for_grid_sample(pixel_val_view3_context[..., :2], self.H, self.W)

            pixel_val_stack = torch.stack([pixel_val_view1_context, pixel_val_view2_context, pixel_val_view3_context], dim=1).flatten(0, 1)
            interp_val_nearest = torch.cat([F.grid_sample(latent, pixel_val_stack, mode='bilinear', padding_mode='zeros', align_corners=False) for latent in z], dim=1)

            s = interp_val_nearest.size()
            interp_val_nearest = interp_val_nearest.view(s[0] // 3, 3, *s[1:])

            interp_val_nearest_1 = interp_val_nearest[:, 0]
            interp_val_nearest_2 = interp_val_nearest[:, 1]
            interp_val_nearest_3 = interp_val_nearest[:, 2]

            # Features on each point
            interp_val_view_2_context_1, interp_val_view_3_context_1 = torch.chunk(interp_val_nearest_1, 2, dim=2)
            interp_val_view_1_context_2, interp_val_view_3_context_2 = torch.chunk(interp_val_nearest_2, 2, dim=2)
            interp_val_view_1_context_3, interp_val_view_2_context_3 = torch.chunk(interp_val_nearest_3, 2, dim=2)

            # Gather the right 3D pts along each image
            pt_view1_context = torch.flatten(torch.stack([pt_view1_context2, pt_view1_context3], dim=1), 1, 2)
            pt_view2_context = torch.flatten(torch.stack([pt_view2_context1, pt_view2_context3], dim=1), 1, 2)
            pt_view3_context = torch.flatten(torch.stack([pt_view3_context1, pt_view3_context2], dim=1), 1, 2)

            interp_val_nearest_1 = torch.cat([interp_val_view_1_context_2, interp_val_view_1_context_3], dim=2)
            interp_val_nearest_2 = torch.cat([interp_val_view_2_context_1, interp_val_view_2_context_3], dim=2)
            interp_val_nearest_3 = torch.cat([interp_val_view_3_context_1, interp_val_view_3_context_2], dim=2)

            pt_view1_context1 = torch.nan_to_num(pt_view1_context1, 0)
            pt_view2_context2 = torch.nan_to_num(pt_view2_context2, 0)
            pt_view3_context3 = torch.nan_to_num(pt_view3_context3, 0)

            pt_view1_context = torch.nan_to_num(pt_view1_context, 0)
            pt_view2_context = torch.nan_to_num(pt_view2_context, 0)
            pt_view3_context = torch.nan_to_num(pt_view3_context, 0)

            pt_view1_context = pt_view1_context.detach()
            pt_view2_context = pt_view2_context.detach()
            pt_view3_context = pt_view3_context.detach()

            # Compute average latent for first view
            interp_val_1_view_1 = torch.cat([interp_val_1, torch.tanh(pt_view1_context1 / 5.).permute(0, 3, 1, 2)], dim=1)
            interp_val_1_view_context = torch.cat([interp_val_nearest_1, torch.tanh(pt_view1_context / 5.).permute(0, 3, 1, 2)], dim=1)

            interp_val_1_encode_1 = self.query_encode_latent_2(F.relu(self.query_encode_latent(interp_val_1_view_1)))
            interp_val_1_encode_context = self.query_encode_latent_2(F.relu(self.query_encode_latent(interp_val_1_view_context)))

            interp_val_1_encode_1 = interp_val_1_encode_1[:, :, None, :, :]
            s = interp_val_1_encode_context.size()
            interp_val_1_encode_context = interp_val_1_encode_context.view(s[0], s[1], 2, s[2] // 2, s[3])

            interp_val_1_avg = torch.cat([interp_val_1_encode_1, interp_val_1_encode_context], dim=2).flatten(1, 2)

            # Compute average latent for second view
            interp_val_2_view_2 = torch.cat([interp_val_2, torch.tanh(pt_view2_context2 / 5.).permute(0, 3, 1, 2)], dim=1)
            interp_val_2_view_context = torch.cat([interp_val_nearest_2, torch.tanh(pt_view2_context / 5.).permute(0, 3, 1, 2)], dim=1)

            interp_val_2_encode_2 = self.query_encode_latent_2(F.relu(self.query_encode_latent(interp_val_2_view_2)))
            interp_val_2_encode_context = self.query_encode_latent_2(F.relu(self.query_encode_latent(interp_val_2_view_context)))

            interp_val_2_encode_2 = interp_val_2_encode_2[:, :, None, :, :]
            s = interp_val_2_encode_context.size()
            interp_val_2_encode_context = interp_val_2_encode_context.view(s[0], s[1], 2, s[2] // 2, s[3])

            interp_val_2_avg = torch.cat([interp_val_2_encode_2, interp_val_2_encode_context], dim=2).flatten(1, 2)

            # Compute average latent for third view

            interp_val_3_view_3 = torch.cat([interp_val_3, torch.tanh(pt_view3_context3 / 5.).permute(0, 3, 1, 2)], dim=1)
            interp_val_3_view_context = torch.cat([interp_val_nearest_3, torch.tanh(pt_view3_context / 5.).permute(0, 3, 1, 2)], dim=1)

            interp_val_3_encode_3 = self.query_encode_latent_2(F.relu(self.query_encode_latent(interp_val_3_view_3)))
            interp_val_3_encode_context = self.query_encode_latent_2(F.relu(self.query_encode_latent(interp_val_3_view_context)))

            interp_val_3_encode_3 = interp_val_3_encode_3[:, :, None, :, :]
            s = interp_val_3_encode_context.size()
            interp_val_3_encode_context = interp_val_3_encode_context.view(s[0], s[1], 2, s[2] // 2, s[3])

            interp_val_3_avg = torch.cat([interp_val_3_encode_3, interp_val_3_encode_context], dim=2).flatten(1, 2)

            interp_val = torch.stack([interp_val_1_avg, interp_val_2_avg, interp_val_3_avg], dim=1).flatten(0, 1)
        elif self.no_latent_concat:
            pass
        else:
            # Find the nearest neighbor latent for a single view (null operation)
            pt, _, _, _ = geometry.get_3d_point_epipolar(lf_coords.flatten(0, 1), pixel_val, context_cam2world.flatten(0, 1), self.H, self.W, context['intrinsics'].flatten(0, 1))

            pt[torch.isnan(pt)] = 0
            pt_context = torch.cat([torch.tanh(pt / 5.), torch.tanh(pt / 100.)], dim=-1)
            interp_val = torch.cat([interp_val, pt_context.permute(0, 3, 1, 2)], dim=1)
            interp_val = self.update_val_merge(interp_val)

        pt, dist, parallel, equivalent = geometry.get_3d_point_epipolar(lf_coords.flatten(0, 1), pixel_val, context_cam2world.flatten(0, 1), self.H, self.W, context['intrinsics'].flatten(0, 1))

         # Get camera ray direction of each epipolar pixel coordinate 
        cam_rays = geometry.get_ray_directions_cam(pixel_val, context['intrinsics'].flatten(0, 1), self.H, self.W)

        # Ray direction of the query ray to be rendered 
        ray_dir = lf_coords[..., :3].flatten(0, 1)
        ray_dir = ray_dir[:, :, None]
        ray_dir = ray_dir.expand(-1, -1, cam_rays.size(2), -1)

        # 3D coordinate of each epipolar point in 3D
        # depth, _, _ = geometry.get_depth_epipolar(lf_coords.flatten(0, 1), pixel_val, query_cam2world, self.H, self.W, context['intrinsics'].flatten(0, 1))

        # Compute the origin of the query ray
        query_ray_orig = geometry.get_ray_origin(query_cam2world).flatten(0, 1)
        query_ray_orig = query_ray_orig[:, None, None]
        query_ray_orig_ex = torch.broadcast_to(query_ray_orig, cam_rays.size())

        cam_origin = torch.zeros_like(query_ray_orig_ex)
        # Compute depth of the computed 3D coordinate (with respect to query camera)
        depth = torch.norm(pt - query_ray_orig, p=2, dim=-1)[..., None]

        # Set NaN and large depth values to a finite value
        depth[torch.isnan(depth)] = 1000000
        depth[torch.isinf(depth)] = 1000000
        depth = depth.detach()

        # Encode depth with tanh to encode different scales of depth values depth values
        depth_encode = torch.cat([torch.tanh(depth), torch.tanh(depth / 10.), torch.tanh(depth / 100.), torch.tanh(depth / 1000.)], dim=-1)

        # Compute query coordinates by combining context ray info, query ray info, and 3D depth of epipolar line
        if self.GTA:
            local_coords = depth_encode.permute(0, 3, 1, 2)
        else:
            local_coords = torch.cat([cam_rays, cam_origin, ray_dir, depth_encode, query_ray_orig_ex], dim=-1).permute(0, 3, 1, 2)
        coords_embed = self.query_embed_2(F.relu(self.query_embed(local_coords)))

        if self.GTA:
            enable_so3 = self.so3 
            def scale_mask(trans_coeff, device):
                msk = trans_coeff * torch.ones(size=(3, 1)).to(device)
                msk = torch.cat([torch.ones(size=(3, 3)).to(device), msk], -1)
                msk = torch.cat([msk, torch.Tensor([[0, 0, 0, 1]]).to(device)], -2)
                return msk

            v = self.latent_value(interp_val) # (B*N, C, Tq, Tk)
            k = self.key_map_2(F.relu(self.key_map(interp_val))) # (B*N, C, Tq, Tk)
            x = self.initial_emb[None, :].repeat(k.shape[0], 1)
            q = self.initial_query_embed(x)[:, :, None, None].repeat(1, 1, k.shape[2], k.shape[3]) + coords_embed
            BN, C, Tq, Tk = q.shape[0], q.shape[1], q.shape[2], q.shape[3]
            C_v = v.shape[1]

            uv_q = query['uv']
            uv_q = torch.stack([uv_q[..., 0] / self.H, uv_q[..., 1] / self.W,], -1).repeat(1, self.n_view, 1, 1).reshape(-1, Tq, 2) #[B*Nk, Tq, 2]
            uv_q = uv_q.unsqueeze(2).repeat(1, 1, Tk, 1) #[B*Nk, Tq, Tk, 2]
            uv_k = pixel_val / 2.0 + 0.5 #[B*Nk, Tq, Tk, 2]
            
            freqs = C//8 if not self.so3 else C//16
            freqs_v = C_v//8 if not self.so3 else C_v//16
            measure_time = False
            if measure_time:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
            if self.kv_trnsfm:
                rep_so2_k = make_SO2mats(uv_k-uv_q, freqs).reshape(BN, Tq, Tk, 2*freqs, 2, 2) # [B*Nk, Tq, Tk, C//4, 2, 2]
                rep_so2_v = make_SO2mats(uv_k-uv_q, freqs_v).reshape(BN, Tq, Tk, 2*freqs_v, 2, 2) # [B*Nk, Tq, Tk, C//4, 2, 2]
            else:
                rep_so2_q = make_SO2mats(uv_q, freqs).reshape(BN, Tq, Tk, 2*freqs, 2, 2) # [B*Nk, Tq, Tk, C//4, 2, 2]
                rep_so2_k = make_SO2mats(uv_k, freqs).reshape(BN, Tq, Tk, 2*freqs, 2, 2) # [B*Nk, Tq, Tk, C//4, 2, 2]
                rep_so2_v = make_SO2mats(uv_k, freqs_v).reshape(BN, Tq, Tk, 2*freqs_v, 2, 2) # [B*Nk, Tq, Tk, C//4, 2, 2]
                inv_rep_so2_v = make_SO2mats(uv_q[:, :, 1], freqs_v).reshape(BN, Tq, 2*freqs_v, 2, 2).transpose(-2,-1) # [B*Nk, Tq,  C//4, 2, 2]

            ref_cam = context['cam2world'][:, 0:1]
            cam_k = torch.matmul(torch.inverse(ref_cam), context['cam2world']).reshape(-1, 4, 4) #[B*Nk, 4, 4]
            cam_q = torch.matmul(torch.inverse(ref_cam), query['cam2world']).repeat(1, self.n_view, 1, 1).reshape(-1, 4, 4) #[B*Nk, 4, 4]
            msk = scale_mask(0.1, cam_k.device)
            for i in range(len(cam_k.shape[:-2])):
                msk.unsqueeze(0)
            cam_q, cam_k = cam_q*msk, cam_k*msk
            if self.kv_trnsfm:
                cam_qk = torch.matmul(torch.inverse(cam_q), cam_k)
            else:
                inv_cam_q = torch.inverse(cam_q)

            if self.so3:
                R_q = cam_q[..., :3, :3]
                R_k = cam_k[..., :3, :3]
                if self.kv_trnsfm:
                    R_qk = torch.matmul(R_q.transpose(-2,-1), R_k)
                    D_k = rotmat_to_wigner_d_matrices(2, R_qk)[1:]
                else:
                    D_q = rotmat_to_wigner_d_matrices(2, R_q)[1:]
                    D_k = rotmat_to_wigner_d_matrices(2, R_k)[1:]
                    inv_D_q = [D.transpose(-2,-1) for D in D_q]

            def gta_routine(q, k, v):
                # routine
                qs = OrderedDict()
                ks = OrderedDict()
                vs = OrderedDict()

                if enable_so3:
                    C_se3_st, C_se3_ed = 0, C//2
                    C_so3_st, C_so3_ed = C//2, 3*C//4
                    C_so2_st, C_so2_ed = 3*C//4, C

                    C_v_se3_st, C_v_se3_ed = 0, C_v//2
                    C_v_so3_st, C_v_so3_ed = C_v//2, 3*C_v//4
                    C_v_so2_st, C_v_so2_ed = 3*C_v//4, C_v
                else:
                    C_se3_st, C_se3_ed = 0, C//2
                    C_so2_st, C_so2_ed = C//2, C

                    C_v_se3_st, C_v_se3_ed = 0, C_v//2
                    C_v_so2_st, C_v_so2_ed = C_v//2, C_v

                (q_se3, k_se3) =  map(lambda x: x[:,  C_se3_st:C_se3_ed], (q, k))
                v_se3 = v[:, C_v_se3_st:C_v_se3_ed]
                q_se3_shape, k_se3_shape, v_se3_shape = q_se3.shape, k_se3.shape, v_se3.shape
                if self.kv_trnsfm:
                    k_se3 = k_se3.reshape(BN, -1, 4, Tq, Tk) 
                    v_se3 = v_se3.reshape(BN, -1, 4, Tq, Tk) 

                    fn_se3 = lambda A, x: torch.einsum('bij,bkjtl->bkitl', A, x)
                    qs['se3'] = q_se3
                    ks['se3'] = fn_se3(cam_qk, k_se3).reshape(k_se3_shape)
                    vs['se3'] = fn_se3(cam_qk, v_se3).reshape(v_se3_shape)
                else:
                    q_se3 = q_se3.reshape(BN, -1, 4, Tq, Tk)
                    k_se3 = k_se3.reshape(BN, -1, 4, Tq, Tk) 
                    v_se3 = v_se3.reshape(BN, -1, 4, Tq, Tk) 

                    fn_se3 = lambda A, x: torch.einsum('bij,bkjtl->bkitl', A, x)
                    qs['se3'] = fn_se3(inv_cam_q.transpose(-2, -1), q_se3).reshape(q_se3_shape)
                    ks['se3'] = fn_se3(cam_k, k_se3).reshape(k_se3_shape)
                    vs['se3'] = fn_se3(cam_k, v_se3).reshape(v_se3_shape)

                if enable_so3:
                    dims = [_D.shape[-1] for _D in D_k]

                    total_dim = np.sum(dims)
                    # Use deg1 and deg2
                    (q_so3, k_so3) = map(lambda x: x[:, C_so3_st:C_so3_ed], (q, k))
                    v_so3 = v[:, C_v_so3_st:C_v_so3_ed]
                    q_so3_shape, k_so3_shape, v_so3_shape = q_so3.shape, k_so3.shape, v_so3.shape
                    
                    if self.kv_trnsfm:
                        k_so3 = k_so3.reshape(BN, -1, total_dim, Tq, Tk)
                        v_so3 = v_so3.reshape(BN, -1, total_dim, Tq, Tk) 
                        k_so3s,v_so3s = [], []
                        fn_so3 = lambda A, x: torch.einsum('bij,bkjtl->bkitl', A, x)
                        for i in range(len(dims)):
                            end_dim = np.sum(dims[:i+1])
                            dim = dims[i]
                            k_so3s.append(fn_so3(D_k[i].detach(), k_so3[:,:,end_dim-dim:end_dim]))
                            v_so3s.append(fn_so3(D_k[i].detach(), v_so3[:,:,end_dim-dim:end_dim]))
                        qs['so3'] = q_so3
                        ks['so3'] = torch.cat(k_so3s, 2).reshape(*k_so3_shape)
                        vs['so3'] = torch.cat(v_so3s, 2).reshape(*v_so3_shape)
                    else:
                        q_so3 = q_so3.reshape(BN, -1, total_dim, Tq, Tk) 
                        k_so3 = k_so3.reshape(BN, -1, total_dim, Tq, Tk)
                        v_so3 = v_so3.reshape(BN, -1, total_dim, Tq, Tk) 
                        q_so3s,k_so3s,v_so3s = [], [], []
                        fn_so3 = lambda A, x: torch.einsum('bij,bkjtl->bkitl', A, x)
                        for i in range(len(dims)):
                            end_dim = np.sum(dims[:i+1])
                            dim = dims[i]
                            q_so3s.append(fn_so3(D_q[i].detach(), q_so3[:,:,end_dim-dim:end_dim]))
                            k_so3s.append(fn_so3(D_k[i].detach(), k_so3[:,:,end_dim-dim:end_dim]))
                            v_so3s.append(fn_so3(D_k[i].detach(), v_so3[:,:,end_dim-dim:end_dim]))
                        qs['so3'] = torch.cat(q_so3s, 2).reshape(*q_so3_shape)
                        ks['so3'] = torch.cat(k_so3s, 2).reshape(*k_so3_shape)
                        vs['so3'] = torch.cat(v_so3s, 2).reshape(*v_so3_shape)


                (q_so2, k_so2) =  map(lambda x: x[:, C_so2_st:C_so2_ed], (q, k))
                v_so2 =  v[:, C_v_so2_st:C_v_so2_ed]
                q_so2_shape, k_so2_shape, v_so2_shape = q_so2.shape, k_so2.shape, v_so2.shape
                
                if self.kv_trnsfm:
                    k_so2 = k_so2.reshape(BN, -1, 2, Tq, Tk)
                    v_so2 = v_so2.reshape(BN, -1, 2, Tq, Tk)

                    def fn_so2(A, x): # Einsum is too slow
                        x = x.permute(0, 3,4,1,2)
                        x = torch.sum(A * x[..., None, :], -1)
                        return x.permute(0,3,4,1,2)

                    qs['so2'] = q_so2.reshape(q_so2_shape)
                    ks['so2'] = fn_so2(rep_so2_k, k_so2).reshape(k_so2_shape)
                    vs['so2'] = fn_so2(rep_so2_v, v_so2).reshape(v_so2_shape)
                else:
                    q_so2 = q_so2.reshape(BN, -1, 2, Tq, Tk)
                    k_so2 = k_so2.reshape(BN, -1, 2, Tq, Tk)
                    v_so2 = v_so2.reshape(BN, -1, 2, Tq, Tk)

                    def fn_so2(A, x): # Einsum is too slow
                        x = x.permute(0, 3,4,1,2)
                        x = torch.sum(A * x[..., None, :], -1)
                        return x.permute(0,3,4,1,2)

                    if measure_time:
                        start.record()
                
                    qs['so2'] = fn_so2(rep_so2_q, q_so2).reshape(q_so2_shape)
                    ks['so2'] = fn_so2(rep_so2_k, k_so2).reshape(k_so2_shape)
                    vs['so2'] = fn_so2(rep_so2_v, v_so2).reshape(v_so2_shape)
                    if measure_time:
                        end.record()
                        torch.cuda.synchronize()
                        print('time for so2:', start.elapsed_time(end))

                qt = torch.cat([x for _,x in qs.items()], 1)
                kt = torch.cat([x for _,x in ks.items()], 1)
                vt = torch.cat([x for _,x in vs.items()], 1)

                dot = torch.einsum('bijk,bijk->bjk', qt, kt) * (C**-0.5) # [BN, Tq, Tk]
                dot = dot.view(b, n_context, n_qry_rays, self.npoints).permute(0, 2, 1, 3) # [B, Tq, N, Tk]
                dot = dot.reshape(b, n_qry_rays, n_context * (self.npoints)) # [B, Tq, N*Tk]
                at_wt = F.softmax(dot, dim=-1) # [B, Tq, N*Tk]
                at_wt = at_wt.view(b, n_qry_rays, self.n_view, self.npoints).permute(0, 2, 1, 3) #[B, N, Tq, Tk]
                at_wt = torch.flatten(at_wt, 0, 1) # [BN, Tq, Tk]
                out = (vt * at_wt[:, None, :, :]).sum(dim=-1) # [BN, C, Tq]
                if not self.kv_trnsfm:
                    outs = OrderedDict()
                    outs['se3'] = torch.einsum(
                        'bij,bkjt->bkit',
                        inv_cam_q, 
                        out[:,  C_v_se3_st:C_v_se3_ed].reshape(BN, -1, 4, Tq)
                        ).reshape(v_se3_shape[:-1])
                    if enable_so3:
                        out_so3 = out[:, C_v_so3_st:C_v_so3_ed].reshape(BN, -1, total_dim, Tq)
                        out_so3s = []
                        for i in range(len(dims)):
                            dim = dims[i]
                            end_dim = np.sum(dims[:i+1])
                            out_so3s.append(
                                torch.einsum(
                                    'bij,bkjt->bkit', 
                                    inv_D_q[i].detach(), 
                                    out_so3[:,:,end_dim-dim:end_dim]))
                        outs['so3'] = torch.cat(out_so3s, 2).reshape(v_so3_shape[:-1])
                    def fn_so2_v(A, x): # Einsum is too slow
                        # [bkjt]->btkj
                        x = x.permute(0,3,1,2)
                        x = torch.sum(A * x[..., None, :], -1)
                        return x.permute(0,2,3,1)
                    outs['so2'] = fn_so2_v(inv_rep_so2_v,
                         out[:, C_v_so2_st:C_v_so2_ed].reshape(BN, -1, 2, Tq)).reshape(v_so2_shape[:-1])
                else:
                    outs = OrderedDict()
                    outs['se3'] = out[:,  C_v_se3_st:C_v_se3_ed]
                    outs['so3'] = out[:,  C_v_so3_st:C_v_so3_ed]
                    outs['so2'] = out[:, C_v_so2_st:C_v_so2_ed]

                return torch.cat([x for _,x in outs.items()], 1), at_wt
            
            z_local, at_wt = gta_routine(q, k, v)
            at_wts.append(at_wt)

            s = z_local.size()
            z_local = z_local.view(b, n_context, s[1], n_qry_rays)
            z_sum = z_local.sum(dim=1)
            z_local = torch.cat([z_sum for i in range(n_context)], dim=1).view(*s)
            x = z_local + x[..., None]

            if self.repeat_attention:
                z_embed = self.encode_latent(x)
                z_embed_local = z_embed[:, :, :, None].expand(-1, -1, -1, Tk)
                query_embed_local = torch.cat([z_embed_local, local_coords], dim=1)
                q = self.query_repeat_embed_2(F.relu(self.query_repeat_embed(query_embed_local)))
                z_local = gta_routine(q, k, v)[0] + x

            z_local = z_local.view(b, n_context, s[1], n_qry_rays)
            z_sum = z_local.sum(dim=1)
            z_local = torch.cat([z_sum for i in range(n_context)], dim=1).view(*s)

        else:
            joint_latent = self.latent_value(interp_val)
            s = interp_val.size()

            # Compute key value
            key_val = self.key_map_2(F.relu(self.key_map(interp_val))) # (b*n_ctxt, latent, n_queries, interval_steps)

            #pixel_dist = pixel_val[:, :, :1, :] - pixel_val[:, :, -1:, :]
            #pixel_dist = torch.norm(pixel_dist, p=2, dim=-1)
            
            # Origin of the context camera ray (always zeros)
            
            # Encode depth with tanh to encode different scales of depth values depth values
            # depth_encode = torch.cat([torch.tanh(depth), torch.tanh(depth / 10.), torch.tanh(depth / 100.), torch.tanh(depth / 1000.)], dim=-1)

            # Compute query coordinates by combining context ray info, query ray info, and 3D depth of epipolar line
            # local_coords = torch.cat([cam_rays, cam_origin, ray_dir, depth_encode, query_ray_orig_ex], dim=-1).permute(0, 3, 1, 2)
            # coords_embed = self.query_embed_2(F.relu(self.query_embed(local_coords)))

            # Multiply key and value pairs
            dot_at_joint = torch.einsum('bijk,bijk->bjk', key_val, coords_embed) / 16.
            dot_at_joint = dot_at_joint.view(b, n_context, n_qry_rays, self.npoints).permute(0, 2, 1, 3).reshape(b, n_qry_rays, n_context * (self.npoints)) # [B, 192, 64*2]
            at_wt_joint = F.softmax(dot_at_joint, dim=-1)
            at_wt_joint = torch.flatten(at_wt_joint.view(b, n_qry_rays, self.n_view, self.npoints).permute(0, 2, 1, 3), 0, 1)

            z_local = (joint_latent * at_wt_joint[:, None, :, :]).sum(dim=-1)
            s = z_local.size()
            z_local = z_local.view(b, n_context, s[1], n_qry_rays)
            z_sum = z_local.sum(dim=1)
            z_local = torch.cat([z_sum for i in range(n_context)], dim=1).view(*s)

            at_wt = at_wt_joint
            at_wts.append(at_wt)

            # A second round of attention to gather additional information
            if self.repeat_attention:
                z_embed = self.encode_latent(z_local)
                z_embed_local = z_embed[:, :, :, None].expand(-1, -1, -1, local_coords.size(-1))

                # Concatenate the previous cross-attention vector as context for second round of attention
                query_embed_local = torch.cat([z_embed_local, local_coords], dim=1)
                query_embed_local = self.query_repeat_embed_2(F.relu(self.query_repeat_embed(query_embed_local)))

                dot_at = torch.einsum('bijk,bijk->bjk', query_embed_local, coords_embed) / 16
                dot_at = dot_at.view(b, n_context, n_qry_rays, self.npoints).permute(0, 2, 1, 3).reshape(b, n_qry_rays, n_context * (self.npoints))
                at_wt_joint = F.softmax(dot_at, dim=-1)

                # Compute second averaged feature after cross-attention 
                at_wt_joint = torch.flatten(at_wt_joint.view(b, n_qry_rays, self.n_view, self.npoints).permute(0, 2, 1, 3), 0, 1)
                z_local = (joint_latent * at_wt_joint[:, None, :, :]).sum(dim=-1) + z_local
                z_local = z_local.view(b, n_context, s[1], n_qry_rays)

                z_sum = z_local.sum(dim=1)
                z_local = torch.cat([z_sum for i in range(n_context)], dim=1).view(*s)

        latents_out.append(z_local)

        z = torch.cat(latents_out, dim=1).permute(0, 2, 1).contiguous()
        out_dict['pixel_val'] = pixel_val.cpu()
        out_dict['at_wts'] = at_wts

        depth_squeeze = depth[..., 0]
        at_max_idx = at_wt[..., :].argmax(dim=-1)[..., None, None].expand(-1, -1, -1, 3)

        # Ignore points that are super far away
        pt_clamp = torch.clamp(pt, -100, 100)
        # Get the 3D point that is the average (along attention weight) across epipolar points
        world_point_3d_max = (at_wt[..., None] * pt_clamp).sum(dim=-2)

        s = world_point_3d_max.size()
        world_point_3d_max = world_point_3d_max.view(b, n_context, *s[1:]).sum(dim=1)
        world_point_3d_max = world_point_3d_max[:, :, None, :]

        # Compute the depth for epipolar line visualization
        world_point_3d_max = geometry.project_cam2world(world_point_3d_max[:, :, 0, :], query['cam2world'][:, 0])
        depth_ray = world_point_3d_max[:, :, 2]

        # Clamp depth to make sure things don't get too large due to numerical instability
        depth_ray = torch.clamp(depth_ray, 0, 10)

        out_dict['at_wt'] = at_wt
        out_dict['at_wt_max'] = at_max_idx[:, :, :, 0]
        out_dict['depth_ray'] = depth_ray[..., None]

        # Append to the origin of ray into coords that we query the MLP so that it can reason by disocclusion
        out_dict['coords'] = torch.cat([out_dict['coords'], query_ray_orig_ex[:, :, 0, :]], dim=-1)

        # Plucker embedding for query ray 
        coords = out_dict['coords']
        s = coords.size()
        coords = torch.flatten(coords.view(b, n_context, n_qry_rays, s[-1]).permute(0, 2, 1, 3), -2, -1)

        zsize = z.size()
        z_flat = z.view(b, n_context, *zsize[1:]).permute(0, 2, 1, 3)
        z_flat = torch.flatten(z_flat, -2, -1)
        if not self.GTA:
            z_flat = torch.cat((z_flat, coords), dim=-1)

        # Light field decoder using the gather geometric context
        lf_out = self.phi(z_flat)
        rgb = lf_out[..., :3]

        # Mask invalid regions (no epipolar line correspondence) to be white
        valid_mask = valid_mask.bool().any(dim=1).float()
        rgb = rgb * valid_mask[:, :, None] + 1 * (1 - valid_mask[:, :, None])
        out_dict['valid_mask'] = valid_mask[..., None]

        rgb = rgb.view(b, n_qry, n_qry_rays, 3)

        out_dict['rgb'] = rgb

        # Return the multiview latent for each image (so we can cache computation of multiview encoder)
        out_dict['z'] = z_orig

        return out_dict


