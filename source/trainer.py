import torch
import numpy as np
from tqdm import tqdm

from typing import Tuple

import source.utils.visualize as vis
from source.utils.common import mse2psnr, reduce_dict, gather_all
from source.utils import nerf
from source.utils.common import get_rank, get_world_size
from source.utils.gta import make_2dcoord
from source.models_nvs import TransformingSRT

import os
import math
from collections import defaultdict

class SRTTrainer:
    def __init__(self, model, optimizer, cfg, device, out_dir, render_kwargs):
        self.model = model
        self.optimizer = optimizer
        self.config = cfg
        self.device = device
        self.out_dir = out_dir
        self.render_kwargs = render_kwargs
        self.mixed_prec = cfg['training']['mixed_prec'] if 'mixed_prec' in cfg['training'] else False
        self.loss_scale = cfg['training']['loss_scale'] if 'loss_scale' in cfg['training'] else False
        
        print('Mixed Precision:', self.mixed_prec, ' Loss scaling:', self.loss_scale)
        self.scaler = torch.cuda.amp.GradScaler()
        if 'num_coarse_samples' in cfg['training']:
            self.render_kwargs['num_coarse_samples'] = cfg['training']['num_coarse_samples']
        if 'num_fine_samples' in cfg['training']:
            self.render_kwargs['num_fine_samples'] = cfg['training']['num_fine_samples']
    

    def evaluate(self, val_loader, **kwargs):
        ''' Performs an evaluation.
        Args:
            val_loader (dataloader): pytorch dataloader
        '''
        self.model.eval()
        eval_lists = defaultdict(list)

        loader = val_loader if get_rank() > 0 else tqdm(val_loader)
        sceneids = []

        for data in loader:
            sceneids.append(data['sceneid'])
            eval_step_dict = self.eval_step(data, **kwargs)

            for k, v in eval_step_dict.items():
                eval_lists[k].append(v)

        sceneids = torch.cat(sceneids, 0).cuda()
        sceneids = torch.cat(gather_all(sceneids), 0)

        print(f'Evaluated {len(torch.unique(sceneids))} unique scenes.')

        eval_dict = {k: torch.cat(v, 0) for k, v in eval_lists.items()}
        # Average across processes
        eval_dict = reduce_dict(eval_dict, average=True)
        eval_dict = {k: v.mean().item()
                     for k, v in eval_dict.items()}  # Average across batch_size
        print('Evaluation results:')
        print(eval_dict)
        return eval_dict

    def train_step(self, data, it):
        self.model.train()
        self.optimizer.zero_grad()
        loss, loss_terms = self.compute_loss(data, it)
        loss = loss.mean(0)
        loss_terms = {k: v.mean(0).item() for k, v in loss_terms.items()}
        if self.mixed_prec and self.loss_scale:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item(), loss_terms

    def compute_loss(self, data, it, no_view_avg=False):
        device = self.device
        input_images = data.get('input_images').to(device, non_blocking=True)
        input_camera_pos = data.get('input_camera_pos').to(device, non_blocking=True)
        input_rays = data.get('input_rays').to(device, non_blocking=True)
        target_pixels = data.get('target_pixels').to(device, non_blocking=True)
        target_camera_pos = data.get('target_camera_pos').to(device, non_blocking=True)
        target_rays = data.get('target_rays').to(device, non_blocking=True)
        extras = {}

        extras['input_transforms'] =  data.get('input_transforms').to(device, non_blocking=True)
        if 'target_transforms' in data:
            target_transforms = data.get('target_transforms').to(device, non_blocking=True)
            extras['target_transforms'] = target_transforms
            extras['input_coord'] = data.get('input_coord').to(device, non_blocking=True)
            extras['target_coord'] = data.get('target_coord').to(device, non_blocking=True)
            extras['input_rays'] = input_rays
            extras['target_rays'] = target_rays
        loss = 0.
        loss_terms = dict()

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16 if self.mixed_prec else torch.float32):
            if 'target_transforms' in data:
                extras.update(self.render_kwargs)
                target_pixels = target_pixels.flatten(1, 2)
                pred_pixels, extras = self.model(input_images, input_camera_pos, input_rays,
                                                target_camera_pos, target_rays, extras)
                pred_pixels = pred_pixels.reshape(*list(target_pixels.shape))
                    
            else:
                pred_pixels, extras = self.model(
                    input_images, input_camera_pos, input_rays, target_camera_pos, target_rays, extras)

        if no_view_avg:
            loss = loss + ((pred_pixels - target_pixels)**2).mean(2)
        else:
            loss = loss + ((pred_pixels - target_pixels)**2).mean((1, 2))
        assert loss.dtype is torch.float32

        loss_terms['mse'] = loss
        return loss, loss_terms


    def eval_step(self, data, full_scale=False):
        with torch.no_grad():
            loss, loss_terms = self.compute_loss(data, 1000000)

        mse = loss_terms['mse']
        psnr = mse2psnr(mse)
        return {'psnr': psnr, 'mse': mse, **loss_terms}


    def render_image(self, z, camera_pos, rays, transforms=None, extras={}):
        """
        Args:
            z [n, k, c]: set structured latent variables
            camera_pos [n, 3]: camera position
            rays [n, h, w, 3]: ray directions
            render_kwargs: extras
        """
        batch_size, height, width = rays.shape[:3]

        coord = torch.stack(batch_size*[torch.Tensor(make_2dcoord(height, width).astype(np.float32))])
        coord = coord.flatten(1,2).to(z.device)

        rays = rays.flatten(1, 2)

        camera_pos = camera_pos.unsqueeze(1).repeat(1, rays.shape[1], 1)

        max_num_rays = self.config['data']['num_points'] * \
            self.config['training']['batch_size'] // (
                rays.shape[0] * get_world_size())
        num_rays = rays.shape[1]
        img = torch.zeros(size=(batch_size, height*width, 3),
                          dtype=camera_pos.dtype).to(camera_pos.device)
        all_extras = []
        for i in range(0, num_rays, max_num_rays):
            if 'target_coord' in extras:
                extras['target_rays'] = rays[:, None, i:i+max_num_rays]
                extras['target_coord'] = coord[:, None, i:i+max_num_rays]
                img[:, i:i+max_num_rays], n_extras = self.model.decode(
                    z=z, x=camera_pos[:, None, i:i+max_num_rays], 
                    rays=rays[:, None, i:i+max_num_rays], transforms=transforms[:, None],
                    extras=extras)
            else:
                img[:, i:i+max_num_rays], n_extras = self.model.decoder(
                    z=z, x=camera_pos[:, i:i+max_num_rays], rays=rays[:, i:i+max_num_rays], extras=extras)
            all_extras.append(n_extras)

        agg_extras = {}
        for key in all_extras[0]:
            agg_extras[key] = torch.cat([extras[key]
                                        for extras in all_extras], 1)
            agg_extras[key] = agg_extras[key].view(
                batch_size, height, width, -1)

        img = img.view(img.shape[0], height, width, 3)
        return img, agg_extras
    

    def visualize(self, data, mode='val'):
        self.model.eval()

        with torch.no_grad():
            device = self.device
            input_images = data.get('input_images').to(device)
            input_camera_pos = data.get('input_camera_pos').to(device)
            input_rays = data.get('input_rays').to(device)
                
            camera_pos_base = input_camera_pos[:, 0]
            if data.get('target_rays').dtype == torch.int64:
                # Make index version of input_rays
                input_rays_base = torch.stack([torch.arange(0, input_rays.shape[2]*input_rays.shape[3]).reshape(
                    input_rays.shape[2], input_rays.shape[3])]*input_rays.shape[0])
            else:
                input_rays_base = input_rays[:, 0]

            extras = {}
            extras['input_transforms'] = data.get('input_transforms').to(device, non_blocking=True)
            target_transforms = data.get('target_transforms')
            if target_transforms is not None:
                extras['input_coord'] = data.get('input_coord').to(device, non_blocking=True)
                extras['target_coord'] = data.get('target_coord').to(device, non_blocking=True)
                extras['input_rays'] = input_rays
                extras['target_rays'] = input_rays_base.unsqueeze(1)

            if 'transform' in data:
                # If the data is transformed in some different coordinate system, where
                # rotating around the z axis doesn't make sense, we first undo this transform,
                # then rotate, and then reapply it.

                transform = data['transform'].to(device)
                inv_transform = torch.inverse(transform)
                if not isinstance(self.model, TransformingSRT):
                    camera_pos_base = nerf.transform_points_torch(
                        camera_pos_base, inv_transform)
                    input_rays_base = nerf.transform_points_torch(
                        input_rays_base, inv_transform.unsqueeze(1).unsqueeze(2), translate=False)
            else:
                transform = None

            input_images_np = np.transpose(
                input_images.cpu().numpy(), (0, 1, 3, 4, 2))

            z = self.model.encoder(input_images, input_camera_pos, input_rays, extras)
            if isinstance(z, (list, tuple)):
                z, extras = z
            else:
                extras = {}

            batch_size, num_input_images, height, width, _ = input_rays.shape

            num_angles = 6

            columns = []
            for i in range(num_input_images):
                header = 'input' if num_input_images == 1 else f'input {i+1}'
                columns.append((header, input_images_np[:, i], 'image'))

            if 'input_masks' in data:
                input_mask = data['input_masks'][:, 0]
                columns.append(
                    ('true seg 0°', input_mask.argmax(-1), 'clustering'))

            row_labels = None

            extras.update(self.render_kwargs.copy())

            for i in range(num_angles):
                angle = i * (2 * math.pi / num_angles)
                angle_deg = (i * 360) // num_angles

                if target_transforms is not None:
                    R = torch.Tensor(np.array([
                        [np.cos(angle), np.sin(-angle), 0, 0],
                        [np.sin(angle), np.cos(angle), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]
                    ])).type(camera_pos_base.dtype).to(camera_pos_base.device)
                    if transform is not None:
                        # BRB^{-1} where B is canonical camera extrinsic
                        target_transforms = transform @ torch.einsum(
                            'ij,njk->nik', R, inv_transform)
                    else:
                        target_transforms = R[None].repeat(batch_size, 1, 1)
                    target_transforms = target_transforms.to(device, non_blocking=True)[:, None]
                    extras['target_transforms'] = target_transforms
                    camera_pos_rot = camera_pos_base
                    rays_rot = input_rays_base
                else:
                    camera_pos_rot = nerf.rotate_around_z_axis_torch(
                        camera_pos_base, angle)
                    rays_rot = nerf.rotate_around_z_axis_torch(
                        input_rays_base, angle)

                    if transform is not None:
                        camera_pos_rot = nerf.transform_points_torch(
                            camera_pos_rot, transform)
                        rays_rot = nerf.transform_points_torch(
                            rays_rot, transform.unsqueeze(1).unsqueeze(2), translate=False)
                    target_transforms = None

                img, n_extras = self.render_image(
                    z, camera_pos_rot, rays_rot, target_transforms, extras)
                
                columns.append(
                    (f'render {angle_deg}°', img.cpu().numpy(), 'image'))

               
            output_img_path = os.path.join(self.out_dir, f'renders-{mode}')
            vis.draw_visualization_grid(
                columns, output_img_path, row_labels=row_labels)
            



