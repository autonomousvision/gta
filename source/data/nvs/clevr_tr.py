# https://github.com/stelzner/srt/blob/main/srt/data/obsurf.py

import numpy as np
import imageio
import yaml
from torch.utils.data import Dataset
import os
import glob
import json
from source.utils.common import make_2dcoord, make_2dimgcoord
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R
from source.utils.nerf import get_camera_rays, get_extrinsic, transform_points, get_rays
import cv2

def SE3_to_lie(SE3):
    """Convert an element of SE(3) to the coefficients of its Lie algebra representation."""
    rotation_part = SE3[:3, :3]
    translation_part = SE3[:3, 3]
    
    rot = R.from_matrix(rotation_part)
    rot_vec = rot.as_rotvec()
    
    return np.hstack((rot_vec, translation_part))

def lie_to_SE3(lie_coeffs):
    """Convert the coefficients of a Lie algebra representation to an element of SE(3)."""
    rot_vec = lie_coeffs[:3]
    translation_part = lie_coeffs[3:]
    
    rot = R.from_rotvec(rot_vec)
    rotation_part = rot.as_matrix()
    
    SE3 = np.eye(4)
    SE3[:3, :3] = rotation_part
    SE3[:3, 3] = translation_part
    
    return SE3


def downsample(x, num_steps=1):
    if num_steps is None or num_steps < 1:
        return x
    stride = 2**num_steps
    return x[stride//2::stride, stride//2::stride]


def camera_basis(return_quaternion=False, kubric_basis=False):
    if kubric_basis:
        X = np.array([1, 0, 0])
        Y = np.array([0, -1, 0])
        Z = np.array([0, 0, -1])
    else:
        X = np.array([-1, 0, 0])
        Y = np.array([0, 1, 0])
        Z = np.array([0, 0, -1])
    front = Z
    right = X
    up = Y
    mat = np.stack([right, up, front])
    return np.asarray(Quaternion(matrix=mat).q, np.float32) if return_quaternion else mat


def get_extrinsic_from_kubric_quats(q, p, kubric_basis=False):
    """
    Args:
        q: quaternion of shape [4]
        p: camera position of shape [3]
    """
    extrinsic = camera_basis(kubric_basis=kubric_basis).T @  Quaternion(q).rotation_matrix.T 
    t = - extrinsic @ p
    extrinsic  = np.concatenate(
        (extrinsic , np.expand_dims(t, -1)), -1)
    filler = np.array([[0., 0., 0., 1.]])
    extrinsic = np.concatenate((extrinsic, filler), 0)
    return extrinsic


class CLEVRTR(Dataset):
    def __init__(self, path, mode,
                 points_per_item=2048, canonical_view=True,
                 num_views=5,
                 max_len=None,
                 full_scale=False,
                 downsample=0,
                 return_transform=False,
                 downsample_target=0,
                 downsample_input_coord=0,
                 num_input_views=4,
                 num_target_views=1,
                 overlap=False,
                 kubric_basis=False,
                 reconstruction=False,
                 return_org_rays=False,
                 return_org_images=False,
                 return_target_transform=False,
                 avoid_zerocamorg=False,
                 camera_noise=False,
                 image_coord=False,
                 load_depth=False,
                 seed=None):
        """ Loads the multi-object datasets
        Args:
            path (str): Path to dataset.
            mode (str): 'train', 'val', or 'test'.
            points_per_item (int): Number of target points per scene.
            max_len (int): Limit to the number of entries in the dataset.
            canonical_view (bool): Return data in canonical camera coordinates (like in SRT), as opposed
                to world coordinates.
            full_scale (bool): Return all available target points, instead of sampling.
            max_objects (int): Load only scenes with at most this many objects.
            downsample (int): Downsample height and width of input image by a factor of 2**downsample
        """
        self.path = path
        self.mode = mode
        self.num_target_pixels = points_per_item
        self.max_len = max_len
        self.canonical = canonical_view
        self.full_scale = full_scale
        self.downsample = downsample
        self.return_transform = return_transform
        self.downsample_input_coord = downsample_input_coord
        self.downsample_target = downsample_target
        self.num_input_views = num_input_views
        self.num_target_views = num_target_views
        self.num_views = num_views
        self.overlap = overlap
        self.reconstruction = reconstruction
        self.kubric_basis = kubric_basis # if True, use kubric basis, which can correct the extrinsic matrix
        self.return_org_rays = return_org_rays
        self.avoid_zerocamorg = avoid_zerocamorg
        self.camera_noise = camera_noise
        self.image_coord = image_coord
        self.load_depth = load_depth

        self.return_org_images = return_org_images
        self.return_target_transform = return_target_transform

        self.h = 240
        self.w = 320

        if self.image_coord:
            self.coord =  make_2dimgcoord(self.h, self.w)
        else:
            self.coord = make_2dcoord(self.h, self.w)

        self.num_max_entities = 7

        self.render_kwargs = {
            'min_dist': 0.035,
            'max_dist': 35.}

        self.dir = os.path.join(path, 'train' if mode in [
                                'train', 'val'] else 'test')

        # metadata path list
        self.metadata_paths = glob.glob(
            os.path.join(self.dir, 'metadata', '*'))
        self.metadata_paths = sorted(self.metadata_paths, key=lambda x: int(
            os.path.basename(x).strip('.json')))  # sort by the filename
        if mode == 'train':  # 90 % of training examples are used for the training
            self.metadata_paths = self.metadata_paths[:9 *
                                                      len(self.metadata_paths)//10]
        elif mode == 'val':  # the rest are used for the validation
            self.metadata_paths = self.metadata_paths[9 *
                                                      len(self.metadata_paths)//10:]

        print('Mode:{}, N:{}'.format(mode,  len(self.metadata_paths)))
        
        if seed is not None:
            self.rng = np.random.RandomState(seed=seed)
        else:
            self.rng = np.random

    def __len__(self):
        return len(self.metadata_paths)

    def __getitem__(self, idx):
        metadata_filename = self.metadata_paths[idx]
        scene_idx = int(os.path.basename(metadata_filename).strip('.json'))
        with open(metadata_filename, 'r') as f:
            metadata = json.loads(f.read())

        input_view_idx = self.rng.choice(
            np.arange(self.num_views), size=self.num_input_views, replace=False)

        if self.reconstruction:
            target_view_idx = input_view_idx
        else:
            if self.overlap:
                target_view_idx = self.rng.choice(
                    np.arange(self.num_views), size=self.num_target_views, replace=False)
            else:
                target_view_idx =  self.rng.choice(
                    list(set(range(self.num_views)) - set(input_view_idx)),
                    size=self.num_target_views, replace=False)
        # load imgs
        imgs = [np.asarray(imageio.imread(
            os.path.join(self.dir, 'imgs', f'img_{scene_idx}_{v}.png')))
            for v in range(self.num_views)]
        imgs = np.stack([img[..., :3].astype(
            np.float32) / 255 for img in imgs])

        # Load masks
        mask_idxs = [imageio.imread(os.path.join(self.dir, 'masks', f'masks_{scene_idx}_{v}.png'))
                     for v in range(self.num_views)]
        masks = np.zeros(
            (self.num_views, self.h, self.w, self.num_max_entities), dtype=np.uint8)
        np.put_along_axis(masks, np.expand_dims(mask_idxs, -1), 1, axis=-1)

        # Load camera extrinsic information
        all_qs, all_camera_pos = metadata['camera']['quaternions'], metadata['camera']['positions']
        all_camera_pos = np.array(all_camera_pos).astype(np.float32)

        extrinsics = np.array([get_extrinsic_from_kubric_quats(
            q, p, kubric_basis=self.kubric_basis) for q, p in zip(all_qs, all_camera_pos)]).astype(np.float32)
        
        if self.camera_noise > 0:
            for i in input_view_idx[1:]:
                liealg_se3 = SE3_to_lie(extrinsics[i]) # get coeffs of lie algebra bases 
                liealg_se3 += self.camera_noise * self.rng.normal(size=liealg_se3.shape)
                extrinsics[i] = lie_to_SE3(liealg_se3)
                
        all_rays = []
        for i in range(self.num_views):
            cur_rays = get_rays(
                all_camera_pos[i],
                front=extrinsics[i][2, :3],
                right=extrinsics[i][0, :3],
                up=extrinsics[i][1, :3],
                noisy=False)
            all_rays.append(cur_rays)
        all_rays = np.stack(all_rays, 0).astype(np.float32)

        # the first input view is set to the canonical view
        canonical_idx = input_view_idx[0]

        target_transforms = extrinsics[target_view_idx]
        input_transforms = extrinsics[input_view_idx]

        if self.canonical:
            canonical_extrinsic = extrinsics[canonical_idx]
            if self.avoid_zerocamorg:
                canonical_extrinsic[:3, 3] += 0.01
            all_rays = transform_points(
                all_rays, canonical_extrinsic, translate=False)
            all_camera_pos = transform_points(
                all_camera_pos, canonical_extrinsic)
            target_transforms = np.stack([extrinsic @ np.linalg.inv(canonical_extrinsic) for extrinsic in target_transforms])
            input_transforms = np.stack([extrinsic @ np.linalg.inv(canonical_extrinsic) for extrinsic in input_transforms])

        input_images = imgs[input_view_idx]
        input_masks = masks[input_view_idx]
        input_camera_pos = all_camera_pos[input_view_idx]
        input_rays = all_rays[input_view_idx]
        input_coord = np.stack([
                downsample(
                self.coord, 
                self.downsample+self.downsample_input_coord if self.downsample is not None else self.downsample_input_coord
                ).reshape(-1, 2)]*len(input_view_idx), 
            0) #[N, H*W, 2]
        if self.return_org_rays:
            input_org_rays = input_rays

        target_pixels = imgs[target_view_idx]  # [nt h w c]
        target_masks = masks[target_view_idx]
        target_camera_pos = all_camera_pos[target_view_idx][:, None, None].repeat(
            self.h, 1).repeat(self.w, 2)  # [nt h w 3]
        target_rays = all_rays[target_view_idx]  # [nt h w 3]

        if self.return_transform:
            h, w = target_pixels.shape[1], target_pixels.shape[2]
            num_points_per_view = h*w
            target_pixels = target_pixels.reshape(-1, h*w, 3)
            target_masks = target_masks.reshape(-1, h*w, self.num_max_entities)
            base_rays = input_rays[0].reshape(-1, 3)  # [h*w, 3]
            base_camera_pos = input_camera_pos[0:1].repeat(h*w, 0)  # [h*w, 3]
            base_coord = self.coord.reshape(-1, 2) # [h*w, 2]
            if not self.full_scale:
                # If we have fewer points than we want, sample with replacement
                points_per_item_and_view = self.num_target_pixels // self.num_target_views
                replace = num_points_per_view < points_per_item_and_view
                target_pixels_list = []
                target_masks_list = []
                target_rays_list = []
                target_camera_pos_list = []
                target_coord_list = []

                for i in range(self.num_target_views):
                    sampled_idxs = np.random.choice(np.arange(num_points_per_view),
                                                    size=(
                                                        points_per_item_and_view,),
                                                    replace=replace)

                    target_pixels_list.append(target_pixels[i, sampled_idxs])
                    target_masks_list.append(target_masks[i, sampled_idxs])

                    target_rays_list.append(base_rays[sampled_idxs])
                    target_camera_pos_list.append(
                        base_camera_pos[sampled_idxs])
                    target_coord_list.append(base_coord[sampled_idxs])

                target_pixels = np.stack(target_pixels_list)
                target_masks = np.stack(target_masks_list)
                target_rays = np.stack(target_rays_list)
                target_camera_pos = np.stack(target_camera_pos_list)
                target_coord = np.stack(target_coord_list)
            else:
                target_rays = np.stack([base_rays]*len(target_pixels))
                target_camera_pos = np.stack(
                    [base_camera_pos]*len(target_pixels))
                target_coord = np.stack([base_coord]*len(target_pixels))

        else:
            target_pixels = target_pixels.reshape(-1, 3)
            target_camera_pos = target_camera_pos.reshape(-1, 3)
            target_rays = target_rays.reshape(-1, 3)
            target_masks = target_masks.reshape(-1, self.num_max_entities)
            num_pixels = target_pixels.shape[0]
            if not self.full_scale:
                sampled_idxs = np.random.choice(np.arange(num_pixels),
                                                size=(self.num_target_pixels,),
                                                replace=False)

                target_pixels = target_pixels[sampled_idxs]
                target_rays = target_rays[sampled_idxs]
                target_camera_pos = target_camera_pos[sampled_idxs]
                target_masks = target_masks[sampled_idxs]

        org_input_images = input_images
        if self.downsample is not None:

            input_images = np.stack(
                [downsample(img, self.downsample) for img in input_images])
            input_rays = np.stack(
                [downsample(rays, self.downsample) for rays in input_rays])
            input_masks = np.stack(
                [downsample(masks, self.downsample) for masks in input_masks])

        result = {
            # [1, 3, h, w]
            'input_images':         input_images.transpose(0, 3, 1, 2),
            'input_camera_pos':     input_camera_pos,     # [1, 3]
            'input_rays':           input_rays,           # [1, h, w, 3]
            # [1, h, w, self.max_num_entities]
            'input_masks':          input_masks,
            # [p, 3] or [n_views, p, 3]
            'target_pixels':        target_pixels,
            # [p, 3] or [n_views, p, 3]
            'target_camera_pos':    target_camera_pos,
            # [p, 3] or [n_views, p, 3]
            'target_rays':          target_rays,
            # [p, self.max_num_entities] or [n_views, p, self.max_num_entities]
            'target_masks':         target_masks,
            'sceneid':              idx,                  # int
        }

        if self.canonical:
            result['transform'] = canonical_extrinsic     # [3, 4] (optional)

        if self.return_transform:
            result['target_transforms'] = target_transforms
            result['target_coord'] = target_coord
            result['input_coord'] = input_coord
        result['input_transforms'] = input_transforms

        if self.return_target_transform:
            result['target_transforms'] = target_transforms
    
        if self.return_org_rays:
            result['input_org_rays'] = input_org_rays
        
        if self.return_org_images:
            result['org_input_images'] = org_input_images

        if self.load_depth:
            #depth = np.asarray(imageio.imread(
            #    os.path.join(self.dir, 'depths', f'depths_{scene_idx}_{input_view_idx[0]}.tiff')))[..., 0]
            depth = np.asarray(cv2.imread(
                os.path.join(self.dir, 'depths', f'depths_{scene_idx}_{input_view_idx[0]}.tiff'),
                cv2.IMREAD_UNCHANGED))
            result['input_depths'] = depth

            
        return result
