import torch
from source.utils.nerf import get_extrinsic, transform_points
from source.utils.common import get_rank, get_world_size, make_2dcoord
from torch.utils.data import get_worker_info, IterableDataset
import numpy as np
from scipy.spatial.transform import Rotation as R

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


class MultishapenetDataset(IterableDataset):
    def __init__(self, path, mode, points_per_item=8192, max_len=None, canonical_view=True,
                 full_scale=False, osrt=False, shuffle=None, 
                 downsample=0,
                 downsample_input_coord=0, 
                 return_transform=False,
                 num_input_views=1, num_target_views=5, seed=None,
                 camera_noise=0.0):
        super(MultishapenetDataset).__init__()
        self.num_target_pixels = points_per_item
        self.path = path
        self.mode = mode
        self.canonical = canonical_view
        self.full_scale = full_scale
        self.osrt = osrt
        self.shuffle = shuffle
        self.downsample = downsample
        self.downsample_input_coord = downsample_input_coord
        self.return_transform = return_transform
        self.num_input_views = num_input_views
        self.num_target_views = num_target_views
        self.camera_noise = camera_noise

        self.h=128
        self.w=128
        self.coord = make_2dcoord(self.h, self.w)

        self.render_kwargs = {
            'min_dist': 0.,
            'max_dist': 20.}

        import sunds  # Import here, so that only this dataset depends on Tensorflow

        try:
            # Disable all GPUS
            import tensorflow as tf
            tf.config.set_visible_devices([], 'GPU')
            visible_devices = tf.config.get_visible_devices()
            for device in visible_devices:
                assert device.device_type != 'GPU'
        except:
            # Invalid device or cannot modify virtual devices once initialized.
            pass

        print('num threads used by TF:',
              tf.config.threading.get_inter_op_parallelism_threads())

        builder = sunds.builder('multi_shapenet', data_dir=self.path)

        self.tf_dataset = builder.as_dataset(
            split=self.mode,
            task=sunds.tasks.Nerf(yield_mode='stacked',
                                  additional_camera_specs={'instance_image'}),
        )

        self.num_items = 1000000 if mode == 'train' else 10000

        if max_len is not None:
            self.num_items = min(max_len, self.num_items)
            print('set num of examples to:', self.num_items)

        self.tf_dataset = self.tf_dataset.take(self.num_items)
        if seed is not None:
            self.rng = np.random.RandomState(seed=seed)
        else:
            self.rng = np.random

    def __len__(self):
        return self.num_items

    def __iter__(self):
        rank = get_rank()
        world_size = get_world_size()

        if torch.utils.data.get_worker_info() is not None:
            worker_id = torch.utils.data.get_worker_info().id
            num_workers = torch.utils.data.get_worker_info().num_workers
            n_shard = world_size * num_workers
            index = num_workers * rank + worker_id
            print('world_size:', world_size, 'num_workers', num_workers)
            print('rank', rank, 'worker_id', worker_id)
        else:
            n_shard = world_size
            index = rank

        dataset = self.tf_dataset

        if n_shard > 1:
            num_shardable_items = (self.num_items // n_shard) * n_shard
            if num_shardable_items != self.num_items:
                print(
                    f'MSN: Using {num_shardable_items} scenes to {self.mode} instead of {self.num_items} to be able to evenly shard to {world_size} processes.')
                dataset = dataset.take(num_shardable_items)
            print('index', index)
            dataset = dataset.shard(num_shards=n_shard, index=index)

            n_examples = self.num_items // n_shard
        else:
            n_examples = self.num_items

        if self.shuffle is not None and self.mode == 'train':
            dataset = dataset.shuffle(self.shuffle)
        tf_iterator = dataset.as_numpy_iterator()

        for i, data in enumerate(tf_iterator):
            yield self.prep_item(data)

    def prep_item(self, data):
        input_views = self.rng.choice(
            np.arange(10), size=self.num_input_views, replace=False)
        target_views = np.array(list(set(range(10)) - set(input_views)))
        target_views = self.rng.choice(
            target_views, size=self.num_target_views, replace=False)

        data['color_image'] = data['color_image'].astype(np.float32) / 255.

        input_images = data['color_image'][input_views]
        input_images = np.stack([downsample(img, self.downsample)
                                for img in input_images]).transpose(0, 3, 1, 2)
        input_rays = data['ray_directions'][input_views]

        input_rays = np.stack([downsample(ray, self.downsample)
                              for ray in input_rays])
        input_camera_pos = data['ray_origins'][input_views][:, 0, 0]  # [Nt, 3]

        instance_indexes = data['instance_image'].clip(1, 34) - 1
        masks = np.zeros((10, 128, 128, 34), dtype=np.uint8)
        np.put_along_axis(masks,  instance_indexes, 1, axis=-1)
        input_masks = masks[input_views]
        input_coord = np.stack([
                downsample(
                self.coord, 
                self.downsample+self.downsample_input_coord if self.downsample is not None else self.downsample_input_coord
                ).reshape(-1, 2)]*len(input_views), 
            0) #[N, H*W, 2]

        target_pixels = data['color_image'][target_views]  # [nt h w c]
        target_rays = data['ray_directions'][target_views]  # [nt h w 3]
        target_camera_pos = data['ray_origins'][target_views]  # [nt h w 3]
        target_masks = masks[target_views]

        # Compute relative camera transformations

        input_transforms = np.stack([get_extrinsic(camera_pos, rays) for (camera_pos, rays) in zip(input_camera_pos, input_rays)])
        target_transforms = np.stack([get_extrinsic(camera_pos[0,0], rays) for (camera_pos, rays) in zip(target_camera_pos, target_rays)])
       
        if self.canonical:
            canonical_extrinsic = get_extrinsic(
                input_camera_pos[0], input_rays[0]).astype(np.float32)
            
            input_rays = transform_points(
                input_rays, canonical_extrinsic, translate=False)
            input_camera_pos = transform_points(
                input_camera_pos, canonical_extrinsic)
            target_rays = transform_points(
                    target_rays, canonical_extrinsic, translate=False)
            target_camera_pos = transform_points(
                    target_camera_pos, canonical_extrinsic)
            
            input_transforms = np.stack([extrinsic @ np.linalg.inv(canonical_extrinsic) for extrinsic in input_transforms])
            target_transforms = np.stack([extrinsic @ np.linalg.inv(canonical_extrinsic) for extrinsic in target_transforms])
        
        if self.camera_noise > 0:
            base_rays = input_rays[0]
            base_cam = input_camera_pos[0]
            # add normal gaussian to the input extrinsic matrices
            n_input_transforms = []
            for i in range(1, len(input_transforms)):
                liealg_se3 = SE3_to_lie(input_transforms[i]) # get coeffs of lie algebra bases 
                liealg_se3 += self.camera_noise * self.rng.normal(size=liealg_se3.shape)
                n_transform = lie_to_SE3(liealg_se3)
                n_input_transforms.append(n_transform)
                inv_n_transform = np.linalg.inv(n_transform)
                input_rays[i] = transform_points(
                        base_rays, inv_n_transform, translate=False)
                input_camera_pos[i] = transform_points(
                        base_cam, inv_n_transform)

            input_transforms = np.stack([input_transforms[0]]+n_input_transforms)

        if self.return_transform:
            h, w = target_pixels.shape[1], target_pixels.shape[2]
            num_points_per_view = h*w
            target_pixels = target_pixels.reshape(-1, h*w, 3)
            target_masks = target_masks.reshape(-1, h*w, 34)
            
            base_rays = input_rays[0]  # [h*w, 3]
            # make all returning input_rays to be base rays:
            input_rays = np.stack([base_rays]*len(input_rays))
            base_rays = base_rays.reshape(-1, 3)

            base_camera_pos = input_camera_pos[0:1].repeat(h*w, 0)  # [h*w, 3]
            base_coord = self.coord.reshape(-1, 2)

          

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
                target_camera_pos = np.stack([base_camera_pos]*len(target_pixels))
                target_coord = np.stack([base_coord]*len(target_pixels))

        else:

            target_pixels = target_pixels.reshape(-1, 3)
            target_camera_pos = target_camera_pos.reshape(-1, 3)
            target_rays = target_rays.reshape(-1, 3)
            target_masks = target_masks.reshape(-1, 34)
            num_pixels = target_pixels.shape[0]
            if not self.full_scale:
                sampled_idxs = np.random.choice(np.arange(num_pixels),
                                                size=(self.num_target_pixels,),
                                                replace=False)

                target_pixels = target_pixels[sampled_idxs]
                target_rays = target_rays[sampled_idxs]
                target_camera_pos = target_camera_pos[sampled_idxs]
                target_masks = target_masks[sampled_idxs]

                

        sceneid = int(data['scene_name'][6:])

        result = {
            'input_images':         input_images,         # [k, 3, h, w]
            'input_camera_pos':     input_camera_pos,     # [k, 3]
            'input_rays':           input_rays,           # [k, h, w, 3]
            'input_masks':          input_masks,
            'target_pixels':        target_pixels,        # [p, 3]
            'target_camera_pos':    target_camera_pos,    # [p, 3]
            'target_rays':          target_rays,          # [p, 3]
            'target_masks':         target_masks,
            'sceneid':              sceneid,              # int
        }

        if self.canonical:
            result['transform'] = canonical_extrinsic     # [3, 4] (optional)

        if self.return_transform:
            result['target_transforms'] = target_transforms.astype(np.float32)
            result['target_coord'] = target_coord
            result['input_coord'] = input_coord
            
        result['input_transforms'] = input_transforms.astype(np.float32)
        

        return result

    def skip(self, n):
        """
        Skip the first n scenes
        """
        self.tf_dataset = self.tf_dataset.skip(n)
