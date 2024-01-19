import json
import logging
import os
import random
import time
import warnings
from collections import defaultdict
from itertools import islice
from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torchvision
from pytorch3d.implicitron.dataset import types
from pytorch3d.implicitron.dataset.json_index_dataset import (
    FrameAnnotsEntry, _bbox_xyxy_to_xywh, _clamp_box_to_image_bounds_and_round,
    _crop_around_box, _get_bbox_from_mask, _get_clamp_bbox, _load_depth,
    _load_depth_mask, _load_image, _load_mask, _load_pointcloud, _rescale_bbox,
    _safe_as_tensor, _seq_name_to_seed)
from pytorch3d.renderer.cameras import PerspectiveCameras

# from gbt.dataset.perturb_camera import perturb_pt3d_cams

logger = logging.getLogger(__name__)

from dataclasses import dataclass, field, fields

from pytorch3d.renderer.camera_utils import join_cameras_as_batch
from pytorch3d.renderer.cameras import CamerasBase, PerspectiveCameras
from pytorch3d.structures.pointclouds import (Pointclouds,
                                              join_pointclouds_as_batch)
from pytorch3d.utils import opencv_from_cameras_projection


# Taken from Co3D Repo: https://github.com/facebookresearch/co3d/blob/d4895dd3976b1c6afb9e9221c047f67c678eaf08/dataset/dataset_zoo.py#L38
CO3D_ALL_CATEGORIES = list(reversed([
    "baseballbat", "banana",  "bicycle", "microwave", "tv",
    "cellphone", "toilet", "hairdryer", "couch", "kite", "pizza",
    "umbrella", "wineglass", "laptop",
    "hotdog", "stopsign", "frisbee", "baseballglove",
    "cup", "parkingmeter", "backpack", "toyplane", "toybus",
    "handbag", "chair", "keyboard", "car", "motorcycle",
    "carrot", "bottle", "sandwich", "remote", "bowl", "skateboard",
    "toaster", "mouse", "toytrain", "book",  "toytruck",
    "orange", "broccoli", "plant", "teddybear",
    "suitcase", "bench", "ball", "cake",
    "vase", "hydrant", "apple", "donut",
]))

# Taken from this github comment: https://github.com/facebookresearch/co3d/issues/43#issuecomment-1103789390
CO3D_NERFORMER_SUBSET_CATEGORIES = CO3D_ALL_CATEGORIES[:10]
# [‘donut’, ‘apple’, ‘hydrant’, ‘vase’, ‘cake’, ‘ball’, ‘bench’, ‘suitcase’, ‘teddybear’, ‘plant’]


ALL_CATEGORY_MAPPING = {
    'all': CO3D_ALL_CATEGORIES,
    'all_nerformer': CO3D_NERFORMER_SUBSET_CATEGORIES,
}


@dataclass
class FrameData(Mapping[str, Any]):
    """
    A type of the elements returned by indexing the dataset object.
    It can represent both individual frames and batches of thereof;
    in this documentation, the sizes of tensors refer to single frames;
    add the first batch dimension for the collation result.
    Args:
        frame_number: The number of the frame within its sequence.
            0-based continuous integers.
        sequence_name: The unique name of the frame's sequence.
        sequence_category: The object category of the sequence.
        frame_timestamp: The time elapsed since the start of a sequence in sec.
        image_size_hw: The size of the image in pixels; (height, width) tensor
                        of shape (2,).
        image_path: The qualified path to the loaded image (with dataset_root).
        image_rgb: A Tensor of shape `(3, H, W)` holding the RGB image
            of the frame; elements are floats in [0, 1].
        mask_crop: A binary mask of shape `(1, H, W)` denoting the valid image
            regions. Regions can be invalid (mask_crop[i,j]=0) in case they
            are a result of zero-padding of the image after cropping around
            the object bounding box; elements are floats in {0.0, 1.0}.
        depth_path: The qualified path to the frame's depth map.
        depth_map: A float Tensor of shape `(1, H, W)` holding the depth map
            of the frame; values correspond to distances from the camera;
            use `depth_mask` and `mask_crop` to filter for valid pixels.
        depth_mask: A binary mask of shape `(1, H, W)` denoting pixels of the
            depth map that are valid for evaluation, they have been checked for
            consistency across views; elements are floats in {0.0, 1.0}.
        mask_path: A qualified path to the foreground probability mask.
        fg_probability: A Tensor of `(1, H, W)` denoting the probability of the
            pixels belonging to the captured object; elements are floats
            in [0, 1].
        bbox_xywh: The bounding box tightly enclosing the foreground object in the
            format (x0, y0, width, height). The convention assumes that
            `x0+width` and `y0+height` includes the boundary of the box.
            I.e., to slice out the corresponding crop from an image tensor `I`
            we execute `crop = I[..., y0:y0+height, x0:x0+width]`
        crop_bbox_xywh: The bounding box denoting the boundaries of `image_rgb`
            in the original image coordinates in the format (x0, y0, width, height).
            The convention is the same as for `bbox_xywh`. `crop_bbox_xywh` differs
            from `bbox_xywh` due to padding (which can happen e.g. due to
            setting `JsonIndexDataset.box_crop_context > 0`)
        camera: A PyTorch3D camera object corresponding the frame's viewpoint,
            corrected for cropping if it happened.
        camera_quality_score: The score proportional to the confidence of the
            frame's camera estimation (the higher the more accurate).
        point_cloud_quality_score: The score proportional to the accuracy of the
            frame's sequence point cloud (the higher the more accurate).
        sequence_point_cloud_path: The path to the sequence's point cloud.
        sequence_point_cloud: A PyTorch3D Pointclouds object holding the
            point cloud corresponding to the frame's sequence. When the object
            represents a batch of frames, point clouds may be deduplicated;
            see `sequence_point_cloud_idx`.
        sequence_point_cloud_idx: Integer indices mapping frame indices to the
            corresponding point clouds in `sequence_point_cloud`; to get the
            corresponding point cloud to `image_rgb[i]`, use
            `sequence_point_cloud[sequence_point_cloud_idx[i]]`.
        frame_type: The type of the loaded frame specified in
            `subset_lists_file`, if provided.
        meta: A dict for storing additional frame information.
    """

    frame_number: Optional[torch.LongTensor]
    sequence_name: Union[str, List[str]]
    sequence_category: Union[str, List[str]]
    frame_timestamp: Optional[torch.Tensor] = None
    image_size_hw: Optional[torch.Tensor] = None
    image_path: Union[str, List[str], None] = None
    image_rgb: Optional[torch.Tensor] = None
    # masks out padding added due to cropping the square bit
    mask_crop: Optional[torch.Tensor] = None
    depth_path: Union[str, List[str], None] = ''
    depth_map: Optional[torch.Tensor] = torch.zeros(1)
    depth_mask: Optional[torch.Tensor] = torch.zeros(1)
    mask_path: Union[str, List[str], None] = None
    fg_probability: Optional[torch.Tensor] = None
    bbox_xywh: Optional[torch.Tensor] = None
    crop_bbox_xywh: Optional[torch.Tensor] = None
    camera: Optional[PerspectiveCameras] = None
    camera_quality_score: Optional[torch.Tensor] = None
    point_cloud_quality_score: Optional[torch.Tensor] = None
    sequence_point_cloud_path: Union[str, List[str], None] = ''
    sequence_point_cloud: Optional[Pointclouds] = torch.zeros(1)
    sequence_point_cloud_idx: Optional[torch.Tensor] = torch.zeros(1)
    frame_type: Union[str, List[str], None] = ''  # known | unseen
    meta: dict = field(default_factory=lambda: {})
    valid_region: Optional[torch.Tensor] = None
    category_one_hot: Optional[torch.Tensor] = None

    def to(self, *args, **kwargs):
        new_params = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, (torch.Tensor, Pointclouds, CamerasBase)):
                new_params[f.name] = value.to(*args, **kwargs)
            else:
                new_params[f.name] = value
        return type(self)(**new_params)

    def cpu(self):
        return self.to(device=torch.device("cpu"))

    def cuda(self):
        return self.to(device=torch.device("cuda"))

    # the following functions make sure **frame_data can be passed to functions
    def __iter__(self):
        for f in fields(self):
            yield f.name

    def __getitem__(self, key):
        return getattr(self, key)

    def __len__(self):
        return len(fields(self))

    @classmethod
    def collate(cls, batch):
        """
        Given a list objects `batch` of class `cls`, collates them into a batched
        representation suitable for processing with deep networks.
        """

        elem = batch[0]

        if isinstance(elem, cls):
            pointcloud_ids = [id(el.sequence_point_cloud) for el in batch]
            id_to_idx = defaultdict(list)
            for i, pc_id in enumerate(pointcloud_ids):
                id_to_idx[pc_id].append(i)

            sequence_point_cloud = []
            sequence_point_cloud_idx = -np.ones((len(batch),))
            for i, ind in enumerate(id_to_idx.values()):
                sequence_point_cloud_idx[ind] = i
                sequence_point_cloud.append(batch[ind[0]].sequence_point_cloud)
            assert (sequence_point_cloud_idx >= 0).all()

            override_fields = {
                "sequence_point_cloud": sequence_point_cloud,
                "sequence_point_cloud_idx": sequence_point_cloud_idx.tolist(),
            }
            # note that the pre-collate value of sequence_point_cloud_idx is unused

            collated = {}
            for f in fields(elem):
                list_values = override_fields.get(
                    f.name, [getattr(d, f.name) for d in batch]
                )
                collated[f.name] = (
                    cls.collate(list_values)
                    if all(list_value is not None for list_value in list_values)
                    else None
                )
            return cls(**collated)

        elif isinstance(elem, Pointclouds):
            return join_pointclouds_as_batch(batch)

        elif isinstance(elem, CamerasBase):
            # TODO: don't store K; enforce working in NDC space
            return join_cameras_as_batch(batch)
        else:
            return torch.utils.data._utils.collate.default_collate(batch)


class CO3Dv2Wrapper(torch.utils.data.Dataset):

    def __init__(self, root='/drive/datasets/co3d/', category='hydrant', subset='fewview_train',
                 stage='train', sample_batch_size=20, image_size=256, masked=True,
                 num_input_views=None, num_query_views=None, query_input_views=False, camera_noise=None):

        from typing import List

        from co3d.dataset.data_types import (FrameAnnotation,
                                             SequenceAnnotation,
                                             load_dataclass_jgzip)

        self.dataset_root = root
        self.path_manager = None
        self.subset = subset
        self.stage = stage
        self.subset_lists_file: List[str] = [os.path.join(self.dataset_root,
                                                          f"{category}/set_lists/set_lists_{subset}.json")]
        self.subsets: Optional[List[str]] = [subset]
        self.sample_batch_size = sample_batch_size
        self.limit_to: int = 0
        self.limit_sequences_to: int = 0
        self.pick_sequence: Tuple[str, ...] = ()
        self.exclude_sequence: Tuple[str, ...] = ()
        self.limit_category_to: Tuple[int, ...] = ()
        self.load_images: bool = True
        self.load_depths: bool = False
        self.load_depth_masks: bool = False
        self.load_masks: bool = True
        self.load_point_clouds: bool = False
        self.max_points: int = 0
        self.mask_images: bool = False
        self.mask_depths: bool = False
        self.image_height: Optional[int] = image_size
        self.image_width: Optional[int] = image_size
        self.box_crop: bool = True
        self.box_crop_mask_thr: float = 0.4
        self.box_crop_context: float = 0.3
        self.remove_empty_masks: bool = True
        self.n_frames_per_sequence: int = -1
        self.seed: int = 0
        self.sort_frames: bool = False
        self.eval_batches: Any = None

        self.img_h = self.image_height
        self.img_w = self.image_width
        self.masked = masked

        self.num_input_views = num_input_views
        self.num_query_views = num_query_views
        self.query_input_views = query_input_views

        self.camera_noise = camera_noise

        start_time = time.time()
        if 'all_' in category or category == 'all':
            self.category_frame_annotations = []
            self.category_sequence_annotations = []
            self.subset_lists_file = []

            if category not in ALL_CATEGORY_MAPPING:
                raise ValueError("Unsupported category.")

            cats = ALL_CATEGORY_MAPPING[category]
            for cat in cats:
                print(cat, 'loaded')
                self.category_frame_annotations.extend(load_dataclass_jgzip(
                    f"{self.dataset_root}/{cat}/frame_annotations.jgz", List[FrameAnnotation]
                ))
                self.category_sequence_annotations.extend(load_dataclass_jgzip(
                    f"{self.dataset_root}/{cat}/sequence_annotations.jgz", List[SequenceAnnotation]
                ))
                self.subset_lists_file.append(f"{self.dataset_root}/{cat}/set_lists/set_lists_{subset}.json")

        else:
            self.category_frame_annotations = load_dataclass_jgzip(
                f"{self.dataset_root}/{category}/frame_annotations.jgz", List[FrameAnnotation]
            )
            self.category_sequence_annotations = load_dataclass_jgzip(
                f"{self.dataset_root}/{category}/sequence_annotations.jgz", List[SequenceAnnotation]
            )

        self.subset_to_image_path = None
        self._load_frames()
        self._load_sequences()
        self._sort_frames()
        self._load_subset_lists()
        self._filter_db()  # also computes sequence indices
        # self._extract_and_set_eval_batches()
        # print(self.eval_batches)
        logger.info(str(self))

        self.seq_to_frames = {}
        for fi, item in enumerate(self.frame_annots):
            if item['frame_annotation'].sequence_name in self.seq_to_frames:
                self.seq_to_frames[item['frame_annotation'].sequence_name].append(fi)
            else:
                self.seq_to_frames[item['frame_annotation'].sequence_name] = [fi]

        if self.stage != 'test' or self.subset != 'fewview_test': # Keep only sequences containing more than 10 frames
            count = 0
            new_seq_to_frames = {}
            for item in self.seq_to_frames:
                if len(self.seq_to_frames[item]) > 10:
                    count += 1
                    new_seq_to_frames[item] = self.seq_to_frames[item]
            self.seq_to_frames = new_seq_to_frames

        self.seq_list = list(self.seq_to_frames.keys())
        print('total training seq', len(self.seq_to_frames))
        print('data loading took', time.time()-start_time, 'seconds')

        self.all_category_list = list(CO3D_ALL_CATEGORIES)
        self.all_category_list.sort()
        self.cat_to_idx = {}
        for ci, cname in enumerate(self.all_category_list):
            self.cat_to_idx[cname] = ci
        print("Initialized Wrapper: with subset", self.subset, "stage", self.stage)

    def __len__(self):
        return len(self.seq_to_frames)

    # def get_random_input_query_split(self, num_input_views, num_query_views, query_input_views, max_avail_views, generator=None):
    #     """This function is used to maintain consistency across baselines and our method"""
    #     V = num_input_views
    #     Q = num_query_views
    #     M = max_avail_views

    #     if (V+Q) > M:
    #         print("INFO: The number of images in the scene ({}), is less than the requested views V({})+Q({})".format(M, V, Q))
    #         Q = max_avail_views if query_input_views else max_avail_views - V
    #         print(f"Setting Q={Q}")

    #     randperm_indices = torch.randperm(min(V+Q, M), generator=generator)  # 2, 4, 0, 3, 1
    #     indices_input = torch.sort(randperm_indices[:V])[0]  # 2, 4, 0 -> 0, 2, 4,
    #     if query_input_views:
    #         indices_query = torch.sort(randperm_indices[:Q])[0]  # 2, 4 -> 2, 4
    #     else:
    #         indices_query = torch.sort(randperm_indices[V:V+Q])[0]  # 3, 1 -> 1, 3

    #     return indices_input, indices_query

    def __getitem__(self, index):
        seed = 10

        seq_index = self.seq_list[index]
        seq_num_frames = len(self.seq_to_frames[seq_index])

        if self.subset == 'fewview_train':
            input_batch_idx = torch.randperm(seq_num_frames)[:self.num_input_views]
            query_batch_idx = torch.sort(torch.randperm(seq_num_frames)[:self.num_query_views])[0]
        elif self.subset == 'fewview_dev' or self.subset == 'fewview_test':
            # print("*********************************************")
            # print("Selecting equally spaced input indices")
            # print("*********************************************")
            # input_batch_idx = torch.linspace(0, seq_num_frames, self.num_input_views+2)[1:-1].to(torch.int32)
            # query_batch_idx = torch.arange(seq_num_frames)[::5]
            input_batch_idx = torch.randperm(seq_num_frames,
                generator=torch.Generator().manual_seed(index))[:self.num_input_views]
            query_batch_idx = torch.sort(torch.randperm(seq_num_frames,
                generator=torch.Generator().manual_seed(index))[-self.num_query_views:])[0]
        else:
            raise NotImplementedError('subset should be - fewview_train/fewview_dev/fewview_test')

        batch_idx = torch.cat((input_batch_idx, query_batch_idx), dim=-1)
        indices = torch.arange(len(batch_idx))
        input_indices = indices[:self.num_input_views]
        query_indices = indices[self.num_input_views:]

        idx_list = [self.seq_to_frames[seq_index][idx] for idx in batch_idx]
        timestamp_list = [self.frame_annots[self.seq_to_frames[seq_index][idx]]["frame_annotation"].frame_timestamp for idx in batch_idx]
        frame_data_list = [self._get_frame(int(self.seq_to_frames[seq_index][idx])) for idx in batch_idx]

        frame_data = FrameData.collate(frame_data_list)
        image_size = torch.Tensor([self.image_height]).repeat(frame_data.camera.R.shape[0],2)

        # if self.camera_noise is not None:
        #     if self.subset != 'fewview_dev' or self.stage != 'test':
        #         raise NotImplementedError()
        #     Rs = frame_data.camera.R # (N, 3, 3)
        #     Ts = frame_data.camera.T # (N, 3)
        #     seeds = (frame_data.frame_number + index).tolist() # (N,)
        #     Rs_perturbed, Ts_perturbed = perturb_pt3d_cams(Rs, Ts, seeds, self.camera_noise)

        #     # Only perturb input views
        #     Rs_perturbed[query_indices] = Rs[query_indices]
        #     Ts_perturbed[query_indices] = Ts[query_indices]

        frame_dict = {
            # 'R':frame_data.camera.R if self.camera_noise is None else Rs_perturbed,
            # 'T':frame_data.camera.T if self.camera_noise is None else Ts_perturbed,
            'R': frame_data.camera.R,
            'T':frame_data.camera.T,
            'f':frame_data.camera.focal_length,
            'c':frame_data.camera.principal_point,
            'images':frame_data.image_rgb*frame_data.fg_probability,
            'images_full':frame_data.image_rgb,
            'masks':frame_data.fg_probability,
            'mask_crop':frame_data.mask_crop,
            'valid_region':frame_data.mask_crop,
            'valid_bbox':frame_data.valid_region,
            'image_size':image_size,
            'frame_type':frame_data.frame_type,
            'idx':seq_index,
            'category':frame_data.category_one_hot,
            'input_indices':input_indices,
            'query_indices':query_indices,
            'frame_indices_scene':batch_idx,
            'frame_indices_global':idx_list,
            'frame_timestamp_list':timestamp_list,
            'frame_number': frame_data.frame_number.tolist()
        }

        return frame_dict

    def _get_frame(self, index):

        entry = self.frame_annots[index]["frame_annotation"]
        point_cloud = self.seq_annots[entry.sequence_name].point_cloud
        frame_data = FrameData(
            frame_number=_safe_as_tensor(entry.frame_number, torch.long),
            frame_timestamp=_safe_as_tensor(entry.frame_timestamp, torch.float),
            sequence_name=entry.sequence_name,
            sequence_category=self.seq_annots[entry.sequence_name].category,
            camera_quality_score=_safe_as_tensor(
                self.seq_annots[entry.sequence_name].viewpoint_quality_score,
                torch.float,
            ),
            point_cloud_quality_score=_safe_as_tensor(
                point_cloud.quality_score, torch.float
            )
            if point_cloud is not None
            else None,
        )

        # The rest of the fields are optional
        frame_data.frame_type = self._get_frame_type(self.frame_annots[index])

        (frame_data.fg_probability, frame_data.mask_path, frame_data.bbox_xywh, clamp_bbox_xyxy,
         frame_data.crop_bbox_xywh) = self._load_crop_fg_probability(entry)

        scale = 1.0
        if self.load_images and entry.image is not None:
            # original image size
            frame_data.image_size_hw = _safe_as_tensor(entry.image.size, torch.long)
            frame_data.image_rgb, frame_data.image_path, frame_data.mask_crop, scale = \
                self._load_crop_images(entry, frame_data.fg_probability, clamp_bbox_xyxy)

        #! UPDATED VALID BBOX
        valid = torch.nonzero(frame_data.mask_crop[0])
        min_y, min_x, max_y, max_x = valid[:,0].min(), valid[:,1].min(), valid[:,0].max(), valid[:,1].max()
        h_half, w_half = (self.image_height-1) / 2, (self.image_width-1) / 2
        valid_bbox = torch.tensor([min_y, min_x, max_y, max_x], device=frame_data.image_rgb.device).float()
        valid_bbox[0] = ((valid_bbox[0] - h_half) / h_half).clip(-1.0, 1.0)
        valid_bbox[2] = ((valid_bbox[2] - h_half) / h_half).clip(-1.0, 1.0)
        valid_bbox[1] = ((valid_bbox[1] - w_half) / w_half).clip(-1.0, 1.0)
        valid_bbox[3] = ((valid_bbox[3] - w_half) / w_half).clip(-1.0, 1.0)
        frame_data.valid_region = valid_bbox

        #! SET CLASS ONEHOT
        frame_data.category_one_hot = torch.zeros((len(self.all_category_list)), device=frame_data.image_rgb.device)
        frame_data.category_one_hot[self.cat_to_idx[frame_data.sequence_category]] = 1

        if self.load_depths and entry.depth is not None:
            (frame_data.depth_map, frame_data.depth_path, frame_data.depth_mask) = \
                self._load_mask_depth(entry, clamp_bbox_xyxy, frame_data.fg_probability)

        if entry.viewpoint is not None:
            frame_data.camera = self._get_pytorch3d_camera(entry, scale, clamp_bbox_xyxy)

        if self.load_point_clouds and point_cloud is not None:
            frame_data.sequence_point_cloud_path = pcl_path = os.path.join(self.dataset_root, point_cloud.path)
            frame_data.sequence_point_cloud = _load_pointcloud(self._local_path(pcl_path), max_points=self.max_points)

        return frame_data

    def _extract_and_set_eval_batches(self):
        """
        Sets eval_batches based on input eval_batch_index.
        """
        if self.eval_batch_index is not None:
            if self.eval_batches is not None:
                raise ValueError(
                    "Cannot define both eval_batch_index and eval_batches."
                )
            self.eval_batches = self.seq_frame_index_to_dataset_index(
                self.eval_batch_index
            )

    def _load_crop_fg_probability(
        self, entry: types.FrameAnnotation
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[str],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        fg_probability = None
        full_path = None
        bbox_xywh = None
        clamp_bbox_xyxy = None
        crop_box_xywh = None

        if (self.load_masks or self.box_crop) and entry.mask is not None:
            full_path = os.path.join(self.dataset_root, entry.mask.path)
            mask = _load_mask(self._local_path(full_path))

            if mask.shape[-2:] != entry.image.size:
                raise ValueError(
                    f"bad mask size: {mask.shape[-2:]} vs {entry.image.size}!"
                )

            bbox_xywh = torch.tensor(_get_bbox_from_mask(mask, self.box_crop_mask_thr))

            if self.box_crop:
                clamp_bbox_xyxy = _clamp_box_to_image_bounds_and_round(
                    _get_clamp_bbox(
                        bbox_xywh,
                        image_path=entry.image.path,
                        box_crop_context=self.box_crop_context,
                    ),
                    image_size_hw=tuple(mask.shape[-2:]),
                )
                crop_box_xywh = _bbox_xyxy_to_xywh(clamp_bbox_xyxy)

                mask = _crop_around_box(mask, clamp_bbox_xyxy, full_path)

            fg_probability, _, _ = self._resize_image(mask, mode="nearest")

        return fg_probability, full_path, bbox_xywh, clamp_bbox_xyxy, crop_box_xywh

    def _load_crop_images(
        self,
        entry: types.FrameAnnotation,
        fg_probability: Optional[torch.Tensor],
        clamp_bbox_xyxy: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, str, torch.Tensor, float]:
        assert self.dataset_root is not None and entry.image is not None
        path = os.path.join(self.dataset_root, entry.image.path)
        image_rgb = _load_image(self._local_path(path))

        if image_rgb.shape[-2:] != entry.image.size:
            raise ValueError(
                f"bad image size: {image_rgb.shape[-2:]} vs {entry.image.size}!"
            )

        if self.box_crop:
            assert clamp_bbox_xyxy is not None
            image_rgb = _crop_around_box(image_rgb, clamp_bbox_xyxy, path)

        image_rgb, scale, mask_crop = self._resize_image(image_rgb)

        if self.mask_images:
            assert fg_probability is not None
            image_rgb *= fg_probability

        return image_rgb, path, mask_crop, scale

    def _load_mask_depth(
        self,
        entry: types.FrameAnnotation,
        clamp_bbox_xyxy: Optional[torch.Tensor],
        fg_probability: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, str, torch.Tensor]:
        entry_depth = entry.depth
        assert entry_depth is not None
        path = os.path.join(self.dataset_root, entry_depth.path)
        depth_map = _load_depth(self._local_path(path), entry_depth.scale_adjustment)

        if self.box_crop:
            assert clamp_bbox_xyxy is not None
            depth_bbox_xyxy = _rescale_bbox(
                clamp_bbox_xyxy, entry.image.size, depth_map.shape[-2:]
            )
            depth_map = _crop_around_box(depth_map, depth_bbox_xyxy, path)

        depth_map, _, _ = self._resize_image(depth_map, mode="nearest")

        if self.mask_depths:
            assert fg_probability is not None
            depth_map *= fg_probability

        if self.load_depth_masks:
            assert entry_depth.mask_path is not None
            mask_path = os.path.join(self.dataset_root, entry_depth.mask_path)
            depth_mask = _load_depth_mask(self._local_path(mask_path))

            if self.box_crop:
                assert clamp_bbox_xyxy is not None
                depth_mask_bbox_xyxy = _rescale_bbox(
                    clamp_bbox_xyxy, entry.image.size, depth_mask.shape[-2:]
                )
                depth_mask = _crop_around_box(
                    depth_mask, depth_mask_bbox_xyxy, mask_path
                )

            depth_mask, _, _ = self._resize_image(depth_mask, mode="nearest")
        else:
            depth_mask = torch.ones_like(depth_map)

        return depth_map, path, depth_mask

    def _get_pytorch3d_camera(
        self,
        entry: types.FrameAnnotation,
        scale: float,
        clamp_bbox_xyxy: Optional[torch.Tensor],
    ) -> PerspectiveCameras:
        entry_viewpoint = entry.viewpoint
        assert entry_viewpoint is not None
        # principal point and focal length
        principal_point = torch.tensor(
            entry_viewpoint.principal_point, dtype=torch.float
        )
        focal_length = torch.tensor(entry_viewpoint.focal_length, dtype=torch.float)

        half_image_size_wh_orig = (
            torch.tensor(list(reversed(entry.image.size)), dtype=torch.float) / 2.0
        )

        # first, we convert from the dataset's NDC convention to pixels
        format = entry_viewpoint.intrinsics_format
        if format.lower() == "ndc_norm_image_bounds":
            # this is e.g. currently used in CO3D for storing intrinsics
            rescale = half_image_size_wh_orig
        elif format.lower() == "ndc_isotropic":
            rescale = half_image_size_wh_orig.min()
        else:
            raise ValueError(f"Unknown intrinsics format: {format}")

        # principal point and focal length in pixels
        principal_point_px = half_image_size_wh_orig - principal_point * rescale
        focal_length_px = focal_length * rescale
        if self.box_crop:
            assert clamp_bbox_xyxy is not None
            principal_point_px -= clamp_bbox_xyxy[:2]

        # now, convert from pixels to PyTorch3D v0.5+ NDC convention
        if self.image_height is None or self.image_width is None:
            out_size = list(reversed(entry.image.size))
        else:
            out_size = [self.image_width, self.image_height]

        half_image_size_output = torch.tensor(out_size, dtype=torch.float) / 2.0
        half_min_image_size_output = half_image_size_output.min()

        # rescaled principal point and focal length in ndc
        principal_point = (
            half_image_size_output - principal_point_px * scale
        ) / half_min_image_size_output
        focal_length = focal_length_px * scale / half_min_image_size_output

        return PerspectiveCameras(
            focal_length=focal_length[None],
            principal_point=principal_point[None],
            R=torch.tensor(entry_viewpoint.R, dtype=torch.float)[None],
            T=torch.tensor(entry_viewpoint.T, dtype=torch.float)[None],
        )

    def _load_frames(self) -> None:
        self.frame_annots = [
            FrameAnnotsEntry(frame_annotation=a, subset=None) for a in self.category_frame_annotations
        ]

    def _load_sequences(self) -> None:

        self.seq_annots = {entry.sequence_name: entry for entry in self.category_sequence_annotations}

    def _load_subset_lists(self) -> None:
        logger.info(f"Loading Co3D subset lists from {self.subset_lists_file}.")
        if not self.subset_lists_file:
            return

        frame_path_to_subset = {}

        for subset_list_file in self.subset_lists_file:
            with open(self._local_path(subset_list_file), "r") as f:
                subset_to_seq_frame = json.load(f)

            #! PRINT SUBSET_LIST STATS
            if len(self.subset_lists_file) == 1:
                print('train frames', len(subset_to_seq_frame['train']))
                print('val frames', len(subset_to_seq_frame['val']))
                print('test frames', len(subset_to_seq_frame['test']))


            for set_ in subset_to_seq_frame:
                for _, _, path in subset_to_seq_frame[set_]:
                    if path in frame_path_to_subset:
                        frame_path_to_subset[path].add(set_)
                    else:
                        frame_path_to_subset[path] = {set_}

        #! DEBUG PRINT PATH_TO_SUBSET
        # print(len(frame_path_to_subset))
        # hist = {}
        # for entry in frame_path_to_subset:
        #     if ','.join(frame_path_to_subset[entry]) in hist:
        #         hist[','.join(frame_path_to_subset[entry])]  += 1
        #     else:
        #         hist[','.join(frame_path_to_subset[entry])]  = 1
        # print(hist)
        # exit(0)

        # pyre-ignore[16]
        for frame in self.frame_annots:
            frame["subset"] = frame_path_to_subset.get(
                frame["frame_annotation"].image.path, None
            )

            if frame["subset"] is None:
                continue
                warnings.warn(
                    "Subset lists are given but don't include "
                    + frame["frame_annotation"].image.path
                )

    def _sort_frames(self) -> None:
        # Sort frames to have them grouped by sequence, ordered by timestamp
        # pyre-ignore[16]
        self.frame_annots = sorted(
            self.frame_annots,
            key=lambda f: (
                f["frame_annotation"].sequence_name,
                f["frame_annotation"].frame_timestamp or 0,
            ),
        )

    def _filter_db(self) -> None:
        if self.remove_empty_masks:
            logger.info("Removing images with empty masks.")
            # pyre-ignore[16]
            old_len = len(self.frame_annots)

            msg = "remove_empty_masks needs every MaskAnnotation.mass to be set."

            def positive_mass(frame_annot: types.FrameAnnotation) -> bool:
                mask = frame_annot.mask
                if mask is None:
                    return False
                if mask.mass is None:
                    raise ValueError(msg)
                return mask.mass > 1

            self.frame_annots = [
                frame
                for frame in self.frame_annots
                if positive_mass(frame["frame_annotation"])
            ]
            logger.info("... filtered %d -> %d" % (old_len, len(self.frame_annots)))

        # this has to be called after joining with categories!!
        subsets = self.subsets
        if subsets:
            if not self.subset_lists_file:
                raise ValueError(
                    "Subset filter is on but subset_lists_file was not given"
                )

            logger.info(f"Limiting Co3D dataset to the '{subsets}' subsets.")

            # truncate the list of subsets to the valid one
            self.frame_annots = [
                entry for entry in self.frame_annots if (entry["subset"] is not None and self.stage in entry["subset"])
            ]

            if len(self.frame_annots) == 0:
                raise ValueError(f"There are no frames in the '{subsets}' subsets!")

            self._invalidate_indexes(filter_seq_annots=True)

        if len(self.limit_category_to) > 0:
            logger.info(f"Limiting dataset to categories: {self.limit_category_to}")
            # pyre-ignore[16]
            self.seq_annots = {
                name: entry
                for name, entry in self.seq_annots.items()
                if entry.category in self.limit_category_to
            }

        # sequence filters
        for prefix in ("pick", "exclude"): # UNUSED
            orig_len = len(self.seq_annots)
            attr = f"{prefix}_sequence"
            arr = getattr(self, attr)
            if len(arr) > 0:
                logger.info(f"{attr}: {str(arr)}")
                self.seq_annots = {
                    name: entry
                    for name, entry in self.seq_annots.items()
                    if (name in arr) == (prefix == "pick")
                }
                logger.info("... filtered %d -> %d" % (orig_len, len(self.seq_annots)))

        if self.limit_sequences_to > 0: # UNUSED
            self.seq_annots = dict(
                islice(self.seq_annots.items(), self.limit_sequences_to)
            )

        # retain only frames from retained sequences
        self.frame_annots = [
            f
            for f in self.frame_annots
            if f["frame_annotation"].sequence_name in self.seq_annots
        ]

        self._invalidate_indexes()

        if self.n_frames_per_sequence > 0:
            logger.info(f"Taking max {self.n_frames_per_sequence} per sequence.")
            keep_idx = []
            # pyre-ignore[16]
            for seq, seq_indices in self._seq_to_idx.items():
                # infer the seed from the sequence name, this is reproducible
                # and makes the selection differ for different sequences
                seed = _seq_name_to_seed(seq) + self.seed
                seq_idx_shuffled = random.Random(seed).sample(
                    sorted(seq_indices), len(seq_indices)
                )
                keep_idx.extend(seq_idx_shuffled[: self.n_frames_per_sequence])

            logger.info(
                "... filtered %d -> %d" % (len(self.frame_annots), len(keep_idx))
            )
            self.frame_annots = [self.frame_annots[i] for i in keep_idx]
            self._invalidate_indexes(filter_seq_annots=False)
            # sequences are not decimated, so self.seq_annots is valid

        if self.limit_to > 0 and self.limit_to < len(self.frame_annots):
            logger.info(
                "limit_to: filtered %d -> %d" % (len(self.frame_annots), self.limit_to)
            )
            self.frame_annots = self.frame_annots[: self.limit_to]
            self._invalidate_indexes(filter_seq_annots=True)

    def _invalidate_indexes(self, filter_seq_annots: bool = False) -> None:
        # update _seq_to_idx and filter seq_meta according to frame_annots change
        # if filter_seq_annots, also uldates seq_annots based on the changed _seq_to_idx
        self._invalidate_seq_to_idx()

        if filter_seq_annots:
            # pyre-ignore[16]
            self.seq_annots = {
                k: v
                for k, v in self.seq_annots.items()
                # pyre-ignore[16]
                if k in self._seq_to_idx
            }

    def _invalidate_seq_to_idx(self) -> None:
        seq_to_idx = defaultdict(list)
        # pyre-ignore[16]
        for idx, entry in enumerate(self.frame_annots):
            seq_to_idx[entry["frame_annotation"].sequence_name].append(idx)
        # pyre-ignore[16]
        self._seq_to_idx = seq_to_idx

    def _resize_image(
        self, image, mode="bilinear"
    ) -> Tuple[torch.Tensor, float, torch.Tensor]:
        image_height, image_width = self.image_height, self.image_width
        if image_height is None or image_width is None:
            # skip the resizing
            imre_ = torch.from_numpy(image)
            return imre_, 1.0, torch.ones_like(imre_[:1])
        # takes numpy array, returns pytorch tensor
        minscale = min(
            image_height / image.shape[-2],
            image_width / image.shape[-1],
        )
        imre = torch.nn.functional.interpolate(
            torch.from_numpy(image)[None],
            scale_factor=minscale,
            mode=mode,
            align_corners=False if mode == "bilinear" else None,
            recompute_scale_factor=True,
        )[0]
        # pyre-fixme[19]: Expected 1 positional argument.
        imre_ = torch.zeros(image.shape[0], self.image_height, self.image_width)
        imre_[:, 0 : imre.shape[1], 0 : imre.shape[2]] = imre
        # pyre-fixme[6]: For 2nd param expected `int` but got `Optional[int]`.
        # pyre-fixme[6]: For 3rd param expected `int` but got `Optional[int]`.
        mask = torch.zeros(1, self.image_height, self.image_width)
        mask[:, 0 : imre.shape[1], 0 : imre.shape[2]] = 1.0
        return imre_, minscale, mask

    def _local_path(self, path: str) -> str:
        if self.path_manager is None:
            return path
        return self.path_manager.get_local_path(path)

    def get_frame_numbers_and_timestamps(
        self, idxs: Sequence[int]
    ) -> List[Tuple[int, float]]:
        out: List[Tuple[int, float]] = []
        for idx in idxs:
            # pyre-ignore[16]
            frame_annotation = self.frame_annots[idx]["frame_annotation"]
            out.append(
                (frame_annotation.frame_number, frame_annotation.frame_timestamp)
            )
        return out

    def get_eval_batches(self) -> Optional[List[List[int]]]:
        return self.eval_batches

    def _get_frame_type(self, entry: FrameAnnotsEntry) -> Optional[str]:
        return entry['frame_annotation'].meta['frame_type']


class Co3dV2Dataset(CO3Dv2Wrapper):

    def __init__(self, cfg, query_sparsity=192):
        if cfg.stage == 'train':
            subset='fewview_train'
        elif cfg.stage == 'val':
            subset='fewview_dev'
        elif cfg.stage == 'test':
            subset='fewview_test'
            print("Using test stage with subset fewview_dev")
        else:
            raise ValueError

        camera_noise = cfg.camera_noise if 'camera_noise' in cfg else None
        cfg.category = 'ball'
        super().__init__(root=cfg.path, category=cfg.category, stage='test', subset=subset,
                         sample_batch_size=cfg.num_input_views+cfg.num_query_views, image_size=cfg.image_size[0],
                         masked=False, num_input_views=cfg.num_input_views, num_query_views=cfg.num_query_views,
                         query_input_views=cfg.query_input_views, camera_noise=camera_noise)

        self.cfg = cfg
        self.query_sparsity = query_sparsity

        # Catch arguments
        self.path = cfg.path
        self.category = cfg.category
        self.training_stage = cfg.stage
        self.image_size = cfg.image_size

        self.num_input_views = cfg.num_input_views
        self.num_query_views = cfg.num_query_views
        self.query_input_views = cfg.query_input_views
        self.mask_images = cfg.mask_images

        # The images are already resized and normalized to 0-1 from the Co3dDataset.
        self.input_view_transform = torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        self.query_view_transform = None

        W, H = 256, 256
        i, j = torch.meshgrid(torch.arange(0, W), torch.arange(0, H))
        self.uv = torch.stack([i.float(), j.float()], dim=-1).permute(1, 0, 2)
        self.uv = self.uv[None].permute(0, -1, 1, 2).permute(0, 2, 3, 1)
        self.uv = self.uv.reshape(-1, 2)

    def _collect_data(self, item, indices, transform=None):
        if self.mask_images:
            images = torch.stack([item['images'][i] for i in indices], dim=0)
        else:
            images = torch.stack([item['images_full'][i] for i in indices], dim=0)

        if transform is not None:
            images = transform(images)

        cameras = [PerspectiveCameras(R=item['R'][i].unsqueeze(0),
                                      T=item['T'][i].unsqueeze(0),
                                      focal_length=item['f'][i].unsqueeze(0),
                                      principal_point=item['c'][i].unsqueeze(0),
                                      image_size=item['image_size'][i].unsqueeze(0))
                   for i in indices]
        frame_numbers = torch.LongTensor([item['frame_number'][i] for i in indices])
        img_indices = torch.LongTensor([item['frame_indices_scene'][i] for i in indices])
        img_bboxes = torch.FloatTensor([[-1, -1, -1, -1] for _ in indices])  # Not used in v2. Kept here for backward compatibility.
        crop_mask = torch.stack([item['mask_crop'][i].squeeze(0) for i in indices], dim=0)
        fg_masks_soft = torch.stack([item['masks'][i].squeeze(0) for i in indices], dim=0)
        fg_masks = (fg_masks_soft > 0.5).float()

        return images, cameras, img_indices, frame_numbers, img_bboxes, crop_mask, fg_masks_soft, fg_masks

    def __getitem__(self, index):
        V = self.num_input_views
        Q = self.num_query_views

        item = super().__getitem__(index) # dict from wrapper getitem
        input_indices = item['input_indices']
        query_indices = item['query_indices']

        (
            sparse_input_images,
            sparse_input_cameras,
            sparse_input_indices,
            sparse_input_frame_numbers,
            sparse_input_bboxes,
            sparse_input_crop_masks,
            sparse_input_fg_masks_soft,
            sparse_input_fg_masks
        ) = self._collect_data(item, input_indices, self.input_view_transform)

        (
            sparse_query_images,
            sparse_query_cameras,
            sparse_query_indices,
            sparse_query_frame_numbers,
            sparse_query_bboxes,
            sparse_query_crop_masks,
            sparse_query_fg_masks_soft,
            sparse_query_fg_masks
        ) = self._collect_data(item, query_indices, self.query_view_transform)

        ctxt_cam2worlds = []
        ctxt_intrinsics = []

        for cam in sparse_input_cameras:
            r, t, k = opencv_from_cameras_projection(cam, torch.ones(1, 2) * 256)

            r = r[0]
            t = t[0]
            k = k[0]
            # k[:2, -1] = k[:2, -1] + 128

            c2w = torch.eye(4, dtype=torch.float32)
            c2w[:3, :3] = r.T
            c2w[:3, 3] = -r.T @ t

            # w2c = torch.eye(4, dtype=torch.float32)
            # w2c[:3, :3] = r
            # w2c[:3, 3] = t
            # c2w = torch.inverse(w2c)

            # c2w = torch.diag(torch.Tensor([-1, -1, 1, 1])) @ c2w
            # c2w[:3, :3] = r.T
            # c2w[:3, 3] = -r.T @ t

            # W, H = cam.get_image_size()[0]
            # pt = cam.principal_point
            # f = cam.focal_length

            # fx = W / 2 * f[0, 0]
            # fy = H / 2 * f[0, 1]

            # cx = W / 2 - cam.principal_point[0, 0] * (W / 2)
            # cy = H / 2 - cam.principal_point[0, 1] * (H / 2)

            # intrinsic = torch.eye(3)
            # intrinsic[0, 0] = fx
            # intrinsic[1, 1] = fy
            # intrinsic[0, 2] = cx
            # intrinsic[1, 2] = cx

            ctxt_cam2worlds.append(c2w)
            ctxt_intrinsics.append(k)

        query_cam2worlds = []
        query_intrinsics = []

        for cam in sparse_query_cameras:
            r, t, k = opencv_from_cameras_projection(cam, torch.ones(1, 2) * 256)

            r = r[0]
            t = t[0]

            c2w = torch.eye(4, dtype=torch.float32)
            c2w[:3, :3] = r.T
            c2w[:3, 3] = -r.T @ t

            # c2w = torch.diag(torch.Tensor([-1, -1, 1, 1])) @ c2w

            # w2c = torch.eye(4, dtype=torch.float32)
            # w2c[:3, :3] = r
            # w2c[:3, 3] = t
            # c2w = torch.inverse(w2c)

            k = k[0]
            # k[:2, -1] = k[:2, -1] + 128

            query_cam2worlds.append(c2w)
            query_intrinsics.append(k)

        uvs = []
        query_rgbs = []
        for rgb, mask in zip(sparse_query_images, sparse_query_fg_masks):
            rgb = rgb.permute(1, 2, 0)
            rgb = ((rgb - 0.5) * 2.0).reshape((-1, 3))

            mask = mask.bool().reshape((-1,))

            if self.query_sparsity is not None:
                uv = self.uv
                rix = np.random.permutation(uv.shape[0])
                # rix = rix[mask]

                rix = rix[:self.query_sparsity]
                uv = uv[rix]
                rgb = rgb[rix]
            else:
                uv = self.uv

            uvs.append(uv)
            query_rgbs.append(rgb)

        ctxt_rgbs = sparse_input_images.permute(0, 2, 3, 1)
        mask_lpips = 0.0

        query = {'rgb': torch.stack(query_rgbs, dim=0),
                 'cam2world': torch.stack(query_cam2worlds, dim=0),
                 'intrinsics': torch.stack(query_intrinsics, dim=0),
                 'uv': torch.stack(uvs, dim=0),
                 'mask': mask_lpips}

        ctxt = {'rgb': ctxt_rgbs,
                'cam2world': torch.stack(ctxt_cam2worlds, dim=0),
                'intrinsics': torch.stack(ctxt_intrinsics, dim=0)}

        return {'query': query, 'context': ctxt}, query


    @staticmethod
    def collate_fn(batch):
        """A function to collate the data across batches. This function must be passed to pytorch's DataLoader to collate batches.
        Args:
            batch(list): List of objects returned by this class' __getitem__ function. This is given by pytorch's dataloader that calls __getitem__
                         multiple times and expects a collated batch.
        Returns:
            dict: The collated dictionary representing the data in the batch.
        """
        result = {
            "path": [],
            "obj_id": [],
            "focal": [],
            "c": [],
            "sparse_input_images": [],
            "sparse_query_images": [],
            "sparse_input_cameras": [],
            "sparse_query_cameras": [],
            "sparse_input_indices": [],
            "sparse_query_indices": [],
            "sparse_input_fg_masks": [],
            "sparse_query_fg_masks": [],
            "sparse_input_crop_masks": [],
            "sparse_query_crop_masks": [],
            "sparse_input_fg_masks_soft": [],
            "sparse_query_fg_masks_soft": []
        }
        for batch_obj in batch:
            for key in result.keys():
                result[key].append(batch_obj[key])

        for key in ['sparse_input_images', 'sparse_query_images',
                    'sparse_input_fg_masks', 'sparse_query_fg_masks',
                    'sparse_input_crop_masks', 'sparse_query_crop_masks',
                    'sparse_input_fg_masks_soft', 'sparse_query_fg_masks_soft']:
            result[key] = torch.stack(result[key], dim=0)

        return result


if __name__ == "__main__":

    def encode_relative_point(ray, transform):
        s = ray.size()
        b, ncontext = transform.size()[:2]

        ray = ray.view(b, ncontext, *s[1:])
        ray = torch.cat([ray, torch.ones_like(ray[..., :1])], dim=-1)
        ray = (ray[:, :, :, :, None, :] * transform[:, :, None, None, :4, :4]).sum(dim=-1)[..., :3]

        ray = ray.view(*s)
        return ray

    import yaml
    from easydict import EasyDict
    from imageio import imwrite

    import sys
    sys.path.append('/nobackup/users/yilundu/my_repos/local_lightfield_networks')
    import geometry

    with open('dataset/co3d.yaml', 'r') as file:
        config = yaml.safe_load(file)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    train_config = config['train_dataset']
    train_config = EasyDict(train_config)
    dataset = Co3dV2Dataset(train_config, query_sparsity=None)
    elem = dataset[0]

    rgb = elem[0]['context']['rgb']

    for counter, (data, target) in enumerate(dataset):

        if counter == 20:
            break

        rgb = data['query']['rgb']
        ctxt_rgb = data['context']['rgb']
        rgb_input = data['query']['rgb']
        intrinsics = data['query']['intrinsics']
        intrinsics_context = data['context']['intrinsics']
        import pdb
        pdb.set_trace()
        print(intrinsics)
        cam2world = data['query']['cam2world']
        context_cam2world = data['context']['cam2world']
        uv = data['query']['uv']
        coord = uv[0:1, 2080:2081]

        coord = torch.Tensor([[[128, 128]]])

        cam2world = torch.matmul(torch.inverse(context_cam2world), cam2world)
        lf_coords = geometry.plucker_embedding(cam2world, coord.expand(3, -1, -1), intrinsics)

        context_rel_cam2world_view1 = torch.matmul(torch.inverse(context_cam2world[0:1]), context_cam2world)

        start, end, diff, valid_mask = geometry.get_epipolar_lines(lf_coords[None], cam2world[None], intrinsics_context[None], 256, 256)

        diff = end - start
        interval = torch.linspace(0, 1, 64, device=lf_coords.device)

        pixel_val = start + diff * interval[None, None, :, None]
        pixel_val = pixel_val[0]


        pt, dist, parallel, equivalent = geometry.get_depth_epipolar(lf_coords, pixel_val, context_cam2world, 256, 256, intrinsics_context)
        pt_view1 = encode_relative_point(pt, context_rel_cam2world_view1[None])

        pixel_val_multiview = geometry.project(pt_view1[..., 0], pt_view1[..., 1], pt_view1[..., 2], intrinsics_context[:1]).detach().cpu().numpy()


        query_rgb = rgb.reshape((256, 256, 3)).numpy()
        ctxt_rgb = ctxt_rgb.numpy()

        u, v = coord.flatten()

        u, v = int(u), int(v)
        vmin, vmax = max(min(v - 5, 255), 0), max(min(v + 5, 255), 0)
        umin, umax = max(min(u - 5, 255), 0), max(min(u + 5, 255), 0)

        query_rgb[vmin:vmax, umin:umax, 0] = 1.0
        rgb_images = []

        for view in range(3):
            pixel_val_i = pixel_val_multiview[view, 0]
            ctxt_rgb_i = ctxt_rgb[0].copy()

            for ui, vi, _ in pixel_val_i:
                u, v = int(ui), int(vi)
                vmin, vmax = max(min(v - 5, 255), 0), max(min(v + 5, 255), 0)
                umin, umax = max(min(u - 5, 255), 0), max(min(u + 5, 255), 0)
                ctxt_rgb_i[vmin:vmax, umin:umax, view] = 1.0

            rgb_images.append(ctxt_rgb_i)

        rgb_img = np.stack(rgb_images, axis=0).transpose((1, 0, 2, 3)).reshape((256, 256*3, 3))

        imwrite("epipolar_line_{}.png".format(counter), rgb_img)


        pixel_val = np.clip((pixel_val.numpy() + 1) / 2. * 255, 0, 255)

        for i, pixel_val_i in enumerate(pixel_val):
            ctxt_rgb_i = ctxt_rgb[i]

            for ui, vi in pixel_val_i:
                u, v = int(ui), int(vi)
                vmin, vmax = max(min(v - 5, 255), 0), max(min(v + 5, 255), 0)
                umin, umax = max(min(u - 5, 255), 0), max(min(u + 5, 255), 0)
                ctxt_rgb_i[vmin:vmax, umin:umax, 0] = 1.0


        panel_im = np.concatenate([query_rgb[None], ctxt_rgb], axis=0)
        panel_im = panel_im.transpose((1, 0, 2, 3)).reshape((256, 256 * 4, 3))
        imwrite("vis_{}.png".format(counter), panel_im)

    import pdb
    pdb.set_trace()
    print(elem)
    import pdb
    pdb.set_trace()
    print(dataset)
    pass
