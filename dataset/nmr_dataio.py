import random
import cv2
import os
import os.path as osp
import torch
import numpy as np
from glob import glob

current_dir = osp.dirname(os.path.abspath(__file__))
import sys
sys.path.append(current_dir + '/../')

from utils import data_util, util
import json
from collections import defaultdict
from tqdm import tqdm


def parse_intrinsics(filepath, trgt_sidelength=None, invert_y=False):
    # Get camera intrinsics
    with open(filepath, 'r') as file:
        f, cx, cy, _ = map(float, file.readline().split())
        grid_barycenter = torch.Tensor(list(map(float, file.readline().split())))
        scale = float(file.readline())
        height, width = map(float, file.readline().split())

        try:
            world2cam_poses = int(file.readline())
        except ValueError:
            world2cam_poses = None

    if world2cam_poses is None:
        world2cam_poses = False

    world2cam_poses = bool(world2cam_poses)
    if trgt_sidelength is not None:
        cx = cx / width * trgt_sidelength
        cy = cy / height * trgt_sidelength
        f = trgt_sidelength / height * f

    fx = f
    if invert_y:
        fy = -f
    else:
        fy = f

    # Build the intrinsic matrices
    full_intrinsic = np.array([[fx, 0., cx, 0.],
                               [0., fy, cy, 0],
                               [0., 0, 1, 0],
                               [0, 0, 0, 1]])

    return full_intrinsic, grid_barycenter, scale, world2cam_poses


def get_rays_np(H, W, focal, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d


string2class_dict = {

        '02691156': 0,  # airplane, aeroplane, plane, nTrain=2831, nTest=808
        '02958343': 1,  # car, auto, automobile, machine, motorcar, nTrain=5247, nTest=1498
        '03636649': 2,  # lamp, nTrain=1623, nTest=462
        '04256520': 3,  # sofa, couch, lounge, nTrain=2221, nTest=633
        '04530566': 4,  # vessel, watercraft, nTrain=1358, nTest=386
        '02828884': 5,  # bench, nTrain=1271, nTest=362
        '03001627': 6,  # chair, nTrain=4745, nTest=1354
        '03691459': 7,  # loudspeaker,speaker,speaker unit,loudspeaker system,speaker system, nTrain=1133, nTest=322
        '04379243': 8,  # table, nTrain=5957, nTest=1700
        '02933112': 9,  # cabinet, nTrain=1100, nTest=313
        '03211117': 10,  # display, video display, nTrain=766, nTest=218
        '04090263': 11,  # rifle, nTrain=1660, nTest=473
        '04401088': 12,  # telephone, phone, telephone set, nTrain=736, nTest=209
    }

class2string_dict = {v:k for k, v in string2class_dict.items()}

def class_string_2_class_id(x):
    return string2class_dict[x]

class SceneInstanceDataset():
    """This creates a dataset class for a single object instance (such as a single car)."""

    def __init__(self, root_dir, full, instance_idx, instance_dir, uv,
                 img_sidelength=None,
                 specific_observation_idcs=None, rotate=False,
                 num_images=None, test=False, view_map=None):
        self.instance_idx = instance_idx
        self.img_sidelength = img_sidelength
        self.instance_dir = instance_dir
        self.instance_name = os.path.basename(self.instance_dir)
        self.root_dir = root_dir
        self.uv = uv
        self.full = full
        self.test = test
        self.view_map = view_map

        class_string = instance_dir.split('/')[-2]  # the object class is the second last directory of the instance directory
        self.instance_class = torch.Tensor([class_string_2_class_id(class_string)])

        color_dir = os.path.join(instance_dir, "image")
        pose_dir = os.path.join(instance_dir, "cameras.npz")

        self.color_paths = sorted(data_util.glob_imgs(color_dir))
        cameras = np.load(pose_dir)
        self.poses = [torch.from_numpy(cameras["world_mat_inv_"+str(idx)]).float() for idx in range(len(self.color_paths))]
        if specific_observation_idcs is not None:
            self.color_paths = util.pick(self.color_paths, specific_observation_idcs)
            self.poses = util.pick(self.poses, specific_observation_idcs)
        elif num_images is not None:
            random_idcs = np.random.choice(len(self.color_paths), size=num_images)
            self.color_paths = util.pick(self.color_paths, random_idcs)
            self.poses = util.pick(self.poses, random_idcs)

        intrinsics, _, _, _ = parse_intrinsics(os.path.join(self.root_dir, "intrinsics.txt"),
                                                    trgt_sidelength=self.img_sidelength)
        self.intrinsics = torch.Tensor(intrinsics)

    def set_img_sidelength(self, new_img_sidelength):
        """For multi-resolution training: Updates the image sidelength with whichimages are loaded."""
        self.img_sidelength = new_img_sidelength

    def __len__(self):
        return len(self.color_paths)

    def __getitem__(self, idx):
        rgb = data_util.load_rgb(self.color_paths[idx])

        tf = torch.eye(4)

        pose = self.poses[idx] @ tf

        # If testing always use the test context
        if self.test:
            view_map = self.view_map
            idx_other = view_map[self.color_paths[0].split("/")[-3]]
        else:
            idx_other = idx

            while idx_other == idx:
                idx_other = random.randint(0, len(self.color_paths) - 1)

        pose_other = self.poses[idx_other] @ tf
        rgb_other = data_util.load_rgb(self.color_paths[idx_other])

        if self.full:
            sample = {
                "instance_idx": torch.Tensor([self.instance_idx]).squeeze(),
                "rgb": torch.from_numpy(rgb).float(),
                "cam2world": pose,
                "cam2world_other": pose_other,
                "rgb_other": torch.from_numpy(rgb_other).float(),
                "uv": self.uv,
                "intrinsics": self.intrinsics,
                "class": self.instance_class,
                "idx_other": idx_other,
                "idx": idx,
                "instance_name": self.instance_name
            }
        else:
            sample = {
                "instance_idx": torch.Tensor([self.instance_idx]).squeeze(),
                "rgb": torch.from_numpy(rgb).float(),
                "cam2world": pose,
                "uv": self.uv,
                "intrinsics": self.intrinsics,
                "idx": idx,
                "class": self.instance_class,
                "instance_name": self.instance_name
            }

        return sample


def get_instance_datasets(root, max_num_instances=None, specific_observation_idcs=None,
                          sidelen=None, rotate=False, max_observations_per_instance=None, dataset_type='train',
                          local=False):
    object_classes = sorted(glob(os.path.join(root, "*/")))
    all_objects = []
    for object_class in object_classes:
        # file_list = open(object_class + dataset_type + ".lst", "r")
        file_list = open(object_class + 'softras_' + dataset_type + ".lst", "r")
        content = file_list.read()
        content_list = content.split("\n")
        content_list.pop()  # remove last element since that is empty after newline
        content_list = [object_class + sub for sub in content_list]  # appends path to every entry
        file_list.close()
        all_objects.append(content_list)
    instance_dirs = [y for x in all_objects for y in x]  # just flattens the list
    assert (len(instance_dirs) != 0), f"No objects in the directory {root}"

    if max_num_instances != None:
        instance_dirs = instance_dirs[:max_num_instances]

    dummy_img_path = data_util.glob_imgs(os.path.join(instance_dirs[0], "image"))[0]
    dummy_img = data_util.load_rgb(dummy_img_path)
    org_sidelength = dummy_img.shape[1]

    i, j = torch.meshgrid(torch.arange(0, 64), torch.arange(0, 64))

    uv = torch.stack([i.float(), j.float()], dim=-1).permute(1, 0, 2)

    uv = uv[None].permute(0, -1, 1, 2).permute(0, 2, 3, 1)
    uv = uv[0].float()

    random.seed(0)
    np.random.seed(0)

    view_map = {}
    for line in open("viewlist/src_dvr.txt", "r"):
        splits = line.split(" ")
        shapenet_id, view_id = splits[1], int(splits[2])
        view_map[shapenet_id] = view_id

    all_instances = [SceneInstanceDataset(root_dir=root, instance_idx=idx, instance_dir=dir,
                                          specific_observation_idcs=specific_observation_idcs, img_sidelength=64,
                                          num_images=max_observations_per_instance, uv=uv, full=True, test=True, view_map=view_map)
                          for idx, dir in enumerate(instance_dirs)]
    return all_instances


def get_num_instances(root_dir, dataset_type):
    object_classes = sorted(glob(os.path.join(root_dir, "*/")))
    all_objects = []
    for object_class in object_classes:
        file_list = open(object_class + 'softras_' + dataset_type + ".lst", "r")
        content = file_list.read()
        content_list = content.split("\n")
        content_list.pop()  # remove last element since that is empty after newline
        content_list = [object_class + sub for sub in content_list]  # appends path to every entry
        file_list.close()
        all_objects.append(content_list)
    all_objects = [y for x in all_objects for y in x]  # just flattens the list
    return len(all_objects)


class SceneClassDataset(torch.utils.data.Dataset):
    """Dataset for a class of objects, where each datapoint is a SceneInstanceDataset."""

    def __init__(self,
                 num_context,
                 full,
                 num_trgt,
                 root_dir,
                 vary_context_number=False,
                 query_sparsity=None,
                 img_sidelength=None,
                 max_num_instances=None,
                 max_observations_per_instance=None,
                 dataset_type='train',
                 specific_observation_idcs=None,
                 test=False,
                 test_context_idcs=None,
                 viewlist=None):

        self.full = full
        self.num_context = num_context
        self.num_trgt = num_trgt
        self.query_sparsity = query_sparsity
        self.img_sidelength = img_sidelength
        self.vary_context_number = vary_context_number
        self.test = test
        self.test_context_idcs = test_context_idcs

        if viewlist is not None:
            with open(viewlist, "r") as f:
                tmp = [x.strip().split() for x in f.readlines()]
            viewlist = {
                x[0] + "/" + x[1]: list(map(int, x[2:]))
                for x in tmp
            }

        object_classes = sorted(glob(os.path.join(root_dir, "*/")))
        all_objects = []
        for object_class in object_classes:
            file_list = open(object_class + 'softras_' + dataset_type + ".lst", "r")
            content = file_list.read()
            content_list = content.split("\n")
            content_list.pop() # remove last element since that is empty after newline
            content_list = [object_class + sub for sub in content_list] #appends path to every entry
            file_list.close()
            all_objects.append(content_list)
        all_objects = [y for x in all_objects for y in x] #just flattens the list
        self.instance_dirs = all_objects
        print(f"Root dir {root_dir}, {len(self.instance_dirs)} instances")

        dummy_img_path = data_util.glob_imgs(os.path.join(all_objects[0], "image"))[0]
        dummy_img = data_util.load_rgb(dummy_img_path)
        org_sidelength = dummy_img.shape[1]

        i, j = torch.meshgrid(torch.arange(0, 64), torch.arange(0, 64))
        uv = torch.stack([i.float(), j.float()], dim=-1).permute(1, 0, 2)

        uv = uv[None].permute(0, -1, 1, 2).permute(0, 2, 3, 1)
        uv = uv[0]
        uv = uv.float()

        assert (len(self.instance_dirs) != 0), "No objects in the data directory"

        if max_num_instances is not None:
            self.instance_dirs = self.instance_dirs[:max_num_instances]

        random.seed(0)
        np.random.seed(0)
        self.all_instances = []
        for idx, dir in tqdm(enumerate(self.instance_dirs)):
            viewlist_key = '/'.join(dir.split('/')[-2:])
            specific_observation_idcs = viewlist[viewlist_key] if viewlist is not None else specific_observation_idcs
            self.all_instances.append(SceneInstanceDataset(root_dir=root_dir,
                                                           full=full,
                                                           instance_idx=idx,
                                                           instance_dir=dir,
                                                           specific_observation_idcs=specific_observation_idcs,
                                                           img_sidelength=img_sidelength,
                                                           uv=uv,
                                                           num_images=max_observations_per_instance))

        self.num_per_instance_observations = [len(obj) for obj in self.all_instances]
        self.num_instances = len(self.all_instances)

    def flatten(self, dict):
        new_dict = {}
        for key in ['rgb', 'uv']:
            new_dict[key] = dict[key].flatten(0,1)
        for key, v in dict.items():
            if key not in ['rgb', 'uv']:
                new_dict[key] = dict[key]
        return new_dict

    def set_img_sidelength(self, new_img_sidelength):
        """For multi-resolution training: Updates the image sidelength with which images are loaded."""
        self.img_sidelength = new_img_sidelength
        for instance in self.all_instances:
            instance.set_img_sidelength(new_img_sidelength)

    def __len__(self):
        return np.sum(self.num_per_instance_observations)

    def get_instance_idx(self, idx):
        """Maps an index into all tuples of all objects to the idx of the tuple relative to the other tuples of that
        object
        """
        if self.test:
            obj_idx = 0
            while idx >= 0:
                idx -= self.num_per_instance_observations[obj_idx]
                obj_idx += 1
            return obj_idx - 1, int(idx + self.num_per_instance_observations[obj_idx - 1])
        else:
            return np.random.randint(self.num_instances), 0

    def collate_fn(self, batch_list):
        keys = batch_list[0].keys()
        result = defaultdict(list)

        for entry in batch_list:
            # make them all into a new dict
            for key in keys:
                result[key].append(entry[key])

        for key in keys:
            try:
                result[key] = torch.stack(result[key], dim=0)
            except:
                continue

        return result

    def __getitem__(self, idx):
        context = []
        trgt = []
        post_input = []

        obj_idx, det_idx = self.get_instance_idx(idx)

        if self.vary_context_number and self.num_context > 0:
            num_context = np.random.randint(1, self.num_context+1)

        if not self.test:
            try:
                sample_idcs = np.random.choice(len(self.all_instances[obj_idx]), replace=False,
                                               size=self.num_context+self.num_trgt)
            except:
                sample_idcs = np.random.choice(len(self.all_instances[obj_idx]), replace=True,
                                               size=self.num_context+self.num_trgt)

        for i in range(self.num_context):
            if self.test:
                sample = self.all_instances[obj_idx][self.test_context_idcs[i]]
            else:
                sample = self.all_instances[obj_idx][sample_idcs[i]]

            context.append(sample)

        for i in range(self.num_trgt):
            if self.test:
                sample = self.all_instances[obj_idx][det_idx]
            else:
                sample = self.all_instances[obj_idx][sample_idcs[i+self.num_context]]

            sample = self.flatten(sample)
            rix = np.random.permutation(sample['rgb'].shape[0])

            if self.query_sparsity is not None:
                rix = rix[:self.query_sparsity]
                sample['rgb'] = sample['rgb'][rix]
                sample['uv'] = sample['uv'][rix]

            trgt.append(sample)

        context = self.collate_fn(context)
        trgt = self.collate_fn(trgt)

        return {'context': context, 'query': trgt}, trgt
