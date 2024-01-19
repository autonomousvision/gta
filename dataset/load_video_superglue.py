import numpy as np
import os, imageio
import os.path as osp
from skimage.transform import resize as imresize
from imageio import imwrite

current_dir = osp.dirname(os.path.abspath(__file__))
import sys
sys.path.append(current_dir + '/../')

from utils import data_util
import torch
from skimage.color import rgb2gray
from estimate_pose.glue_match import Matching
import cv2
import roma
from imageio import imread
from skimage.transform import resize as imresize

class Camera(object):
    def __init__(self, entry):
        fx, fy, cx, cy = entry[1:5]
        self.intrinsics = np.array([[fx, 0, cx, 0],
                                    [0, fy, cy, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]])
        w2c_mat = np.array(entry[7:]).reshape(3, 4)
        w2c_mat_4x4 = np.eye(4)
        w2c_mat_4x4[:3, :] = w2c_mat
        self.w2c_mat = w2c_mat_4x4
        self.c2w_mat = np.linalg.inv(w2c_mat_4x4)

def linear_interpolate(poses, n):
    start_pose = poses[0]
    end_pose = poses[1]

    start_t = start_pose[:3, -1]
    end_t = end_pose[:3, -1]
    start_R = start_pose[:3, :3]
    end_R = end_pose[:3, :3]

    interval = torch.linspace(0, 1, n)
    rots = roma.rotmat_slerp(torch.Tensor(start_R), torch.Tensor(end_R), interval)

    trans = start_t[None, :] + (end_t[None, :] - start_t[None, :]) * interval[:, None].numpy()

    rots = rots.numpy()

    render_poses = np.tile(np.eye(4)[None, :, :], (n, 1, 1))
    render_poses[:, :3, :3] = rots
    render_poses[:, :3, -1] = trans

    return render_poses


def make_circle(dir, n, radius=0.03):
    angles = np.linspace(0, 4 * np.pi, n)
    x = np.cos(angles) * radius
    y = np.sin(angles) * radius

    # x[:50] = 0
    # y[:50] = 0

    coord = np.stack([x, y, np.linspace(0, 1, n)], axis=-1)

    # Compute the axis of the rotation matrix
    axis_1 = np.array([1, 0, 0])
    axis_1 = axis_1 - (dir * axis_1).sum() * dir
    axis_1 = axis_1 / np.linalg.norm(axis_1)

    axis_2 = np.cross(axis_1, dir)

    rot_matrix = np.zeros((3, 3))

    rot_matrix[:, 0] = axis_1
    rot_matrix[:, 1] = axis_2
    rot_matrix[:, 2] = dir

    coord = np.matmul(rot_matrix, coord[:, :, None])[:, :, 0]
    return coord


def rotate_interpolate(poses, n):
    start_pose = poses[0]
    end_pose = poses[1]

    start_t = start_pose[:3, -1]
    end_t = end_pose[:3, -1]
    dir = end_t - start_t

    norm = np.linalg.norm(dir)
    dir_norm = dir / norm

    # trans_straight = start_t[None, :] + (end_t[None, :] - start_t[None, :]) * interval[:, None].numpy()

    trans = make_circle(dir_norm, n, radius=0.05)
    trans = trans * norm
    start_R = start_pose[:3, :3]
    end_R = end_pose[:3, :3]

    interval = torch.linspace(0, 1, n)
    rots = roma.rotmat_slerp(torch.Tensor(start_R), torch.Tensor(end_R), interval)

    rots = rots.numpy()

    render_poses = np.tile(np.eye(4)[None, :, :], (n, 1, 1))
    render_poses[:, :3, :3] = rots
    render_poses[:, :3, -1] = trans
    render_poses = render_poses[2:-2]

    return render_poses


def estimate_pose(kpts0, kpts1, K0, K1, thresh, conf=0.99999):
    if len(kpts0) < 5:
        return None

    f_mean = np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])
    norm_thresh = thresh / f_mean

    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    E, mask = cv2.findEssentialMat(
        kpts0, kpts1, np.eye(3), threshold=norm_thresh, prob=conf,
        method=cv2.RANSAC)

    assert E is not None

    best_num_inliers = 0
    ret = None
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(
            _E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            best_num_inliers = n
            ret = (R, t[:, 0], mask.ravel() > 0)
    return ret

########## Slightly modified version of LLFF data loading code 
##########  see https://github.com/Fyusion/LLFF for original

def _minify(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return
    
    from shutil import copy
    from subprocess import check_output
    
    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgdir_orig = imgdir
    
    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
            resizearg = '{}%'.format(100./r)
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue
            
        print('Minifying', r, basedir)
        
        os.makedirs(imgdir)
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)
        
        ext = imgs[0].split('.')[-1]
        args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)
        
        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')
            
        
        
        
def _load_data(basedir, factor=None, width=None, height=None, load_imgs=True):
    
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0])
    bds = poses_arr[:, -2:].transpose([1,0])
    
    img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    sh = imageio.imread(img0).shape
    
    sfx = ''
    
    if factor is not None:
        sfx = '_{}'.format(factor)
        _minify(basedir, factors=[factor])
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1
    
    imgdir = os.path.join(basedir, 'images' + sfx)
    if not os.path.exists(imgdir):
        print( imgdir, 'does not exist, returning' )
        return
    
    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    if poses.shape[-1] != len(imgfiles):
        print( 'Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), poses.shape[-1]) )
        return
    
    sh = imageio.imread(imgfiles[0]).shape
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1./factor

    h, w = sh[0], sh[1]
    im_size = 256

    sh, sw = int(256 / h * h), int(256 / h * w)
    # height is less than width

    poses[:2, 4, :] = np.array((256, 256)).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 256 / h
    
    if not load_imgs:
        return poses, bds
    
    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)

    imgs = []

    for img_file in imgfiles:
        im = imread(img_file)
        im = imresize(im, (sh, sw))
        im = data_util.square_crop_img(im)

        imgs.append(im)
    imgs = np.stack(imgs, -1)  
    
    print('Loaded image data', imgs.shape, poses[:,-1,0])
    return poses, bds, imgs

def expand_pose(pose):
    ncam = pose.shape[0]
    pose_fill = np.tile(np.eye(4)[None, :, :], (ncam, 1, 1))
    pose_fill[:, :3, :] = pose

    return pose_fill

def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3,:3].T, (pts-c2w[:3,3])[...,np.newaxis])[...,0]
    return tt

def poses_avg(poses):

    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    
    return c2w



def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w[:,4:5]
    
    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads) 
        z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses
    

def transform_pose(poses, tf):
    # cam2world = poses[:, :, :-1]
    # aux_info = poses[:, :, -1:]

    poses[:, :3, 1] = -1 * poses[:, :3, 1]
    poses[:, :3, 2] = -1 * poses[:, :3, 2]
    # cam2world = np.sum(tf[None, :, :, None] * poses[:, None, :, :], axis=-2)
    # cam2world = np.matmul(cam2world[:, :, :], tf[None, :, :])
    # cam2world = np.matmul(tf[None, :, :], cam2world[:, :, :])
    # cam2world = np.concatenate([cam2world, aux_info], axis=-1)
    return poses


def recenter_poses(poses):

    poses_ = poses+0
    bottom = np.reshape([0,0,0,1.], [1,4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3,:4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1])
    poses = np.concatenate([poses[:,:3,:4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:,:3,:4] = poses[:,:3,:4]
    poses = poses_
    return poses


#####################


def spherify_poses(poses, bds):
    
    p34_to_44 = lambda p : np.concatenate([p, np.tile(np.reshape(np.eye(4)[-1,:], [1,1,4]), [p.shape[0], 1,1])], 1)
    
    rays_d = poses[:,:3,2:3]
    rays_o = poses[:,:3,3:4]

    def min_line_dist(rays_o, rays_d):
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0,2,1])
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i, [0,2,1]) @ A_i).mean(0)) @ (b_i).mean(0))
        return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)
    
    center = pt_mindist
    up = (poses[:,:3,3] - center).mean(0)

    vec0 = normalize(up)
    vec1 = normalize(np.cross([.1,.2,.3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)

    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:,:3,:4])

    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:,:3,3]), -1)))
    
    sc = 1./rad
    poses_reset[:,:3,3] *= sc
    bds *= sc
    rad *= sc
    
    centroid = np.mean(poses_reset[:,:3,3], 0)
    zh = centroid[2]
    radcircle = np.sqrt(rad**2-zh**2)
    new_poses = []
    
    for th in np.linspace(0.,2.*np.pi, 120):

        camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
        up = np.array([0,0,-1.])

        vec2 = normalize(camorigin)
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = camorigin
        p = np.stack([vec0, vec1, vec2, pos], 1)

        new_poses.append(p)

    new_poses = np.stack(new_poses, 0)
    
    new_poses = np.concatenate([new_poses, np.broadcast_to(poses[0,:3,-1:], new_poses[:,:3,-1:].shape)], -1)
    poses_reset = np.concatenate([poses_reset[:,:3,:4], np.broadcast_to(poses[0,:3,-1:], poses_reset[:,:3,-1:].shape)], -1)
    
    return poses_reset, new_poses, bds
    

def parse_pose_file(file):
    f = open(file, 'r')
    cam_params = {}
    for i, line in enumerate(f):
        if i == 0:
            continue
        entry = [float(x) for x in line.split()]
        id = int(entry[0])
        cam_params[id] = Camera(entry)
    return cam_params

def get_camera_pose(path, path2, uv, views=40, sf=1.2):

    config = {
        'superpoint': {
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': 1024
        },
        'superglue': {
            'weights': 'indoor',
            'sinkhorn_iterations': 20,
            'match_threshold': 0.2,
        }
    }

    matching = Matching(config).eval().cuda()

    im0 = imread(path)
    im1 = imread(path2)

    h, w, _ = im0.shape
    offset = (w - h) // 2

    image0 = imresize(im0[:, offset:-offset], (256, 256))
    image1 = imresize(im1[:, offset:-offset], (256, 256))

    context_rgbs = np.stack([(image0 - 0.5) * 2, (image1 - 0.5) * 2], axis=0)
    image0 = rgb2gray(image0)
    image1 = rgb2gray(image1)

    inp0 = torch.Tensor(image0).cuda()[None, None]
    inp1 = torch.Tensor(image1).cuda()[None, None]


    with torch.no_grad():
        pred = matching({'image0': inp0, 'image1': inp1})
    pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
    kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
    matches, conf = pred['matches0'], pred['matching_scores0']

    # Keep the matching keypoints.
    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    mconf = conf[valid]

    K0 = np.array([[225., 0.0, 128.0, 0.0], [0.0, 225., 128.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    intrinsics = K0

    K0 = K0[:3, :3]
    K1 = K0

    thresh = 1.  # In pixels relative to resized image size.
    ret = estimate_pose(mkpts0, mkpts1, K0, K1, thresh)
    R, t = ret[:2]

    pose1 = np.eye(4)
    pose2 = np.eye(4)
    pose2[:3, :3] = R
    pose2[:3, -1] = t
    pose2 = np.linalg.inv(pose2)

    # The sf will need to be tuned for different images
    pose2_superglue = pose2
    pose2[:3, -1] = pose2[:3, -1] / sf


    pose_new = np.stack([pose1, pose2], axis=0)

    context_c2w = pose_new # poses[idxs]
    context_intrinsics = np.tile(intrinsics[None, :, :], (context_c2w.shape[0], 1, 1))

    render_poses = rotate_interpolate(context_c2w, 80)

    query_rgbs = np.tile(context_rgbs[:1], (render_poses.shape[0], 1, 1, 1))
    query_c2w = render_poses

    query_intrinsics = np.tile(intrinsics[None, :, :], (query_c2w.shape[0], 1, 1))

    query = {'rgb': torch.Tensor(query_rgbs)[None].float(),
             'cam2world': torch.Tensor(query_c2w)[None].float(),
             'intrinsics': torch.Tensor(query_intrinsics)[None].float(),
             'uv': uv.view(-1, 2)[None, None].expand(1, query_c2w.shape[0], -1, -1)}
    ctxt = {'rgb': torch.Tensor(context_rgbs)[None].float(),
            'cam2world': torch.Tensor(context_c2w)[None].float(),
            'intrinsics': torch.Tensor(context_intrinsics)[None].float()}

    return {'query': query, 'context': ctxt}
