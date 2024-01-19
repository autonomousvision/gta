"""
checkpoint (*WITH SOFTRAS SPLIT*) under 
/om2/user/sitzmann/logs/light_fields/NMR_hyper_1e2_reg_layernorm/64_256_None/checkpoints/model_epoch_0087_iter_250000.pth
"""

# Enable import from parent package
import sys
import os
sys.path.append(os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/../' )
import random

import torch
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from dataset.realestate10k_dataio import RealEstate10k, get_camera_pose
import torch
import models
import training
import configargparse
from torch.utils.data import DataLoader
import loss_functions
import summaries
from utils import util
import config
from tqdm import tqdm
from imageio import imwrite, get_writer
from glob import glob
import lpips
from skimage.metrics import structural_similarity
import cv2
import matplotlib.pyplot as plt

img2mse = lambda x, y: torch.mean((x - y) ** 2)
mse2psnr = lambda x: -10. * torch.log(x) / torch.log(torch.Tensor([10.]).to(x.device))

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True)

p.add_argument('--save_img', action='store_true', default=False)
p.add_argument('--logging_root', type=str, default=config.logging_root)
p.add_argument('--data_root', type=str, default='/om2/user/egger/MultiClassSRN/data/NMR_Dataset', required=False)
p.add_argument('--val_root', type=str, default=None, required=False)
p.add_argument('--network', type=str, default='relu')
p.add_argument('--category', type=str, default='donut')
p.add_argument('--conditioning', type=str, default='hyper')
p.add_argument('--experiment_name', type=str, required=True)
p.add_argument('--num_context', type=int, default=0)
p.add_argument('--batch_size', type=int, default=48)
p.add_argument('--max_num_instances', type=int, default=None)
p.add_argument('--num_trgt', type=int, default=1)
p.add_argument('--gpus', type=int, default=1)
p.add_argument('--views', type=int, default=1)
p.add_argument('--RTA', action='store_true', default=False)
p.add_argument('--so3', action='store_true', default=False)
p.add_argument('--idx', type=int, default=0)
p.add_argument('--npoints', type=int, default=64)
p.add_argument('--kv_trnsfm', action='store_true', default=False)

# General training options
p.add_argument('--lr', type=float, default=5e-4)
p.add_argument('--num_epochs', type=int, default=40001)
p.add_argument('--reconstruct', action='store_true', default=False)
p.add_argument('--nearest', action='store_true', default=False)
p.add_argument('--model', type=str, default='midas_vit')
p.add_argument('--autodecoder', action='store_true', default=False)
p.add_argument('--epochs_til_ckpt', type=int, default=10)
p.add_argument('--steps_til_summary', type=int, default=500)
p.add_argument('--iters_til_ckpt', type=int, default=10000)
p.add_argument('--checkpoint_path', default=None)

p.add_argument('--load_dino', default=False)
opt = p.parse_args()


def sync_model(model):
    for param in model.parameters():
        dist.broadcast(param.data, 0)

def worker_init_fn(worker_id):
    random.seed(int(torch.utils.data.get_worker_info().seed)%(2**32-1))
    np.random.seed(int(torch.utils.data.get_worker_info().seed)%(2**32-1))

def make_circle(n, radius=0.2):
    angles = np.linspace(0, 4 * np.pi, n)
    x = np.cos(angles) * radius
    y = np.sin(angles) * radius

    coord = np.stack([x, y, np.zeros(n)], axis=-1)
    return coord

def get_img_writer(dirpath, title):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    def save_img(rgb, i):
        H, W = rgb.shape[0:2]
        fig, ax = plt.subplots(1, 1, frameon=False)
        fig.set_size_inches(W, H)  # Note the order
        im = ax.imshow(rgb, aspect=1)
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(dirpath, '{:04d}.jpg'.format(i)), bbox_inches='tight', pad_inches=0, dpi=1)  # Increased dpi
        
        plt.close(fig)  # Close the figure to free up memory

    return save_img

def append_white_space(img):
    H, W = img.shape[:2]
    return np.concatenate([
        (255*np.ones((32, W, 3))).astype(np.uint8), 
        img
    ], 0)
        

def render_data(model_input, scene, model, idx, method, nrays_per_oneff=8192, save_img=False):
    model_input = util.dict_to_gpu(model_input)
    print(model_input['context']['rgb'].shape, 
          model_input['context']['rgb'].max(), 
          model_input['context']['rgb'].min())
    nrender = model_input['query']['cam2world'].size(1)
    query_cam2world = model_input['query']['cam2world']

    context_cam2world = model_input['context']['cam2world']

    query_intrinsic = model_input['query']['intrinsics']
    uv = model_input['query']['uv']
    nrays = uv.size(-2)

    chunks = nrays // nrays_per_oneff
    z = model.get_z(model_input)

    query_cam2world = model_input['query']['cam2world']
    query_intrinsic = model_input['query']['intrinsics']
    trans = torch.Tensor(make_circle(query_cam2world.shape[1])).cuda()
    query_cam2world[0, :, :3, -1] = query_cam2world[0, :, :3, -1]

    scene = str(scene)
    scene_path = scene.split("/")[-1]

    if not os.path.exists('vis/'):
        os.makedirs('vis/')
    if save_img:
        if method == 'baseline':
            title = 'Du et al. (2023)'
        else:
            title = method
        writer = get_img_writer("imgs_realestate/{}/{}/".format(idx, method), title=title)
        gt_writer = get_img_writer("imgs_realestate/{}/gt/".format(idx), title='Ground truth')
        cntxt_writer = get_img_writer("imgs_realestate/{}/cntxt/".format(idx), title='Context images')
    else:
        dirpath = 'imgs_realestate/{}/'.format(idx)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        writer = get_writer(os.path.join(dirpath, "{}.mp4".format(method)))
        gt_writer = get_writer(os.path.join(dirpath,"gt.mp4"))
        cntxt_writer = get_writer(os.path.join(dirpath, "cntxt.mp4"))
    loss_fn_alex = lpips.LPIPS(net='vgg').cuda()
    mses = []
    ssims = []
    psnrs = []
    lpips_list = []

    with torch.no_grad():
        for i in tqdm(range(nrender)):
            model_input['query']['cam2world'] = query_cam2world[:, i:i+1]
            model_input['query']['intrinsics'] = query_intrinsic[:, i:i+1]

            uv_i = uv[:, i:i+1]

            uv_chunks = torch.chunk(uv_i, chunks, dim=2)

            model_outputs = []

            for uv_chunk in uv_chunks:
                model_input['query']['uv'] = uv_chunk
                model_output = model(model_input, z=z)
                del model_output['z']
                del model_output['coords']
                del model_output['uv']
                del model_output['pixel_val']
                del model_output['at_wts']

                model_outputs.append(model_output)

            model_output_full = {}

            for k in model_outputs[0].keys():
                outputs = [model_output[k] for model_output in model_outputs]
                val = torch.cat(outputs, dim=-2)
                model_output_full[k] = val

            rgb = model_output_full['rgb'].view(256, 256, 3)
            rgb_gt = model_input['query']['rgb'][0, i]

            rgb = rgb_np = rgb.detach().cpu().numpy()
            rgb_gt = target_np = rgb_gt.detach().cpu().numpy()

            rgb = np.clip(rgb, -1, 1)
            rgb_gt = np.clip(rgb_gt, -1, 1)

            rgb = (rgb + 1) / 2.
            target = (rgb_gt + 1) / 2

            rgb = torch.Tensor(rgb).cuda()
            target = torch.Tensor(target).cuda()

            rgb_lpips = ((rgb.permute(2, 0, 1) - 0.5) * 2)[None, :, :, :].cuda()
            target_lpips = ((target.permute(2, 0, 1) - 0.5) * 2)[None, :, :, :].cuda()

            mse = img2mse(rgb, target)
            psnr = mse2psnr(mse)

            mses.append(mse.item())
            psnrs.append(psnr.item())

            lpip = loss_fn_alex(rgb_lpips, target_lpips).item()
            lpips_list.append(lpip)

            ssim = structural_similarity(rgb_np, target_np, win_size=11, multichannel=True, gaussian_weights=True)
            ssims.append(ssim)

            print("mse, psnr, lpip, ssim", np.mean(mses), np.mean(psnrs), np.mean(lpips_list), np.mean(ssims))

            target = (target.detach().cpu().numpy() * 255).astype(np.uint8)
            rgb_np = (rgb.detach().cpu().numpy() * 255).astype(np.uint8)

            cntxt_rgb = model_input['context']['rgb']
            cntxt_rgb = (cntxt_rgb + 1) / 2.
            cntxt_rgb = torch.cat(
                [cntxt_rgb[0, i] for i in range(2)]
                , 1).detach().cpu().numpy() * 255
            cntxt_rgb=cntxt_rgb.astype(np.uint8)

            if opt.save_img:
                
                writer(rgb_np, i)
                gt_writer(target, i)
                cntxt_writer(cntxt_rgb, i)
            else:

                writer.append_data(append_white_space(rgb_np))
                gt_writer.append_data(append_white_space(target))
                cntxt_writer.append_data(append_white_space(cntxt_rgb))

    pass

def render(gpu, opt):
    if opt.gpus > 1:
        dist.init_process_group(backend='nccl', init_method='tcp://localhost:1492', world_size=opt.gpus, rank=gpu)

    torch.cuda.set_device(gpu)

    test_dataset = RealEstate10k(img_root="data_download/realestate/test",
                                 pose_root="poses/realestate/test.mat",
                                 num_ctxt_views=1, num_query_views=1, query_sparsity=256,
                                 )

    train_scenes = [str(s).split("/")[-1] for s in test_dataset.all_scenes]

    model = models.CrossAttentionRenderer(model=opt.model, n_view=opt.views, GTA=opt.GTA, npoints=opt.npoints, kv_trnsfm=opt.kv_trnsfm)
    old_state_dict = model.state_dict()

    if opt.checkpoint_path is not None:
        print(f"Loading weights from {opt.checkpoint_path}...")
        state_dict = torch.load(opt.checkpoint_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict['model'], strict=False)#, strict=not opt.reconstruct)

    model = model.cuda()
    scenes = [s.stem for s in test_dataset.all_scenes]
    all_scene = test_dataset.all_scenes[opt.idx]

    #try:
    data = get_camera_pose(all_scene, "RealEstate10K/test", test_dataset.uv, views=opt.views)
    #except:
    #    continue
    if opt.RTA:
        if opt.so3:
            method = 'GTA'
        else:
            method = 'GTAwoSO3'
    else:
        method = 'baseline'
    render_data(data, all_scene, model, opt.idx, method, int(8192/(opt.npoints/64)), opt.save_img)

if __name__ == "__main__":
    opt = p.parse_args()
    render(0, opt)
