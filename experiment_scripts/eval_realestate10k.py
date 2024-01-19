# Enable import from parent package
import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/../' )
import random

import torch
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from dataset.realestate10k_dataio import RealEstate10kVis, get_camera_pose
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
import time
import matplotlib.pyplot as plt

import lpips
from skimage.metrics import structural_similarity
from skimage.transform import resize as imresize
import cv2

# torch.manual_seed(0)

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True)

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

# General training options
p.add_argument('--lr', type=float, default=5e-4)
p.add_argument('--num_epochs', type=int, default=40001)
p.add_argument('--reconstruct', action='store_true', default=False)
p.add_argument('--local', action='store_true', default=False)
p.add_argument('--local_coord', action='store_true', default=False)
p.add_argument('--learned_local_coord', action='store_true', default=False)
p.add_argument('--global_local_coord', action='store_true', default=False)
p.add_argument('--model', type=str, default='midas_vit')
p.add_argument('--autodecoder', action='store_true', default=False)
p.add_argument('--epochs_til_ckpt', type=int, default=10)
p.add_argument('--steps_til_summary', type=int, default=500)
p.add_argument('--iters_til_ckpt', type=int, default=10000)
p.add_argument('--checkpoint_path', default=None)
p.add_argument('--GTA', action='store_true', default=False)
p.add_argument('--npoints', type=int, default=64)
p.add_argument('--kv_trnsfm', action='store_true', default=False)
p.add_argument('--imsize', type=int, default=256)


# Ablations
p.add_argument('--no_multiview', action='store_true', default=False)
p.add_argument('--no_sample', action='store_true', default=False)
p.add_argument('--no_latent_concat', action='store_true', default=False)
p.add_argument('--no_data_aug', action='store_true', default=False)

opt = p.parse_args()

img2mse = lambda x, y: torch.mean((x - y) ** 2)
mse2psnr = lambda x: -10. * torch.log(x) / torch.log(torch.Tensor([10.]).to(x.device))

imsize = opt.imsize

def sync_model(model):
    for param in model.parameters():
        dist.broadcast(param.data, 0)

def worker_init_fn(worker_id):
    random.seed(int(torch.utils.data.get_worker_info().seed)%(2**32-1))
    np.random.seed(int(torch.utils.data.get_worker_info().seed)%(2**32-1))


def make_circle(n, radius=0.1):
    angles = np.linspace(0, 4 * np.pi, n)
    x = np.cos(angles) * radius
    y = np.sin(angles) * radius

    coord = np.stack([x, y, np.zeros(n)], axis=-1)
    return coord


def multigpu_train(gpu, opt):
    if opt.gpus > 1:
        dist.init_process_group(backend='nccl', init_method='tcp://localhost:1492', world_size=opt.gpus, rank=gpu)

    torch.cuda.set_device(gpu)

    val_dataset = RealEstate10kVis(img_root="data_download/realestate/test",
                                 pose_root="poses/realestate/test.mat",
                                 num_ctxt_views=opt.views, num_query_views=1, augment=False, always_resize=True,
                                 imsize=opt.imsize)

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, drop_last=True, num_workers=0, pin_memory=False, worker_init_fn=worker_init_fn)

    model = models.CrossAttentionRenderer(no_multiview=opt.no_multiview, no_sample=opt.no_sample, no_latent_concat=opt.no_latent_concat, model=opt.model, n_view=opt.views, GTA=opt.GTA, kv_trnsfm=opt.kv_trnsfm, npoints=opt.npoints)
    old_state_dict = model.state_dict()

    if opt.checkpoint_path is not None:
        print(f"Loading weights from {opt.checkpoint_path}...")
        state_dict = torch.load(opt.checkpoint_path, map_location=torch.device('cpu'))
        if opt.reconstruct:
            state_dict['latent_codes.weight'] = torch.zeros_like(state_dict['latent_codes.weight'])

        # state_dict['encoder.latent'] = old_state_dict['encoder.latent']

        model.load_state_dict(state_dict['model'], strict=False)

    model = model.cuda().eval()
    device = "gpu"

    with torch.no_grad():
        loss_fn_alex = lpips.LPIPS(net='vgg').cuda()

        mses = []
        psnrs = []
        lpips_list = []
        ssims = []

        for val_i, (model_input, gt) in enumerate(val_loader):
            if device == 'gpu':
                model_input = util.dict_to_gpu(model_input)
                gt = util.dict_to_gpu(gt)

            model_input_full = model_input
            rgb_full = model_input['query']['rgb']
            uv_full = model_input['query']['uv']

            nrays = uv_full.size(2)

            z = model.get_z(model_input)
    
            if opt.views == 3 or imsize > 256:
                rgb_chunks = torch.chunk(rgb_full, 18, dim=2)
                uv_chunks = torch.chunk(uv_full, 18, dim=2)
            else:
                rgb_chunks = torch.chunk(rgb_full, 9, dim=2)
                uv_chunks = torch.chunk(uv_full, 9, dim=2)

            start = time.time()

            model_outputs = []

            for rgb_chunk, uv_chunk in zip(rgb_chunks, uv_chunks):
                model_input['query']['rgb'] = rgb_chunk
                model_input['query']['uv'] = uv_chunk

                model_output = model(model_input, z=z, val=True)

                model_outputs.append(model_output)

            end = time.time()
            print("elapsed: ", end - start)
            model_output_full = {}
            for k in ['rgb', 'valid_mask', 'depth_ray']:
                outputs = [model_output[k] for model_output in model_outputs]

                if k == "pixel_val":
                    val = torch.cat(outputs, dim=-3)
                else:
                    val = torch.cat(outputs, dim=-2)
                model_output_full[k] = val

            rgb = model_output_full['rgb'].view(imsize, imsize, 3)
            valid_mask = model_output_full['valid_mask'].view(imsize, imsize, 1)

            # Saving output image
            target = gt['rgb'].view(imsize, imsize, 3)

            rgb = ((rgb + 1) * 0.5).detach() * valid_mask + 0.5 * (1 - valid_mask) * torch.ones_like(rgb)
            target = ((target + 1) * 0.5).detach() * valid_mask + 0.5 * (1 - valid_mask) * torch.ones_like(target)

            mse = img2mse(rgb, target)
            psnr = mse2psnr(mse)
            mses.append(mse.item())
            psnrs.append(psnr.item())

            rgb_lpips = ((rgb.permute(2, 0, 1) - 0.5) * 2)[None, :, :, :].cuda()
            target_lpips = ((target.permute(2, 0, 1) - 0.5) * 2)[None, :, :, :].cuda()
            lpip = loss_fn_alex(rgb_lpips, target_lpips).item()
            lpips_list.append(lpip)

            rgb_np = rgb.detach().cpu().numpy()
            target_np = target.detach().cpu().numpy()
            ssim = structural_similarity(rgb_np, target_np, win_size=11, multichannel=True, gaussian_weights=True)
            ssims.append(ssim)

            print("[last ex] mse, psnr, lpip, ssim", mses[-1], psnrs[-1], lpips_list[-1], ssims[-1])
            print("i, mse, psnr, lpip, ssim", val_i, np.mean(mses), np.mean(psnrs), np.mean(lpips_list), np.mean(ssims))

        import pdb
        pdb.set_trace()
        print("here")


if __name__ == "__main__":
    # manager = Manager()
    # shared_dict = manager.dict()

    opt = p.parse_args()
    if opt.gpus > 1:
        mp.spawn(multigpu_train, nprocs=opt.gpus, args=(opt,))
    else:
        multigpu_train(0, opt)
