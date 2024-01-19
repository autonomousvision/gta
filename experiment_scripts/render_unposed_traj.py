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
from dataset.load_video_superglue import get_camera_pose
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
p.add_argument('--views', type=int, default=2)

# General training options
p.add_argument('--lr', type=float, default=5e-4)
p.add_argument('--num_epochs', type=int, default=40001)
p.add_argument('--reconstruct', action='store_true', default=False)
p.add_argument('--nearest', action='store_true', default=False)
p.add_argument('--model', type=str, default='midas_vit')
p.add_argument('--im1', type=str, default='image/demo_first.png')
p.add_argument('--im2', type=str, default='image/demo_second.png')
p.add_argument('--epochs_til_ckpt', type=int, default=10)
p.add_argument('--steps_til_summary', type=int, default=500)
p.add_argument('--iters_til_ckpt', type=int, default=10000)
p.add_argument('--checkpoint_path', default=None)
opt = p.parse_args()


def sync_model(model):
    for param in model.parameters():
        dist.broadcast(param.data, 0)

def worker_init_fn(worker_id):
    random.seed(int(torch.utils.data.get_worker_info().seed)%(2**32-1))
    np.random.seed(int(torch.utils.data.get_worker_info().seed)%(2**32-1))


def render_data(model_input, scene, model):
    model_input = util.dict_to_gpu(model_input)

    nrender = model_input['query']['cam2world'].size(1)

    uv = model_input['query']['uv']
    nrays = uv.size(-2)
    chunks = nrays // 8192 + 1
    z = model.get_z(model_input)

    query_cam2world = model_input['query']['cam2world']
    query_intrinsic = model_input['query']['intrinsics']

    writer = get_writer("unposed.mp4")

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
                # del model_output['ent']
                del model_output['uv']
                del model_output['pixel_val']
                del model_output['at_wts']

                model_outputs.append(model_output)

            model_output_full = {}

            for k in model_outputs[0].keys():
                outputs = [model_output[k] for model_output in model_outputs]
                print(k, [output.size() for output in outputs])
                val = torch.cat(outputs, dim=-2)
                model_output_full[k] = val

            rgb = model_output_full['rgb'].view(256, 256, 3)
            rgb_gt = model_input['query']['rgb'][0, i]

            panel_im = torch.cat([rgb], dim=1).detach().cpu().numpy()
            panel_im = np.clip(panel_im, -1, 1)
            writer.append_data(panel_im)

    writer.close()

    pass

def render(gpu, opt):
    torch.cuda.set_device(gpu)

    model = models.CrossAttentionRenderer(model=opt.model, n_view=opt.views)
    old_state_dict = model.state_dict()

    i, j = torch.meshgrid(torch.arange(0, 256), torch.arange(0, 256))
    uv = torch.stack([i.float(), j.float()], dim=-1).permute(1, 0, 2)

    uv = uv[None].permute(0, -1, 1, 2).permute(0, 2, 3, 1)
    uv = uv.reshape(-1, 2)

    if opt.checkpoint_path is not None:
        print(f"Loading weights from {opt.checkpoint_path}...")
        state_dict = torch.load(opt.checkpoint_path, map_location=torch.device('cpu'))

        model.load_state_dict(state_dict['model'], strict=not opt.reconstruct)

    model = model.cuda()

    data = get_camera_pose(opt.im1, opt.im2, uv)
    render_data(data, "./", model)

if __name__ == "__main__":
    render(0, opt)
