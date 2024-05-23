import numpy as np
import torch
import torch.nn as nn
import yaml
import os
import sys
import source
import random
from source.data import nvs as data
from source.models_nvs import SRT, TransformingSRT
from source.trainer import SRTTrainer
from source.checkpoint import Checkpoint
from importlib import reload
import source.encoder 
import source.layers
import torch.optim as optim
import matplotlib.pyplot as plt
import einops
from source.utils.common import mse2psnr
import argparse
from pytorch_msssim import ssim as ssim_fn
from skimage.metrics import structural_similarity

# init
device = 'cuda'

# Learned Perceptual Image Patch Similarity
class LPIPS(object):
    '''
    borrowed from https://github.com/huster-wgm/Pytorch-metrics/blob/master/metrics.py
    '''
    def __init__(self, net='alex'):
        import lpips
        self.model = lpips.LPIPS(net=net).cuda()

    def __call__(self, y_pred, y_true, normalized=True):
        """
        args:
            y_true : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            y_pred : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            normalized : change [0,1] => [-1,1] (default by LPIPS)
        return LPIPS, smaller the better
        """
        if normalized:
            y_pred = y_pred * 2.0 - 1.0
            y_true = y_true * 2.0 - 1.0
        error =  self.model.forward(y_pred, y_true)
        return error
    

def load_model(config_name, datapath, checkpoint_path):
    with open(config_name, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.CLoader)
    cfg['data']['path']=datapath
    cfg['data']['kwargs']['return_org_rays'] = True
    out_dir = os.path.dirname(config_name)
    print('Loading eval set...')
    ### num of examples 
    batch_size = 1
    mode = 'test'
    test_dataset = data.get_dataset(mode, cfg['data'], full_scale=True)
    test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, num_workers=0, 
            sampler=None, shuffle=False,
            pin_memory=False, worker_init_fn=data.worker_init_fn, persistent_workers=False)

    if cfg['model']['model_type'] == 'srt':
        model = SRT(cfg['model']['args']).to(device)
    elif cfg['model']['model_type'] == 'tsrt':
        print(cfg['model'])
        model = TransformingSRT(cfg['model']['args']).to(device)
    print('Model created.')
    _optimizer = optim.AdamW(model.parameters(), lr=0.0001) # dummy
    _trainer = SRTTrainer(model, _optimizer, cfg, device, out_dir, test_dataset.render_kwargs)
    checkpoint = Checkpoint(out_dir, device=device, encoder=model.encoder,
                            decoder=model.decoder, optimizer=_optimizer)
    load_dict = checkpoint.load(checkpoint_path)
    return cfg, model, test_dataset, test_loader 


def evaluate(model, loader, cfg, n_target_views):
    mses= []
    psnrs = []
    ssims = []
    lpipss = []
    lpipss_alex = []
    lpips_fn = LPIPS(net='vgg')
    lpips_fn_alex = LPIPS(net='alex')

    if cfg['data']['dataset']=='clevrtr':
        H, W = 240, 320
    elif cfg['data']['dataset'] == 'msn':
        H, W = 128, 128
    else:
        raise NotImplementedError
    
    model.eval()
    loader = iter(loader)
    with torch.no_grad():
        for i, batch in enumerate(loader): 
            if cfg['model']['model_type'] == 'tsrt':
                extras = {'input_rays': batch.get('input_rays').to(device, non_blocking=True),
                          'input_transforms':  batch.get('input_transforms').to(device, non_blocking=True),
                          'input_coord': batch.get('input_coord').to(device, non_blocking=True)
                          }
                target_rays = batch['target_rays'].to(device, non_blocking=True)
                target_camera_pos = batch['target_camera_pos'].to(device, non_blocking=True)
                target_transforms = batch['target_transforms'].to(device, non_blocking=True)
                target_coord = batch.get('target_coord').to(device, non_blocking=True)
            else:
                extras = {'input_transforms':  batch.get('input_transforms').to(device, non_blocking=True)}
                target_rays = batch['target_rays'].to(device, non_blocking=True)
                target_rays = target_rays.reshape(target_rays.shape[0], n_target_views, -1, 3)
                target_camera_pos = batch['target_camera_pos'].to(device, non_blocking=True)
                target_camera_pos = target_camera_pos.reshape(target_rays.shape[0], n_target_views, -1, 3)
            model.eval()

            with torch.no_grad():
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16 if cfg['training']['mixed_prec'] else torch.float32):
                    z, extras = model.encoder(batch['input_images'].to(device), batch['input_camera_pos'].to(device), batch['input_rays'].to(device), extras=extras)
                    preds = []
                    for v in range(n_target_views):
                        if cfg['model']['model_type'] == 'tsrt':
                            extras.update({
                                'target_rays':  target_rays[:, v:v+1],
                                'target_transforms': target_transforms[:, v:v+1],
                                'target_coord': target_coord[:, v:v+1]})
                        _target_camera_pos = target_camera_pos[:, v:v+1].flatten(1,2)
                        _target_rays = target_rays[:, v:v+1].flatten(1,2)
                        pred, _ = model.decoder(z, _target_camera_pos, _target_rays, extras=extras)
                        preds.append(pred)
                    pred = torch.cat(preds, 1)
            
            pred = pred.reshape(-1, n_target_views, H, W, 3).permute(0,1,4,2,3).flatten(0,1)
            target = batch['target_pixels'].reshape(-1,  n_target_views, H, W, 3).permute(0,1,4,2,3).to(device).flatten(0,1)
            mse = torch.mean((pred-target)**2, axis=(1,2,3))
            psnr = mse2psnr(mse).mean().item()
            lpipss.append(lpips_fn(pred, target).mean().item())
            lpipss_alex.append(lpips_fn_alex(pred, target).mean().item())
            ssims.append(ssim_fn(pred, target, data_range=1.).mean().item())
            mses.append(mse.mean().item())
            psnrs.append(psnr)
            if (i+1) % 50 == 0:
                print(i, np.mean(psnrs), np.mean(ssims),  np.mean(lpipss), np.mean(lpipss_alex), np.mean(mses))
    return np.mean(psnrs), np.mean(ssims),  np.mean(lpipss),  np.mean(lpipss_alex), np.mean(mses)


if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate a scene representation model.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('datapath', type=str, help='Path to datadir')
    parser.add_argument('checkpoint_path', type=str, help='Path to checkpoint')
    
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True    

    cfg, model, test_dataset, test_loader = load_model(
        args.config,
        args.datapath,
        args.checkpoint_path
    )
    evaluate(model, test_loader, cfg, cfg['data']['kwargs']['num_target_views'])
