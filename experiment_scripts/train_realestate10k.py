# Enable import from parent package
import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
import random

import torch
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from dataset.realestate10k_dataio import RealEstate10k
import torch
import models
import training
import configargparse
from torch.utils.data import DataLoader
import loss_functions
import summaries
import config


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
p.add_argument('--batch_size', type=int, default=12)
p.add_argument('--num_trgt', type=int, default=1)
p.add_argument('--views', type=int, default=2)
p.add_argument('--gpus', type=int, default=1)
p.add_argument('--port', type=int, default=1492)

# General training options
p.add_argument('--lr', type=float, default=5e-5)
p.add_argument('--num_epochs', type=int, default=40001)
p.add_argument('--lpips', action='store_true', default=False)
p.add_argument('--depth', action='store_true', default=False)
p.add_argument('--lpips_weight', type=float, default=0.1)
p.add_argument('--depth_weight', type=float, default=0.05)
p.add_argument('--model', type=str, default='midas_vit')
p.add_argument('--epochs_til_ckpt', type=int, default=10)
p.add_argument('--steps_til_summary', type=int, default=500)
p.add_argument('--iters_til_ckpt', type=int, default=50000)
p.add_argument('--checkpoint_path', default=None)
p.add_argument('--GTA', action='store_true', default=False)
p.add_argument('--imsize', type=int, default=256)
p.add_argument('--upsample', action='store_true', default=False)

# Ablations
p.add_argument('--no_multiview', action='store_true', default=False)
p.add_argument('--no_sample', action='store_true', default=False)
p.add_argument('--no_latent_concat', action='store_true', default=False)
p.add_argument('--no_data_aug', action='store_true', default=False)
p.add_argument('--no_high_freq', action='store_true', default=False)

opt = p.parse_args()


def sync_model(model):
    for param in model.parameters():
        dist.broadcast(param.data, 0)

def worker_init_fn(worker_id):
    random.seed(int(torch.utils.data.get_worker_info().seed)%(2**32-1))
    np.random.seed(int(torch.utils.data.get_worker_info().seed)%(2**32-1))


def multigpu_train(gpu, opt):
    if opt.gpus > 1:
        dist.init_process_group(backend='nccl', init_method='tcp://localhost:{}'.format(opt.port), world_size=opt.gpus, rank=gpu)

    torch.cuda.set_device(gpu)

    def create_dataloader_callback(sidelength, batch_size, query_sparsity):
        train_dataset = RealEstate10k(img_root="data_download/realestate/train",
                                     pose_root="poses/realestate/train.mat",
                                     num_ctxt_views=opt.views, num_query_views=1, query_sparsity=192,
                                     lpips=opt.lpips, augment=(not opt.no_data_aug), always_resize=True,
                                     imsize=opt.imsize)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  drop_last=True, num_workers=8, pin_memory=False, worker_init_fn=worker_init_fn)

        val_dataset = RealEstate10k(img_root="data_download/realestate/test",
                                      pose_root="poses/realestate/test.mat",
                                      num_ctxt_views=opt.views, num_query_views=1, augment=False, always_resize=True,
                                      imsize=opt.imsize)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True, drop_last=True, num_workers=4, pin_memory=False, worker_init_fn=worker_init_fn)

        return train_loader, val_loader

    model = models.CrossAttentionRenderer(
        no_multiview=opt.no_multiview, 
        no_sample=opt.no_sample, 
        no_latent_concat=opt.no_latent_concat, 
        no_high_freq=opt.no_high_freq, 
        model=opt.model,
        n_view=opt.views,
        GTA=opt.GTA)
    
    old_state_dict = model.state_dict()

    optimizer = torch.optim.Adam(lr=opt.lr, params=model.parameters(), betas=(0.99, 0.999))

    if opt.checkpoint_path is not None:
        print(f"Loading weights from {opt.checkpoint_path}...")
        state_dict = torch.load(opt.checkpoint_path, map_location=torch.device('cpu'))
        state_dict, optimizer_dict = state_dict['model'], state_dict['optimizer']

        model.load_state_dict(state_dict, strict=False)
        # optimizer.load_state_dict(optimizer_dict)

        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

    model = model.cuda()

    if opt.gpus > 1:
        sync_model(model)

    # Define the loss
    summary_fn = summaries.img_summaries
    val_summary_fn = summaries.img_summaries
    root_path = os.path.join(opt.logging_root, opt.experiment_name)
    loss_fn = val_loss_fn = loss_functions.LFLoss(
        opt.lpips_weight if opt.lpips else 0,
        opt.depth_weight if opt.depth else 0,
        upsample=opt.upsample)

    training.training(model=model, dataloader_callback=create_dataloader_callback,
                                 dataloader_iters=(1000000,), dataloader_params=((64, opt.batch_size, 512), ),
                                 epochs=opt.num_epochs, lr=opt.lr, steps_til_summary=opt.steps_til_summary,
                                 epochs_til_checkpoint=opt.epochs_til_ckpt,
                                 model_dir=root_path, loss_fn=loss_fn, val_loss_fn=val_loss_fn,
                                 iters_til_checkpoint=opt.iters_til_ckpt, val_summary_fn=val_summary_fn,
                                 overwrite=True,
                                 optimizer=optimizer,
                                 clip_grad=True,
                                 rank=gpu, train_function=training.train, gpus=opt.gpus, n_view=opt.views)

if __name__ == "__main__":
    opt = p.parse_args()
    if opt.gpus > 1:
        mp.spawn(multigpu_train, nprocs=opt.gpus, args=(opt,))
    else:
        multigpu_train(0, opt)
