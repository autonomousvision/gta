"""
checkpoint (*WITH SOFTRAS SPLIT*) under 
/om2/user/sitzmann/logs/light_fields/NMR_hyper_1e2_reg_layernorm/64_256_None/checkpoints/model_epoch_0087_iter_250000.pth
"""

# Enable import from parent package
import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
import random

import torch
import numpy as np
# torch.multiprocessing.set_sharing_strategy('file_system')
# torch.multiprocessing.set_start_method('spawn', force=True)
import torch.distributed as dist
import torch.multiprocessing as mp
import dataset.acid_dataio
from dataset.acid_dataio import ACID
# from multiprocessing import Manager
import torch
import models
import training
import configargparse
from torch.utils.data import DataLoader
import loss_functions
import summaries
import config

# torch.manual_seed(2)

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
p.add_argument('--batch_size', type=int, default=24)
p.add_argument('--views', type=int, default=2)
p.add_argument('--max_num_instances', type=int, default=None)
p.add_argument('--num_trgt', type=int, default=1)
p.add_argument('--gpus', type=int, default=1)
p.add_argument('--port', type=int, default=1492)

# General training options
p.add_argument('--lr', type=float, default=5e-5)
p.add_argument('--num_epochs', type=int, default=40001)
p.add_argument('--lpips', action='store_true', default=False)
p.add_argument('--depth', action='store_true', default=False)
p.add_argument('--lpips_weight', type=float, default=0.1)
p.add_argument('--depth_weight', type=float, default=0.1)
p.add_argument('--model', type=str, default='midas_vit')
p.add_argument('--epochs_til_ckpt', type=int, default=10)
p.add_argument('--steps_til_summary', type=int, default=500)
p.add_argument('--iters_til_ckpt', type=int, default=10000)
p.add_argument('--checkpoint_path', default=None)
p.add_argument('--GTA', action='store_true', default=False)
p.add_argument('--upsample', action='store_true', default=False)
p.add_argument('--kv_trnsfm', action='store_true', default=False)
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
        train_dataset = ACID(img_root="data_download/acid/train",
                                     pose_root="poses/acid/train.mat",
                                     num_ctxt_views=opt.views, num_query_views=1, query_sparsity=192,
                                     lpips=opt.lpips, augment=True, dual_view=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  drop_last=True, num_workers=8, pin_memory=False, worker_init_fn=worker_init_fn)

        # if opt.val_root is not None:
        val_dataset = ACID(img_root="data_download/acid/test",
                                      pose_root="poses/acid/test.mat",
                                      num_ctxt_views=opt.views, num_query_views=1, augment=False, dual_view=True)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True, drop_last=True, num_workers=4, pin_memory=False, worker_init_fn=worker_init_fn)
        return train_loader, val_loader

    model = models.CrossAttentionRenderer(model=opt.model, n_view=opt.views, GTA=opt.GTA, kv_trnsfm=opt.kv_trnsfm)
    old_state_dict = model.state_dict()

    optimizer = torch.optim.Adam(lr=opt.lr, params=model.parameters(), betas=(0.99, 0.999))
    # optimizer = torch.optim.Adam(lr=opt.lr, params=model.parameters(), betas=(0.9, 0.999))

    if opt.checkpoint_path is not None:
        print(f"Loading weights from {opt.checkpoint_path}...")
        state_dict = torch.load(opt.checkpoint_path, map_location=torch.device('cpu'))
        state_dict, optimizer_dict = state_dict['model'], state_dict['optimizer']
        #if opt.reconstruct:
        #    state_dict['latent_codes.weight'] = torch.zeros_like(state_dict['latent_codes.weight'])

        # state_dict['encoder.latent'] = old_state_dict['encoder.latent']

        model.load_state_dict(state_dict)
        optimizer.load_state_dict(optimizer_dict)

        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

    model = model.cuda()
    # optimizer = optimizer.cuda()
    if opt.gpus > 1:
        sync_model(model)

    # Define the loss
    summary_fn = summaries.img_summaries
    val_summary_fn = summaries.img_summaries
    root_path = os.path.join(opt.logging_root, opt.experiment_name)
    print("LPIPS loss:", opt.lpips)
    loss_fn = val_loss_fn = loss_functions.LFLoss(
        opt.lpips_weight if opt.lpips else 0,
        opt.depth_weight if opt.depth else 0
        )

    training.training(model=model, dataloader_callback=create_dataloader_callback,
                                 dataloader_iters=(1000000,), dataloader_params=((64, opt.batch_size, 512), ),
                                 epochs=opt.num_epochs, lr=opt.lr, steps_til_summary=opt.steps_til_summary,
                                 epochs_til_checkpoint=opt.epochs_til_ckpt,
                                 model_dir=root_path, loss_fn=loss_fn, val_loss_fn=val_loss_fn,
                                 iters_til_checkpoint=opt.iters_til_ckpt, val_summary_fn=val_summary_fn,
                                 overwrite=True,
                                 optimizer=optimizer,
                                 clip_grad=True,
                                 rank=gpu, train_function=training.train, gpus=opt.gpus, n_view=2)

if __name__ == "__main__":
    # manager = Manager()
    # shared_dict = manager.dict()

    opt = p.parse_args()
    if opt.gpus > 1:
        mp.spawn(multigpu_train, nprocs=opt.gpus, args=(opt,))
    else:
        multigpu_train(0, opt)
