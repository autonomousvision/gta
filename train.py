import datetime
import time
from torch.profiler import profile, record_function, ProfilerActivity
from source.utils.common import init_ddp
from source.checkpoint import Checkpoint
from source.trainer import SRTTrainer
from source.models_nvs import SRT, TransformingSRT
from source.data import nvs as nvsdata
import yaml
import argparse
import os
import torch
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
import numpy as np
import random

#import cv2
#cv2.setNumThreads(0)


class LrScheduler():
    """ Implements a learning rate schedule with warum up and decay """

    def __init__(self, peak_lr=4e-4, peak_it=10000, decay_rate=0.5, decay_it=100000):
        self.peak_lr = peak_lr
        self.peak_it = peak_it
        self.decay_rate = decay_rate
        self.decay_it = decay_it

    def get_cur_lr(self, it):
        if it < self.peak_it:  # Warmup period
            return self.peak_lr * (it / self.peak_it)
        it_since_peak = it - self.peak_it
        return self.peak_lr * (self.decay_rate ** (it_since_peak / self.decay_it))


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(
        description='Train a 3D scene representation model.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('datapath', type=str, help='Path to datadir')
    parser.add_argument('--seed', type=int, help='random seed', default=None)
    parser.add_argument('--outdir', type=str, help='Path to logdir', default=None)
    parser.add_argument('--exit-after', type=int,
                        help='Exit after this many training iterations.')
    parser.add_argument('--test', action='store_true',
                        help='When evaluating, use test instead of validation split.')
    parser.add_argument('--evalnow', action='store_true',
                        help='Run evaluation on startup.')
    parser.add_argument('--visnow', action='store_true',
                        help='Run visualization on startup.')
    parser.add_argument('--wandb', action='store_true',
                        help='Log run to Weights and Biases.')
    parser.add_argument('--max-eval', type=int,
                        help='Limit the number of scenes in the evaluation set.')
    parser.add_argument('--full-scale', action='store_true',
                        help='Evaluate on full images.')
    parser.add_argument('--print-model', action='store_true',
                        help='Print model and parameters on startup.')
    parser.add_argument(
        '--rtpt', type=str, help='Use rtpt to set process name with given initials.')
    parser.add_argument(
        '--speed_test', type=int, default=0, help='if > 0, asynchronous computation in CUDA becomes disabled, and time to perform one step gradient step is profiled. Also, batchsize is divided by this value')

    args = parser.parse_args()
    with open(args.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.CLoader)

    if args.seed is not None:
        cfg['seed'] = args.seed
        torch.manual_seed(cfg['seed'])
        random.seed(cfg['seed'])
        np.random.seed(cfg['seed'])

    torch.backends.cudnn.benchmark = True    

    rank, world_size = init_ddp()
    device = torch.device(f"cuda:{rank}")

    # replace datapath
    if args.datapath is not None:
        cfg['data']['path'] = args.datapath

    args.wandb = args.wandb and rank == 0  # Only log to wandb in main process

    if args.exit_after is not None:
        max_it = args.exit_after
    elif 'max_it' in cfg['training']:
        max_it = cfg['training']['max_it']
    else:
        max_it = 1000000

    exp_name = os.path.basename(os.path.dirname(args.config))

    if args.rtpt is not None:
        from rtpt import RTPT
        rtpt = RTPT(name_initials=args.rtpt,
                    experiment_name=exp_name, max_iterations=max_it)
    
    out_dir = os.path.dirname(args.config) if args.outdir is None else args.outdir

    if 'seed' in cfg:
        out_dir = os.path.join(out_dir, 'seed'+str(cfg['seed']))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    batch_size = cfg['training']['batch_size'] // world_size
    if args.speed_test:
        batch_size //= args.speed_test
    model_selection_metric = cfg['training']['model_selection_metric']
    if cfg['training']['model_selection_mode'] == 'maximize':
        model_selection_sign = 1
    elif cfg['training']['model_selection_mode'] == 'minimize':
        model_selection_sign = -1
    else:
        raise ValueError(
            'model_selection_mode must be either maximize or minimize.')

    # Initialize datasets
    print('Loading training set...')
    train_dataset = nvsdata.get_dataset('train', cfg['data'])
    eval_split = 'test' if args.test else 'val'
    print(f'Loading {eval_split} set...')
    eval_dataset = nvsdata.get_dataset(eval_split, cfg['data'],
                                    max_len=args.max_eval, full_scale=args.full_scale)

    num_workers = cfg['training']['num_workers'] if 'num_workers' in cfg['training'] else 1
    print(f'Using {num_workers} workers per process for data loading.')

    # Initialize data loaders
    train_sampler = val_sampler = None
    shuffle = False
    if isinstance(train_dataset, torch.utils.data.IterableDataset):
        assert num_workers == 1, "Our MSN dataset is implemented as Tensorflow iterable, and does not currently support multiple PyTorch workers per process. Is also shouldn't need any, since Tensorflow uses multiple workers internally."
        persistent_worker = False
    else:
        persistent_worker = True
        if world_size > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset, shuffle=True, drop_last=False)
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                eval_dataset, shuffle=True, drop_last=False)
        else:
            shuffle = True

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
        sampler=train_sampler, shuffle=shuffle,
        worker_init_fn=nvsdata.worker_init_fn, persistent_workers=persistent_worker)

    val_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=max(1, batch_size//8), num_workers=0,
        sampler=val_sampler, shuffle=shuffle,
        pin_memory=False, worker_init_fn=nvsdata.worker_init_fn, persistent_workers=False)

    # Loaders for visualization scenes
    vis_loader_val = torch.utils.data.DataLoader(
        eval_dataset, batch_size=12, shuffle=shuffle, worker_init_fn=nvsdata.worker_init_fn)
    vis_loader_train = torch.utils.data.DataLoader(
        train_dataset, batch_size=12, shuffle=shuffle, worker_init_fn=nvsdata.worker_init_fn)
    print('Data loaders initialized.')

    # Validation set data for visualization
    data_vis_val = next(iter(vis_loader_val))
    train_dataset.mode = 'val'  # Get validation info from training set just this once
    # Validation set data for visualization
    data_vis_train = next(iter(vis_loader_train))
    train_dataset.mode = 'train'
    print('Visualization data loaded.')

    if cfg['model']['model_type'] == 'srt':
        model = SRT(cfg['model']['args']).to(device)
    elif cfg['model']['model_type'] == 'tsrt':
        print(cfg['model'])
        model = TransformingSRT(cfg['model']['args']).to(device)
    print('Model created.')

    modules_to_be_saved = {}
    if world_size > 1:
        model.encoder = DistributedDataParallel(
            model.encoder, device_ids=[rank], output_device=rank)
        model.decoder = DistributedDataParallel(
            model.decoder, device_ids=[rank], output_device=rank)
        modules_to_be_saved['encoder'] = model.encoder.module
        modules_to_be_saved['decoder'] = model.decoder.module
    else:
        modules_to_be_saved['encoder'] = model.encoder
        modules_to_be_saved['decoder'] = model.decoder
    if hasattr(model, 'module_dict'):
        print('Misc variables exist in model')
        modules_to_be_saved['module_dict'] = model.module_dict

    if 'lr_warmup' in cfg['training']:
        peak_it = cfg['training']['lr_warmup']
    else:
        peak_it = 2500

    decay_it = cfg['training']['decay_it'] if 'decay_it' in cfg['training'] else 4000000

    lr_scheduler = LrScheduler(peak_lr=cfg['training']['lr'] if 'lr' in cfg['training']
                               else 1e-4, peak_it=peak_it, decay_it=decay_it, decay_rate=0.16)

    # Intialize training
    if 'noadamW' in cfg['training'] and cfg['training']['noadamW']:
        print("""
         no adamW no adamW no adamW
        """)
        optimizer = optim.Adam(
            model.parameters(), lr=lr_scheduler.get_cur_lr(0))
    else:
        optimizer = optim.AdamW(
            model.parameters(), lr=lr_scheduler.get_cur_lr(0), weight_decay=0.01)
    trainer = SRTTrainer(model, optimizer, cfg, device,
                         out_dir, train_dataset.render_kwargs)
    checkpoint = Checkpoint(out_dir, device=device,
                            optimizer=optimizer, **modules_to_be_saved)

    # Try to automatically resume
    try:
        if os.path.exists(os.path.join(out_dir, f'model_{max_it}.pt')):
            load_dict = checkpoint.load(f'model_{max_it}.pt')
        else:
            load_dict = checkpoint.load('model.pt')
    except FileNotFoundError:
        load_dict = dict()

    epoch_it = load_dict.get('epoch_it', -1)
    it = load_dict.get('it', -1)
    time_elapsed = load_dict.get('t', 0.)
    run_id = load_dict.get('run_id', None)
    metric_val_best = load_dict.get(
        'loss_val_best', -model_selection_sign * np.inf)

    print(
        f'Current best validation metric ({model_selection_metric}): {metric_val_best:.8f}.')

    if args.wandb:
        import wandb
        wandb.login(key='802f89688040242c7511a529e938e730df4d351c')
        if run_id is None:
            run_id = wandb.util.generate_id()
            print(f'Sampled new wandb run_id {run_id}.')
        else:
            print(f'Resuming wandb with existing run_id {run_id}.')
        log_name = os.path.dirname(args.config)
        if 'seed' in cfg:
            log_name = os.path.join(log_name, 'seed'+str(cfg['seed']))
        wandb.init(project='osrt', name=log_name,
                   id=run_id, resume=True, settings=wandb.Settings(start_method="fork"))
        wandb.config = cfg

    if args.print_model:
        print(model)
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(
                    f'{name:80}{str(list(param.data.shape)):20}{int(param.data.numel()):10d}')

    num_encoder_params = sum(p.numel() for p in model.encoder.parameters())
    num_decoder_params = sum(p.numel() for p in model.decoder.parameters())
    print('Number of parameters:')
    print(f'Encoder: {num_encoder_params}, Decoder: {num_decoder_params}, Total: {num_encoder_params + num_decoder_params}')

    if args.rtpt is not None:
        rtpt.start()

    # Shorthands
    print_every = cfg['training']['print_every']
    checkpoint_every = cfg['training']['checkpoint_every']
    validate_every = cfg['training']['validate_every']
    visualize_every = cfg['training']['visualize_every']
    backup_every = cfg['training']['backup_every']

    # On MSN dataset with num_workers=1, the iterator stops after it returns the final batch without rasing any errors,
    # possibly due to imcompatibility between pytorch and tensorflow's multiprocess system. Thus, the code calculates
    # the length of the iterator beforehand (n_iters_per_epoch) and specifies that the iterator explictly iterates n_iters_per_epoch times.
    n_examples_per_worker = len(train_dataset)//world_size
    print('Total_examples per worker:',
          n_examples_per_worker, 'batch_size:', batch_size)
    n_iters_per_epoch = n_examples_per_worker // batch_size

    # Training loop
    if args.speed_test:
        it_sp = 0
        time_per_iter = []
    while True:
        epoch_it += 1
        if train_sampler is not None:
            train_sampler.set_epoch(epoch_it)

        train_iterator = iter(train_loader)

        for _ in range(n_iters_per_epoch):
            batch = next(train_iterator)
            it += 1

            # Special responsibilities for the main process
            if rank == 0:
                checkpoint_scalars = {'epoch_it': epoch_it,
                                      'it': it,
                                      't': time_elapsed,
                                      'loss_val_best': metric_val_best,
                                      'run_id': run_id}
                # Save checkpoint
                if (checkpoint_every > 0 and (it % checkpoint_every) == 0) and it > 0:
                    checkpoint.save('model.pt', **checkpoint_scalars)
                    print('Checkpoint saved.')

                # Backup if necessary
                if (backup_every > 0 and (it % backup_every) == 0):
                    checkpoint.save('model_%d.pt' % it, **checkpoint_scalars)
                    print('Backup checkpoint saved.')

                # Visualize output
                if args.visnow or (it > 0 and visualize_every > 0 and (it % visualize_every) == 0):
                    print('Visualizing...')
                    trainer.visualize(data_vis_val, mode='val')
                    trainer.visualize(data_vis_train, mode='train')

            # Run evaluation
            if args.evalnow or (it > 0 and validate_every > 0 and (it % validate_every) == 0):
                print('Evaluating...')
                eval_dict = trainer.evaluate(val_loader)
                metric_val = eval_dict[model_selection_metric]
                print(
                    f'Validation metric ({model_selection_metric}): {metric_val:.4f}')

                if args.wandb:
                    wandb.log(eval_dict, step=it)

                if model_selection_sign * (metric_val - metric_val_best) > 0:
                    metric_val_best = metric_val
                    if rank == 0:
                        checkpoint_scalars['loss_val_best'] = metric_val_best
                        print(f'New best model (loss {metric_val_best:.6f})')
                        checkpoint.save('model_best.pt', **checkpoint_scalars)

            # Update learning rate
            new_lr = lr_scheduler.get_cur_lr(it)
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr

            # Run training step
            
            # with profile(activities=
            #        [ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
            #    with record_function("One train step"):
            if args.speed_test:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
            else:
                t0 = time.perf_counter()
            loss, log_dict = trainer.train_step(batch, it)

            if args.speed_test:
                
                end.record()
                torch.cuda.synchronize()
                time_elapsed_per_iter = start.elapsed_time(end)
                time_per_iter.append(time_elapsed_per_iter)
                print(time_elapsed_per_iter)
                it_sp += 1
                if it_sp == 100:
                    np.save(os.path.join(out_dir, 'time.npy'), np.array(time_per_iter))
                    exit(0)
                time_elapsed += time_elapsed_per_iter
            else:
                time_elapsed += time.perf_counter() - t0
            
            # if rank==0:
            #    prof.export_chrome_trace("trace.json")
            time_elapsed_str = str(datetime.timedelta(seconds=time_elapsed))
            log_dict['lr'] = new_lr
            # Print progress
            if print_every > 0 and (it % print_every) == 0:
                log_str = ['{}={:f}'.format(k, v) for k, v in log_dict.items()]
                print(out_dir, 't=%s [Epoch %02d] it=%03d, loss=%.4f'
                      % (time_elapsed_str, epoch_it, it, loss), log_str)
                log_dict['t'] = time_elapsed
                if args.wandb:
                    wandb.log(log_dict, step=it)

            if args.rtpt is not None:
                rtpt.step(subtitle=f'score={metric_val_best:2.2f}')

            args.evalnow = False
            args.visnow = False

            if it >= max_it:
                print('Iteration limit reached. Exiting.')
                if rank == 0:
                    checkpoint.save('model.pt', **checkpoint_scalars)
                exit(0)
