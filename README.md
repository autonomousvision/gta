# DiT + GTA

[Diffusion transformers](https://arxiv.org/abs/2212.09748) with [Geometric Transform Attention](https://openreview.net/forum?id=uJVHygNeSZ).
This codebase is built on [fast-DiT](https://github.com/chuanyangjin/fast-DiT) and [DiT](https://github.com/facebookresearch/DiT). We thank their open-source contributions.

## Setup

First, download and set up the repo:

```bash
git clone git@github.com:autonomousvision/gta.git
cd gta
git checkout DiT
```

We provide an [`environment.yml`](environment.yml) file that can be used to create a Conda environment. If you only want 
to run pre-trained models locally on CPU, you can remove the `cudatoolkit` and `pytorch-cuda` requirements from the file.

```bash
conda env create -f environment.yml
conda activate DiT
```

**Pre-trained DiT + GTA checkpoints.**
- [DiT+GTA at 2.5M-th iteration](https://drive.google.com/drive/folders/1-FoJHUQqn1ZXeA3g4Bko2iWnDu3k7OpC)
- [DiT+RoPE at 2.5M-th iteration](https://drive.google.com/drive/folders/1W7tStpQVssNz5kNAmCKi7PgbuywbyML6) 

## Training
### Preparation Before Training
To extract ImageNet features with `1` GPUs on one node:

```bash
torchrun --nnodes=1 --nproc_per_node=1 extract_features.py --model DiT-B/2 --data-path /path/to/imagenet/train --features-path /path/to/store/features
```

### Training DiT
We provide a training script for DiT in [`train.py`](train.py). This script can be used to train class-conditional 
DiT models, but it can be easily modified to support other types of conditioning. 

To launch DiT-B/2 (256x256) training with `N` GPUs on one node:
```bash
# DiT + GTA
accelerate launch train.py --multi_gpu --num_processes N --model DiT-B/2 --features-path /path/to/store/features --posenc=gta --image-size=256 --results-dir=outputs/GTA --epochs=500 --ckpt-every=250000
# DiT + RoPE
accelerate launch train.py --multi_gpu --num_processes N --model DiT-B/2 --features-path /path/to/store/features --posenc=rope --image-size=256 --results-dir=outputs/RoPE --epochs=500 --ckpt-every=250000 
```

### PyTorch Training Results

We've trained DiT-XL/2 and DiT-B/4 models from scratch with the PyTorch training script
to verify that it reproduces the original JAX results up to several hundred thousand training iterations. Across our experiments, the PyTorch-trained models give 
similar (and sometimes slightly better) results compared to the JAX-trained models up to reasonable random variation. Some data points:

| DiT Model  | Train Steps | FID-50K<br> (PyTorch Training)|
|------------|-------------|----------------------------|
| B/2        | 200K        | 31.93                      |
| B/2+RoPE   | 200K        | 25.71                      |
| B/2+GTA    | 200K        | **25.15** (2.2% relative improvment over RoPE)                 |
| B/2+RoPE   | 2.5M        | 6.26                       | 
| B/2+GTA    | 2.5M        | **5.87**  (6.2% relative improvment over RoPE)                 |

These models were trained at 256x256 resolution; we used 4x A100s to train B/2
here is computed with 250 DDPM sampling steps, with the `ema` VAE decoder and with guidance (`cfg-scale=1.5`). 


## Evaluation (FID, Inception Score, etc.)

We include a [`sample_ddp.py`](sample_ddp.py) script which samples a large number of images from a DiT model in parallel. This script 
generates a folder of samples as well as a `.npz` file which can be directly used with [ADM's TensorFlow
evaluation suite](https://github.com/openai/guided-diffusion/tree/main/evaluations) to compute FID, Inception Score and
other metrics. For example, to sample 50K images from our pre-trained DiT-B/2 model over `N` GPUs, run:

```bash
# DiT + GTA
torchrun --nnodes=1 --nproc_per_node=N sample_ddp.py --model DiT-B/2 --num-fid-samples 50000 --ckpt=/path/to/checkpoint --posenc=gta
# DiT + RoPE
torchrun --nnodes=1 --nproc_per_node=N sample_ddp.py --model DiT-B/2 --num-fid-samples 50000 --ckpt=/path/to/checkpoint --posenc=rope
```

There are several additional options; see [`sample_ddp.py`](sample_ddp.py) for details.
