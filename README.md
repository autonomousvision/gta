<p align="center">

  <h1 align="center">Geometric Transform Attention</h1>
  <p align="center">
    <a href="https://takerum.github.io/">Takeru Miyato</a>
    ·
    <a href="https://kait0.github.io/">Bernhard Jaeger</a>
    ·
    <a href="https://staff.fnwi.uva.nl/m.welling/">Max Welling</a>
    ·
    <a href="https://www.cvlibs.net/">Andreas Geiger</a>

  </p>
  <h3 align="center"><a href="https://openreview.net/forum?id=uJVHygNeSZ">OpenReview</a> | <a href="https://arxiv.org/abs/2310.10375">arXiv</a>  | <a href="https://takerum.github.io/gta/">Project Page</a> </h3>


  <p align="center">
  <img width="180" alt="gta_mech" src="https://github.com/autonomousvision/gta/assets/11573649/939bc44b-3fde-48b8-96ce-4f68c6c55ccd">
  </p>
</p>

Official reproducing code of our ICLR2024 work: "GTA: A Geometry-Aware Attention Mechanism for Multi-view Transformers", a simple way to make your multi-view transformer more expressive! 

(3/15/2024): While we mainly focus on NVS tasks in the paper, the GTA mechanism is also effective for image generation, which is a purely 2D image task. You can find the experimental details in our camera-ready paper and the implementation at this [branch](https://github.com/autonomousvision/gta/tree/DiT).


## Contents

This repository contains the following different codebases, and you can access each of them by switching to the corresponding branch:
- NVS experiments on CLEVR-TR and MSN-Hard (this branch)
- NVS experiments on ACID and RealEstate ([link](https://github.com/autonomousvision/gta/tree/crsrndr))
- ImageNet generation with Diffusion transformers (DiT) ([link](https://github.com/autonomousvision/gta/tree/DiT))


You can find the code of GTA for multi-view ViTs 
[here](https://github.com/autonomousvision/gta/blob/main/source/utils/gta.py) and for image ViTs [here](https://github.com/autonomousvision/gta/blob/main/source/utils/gta.py). 

Please feel free to reach out to us if you have any questions!

# Setup

## 1. Create env and install python libraries

```
conda create -n gta python=3.9
conda activate gta
pip3 install -r requirements.txt
```

## 2. Download dataset

```
export DATADIR=<path_to_datadir>
mkdir -p $DATADIR
```

### CLEVR-TR 

Download the dataset from this [link](https://drive.google.com/file/d/1iT3LjOPm1etcLKs7nVoHhYWU6qR1PdRG/view?usp=drive_link) and place it under `$DATADIR`
<p align="center">
<img width="400" alt="clevr1" src="https://github.com/autonomousvision/gta/assets/11573649/0952c2f7-47e4-41c1-93a4-bad57a8df12e">
<img width="400" alt="clevr2" src="https://github.com/autonomousvision/gta/assets/11573649/1aca1f94-8b7e-42c8-9703-13f2dd60de38">
</p>

### MultiShapeNet Hard (MSN-Hard)
```
gsutil -m cp -r gs://kubric-public/tfds/kubric_frames/multi_shapenet_conditional/2.8.0/ ${DATADIR}/multi_shapenet_frames/
```
<p align="center">
<img width="800" alt="gta_mech" src="https://github.com/autonomousvision/gta/assets/11573649/de09e2d9-1eb1-4833-981b-3ef11c1c5fa3">
</p>

## *Pretrained models (MSN-Hard pre-trained models will be uploaded soon)
- CLEVR-TR: [link](https://drive.google.com/drive/folders/1YPhMpvrVFOgJhCMuyUQqrUx2YC25oLlh)
- MSN-Hard: [link](https://drive.google.com/drive/folders/1HJUfXPnslRvDo2hez2GJ8tFh_O7Lk5iL) 

# Training

### CLEVR-TR
```
torchrun --standalone --nnodes 1 --nproc_per_node 4 train.py runs/clevrtr/GTA/gta/config.yaml  ${DATADIR}/clevrtr --seed=0 
```

### MSN-Hard
```
torchrun --standalone --nnodes 1 --nproc_per_node 4 train.py runs/msn/GTA/gta_so3/config.yaml  ${DATADIR} --seed=0 
```


# Evaluation of PSNR, SSIM and LPIPS
```
python evaluate.py runs/clevrtr/GTA/gta/config.yaml ${DATADIR}/clevrtr $PATH_TO_CHECKPOINT # CLEVR-TR
python evaluate.py runs/msn/GTA/gta_so3/config.yaml ${DATADIR} $PATH_TO_CHECKPOINT # MSN-Hard
```

# Acknowledgements
This repository is built on top of [SRT](https://github.com/stelzner/srt) and [OSRT](https://github.com/stelzner/osrt) created by @stelzner. We would like to thank him for his open-source contribution of the SRT models.
We also thank @lucidrains for providing the values of J matrices, which are needed to compute the irreps of SO(3) efficiently.
 


# Citation
```bibtex
@inproceedings{Miyato2024GTA,
    title={GTA: A Geometry-Aware Attention Mechanism for Multi-View Transformers},
    author={Miyato,Takeru and Jaeger, Bernhard and Welling, Max and Geiger, Andreas},
    booktitle={International Conference on Learning Representations (ICLR)},
    year={2024}
}
```
