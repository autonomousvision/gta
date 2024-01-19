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
  <img width="800" alt="gta_mech" src="https://github.com/takerum/gta_private/assets/11573649/b326d2c1-41f9-49da-bcc5-5aee6d9ed7fe">
  </p>
</p>
This repository contains the reproducing code for our ICLR2024 work: "GTA: A Geometry-Aware Attention Mechanism for Multi-view Transformers", a simple way to make your multi-view transformer more expressive!

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

<img width="400" alt="clevr1" src="https://github.com/takerum/gta_private/assets/11573649/687ab4d3-dd67-4f5b-bf14-f1d96e94608d">
<img width="400" alt="clevr2" src="https://github.com/takerum/gta_private/assets/11573649/5fbc336b-739c-4815-8c18-2ac7f2644625">

### MultiShapeNet Hard (MSN-Hard)
```
gsutil -m cp -r gs://kubric-public/tfds/kubric_frames/multi_shapenet_conditional/2.8.0/ ${DATADIR}/multi_shapenet_frames/
```

<img width="800" alt="gta_mech" src="https://github.com/takerum/gta_private/assets/11573649/30c88c52-ae82-41a6-be1a-f526d224a9f0">

# Training

### CLEVR-TR
```
torchrun --standalone --nnodes 1 --nproc_per_node 4 train.py runs/clevrtr/GTA/gta/config.yaml  ${DATADIR}/clevrtr --seed=0 
```

### MSN-Hard
```
torchrun --standalone --nnodes 1 --nproc_per_node 4 train.py runs/msn/GTA/gta/config.yaml  ${DATADIR} --seed=0 
```
## Pretrained models
- CLEVR-TR: [link]()
- MSN-Hard: [link]()

# Evaluation of PSNR, SSIM and LPIPS
```
python evaluate.py runs/clevrtr/GTA/gta/config.yaml ${DATADIR}/clevrtr --checkpoint_path=$PATH_TO_CHECKPOINT
```

# Acknowledgements
This repository is built on top of [SRT](https://github.com/stelzner/srt) and [OSRT](https://github.com/stelzner/osrt) created by @stelzner. We would like to thank him for his clean and well-organized codebase.



# Citation
```bibtex
@inproceedings{Miyato2024GTA,
    title={GTA: A Geometry-Aware Attention Mechanism for Multi-View Transformers},
    author={Miyato,Takeru and Jaeger, Bernhard and Welling, Max and Geiger, Andreas},
    booktitle={International Conference on Learning Representations (ICLR)},
    year={2024}
}
```
