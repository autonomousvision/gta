
# GTA with the CrossAttentionRenderer(Du et al.(CVPR2023))

This codebase is for reproducing results of our ICLR2024 paper on the ACID and RealEstate10k datasets and is built on [the original codebase of Du et al.(CVPR2023)](https://github.com/yilundu/cross_attention_renderer).


## Setup
You can install the packages used in this codebase using the following command
```
conda create -n gta_crsrndr python=3.9
conda activate gta_crsrndr
pip install -r requirements.txt
conda install h5py
```
You will need to download the Realestate10k and ACID dataset.  
The authors of Du et al.(CVPR2023) provide a small subset of the RealEstate10K dataset [here](https://www.dropbox.com/s/qo8b7odsms722kq/cvpr2023_wide_baseline_data.tar.gz?dl=0).

To download the full datasets for Realestate10k and ACID, please follow the README [here](./data_download/README.md). 
Please extract the folder in the root directory of the repo.

## Pretrained models
- ACID: [link](https://drive.google.com/drive/folders/1urQpbR5wd9BREgKY6idOXRMInx9nca9k)
- RealEstate10K: [link](https://drive.google.com/drive/folders/1xmMphAbZeqmtIz-KeNUP3_KrLbOUmo60)


## High-Level structure
The code is organized as follows:
* ./dataset/ contains code loading data
* ./data_download/ contains code to download Realestate10k and ACID datasets
* ./utils/ contains code for different utility functions on the dataset
* ./estimate_pose/ contains code for estimate the pose between two images using superpoint
* models.py contains the underlying code for instantiating the model
* training.py contains the underlying code for training the model
* loss_functions.py contains loss functions for the different experiments.
* summaries.py contains summary functions for training
* ./experiment_scripts/ contains scripts to reproduce experiments in the paper.

## Reproducing experiments with GTA 
The directory `experiment_scripts` contains one script per experiment in the paper.

To monitor progress, the training code writes tensorboard summaries into a "summaries"" subdirectory in the logging_root. The code will run 
infinitely -- you may stop the code after a day or two of training.

### Realestate 
The Realestate experiment with GTA can be reproduced by running the following command. We iterated 300,000 gradient steps.
```
python experiment_scripts/train_realestate10k.py --experiment_name realestate_gta --batch_size 12 --gpus 4 --GTA
```

### ACID 
First train the model with the below command for 30,000 iterations
```
python experiment_scripts/train_acid.py --experiment_name acid_gta --batch_size 12 --gpus 4 --GTA
```

followed by running the command below (adding lpips and depth loss later in training)
```
python experiment_scripts/train_acid.py --experiment_name acid_gta_lpips_depth --batch_size 4 --gpus 4 --GTA --lpips --depth --checkpoint_path logs/acid_gta/checkpoints/model_current.pth
```

## Evaluation 
You can evaluate the results of the trained model using
```
python experiment_scripts/eval_realestate10k.py --experiment_name vis_realestate --batch_size 1 --gpus 1 --GTA --checkpoint_path logs/realestate_gta/checkpoints/model_current.pth
``` 

You can also visualize the results of applying the trained model on videos using the command
```
python experiment_scripts/render_realestate10k_traj.py --experiment_name vis_realestate --batch_size 1 --gpus 1 --GTA --checkpoint_path logs/realestate_gta/checkpoints/model_current.pth
```

and on unposed images using

```
python experiment_scripts/render_unposed_traj.py --experiment_name vis_realestate --batch_size 1 --gpus 1 --checkpoint_path logs/realestate_gta/checkpoints/model_current.pth
```

## Citation

If you find GTA useful for your research/product, please consider citing the following ICLR2024 paper:
```bibtex
@inproceedings{Miyato2024GTA,
          title={GTA: A Geometry-Aware Attention Mechanism for Multi-View Transformers},
          author={Miyato,Takeru and Jaeger, Bernhard and Welling, Max and Geiger, Andreas},
          booktitle={International Conference on Learning Representations (ICLR)},
          year={2024}
}
```

Our GTA on ACID and RealEstate10k is implemented on top of the network proposed in the following work:
```bibtex
@article{du2023widerender,
          title={Learning to Render Novel Views from
            Wide-Baseline Stereo Pairs},
          author={Du, Yilun and Smith, Cameron and Tewari,
                Ayush and Sitzmann, Vincent},
          journal={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
          year={2023}
}
```
