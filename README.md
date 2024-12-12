# GaussianMarker

GaussianMarker: Uncertainty-Aware Copyright Protection of 3D Gaussian Splatting (NeurIPS 2024)

Paper: [arXiv](https://arxiv.org/pdf/2410.23718)

Project Page: [https://kevinhuangxf.github.io/GaussianMarker/](https://kevinhuangxf.github.io/GaussianMarker/)

## Clone this repository

```
git clone https://kevinhuangxf.github.io/GaussianMarker/ --recursive
```

## Installation

```
# create conda environment
conda create -n gaussianmarker python=3.8
conda activate gaussianmarker

# install pytorch
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121

# install submodules
pip install submodules/diff-gaussian-rasterization/
pip install submodules/simple-knn/
pip install -e submodules/FisherRF/diff/ -v

# install requirements
pip install -r requirements.txt
```

## Data preparation

Please download the [Blender](https://github.com/bmild/nerf), [LLFF](https://github.com/Fyusion/LLFF) and [MipNeRF360](https://jonbarron.info/mipnerf360/) datasets in the official websites.

## Training

### Stage 1

You can train your 3DGS model using this repo, or use the official [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting):

```
python train.py -s /path/to/dataset
```

We provide the pretrained HiDDeN model in the [hidden](hidden) folder based on the [Stable Signature](https://github.com/facebookresearch/stable_signature). You can run the pretrained HiDDeN model by:

```
PYTHONPATH=. python hidden/hidden_images.py hidden/imgs/ hidden/results
```

You can check the [README.md](hidden/README.md) file for how to train the HiDDeN model from scratch.

### Stage 2

Train and evaluate the GaussianMarker on the 3DGS models:

```
# Train on Blender
python train_gaussianmarker.py -s /path/to/nerf_synthetic/lego -m path/to/3dgs_model --iterations 2000

# Train on LLFF and save visualization results
python train_gaussianmarker.py -s /path/to/nerf_llff_colmap/trex -m path/to/3dgs_model --iterations 1000 --test_iterations 1000 --save_iterations 1000 --save_vis

# Train on MipNeRF360 with 1/4 resolotion for saving memory and save visualization results
python train_gaussianmarker.py -s /path/to/mip360/bicycle -m path/to/3dgs_model --iterations 2000 --save_vis -r 4
```

Train 3D decoder (optional):

```
python train_gaussianmarker.py -m /path/to/nerf_synthetic/lego --train_3d_decoder --iterations 2000 --test_iterations 2000
```

## Ciatation

```
@article{huang2024gaussianmarker,
  title     = {GaussianMarker: Uncertainty-Aware Copyright Protection of 3D Gaussian Splatting},
  author    = {Xiufeng Huang, Ruiqi Li, Yiu-ming Cheung, Ka Chun Cheung, Simon See, Renjie Wan},
  journal   = {Neural Information Processing Systems (NeurIPS)},
  year      = {2024},
}
```

## Acknowledgements

This project builds heavily on [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting), [FisherRF](https://github.com/JiangWenPL/FisherRF) and [Stable Signature](https://github.com/facebookresearch/stable_signature). We thanks the authors for their excellent works! If you use our code, please also consider citing their papers as well.
