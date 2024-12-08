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

You can train your 3DGS model using this repo, or use the official [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) repo:

```
python train.py -s /path/to/dataset
```

Train and evaluate the GaussianMarker on the 3DGS models:

```
# Train on Blender
python train_gaussianmarker.py -s /path/to/nerf_synthetic/ship -m path/to/3dgs_model --results_name ship

# Train on LLFF
python train_gaussianmarker.py -s /path/to/nerf_llff_colmap/trex -m path/to/3dgs_model --results_name trex

# Train on MipNeRF360
python train_gaussianmarker.py -s /path/to/mip360/bicycle -m path/to/3dgs_model --results_name bicycle -r 4
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
