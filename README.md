# Continuous Vision Transformer (CViT)


![master_figure-2](figures/cvit_arch.png)

This repository contains code and data accompanying the paper 
titled [CViT: Continuous Vision Transformer for Operator Learning](https://arxiv.org/abs/2405.13998), published in ICLR 2025.

## Abstract

Operator learning, which aims to approximate maps between infinite-dimensional function spaces, is an important area in scientific machine learning with applications across various physical domains. Here we introduce the Continuous Vision Transformer (CViT), a novel neural operator architecture that leverages advances in computer vision to address challenges in learning complex physical systems.  CViT combines a vision transformer encoder, a novel grid-based coordinate embedding, and a query-wise cross-attention mechanism to effectively capture multi-scale dependencies. This design allows for flexible output representations and consistent evaluation at arbitrary resolutions.
We demonstrate CViT's effectiveness across a diverse range of partial differential equation (PDE) systems, including fluid dynamics, climate modeling, and reaction-diffusion processes. Our comprehensive experiments show that CViT achieves state-of-the-art performance on multiple benchmarks, often surpassing larger foundation models, even without extensive pretraining and roll-out fine-tuning. Taken together, CViT exhibits robust handling of discontinuous solutions, multi-scale features, and intricate spatio-temporal dynamics. Our contributions can be viewed as a significant step towards adapting advanced computer vision architectures for building more flexible and accurate machine learning models in the physical sciences.

## Installation

First install the required dependencies by running the following commands:

```
pip3 install -U pip
pip3 install --upgrade jax jaxlib
pip3 install --upgrade -r requirements.txt
```

Then install the `cvit` package by running the following command:

```
git clone https://github.com/PredictiveIntelligenceLab/cvit.git
cd cvit
pip install -e .
```


## Experiments

### Advection 

Further instructions and details on the training and evaluation of the models can be found [here](./adv/README.md).

### Shallow Water 

Further instructions and details on the training and evaluation of the models can be found [here](./swe/README.md).

### Navier-Stokes 

Further instructions and details on the training and evaluation of the models can be found [here](./ns/README.md).

### Diffusion-Reaction

Further instructions and details on the training and evaluation of the models can be found [here](./dr/README.md).




## Citation
    @article{wang2024cvit,
      title={Cvit: Continuous vision transformer for op-erator learning},
      author={Wang, Sifan and Seidman, Jacob H and Sankaran, Shyam and Wang, Hanwen and Paris, George J Pappas},
      journal={arXiv preprint arXiv:2405.13998},
      volume={3},
      year={2024}
    }



