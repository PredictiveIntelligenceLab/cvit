
# 2D Navier-Stokes equation

This directory contains the code to train and evaluate models on the 2D Navier-Stokes equation.


## Download datasets


SSH

```
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone git@hf.co:datasets/pdearena/NavierStokes-2D

# if you want to clone without large files – just their pointers
# prepend your git clone with the following env var:
GIT_LFS_SKIP_SMUDGE=1
```

HTTPS:

```
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/datasets/pdearena/NavierStokes-2D

# if you want to clone without large files – just their pointers
# prepend your git clone with the following env var:
GIT_LFS_SKIP_SMUDGE=1
```


## Training and Evaluation

Make sure you are in the `ns` directory before running the following commands and the path to the data is correctly specified in the config file.

To train the model,
specify the GPU and config and run the following command

```CUDA_VISIBLE_DEVICES=0 python3 main.py --config=configs/cvit_4x4.py```

To evaluate the model, specify the GPU and config and run the following command

```CUDA_VISIBLE_DEVICES=0 python3 eval.py --config=configs/cvit_4x4.py```




## Results


| **Model**                    | **# Params** | **Rel. $L^2$ error ($\downarrow$)** |
|------------------------------|--------------|-------------------------------------|
| FNO                          | 0.5 M        | 9.12%                              |
| FFNO                         | 1.3 M        | 8.39%                              |
| GK-T                         | 1.6 M        | 9.52%                              |
| GNOT                         | 1.8 M        | 17.20%                             |
| Oformer                      | 1.9 M        | 13.50%                             |
| DPOT-Ti                      | 7 M          | 12.50%                             |
| DPOT-S                       | 30 M         | 9.91%                              |
| DPOT-L (Pre-trained)         | 500 M        | 7.98%                              |
| DPOT-L (Fine-tuned)          | 500 M        | 2.78%                              |
| **CViT-S**                   | 13 M         | **3.75%**                          |
| **CViT-B**                   | 30 M         | **3.18%**                          |
| **CViT-L**                   | 92 M         | **2.35%**                          |

## Visualizations

The following figures show the predictions of the CViT-L model on the Navier-Stokes equation.

### Smoke

![u](../figures/ns_u_pred.png)

### Velocity field

![vx](../figures/ns_vx_pred.png)

![vy](../figures/ns_vy_pred.png)
