# SSL4HFM

An open-source Self-supervised Learning Package for Hyperspectral Earth Observations 

---

## Introduction

SSL4HFM is an open-source Python/PyTorch toolbox for Pretraining Hyperspectral Foundation Models.

## SSL4HFM Key Features (will be updated)

* Models: MAE: Mased Autoencoders, SimMIM: Simple Masked Image Modeling, SatMAE: MAE for Temporal and Multi-Spectral Satellite Imagery, AIM: Autoregressive Image Modeling
* metrics: PSNR, SAD, SSIM, MSE
* Losses: MSE, SAD, SSIM, Group MSE, L1_L2
* Datasets: hyspecnet, SpectralEarth

## License

HySUPP is distributed under MIT license.

## Citing SSL4HFM

TBD

## Installation

### Using `conda`

We recommend using a `conda` virtual Python environment to install SSL4HFM.

In the following steps we will use `conda` to handle the Python distribution and `pip` to install the required Python packages.
If you do not have `conda`, please install it using [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

```
conda create --name SSL4HFM python=3.10
```

Activate the new `conda` environment to install the Python packages.

```
conda activate SSL4HFM
```

Clone the Github repository.

```
git clone git@github.com:BehnoodRasti/SSL4HFM.git
```

Change directory and install the required Python packages.

```
cd SSL4HFM && pip install -r requirements.txt
```


## Getting started

For Slurm Users

Check the arguments in train_srun and pass the desired ones for your training 

```shell
sbatch train_srun.sh
```

If you dont use slurm

```shell
sbatch train.sh
```
