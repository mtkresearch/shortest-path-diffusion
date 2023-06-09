# Shortest Path Diffusion (SPD)

*Official code for the paper "Image generation with shortest path diffusion" from [MediaTek Research](https://www.mtkresearch.com/en/), accepted at [International Conference on Machine Learning](https://icml.cc) 2023.*

[**Image generation with shortest path diffusion**]() <br />
Ayan Das*, Stathi Fotiadis*, Anil Batra, Farhang Nabiei, FengTing Liao, Sattar Vakili, Da-shan Shiu, Alberto Bernaccia<br /> 
[MediaTek Research, Cambourne UK](https://www.mtkresearch.com/en/)<br /><br />
(* Equal Contributions)<br />

---

Please note that this codebase is built on the publicly available implementation of OpenAI's "[Guided Diffusion](https://github.com/openai/guided-diffusion)". Below we provide instructions for downloading data, training the model and sampling from it.

**NOTE:** Running this code requires at least one GPU available on the system.

---

## Downloading CIFAR10

Please use the `datasets/cifar10.py` script to download the CIFAR10 dataset at a directory using the following command

```
python datasets/cifar10.py </tmp/dir/>
```

Use the argument `--data_dir /tmp/dir/cifar_train/` in all our scripts to use the dataset.

---

## Preparing the env and installing the code

The entire codebase is written as a python package, hence you need to run ..

```
pip install -e .
```

.. in order to install the package. This will also install all necessary dependencies (require internet connection).

---

## Reference batch for FID

First, create a reference batch of 50000 images for FID computation. Run the following script pointing to the data directory

```
python evaluations/create_ref_batch.py </tmp/dir/cifar_train/>
```

.. which creates a file named `cifar10_reference_50000x32x32x3.npz`, which we will be required for going forward.

## Training

We provide the code for training both our SPD model and the original DDPM model. To train our proposed SPD model, run the following

```
python scripts/image_fourier_train.py --config ./configs/cifar10_fourier.yml --data_dir </tmp/dir/cifar_train/> --reference_batch_path ./cifar10_reference_50000x32x32x3.npz --output_dir ./logs --exp_name my_training --debug True --batch_size 24 --num_samples 50000 --diffusion_steps 4000
```

1. Please make sure to use the `--debug True` flag for running in a non-distributed setting, otherwise use [torchrun](https://pytorch.org/docs/stable/elastic/run.html) appropriately.
2. Use appropriate `--batch_size xx` depending on the GPU used.
3. You may use on-the-fly FID computation with `--num_samples xx` but we discourage doing so due to it's time-consuming nature. We recommend a training-only run with `--num_samples 0` followed by separate sampling run.

> **An important argument** for training SPD is the `--diffusion_steps xx` which sets `T`, the total number of diffusion steps. Use this argument with the training as well as the sampling script (explained below) to produce the results in the paper.

## Sampling 

The training process will produce EMA-checkpoints on certain interval (configurable with `--save_interval xx`) inside the `./logs/my_training/rank_x` folder. Choose a checkpoint, e.g. `./logs/my_training/rank_x/<checkpoint>.pt` and run the sampling as explained below

```
python scripts/image_fourier_sample.py --config ./configs/cifar10_fourier.yml --output_dir ./logs --exp_name sampling --debug True --batch_size 128 --model_path ./logs/my_training/rank_0/ema_0.9999_000000.pt --num_samples 50000
```

This will create an `.npz` file containin samples from the model provided as checkpoint with `--model_path ./logs/my_training/rank_0/<checkpoint>.pt`. It will also compute the FID and display it. Please note that you must have internet connection in order to download the inception weight necessary for FID computation.

---

## Training and sampling with DDPM

The original DDPM and DDIM implementation is also provided for the sake of completeness. The training and sampling process is exactly same as explained above with, only the name of the scripts change. Use the following scripts for DDPM training

```
python scripts/image_train.py --config ./configs/cifar10_fourier.yml ... <arguments>
```

and sampling

```
python scripts/image_sample.py --config ./configs/cifar10_fourier.yml ... <arguments>
```

Please note that the every script reads all necessary hyperparameters from various sections of the `./configs/cifar10_fourier.yml` config file.

## Sampling with DDIM

You may also use DDIM sampler in the sampling script in the same manner as [explained here](https://github.com/openai/improved-diffusion#sampling)

```
python scripts/image_sample.py ... <arguments> --use_ddim True --timestep_respacing ddimXXX
```

where `XXX` is the number of desired steps.