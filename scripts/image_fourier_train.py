"""
Train a diffusion model on images.
"""

import argparse
import os

import torch.distributed as dist
from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    args_to_dict,
    add_dict_to_argparser,
)
from nonunif_diffusion.script_util import (
    model_and_fourier_diffusion_defaults,
    create_model_and_fourier_diffusion
)
from guided_diffusion.train_util import TrainLoop

from cmdline_util import to_yaml_interface

def main():
    args = create_argparser().parse_args()

    # Hack to fix issue for debugging the code in VSCode.
    # As the default VSCode unable to launch the distributed Training.
    # Moreover, the entire Code base is developed with Distributed training.
    if args.debug:
        print("Manual Setting Environment Variables for Debugging!")
        dist_util.set_fixed_environ_vars()
    
    assert not(args.use_kl), "VLB not supported; use MSE loss"
    assert not(args.learn_sigma), "Learn sigma not supported with fourier; use beta or beta_tilde"
    assert not(args.use_ddim), "DDIM sampler not yet supported"

    dist_util.setup_dist()
    out_dir = os.path.join(args.output_dir, args.exp_name, "rank_{}".format(dist.get_rank()))
    logger.configure(dir=out_dir, format_strs=args.log_formats, hps={'common': vars(args)})

    logger.log("creating model and shortest path fourier diffusion...")
    model, diffusion = create_model_and_fourier_diffusion(
        **args_to_dict(args, model_and_fourier_diffusion_defaults().keys()),
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        resumable=args.resumable,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        num_samples=args.num_samples,
        reference_batch_path=args.reference_batch_path,
        use_ddim=args.use_ddim,
        clip_denoised=args.clip_denoised,
        infer_batch_size=args.infer_batch_size
    ).run_loop()

@to_yaml_interface(__file__)
def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        resumable=True,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        output_dir="",
        log_formats=['stdout', 'csv', 'json', 'tensorboard'],
        exp_name="default",
        debug=False,
        # inference related kwargs
        num_samples=32,
        infer_batch_size=-1,
        use_ddim=False,
        clip_denoised=True,
        reference_batch_path=""
    )
    defaults.update(model_and_fourier_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
