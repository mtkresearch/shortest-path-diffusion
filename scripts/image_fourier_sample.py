"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import yaml

import numpy as np
import torch as th
import torch.distributed as dist

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    add_dict_to_argparser,
    args_to_dict,
)
from nonunif_diffusion.script_util import (
    model_and_fourier_diffusion_defaults,
    create_model_and_fourier_diffusion,
)

from cmdline_util import to_yaml_interface
from evaluations.fid_score import compute_fid

def sample(args):
    # Hack to fix issue for debugging the code in VSCode.
    # As the default VSCode unable to launch the distributed Training.
    # Moreover, the entire Code base is developed with Distributed training.
    if args.debug:
        print("Manual Setting Environment Variables for Debugging!")
        dist_util.set_fixed_environ_vars()

    dist_util.setup_dist()
    out_dir = os.path.join(args.output_dir, args.exp_name, "rank_{}".format(dist.get_rank()))
    logger.configure(dir=out_dir, format_strs=args.log_formats, hps={'common': vars(args)})

    logger.log("creating model and non-uniform diffusion...")
    model, diffusion = create_model_and_fourier_diffusion(
        **args_to_dict(args, model_and_fourier_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")
    all_images = []
    all_labels = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample_ts = sample_fn(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        sample = sample_ts[-1]
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    out_path=None
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")
    print('GPU Memory used:', th.cuda.memory_allocated())
    return out_path

def write_fid_to_disk(args, out_path):
    th.cuda.synchronize()
    th.cuda.empty_cache()
    import time
    time.sleep(60)
    print('GPU Memory after empty_cache:', th.cuda.memory_allocated())
    logger.log("computing fid")
    
    if dist.get_rank() == 0:
        out_dir = os.path.join(args.output_dir, args.exp_name, "rank_{}".format(dist.get_rank()))

        fid = compute_fid(out_path, args.reference_batch_path,
                          batch_size=args.batch_size)
        logger.log(f'FID: {fid}')

        fid_dict = {
            'samples_path': out_path,
            'reference_path': args.reference_batch_path,
            'FID': fid.item()
        }

        with open(os.path.join(out_dir, 'fid.yaml'), 'w') as f:
            yaml.dump(fid_dict, f)

@to_yaml_interface(__file__)
def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=64,
        use_ddim=False,
        model_path="",
        output_dir="",
        log_formats=['stdout', 'csv', 'json', 'tensorboard'],
        exp_name="default",
        debug=False
    )
    defaults.update(model_and_fourier_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":    
    args = create_argparser().parse_args()
    out_path = sample(args)
    write_fid_to_disk(args, out_path)
