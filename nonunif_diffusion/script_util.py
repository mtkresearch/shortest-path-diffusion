import os
import warnings
import numpy as np
import torch as th
import blobfile as bf
from PIL import Image

import guided_diffusion.gaussian_diffusion as gd
from guided_diffusion.script_util import create_model, model_and_diffusion_defaults
from guided_diffusion.respace import space_timesteps
from nonunif_diffusion.respace import FourierSpacedDiffusion
from nonunif_diffusion.nonunif_diffusion import NonUnifGaussianDiffusion, get_named_beta_schedule
from nonunif_diffusion.fourier_diffusion import FourierShortestPathDiffusion
from guided_diffusion.image_datasets import center_crop_arr

def model_and_nonunif_diffusion_defaults():
    "Defaults for non-uniform diffusion training"
    res = dict(
        # non uniform specific defaults must go here
        low_noise=1e-4, # \beta_min for the linear schedule
        high_noise=2e-2, # \beta_max for the linear schedule
        high_noise_multiplier=10, # relative speed of the minimally salient dimension
    )
    res.update(model_and_diffusion_defaults())
    return res

def model_and_fourier_diffusion_defaults():
    "Defaults for (non-uniform) fourier diffusion training"
    res = dict(
        freq_filter_c1=1.59,
        freq_filter_c2=0.0086,
        freq_filter_m=2.0,
        diffusion_steps=4000,
        learn_sigma=False,
        sigma_small=False,
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        # timestep_respacing="",
        image_size=64
    )
    res.update(model_and_diffusion_defaults())
    del res['noise_schedule'] # this is implicit in fourier case
    del res['timestep_respacing'] # this was also in original diffusion defaults
    return res

def load_saliency_map(path, image_resolution):
    if path is None:
        # default saliency map is just a map with all pixels maximally salient
        return np.ones(image_resolution, dtype=np.float64)

    if not os.path.exists(path):
        raise FileNotFoundError(f'Saliency map file {path} not found')
    
    if not path.endswith('npz'):
        warnings.warn("saliency map file doesn't end with .npz, make sure it's a numpy archive")

    with bf.BlobFile(path, "rb") as f:
        # load saliency map file from npz
        saliencies = np.load(f)['maps']
        saliency = saliencies[0] # just hard-coding the first saliency map
        # TODO: incorporate other saliency map types too
        
        saliency = Image.fromarray(
            saliency.squeeze() # if the channel is 1 (i.e. grayscale), it will be squeezed
        )
    
    # center-crop to image-size & rescale image to [0., 1.]
    # TODO: can we resize instead of center-crop ?
    np_image = center_crop_arr(saliency, image_size=image_resolution[-1]).astype(np.float64) / 255.
    # same saliency map accross channels
    np_image_rgb = np.stack([np_image,] * 3, axis=0)
    assert np_image_rgb.shape == image_resolution, \
        f"saliency map size {np_image_rgb.shape} must be same as {image_resolution}"

    return np_image_rgb


def compute_beta_from_alpha_bar_nonunif(alpha_bar):
    steps, *_ = alpha_bar.shape
    betas = th.zeros_like(alpha_bar)
    for t in range(steps):
        alpha_bar_t = alpha_bar[t]
        alpha_bar_t_1 = alpha_bar[t-1] if t != 0 else 1.
        betas[t] = (1. - alpha_bar_t / alpha_bar_t_1)
    return th.clamp(betas, max=0.999)


def create_model_and_nonunif_diffusion(
    image_size,
    class_cond,
    learn_sigma,
    sigma_small,
    num_channels,
    num_res_blocks,
    channel_mult,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    diffusion_steps,
    noise_schedule,
    low_noise,
    high_noise,
    high_noise_multiplier,
    saliency_map_path,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    use_checkpoint,
    use_scale_shift_norm,
    resblock_updown,
    use_fp16,
    use_new_attention_order,
):
    model = create_model(
        image_size,
        num_channels,
        num_res_blocks,
        channel_mult=channel_mult,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
        use_new_attention_order=use_new_attention_order,
    )

    nonunif_diffusion = create_nonunif_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        sigma_small=sigma_small,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
        # diffusion must be aware of data dimension
        image_resolution=(3, image_size, image_size),
        low_noise=low_noise,
        high_noise=high_noise,
        high_noise_multiplier=high_noise_multiplier,
        saliency_map_path=saliency_map_path,
    )
    return model, nonunif_diffusion


def create_model_and_fourier_diffusion(
    image_size,
    class_cond,
    learn_sigma,
    sigma_small,
    num_channels,
    num_res_blocks,
    channel_mult,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    diffusion_steps,
    freq_filter_c1,
    freq_filter_c2,
    freq_filter_m,
    # timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    use_checkpoint,
    use_scale_shift_norm,
    resblock_updown,
    use_fp16,
    use_new_attention_order,
):
    model = create_model(
        image_size,
        num_channels,
        num_res_blocks,
        channel_mult=channel_mult,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
        use_new_attention_order=use_new_attention_order,
    )
    
    # for default parameters check model_and_fourier_diffusion_defaults
    nonunif_diffusion = create_fourier_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        sigma_small=sigma_small,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        # timestep_respacing=timestep_respacing,
        # diffusion needs to be aware of image size
        image_resolution=(3, image_size, image_size), 
        freq_filter_c1=freq_filter_c1,
        freq_filter_c2=freq_filter_c2,
        freq_filter_m=freq_filter_m,
    )
    return model, nonunif_diffusion


def create_nonunif_gaussian_diffusion(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    low_noise=1e-4,
    high_noise=2e-2,
    high_noise_multiplier=10,
    saliency_map_path=None,
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
    # arguments that are not in `create_gaussian_diffusion(..)`
    image_resolution=(3, 64, 64)
):
    # almost a clone of `create_gaussian_diffusion(..)` but schedule varies spatially

    channels, dimx, dimy = image_resolution
    assert (channels == 3) and (dimx == dimy), "Images must be square and 3-channel (colored)"
    
    saliency_map = load_saliency_map(saliency_map_path, image_resolution)

    assert noise_schedule == "linear", "Only linear schedules supported for spatially varying diffusion"
    
    # The following code decided the "dimension to schedule" mapping
    betas = np.zeros((steps, *image_resolution), dtype=np.float64)
    # saliency map is rescaled from [0, 1] to [1, high_noise_multiplier] before multiplying ..
    # .. with `high_noise`. This determines the realtive speed of the fastest pixel (i.e. saliency 0).
    map_01_to_1_high_noise_multiplier = lambda s: s * (high_noise_multiplier - 1) + 1
    for c, i, j in np.ndindex(image_resolution):
        betas[:, c, i, j] = get_named_beta_schedule(
            noise_schedule,
            steps,
            low_noise=low_noise,
            high_noise=high_noise * map_01_to_1_high_noise_multiplier(1. - saliency_map[c, i, j])
        )

    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
    
    return NonUnifGaussianDiffusion(
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )

def create_fourier_diffusion(
    *,
    steps,
    learn_sigma,
    sigma_small,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    # timestep_respacing,
    image_resolution,
    freq_filter_c1,
    freq_filter_c2,
    freq_filter_m
):
    # almost a clone of `create_gaussian_diffusion(..)` but schedule determined by Sigma0 in freq domain
    # for default parameters check model_and_fourier_diffusion_defaults
    
    def freq_domain_saliency(size: int, channel: int, c1: float, c2: float, m: float):
        ''' This creates the \Sigma0 in frequency domain '''
        fr = th.fft.fftfreq(size, d=1./size)
        spec_channel = c1 / th.pow(th.abs(c2 + th.pow(th.pow(fr.unsqueeze(-1), 2.) + th.pow(fr.unsqueeze(0), 2.), 1. / 2.)), m)
        return spec_channel.unsqueeze(0).repeat(channel, 1, 1)

    channels, dimx, dimy = image_resolution
    assert (channels == 3) and (dimx == dimy), "Images must be square and 3-channel (colored)"
    
    Sigma0 = freq_domain_saliency(dimx, # or 'dimy', same thing
                                        channel=channels, # always 3 (color image)
                                        c1=freq_filter_c1, c2=freq_filter_c2, m=freq_filter_m)
    Sigma0 = Sigma0.unsqueeze(0).repeat(steps, 1, 1, 1)
    
    t = th.linspace(1., steps - 1, steps)[:, None, None, None]
    tau = (steps - t) / steps
    alphas_cumprod_implicit = (1. - Sigma0**tau) / (1. - Sigma0)
    
    # This is used by sampling process. This needs to tweaked for experimentation
    betas_implicit = compute_beta_from_alpha_bar_nonunif(alphas_cumprod_implicit)

    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    
    timestep_respacing = [steps]
    
    return FourierSpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas_implicit,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )