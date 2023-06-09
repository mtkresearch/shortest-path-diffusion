import math
import numpy as np
import torch as th

from guided_diffusion import gaussian_diffusion as gd


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps, low_noise=0.0001, high_noise=0.02):
    """
    Get a pre-defined beta schedule for the given name.
    
    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.

    NOTE: Adapted from `guided_diffusion.gassuain_diffusion.get_named_beta_schedule` but with upgraded API.
    
    `low_noise` is equivalent to $\beta_1$
    `high_noise` is equivalent to $\beta_T$
    """

    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * low_noise
        beta_end = scale * high_noise
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return gd.betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


class NonUnifGaussianDiffusion(gd.GaussianDiffusion):
    # Gaussian Diffusion with dimension-wise varying schedule.
    # Implementation mostly based on `guided_diffusion.gaussian_diffusion.GaussianDiffusion`

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        self.image_resolution = self.betas.shape[1:] # first dim is diffusion steps

        self.num_timesteps = int(betas.shape[0])

        self.alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = np.concatenate([
            np.ones((1, *self.image_resolution), dtype=np.float64), 
            self.alphas_cumprod[:-1]
        ])
        self.alphas_cumprod_next = np.concatenate([
            self.alphas_cumprod[1:],
            np.zeros((1, *self.image_resolution), dtype=np.float64)
        ])
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps, *self.image_resolution)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.concatenate([
                self.posterior_variance[None, 1, ...],
                self.posterior_variance[1:]
            ])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        )
    
    def q_posterior_mean_variance(self, x_start, x_t, t):
        # As long as the matrices in mean and variance experssions of q(x_{t-1} | x_t, x_0)
        # remain diagonal, the base method will _just work_ without any modification. This is
        # due to the fact that expressions for scalar operations and point-wise vector operations
        # are roughly same.
        return super().q_posterior_mean_variance(x_start, x_t, t)
    
    def p_mean_variance(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        This function is mostly taken from the super's `p_mean_variance()` but modified for
        supporting non-unif case. See the original function for API signature.
        """
        
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model(x, self._scale_timesteps(t), **model_kwargs)

        if self.model_var_type in [gd.ModelVarType.LEARNED, gd.ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            if self.model_var_type == gd.ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            else:
                min_log = gd._extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = gd._extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                gd.ModelVarType.FIXED_LARGE: (
                    np.concatenate([
                        self.posterior_variance[None, 1, ...], self.betas[1:, ...]
                    ], axis=0),
                    np.log(np.concatenate([
                        self.posterior_variance[None, 1, ...], self.betas[1:, ...]
                    ], axis=0)),
                ),
                gd.ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = gd._extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = gd._extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == gd.ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [gd.ModelMeanType.START_X, gd.ModelMeanType.EPSILON]:
            if self.model_mean_type == gd.ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
            "eps": model_output if self.model_mean_type == gd.ModelMeanType.EPSILON else None,
            "t": t
        }