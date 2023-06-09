import numpy as np
import torch as th

from nonunif_diffusion.nonunif_diffusion import NonUnifGaussianDiffusion
from guided_diffusion.gaussian_diffusion import _extract_into_tensor


def pixel_and_fourier_noise(batch_size, size, channels=3, device=th.device('cuda')):
    p_noise = th.randn(batch_size, channels, size, size, device=device) # pixel space noise
    f_noise = th.fft.fft2(p_noise, dim=(-2, -1), norm='ortho') # fourier space noise
    return p_noise, f_noise


class FourierShortestPathDiffusion(NonUnifGaussianDiffusion):

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps using shortest
        path theory in fourier domain, i.e. sample from q(x_t | x_0) by first going
        into FFT domain -> noising it using implicit schedule -> back to pixels
        """

        batch_size, channel, side, _ = x_start.shape
        pixel_noise, fourier_noise = pixel_and_fourier_noise(batch_size, side, channels=channel)
        assert pixel_noise.shape == x_start.shape

        fft_x_start = th.fft.fft2(x_start, dim=(-2, -1), norm='ortho')

        return pixel_noise, th.fft.ifft2(
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * fft_x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * fourier_noise,
            dim=(-2, -1), norm='ortho'
        ).real # .imag is _very_ small, so no issue
    
    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
    ):
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        batch_size, channels, dim, _ = x.shape
        _, noise_fourier = pixel_and_fourier_noise(batch_size, dim, channels, device=x.device)
        
        k1 = np.sqrt(1. / self.alphas)
        k2 = self.betas / self.sqrt_one_minus_alphas_cumprod

        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        if cond_fn is not None:
            out["mean"] = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )
        
        eps_fourier = th.fft.fft2(out["eps"], dim=(-2, -1), norm='ortho')
        x_fourier = th.fft.fft2(x, dim=(-2, -1), norm='ortho')

        sample_mean_fourier = _extract_into_tensor(k1, t, x.shape) * \
                (x_fourier - _extract_into_tensor(k2, t, x.shape) * eps_fourier)
        sample = th.fft.ifft2(sample_mean_fourier + \
                nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise_fourier, dim=(-2, -1), norm='ortho').real
        
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}
    
    def ddim_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)

        k1 = np.sqrt(1. / self.alphas)
        k2 = self.sqrt_one_minus_alphas_cumprod
        k3 = np.sqrt(1. - self.alphas_cumprod_prev)

        eps_fourier = th.fft.fft2(out["eps"], dim=(-2, -1), norm='ortho')
        x_fourier = th.fft.fft2(x, dim=(-2, -1), norm='ortho')
        
        sample_mean_fourier = (
            _extract_into_tensor(k1, t, x.shape) * (
                x_fourier - _extract_into_tensor(k2, t, x.shape) * eps_fourier
            ) + _extract_into_tensor(k3, t, x.shape) * eps_fourier
        )

        sample = th.fft.ifft2(sample_mean_fourier, dim=(-2, -1), norm='ortho').real

        return {"sample": sample, "pred_xstart": out["pred_xstart"]}