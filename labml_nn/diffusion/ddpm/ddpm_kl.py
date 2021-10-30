from typing import Tuple, Optional

import torch
from labml_nn.diffusion.ddpm import DenoiseDiffusion
import torch.utils.data
from torch import nn

from labml_nn.diffusion.ddpm.ddpm_utils import gather, normal_kl, mean_except_batch

import numpy as np


class DenoiseDiffusionKL(DenoiseDiffusion):
    """
    ## Denoise Diffusion with KL
    """

    def __init__(self, eps_model: nn.Module, n_steps: int, device: torch.device):
        """
        * `eps_model` is $\textcolor{cyan}{\epsilon_\theta}(x_t, t)$ model
        * `n_steps` is $t$
        * `device` is the device to place constants on
        """
        super().__init__(eps_model=eps_model, n_steps=n_steps, device=device)
        self.eps_model = eps_model

        # Create $\beta_1, \dots, \beta_T$ linearly increasing variance schedule
        self.beta = torch.linspace(0.0001, 0.02, n_steps).to(device)

        # $\alpha_t = 1 - \beta_t$
        self.alpha = 1. - self.beta
        # $\bar\alpha_t = \prod_{s=1}^t \alpha_s$
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

        self.alpha_bar_prev = torch.tensor(np.append(1., self.alpha_bar[:-1]))

        # $T$
        self.n_steps = n_steps
        # $\sigma^2 = \beta$
        self.sigma2 = self.beta

        self.gaussian_nll = torch.nn.GaussianNLLLoss(reduction="none")

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.beta * (1. - self.alpha_bar_prev) / (1. - self.alpha_bar)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = torch.tensor(
            np.log(np.append(self.posterior_variance[1], self.posterior_variance[1:])))
        self.posterior_mean_coef1 = self.beta * np.sqrt(self.alpha_bar_prev) / (1. - self.alpha_bar)
        self.posterior_mean_coef2 = (1. - self.alpha_bar_prev) * torch.sqrt(self.alpha) / (1. - self.alpha_bar)

    def q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        #### Get $q(x_t|x_0)$ distribution

        \begin{align}
        q(x_t|x_0) &= \mathcal{N} \Big(x_t; \sqrt{\bar\alpha_t} x_0, (1-\bar\alpha_t) \mathbf{I} \Big)
        \end{align}
        """

        # [gather](utils.html) $\alpha_t$ and compute $\sqrt{\bar\alpha_t} x_0$
        mean = gather(self.alpha_bar, t) ** 0.5 * x0
        # $(1-\bar\alpha_t) \mathbf{I}$
        var = 1 - gather(self.alpha_bar, t)
        #
        return mean, var

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None):
        """
        #### Sample from $q(x_t|x_0)$

        \begin{align}
        q(x_t|x_0) &= \mathcal{N} \Big(x_t; \sqrt{\bar\alpha_t} x_0, (1-\bar\alpha_t) \mathbf{I} \Big)
        \end{align}
        """

        # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
        if eps is None:
            eps = torch.randn_like(x0)

        # get $q(x_t|x_0)$
        mean, var = self.q_xt_x0(x0, t)
        # Sample from $q(x_t|x_0)$
        return mean + (var ** 0.5) * eps

    def p_sample(self, xt: torch.Tensor, t: torch.Tensor):
        """
        #### Sample from $\textcolor{cyan}{p_\theta}(x_{t-1}|x_t)$

        \begin{align}
        \textcolor{cyan}{p_\theta}(x_{t-1} | x_t) &= \mathcal{N}\big(x_{t-1};
        \textcolor{cyan}{\mu_\theta}(x_t, t), \sigma_t^2 \mathbf{I} \big) \\
        \textcolor{cyan}{\mu_\theta}(x_t, t)
          &= \frac{1}{\sqrt{\alpha_t}} \Big(x_t -
            \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\textcolor{cyan}{\epsilon_\theta}(x_t, t) \Big)
        \end{align}
        """

        # $\textcolor{cyan}{\epsilon_\theta}(x_t, t)$
        eps_theta = self.eps_model(xt, t)
        # [gather](utils.html) $\bar\alpha_t$
        alpha_bar = gather(self.alpha_bar, t)
        # $\alpha_t$
        alpha = gather(self.alpha, t)
        # $\frac{\beta}{\sqrt{1-\bar\alpha_t}}$
        eps_coef = (1 - alpha) / (1 - alpha_bar) ** .5
        # $$\frac{1}{\sqrt{\alpha_t}} \Big(x_t -
        #      \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\textcolor{cyan}{\epsilon_\theta}(x_t, t) \Big)$$
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        # $\sigma^2$
        var = gather(self.sigma2, t)

        # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
        eps = torch.randn(xt.shape, device=xt.device)
        # Sample
        return mean + (var ** .5) * eps

    def posterior(self, u_0, t, u_t):
        """
        Get posterior distribution
        Parameters
        ----------
        u_0 : logits at time 0
        t : timesteps
        u_t : logits at time t

        Returns
        -------
        gaussian distribution
        """
        posterior_mean = (
                gather(self.posterior_mean_coef1, t) * u_0 +
                gather(self.posterior_mean_coef2, t) * u_t
        )
        posterior_variance = gather(self.posterior_variance, t)
        posterior_log_variance_clipped = gather(self.posterior_log_variance_clipped, t)
        assert (posterior_mean.shape[0] == posterior_variance.shape[0] == posterior_log_variance_clipped.shape[0] ==
                u_0.shape[0])
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def loss(self, x0: torch.Tensor, noise: Optional[torch.Tensor] = None):
        """
        #### KL Loss
        """
        # Get batch size
        batch_size = x0.shape[0]
        # Get random $t$ for each sample in the batch
        t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long)

        # Sample $x_t$ for $q(x_t|x_0)$
        xt = self.q_sample(x0, t, eps=noise)

        # Compute the Lt-1 term
        true_prob_mean, log_true_gauss_prob_var, log_true_gauss_prob_var_clipped = self.posterior(
            u_0=x0, u_t=xt, t=t)

        log_x_recon = self.eps_model(xt, t)
        model_prob_mean, log_model_gauss_prob_var, log_model_gauss_prob_var_clipped = self.posterior(
            u_0=log_x_recon, u_t=xt, t=t)

        kl = normal_kl(true_prob_mean, log_true_gauss_prob_var_clipped, model_prob_mean,
                       log_model_gauss_prob_var_clipped)

        kl_mean = mean_except_batch(kl) / torch.log(torch.tensor(2.))

        # handle the case t=1 for the L0 term
        model_var = torch.exp(log_model_gauss_prob_var_clipped) * torch.ones(x0.shape)
        decoder_nll = self.gaussian_nll(input=model_prob_mean, var=model_var,
                                        target=x0)
        decoder_nll_mean = mean_except_batch(decoder_nll) / torch.log(torch.tensor(2.))
        loss = torch.where(t == 0, decoder_nll_mean, kl_mean)

        # KL loss
        return loss.sum()
