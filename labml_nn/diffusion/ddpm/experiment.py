"""
---
title: Denoising Diffusion Probabilistic Models (DDPM) training
summary: >
  Training code for
  Denoising Diffusion Probabilistic Model.
---

# [Denoising Diffusion Probabilistic Models (DDPM)](index.html) training

This trains a DDPM based model on CelebA HQ dataset. You can find the download instruction in this
[discussion on fast.ai](https://forums.fast.ai/t/download-celeba-hq-dataset/45873/3).
Save the images inside [`data/celebA` folder](#dataset_path).

The paper had used a exponential moving average of the model with a decay of $0.9999$. We have skipped this for
simplicity.
"""
import os
from datetime import datetime
from typing import List

import torch
import torch.utils.data
import wandb
from labml import experiment, monit
from labml.configs import BaseConfigs, option
from labml_helpers.device import DeviceConfigs
from pytorch_vqvae.modules import VectorQuantizedVAE

from labml_nn.diffusion.ddpm import DenoiseDiffusion
from labml_nn.diffusion.ddpm.unet import UNet
from torch.utils.data.dataloader import default_collate


class Configs(BaseConfigs):
    """
    ## Configurations
    """
    # Device to train the model on.
    # [`DeviceConfigs`](https://docs.labml.ai/api/helpers.html#labml_helpers.device.DeviceConfigs)
    #  picks up an available CUDA device or defaults to CPU.
    device: torch.device = DeviceConfigs()

    # U-Net model for $\textcolor{cyan}{\epsilon_\theta}(x_t, t)$
    eps_model: UNet
    # [DDPM algorithm](index.html)
    diffusion: DenoiseDiffusion

    # Number of channels in the image. $3$ for RGB.
    image_channels: int = 64
    # Image size
    image_size: int = 32
    # Number of channels in the initial feature map
    n_channels: int = 64
    # The list of channel numbers at each resolution.
    # The number of channels is `channel_multipliers[i] * n_channels`
    channel_multipliers: List[int] = [1, 2, 2, 4]
    # The list of booleans that indicate whether to use attention at each resolution
    is_attention: List[int] = [False, False, False, True]

    # Number of time steps $T$
    n_steps: int
    # Batch size
    batch_size: int = 64
    # Number of samples to generate
    n_samples: int = 16
    # Learning rate
    learning_rate: float = 2e-5

    # Number of training epochs
    epochs: int = 1_000

    beta_start: float
    beta_end: float

    vq_path = "data/vqvae_mini.pt"

    # Dataset
    dataset: torch.utils.data.Dataset
    # Dataloader
    data_loader: torch.utils.data.DataLoader

    # Adam optimizer
    optimizer: torch.optim.Adam

    vqvae_model: VectorQuantizedVAE

    # path to save the model
    eps_model_save_path: str
    # path to the data
    data_path: str

    def vq_load(self):
        vqvae_model = VectorQuantizedVAE(3, 64, 64,
                                         pad=1).to(self.device)
        vqvae_model.load_state_dict(torch.load(self.vq_path, map_location=self.device))
        vqvae_model.eval()
        return vqvae_model

    def collate_fn_mila(self):
        def collate_fn_vq_(x):
            batch = default_collate(x).to(next(self.vqvae_model.parameters()).device)
            z_e_x = self.vqvae_model.encoder(batch)
            return z_e_x.detach()

        return collate_fn_vq_

    def init(self, **kwargs):
        self.is_attention = [False] * len(self.channel_multipliers)
        self.is_attention[-1] = True

        self.vqvae_model = self.vq_load()
        path_folder = os.path.join('output', kwargs["run_name"])
        os.makedirs(path_folder)
        self.eps_model_save_path = os.path.join(path_folder, "eps_model.pt")

        # Create $\textcolor{cyan}{\epsilon_\theta}(x_t, t)$ model
        self.eps_model = UNet(
            image_channels=self.image_channels,
            n_channels=self.n_channels,
            ch_mults=self.channel_multipliers,
            is_attn=self.is_attention,
        ).to(self.device)

        # Create [DDPM class](index.html)
        self.diffusion = DenoiseDiffusion(
            eps_model=self.eps_model,
            n_steps=self.n_steps,
            device=self.device,
            beta_start=self.beta_start,
            beta_end=self.beta_end
        )
        # Create dataloader
        self.data_loader = torch.utils.data.DataLoader(self.dataset, self.batch_size, shuffle=True,
                                                       collate_fn=self.collate_fn_mila())
        # Create optimizer
        self.optimizer = torch.optim.Adam(self.eps_model.parameters(), lr=self.learning_rate)

    def quantize_diffused(self, diffused_vector):
        """
        Quantize the diffused vector
        Parameters
        ----------
        diffused_vector: diffused vector

        Returns
        -------
        the quantized vector
        """
        z_q_x_st, z_q_x = self.vqvae_model.codebook.straight_through(diffused_vector)
        return z_q_x_st

    def vq_decode(self, quantized_vector):
        """
        Decode the quantized vector

        Parameters
        ----------
        quantized_vector: the quantized vector ready to be decoded

        Returns
        -------
        the reconstruction
        """

        return self.vqvae_model.decoder(quantized_vector)

    def sample(self, x=None):
        """
        ### Sample images
        """
        with torch.no_grad():
            if x is None:
                # $x_T \sim p(x_T) = \mathcal{N}(x_T; \mathbf{0}, \mathbf{I})$
                x = torch.randn([self.n_samples, self.image_channels, self.image_size, self.image_size],
                                device=self.device)

            # Remove noise for $T$ steps
            for t_ in monit.iterate('Sample', self.n_steps):
                # $t$
                t = self.n_steps - t_ - 1
                # Sample from $\textcolor{cyan}{p_\theta}(x_{t-1}|x_t)$
                x = self.diffusion.p_sample(x, x.new_full((self.n_samples,), t, dtype=torch.long))

            samples = self.vq_decode(self.quantize_diffused(x))
            if x is None:
                wandb.log({"sample": [wandb.Image(sample) for sample in samples]})
            return x, samples

    def reconstruct(self):
        with torch.no_grad():
            # get original logits x0 from image data
            number_of_rec = min(self.n_samples, self.data_loader.batch_size)
            originals = next(iter(self.data_loader))[:number_of_rec].to(self.device)
            # noise the original logits to T
            t = torch.ones((originals.size(0),), device=originals.device, dtype=torch.long) * self.n_steps - 1
            xT = self.diffusion.q_sample(x0=originals, t=t)
            # denoise the xT to get x0_tilde
            x0_tilde, reconstructions = self.sample(x=xT)
            # get the l2 distance between original logits and reconstructions
            l2 = torch.square(originals - x0_tilde).mean()
            # get the rank of the original code in reconstructions
            max_values = (x0_tilde.gather(1, self.vqvae_model.codebook(originals).unsqueeze(1)) < x0_tilde)
            rank = max_values.sum(dim=1).float().mean()
            # plot xT to check if it seems gaussian
            xT_ = xT.cpu().detach().view(xT.size(0), xT.size(1), -1)
            # xT_lines = [xT_[0, :, i] for i in range(xT_.size(-1))]
            xT_lines = [originals[0, :, 0, 0].cpu().detach(), xT_[0, :, 0]]
            xT_lines_plot = wandb.plot.line_series(xs=range(xT.size(1)), ys=xT_lines,
                                                   keys=["x0[0,0,0,:]", "xT[0,0,0,:]"],
                                                   title="logits", xname="Codebook vectors")
            wandb.log({"rank": rank,
                       "l2": l2,
                       "reconstructions": [wandb.Image(image) for image in reconstructions],
                       "images": [wandb.Image(image) for image in self.vq_decode(originals)],
                       "xT_mean": xT.mean(), "xT": xT_lines_plot})

    def train(self):
        """
        ### Train
        """

        # Iterate through the dataset
        for data in self.data_loader:
            # Move data to device
            data = data.to(self.device)

            # Make the gradients zero
            self.optimizer.zero_grad()
            # Calculate loss
            loss = self.diffusion.loss(data)
            # Compute gradients
            loss.backward()
            # Take an optimization step
            self.optimizer.step()
            # Track the loss
            wandb.log({'loss': loss})

    def run(self):
        """
        ### Training loop
        """
        for _ in monit.loop(self.epochs):
            # Train the model
            self.train()
            # Sample some images
            self.sample()
            # reconstruct()
            self.reconstruct()
            # Save the model
            torch.save(self.eps_model, self.eps_model_save_path)


class MiniimagenetDataset(torch.utils.data.Dataset):
    """
    ### Mini imagenet dataset
    # you can download it with !gdown --id 1Ewc4jLHtxgXsD40IVJrBcjVSURQcYKRg  (12Go)
    """

    def __init__(self, image_size: int, data_path: str):
        super().__init__()
        import h5py

        hf = h5py.File(data_path, 'r')
        self.dataset = hf.get('train')

    def __len__(self):
        """
        Size of the dataset
        """
        return len(self.dataset)

    def __getitem__(self, index: int):
        """
        Get an image
        """
        return torch.Tensor(self.dataset[index])


@option(Configs.dataset, 'Miniimagenet')
def miniimagenet(c: Configs):
    """
    Create miniimagenet dataset
    """
    return MiniimagenetDataset(c.image_size, c.data_path)


def get_parser():
    import argparse
    parser = argparse.ArgumentParser(description='parser')
    parser.add_argument('--n_steps', type=int, default=1000)
    parser.add_argument('--beta_start', type=float, default=0.0001)
    parser.add_argument('--beta_end', type=float, default=0.02)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument("--channel_multipliers", default=[1, 2, 2, 4], nargs='+', type=int, help="channel multipliers")
    parser.add_argument('--data_path', type=str)

    return parser


def main():
    # Name of the experiment
    run_name = datetime.now().strftime("train-%Y-%m-%d-%H-%M-%S")
    # Create configurations
    run = wandb.init(
        project="ho_ze_mila_original",
        entity='cmap_vq',
        config=None,
        name=run_name,
    )
    parser = get_parser()

    args = parser.parse_args()

    # Create experiment
    experiment.create(name='diffuse')

    # Create configurations
    configs = Configs()

    # Set configurations. You can override the defaults by passing the values in the dictionary.
    experiment.configs(configs, vars(args))

    # Initialize
    configs.init(run_name=run_name)

    # Set models for saving and loading
    experiment.add_pytorch_models({'eps_model': configs.eps_model})

    # Start and run the training loop
    with experiment.start():
        configs.run()

    run.finish()


#
if __name__ == '__main__':
    main()
