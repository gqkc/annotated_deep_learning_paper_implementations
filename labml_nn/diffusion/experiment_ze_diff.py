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
import torchvision
import wandb
from PIL import Image
from labml import lab, experiment, monit
from labml.configs import BaseConfigs, option
from labml_helpers.device import DeviceConfigs
from torchvision import transforms

from labml_nn import ROOT_DIR
from labml_nn.diffusion.dataset import MiniimagenetDataset, MiniImagenetMax, MiniImagenet128
from labml_nn.diffusion.ddpm import DenoiseDiffusion
from labml_nn.diffusion.ddpm.unet import UNet
from labml_nn.diffusion.vqvae import MilaZeVQ


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
    image_channels: int = 3
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
    n_steps: int = 1_000
    # Batch size
    batch_size: int = 64
    # Number of samples to generate
    n_samples: int = 16
    # Learning rate
    learning_rate: float = 2e-5

    # Number of training epochs
    epochs: int = 1_000

    # Dataset
    dataset: torch.utils.data.Dataset
    # Dataloader
    data_loader: torch.utils.data.DataLoader

    # Adam optimizer
    optimizer: torch.optim.Adam

    data_path: str

    k: int
    vq_path: str
    pad_vqvae: int
    eps_model_save_path: str
    vq: MilaZeVQ
    load_checkpoint: str

    def init(self):
        run_name = datetime.now().strftime("train-%Y-%m-%d-%H-%M-%S")

        path_to_model_save = os.path.join(ROOT_DIR, "output", "prior", "ze_ho", run_name)
        self.eps_model_save_path = os.path.join(path_to_model_save, "eps_model.pt")
        os.makedirs(path_to_model_save)

        # Create $\textcolor{cyan}{\epsilon_\theta}(x_t, t)$ model
        eps_model = UNet(
            image_channels=self.image_channels,
            n_channels=self.n_channels,
            ch_mults=self.channel_multipliers,
            is_attn=self.is_attention,
        ).to(self.device)
        if self.load_checkpoint is not None:
            eps = torch.load(self.load_checkpoint, map_location=self.device)
            if eps.__class__ == eps_model.__class__:
                eps_model = eps
            else:
                eps_model.load_state_dict(eps_model)
        self.eps_model = eps_model
        # Create [DDPM class](index.html)
        self.diffusion = DenoiseDiffusion(
            eps_model=self.eps_model,
            n_steps=self.n_steps,
            device=self.device,
        )
        self.vq = MilaZeVQ(device=self.device, k=self.k, num_channels=3, latent_dim=self.image_channels,
                           vq_path=self.vq_path,
                           pad_vqvae=self.pad_vqvae)
        # Create dataloader

        self.data_loader = torch.utils.data.DataLoader(self.dataset, self.batch_size, shuffle=True,
                                                       drop_last=True, collate_fn=self.vq.collate)
        # self.data_loader = torch.utils.data.DataLoader(self.dataset, self.batch_size, shuffle=True, pin_memory=True)
        # Create optimizer
        self.optimizer = torch.optim.Adam(self.eps_model.parameters(), lr=self.learning_rate)

        # Image logging
        # tracker.set_image("sample", True)

    def sample(self):
        """
        ### Sample images
        """
        with torch.no_grad():
            # $x_T \sim p(x_T) = \mathcal{N}(x_T; \mathbf{0}, \mathbf{I})$
            x = torch.randn([self.n_samples, self.image_channels, self.image_size, self.image_size],
                            device=self.device)

            log_iteration = int(self.n_steps / 20)
            logits_chain = []
            # Remove noise for $T$ steps
            for t_ in monit.iterate('Sample', self.n_steps):
                # $t$
                t = self.n_steps - t_ - 1
                # Sample from $\textcolor{cyan}{p_\theta}(x_{t-1}|x_t)$
                x = self.diffusion.p_sample(x, x.new_full((self.n_samples,), t, dtype=torch.long))
                if t_ % log_iteration == 0:
                    logits_chain.append(x[0].detach().clone().unsqueeze(0))
            logits_chain_ = torch.cat(logits_chain, dim=0)
            samples_chain = self.vq.vq_decode(self.vq.quantize_diffused(logits_chain_.to(self.device)))
            # Log samples
            samples = self.vq.vq_decode(self.vq.quantize_diffused(x.to(self.device)))
            # just plot original to see the quality of the reconstructions
            number_of_rec = min(self.n_samples, self.data_loader.batch_size)
            originals = next(iter(self.data_loader))[:number_of_rec].to(self.device)
            wandb.log(
                {"sample": [wandb.Image(sample) for sample in samples], "samples_chain": [wandb.Image(sample) for sample
                                                                                          in samples_chain],
                 "originals": [
                     wandb.Image(image) for image in
                     self.vq.vq_decode(self.vq.quantize_diffused(originals))]})

    def train(self):
        """
        ### Train
        """

        # Iterate through the dataset
        for data in self.data_loader:
            # Increment global step
            # Move data to device
            if type(data) == list:
                data = data[0]
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
            wandb.log({"loss": loss.cpu().detach()})
            # tracker.save('loss', loss)

    def run(self):
        """
        ### Training loop
        """
        for _ in monit.loop(self.epochs):
            # Train the model
            self.train()
            # Sample some images
            self.sample()
            # New line in the console
            # tracker.new_line()
            # Save the model
            # experiment.save_checkpoint()
            torch.save(self.eps_model, self.eps_model_save_path)


class CelebADataset(torch.utils.data.Dataset):
    """
    ### CelebA HQ dataset
    """

    def __init__(self, image_size: int):
        super().__init__()

        # CelebA images folder
        folder = lab.get_data_path() / 'celebA'
        # List of files
        self._files = [p for p in folder.glob(f'**/*.jpg')]

        # Transformations to resize the image and convert to tensor
        self._transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(image_size),
            torchvision.transforms.ToTensor(),
        ])

    def __len__(self):
        """
        Size of the dataset
        """
        return len(self._files)

    def __getitem__(self, index: int):
        """
        Get an image
        """
        img = Image.open(self._files[index])
        return self._transform(img)


@option(Configs.dataset, 'CelebA')
def celeb_dataset(c: Configs):
    """
    Create CelebA dataset
    """
    return CelebADataset(c.image_size)


class MNISTDataset(torchvision.datasets.MNIST):
    """
    ### MNIST dataset
    """

    def __init__(self, image_size):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(image_size),
            torchvision.transforms.ToTensor(),
        ])

        super().__init__(str(lab.get_data_path()), train=True, download=True, transform=transform)

    def __getitem__(self, item):
        return super().__getitem__(item)[0]


@option(Configs.dataset, 'MNIST')
def mnist_dataset(c: Configs):
    """
    Create MNIST dataset
    """
    return MNISTDataset(c.image_size)


@option(Configs.dataset, 'mini84')
def mini84_dataset(c: Configs):
    """
    Create mini imagenet dataset with 84 pixels
    """
    return MiniImagenetMax(c.data_path)


@option(Configs.dataset, 'mini128')
def mini128_dataset(c: Configs):
    """
    Create mini imagenet dataset with 128 pixels
    """
    transform = transforms.Compose([transforms.Resize((128, 128)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ])
    dataset = MiniImagenet128(c.data_path, train=True, transform=transform, download=True)
    return dataset


@option(Configs.dataset, 'minih5')
def minih5_dataset(c: Configs):
    """
    Create mini imagenet dataset with 128 pixels saved in h5
    """
    dataset = MiniimagenetDataset(c.image_size, c.data_path)
    return dataset


def get_parser():
    import argparse
    parser = argparse.ArgumentParser(description='parser')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--dataset', type=str, default="mini")
    parser.add_argument('--n_channels', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--vq_path', type=str, required=True)
    parser.add_argument('--k', type=int, required=True)
    parser.add_argument("--channel_multipliers", default=[1, 2, 2, 4], nargs='+', type=int, help="channel multipliers")
    parser.add_argument('--pad_vqvae', type=int, default=1)
    parser.add_argument('--image_channels', type=int, default=32, help="hidden dim of ze")
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--load_checkpoint', type=str, default=None)
    return parser


def main():
    run_name = datetime.now().strftime("train-%Y-%m-%d-%H-%M-%S")

    wandb.init(
        project="ho_original",
        entity='cmap_vq',
        config=None,
        name=run_name,
    )

    # Create experiment
    experiment.create(name='diffuse')

    # Create configurations
    configs = Configs()

    parser = get_parser()
    args = parser.parse_args()

    # params["dataset"] = dataset
    # params["image_size"] = image_size
    # Set configurations. You can override the defaults by passing the values in the dictionary.
    experiment.configs(configs, vars(args)
                       )

    # Initialize
    configs.init()

    # Set models for saving and loading
    experiment.add_pytorch_models({'eps_model': configs.eps_model})

    # Start and run the training loop
    with experiment.start():
        configs.run()


#
if __name__ == '__main__':
    main()
