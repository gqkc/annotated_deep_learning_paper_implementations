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
from datetime import datetime
from typing import List

import torch
import torch.utils.data
import torchvision
import wandb
from labml import tracker, experiment, monit
from labml.configs import BaseConfigs, option
from labml_helpers.device import DeviceConfigs
from torch.utils.data import DataLoader

from labml_nn.diffusion.ddpm import DenoiseDiffusion
from labml_nn.diffusion.ddpm.ddpm_kl import DenoiseDiffusionKL
from labml_nn.diffusion.ddpm.unet import UNet
from mixturevqvae.models import VAE
import os


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
    image_channels: int
    # Image size
    image_size: int
    # Number of channels in the initial feature map
    n_channels: int = 128
    # The list of channel numbers at each resolution.
    # The number of channels is `channel_multipliers[i] * n_channels`
    channel_multipliers: List[int]
    # The list of booleans that indicate whether to use attention at each resolution
    is_attention: List[int]

    # Number of time steps $T$
    n_steps: int
    # Batch size
    batch_size: int = 64
    # Number of samples to generate
    n_samples: int = 16
    # Learning rate
    learning_rate: float

    # Number of training epochs
    epochs: int = 1000

    # Dataset
    dataset: torch.utils.data.Dataset
    # Dataloader
    data_loader: torch.utils.data.DataLoader
    vqvae_model: torch.nn.Module
    # Adam optimizer
    optimizer: torch.optim.Adam
    # path to the train dataset
    train_dataset_path: str
    # path where to save the eps model
    eps_model_save_path: str
    # whether to save checkpoints the labml way
    save_checkpoint: bool

    def vq_load(self, **kwargs):
        vqvae_model = VAE.VQVAE_(kwargs["latent_dim"], kwargs["k"], kwargs["gumbel"], beta=1., alpha=1.,
                                 archi=kwargs["archi"], data_type="continuous", ema=kwargs["ema"],
                                 num_channels=kwargs["channels"], compare=kwargs["compare"]).to(self.device)
        vqvae_model.load_state_dict(torch.load(kwargs["vq_path"], map_location=self.device))
        vqvae_model.eval()
        return vqvae_model

    def vq_decode(self, codes):
        return self.vqvae_model.decode(codes.argmax(1))

    def load_dataset(self, path):
        return torch.load(path, map_location="cpu")

    def init(self, **kwargs):
        # to save eps_model
        self.save_checkpoint = kwargs["save_checkpoint"]
        path_to_model_save = os.path.join("output", kwargs["run_name"])
        self.eps_model_save_path = os.path.join(path_to_model_save, "eps_model.pt")
        os.makedirs(path_to_model_save)
        # load the vqvae model from the given path
        self.vqvae_model = self.vq_load(**kwargs)
        self.train_dataset_path = kwargs["train_dataset_path"]
        dataset = self.load_dataset(self.train_dataset_path)
        full_train = next(iter(DataLoader(dataset, batch_size=len(dataset))))[0]
        train_mean = full_train.exp().mean(0).mean(-1).unsqueeze(-1)
        self.image_size, self.image_channels = full_train.size(1), full_train.size(-1)
        self.learning_rate = kwargs["lr"]

        self.n_steps = kwargs["n_steps"]

        transforms = {"oh": get_transform_oh(), "exp_mean": get_transform_exp_mean(train_mean),
                      "exp": get_transform_exp(), "l2": get_transform_l2(), "mean_std": get_transform_mean_std(),
                      "mean_max": get_transform_mean_max(), "permute": Permute()}

        self.dataset = TransformDataset(dataset, transform=transforms[kwargs["transform"]])

        # Create dataloader
        self.data_loader = torch.utils.data.DataLoader(self.dataset, self.batch_size, shuffle=True, pin_memory=True,
                                                       drop_last=True)
        self.channel_multipliers = kwargs["channel_multipliers"]
        self.is_attention = [False] * len(self.channel_multipliers)
        self.is_attention[-1] = True
        # Create $\textcolor{cyan}{\epsilon_\theta}(x_t, t)$ model
        self.eps_model = UNet(
            image_channels=self.image_channels,
            n_channels=self.n_channels,
            ch_mults=self.channel_multipliers,
            is_attn=self.is_attention, bn=kwargs["bn"]
        ).to(self.device)

        if kwargs["load_checkpoint"] is not None:
            self.eps_model.load_state_dict(torch.load(kwargs["load_checkpoint"], map_location=self.device))

        ddpm_class = DenoiseDiffusion if not kwargs["kl"] else DenoiseDiffusionKL
        # Create [DDPM class](index.html)
        self.diffusion = ddpm_class(
            eps_model=self.eps_model,
            n_steps=self.n_steps,
            device=self.device,
            beta_start=kwargs["beta_start"],
            beta_end=kwargs["beta_end"]
        )

        # Create optimizer
        self.optimizer = torch.optim.Adam(self.eps_model.parameters(), lr=self.learning_rate)

        # Image logging
        # tracker.set_image("sample", True)

    def sample(self, x=None):
        """
        ### Sample images
        """
        with torch.no_grad():
            # $x_T \sim p(x_T) = \mathcal{N}(x_T; \mathbf{0}, \mathbf{I})$
            if x is None:
                x = torch.randn([self.n_samples, self.image_channels, self.image_size, self.image_size],
                                device=self.device)

            # Remove noise for $T$ steps
            for t_ in monit.iterate('Sample', self.n_steps):
                # $t$
                t = self.n_steps - t_ - 1
                # Sample from $\textcolor{cyan}{p_\theta}(x_{t-1}|x_t)$
                x = self.diffusion.p_sample(x, x.new_full((self.n_samples,), t, dtype=torch.long))

            # Log samples
            samples = self.vq_decode(x.to(self.device))
            # tracker.save('sample', samples)
            wandb.log({"sample": [wandb.Image(sample) for sample in samples]})
            return x, x.argmax(1), samples

    def reconstruct(self):
        with torch.no_grad():
            number_of_rec = min(self.n_samples, self.data_loader.batch_size)
            originals = next(iter(self.data_loader))[:number_of_rec].to(self.device)
            t = torch.ones((originals.size(0),), device=originals.device, dtype=torch.long) * self.n_steps - 1
            xT = self.diffusion.q_sample(x0=originals, t=t)
            x0_tilde, _, reconstructions = self.sample(x=xT)
            l2 = torch.square(originals - x0_tilde).mean()
            max_values = (x0_tilde.gather(1, originals.argmax(1).unsqueeze(1)) < x0_tilde)
            rank = max_values.sum(dim=1).float().mean()
            wandb.log({"rank": rank,
                       "l2": l2,
                       "reconstructions": [wandb.Image(image) for image in reconstructions],
                       "images": [wandb.Image(image) for image in self.vq_decode(originals)],
                       "xT_mean": xT.mean()})

    def train(self):
        """
        ### Train
        """

        # Iterate through the dataset
        for data in monit.iterate('Train', self.data_loader):
            # Increment global step
            # tracker.add_global_step()
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
            wandb.log({"loss": loss})
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
            # Reconstructions
            self.reconstruct()
            # New line in the console
            # tracker.new_line()
            # Save the model
            if self.save_checkpoint:
                experiment.save_checkpoint()
            torch.save(self.eps_model, self.eps_model_save_path)


class TransformDataset(torch.utils.data.Dataset):
    def __init__(self, dataset=None, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        data, target = self.dataset[index]

        if self.transform is not None:
            data = self.transform(data)
        return data  # , target

    def __len__(self):
        return self.dataset.__len__()


class Exp(object):
    def __call__(self, sample):
        return sample.exp()


class Permute(object):
    def __call__(self, sample):
        return sample.permute(2, 0, 1)


def get_transform_oh():
    class Oh(object):
        def __call__(self, sample):
            one_hot = torch.nn.functional.one_hot(sample.argmax(-1), sample.size(-1))
            return one_hot * 0.5

    return torchvision.transforms.Compose([Oh(), Permute()])


def get_transform_exp_mean(mean):
    class Mean(object):
        def __call__(self, sample):
            return (sample - mean).detach()

    return torchvision.transforms.Compose([Exp(), Mean(), Permute()])


def get_transform_exp():
    class Rescale(object):
        def __call__(self, sample):
            return (sample).detach()

    return torchvision.transforms.Compose([Exp(), Rescale(), Permute()])


def get_transform_l2():
    class L2(object):
        def __call__(self, sample):
            return torch.nn.functional.normalize(sample, dim=-1)

    return torchvision.transforms.Compose([L2(), Permute()])


def get_transform_mean_max():
    class Mean_Max(object):
        def __call__(self, sample):
            return (sample - sample.mean(-1).unsqueeze(-1)) / sample.abs().max(-1)[0].unsqueeze(-1)

    return torchvision.transforms.Compose([Mean_Max(), Permute()])


def get_transform_mean_std():
    class Mean_Std(object):
        def __call__(self, sample):
            return (sample - sample.mean(-1).unsqueeze(-1)) / sample.std(-1).unsqueeze(-1)

    return torchvision.transforms.Compose([Mean_Std(), Permute()])


def main(**kwargs):
    # Create experiment
    experiment.create(name=kwargs["name_exp"])
    run_name = datetime.now().strftime("train-%Y-%m-%d-%H-%M")

    # Create configurations
    configs = kwargs["config"]
    kwargs["run_name"] = run_name
    configs_dict = {}
    if kwargs["uuid"] is not None:
        configs_dict = experiment.load_configs(kwargs["uuid"])
    # Set configurations. You can override the defaults by passing the values in the dictionary.
    experiment.configs(configs, configs_dict)

    # Initialize
    configs.init(**kwargs)

    # Load training experiment
    experiment.load(kwargs["uuid"])

    # Set models for saving and loading
    experiment.add_pytorch_models({'eps_model': configs.eps_model})

    run = wandb.init(
        project=kwargs["name_exp"],
        entity='cmap_vq',
        config=None,
        name=run_name,
    )
    # Start and run the training loop
    with experiment.start():
        configs.run()
    run.finish()


def get_parser():
    import argparse
    parser = argparse.ArgumentParser(description='parser')
    parser.add_argument('--vq_path', type=str)
    parser.add_argument('--train_dataset_path', type=str)
    parser.add_argument('--kl', type=bool, default=False)
    parser.add_argument('--n_steps', type=int, default=1000)
    parser.add_argument('--transform', type=str, default="permute")

    parser.add_argument("--latent_dim", default=32, type=int)
    parser.add_argument("--k", default=64, type=int)
    parser.add_argument("--gumbel", default=True, type=bool)
    parser.add_argument("--archi", default="convMnist",
                        choices=['basic', 'convMnist', 'convCifar', 'ResNetMnist', "ResNet"])
    parser.add_argument("--compare", default="l2", choices=['dot', 'cosine', 'l2'], help="similarity for vq vae")
    parser.add_argument('--ema', default=False, type=bool,
                        help="exponential moving average to update the codebook vectors")
    parser.add_argument("--channels", default=1, help="number of channels")
    parser.add_argument("--uuid", default=None, help="uuid for the checkpoint")
    parser.add_argument("--channel_multipliers", default=[1, 2], nargs='+', type=int, help="channel multipliers")
    parser.add_argument('--save_checkpoint', default=False, type=bool)
    parser.add_argument('--load_checkpoint', type=str, default=None)
    parser.add_argument('--num_channels', type=int, default=1)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--bn', type=bool, default=True)
    parser.add_argument('--beta_start', type=float, default=0.0001)
    parser.add_argument('--beta_end', type=float, default=0.02)

    return parser


if __name__ == '__main__':
    parser = get_parser()

    args = parser.parse_args()
    main(config=Configs(), name_exp="diffusion_logits", **vars(args))
