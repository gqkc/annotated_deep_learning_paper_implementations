from labml_nn.diffusion.ddpm.experiment_logits import Configs
import torch

import wandb
from labml import experiment
from datetime import datetime
from labml_nn.diffusion.ddpm.vqvae import VQVAE
from torch.utils.data import TensorDataset, DataLoader
from pytorch_vqvae.modules import VectorQuantizedVAE


class MilaConfigs(Configs):

    def vq_load(self, **kwargs):
        vqvae_model = VectorQuantizedVAE(kwargs["num_channels"], kwargs["hidden_size"], kwargs["k"]).to(self.device)
        vqvae_model.load_state_dict(torch.load(kwargs["vq_path"], map_location=self.device))
        vqvae_model.eval()
        return vqvae_model

    def load_dataset(self, path: str) -> torch.utils.data.Dataset:
        dataset = torch.load(path, map_location="cpu")
        return dataset

    def vq_decode(self, logits: torch.Tensor) -> torch.Tensor:
        latents = logits.argmin(1)
        reconstructions = self.vqvae_model.decode(latents)
        return reconstructions


def main(**kwargs):
    # Create experiment
    experiment.create(name='diffuse_logits_mila')

    # Create configurations
    configs = MilaConfigs()

    # Set configurations. You can override the defaults by passing the values in the dictionary.
    experiment.configs(configs, {
    })

    # Initialize
    configs.init(**kwargs)

    # Set models for saving and loading
    experiment.add_pytorch_models({'eps_model': configs.eps_model})

    run_name = datetime.now().strftime("train-%Y-%m-%d-%H-%M")

    run = wandb.init(
        project="diffusion_logits_mila",
        entity='cmap_vq',
        config=None,
        name=run_name,
    )
    # Start and run the training loop
    with experiment.start():
        configs.run()
    run.finish()


#
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='parser')
    parser.add_argument('--vq_path', type=str)
    parser.add_argument('--train_dataset_path', type=str)
    parser.add_argument('--kl', type=bool, default=False)
    parser.add_argument('--n_steps', type=int, default=200)
    parser.add_argument('--transform', type=str, default="l2")
    parser.add_argument('--k', type=int, default=512)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--num_channels', type=int, default=3)

    global args
    args = parser.parse_args()
    main(**vars(args))
