import torch
from pytorch_vqvae.modules import VectorQuantizedVAE
from torch.utils.data import TensorDataset

from labml_nn.diffusion.ddpm.experiment_logits import Configs, main


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
    parser.add_argument("--uuid", default=None, help="uuid for the checkpoint")

    global args
    args = parser.parse_args()
    main(config=MilaConfigs(), name_exp="diffusion_logits_mila", **vars(args))
