import torch
from torch.utils.data import TensorDataset

from labml_nn.diffusion.ddpm.experiment_logits import Configs, main
from labml_nn.diffusion.ddpm.vqvae import VQVAE


class BShallConfigs(Configs):

    def vq_load(self, **kwargs):
        vqvae_model = VQVAE(channels=256,
                            latent_dim=1,
                            num_embeddings=1024,
                            embedding_dim=32).to(self.device)
        vqvae_model.load_state_dict(torch.load(kwargs["vq_path"], map_location=self.device)["model"])
        vqvae_model.eval()
        return vqvae_model

    def load_dataset(self, path: str) -> torch.utils.data.Dataset:
        data = torch.load(path, map_location="cpu")
        dataset = TensorDataset(data, torch.zeros(data.size(0)))
        return dataset

    def vq_decode(self, logits: torch.Tensor) -> torch.Tensor:
        logits = logits.permute(0, 2, 3, 1).unsqueeze(0).contiguous().to(self.device)
        dist = self.vqvae_model.decode(logits)
        return dist.probs.argmax(-1).float()


#
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='parser')
    parser.add_argument('--vq_path', type=str)
    parser.add_argument('--train_dataset_path', type=str)
    parser.add_argument('--kl', type=bool, default=False)
    parser.add_argument('--n_steps', type=int, default=200)
    parser.add_argument('--transform', type=str, default="l2")
    parser.add_argument("--uuid", default=None, help="uuid for the checkpoint")
    parser.add_argument("--channel_multipliers", default=[1, 2], nargs='+', type=int, help="channel multipliers")
    parser.add_argument("--channels", default=1, help="number of channels")

    global args
    args = parser.parse_args()
    main(config=BShallConfigs(), name_exp="diffusion_logits_bshall", **vars(args))
