import torch
from pytorch_vqvae.modules import VectorQuantizedVAE
from torch.utils.data import TensorDataset

from labml_nn.diffusion.ddpm.experiment_logits import Configs, main, get_parser


class MilaConfigs(Configs):

    def vq_load(self, **kwargs):
        vqvae_model = VectorQuantizedVAE(kwargs["num_channels"], kwargs["latent_dim"], kwargs["k"]).to(self.device)
        vqvae_model.load_state_dict(torch.load(kwargs["vq_path"], map_location=self.device))
        vqvae_model.eval()
        return vqvae_model

    def load_dataset(self, path: str) -> torch.utils.data.Dataset:
        dataset = torch.load(path, map_location="cpu")
        return dataset

    def vq_decode(self, logits: torch.Tensor) -> torch.Tensor:
        latents = self.select_best_codebook_vector(logits)
        reconstructions = self.vqvae_model.decode(latents)
        return reconstructions

    def quantize_logits(self, codes):
        return codes.argmin(1)


if __name__ == '__main__':
    parser = get_parser()
    global args
    args = parser.parse_args()
    main(config=MilaConfigs(), name_exp="diffusion_logits_mila", **vars(args))
