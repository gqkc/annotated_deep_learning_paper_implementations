import torch
from pytorch_vqvae.modules import VectorQuantizedVAE
from torch.utils.data import TensorDataset

from labml_nn.diffusion.ddpm.experiment_logits import Configs, main, get_parser


class MilaConfigs(Configs):
    # override mult_inputs because we have distances not similarities
    mult_inputs = -1.0

    def vq_load(self, **kwargs):
        vqvae_model = VectorQuantizedVAE(kwargs["num_channels"], kwargs["latent_dim"], kwargs["k"],
                                         pad=kwargs["pad_vqvae"]).to(self.device)
        if kwargs["vq_path"] is not None:
            vqvae_model.load_state_dict(torch.load(kwargs["vq_path"], map_location=self.device))
        vqvae_model.eval()
        return vqvae_model

    def load_dataset(self, path: str, **kwargs) -> torch.utils.data.Dataset:
        dataset = torch.load(path, map_location="cpu")
        return dataset

    def vq_decode(self, logits: torch.Tensor) -> torch.Tensor:
        latents = self.quantize_logits(logits)
        reconstructions = self.vqvae_model.decode(latents)
        return reconstructions

    def quantize_logits(self, codes):
        return codes.argmax(1)


if __name__ == '__main__':
    parser = get_parser()
    parser.add_argument('--pad_vqvae', type=int, default=1)

    global args
    args = parser.parse_args()
    main(config=MilaConfigs(), name_exp="diffusion_logits_mila", **vars(args))
