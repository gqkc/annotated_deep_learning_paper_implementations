import torch
from pytorch_vqvae.modules import VectorQuantizedVAE
from torch.utils.data import TensorDataset

from labml_nn.diffusion.ddpm.experiment_logits import main, get_parser
from labml_nn.diffusion.ddpm.mila.experiment_logits_mila import MilaConfigs
from labml_nn.diffusion.ddpm.data_utils import get_datasets, collate_fn_mila
import torchvision
from torch.utils.data import DataLoader


class MilaVQConfigs(MilaConfigs):
    mult_inputs = 1.0

    def load_dataset(self, path: str, **kwargs) -> torch.utils.data.Dataset:
        dataset = get_datasets(kwargs["dataset"])
        return dataset[0]

    def get_collate(self, **kwargs):
        return collate_fn_mila(self.vqvae_model)

    def vq_decode(self, logits: torch.Tensor) -> torch.Tensor:
        z_q_x_st, z_q_x = self.vqvae_model.codebook.straight_through(logits)
        reconstructions = self.vqvae_model.decoder(z_q_x_st)
        return reconstructions

    def get_sizes(self, dataset):
        full_train = next(iter(DataLoader(dataset, batch_size=2)))[0]
        sizes = self.vqvae_model.encoder(full_train.to(self.device)).size()
        return sizes[-1], sizes[1]


if __name__ == '__main__':
    parser = get_parser()
    parser.add_argument('--pad_vqvae', type=int, default=1)
    parser.add_argument('--dataset', type=str)

    global args
    args = parser.parse_args()
    main(config=MilaVQConfigs(), name_exp="diffusion_logits_mila_vq", **vars(args))
