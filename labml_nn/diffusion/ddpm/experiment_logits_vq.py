import torch
from torch.utils.data import TensorDataset

from labml_nn.diffusion.ddpm.experiment_logits import Configs, main, get_parser
from labml_nn.diffusion.ddpm.data_utils import get_datasets, collate_fn_vq
from torch.utils.data import DataLoader


class VQConfigs(Configs):

    def load_dataset(self, path: str, **kwargs) -> torch.utils.data.Dataset:
        dataset = get_datasets(kwargs["dataset"])
        return dataset[0]

    def get_collate(self, **kwargs):
        return collate_fn_vq(self.vqvae_model)

    def get_sizes(self, dataset):
        full_train = next(iter(DataLoader(dataset, batch_size=2)))[0]
        sizes = self.vqvae_model.net.encode(full_train)[0].size()
        return sizes[1], sizes[-1]

    def vq_decode(self, codes):
        z_q_x_st, z_q_x = self.vqvae_model.codebook.straight_through(codes.permute(0, 2, 3, 1).contiguous())
        return self.vqvae_model.net.decode(z_q_x_st)


if __name__ == '__main__':
    parser = get_parser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--vq_class', type=str, default="default")

    global args
    args = parser.parse_args()
    main(config=VQConfigs(), name_exp="diffusion_logits_vq", **vars(args))
