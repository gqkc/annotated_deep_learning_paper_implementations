import torch
import torch.utils.data
from pytorch_vqvae.modules import VectorQuantizedVAE
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate


def collate_fn_mila(vq_vae):
    def collate_fn_vq_(x):
        batch = default_collate(x).to(next(vq_vae.parameters()).device)
        z_e_x = vq_vae.encoder(batch)
        return z_e_x.detach()

    return collate_fn_vq_


class MilaZeVQ:
    def __init__(self, **kwargs):
        self.device = kwargs["device"]
        self.vqvae_model = self.vq_load(**kwargs)
        self.collate = self.get_collate()

    def vq_load(self, **kwargs):
        vqvae_model = VectorQuantizedVAE(kwargs["num_channels"], kwargs["latent_dim"], kwargs["k"],
                                         pad=kwargs["pad_vqvae"]).to(self.device)
        if kwargs["vq_path"] is not None:
            vqvae_model.load_state_dict(torch.load(kwargs["vq_path"], map_location=self.device))
        vqvae_model.eval()
        return vqvae_model

    def get_collate(self, **kwargs):
        """
        Get the collate which preprocess the data on batch
        Parameters
        ----------
        kwargs

        Returns
        -------
        The collate function
        """
        return collate_fn_mila(self.vqvae_model)

    def get_sizes(self, dataset):
        full_train = next(iter(DataLoader(dataset, batch_size=2)))
        if full_train.__class__ == list:
            full_train = full_train[0]
        sizes = self.vqvae_model.encoder(full_train.to(self.device)).size()
        image_size, image_channels = sizes[-1], sizes[1]

        return image_size, image_channels

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
