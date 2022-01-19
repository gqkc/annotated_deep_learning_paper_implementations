import torch
import torch.nn.functional as F
import torch.utils.data
from mixturevqvae.utils.collate import collate_fn_mila
from mixturevqvae.utils.collate import collate_fn_vq_encode_max
from pytorch_vqvae.modules import VectorQuantizedVAE
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from vqvae.model import VQVAE


class MaxLogitsVQ:
    def __init__(self, **kwargs):
        self.device = kwargs["device"]
        self.vqvae_model = self.vq_load(**kwargs)
        self.collate = self.get_collate()

    def vq_load(self, **kwargs):
        """
        Load the VQVAE model from checkpoint state dict
        Parameters
        ----------
        kwargs

        Returns
        -------
        the vqvae model
        """
        assert kwargs["k"] == 256, f"Can't load VQVAE model with K={kwargs['k']} \
        codebooks, only K=256 has been trained."
        channel_sizes = [16, 32, 32, kwargs["latent_dim"]]
        strides = [2, 2, 1, 1]
        vqvae_model = VQVAE(
            in_channel=3,
            channel_sizes=channel_sizes,
            n_codebook=kwargs["k"],
            dim_codebook=kwargs["latent_dim"],
            strides=strides,
        ).to(self.device)
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
        return collate_fn_vq_encode_max(self.vqvae_model)

    def get_sizes(self, dataset):
        train = next(iter(DataLoader(dataset, batch_size=2)))[0]
        if not self.already_latents:
            sizes = self.vqvae_model.encode(train.to(self.device)).size()
        else:
            sizes = train.sizes()
        return sizes[-1], sizes[1]

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
        return diffused_vector.argmax(1)

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
        with torch.no_grad():
            return self.vqvae_model.decode(quantized_vector)


class MaxZeVQ(MaxLogitsVQ):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_collate(self, **kwargs):
        def collate_fn_vq_(x):
            batch = default_collate(x).to(next(self.vqvae_model.parameters()).device)
            z_e_x = self.vqvae_model.encode(batch)
            return z_e_x.detach()

        return collate_fn_vq_

    def quantize_diffused(self, inputs):
        return self.vqvae_model.quantize(inputs)

    def get_sizes(self, dataset):
        train = next(iter(DataLoader(dataset, batch_size=2)))[0]
        sizes = self.vqvae_model.encode(train.to(self.device)).size()
        return sizes[-1], sizes[1]


class MaxZqProjVQ(MaxLogitsVQ):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def quantize_diffused(self, diffused_vector):
        """
        Quantize the diffused vector / encoding with the normalized codebook embeddings

        Parameters
        ----------
        diffused_vector: encodings with shape (B, C, W, H)

        Returns
        -------
        quantized vector
        """
        normalized_diffused_vector = F.normalize(diffused_vector, dim=-1).permute(0, 2, 3, 1).unsqueeze(-2)
        distances = (normalized_diffused_vector - F.normalize(self.vqvae_model.emb.weight, dim=-1)).square().mean(-1)
        quantized = torch.argmin(distances, dim=-1)
        encodings = self.vqvae_model.emb.codebook_lookup(quantized)
        return encodings.permute(0, 3, 1, 2)

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

        def project_spherical(x: torch.Tensor) -> torch.Tensor:
            """Compute E(X), the spherical projection of an embedding vector.

            Compute distances with the codebooks, select closest codebook,
            normalize the result.

            Notes
            -----
            Uses the `normalize` pytorch function, see docs
            https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html .

            Parameters
            ----------
            x: encoding tensor with shape `(B, C, W, H)`.

            Returns
            -------
            Projected tensor with shape `(B, C, W, H)`
            """
            batch = default_collate(x).to(next(self.vqvae_model.parameters()).device)
            encodings = self.vqvae_model.encode(batch)
            lookup = self.vqvae_model.emb.compute_distances(encodings.permute(0, 2, 3, 1))
            quantized = lookup.argmin(-1)
            return F.normalize(self.vqvae_model.emb(quantized), dim=-1).permute(0, 3, 1, 2)

        return project_spherical


class MilaZeVQ(MaxLogitsVQ):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
        if not self.already_latents:
            sizes = self.vqvae_model.encoder(full_train.to(self.device)).size()
            image_size, image_channels = sizes[-1], sizes[1]
        else:
            sizes = full_train[0].size()
            image_size, image_channels = sizes[1], sizes[-1]

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
