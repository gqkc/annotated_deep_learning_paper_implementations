import torch
import torchvision
from torch.utils.data.dataloader import default_collate
from torchvision import datasets


def collate_fn_bn(x):
    batch = default_collate(x)
    mean, std = batch.mean(1, keepdim=True), batch.std(1, keepdim=True) + 1e-5
    bn_batch = (batch - mean) / std
    return bn_batch


def collate_fn_bn2d(initial_bn):
    def collate_fn_bn2d_(x):
        batch = default_collate(x)
        return initial_bn(batch)

    return collate_fn_bn2d_


def collate_fn_vq(vq_vae):
    def collate_fn_vq_(x):
        batch = default_collate(x).to(next(vq_vae.parameters()).device)
        z_e_x, _ = vq_vae.net.encode(batch)
        return z_e_x.detach().permute(0, 3, 1, 2)

    return collate_fn_vq_


def collate_fn_mila(vq_vae):
    def collate_fn_vq_(x):
        batch = default_collate(x).to(next(vq_vae.parameters()).device)
        z_e_x = vq_vae.encoder(batch)
        return z_e_x.detach()

    return collate_fn_vq_


collates = {"default": default_collate, "bn": collate_fn_bn2d}


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


def get_transform_default(**kwargs):
    return torchvision.transforms.Compose([GetSimilarity(kwargs["mult_input"]), Permute()])


def get_transform_mila_3(**kwargs):
    return torchvision.transforms.Compose([
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def get_transform_mila_1(**kwargs):
    return torchvision.transforms.Compose([
        torchvision.transforms.Normalize((0.5), (0.5))
    ])


class GetSimilarity(object):
    def __init__(self, mult_input=1.):
        self.mult_input = mult_input

    def __call__(self, sample):
        return self.mult_input * sample


def get_transform_oh(**kwargs):
    class Oh(object):
        def __call__(self, sample):
            one_hot = torch.nn.functional.one_hot(sample.argmax(-1), sample.size(-1))
            return one_hot * 0.5

    return torchvision.transforms.Compose([GetSimilarity(kwargs["mult_input"]), Oh(), Permute()])


def get_transform_exp(**kwargs):
    class Rescale(object):
        def __call__(self, sample):
            return (sample).detach()

    return torchvision.transforms.Compose([GetSimilarity(kwargs["mult_input"]), Exp(), Rescale(), Permute()])


def get_transform_l2(**kwargs):
    class L2(object):
        def __call__(self, sample):
            return torch.nn.functional.normalize(sample, dim=-1)

    return torchvision.transforms.Compose([GetSimilarity(kwargs["mult_input"]), L2(), Permute()])


def get_transform_mean_max(**kwargs):
    class Mean_Max(object):
        def __call__(self, sample):
            return (sample - sample.mean(-1).unsqueeze(-1)) / sample.abs().max(-1)[0].unsqueeze(-1)

    return torchvision.transforms.Compose([GetSimilarity(kwargs["mult_input"]), Mean_Max(), Permute()])


def get_transform_mean_std(**kwargs):
    class Mean_Std(object):
        def __call__(self, sample):
            return (sample - sample.mean(-1, keepdim=True)) / sample.std(-1, keepdim=True)

    return torchvision.transforms.Compose([GetSimilarity(kwargs["mult_input"]), Mean_Std(), Permute()])


def get_transform_temp_softmax(**kwargs):
    class TempSoftmax(object):
        def __init__(self, temperature=1.):
            self.temperature = temperature

        def __call__(self, sample):
            return torch.nn.functional.softmax(sample / self.temperature, dim=-1)

    return torchvision.transforms.Compose(
        [GetSimilarity(kwargs["mult_input"]), TempSoftmax(kwargs["temperature"]), Permute()])


transforms = {"oh": get_transform_oh, "exp": get_transform_exp, "l2": get_transform_l2,
              "mean_std": get_transform_mean_std,
              "mean_max": get_transform_mean_max, "permute": Permute, "default": get_transform_default,
              "softmax": get_transform_temp_softmax, "mila_1": get_transform_mila_1, "mila_3": get_transform_mila_3}


def str2bool(v):
    import argparse
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_datasets(dataset, subset=None, transform=None):
    '''
    Build dataloaders for different datasets. The dataloader can be easily iterated on.
    Supports Mnist, FashionMNIST, more to come
    '''
    if transform == None:
        transform = torchvision.transforms.ToTensor()

    if dataset == 'mnist':
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        val_dataset = datasets.MNIST('./data', train=False, transform=transform)
    elif dataset == 'fashionmnist':
        train_dataset = datasets.FashionMNIST('./data', train=True, download=True,
                                              transform=transform)
        val_dataset = datasets.FashionMNIST('./data', train=False, transform=transform)
    elif dataset == 'cifar':
        train_dataset = datasets.CIFAR10('./data', train=True, download=True,
                                         transform=transform)
        val_dataset = datasets.CIFAR10('./data', train=False, transform=transform)
    else:
        raise ValueError(dataset)

    if subset:
        train_dataset = torch.utils.data.Subset(train_dataset, list(range(subset)))
        val_dataset = torch.utils.data.Subset(val_dataset, list(range(subset)))

    return train_dataset, val_dataset
