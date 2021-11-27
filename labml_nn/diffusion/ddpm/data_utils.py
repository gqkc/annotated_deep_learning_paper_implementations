import torch
import torchvision
from torch.utils.data.dataloader import default_collate


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


def get_transform_oh():
    class Oh(object):
        def __call__(self, sample):
            one_hot = torch.nn.functional.one_hot(sample.argmax(-1), sample.size(-1))
            return one_hot * 0.5

    return torchvision.transforms.Compose([Oh(), Permute()])


def get_transform_exp_mean(mean):
    class Mean(object):
        def __call__(self, sample):
            return (sample - mean).detach()

    return torchvision.transforms.Compose([Exp(), Mean(), Permute()])


def get_transform_exp():
    class Rescale(object):
        def __call__(self, sample):
            return (sample).detach()

    return torchvision.transforms.Compose([Exp(), Rescale(), Permute()])


def get_transform_l2():
    class L2(object):
        def __call__(self, sample):
            return torch.nn.functional.normalize(sample, dim=-1)

    return torchvision.transforms.Compose([L2(), Permute()])


def get_transform_mean_max():
    class Mean_Max(object):
        def __call__(self, sample):
            return (sample - sample.mean(-1).unsqueeze(-1)) / sample.abs().max(-1)[0].unsqueeze(-1)

    return torchvision.transforms.Compose([Mean_Max(), Permute()])


def get_transform_mean_std():
    class Mean_Std(object):
        def __call__(self, sample):
            return (sample - sample.mean(-1).unsqueeze(-1)) / sample.std(-1).unsqueeze(-1)

    return torchvision.transforms.Compose([Mean_Std(), Permute()])


transforms = {"oh": get_transform_oh(), "exp": get_transform_exp(), "l2": get_transform_l2(),
              "mean_std": get_transform_mean_std(),
              "mean_max": get_transform_mean_max(), "permute": Permute()}


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
