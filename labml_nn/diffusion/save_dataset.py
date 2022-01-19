import h5py
import torch
from torchvision import transforms

from labml_nn.diffusion.dataset import MiniImagenet128
import tqdm
from labml_nn import ROOT_DIR
import os

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='parser')
    parser.add_argument('--data_path', type=str, required=True)
    args = parser.parse_args()

    BATCH_SIZE = 32

    transform = transforms.Compose([transforms.Resize((128, 128)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ])
    dataset = MiniImagenet128(args.data_path, train=True, transform=transform, download=True)
    data_loader = torch.utils.data.DataLoader(dataset, BATCH_SIZE, shuffle=False)
    new_dataset = []
    for data in tqdm.tqdm(data_loader):
        new_dataset.append(data[0])
    new_dataset = torch.cat(new_dataset, dim=0)
    hf = h5py.File(os.path.join(ROOT_DIR, 'mini128.h5'), 'w')
    hf.create_dataset('train_dataset', data=new_dataset)
