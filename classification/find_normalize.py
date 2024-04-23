import argparse
import os
import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms

parser = argparse.ArgumentParser(description='PyTorchLab')
parser.add_argument('-d', '--data-path', metavar='DATA', 
                    help='path to ImageFolder format dataset')

args = parser.parse_args()

root = args.data_path
transform = transforms.Compose([transforms.ToTensor()])
base_dataset = datasets.ImageFolder(root, transform=transform)

loader = torch.utils.data.DataLoader(
    base_dataset,
    batch_size=1,
    num_workers=1,
    shuffle=False
)

psum    = torch.tensor([0.0,0.0,0.0])
psum_sq = torch.tensor([0.0,0.0,0.0])
count = torch.tensor([0,0,0])

dataset_length = len(base_dataset)
print("Dataset Length:" + str(dataset_length))

print("Dataset shape:" + str(base_dataset[0][0].shape))

for i,data in enumerate(loader):
    for channel in range(3):
        image = data[0][channel]
        psum[channel]    += image.sum()
        psum_sq[channel] += (image ** 2).sum()
        count[channel] += np.prod(image.shape)

print("total pixels:" + str(count[0]))

total_mean = psum / count
total_var  = (psum_sq / count) - (total_mean ** 2)
total_std  = torch.sqrt(total_var)
    
print("Dataset Mean:" + str(total_mean))
print("Dataset Std.Dev.:" + str(total_std))