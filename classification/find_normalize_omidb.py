import argparse
import os
import sys
import shutil
import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms
from PIL import Image

parser = argparse.ArgumentParser(description='PyTorchLab')
parser.add_argument('-d', '--data-path', metavar='DATA', 
                    help='path to ImageFolder format dataset')
parser.add_argument('--input-size', default=256, type=int, help='images input size')
parser.add_argument('--check', action='store_true')

def png16_loader(path):
    sample = Image.open(path).convert("F")
    return sample

args = parser.parse_args()

t = []
t.append(transforms.Resize((args.input_size,args.input_size),interpolation=3))
t.append(transforms.Grayscale())
t.append(transforms.ToTensor())
transform = transforms.Compose(t)

root = args.data_path

if(args.check):
    print("File handling check")

    rootfolders = [f for f in os.listdir(root) if os.path.isdir(os.path.join(root, f))]

    for ff in rootfolders:
        source_dir = os.path.join(root,ff)
        filepaths = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir,f))]
        print("Folder:" + str(source_dir))

        for fpf in filepaths:
            fpath = os.path.join(source_dir,fpf)
            #print(str(fpath))
            try:
                image = png16_loader(fpath)
            except(Exception):
                print("Broken on:" + str(fpath))

base_dataset = datasets.ImageFolder(root, transform=transform, loader=png16_loader)

loader = torch.utils.data.DataLoader(
    base_dataset,
    batch_size=1,
    num_workers=1,
    shuffle=False
)

psum    = torch.tensor([0.0])
psum_sq = torch.tensor([0.0])
count = torch.tensor([0])

dataset_length = len(base_dataset)
print("Dataset Length:" + str(dataset_length))

#print("Dataset item shape:" + str(base_dataset[0][0].shape))
#print("Dataset item shape:" + str(base_dataset.shape))

for i,data in enumerate(loader):
    #print("Item Shape:"+str(data[0].shape))
    image = data[0]
    psum    += image.sum()
    psum_sq += (image ** 2).sum()
    count += np.prod(image.shape)
    #print("Local min:"+str(image.min())+" max:"+str(image.max())+" mean:"+str(image.mean())+" std:"+str(image.std()))

print("total pixels:" + str(count))

total_mean = psum / count
total_var  = (psum_sq / count) - (total_mean ** 2)
total_std  = torch.sqrt(total_var)
    
print("Dataset Mean:" + str(total_mean))
print("Dataset Std.Dev.:" + str(total_std))