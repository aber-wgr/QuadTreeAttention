# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import json
import torch

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader
from torch.utils.data import random_split

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
from mcloader import ClassificationDataset
from CustomDataSet import CustomDataSet
from PIL import Image

class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder


def png16_loader(path):
    sample = Image.open(path).convert("F")
    return sample

def build_dataset(is_train, args):

    if args.data_set == 'CIFAR':
        args.normalise_to = (IMAGENET_DEFAULT_MEAN,IMAGENET_DEFAULT_STD)
        transform = build_transform(is_train, args)
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'OMIDB':
        root = args.data_path
        args.normalise_to = (0.5,0.25)
        transform = build_transform(is_train, args)
        base_dataset = ImageFolder(root, transform=transform, loader=png16_loader) # custom loader to handle the 16-bit inputs
        train_dataset,test_dataset = random_split(base_dataset, [0.9,0.1], generator=torch.Generator().manual_seed(args.seed))
        dataset = train_dataset if is_train else test_dataset
        print( ("train:" if is_train else "test:") + str(len(dataset)) + " shape:" + str(dataset[0][0].shape) )
        nb_classes = 5
    elif args.data_set == 'ISIC2018':
        root = args.data_path
        args.normalise_to = ((0.6276, 0.6257, 0.6292),(0.1824, 0.1813, 0.1850))
        transform = build_transform(is_train, args)
        base_dataset = datasets.ImageFolder(root, transform=transform)
        train_dataset,test_dataset = random_split(base_dataset, [0.9,0.1], generator=torch.Generator().manual_seed(args.seed))
        dataset = train_dataset if is_train else test_dataset
        print( ("train:" if is_train else "test:") + str(len(dataset)) + " shape:" + str(dataset[0][0].shape) )
        nb_classes = 7
    elif args.data_set == 'ISIC2019':
        root = args.data_path
        args.normalise_to = ((0.6276, 0.6257, 0.6292),(0.1824, 0.1813, 0.1850))
        transform = build_transform(is_train, args)
        base_dataset = datasets.ImageFolder(root, transform=transform)
        train_dataset,test_dataset = random_split(base_dataset, [0.9,0.1], generator=torch.Generator().manual_seed(args.seed))
        dataset = train_dataset if is_train else test_dataset
        print( ("train:" if is_train else "test:") + str(len(dataset)) + " shape:" + str(dataset[0][0].shape) )
        nb_classes = 8
    elif args.data_set == 'IMNET':
        args.normalise_to = (IMAGENET_DEFAULT_MEAN,IMAGENET_DEFAULT_STD)
        transform = build_transform(is_train, args)
        if not args.use_mcloader:
            root = os.path.join(args.data_path, 'train' if is_train else 'val')
            dataset = datasets.ImageFolder(root, transform=transform)
        else:
            dataset = ClassificationDataset(
                'train' if is_train else 'val',
                pipeline=transform
            )
        nb_classes = 1000
    elif args.data_set == 'INAT':
        args.normalise_to = (IMAGENET_DEFAULT_MEAN,IMAGENET_DEFAULT_STD)
        transform = build_transform(is_train, args)
        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'INAT19':
        args.normalise_to = (IMAGENET_DEFAULT_MEAN,IMAGENET_DEFAULT_STD)
        transform = build_transform(is_train, args)
        dataset = INatDataset(args.data_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    #if is_train:
        # this should always dispatch to transforms_imagenet_train
        #transform = create_transform(
        #    input_size=args.input_size,
        #    is_training=True,
        #    color_jitter=args.color_jitter,
        #    auto_augment=args.aa,
        #    interpolation=args.train_interpolation,
        #    re_prob=args.reprob,
        #    re_mode=args.remode,
        #    re_count=args.recount,
        #)
        
        #if(args.channels == 1):
        #    transform.transforms.append(transforms.Grayscale(num_output_channels=3))
        #print("transform:"+str(transform))
        #return transform

    t = []
    #if resize_im:
    #   size = int((256 / 224) * args.input_size)
    #   t.append(
    #       transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
    #   )

    t.append(transforms.Resize((args.input_size,args.input_size),interpolation=3))
    if(args.channels == 1):
        t.append(transforms.Grayscale(num_output_channels=1))
    
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(args.normalise_to[0], args.normalise_to[1]))
       
    transform = transforms.Compose(t)
    print("transform:"+str(transform))
    return transform
