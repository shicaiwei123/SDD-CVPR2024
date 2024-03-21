"""
get data loaders
"""
from __future__ import print_function

import os
import socket
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from PIL import Image
import torch



def get_data_folder():
    """
    return server-dependent path to store the data
    """
    hostname = socket.gethostname()
    if hostname.startswith('visiongpu'):
        data_folder = '/data/vision/phillipi/rep-learn/datasets/imagenet'
    elif hostname.startswith('yonglong-home'):
        data_folder = '/home/yonglong/Data/data/imagenet'
    else:
        data_folder = './data/imagenet'

    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)

    return data_folder


class ImageFolderInstance(datasets.ImageFolder):
    """: Folder datasets which returns the index of the image as well::
    """

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)

        # transform
        img0 = self.transform(img)

        img1 = img.transpose(Image.ROTATE_90)
        img1 = self.transform(img1)

        img2 = img.transpose(Image.ROTATE_180)
        img2 = self.transform(img2)

        img3 = img.transpose(Image.ROTATE_270)
        img3 = self.transform(img3)

        img = torch.stack([img0, img1, img2, img3])

        return img, target


def get_test_loader(dataset='imagenet', batch_size=128, num_workers=8):
    """get the test data loader"""

    if dataset == 'imagenet':
        data_folder = get_data_folder()
    else:
        raise NotImplementedError('dataset not supported: {}'.format(dataset))

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    test_folder = os.path.join(data_folder, 'val')
    test_set = datasets.ImageFolder(test_folder, transform=test_transform)
    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=True)

    return test_loader


def get_imagenet_dataloader(dataset='imagenet', batch_size=128, num_workers=16, is_ss=False):
    """
    Data Loader for imagenet
    """
    if dataset == 'imagenet':
        data_folder = get_data_folder()
    else:
        raise NotImplementedError('dataset not supported: {}'.format(dataset))

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    train_folder = os.path.join(data_folder, 'train')
    test_folder = os.path.join(data_folder, 'val')

    if is_ss:
        train_set = ImageFolderInstance(train_folder, transform=train_transform)

    else:
        train_set = datasets.ImageFolder(train_folder, transform=train_transform)

    test_set = datasets.ImageFolder(test_folder, transform=test_transform)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True)

    test_loader = DataLoader(test_set,
                             batch_size=batch_size//2,
                             shuffle=False,
                             num_workers=num_workers // 2,
                             pin_memory=True)

    return train_loader, test_loader

def get_imagenet(dataset='imagenet', batch_size=128, num_workers=16, is_ss=False):
    """
    Data Loader for imagenet
    """
    if dataset == 'imagenet':
        data_folder = get_data_folder()
    else:
        raise NotImplementedError('dataset not supported: {}'.format(dataset))

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    train_folder = os.path.join(data_folder, 'train')
    test_folder = os.path.join(data_folder, 'val')

    if is_ss:
        train_set = ImageFolderInstance(train_folder, transform=train_transform)

    else:
        train_set = datasets.ImageFolder(train_folder, transform=train_transform)

    test_set = datasets.ImageFolder(test_folder, transform=test_transform)



    return train_set,test_set
