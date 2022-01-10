import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from PIL import Image, ImageOps, ImageFilter
import pandas as pd



class Resize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        # padding
        ratio = self.size[0] / self.size[1]
        w, h = img.size
        if w / h < ratio:
            t = int(h * ratio)
            w_padding = (t - w) // 2
            img = img.crop((-w_padding, 0, w+w_padding, h))
        else:
            t = int(w / ratio)
            h_padding = (t - h) // 2
            img = img.crop((0, -h_padding, w, h+h_padding))

        img = img.resize(self.size, self.interpolation)

        return img



class RandomRotate(object):
    def __init__(self, degree, p=0.5):
        self.degree = degree
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            rotate_degree = random.uniform(-1*self.degree, self.degree)
            img = img.rotate(rotate_degree, Image.BILINEAR)
        return img



class RandomGaussianBlur(object):
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, img):
        if random.random() < self.p:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
        return img



class DrowsiessDetectionDataset(Dataset):

    def __init__(self, txt_path, train_mode):
        """
        txt_path: txt路径
        train_mode:  "train", "val", "test" 三种
        """
        assert train_mode in ["train", "val", "test"]

        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))
        self.imgs = imgs

        if train_mode == 'train':
            self.transform = transforms.Compose([
                Resize((224, 224)),
                transforms.RandomCrop(224, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.0434, 0.0434, 0.0434], std=[0.0261, 0.0261, 0.0261]),
            ])

        elif train_mode == 'val':
            self.transform = transforms.Compose([
                Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.0434, 0.0434, 0.0434], std=[0.0261, 0.0261, 0.0261]),
            ])

        elif train_mode == 'test':
            self.transform = transforms.Compose([
                Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.0434, 0.0434, 0.0434], std=[0.0261, 0.0261, 0.0261]),
            ])


    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = Image.open(fn).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label


    def __len__(self):
        return len(self.imgs)


if __name__ == "__main__":
    txt_path = "/home/wang/PycharmProjects/GSCNet/Datasets/SampledDDDImages/train.txt"
    train_datasets = DrowsiessDetectionDataset(txt_path=txt_path, train_mode="train")
    img, label = train_datasets[1]


