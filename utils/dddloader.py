import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import random
from PIL import Image, ImageOps, ImageFilter
import pandas as pd


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

    def __init__(self, img_dir, csv_path, train_mode):
        """
        csv_path: csv路径
        train_mode:  "train", "val", "test" 三种
        """
        assert train_mode in ["train", "val", "test"]

        names = ['drowsiness', 'picname', 'subject_no', 'situation', 'condition']
        df = pd.read_csv(csv_path, header=0, sep=',', index_col=None, usecols=names)
        df.loc[df['drowsiness'] == '\n', 'drowsiness'] = 0
        self.imgs = df["picname"].tolist()
        self.labels = df["drowsiness"].tolist()
        self.img_dir = img_dir

        if train_mode == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(224),
                transforms.RandomCrop(224, padding=4),
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                RandomRotate(15, 0.3),
                # RandomGaussianBlur(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.0434, 0.0434, 0.0434], std=[0.0261, 0.0261, 0.0261]),
            ])

        elif train_mode == 'val':
            self.transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.0434, 0.0434, 0.0434], std=[0.0261, 0.0261, 0.0261]),
            ])

        elif train_mode == 'test':
            self.transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.0434, 0.0434, 0.0434], std=[0.0261, 0.0261, 0.0261]),
            ])


    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.imgs[index])
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            transform_img = self.transform(img)
        img_label = self.labels[index]
        label = int(img_label)

        # 0: stillness, 1: drowsy
        #text_labels = ['stillness', 'drowsy']
        #text_label = text_labels[label]
        return transform_img, label


    def __len__(self):
        return len(self.imgs)


if __name__ == "__main__":
    val_img_dir = "/home/wang/PycharmProjects/GSCNet/Datasets/SampledDDDImages/TrainSampledImages/"
    val_csv_path = "/home/wang/PycharmProjects/GSCNet/Datasets/SampledDDDImages/SampledTrain.csv"
    val_datasets = DrowsiessDetectionDataset(img_dir=val_img_dir, csv_path=val_csv_path, train_mode="train")
    val_datasets[1]

