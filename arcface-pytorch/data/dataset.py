import os
from PIL import Image
import torch
import numpy as np
from torchvision import transforms as T
import torchvision
import cv2
import sys


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path_csv_file, input_shape=(1, 512, 512), phase='train'):
        self.phase = phase
        self.input_shape = input_shape

        with open(path_csv_file, 'r') as fd:
            data_imgs = fd.readlines()

        data_imgs = [data_img[:-1] for data_img in data_imgs]

        if self.phase == 'train':
            self.data_imgs = np.random.permutation(data_imgs)
            self.transforms1 = T.Compose([
                # T.Grayscale(num_output_channels=1),
                # T.RandomAffine(degrees=180, translate=(0.45,0.45), scale=(0.5,1.5), shear=None, resample=False, fillcolor=0),
                T.Resize(self.input_shape[1:], interpolation=2),
                T.RandomVerticalFlip(p=0.5),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
                T.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                # T.transforms.Normalize(mean=[0.5], std=[0.5])
            ])
            self.transforms2 = T.Compose([
                # T.Grayscale(num_output_channels=1),
                # T.RandomAffine(degrees=180, translate=(0.45,0.45), scale=(0.5,1.5), shear=None, resample=False, fillcolor=0),
                T.RandomCrop(self.input_shape[1:]),
                T.RandomVerticalFlip(p=0.5),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
                T.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                # T.transforms.Normalize(mean=[0.5], std=[0.5])
            ])
            self.transforms3 = T.Compose([
                T.RandomResizedCrop(size = 224, scale = (0.5, 1.0)), 
                T.RandomHorizontalFlip(), 
                T.RandomVerticalFlip(), 
                T.ColorJitter(brightness = 32. / 255., saturation = 0.5),
                T.ToTensor(), 
                T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
            ])
        else:
            self.data_imgs = data_imgs
            self.transforms = T.Compose([
                T.Resize(self.input_shape[1:], interpolation=2),
                T.ToTensor(),
                T.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                # T.transforms.Normalize(mean=[0.5], std=[0.5])
            ])

    def __getitem__(self, index):
        sample = self.data_imgs[index]
        splits = sample.split(',')
        img_path = splits[0]
        img = Image.open(img_path)
        # img = img.convert('L')
        if self.phase == 'train':
            img = self.transforms1(img)
            # if np.random.randint(2) == 0:
            #     img = self.transforms1(img)
            # else:
            #     img = self.transforms2(img)
        else:
            img = self.transforms(img)
        if self.phase == 'test':
            return img.float(), img_path.split('\\')[-1].split('.')[0]
        label = np.int32(splits[-1])
        return img.float(), label

    def __len__(self):
        return len(self.data_imgs)


def transform_show_img(img):
    img = np.transpose(img, (1, 2, 0))
    img += np.array([1, 1, 1])
    img *= 127.5
    img = img.astype(np.uint8)
    img = img[:, :, [2, 1, 0]]
    # cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.imshow('img', img)
    cv2.waitKey()


def show_dataset_img(batch_imgs=True):
    trainset = Dataset(path_csv_file=r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\dataset1\jpeg_train\train.csv',
        input_shape=(3, 224, 224), phase='train')
    testset = Dataset(path_csv_file=r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\testset\test.csv',
        input_shape=(3, 224, 224), phase='test')

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16)
    testloader = torch.utils.data.DataLoader(testset, batch_size=16)
    for i, (data, label) in enumerate(trainloader):
        if i > 0:
            break
        if batch_imgs:
            img = torchvision.utils.make_grid(data).numpy()
        else:
            img = data.numpy()[0]
        transform_show_img(img)

    for i, (data, label) in enumerate(testloader):
        if i > 0:
            break
        if batch_imgs:
            img = torchvision.utils.make_grid(data).numpy()
        else:
            img = data.numpy()[0]
        transform_show_img(img)


if __name__ == '__main__':
    show_dataset_img()