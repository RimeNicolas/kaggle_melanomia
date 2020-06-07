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

        # data_imgs = [data_img[:-1] for data_img in data_imgs[1:]]
        data_imgs = [data_img[:-1] for data_img in data_imgs]

        if self.phase == 'train':
            self.data_imgs = np.random.permutation(data_imgs)
            self.transforms = T.Compose([
                # T.Grayscale(num_output_channels=1),
                # T.RandomAffine(degrees=180, translate=(0.45,0.45), scale=(0.5,1.5), shear=None, resample=False, fillcolor=0),
                # T.CenterCrop(self.input_shape[1:]),
                T.Resize(self.input_shape[1:], interpolation=2),
                T.RandomVerticalFlip(p=0.5),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
                T.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                # T.transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        else:
            self.data_imgs = data_imgs
            self.transforms = T.Compose([
                # T.Grayscale(num_output_channels=1),
                # T.CenterCrop(self.input_shape[1:]),
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
        img = self.transforms(img)
        if self.phase == 'test':
            return img.float(), img_path[-16:-4]
        label = np.int32(splits[-1])
        return img.float(), label

    def __len__(self):
        return len(self.data_imgs)


class RandomDataset(torch.utils.data.Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, 1, size, size)
        self.label = torch.randint(2, (length,)).long()

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return self.len


def test_dataset_img(batch_imgs=True):
    dataset = Dataset(path_csv_file=r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\dataset1\jpeg_train\train.csv',
        input_shape=(1, 100, 100))

    trainloader = torch.utils.data.DataLoader(dataset, batch_size=10)
    for i, (data, label) in enumerate(trainloader):
        if i > 0:
            break
        if batch_imgs:
            img = torchvision.utils.make_grid(data).numpy()
        else:
            img = data.numpy()[0]
        img = np.transpose(img, (1, 2, 0))
        img += np.array([1, 1, 1])
        img *= 127.5
        img = img.astype(np.uint8)
        img = img[:, :, [2, 1, 0]]
        # cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        cv2.imshow('img', img)
        cv2.waitKey()


if __name__ == '__main__':
    pass
    # test_dataset_img()