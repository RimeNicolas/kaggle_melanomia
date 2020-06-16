import os
from PIL import Image
import torch
import numpy as np
from torchvision import transforms as T
import torchvision
import cv2 as cv
import sys
from random import randint


class RandomDataset(torch.utils.data.Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, 1, size, size)
        self.label = torch.randint(2, (length,)).long()

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return self.len


class CircleSquareImg():
    def __init__(self, dir_img, nb_img, size_img, is_train=True):
        os.makedirs(dir_img, exist_ok=True)
        self.dir_img = dir_img
        self.nb_img = nb_img
        self.size_img = size_img
        self.csv_data = list()
        self.ii = 1
        self.is_train = is_train

    def __len__(self):
        return self.nb_img

    def _random_color(self):
        a = randint(0,3)
        return (255, 0, 0)
        # if a == 0:
        #     return (255, 0, 0)
        # elif a == 1:
        #     return (0, 255, 0)
        # else:
        #     return (0, 0, 255)

    def _generate_circle(self, path_img, img, p1, size):
        cv.circle(img, p1, size, color=self._random_color(), thickness=-1)
        self.csv_data.append(path_img + ',0')
        return img

    def _generate_square(self, path_img, img, p1, size):
        p2 = (p1[0] + size, p1[1] + size)
        cv.rectangle(img, p1, p2, color=self._random_color(), thickness=-1)
        self.csv_data.append(path_img + ',1')
        return img

    def _generate_img(self):
        if self.is_train:
            os.makedirs(os.path.join(self.dir_img, 'train'), exist_ok=True)
            csv_file = os.path.join(self.dir_img, 'train.csv')
        else:
            os.makedirs(os.path.join(self.dir_img, 'test'), exist_ok=True)
            csv_file = os.path.join(self.dir_img, 'test.csv')
        for i in range(self.nb_img):
            img = np.zeros((self.size_img, self.size_img, 3), dtype=np.uint8)
            if self.is_train:
                path_img = os.path.join(self.dir_img, 'train', str(i) + '.jpg')
            else:
                path_img = os.path.join(self.dir_img, 'test', str(i) + '.jpg')

            if i % 2 == 0:
                img = self._generate_circle(path_img, img, (randint(0, self.size_img), randint(0, self.size_img)), randint(0, self.size_img // 2))
                # self._generate_circle(img, (self.size_img // 2, self.size_img // 2), randint(0, self.size_img // 2))
            else:
                img = self._generate_square(path_img, img, (randint(0, self.size_img), randint(0, self.size_img)), randint(0, self.size_img // 2))
                # self._generate_square(img, (self.size_img // 2, self.size_img // 2), randint(0, self.size_img // 2))
            cv.imwrite(path_img, img)
            
        with open(csv_file, 'w') as csvfile:
            for el in self.csv_data:
                csvfile.write(el + '\n')

    def __call__(self):
        self._generate_img()

def generate_imgs():
    dir_img = r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\bench2'
    nb_img = 3000
    size_img = 256
    np.random.seed(0)
    CircleSquareImg(dir_img, nb_img, size_img, is_train=True)()
    CircleSquareImg(dir_img, nb_img // 10, size_img, is_train=False)()

def generate_csv_test():
    path_csv = r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\bench2\test.csv'
    path_csv2 = r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\bench2\test_unlabelled.csv'
    with open(path_csv, 'r') as fd:
        data_imgs = fd.readlines()
    with open(path_csv2, 'w') as csvfile:
        for el in data_imgs:
            path = el.split(',')[0]
            csvfile.write(path + '\n')


if __name__ == '__main__':
    # generate_imgs()
    generate_csv_test()

