import os
import numpy as np
import csv
import cv2 as cv
from shutil import copy

from joblib import Parallel, delayed
import multiprocessing

from collections import Counter




class Enhance:
    def __init__(self, csv_in, csv_out, dir_imgs, scales=[1.5, 2.0]):
        self.scales = scales
        self.csv_in = csv_in
        self.csv_out = csv_out
        self.dir_imgs = dir_imgs
        self.csv_data = self._read_csv1()

    def _read_csv1(self):
        data = list()
        aa = list()
        with open(self.csv_in, newline='') as f:
            reader = csv.reader(f)
            data = list(reader)
        for el in data:
            if el[1] == '1':
                aa.append(','.join(el))
        return aa

    def _read_csv0(self):
        data = list()
        with open(self.csv_in, newline='') as f:
            reader = csv.reader(f)
            data = list(reader)
        for el in data:
            if el[1] == '0':
                self.csv_data.append(','.join(el))

    def zoom_in(self, img, scale):
        size = img.shape[0]
        new_size = int(scale * size)
        img = cv.resize(img, (new_size, new_size), cv.INTER_AREA)
        aa = new_size // 2 - size // 2
        return img[aa:aa+size, aa:aa+size]

    def _enhance(self):
        path_imgs = [os.path.join(self.dir_imgs, path_img) for path_img in os.listdir(self.dir_imgs)]
        for path_img in path_imgs:
            img0 = cv.imread(path_img)
            img1 = self.zoom_in(img0, self.scales[0])
            img2 = self.zoom_in(img0, self.scales[1])
            path_img1 = path_img[:-4] + '_' + str(int(self.scales[0])) + '.jpg'
            path_img2 = path_img[:-4] + '_' + str(int(self.scales[1])) + '.jpg'
            cv.imwrite(path_img1, img1)
            cv.imwrite(path_img2, img2)
            self.csv_data.append(path_img1 + ',1')
            self.csv_data.append(path_img2 + ',1')
        
        # take new images
        path_imgs = [os.path.join(self.dir_imgs, path_img) for path_img in os.listdir(self.dir_imgs)]
        for path_img in path_imgs:
            img0 = cv.imread(path_img)
            img1 = cv.rotate(img0, cv.ROTATE_90_CLOCKWISE)
            img2 = cv.rotate(img0, cv.ROTATE_90_COUNTERCLOCKWISE) 
            img3 = cv.rotate(img0, cv.ROTATE_180)
            path_img1 = path_img[:-4] + '_' + str(90) + '.jpg'
            path_img2 = path_img[:-4] + '_' + str(270) + '.jpg'
            path_img3 = path_img[:-4] + '_' + str(180) + '.jpg'
            cv.imwrite(path_img1, img1)
            cv.imwrite(path_img2, img2)
            cv.imwrite(path_img3, img3)
            self.csv_data.append(path_img1 + ',1')
            self.csv_data.append(path_img2 + ',1')
            self.csv_data.append(path_img3 + ',1')

        # take new images
        path_imgs = [os.path.join(self.dir_imgs, path_img) for path_img in os.listdir(self.dir_imgs)]
        for path_img in path_imgs:
            img0 = cv.imread(path_img)
            img1 = cv.flip(img0, 0)
            path_img1 = path_img[:-4] + '_f' + '.jpg'
            cv.imwrite(path_img1, img1)
            self.csv_data.append(path_img1 + ',1')

    def __call__(self):
        self._enhance()
        self._read_csv0()
        with open(self.csv_out, 'w') as csvfile:
            for el in self.csv_data:
                csvfile.write(el + '\n')


if __name__ == '__main__':
    csv_in = r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\dataset3\jpeg_train\train.csv'
    csv_out = r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\dataset3\jpeg_train\train2.csv'
    dir_imgs = r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\dataset3\jpeg_train\1'
    # Enhance(csv_in, csv_out, dir_imgs)()

