import os
import numpy as np
import csv
import cv2 as cv
from shutil import copy

from joblib import Parallel, delayed
import multiprocessing

from collections import Counter


def preprocess_dataset2(data_list_file, dir_imgs_old, dir_imgs_new):
    os.makedirs(dir_imgs_new, exist_ok=True)
    os.makedirs(os.path.join(dir_imgs_new,'0'), exist_ok=True)
    os.makedirs(os.path.join(dir_imgs_new,'1'), exist_ok=True)

    with open(os.path.join(data_list_file), 'r') as fd:
            data_imgs = fd.readlines()

    csv_data = list()
    n0 = 1
    n1 = 1

    for i, data_img in enumerate(data_imgs[1:]):
        # if i >= 1000:
        #     break
        split = data_img.split(',')
        img_name = split[0] + '.jpg'
        label = str(split[-1][:-1])
        if label != '0':
            n1 += 1
            n0 -= 5
        if n1 > n0:
            n0 += 1
            path_img_old = os.path.join(dir_imgs_old, img_name)
            path_img_new = os.path.join(dir_imgs_new, label, img_name)
            copy(path_img_old, path_img_new)
            csv_data.append(','.join([path_img_new, label]))

    print('n 0 =', n0)
    print('n 1 =', n1)

    csv_file = os.path.join(dir_imgs_new, 'train.csv')
    with open(csv_file, 'w') as csvfile:
        for el in csv_data:
            csvfile.write(el + '\n')


def preprocess_test_data(data_list_file, dir_imgs_old, dir_imgs_new):
    os.makedirs(dir_imgs_new, exist_ok=True)
    os.makedirs(os.path.join(dir_imgs_new, 'img'), exist_ok=True)

    with open(os.path.join(data_list_file), 'r') as fd:
            data_imgs = fd.readlines()

    csv_data = list()
    for i, data_img in enumerate(data_imgs[1:]):
        # if i >= 1000:
        #     break
        split = data_img.split(',')
        img_name = split[0] + '.jpg'
        path_img_old = os.path.join(dir_imgs_old, img_name) 
        path_img_new = os.path.join(dir_imgs_new, 'img', img_name)
        copy(path_img_old, path_img_new)
        csv_data.append(path_img_new)

    csv_file = os.path.join(dir_imgs_new, 'test.csv')
    with open(csv_file, 'w') as csvfile:
        for el in csv_data:
            csvfile.write(el + '\n')


def center_crop(img):
    height, width, _ = img.shape
    if height == width:
        return img
    elif height > width:
        img = img[int(height / 2 - width / 2)
            : int(height / 2 + width / 2 + 0.5),:]
    else:
        img = img[:,int(width / 2 - height / 2)
            : int(width / 2 + height / 2 + 0.5)]
    return img


def f3(path_img, min_size=1024):
    img = cv.imread(path_img)
    img = center_crop(img)
    if np.min(img.shape) == min_size:
        return 
    if img.shape[0] >= img.shape[1]:
        scale = min_size / img.shape[1]
    else:
        scale = min_size / img.shape[0]
    width = int(img.shape[1] * scale + 0.5)
    height = int(img.shape[0] * scale + 0.5)
    dim = (width, height)
    resized = cv.resize(img, dim, cv.INTER_AREA)
    cv.imwrite(path_img, resized)


def resize_img(dir_img, min_size = 1024, f=f3):
    name_imgs = os.listdir(dir_img)
    path_imgs = [os.path.join(dir_img, name_img) for name_img in name_imgs]
    num_cores = multiprocessing.cpu_count()
    Parallel(n_jobs=num_cores)(delayed(f)(path_img, min_size) for path_img in path_imgs)


##############################################


def create_balanced_dataset():
    dir_imgs_old = r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\original\jpeg\train'
    dir_imgs_new = r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\dataset2\jpeg_train'
    data_list_file = r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\original\train.csv'
    preprocess_dataset2(data_list_file, dir_imgs_old, dir_imgs_new)


def create_testset():
    dir_imgs_old = r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\original\jpeg\test'
    dir_imgs_new = r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\dataset2\jpeg_test'
    data_list_file = r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\original\test.csv'
    preprocess_test_data(data_list_file, dir_imgs_old, dir_imgs_new)


def resize_imgs():
    dir_imgs0 = r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\dataset2\jpeg_train\0'
    dir_imgs1 = r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\dataset2\jpeg_train\1'
    dir_imgs_test = r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\dataset2\jpeg_test\img'
    resize_img(dir_imgs0)
    resize_img(dir_imgs1)
    resize_img(dir_imgs_test)


if __name__ == '__main__':
    resize_imgs()