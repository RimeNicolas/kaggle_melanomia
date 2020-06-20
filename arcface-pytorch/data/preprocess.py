import os
import numpy as np
import csv
import cv2 as cv
from shutil import copy

from joblib import Parallel, delayed
import multiprocessing

from collections import Counter


def center_crop(img):
    """
    crop maximum square of img
    example: 6000x4000 -> 4000x4000
    """
    height, width, _ = img.shape
    # print('img shape before center crop :', img.shape)
    if height == width:
        return img
    elif height > width:
        img = img[int(height // 2 - width // 2)
            : int(height // 2 + width // 2),:]
        if width % 2 == 1:
            img = img[:,:-1]
    else:
        img = img[:,int(width // 2 - height // 2)
            : int(width // 2 + height // 2)]
        if height % 2 == 1:
            img = img[:-1,:]
    # print('img shape after center crop :', img.shape)
    # print('-'*10)
    return img


def crop_resize_img(path_img_old, path_img_new, min_size=1024):
    """
    crop center of img from path_img_old using center_crop
    resize it to min_size
    write it at path_img_new
    """
    img = cv.imread(path_img_old)
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
    cv.imwrite(path_img_new, resized)


def preprocess_trainset(path_csv_old, dir_imgs_old, dir_imgs_new, diff=0):
    """
    process img using crop_resize_img and create new .csv file
    diff is difference number between class 0 (majority) and 1 (minority)
    example: for diff = 2, there are 3 class 0 img for 1 class 1 img
    """
    os.makedirs(dir_imgs_new, exist_ok=True)
    os.makedirs(os.path.join(dir_imgs_new,'0'), exist_ok=True)
    os.makedirs(os.path.join(dir_imgs_new,'1'), exist_ok=True)
    path_csv_new = os.path.join(dir_imgs_new, 'train.csv')

    with open(os.path.join(path_csv_old), 'r') as fd:
        data_imgs = fd.readlines()

    csv_data = list()
    n0 = 1
    n1 = 1

    for i, data_img in enumerate(data_imgs[1:]):
        # if i>1000:
        #     break
        split = data_img.split(',')
        img_name = split[0] + '.jpg'
        label = str(split[-1][:-1])
        if label == '1':
            n1 += 1
            n0 -= (1 + diff)
        if n1 > n0:
            n0 += 1
            path_img_old = os.path.join(dir_imgs_old, img_name)
            path_img_new = os.path.join(dir_imgs_new, label, img_name)
            csv_data.append(','.join([path_img_new, label]))
            crop_resize_img(path_img_old, path_img_new)

    with open(path_csv_new, 'w') as csvfile:
        for el in csv_data:
            csvfile.write(el + '\n')


def preprocess_train_data(path_csv_old, dir_imgs_old, dir_imgs_new):
    """
    process all images using crop_resize_img and create new .csv file
    """
    os.makedirs(dir_imgs_new, exist_ok=True)
    os.makedirs(os.path.join(dir_imgs_new,'0'), exist_ok=True)
    os.makedirs(os.path.join(dir_imgs_new,'1'), exist_ok=True)
    path_csv_new = os.path.join(dir_imgs_new, 'train.csv')

    with open(os.path.join(path_csv_old), 'r') as fd:
        data_imgs = fd.readlines()

    csv_data = list()

    def f(data_img):
        split = data_img.split(',')
        img_name = split[0] + '.jpg'
        label = str(split[-1][:-1])
        path_img_old = os.path.join(dir_imgs_old, img_name)
        path_img_new = os.path.join(dir_imgs_new, label, img_name)
        crop_resize_img(path_img_old, path_img_new)
        return ','.join([path_img_new, label])
    num_cores = multiprocessing.cpu_count()
    csv_data = Parallel(n_jobs=num_cores)(delayed(f)(data_img) for data_img in data_imgs[1:])

    with open(path_csv_new, 'w') as csvfile:
        for el in csv_data:
            csvfile.write(el + '\n')


def preprocess_test_data(path_csv_old, dir_imgs_old, dir_imgs_new):
    os.makedirs(dir_imgs_new, exist_ok=True)
    os.makedirs(os.path.join(dir_imgs_new, 'img'), exist_ok=True)

    with open(os.path.join(path_csv_old), 'r') as fd:
        data_imgs = fd.readlines()

    csv_data = list()

    def f(data_img):
        split = data_img.split(',')
        img_name = split[0] + '.jpg'
        path_img_old = os.path.join(dir_imgs_old, img_name) 
        path_img_new = os.path.join(dir_imgs_new, 'img', img_name)
        crop_resize_img(path_img_old, path_img_new)
        return path_img_new

    num_cores = multiprocessing.cpu_count()
    csv_data = Parallel(n_jobs=num_cores)(delayed(f)(data_img) for data_img in data_imgs[1:])

    csv_file = os.path.join(dir_imgs_new, 'test.csv')
    with open(csv_file, 'w') as csvfile:
        for el in csv_data:
            csvfile.write(el + '\n')


def parallel_cnt_img_shapes(dir_imgs):
    def f(dir_imgs, name_img):
        path_img = os.path.join(dir_imgs, name_img)
        img = cv.imread(path_img, 1)
        return img.shape
    list_shapes = list()
    name_imgs = os.listdir(dir_imgs)

    num_cores = multiprocessing.cpu_count()
    list_shapes = Parallel(n_jobs=num_cores)(delayed(f)(dir_imgs, i) for i in name_imgs)
    print('img nb in {} : {}'.format(dir_imgs, len(list_shapes)))
    cnt = Counter(list_shapes)
    print(cnt)


##############################################


def create_dataset1():
    path_csv_old = r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\original\train.csv'
    dir_imgs_old = r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\original\jpeg\train'
    dir_imgs_new = r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\dataset1\jpeg_train'
    preprocess_trainset(path_csv_old, dir_imgs_old, dir_imgs_new)

def create_dataset2():
    path_csv_old = r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\original\train.csv'
    dir_imgs_old = r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\original\jpeg\train'
    dir_imgs_new = r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\dataset2\jpeg_train'
    preprocess_trainset(path_csv_old, dir_imgs_old, dir_imgs_new, diff=5)

def create_dataset3():
    path_csv_old = r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\original\train.csv'
    dir_imgs_old = r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\original\jpeg\train'
    dir_imgs_new = r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\dataset3\jpeg_train'
    preprocess_train_data(path_csv_old, dir_imgs_old, dir_imgs_new)

def cnt_imgs_shape():
    dir_imgs0 = r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\dataset3\jpeg_train\0'
    dir_imgs1 = r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\dataset3\jpeg_train\1'
    dir_imgs_test = r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\testset\img'
    parallel_cnt_img_shapes(dir_imgs0)
    parallel_cnt_img_shapes(dir_imgs1)
    # parallel_cnt_img_shapes(dir_imgs_test)

def create_testset():
    path_csv_old = r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\original\test.csv'
    dir_imgs_old = r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\original\jpeg\test'
    dir_imgs_new = r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\testset'
    preprocess_test_data(path_csv_old, dir_imgs_old, dir_imgs_new)

if __name__ == '__main__':
    # create_dataset1()
    # create_dataset2()
    create_dataset3()
    # create_testset()
    cnt_imgs_shape()

