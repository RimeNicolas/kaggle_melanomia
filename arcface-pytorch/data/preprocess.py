import os
import numpy as np
import csv
import cv2 as cv
from shutil import copy

from joblib import Parallel, delayed
import multiprocessing

from collections import Counter

# old
# def cnt_img_shapes(dir_imgs):
#     list_shapes = list()
#     name_imgs = os.listdir(dir_imgs)
#     for i, name_img in enumerate(name_imgs):
#         if (i+1)%100 == 0:
#             print(i+1)
#             break
#         path_img = os.path.join(dir_imgs, name_img)
#         img = cv.imread(path_img, 1)
#         list_shapes.append(img.shape)
#     cnt = Counter(list_shapes)
#     return cnt


def f1(dir_imgs, name_img):
    path_img = os.path.join(dir_imgs, name_img)
    img = cv.imread(path_img, 1)
    return img.shape


def parallel_cnt_img_shapes(dir_imgs, f=f1):
    list_shapes = list()
    name_imgs = os.listdir(dir_imgs)

    num_cores = multiprocessing.cpu_count()
    list_shapes = Parallel(n_jobs=num_cores)(delayed(f)(dir_imgs, i) for i in name_imgs)
    print('list length = ', len(list_shapes))
    return list_shapes


def f2(path_img, min_size):
    img = cv.imread(path_img)
    if np.min(img.shape) == min_size:
        return 
    if img.shape[0] > img.shape[1]:
        scale = min_size / img.shape[1]
    else:
        scale = min_size / img.shape[0]
    width = int(img.shape[1] * scale + 0.5)
    height = int(img.shape[0] * scale + 0.5)
    dim = (width, height)
    resized = cv.resize(img, dim)
    cv.imwrite(path_img, resized)


def resize_img(dir_img, min_size = 1024, f=f2):
    name_imgs = os.listdir(dir_img)
    path_imgs = [os.path.join(dir_img, name_img) for name_img in name_imgs]
    num_cores = multiprocessing.cpu_count()
    Parallel(n_jobs=num_cores)(delayed(f)(path_img, min_size) for path_img in path_imgs)


def preprocess_dataset1(data_list_file, dir_imgs_old, dir_imgs_new):
    os.makedirs(dir_imgs_new, exist_ok=True)

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
            n0 -= 1
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


def process_csv(data_list_file, data_list_file_out, dir_img_new, test=False):
    with open(os.path.join(data_list_file), 'r') as fd:
            data_imgs = fd.readlines()

    csv_data = list()

    for i, data_img in enumerate(data_imgs[1:]):
        # if i >= 1000:
        #     break
        split = data_img.split(',')
        img_name = split[0] + '.jpg'
        label = str(split[-1][:-1])
        path_img_new = os.path.join(dir_img_new, img_name)
        if test == False:
            csv_data.append(','.join([path_img_new, label]))
        else:
            csv_data.append(path_img_new)

    with open(data_list_file_out, 'w') as csvfile:
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


def read_traincsv(csv_file):
    csv_data = list()
    with open(csv_file, 'r') as csvfile:
        content = csvfile.readlines()
    csv_data = [x.strip() for x in content]
    path_img = csv_data[0].split(',')[0]
    img = cv.imread(path_img)
    cv.imshow('img', img)
    cv.waitKey()



##############################################


def create_balanced_dataset():
    dir_imgs_old = r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\original\jpeg\train'
    dir_imgs_new = r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\dataset1\jpeg_train'
    data_list_file = r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\original\train.csv'
    preprocess_dataset1(data_list_file, dir_imgs_old, dir_imgs_new)


def create_testset():
    dir_imgs_old = r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\original\jpeg\test'
    dir_imgs_new = r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\dataset1\jpeg_test'
    data_list_file = r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\original\test.csv'
    preprocess_test_data(data_list_file, dir_imgs_old, dir_imgs_new)


def cnt_imgs_shape():
    dir_imgs = r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\dataset1\jpeg_train'
    dir_imgs0 = os.path.join(dir_imgs, str(0))
    dir_imgs1 = os.path.join(dir_imgs, str(1))
    dir_imgs_test = r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\dataset1\jpeg_test\img'
    list_shape0 = parallel_cnt_img_shapes(dir_imgs0)
    list_shape1 = parallel_cnt_img_shapes(dir_imgs1)
    list_shape_test = parallel_cnt_img_shapes(dir_imgs_test)
    cnt0 = Counter(list_shape0)
    cnt1 = Counter(list_shape1)
    cnt  = Counter(list_shape0 + list_shape1)
    cnt_test = Counter(list_shape_test)
    print(cnt0)
    print(' ')
    print(cnt1)
    print(' ')
    print(cnt)
    print(' ')
    print(cnt_test)


def resize_imgs():
    dir_imgs = r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\dataset1\jpeg_train'
    dir_imgs0 = os.path.join(dir_imgs, str(0))
    dir_imgs1 = os.path.join(dir_imgs, str(1))
    dir_imgs_test = r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\dataset1\jpeg_test\img'
    # resize_img(dir_imgs0)
    resize_img(dir_imgs1)
    resize_img(dir_imgs_test)


def create_csv():
    csv_in = r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\original\train.csv'
    csv_out = r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\original\jpeg\train.csv'
    dir_img = r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\original\jpeg\train'
    process_csv(csv_in, csv_out, dir_img)
    csv_in = r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\original\test.csv'
    csv_out = r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\original\jpeg\test.csv'
    dir_img = r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\original\jpeg\test'
    process_csv(csv_in, csv_out, dir_img, test=True)
    


if __name__ == '__main__':
    # resize_imgs()
    # cnt_imgs_shape()
    create_csv()
    

