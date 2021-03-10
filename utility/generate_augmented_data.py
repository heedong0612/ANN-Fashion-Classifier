"""
hi

Functions
- rotate, blur, add_salt_pepper_noise, distort, translate, scale, flip
- augment: performs some or all of those augmentation for each data sample in train set

Instructions
1) In 'uwb-css-485-winter-2021' directory, make duplicate of 'train.csv' and name it 'train_for_aug.csv'
2) Make following changes to 'train_for_aug.csv'
    - delete the first row ("id", "label", "pixel1", "pixel2", ...)
    - delete the validation data (rows from 50,001 - 60,000)
3) In the same depth as the 'utility' folder, make a directory called 'augmented'
4) Currently this code performs 4 different augmentation. if you want, make changes to augment() function
5) Run this code and the output will be saved in 'augmented' directory
6) When you train your model on MATLAB, train with main_agumented() instead of main()

"""
import pandas as pd
import numpy as np
from numpy import genfromtxt
from IPython.display import Image
from PIL import Image as PILimage
import scipy.ndimage
import cv2
import random
import csv
from csv import reader, writer
from datetime import datetime

import gc

def blur(sample, sigma=0.58):
    return scipy.ndimage.filters.gaussian_filter(sample, sigma=sigma) 

def add_salt_pepper_noise(sample, prob=0.05):
    output = np.zeros(sample.shape,np.uint8)
    thres = 1 - prob 
    for i in range(sample.shape[0]):
        for j in range(sample.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = sample[i][j]
    return output

def distort(img, orientation='horizontal', func=np.sin, x_scale=0.05, y_scale=5):
    assert orientation[:3] in ['hor', 'ver'], "dist_orient should be 'horizontal'|'vertical'"
    assert func in [np.sin, np.cos], "supported functions are np.sin and np.cos"
    assert 0.00 <= x_scale <= 0.1, "x_scale should be in [0.0, 0.1]"
    assert 0 <= y_scale <= min(img.shape[0], img.shape[1]), "y_scale should be less then image size"
    img_dist = img.copy()
    
    def shift(x):
        return int(y_scale * func(np.pi * x * x_scale))
    
    for _ in range(3):
        for i in range(img.shape[orientation.startswith('ver')]):
            if orientation.startswith('ver'):
                img_dist[:, i] = np.roll(img[:, i], shift(i))
            else:
                img_dist[i, :] = np.roll(img[i, :], shift(i))
            
    return img_dist

def translate(sample, shift=1, direction = 1):
    # direction 1,2,3,4 correstponds to 
    # left, right, upward, downward

    if direction < 3:
        shifted_area = np.zeros((28, shift))
        ax = 1
    else:
        shifted_area = np.zeros((shift, 28))
        ax = 0
    
    if direction == 1:
        sample = sample[:, shift:]
    elif direction == 2:
        sample = sample[:, :-shift]
    elif direction == 3:
        sample = sample[shift:, :]
    else:
        sample = sample[:-shift, :]
    
    if direction % 2 == 0:  # downard and right
        shifted = np.concatenate((shifted_area, sample),axis=ax)
    else:
        shifted = np.concatenate((sample, shifted_area),axis=ax)
    
    return shifted

def scale(sample, dsize=(28,28), resultsize=(28,28), interpolation=cv2.INTER_CUBIC):
    res = cv2.resize(sample, dsize=dsize, interpolation=cv2.INTER_CUBIC)

    if dsize[0] > resultsize[0]: # cut from each side almost equally
        diff = dsize[0] - resultsize[0]
        res = res[:,diff//2 : -(diff)//2]

    if dsize[1] > resultsize[1]:
        diff = dsize[1] - resultsize[1]
        res = res[diff//2:-(diff)//2,:]
        
    return res

def rotate(sample, angle=2):
    return scipy.ndimage.rotate(sample, angle, reshape=False)

def flip(sample):
    return np.fliplr(sample)

def augment(filename):
    t = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    count = 0

    with open(f'../augmented/augmented_train_{t}.csv', 'a', newline='') as f1:
        csvwriter1 = writer(f1)
        with open(f'../augmented/augmented_label_{t}.csv', 'a', newline='') as f2:
            csvwriter2 = writer(f2)

            with open(filename, 'r') as read_obj:
                csv_reader = reader(read_obj)

                for row in csv_reader:
                    count += 1
                    if count % 1000 == 0: print(f'Progress: {count}/50000')
                    if count % 500 == 0: gc.collect()

                    arr = np.asarray(row).astype(float)
                    label = arr[1]
                    arr = arr[2:]

                    sample = arr.reshape(28,28)

                    """
                    !!! Make changes here if you want more or less augmentation !!!
                        Don't forget to .ravel()
                        Don't forget to change the range(n) in the for loop for writing labels
                    """
                    # writing origianl and augmented images
                    csvwriter1.writerows([sample.ravel(),\
                                        rotate(sample,-2).ravel(),\
                                        rotate(sample, 2).ravel(),\
                                        flip(sample).ravel(),\
                                        distort(sample, x_scale=0.03, y_scale=2).ravel()])
                    
                    # writing labels
                    for _ in range(5):
                        csvwriter2.writerow([label])
    
    return t

def main():

    t = augment("../uwb-css-485-winter-2021/train_for_aug.csv") # this file should not contain the first row and the rows from 50,001 - 60,000
    
    print("\nFiles generated: ")
    print(f'augmented/augmented_train_{t}.csv')
    print(f'augmented/augmented_label_{t}.csv')

if __name__ == "__main__":
    main()