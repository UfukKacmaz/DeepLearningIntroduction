import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy
from skimage import color

from load_gtsrb import *

################
## EXERCISE 1 ##
################
dataset_path = "C:/Users/schaf/Documents/GTSRB/Final_Training/Images/"
classes = [1, 2]
[imgs, labels, class_descs, sign_ids] = load_gtsrb_images(dataset_path, classes, 500)    

################
## EXERCISE 2 ##
################
imgs_50 = np.array([img for img, label in zip(imgs, labels) if label == 2])
labels_50 = np.array([2 for _ in range(imgs_50.shape[0])])

indx = np.random.randint(0, imgs_50.shape[0], 5)
i_s = imgs_50[indx]
l_s = labels_50[indx]
 
for i, l in zip(i_s, l_s):
    plt.imshow(i)
    plt.title(l)
    plt.show()

imgs_30 = np.array([img for img, label in zip(imgs, labels) if label == 1], dtype=np.uint8)
labels_30 = np.array([1 for _ in range(imgs_50.shape[0])], dtype=np.uint8)

indx = np.random.randint(0, imgs_30.shape[0], 5)
i_s = imgs_30[indx]
l_s = labels_30[indx]
 
for i, l in zip(i_s, l_s):
    plt.imshow(i)
    plt.title(l)
    plt.show()

################
## EXERCISE 3 ##
################
# img = color.rgb2gray(io.imread('image.png'))

#for img in imgs_30