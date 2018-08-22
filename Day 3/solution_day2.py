import numpy as np
import os
import math
import matplotlib.pyplot as plt
import cv2

from load_gtsrb import *  

################
## EXERCISE 1 ##
################

# Load dataset
dataset_path = "C:/Users/schaf/Documents/GTSRB/Final_Training/Images/"
classes = range(43)
[imgs, labels, class_descs, sign_ids] = load_gtsrb_images(dataset_path, classes, 150)  
imgs = imgs.astype(np.uint8)

imgs = np.array([img.astype(np.uint8) for img in imgs])
imgs = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in imgs])