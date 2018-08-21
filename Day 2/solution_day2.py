import numpy as np
import matplotlib.pyplot as plt
import cv2

from load_gtsrb import *

# Load dataset
dataset_path = "C:/Users/schaf/Documents/GTSRB/Final_Training/Images/"
classes = range(43)
[imgs, labels, class_descs, sign_ids] = load_gtsrb_images(dataset_path, classes, 50)    

################
## EXERCISE 1 ##
################

print(imgs.shape)
print(labels.shape)