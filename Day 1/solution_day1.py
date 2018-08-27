import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy
import scipy.ndimage
from skimage import color
import matplotlib.patches as patches

from load_gtsrb import *

################
## EXERCISE 1 ##
################
# Load dataset
dataset_path = "C:/Users/schaf/Documents/GTSRB/Final_Training/Images/"
classes = [1, 2]
[imgs, labels, class_descs, sign_ids] = load_gtsrb_images(dataset_path, classes, 50)    

################
## EXERCISE 2 ##
################

# Get all 30 and 50 images
imgs_50 = np.array([img for img, label in zip(imgs, labels) if label == 2], dtype=np.uint8)
labels_50 = np.array([2 for _ in range(imgs_50.shape[0])], dtype=np.uint8)
imgs_30 = np.array([img for img, label in zip(imgs, labels) if label == 1], dtype=np.uint8)
labels_30 = np.array([1 for _ in range(imgs_50.shape[0])], dtype=np.uint8)

# Plot 5 random images from class 50
idx = np.random.randint(0, imgs_50.shape[0], 5)
i_s = imgs_50[idx]
l_s = labels_50[idx]
for i, l in zip(i_s, l_s):
    plt.imshow(i)
    plt.title(l)
    plt.show()

# Plot 5 random images from class 30
idx = np.random.randint(0, imgs_30.shape[0], 5)
i_s = imgs_30[idx]
l_s = labels_30[idx]
for i, l in zip(i_s, l_s):
    plt.imshow(i)
    plt.title(l)
    plt.show()

################
## EXERCISE 3 ##
################

# Convert images to uint8 and to grayscale
imgs = imgs.astype(np.uint8)
imgs = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in imgs], dtype=np.uint8)

################
## EXERCISE 4 ##
################

# Extract ROI
# Get index image for 3
img = imgs[7]
croped_img_30 = img[13:23, 11:16]
# Get index image for 5
img = imgs[60]
croped_img_50 = img[11:21, 10:15]

################
## EXERCISE 5 ##
################

# Conv image with given conv filter
def convolution(image, conv_filter):
    filter_h, filter_w = conv_filter.shape
    image_h, image_w = image.shape
    # Image padding for convolution
    image = np.pad(
        image, pad_width=((0, filter_h), (0, filter_w)), mode="constant", constant_values=0.0)
    max_value = 0.0
    max_value_index = 0, 0
    conv_img = np.empty(shape=(32, 32))
    conv_filter = conv_filter - np.mean(conv_filter)
    conv_sum = np.sum(conv_filter)
    # Iterate over image
    for i in range(image_h):
        for j in range(image_w):
            # For every pixel compute filter result
            value = np.sum(conv_filter * image[i:i+filter_h, j:j+filter_w])
            if i < image_h - filter_h and j < image_w - filter_w:
                conv_img[i][j] = value / conv_sum
            if value > max_value:
                max_value = value
                max_value_index = i, j
    return max_value, max_value_index, conv_img

# Conv image with 3 and 5 ROI image to get features
# and save computed features to disk
def extract_features(croped_img_30, croped_img_50):
    x = np.empty(shape=(imgs.shape[0], 2))
    for i, img in enumerate(imgs):
        feature_30, feature_30_pos,_ = convolution(img, croped_img_30)
        feature_50, feature_50_pos,_ = convolution(img, croped_img_50)
        x[i] = [feature_30, feature_50]
    y = labels
    np.save(dataset_path+"x.npy", x)
    np.save(dataset_path+"y.npy", y)

def load_features():
    x = np.load(dataset_path+"x.npy")
    y = np.load(dataset_path+"y.npy")
    return x, y

extract_features(croped_img_30, croped_img_50)
x, y = load_features()

################
## EXERCISE 6 ##
################

# Scatter Plot for both given classes
colors = [None, "blue", "red"]
for x_p, cl in zip(x, y):
    plt.scatter(x_p[0], x_p[1], color=colors[cl])
plt.show()

################
## EXERCISE 7 ##
################

# Test some weights for the linear classifier
w = [1.0, -1.0]
b = 0.0

# The linear clasifier
def linear_classifier(w, b, x_p):
    c = w[0] * x_p[0] + w[1] * x_p[1]
    if c > b:
        pred = 1
    else:
        pred = 2
    return pred

# Test the above defined linear classifier for 5 random samples
idx = np.random.randint(0, x.shape[0], 5)
xs = x[idx]
ys = y[idx]
ims = imgs[idx]
for i, x_p in enumerate(xs):
    pred = linear_classifier(w, b, x_p)
    print("Prediction: ", pred)
    print("Class:", ys[i])

################
## EXERCISE 8 ##
################

# Adjusted weights for a better performance
w = [2.0, -1.0]
b = 0.0
def predictions(x):
    predictions = np.array([linear_classifier(w, b, x_p) for x_p in x])
    return predictions
    
predictions = predictions(x)

# Scatter plot for right (green)/false (red) classifications
for i in range(x.shape[0]):
    if predictions[i] == y[i]:
        plt.scatter(x[i][0], x[i][1], color="green")
    else:
        plt.scatter(x[i][0], x[i][1], color="red")
plt.show()

################
## EXERCISE 9 ##
################

# Compute the accuracy for the given dataset
def accuracy():
    right_classification = 0
    for pred, cl in zip(predictions, y):
        if pred == cl:
            right_classification += 1
    accuracy = right_classification / imgs.shape[0]
    return accuracy
    
acc = accuracy()
print("Acc: ", acc)

################
## EXERCISE 9 ##
################

# Get random sample and check if its false classified
found = False
while not found:
    idx = np.random.randint(0, x.shape[0])
    x_p, img, y_p = x[idx], imgs[idx], y[idx]
    pred = predictions[idx]
    if pred != y_p:
        found = True

# Show the given image with the true label and the prediction
plt.imshow(img, cmap="gray")
plt.title(str(y_p) + " != "+ str(pred))
plt.show()