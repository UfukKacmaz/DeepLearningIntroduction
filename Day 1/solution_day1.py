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
# imgs_50 = np.array([img for img, label in zip(imgs, labels) if label == 2], dtype=np.uint8)
# labels_50 = np.array([2 for _ in range(imgs_50.shape[0])], dtype=np.uint8)

# indx = np.random.randint(0, imgs_50.shape[0], 5)
# i_s = imgs_50[indx]
# l_s = labels_50[indx]
 
# for i, l in zip(i_s, l_s):
#     plt.imshow(i)
#     plt.title(l)
#     plt.show()

# imgs_30 = np.array([img for img, label in zip(imgs, labels) if label == 1], dtype=np.uint8)
# labels_30 = np.array([1 for _ in range(imgs_50.shape[0])], dtype=np.uint8)

# indx = np.random.randint(0, imgs_30.shape[0], 5)
# i_s = imgs_30[indx]
# l_s = labels_30[indx]
 
# for i, l in zip(i_s, l_s):
#     plt.imshow(i)
#     plt.title(l)
#     plt.show()

################
## EXERCISE 3 ##
################

# Convert images to uint8 and to grayscale
imgs = imgs.astype(np.uint8)
imgs = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in imgs], dtype=np.uint8)

################
## EXERCISE 4 ##
################

# Get index image for 3
img = imgs[7]
croped_img_30 = img[13:23, 11:16]
# Get index image for 5
img = imgs[60]
croped_img_50 = img[11:21, 10:15]

################
## EXERCISE 5 ##
################

def convolution(image, conv_filter):
    filter_h, filter_w = conv_filter.shape
    image_h, image_w = image.shape
    # Image padding for convolution
    image = np.pad(image, pad_width=((0, filter_h), (0, filter_w)), mode="constant", constant_values=0.0)
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

def extract_features(croped_img_30, croped_img_50):
    x = np.empty(shape=(imgs.shape[0], 2))
    for i, img in enumerate(imgs):
        feature_30, feature_30_pos,_ = convolution(img, croped_img_30)
        feature_50, feature_50_pos,_ = convolution(img, croped_img_50)
        x[i] = [feature_30, feature_50]
        # fig, ax = plt.subplots(1)
        # ax.imshow(img, cmap="gray")
        # rect1 = patches.Rectangle((feature_30_pos[0], feature_30_pos[1]), 
        #     6, 10, linewidth=1, edgecolor='b', facecolor='none')
        # rect2 = patches.Rectangle((feature_50_pos[0], feature_50_pos[1]), 
        #     6, 10, linewidth=1, edgecolor='r', facecolor='none')
        # ax.add_patch(rect1)
        # ax.add_patch(rect2)
        # plt.legend(["30", "50"])
        # plt.show()
    y = labels
    np.save("x.npy", x)
    np.save("y.npy", y)

extract_features(croped_img_30, croped_img_50)
x = np.load("x.npy")
y = np.load("y.npy")

################
## EXERCISE 6 ##
################

colors = [None, "blue", "red"]
for x_p, cl in zip(x, y):
    plt.scatter(x_p[0], x_p[1], color=colors[cl])
plt.show()

################
## EXERCISE 7 ##
################

w = [2.0, -1.0]
print(w)
b = 0.0

# indx = np.random.randint(0, 100, 30)
# xs = x[indx]
# ys = y[indx]
# ims = imgs[indx]
# for i, x_p in enumerate(xs):
#     c = w[0] * x_p[0] + w[1] * x_p[1]
#     if c > b:
#         pred = 1
#     else:
#         pred = 2
#     print("Prediction: ", pred)
#     print("Clas::", ys[i])

################
## EXERCISE 8 ##
################

def predict(x_p):
    w = [2.0, -1.0]
    b = 0.0
    c = w[0] * x_p[0] + w[1] * x_p[1]
    if c > b:
        prediction = 1
    else:
        prediction = 2
    return prediction

def predictions(x):
    predictions = np.array([predict(x_p) for x_p in x])
    return predictions
    
predictions = predictions(x)

for i in range(x.shape[0]):
    if predictions[i] == y[i]:
        plt.scatter(x[i][0], x[i][1], color="green")
    else:
        plt.scatter(x[i][0], x[i][1], color="red")
plt.show()

################
## EXERCISE 9 ##
################

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

x_p, img, y_p = x[99], imgs[99], y[99]
pred = predict(x_p)
plt.imshow(img, cmap="gray")
plt.title(str(y_p) + " != "+ str(pred))
plt.show()
