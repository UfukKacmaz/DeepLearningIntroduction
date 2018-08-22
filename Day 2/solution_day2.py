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
classes = [2, 12, 13]
[imgs, labels, class_descs, sign_ids] = load_gtsrb_images(dataset_path, classes, 150)  
imgs = imgs.astype(np.uint8)

# plt.imshow(imgs[0])
# plt.show()

################
## EXERCISE 2 ##
################

imgs = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in imgs])

################
## EXERCISE 3 ##
################

# Compute HOG Features for given image
def hog_image(img, cell_size, block_size, nbins):
    # winSize is the size of the image cropped to an multiple of the cell size
    hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1],
                                    img.shape[0] // cell_size[0] * cell_size[0]),
                            _blockSize=(block_size[1] * cell_size[1],
                                        block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=nbins)
    n_cells = (img.shape[0] // cell_size[0], img.shape[1] // cell_size[1])
    hog_feats = hog.compute(img).reshape(n_cells[1] - block_size[1] + 1, n_cells[0] - block_size[0] + 1,
                            block_size[0], block_size[1], nbins).transpose((1, 0, 2, 3, 4))
    # hog_feats now contains the gradient amplitudes for each direction,
    # for each cell of its group for each group. Indexing is by rows then columns.
    gradients = np.zeros((n_cells[0], n_cells[1], nbins))
    # count cells (border cells appear less often across overlapping groups)
    cell_count = np.full((n_cells[0], n_cells[1], 1), 0, dtype=int)
    for off_y in range(block_size[0]):
        for off_x in range(block_size[1]):
            gradients[off_y:n_cells[0] - block_size[0] + off_y + 1,
                    off_x:n_cells[1] - block_size[1] + off_x + 1] += hog_feats[:, :, off_y, off_x, :]
            cell_count[off_y:n_cells[0] - block_size[0] + off_y + 1,
                    off_x:n_cells[1] - block_size[1] + off_x + 1] += 1
    # Average gradients
    gradients /= cell_count
    return gradients

def extract_hog_features(imgs, cell_size, block_size, nbins):
    # Save HOG Features for every image
    bin = 3
    x = np.empty(shape=(imgs.shape[0], imgs[0].shape[0] // cell_size[0] * imgs[0].shape[1] // cell_size[1] * nbins))
    for i, img in enumerate(imgs):
        gradients = hog_image(img, cell_size, block_size, nbins)
        x[i] = gradients.ravel()
    return x, labels

cell_size = (8, 8)
block_size = (4, 4)
nbins = 9
# x, y = extract_hog_features(imgs, cell_size, block_size, nbins)
# np.save(dataset_path+"x_hog.npy", x)
# np.save(dataset_path+"y_hog.npy", labels)
x = np.load(dataset_path+"x_hog.npy")
y = np.load(dataset_path+"y_hog.npy")

################
## EXERCISE 4 ##
################

from sklearn.decomposition import PCA

def perform_pca(x):
    pca = PCA(n_components=2)
    pca.fit(x)
    x_transformed = pca.transform(x)
    return x_transformed

# x_transformed = perform_pca(x)
# np.save(dataset_path+"x_hog_pca.npy", x_transformed)
# x_transformed = np.load(dataset_path+"x_hog_pca.npy")

################
## EXERCISE 5 ##
################

# colors = {2: "red", 12: "blue", 13: "green"}
# for x_p, y_p in zip(x_transformed, y):
#     plt.scatter(x_p[0], x_p[1], color=colors[y_p])
# ax = plt.gca()
# ax.legend(["Speed limit 50", "Right of way on this street", "Yield way"])
# leg = ax.get_legend()
# leg.legendHandles[0].set_color('red')
# leg.legendHandles[1].set_color('blue')
# leg.legendHandles[2].set_color('green')
# plt.show()

################
## EXERCISE 6 ##
################

import matplotlib
from matplotlib import colors as mcolors
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

def get_train_valid_test(x, y):
    test_size = 0.3
    valid_size = 0.2 / (1 - test_size)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=42)
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train, y_train, test_size=valid_size, random_state=42)
    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)

num_classes = 3
(x_train, y_train), (x_valid, y_valid), (x_test, y_test) = get_train_valid_test(x, y)
parameters = {
    'kernel':('poly', 'linear', 'rbf'), 
    'C':[0.5, 1, 5, 10], 
    'gamma': ["auto", 1]
    }

# print("\n\nTRAINING FOR ALL CLASSES!")
# svc = SVC(decision_function_shape="ovr")
# clf = GridSearchCV(svc, parameters)
# clf.fit(x_train, y_train)

# print("VALID FOR ALL CLASSES!")
# print("Best parameters set found on development set:")
# print(clf.best_params_)
# y_valid, y_pred = y_valid, clf.predict(x_valid)
# print("Validation acc: ", clf.score(x_valid, y_valid))
# print(confusion_matrix(y_valid, y_pred, labels=range(num_classes)))

print("TESTING FOR ALL CLASSES!")
best_c = 0.5
best_gamma = "auto"
best_kernel = "linear"
clf = SVC(decision_function_shape="ovr", C=best_c, gamma=best_gamma, kernel=best_kernel)
clf.fit(x_train, y_train)
y_test, y_pred = y_test, clf.predict(x_test)
print("Validation acc: ", clf.score(x_test, y_test))
print(confusion_matrix(y_test, y_pred, labels=range(num_classes)))

################
## EXERCISE 7 ##
################

# Load full dataset
num_classes = 43
classes = range(num_classes)
[imgs, labels, class_descs, sign_ids] = load_gtsrb_images(dataset_path, classes, 500)  
imgs = imgs.astype(np.uint8)
imgs = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in imgs])

# Compute hog features for all classes
cell_size = (8, 8)
block_size = (4, 4)
nbins = 9
#x, y = extract_hog_features(imgs, cell_size, block_size, nbins)
#np.save(dataset_path+"x_hog_full.npy", x)
#np.save(dataset_path+"y_hog_full.npy", labels)
x = np.load(dataset_path+"x_hog_full.npy")
y = np.load(dataset_path+"y_hog_full.npy")

# Compute PCA for hog features for all classes
#x_transformed = perform_pca(x)
#np.save(dataset_path+"x_hog_pca_full.npy", x_transformed)
x_transformed = np.load(dataset_path+"x_hog_pca_full.npy")

# Plot PCA results
# cmap = matplotlib.cm.get_cmap('Spectral')
# colors = {i: cmap(i*(1/num_classes)) for i in range(num_classes)}
# for x_p, y_p in zip(x_transformed, y):
#     plt.scatter(x_p[0], x_p[1], color=colors[y_p])
# plt.show()

# Split Dataset and compute SVM classification
(x_train, y_train), (x_valid, y_valid), (x_test, y_test) = get_train_valid_test(x, y)
parameters = {
    'kernel':('poly', 'linear', 'rbf'), 
    'C':[0.5, 1, 5, 10], 
    'gamma': ["auto", 1]
    }

# print("\n\nTRAINING FOR ALL CLASSES!")
# svc = SVC(decision_function_shape="ovr")
# clf = GridSearchCV(svc, parameters, n_jobs=-1)
# clf.fit(x_train, y_train)

# print("VALID FOR ALL CLASSES!")
# print("Best parameters set found on development set:")
# print(clf.best_params_)
# y_valid, y_pred = y_valid, clf.predict(x_valid)
# print("Validation acc: ", clf.score(x_valid, y_valid))
# print(confusion_matrix(y_valid, y_pred, labels=range(num_classes)))

print("TESTING FOR ALL CLASSES!")
best_c = 10
best_gamma = 1.0
best_kernel = "rbf"
clf = SVC(decision_function_shape="ovr", C=best_c, gamma=best_gamma, kernel=best_kernel)
clf.fit(x_train, y_train)
y_test, y_pred = y_test, clf.predict(x_test)
print("Testing acc: ", clf.score(x_test, y_test))
print(confusion_matrix(y_test, y_pred, labels=range(num_classes)))

################
## EXERCISE 8 ##
################

rand_indx = [i for i in range(x_test.shape[0]) if y_pred[i] != y_test[i]]
if len(rand_indx) > 0:
    img = imgs[rand_indx[0]]
    plt.imshow(img)
    title = "Target: " + str(y_test[rand_indx[0]]) + " - Pred: " + str(y_pred[rand_indx[0]]) 
    plt.title(title)
    plt.show()