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
[imgs, labels, class_descs, sign_ids] = load_gtsrb_images(dataset_path, classes, 50)  
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

def extract_hog_features():
    # Save HOG Features for every image
    cell_size = (8, 8)
    block_size = (2, 2)
    nbins = 9
    bin = 3
    x = np.empty(shape=(imgs.shape[0], cell_size[0]//block_size[0] * cell_size[1]//block_size[1] * nbins))
    for i, img in enumerate(imgs):
        gradients = hog_image(img, cell_size, block_size, nbins)
        x[i] = gradients.ravel()
    np.save(dataset_path+"x_hog.npy", x)
    np.save(dataset_path+"y_hog.npy", labels)

extract_hog_features()
x = np.load(dataset_path+"x_hog.npy")
y = np.load(dataset_path+"y_hog.npy")

################
## EXERCISE 4 ##
################

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(x)
x_transformed = pca.transform(x)

colors = {2: "red", 12: "blue", 13: "green"}
for x_p, y_p in zip(x_transformed, y):
    plt.scatter(x_p[0], x_p[1], color=colors[y_p])
ax = plt.gca()
ax.legend(["Speed limit 50", "Right of way on this street", "Yield way"])
leg = ax.get_legend()
leg.legendHandles[0].set_color('red')
leg.legendHandles[1].set_color('blue')
leg.legendHandles[2].set_color('green')
plt.show()