import numpy as np
import matplotlib.pyplot as plt
from load_gtsrb import *
import tensorflow as tf

def occlusion_plot(prediction_image):
    cMap = "coolwarm"
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(prediction_image, cmap=cMap)
    cbar = plt.colorbar(heatmap)
    plt.show()

# Display the convergence of the errors
def display_convergence_error(train_error, valid_error):
    plt.plot(train_error, color="red")
    plt.plot(valid_error, color="blue")
    plt.legend(["Train", "valid"])
    plt.title('Error of the NN')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.show()

# Display the convergence of the accs
def display_convergence_acc(train_acc, valid_acc):
    plt.plot(train_acc, color="red")
    plt.plot(valid_acc, color="blue")
    plt.legend(["Train", "valid"])
    plt.title('Accs of the NN')
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.show()