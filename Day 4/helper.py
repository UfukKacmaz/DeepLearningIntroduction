import numpy as np
import matplotlib.pyplot as plt
from load_gtsrb import *
import tensorflow as tf

def occlusion_plot(prediction_heatmap):
    cMap = "Spectral"
    #cMap = "RdYlBu"
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(prediction_heatmap, cmap=cMap)
    cbar = plt.colorbar(heatmap)
    plt.show()

def occlusion(img, label, box_size, sess, pred_op, x):
    prediction_heatmap = np.empty(
        shape=(img.shape[0], img.shape[1]), dtype=np.float32)
    #print(box_size//2, img.shape[0]+box_size//2, box_size//2, img.shape[1]+box_size//2)
    padded_img = np.pad(
        img, ((box_size//2, box_size//2), (box_size//2, box_size//2), (0, 0)), mode="constant", constant_values=0.0)
    #print((img.shape[0] + box_size//2))
    #print((img.shape[0]+ box_size//2))
    for y_coordinates in range(img.shape[0]):
        for x_coordinates in range(img.shape[1]):
            padded_img[box_size // 2 + y_coordinates][box_size // 2 + x_coordinates] = img[y_coordinates][x_coordinates]

    #padded_img[(box_size//2):(img.shape[0] + box_size//2)][(box_size//2):(img.shape[1]+box_size//2)] = img
    #plt.imshow(img)
    #plt.show()
    gray_box = np.full((box_size,box_size,3), 100)
    for i in range(prediction_heatmap.shape[0]):
        for j in range(prediction_heatmap.shape[0]):
            img_s = padded_img.copy()
            #print(img_s.shape)
            #img_s[i:i+box_size, j:j+box_size] = gray_box
            #img_s = img_s[box_size//2:img.shape[0]+box_size//2][box_size//2:img.shape[1]+box_size//2]
            for y_coordinates in range(gray_box.shape[0]):
                for x_coordinates in range(gray_box.shape[1]):
                        img_s[y_coordinates // 2 + i, x_coordinates // 2 + j] = gray_box[y_coordinates][x_coordinates]

            input = img.copy()
            for y_coordinates in range(img.shape[0]):
                for x_coordinates in range(img.shape[1]):
                    #img_s[y_coordinates][x_coordinates] = img_s[box_size // 2 + y_coordinates][box_size // 2 + x_coordinates]
                    input[y_coordinates][x_coordinates] = img_s[box_size // 2 + y_coordinates][box_size // 2 + x_coordinates]
                    #input

            #print(img_s.shape)
            input = [input]

            #input = img_s.reshape(-1, img_s.shape[0], img_s.shape[1], img_s.shape[2])
            x_p = sess.run([pred_op], feed_dict={x: input})
            x_p = np.reshape(x_p, (label.shape[0]))
            print("np.exp(x_p): " + str(np.exp(x_p)))
            print("np.sum(np.exp(x_p), axis=0):" + str(np.sum(np.exp(x_p), axis=0)))
            x_p = np.exp(x_p) / np.sum(np.exp(x_p), axis=0)
            prediction_heatmap[i][j] = x_p[np.argmax(label)]
    #print(prediction_heatmap)
    occlusion_plot(prediction_heatmap)

# Display the convergence of the errors
def display_convergence_error(train_error, valid_error):
    plt.plot(range(1, len(train_error)+1), train_error, color="red")
    plt.plot(range(1, len(valid_error)+1), valid_error, color="blue")
    plt.legend(["Train", "valid"])
    plt.title('Error of the NN')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.show()

# Display the convergence of the accs
def display_convergence_acc(train_acc, valid_acc):
    plt.plot(range(1, len(train_acc)+1), train_acc, color="red")
    plt.plot(range(1, len(valid_acc)+1), valid_acc, color="blue")
    plt.legend(["Train", "valid"])
    plt.title('Accs of the NN')
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.show()