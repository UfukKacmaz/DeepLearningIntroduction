import numpy as np
import matplotlib.pyplot as plt
from load_gtsrb import *
import tensorflow as tf
from layers import *
from skimage import transform
import matplotlib.cm

def occlusion_plot(occlusion_map, img):
    cMap = "Spectral"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), subplot_kw={'aspect': 1})
    heatmap = ax2.pcolor(occlusion_map, cmap=cMap)
    cbar = plt.colorbar(heatmap)
    ax1.imshow(img/255.0)
    plt.show()

def heatmap_plot(heat_map, img):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), subplot_kw={'aspect': 1})
    heatmap = ax2.pcolor(heat_map, cmap=plt.cm.jet)
    cbar = plt.colorbar(heatmap)
    ax1.imshow(img/255.0)
    plt.show()

def get_occlusion(img, label, box_size, sess, pred_op, x):
    occlusion_map = np.full((img.shape[0], img.shape[1]), 1.0)
    gray_box = np.full((box_size,box_size,3), 100)
    for i in range(img.shape[0]-gray_box.shape[0]):
        for j in range(img.shape[1]-gray_box.shape[1]):
            img_s = img.copy()
            img_s[i:i+gray_box.shape[0], j:j+gray_box.shape[1]] = gray_box
            x_p = sess.run([pred_op], 
                feed_dict={x: img_s.reshape(-1, img_s.shape[0], img_s.shape[1], img_s.shape[2])})[0]
            x_p = sess.run(tf.nn.softmax(x_p))
            x_p = np.reshape(x_p, (label.shape[0]))
            occlusion_map[i+gray_box.shape[0]//2][j+gray_box.shape[1]//2] = x_p[np.argmax(label)]
    occlusion_plot(occlusion_map, img)

def get_heatmap(img, sess, pred_op, heatmaps, x):
    (x, heatmaps) = sess.run([pred_op, heatmaps], 
                feed_dict={x: img.reshape(-1, img.shape[0], img.shape[1], img.shape[2])})
    for heatmap in heatmaps.values():
        # Heatmap from Conv Layer
        heatmap = heatmap.squeeze(axis=(0, 3))
        heatmap = transform.resize(heatmap/255.0, (32, 32))
        heatmap_plot(heatmap, img)

# Display the convergence of the errors
def display_convergence_error(train_error, valid_error):
    plt.plot(range(1, len(train_error)+1), train_error, color="red")
    plt.plot(range(1, len(valid_error)+1), valid_error, color="blue")
    plt.legend(["Train", "valid"])
    plt.title('Errors of the NN')
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

def get_model_params():
    gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    return {gvar.op.name: value for gvar, value in zip(gvars, tf.get_default_session().run(gvars))}

def restore_model_params(model_params):
    gvar_names = list(model_params.keys())
    assign_ops = {gvar_name: tf.get_default_graph().get_operation_by_name(gvar_name + "/Assign")
                  for gvar_name in gvar_names}
    init_values = {gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()}
    feed_dict = {init_values[gvar_name]: model_params[gvar_name] for gvar_name in gvar_names}
    tf.get_default_session().run(assign_ops, feed_dict=feed_dict)