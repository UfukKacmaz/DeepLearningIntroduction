import numpy as np
import matplotlib.pyplot as plt
from load_gtsrb import *
import tensorflow as tf

def occlusion_plot(prediction_heatmap):
    cMap = "Spectral"
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(prediction_heatmap, cmap=cMap)
    cbar = plt.colorbar(heatmap)
    plt.show()

def occlusion(img, label, box_size, sess, pred_op, x):
    prediction_heatmap = np.empty(shape=(img.shape[0]-box_size+1, img.shape[1]-box_size+1), dtype=np.float32)
    gray_box = np.full((box_size,box_size,3), 100)
    for i in range(img.shape[0]-box_size+1):
        for j in range(img.shape[1]-box_size+1):
            img_s = img.copy()
            img_s[i:box_size+i, j:box_size+j] = gray_box
            x_p = sess.run([pred_op], feed_dict={x: img_s.reshape(-1, img_s.shape[0], img_s.shape[1], img_s.shape[2])})[0]
            x_p = sess.run(tf.nn.softmax(x_p))
            x_p = np.reshape(x_p, (label.shape[0]))
            prediction_heatmap[i][j] = x_p[np.argmax(label)]
    occlusion_plot(prediction_heatmap)

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