import numpy as np
import tensorflow as tf

def lapl_pen_stim(nx, ny, n_basis_stim, lapl_axis, layer):
    if lapl_axis == 0 or (lapl_axis == 'all' and ny==1):
        kernel = tf.constant([1, -2, 1], dtype='float32')
        kernel = kernel[:, tf.newaxis, tf.newaxis, tf.newaxis]
    elif lapl_axis == 1:
        kernel = tf.constant([1, 2, 1], dtype='float32')
        kernel = tf.expand_dims(kernel[:, tf.newaxis, tf.newaxis], 0)
    elif lapl_axis == 'all' and ny>1:
        kernel = tf.constant([[0, 1, 0],[1, -4, 1],[0, 1, 0]], dtype='float32')
        kernel = kernel[:,:,tf.newaxis, tf.newaxis]       
    else:
        raise Exception("lapl_axis must be 0, 1 or 'all'")
    weightsxy = tf.reshape(layer.weights[0], [ny, nx, n_basis_stim])
    weightsxy = tf.transpose(weightsxy, [2, 1, 0])
    weightsxy = weightsxy[:,:,:,tf.newaxis]
    lapl_val = tf.nn.conv2d(weightsxy, kernel, strides = 1, padding='SAME')
    loss_val = tf.reduce_sum(lapl_val**2)
    return loss_val

def lapl_pen_weights(layer):
    kernel = tf.constant([1, -2, 1], dtype='float32')
    kernel = kernel[:, tf.newaxis, tf.newaxis]
    h = layer.weights[0]
    h = h[tf.newaxis,:]
    lapl_val = tf.nn.conv1d(h, kernel, stride = 1, padding='SAME')
    loss_val = tf.reduce_sum(lapl_val**2)
    return loss_val