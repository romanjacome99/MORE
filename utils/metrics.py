import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
def psnr(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=K.max(y_true))

def psnr_cs(y_true, y_pred):
    print(y_true.shape,y_pred.shape)
    y_true = tf.reshape(y_true,[-1,32,32])
    y_pred = tf.reshape(y_pred, [-1, 32, 32])
    return tf.image.psnr(y_true, y_pred, max_val=K.max(y_true))
def cos_distance(y_true, y_pred):
    def l2_normalize(x, axis):
        norm = K.sqrt(K.sum(K.square(x), axis=axis, keepdims=True))
        return K.maximum(x, K.epsilon()) / K.maximum(norm, K.epsilon())

    y_true = l2_normalize(y_true, axis=-1)
    y_pred = l2_normalize(y_pred, axis=-1)
    return K.mean(K.sum(y_true * y_pred, axis=-1))
def relRMSE(y_true,y_pred):
    true_norm = K.sqrt(K.sum(K.square(y_true), axis=-1))
    return K.mean(K.sqrt(tf.keras.losses.mean_squared_error(y_true, y_pred))/true_norm)
def SSIM(y_true,y_pred):
    return tf.image.ssim(y_pred,y_true,K.max(y_true))
