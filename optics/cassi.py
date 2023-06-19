import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer
import numpy as np
from regularizers import implementation



class ForwardCASSI(Layer):
    def __init__(self, input_dim=(512, 512, 31), noise=False, bin_param=0.5, opt_H=True,
                 name=False, shots=1, batch_size=1, snr=30, **kwargs):

        self.input_dim = input_dim
        self.noise = noise
        self.shots = shots
        self.snr = snr
        self.batch_size = batch_size
        self.L = int(input_dim[-1])
        self.my_regularizer = implementation.Reg_Binary_0_1(bin_param)
        self.M = int(input_dim[0])
        self.opt_H = opt_H
        super(ForwardCASSI, self).__init__(name=name, **kwargs)

    def build(self, input_shape):

        if self.opt_H:
            print('Trainable CASSI')

            H_init = np.random.normal(0,1, (1, self.M, self.M, 1,
                                              self.shots))
            H_init = tf.constant_initializer(H_init)
            self.H = self.add_weight(name='H', shape=(1, self.M, self.M, 1, self.shots),
                                     initializer=H_init, trainable=True, regularizer=self.my_regularizer)

    def call(self, inputs, **kwargs):

        X = inputs
        H = self.H

        if len(H.shape) < 5:
            H = tf.expand_dims(H, -1)
        if len(X.shape) <5:
            X = tf.expand_dims(X,-1)
        L = self.L
        M = self.M
        # CASSI Sensing Model

        Aux1 = tf.multiply(H,X)
        Aux1 = tf.pad(Aux1, [[0, 0], [0, 0], [0, L - 1], [0, 0], [0, 0]])
        Y = None
        for i in range(L):
            Tempo = tf.roll(Aux1, shift=i, axis=2)
            if Y is not None:
                Y = tf.concat([Y, tf.expand_dims(Tempo[:, :, :, i], -1)], axis=4)
            else:
                Y = tf.expand_dims(Tempo[:, :, :, i], -1)
        Y = tf.reduce_sum(Y, 4)
        if self.noise:
            sigm = tf.reduce_sum(tf.math.pow(Y, 2)) / ((M * (M + (L - 1)) * self.batch_size) * 10 ** (self.noise / 10))
            Y = Y + tf.random.normal(shape=(self.batch_size, M, M + L - 1, 1), mean=0, stddev=tf.math.sqrt(sigm)
                                     , dtype=Y.dtype)
        Y = Y/tf.reduce_max(Y)
        # CASSI Transpose model (x = H'*y)


        if self.opt_H:
            bn_reg = tf.reduce_sum(tf.multiply(tf.square(H), tf.square(1 - H)))
            self.add_metric(bn_reg, 'bin_regularizer')

        return Y, H



class ForwardCASSIReal(Layer):
    def __init__(self, input_dim=(512, 512, 31), noise=False, H=None, bin_param=0.5, opt_H=True,
                 name=False, shots=1, batch_size=1, snr=30, **kwargs):

        self.input_dim = input_dim
        self.noise = noise
        self.shots = shots
        self.snr = snr
        self.batch_size = batch_size
        self.L = int(input_dim[-1])
        self.M = int(input_dim[0])
        self.H = H
        super(ForwardCASSIReal, self).__init__(name=name, **kwargs)


    def call(self, inputs, **kwargs):

        X = inputs
        H = self.H

        if len(H.shape) < 5:
            H = tf.expand_dims(H, -1)
        if len(X.shape) <5:
            X = tf.expand_dims(X,-1)
        L = self.L
        M = self.M
        # CASSI Sensing Model

        Aux1 = tf.multiply(H,X)
        Aux1 = tf.pad(Aux1, [[0, 0], [0, 0], [0, L - 1], [0, 0], [0, 0]])
        Y = None
        for i in range(L):
            Tempo = tf.roll(Aux1, shift=i, axis=2)
            if Y is not None:
                Y = tf.concat([Y, tf.expand_dims(Tempo[:, :, :, i], -1)], axis=4)
            else:
                Y = tf.expand_dims(Tempo[:, :, :, i], -1)
        Y = tf.reduce_sum(Y, 4)
        if self.noise:
            sigm = tf.reduce_sum(tf.math.pow(Y, 2)) / ((M * (M + (L - 1)) * self.batch_size) * 10 ** (self.noise / 10))
            Y = Y + tf.random.normal(shape=(self.batch_size, M, M + L - 1, 1), mean=0, stddev=tf.math.sqrt(sigm)
                                     , dtype=Y.dtype)
        Y = Y/tf.reduce_max(Y)

        return Y, H



class TransposeCASSI(Layer):
    def __init__(self, input_dim=(512, 512, 31), noise=False, bin_param=0.5, opt_H=True,
                 name=False, shots=1, batch_size=1, snr=30, **kwargs):

        self.input_dim = input_dim
        self.noise = noise
        self.shots = shots
        self.snr = snr
        self.batch_size = batch_size
        self.L = int(input_dim[-1])
        self.M = int(input_dim[0])
        self.opt_H = opt_H
        super(TransposeCASSI, self).__init__(name=name, **kwargs)

    def build(self, input_shape):

        super(TransposeCASSI, self).build(input_shape)

    def call(self, inputs, **kwargs):

        L = self.input_dim[-1]
        M = self.input_dim[0]

        [Y, H] = inputs
        if len(H.shape) < 5:
            H = tf.expand_dims(H, -1)

        X = None
        for i in range(L):
            Tempo = tf.roll(Y, shift=-i, axis=2)
            if X is not None:
                X = tf.concat([X, tf.expand_dims(Tempo[:, :, 0:M], -1)], axis=4)
            else:
                X = tf.expand_dims(Tempo[:, :, 0:M], -1)

        X = tf.transpose(X, [0, 1, 2, 4, 3])
        X = tf.multiply(H, X)
        X = tf.reduce_sum(X, 4)

        X = X / tf.math.reduce_max(X)
        return X
