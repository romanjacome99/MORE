
import tensorflow as tf
import numpy as np 
from tensorflow.keras.layers import Layer
import numpy as np
from regularizers import implementation
class ForwardSPC(Layer):

    def __init__(self, output_dim=(28, 28, 1), input_dim=(28, 28, 1), compression=0.2,bin_param=0.03,opt_H = True,**kwargs):
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.bin_param = bin_param
        self.compression = compression
        self.my_regularizer = implementation.Reg_Binary_1_1(bin_param)
        self.opt_H = opt_H
        super(ForwardSPC, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'output_dim': self.output_dim,
            'input_dim': self.input_dim,
            'compression': self.compression,
        })
        return config

    def build(self, input_shape):

        M = round((self.input_dim[0] * self.input_dim[1] * self.input_dim[2]) * self.compression)
        if self.opt_H:

            H_init = np.random.normal(0, 1, (1, self.input_dim[0], self.input_dim[0], 1, M)) / np.sqrt(
                self.input_dim[0] * self.input_dim[0])
            H_init = tf.constant_initializer(H_init)

            self.H = self.add_weight(name='H', shape=(1, self.input_dim[0], self.input_dim[0],1, M),
                                    initializer=H_init, trainable=True,regularizer=self.my_regularizer)
        else:
            H_init = np.random.randint(0, 2, (1, self.input_dim[0], self.input_dim[0],1, M))*2 - 1
            H_init = tf.constant_initializer(H_init)

            self.H = self.add_weight(name='H', shape=(1, self.input_dim[0], self.input_dim[0],1, M),
                                    initializer=H_init, trainable=False)
        super(ForwardSPC, self).build(input_shape)

    def call(self, inputs, **kwargs):

        if self.opt_H:
            H = self.H
            x = inputs
        else:
            H = inputs[1]
            x = inputs[0]

        y = tf.reduce_sum(tf.reduce_sum(tf.multiply(tf.expand_dims(x,-1), H), axis=1), axis=1)
        y = y / tf.reduce_max(y)
        print('y',y.shape)

        if self.opt_H:
            bn_reg = tf.reduce_sum(tf.multiply(tf.square(1+H), tf.square(1 - H)))

            self.add_metric(bn_reg, 'bin_regularizer')



        return y, H

    def get_config(self):
        return {'bin_param': float(tf.keras.backend.get_value(self.bin_param))}
    def update_reg_param(self,new_param):
        self.reg_param.assign(new_param)


class TransposeSPC(Layer):

    def __init__(self, output_dim=(28, 28, 1), input_dim=(28, 28, 1), **kwargs):
        self.output_dim = output_dim
        self.input_dim = input_dim
        super(TransposeSPC, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'output_dim': self.output_dim,
            'input_dim': self.input_dim,
        })
        return config


    def call(self, inputs):

        H = inputs[1]
        y = inputs[0]

        yd = tf.expand_dims(tf.expand_dims(y, -1), -1)
        print('y',yd.shape)

        H_t = tf.transpose(H, [0, 4, 1, 2,3])
        x = tf.reduce_sum(tf.multiply(yd, H_t), axis=1)
        x = tf.expand_dims(x, -1)
        x = x/tf.math.reduce_max(x)
        print(x.shape)
        return x


class SinglePixel(Layer):

    def __init__(self, output_dim=(28, 28, 1), input_dim=(28, 28, 1), compression=0.2, bin_param=0.03, opt_H=True, name='SPC',
                 **kwargs):
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.bin_param = bin_param
        self.compression = compression
        self.my_regularizer = implementation.Reg_Binary_1_1(bin_param)
        self.opt_H = opt_H
        super(SinglePixel, self).__init__(name=name,**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'output_dim': self.output_dim,
            'input_dim': self.input_dim,
            'compression': self.compression,
        })
        return config

    def build(self, input_shape):

        M = round((self.input_dim[0] * self.input_dim[1] * self.input_dim[2]) * self.compression)
        if self.opt_H:

            H_init = np.random.normal(0, 1, (1, self.input_dim[0], self.input_dim[0], M)) / np.sqrt(
                self.input_dim[0] * self.input_dim[0])
            H_init = tf.constant_initializer(H_init)

            self.H = self.add_weight(name='H', shape=(1, self.input_dim[0], self.input_dim[0], M),
                                     initializer=H_init, trainable=True, regularizer=self.my_regularizer)
        else:
            H_init = np.random.randint(0, 2, (1, self.input_dim[0], self.input_dim[0], M)) * 2 - 1
            H_init = tf.constant_initializer(H_init)

            self.H = self.add_weight(name='H', shape=(1, self.input_dim[0], self.input_dim[0], M),
                                     initializer=H_init, trainable=False)
        super(SinglePixel, self).build(input_shape)

    def call(self, inputs, **kwargs):

        H = self.H

        y = tf.reduce_sum(tf.multiply(inputs, H),axis=1)

        y = tf.reduce_sum(y, axis=1)
        y = y / tf.reduce_max(y)
        yd = tf.expand_dims(tf.expand_dims(y, -1), -1)
        H_t = tf.transpose(H, [0, 3, 1, 2])
        x = tf.multiply(yd, H_t)
        print(x.shape)

        x = tf.reduce_sum(x, axis=1,keepdims=False)
        print(x.shape)
        x = tf.expand_dims(x, -1)
        x = x / tf.math.reduce_max(x)
        bn_reg = tf.reduce_sum(tf.multiply(tf.square(1 + H), tf.square(1 - H)))
        self.add_metric(bn_reg, 'bin_regularizer')

        return x, y, H

    def get_config(self):
        return {'bin_param': float(tf.keras.backend.get_value(self.bin_param))}

    def update_reg_param(self, new_param):
        self.reg_param.assign(new_param)
