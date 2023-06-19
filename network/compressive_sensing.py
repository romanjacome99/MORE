from optics.cassi import *
from optics.single_pixel import ForwardSPC, TransposeSPC
from optics.doe.diffractive import *
from network.unet import UNetCompiled, UNetCompiled_Transpose_DOE
from network.unrolling import Unrolling
from regularizers.proposed import *
from regularizers import proposed
from tensorflow.keras.layers import Conv2D, Dense, Input, Add, Subtract, Conv2DTranspose, BatchNormalization, \
    concatenate
from tensorflow.keras.models import Model

import tensorflow as tf


class E2E_CS(tf.keras.Model):
    def __init__(self, m=100, n=1000, type_reg='min-variance', regularization=True, batch_size=100, mean=1, stddev=0.1,
                 param=0.1, max_var=2,opt_H=True):


        super(E2E_CS, self).__init__()
        self.m = m
        self.n = n
        self.type_reg = type_reg
        self.regularization = regularization

        self.encoder = Dense(m, activation=None,use_bias=False,trainable=opt_H)
        self.decoder = Dense(n, activation='relu')
        reg_function = {'low_rank': LowRank(param=param, batch_size=batch_size),
                        'min-variance': MinVariance(param=param),
                        'min-varvariance': MinVarVariance(param=param),
                        'sparsity': Sparsity(param=param),
                        'max-variance': MaxVariance(param=param, max_var=max_var),
                        'kl-gaussian': KLGaussian(mean=mean, stddev=stddev),
                        'kl-laplace': KLLaplacian(mean=mean, stddev=stddev)
                        }
        if self.regularization:
            self.reg = reg_function[type_reg]



    def call(self, inputs):
        print(inputs)
        y = self.encoder(inputs)
        x = self.decoder(y)

        if self.regularization:
            self.reg(y)

        return x




class SensingLayer(tf.keras.layers.Layer):

    def __init__(self, m=1000, n=100,opt_H = True,**kwargs):
        self.m = m
        self.n = n
        self.opt_H = opt_H

        super(SensingLayer, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'm': self.m,
            'n': self.n,
            'opt_H': self.opt_H
        })
        return config

    def build(self, input_shape):


        if self.opt_H:

            H_init = np.random.normal(0, 1, (self.n, self.m))
            H_init = tf.constant_initializer(H_init)

            self.H = self.add_weight(name='H', shape=( self.n, self.m),
                                    initializer=H_init, trainable=True)
        else:
            H_init = np.random.normal(0, 1, (self.n, self.m))

            H_init = tf.constant_initializer(H_init)

            self.H = self.add_weight(name='H', shape=(self.n, self.m),
                                     initializer=H_init, trainable=False)
        super(SensingLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):

        H  = self.H
        y = tf.matmul(inputs,H)
        y = y / tf.reduce_max(y)



        return y, H

    def get_config(self):
        return {'bin_param': float(tf.keras.backend.get_value(self.bin_param))}
    def update_reg_param(self,new_param):
        self.reg_param.assign(new_param)
