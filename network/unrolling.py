import numpy as np
from optics.cassi import *
from scipy.io import loadmat
from tensorflow.keras.constraints import NonNeg
import tensorflow as K  # se puede cambiar por from keras.import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from network.prior_networks import *
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model


class Unrolling(Layer):
    def __init__(self, input_dim=(128, 128, 25), prior='unet', type_unr = 'hqs',stages=10):
        super(Unrolling, self).__init__()

        self.input_dim = input_dim
        self.prior = prior
        self.type_unr = type_unr
        self.stages = stages
        self.iteration = []
        if type_unr == 'hqs':

            for i in range(stages):
                self.iteration.append(HQS_Update(input_dim=input_dim,rho_initial=0.1,alpha_initial=0.1,prior=prior,kernel_size=3))



    def call(self,inputs):

        [X, y, F, T,H] = inputs
        Xt = [X]
        for i in range(self.stages):
            X = self.iteration[i]([X, y, F, T,H])
            Xt.append(X)
        return Xt




class HQS_Update(Layer):
    def __init__(self, input_dim=(128, 128, 25), name='HQS_update', rho_initial=0.1, alpha_initial=0.1, prior='unet',
                 kernel_size=3, **kwargs):
        super(HQS_Update, self).__init__(name=name, **kwargs)
        self.input_dim = input_dim

        if prior == 'hssp':
            self.prior = hssp_prior(input_size=input_dim, num_filters=20, trainable=True)
        if prior == 'unet':
            self.prior = deep_prior_unet(input_dim=input_dim, Bands=input_dim[-1], Kernels_Size=(3, 3), num_filters=20,
                                         trainable=True)
        if prior == 'high_res':
            self.prior = prior_highres(input_dim=input_dim, Bands=input_dim[-1], Kernels_Size=(3, 3), num_filters=20,
                                       trainable=True)

        self.rho_initial = rho_initial
        self.alpha_initial = alpha_initial
        self.Grad = Gradient(input_size=input_dim)

    def build(self, input_shape):
        rho_init = tf.keras.initializers.Constant(self.rho_initial)
        self.rho = self.add_weight(name='rho', trainable=True, constraint=NonNeg(), initializer=rho_init)

        alpha_init = tf.keras.initializers.Constant(self.alpha_initial)
        self.alpha = self.add_weight(name='alpha', trainable=True, constraint=NonNeg(), initializer=alpha_init)
        super(HQS_Update, self).build(input_shape)

    def call(self, inputs):
        [X, y, F, T, H] = inputs
        Xn = X - self.alpha * (self.Grad([X, y, F, T, H]) + self.rho * (X - self.prior(X)))
        return Xn


class Gradient(Layer):
    def __init__(self, input_size=(128, 128, 25), arch='cassi', shots=4, name='Grad_cassi', **kwargs):
        super(Gradient, self).__init__(name=name, **kwargs)
        self.input_size = input_size


    def call(self, inputs):
        [X, y, F, T,H] = inputs

        yk,_ = F(X)

        res = yk - y
        Xk = T([res,H])

        return Xk

