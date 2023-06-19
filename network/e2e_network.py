from regularizers import proposed
from tensorflow.keras.layers import Conv2D, Dense, Input, Add, Subtract, Conv2DTranspose, BatchNormalization, \
    concatenate
from tensorflow.keras.models import Model
from optics.cassi import *
from optics.single_pixel import ForwardSPC, TransposeSPC, SinglePixel
from optics.doe.diffractive import *
from network.unet import UNetCompiled, UNetCompiled_Transpose_DOE
from network.unrolling import Unrolling
from regularizers.proposed import *
import tensorflow as tf


class E2E_Network(tf.keras.Model):
    def __init__(self, arch='spc', ct='recovery', input_dim=(128, 128, 25), decoder='unet', compression=0.1, Nterms=15,
                 n_stages=10, shots=1, type_reg='variational', param=0.1, max_var=1.0,
                 noise=True, bin_param=0.5, opt_H=True, batch_size=32, snr=30, epsilon=0.2, sigma=0.2,
                 type_unr='admm', regularization=True, mean=1, stddev=0.1, prior='unet', intermediate_out=False):
        super(E2E_Network, self).__init__()

        self.arch = arch
        self.input_dim = input_dim
        self.param = param
        self.Nterm = Nterms
        self.type_unr = type_unr
        self.regularization = regularization
        self.mean = mean
        self.stddev = stddev
        self.bin_param = bin_param
        self.type_reg = type_reg
        self.batch_size = batch_size
        self.max_var = max_var
        self.decoder = decoder
        self.compression = compression
        self.opt_H = opt_H
        self.noise = noise
        self.n_stages = n_stages
        self.ct = ct
        self.snr = snr

        reg_function = {'low_rank':LowRank(param=param,batch_size=batch_size),
                        'min-variance':MinVariance(param=param),
                        'sparsity': Sparsity(param=param),
                        'min-varvariance': MinVarVariance(param=param),
                        'max-variance': MaxVariance(param=param,max_var=max_var),
                        'kl-gaussian': KLGaussian(mean=mean,stddev=stddev),
                        'kl-laplace': KLLaplacian(mean=mean,stddev=stddev),
                        'graph': GraphRegularized(epsilon=epsilon,sigma=sigma,batch_size=batch_size),
                        'corr': Correlation(param=param,batch_size=batch_size)
                        }
        if self.regularization:
            self.reg = reg_function[type_reg]

        if self.arch == 'spc':
            self.F = SinglePixel(input_dim=input_dim, compression=compression, bin_param=bin_param, opt_H=True,name='optical_encoder')

        elif self.arch == 'cassi':
            self.F = ForwardCASSI(input_dim=input_dim, noise=False, bin_param=bin_param, opt_H=opt_H,
                                  name='forward', shots=1, batch_size=batch_size, snr=30)
            self.T = TransposeCASSI(input_dim=input_dim, noise=False, bin_param=bin_param, opt_H=True,
                                    name='transpose', shots=1, batch_size=batch_size, snr=30)
        elif self.arch == 'doe':
            self.F = ForwardDOE(input_dim=input_dim, train=True, Nterms=Nterms, name='Forward_Model')

            self.T = UNetCompiled(input_size=(input_dim[0],input_dim[0],3), n_filters=32, n_classes=input_dim[-1])

        if self.ct == 'recovery':
            if self.decoder == 'unet':
                self.network = UNetCompiled(input_size=input_dim, n_filters=32, n_classes=input_dim[-1])
            if self.decoder == 'unrolling':
                if self.type_unr == 'hqs':
                    self.network = Unrolling(input_dim=input_dim, prior=prior, type_unr=type_unr, stages=n_stages)

        if ct == 'classification':

            if decoder == 'mobilnetv2':
                self.network = tf.keras.applications.mobilenet_v2.MobileNetV2(
                    input_shape=(input_dim),
                    alpha=1.0,
                    include_top=True,
                    weights=None,
                    pooling=None,
                    classes=10,
                    classifier_activation='softmax')
            if decoder == 'vgg19':
                self.network = tf.keras.applications.VGG19(
                    include_top=True,
                    weights=None,
                    input_shape=None,
                    pooling=None,
                    classes=10,
                    classifier_activation="softmax")
            if decoder == 'resent50':
                self.network = tf.keras.applications.ResNet50(
                    include_top=True,
                    weights=None,
                    input_shape=None,
                    pooling=None,
                    classes=10)
        #self.reg = LowRank(param=param)

    def call(self, inputs):
        if self.arch == 'spc':
            X,y, H= self.F(inputs)
            if self.noise:
                print('here')
                sigma = tf.reduce_sum(tf.math.pow(y, 2)) / (self.batch_size * y.shape[1]) * 10 ** (self.snr / 10)
                y = y + tf.random.normal(shape=y.shape, mean=0, stddev=tf.math.sqrt(sigma), dtype=y.dtype)

        elif self.arch == 'cassi':
            y, H= self.F(inputs)
            if self.noise:
                print('here')
                sigma = tf.reduce_sum(tf.math.pow(y, 2)) / (self.batch_size * y.shape[1]) * 10 ** (self.snr / 10)
                y = y + tf.random.normal(shape=y.shape, mean=0, stddev=tf.math.sqrt(sigma), dtype=y.dtype)
            X = self.T([y,H])

        else:
            y, H = self.F(inputs)
            if self.noise:
                print('here')
                sigma = tf.reduce_sum(tf.math.pow(y, 2)) / (self.batch_size * y.shape[1]) * 10 ** (self.snr / 10)
                y = y + tf.random.normal(shape=y.shape, mean=0, stddev=tf.math.sqrt(sigma), dtype=y.dtype)

            X = self.T(y)

        if self.decoder == 'unrolling':
            X = self.network([X, y, self.F, self.T, H])

        else:
            X = self.network(X)

        if self.regularization:
            if self.type_reg=='graph' or self.type_reg=='corr':
                print('here')
                self.reg([inputs,y])
            else:
                self.reg(y)
        return X




class E2E_NetworkRealCassi(tf.keras.Model):
    def __init__(self, input_dim=(128, 128, 25), noise=True, bin_param=0.5, H=True, batch_size=32, snr=30):
        super(E2E_NetworkRealCassi, self).__init__()

        self.input_dim = input_dim
        self.batch_size = batch_size
        self.H = H
        self.noise = noise
        self.snr = snr



        self.F = ForwardCASSIReal(input_dim=input_dim, noise=False, bin_param=bin_param, H=H,
                              name='forward', shots=1, batch_size=batch_size, snr=snr)
        self.T = TransposeCASSI(input_dim=input_dim, noise=False, bin_param=bin_param, opt_H=True,
                                name='transpose', shots=1, batch_size=batch_size, snr=30)

        self.network = UNetCompiled(input_size=input_dim, n_filters=32, n_classes=input_dim[-1])




    def call(self, inputs,isinference=False):
        if isinference:
            y, H = inputs
        else:
            y, H = self.F(inputs)
        if self.noise:
            print('here')
            sigma = tf.reduce_sum(tf.math.pow(y, 2)) / (self.batch_size * y.shape[1]) * 10 ** (self.snr / 10)
            y = y + tf.random.normal(shape=y.shape, mean=0, stddev=tf.math.sqrt(sigma), dtype=y.dtype)
        X = self.T([y,H])



        X = self.network(X)


        return X



class E2E_NetworkCASSI(tf.keras.Model):
    def __init__(self, input_dim=(128, 128, 25), noise=True, bin_param=0.5, H=True, batch_size=32, snr=30):
        super(E2E_NetworkCASSI, self).__init__()

        self.input_dim = input_dim
        self.batch_size = batch_size
        self.H = H
        self.noise = noise
        self.snr = snr



        self.F = ForwardCASSI(input_dim=input_dim, noise=False, bin_param=bin_param,
                                  name='forward', shots=1, batch_size=batch_size, snr=30)
        self.T = TransposeCASSI(input_dim=input_dim, noise=False, bin_param=bin_param, opt_H=True,
                                name='transpose', shots=1, batch_size=batch_size, snr=30)

        self.network = UNetCompiled(input_size=input_dim, n_filters=32, n_classes=input_dim[-1])



    def call(self, inputs,isinference=False):
        if isinference:
            y, H = inputs
        else:
            y, H = self.F(inputs)
        if self.noise:
            print('here')
            sigma = tf.reduce_sum(tf.math.pow(y, 2)) / (self.batch_size * y.shape[1]) * 10 ** (self.snr / 10)
            y = y + tf.random.normal(shape=y.shape, mean=0, stddev=tf.math.sqrt(sigma), dtype=y.dtype)
        X = self.T([y,H])



        X = self.network(X)


        return X
