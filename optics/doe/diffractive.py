import tensorflow as K  # se puede cambiar por from keras.import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.constraints import NonNeg, MinMaxNorm
import numpy as np
import math as m
import os
from scipy.io import loadmat
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import poppy

import random

class ForwardDOE(Layer):
    def __init__(self,input_dim=(128, 128,25), train=True,Nterms = 15,name='Forward_Model',**kwargs):
        super(ForwardDOE, self).__init__()
        self.prop1 = Propagation(Mp=input_dim[0], wl=input_dim[-1], L=0.01, zi=0.06, Trai=False)
        self.prop2 = Propagation(Mp=input_dim[0],wl=input_dim[-1],   L=0.0048, zi=0.01, Trai=False)
        if Nterms == 12:

            self.doe =  DOE_imple(Mdoe=input_dim[0], Mesce=input_dim[0],   Train=train)
        else:
            self.doe = DOE(Mdoe=input_dim[0], Mesce=input_dim[0], Train=train, Nterms=Nterms)
        self.sensor = Sensing(Ms=input_dim[0], Trai=False)

    def call(self,inputs):
        x1 = self.prop1(inputs)
        x2,H = self.doe(x1)
        x3 = self.prop2(x2)
        x4 = self.sensor(x3)

        return x4,x4

    

class DOE(Layer):

    def __init__(self, Mdoe=128, wl = 15,Mesce=128, Nterms=15, DOE_type='New', Train=True):

        self.Mdoei = Mdoe
        self.Mesce = Mesce
        self.wl = wl
        self.DOE_type = DOE_type
        self.Train = Train
        self.wave_lengths = np.linspace(420, 660, 25) * 1e-9
        print(Train)
        if Train:

            Nterms_1 = Nterms
            print(Nterms_1)
            if not os.path.exists('optics/doe/zernike_volume1_%d_Nterms_%d.npy' % (Mdoe, Nterms_1)):
                znew = 1e-6 * poppy.zernike.zernike_basis(nterms=Nterms_1, npix=self.Mdoei, outside=0.0)
                self.zernike_volume = znew[0:, :, :]
                np.save('optics/doe/zernike_volume1_%d_Nterms_%d.npy' % (Mdoe, Nterms_1), self.zernike_volume)

            else:
                self.zernike_volume = np.load('optics/doe/zernike_volume1_%d_Nterms_%d.npy' % (Mdoe, Nterms_1))
        else:

            Hm_DOE = loadmat('optics/doe/Spiral_128x128_nopadd.mat').get('Hm').astype(np.float32)
            self.Hm_DOE = Hm_DOE
        super(DOE, self).__init__()

    def build(self, input_shape):
        print('aaaaa',self.Train)
        if self.Train:
            print('here train')
            num_zernike_coeffs = self.zernike_volume.shape[0]
            zernike_inits = np.zeros((num_zernike_coeffs, 1, 1))
            zernike_initializer = K.constant_initializer(zernike_inits)
            self.zernike_coeffs = self.add_weight(name='zernike_coeffs', shape=zernike_inits.shape,
                                                  constraint=MinMaxNorm(min_value=-1.0, max_value=1.0, rate=1.0,
                                                                        axis=2),
                                                  initializer=zernike_initializer, trainable=self.Train)

            super(DOE, self).build(num_zernike_coeffs)

    def call(self, input, **kwargs):

        Lambda = self.wave_lengths
        Mdoe = self.Mdoei
        Mesce = self.Mesce
        XX = K.linspace(-Mdoe // 2, Mdoe // 2, Mdoe)
        [x, y] = K.meshgrid(XX, XX)

        max_val = 1.4 * K.reduce_max(x)
        r = K.math.sqrt(x ** 2 + y ** 2)
        P = K.cast(r < max_val, K.complex64)
        if self.Train:
            Hm = K.cast(K.reduce_sum(self.zernike_coeffs * self.zernike_volume, axis=0), K.complex64)
            Hm_i = K.cast(Hm,K.float32)
        else:
            Hm_i = self.Hm_DOE
            Hm = K.cast(self.Hm_DOE, K.complex64)
        for NLam in range(self.wl):
            IdLens = 1.5375 + 0.00829045 * (Lambda[NLam] * 1e6) ** (-2) - 0.000211046 * (Lambda[NLam] * 1e6) ** (-4)
            IdLens = IdLens - 1
            # Falta construir el Hm
            # PD = np.int32((Mesce-Mdoe)/2)
            # paddings = K.constant([[PD, PD], [PD, PD], [0, 0]])
            Aux = K.expand_dims(K.math.multiply(P, K.math.exp(1j * (2 * m.pi / Lambda[NLam]) * IdLens * Hm)), 2)
            # Aux = K.pad(Aux, paddings, "CONSTANT")
            if NLam > 0:
                P_DOE = K.concat([P_DOE, Aux], axis=2, name='stack')
            else:
                P_DOE = Aux
        #### OJO IMPROVISACION
        input2 = input[:, ::np.int32(Mesce / Mdoe), ::np.int32(Mesce / Mdoe), :]
        u2 = K.math.multiply(input2, P_DOE)
        ###
        # u2 = K.math.multiply(input, P_DOE)
        # u2 = u2[:,np.int(Mesce / 2 - Mdoe / 2): np.int(Mesce / 2 + Mdoe / 2),
        #          np.int(Mesce / 2 - Mdoe / 2): np.int(Mesce / 2 + Mdoe / 2),:]
        return u2,Hm_i
        # return K.math.multiply(input, self.kernel)


class DOE_imple(Layer):

    def __init__(self, Mdoe=128, Mesce=128, wave_lengths=None, DOE_type='New', Trai=True, **kwargs):

        self.Mdoei = Mdoe
        self.Mesce = Mesce
        self.DOE_type = DOE_type
        if wave_lengths is not None:
            self.wave_lengths = wave_lengths
        else:
            self.wave_lengths = np.linspace(420, 660, 15) * 1e-9
        if DOE_type == 'New' or DOE_type == 'Zeros':
            self.Trai = Trai  # Trai
            self.zernike_volume = loadmat('optics/doe/Base_zernike_128x128_nopadd.mat').get('HmBase').astype(np.float32)
        else:
            Hm_DOE = loadmat('optics/doe/Spiral_128x128_nopadd.mat').get('Hm').astype(np.float32)
            self.Hm_DOE = Hm_DOE

            self.Trai = False
        self.P = loadmat('optics/doe/Spiral_128x128_nopadd.mat').get('P').astype(np.float32)
        super(DOE_imple, self).__init__()

    def build(self, input_shape):
        if self.DOE_type == 'New':
            num_zernike_coeffs = self.zernike_volume.shape[2]
            zernike_inits = np.zeros((1, 1, num_zernike_coeffs))
            # zernike_inits[0] = -1  # This sets the defocus value to approximately focus the image for a distance of 1m.
            zernike_inits[0, 0, 0] = random.random() * 2 - 1
            zernike_inits[0, 0, 1] = random.random() * 2 - 1
            zernike_inits[0, 0, 2] = random.random() * 2 - 1
            zernike_inits[0, 0, 3] = random.random() * 2 - 1
            zernike_inits[0, 0, 4] = random.random() * 2 - 1
            zernike_inits[0, 0, 5] = random.random() * 2 - 1
            zernike_inits[0, 0, 6] = random.random() * 2 - 1
            zernike_inits[0, 0, 7] = random.random() * 2 - 1
            zernike_inits[0, 0, 8] = random.random() * 2 - 1
            zernike_inits[0, 0, 9] = random.random() * 2 - 1
            zernike_inits[0, 0, 10] = random.random() * 2 - 1
            zernike_inits[0, 0, 11] = random.random() * 2 - 1
            zernike_initializer = K.constant_initializer(zernike_inits)
            self.zernike_coeffs = self.add_weight(name='zernike_coeffs', shape=zernike_inits.shape,
                                                  constraint=MinMaxNorm(min_value=-1.0, max_value=1.0, rate=1.0,
                                                                        axis=2),
                                                  initializer=zernike_initializer, trainable=self.Trai)
            super(DOE_imple, self).build(num_zernike_coeffs)
        if self.DOE_type == 'Zeros':
            num_zernike_coeffs = self.zernike_volume.shape[2]
            zernike_inits = np.zeros((1, 1, num_zernike_coeffs))
            zernike_inits[0, 0, 0:11] = 0
            zernike_initializer = K.constant_initializer(zernike_inits)
            self.zernike_coeffs = self.add_weight(name='zernike_coeffs', shape=zernike_inits.shape,
                                                  constraint=MinMaxNorm(min_value=-1.0, max_value=1.0, rate=1.0,
                                                                        axis=2),
                                                  initializer=zernike_initializer, trainable=False)
            super(DOE_imple, self).build(num_zernike_coeffs)

    def call(self, input, **kwargs):

        # Hm = Hm  # Learnable
        # Lambda = Lambda  # Input to construct
        Lambda = self.wave_lengths
        Mdoe = self.Mdoei
        Mesce = self.Mesce
        XX = K.linspace(-Mdoe // 2, Mdoe // 2, Mdoe)
        [x, y] = K.meshgrid(XX, XX)

        # max_val = K.reduce_max(x)
        # r = K.math.sqrt(x ** 2 + y ** 2)
        # P = K.cast(r < max_val, K.complex64)
        P = K.cast(self.P, K.complex64)
        if self.DOE_type == 'New' or self.DOE_type == 'Zeros':

            Hm = K.cast(K.reduce_sum(self.zernike_coeffs * self.zernike_volume, axis=2), K.complex64)
            Hm_i  = K.cast(Hm,K.float32)
        else:
            Hm = K.cast(self.Hm_DOE, K.complex64)
            Hm_i =  K.cast(self.Hm_DOE, K.float32)
        for NLam in range(15):
            IdLens = 1.5375 + 0.00829045 * (Lambda[NLam] * 1e6) ** (-2) - 0.000211046 * (Lambda[NLam] * 1e6) ** (-4)
            IdLens = IdLens - 1
            # Falta construir el Hm
            # PD = np.int32((Mesce-Mdoe)/2)
            # paddings = K.constant([[PD, PD], [PD, PD], [0, 0]])
            # Aux = K.expand_dims(K.math.multiply(P, K.math.exp(1j * (2 * m.pi / Lambda[NLam]) * IdLens * Hm)), 2)
            Aux = K.expand_dims(K.math.exp(1j * (2 * 10.0* m.pi / Lambda[NLam]) * IdLens * Hm), 2)
            # Aux = K.pad(Aux, paddings, "CONSTANT")
            if NLam > 0:
                P_DOE = K.concat([P_DOE, Aux], axis=2, name='stack')
            else:
                P_DOE = Aux
        #### OJO IMPROVISACION
        input2 = input[:, ::np.int32(Mesce / Mdoe), ::np.int32(Mesce / Mdoe), :]
        u2 = K.math.multiply(input2, P_DOE)

        return u2,K.expand_dims(K.expand_dims(Hm_i,0),-1)/K.reduce_max(Hm_i)
        # return K.math.multiply(input, self.kernel)


class Sensing(Layer):

    def __init__(self, Ms=1000, wl = 15,wave_lengths=None, bgr_response=None, Trai=False, **kwargs):
        self.M = Ms
        self.wl = wl
        '''
        if bgr_response is not None:
            self.bgr_response = K.cast(bgr_response, dtype=K.float32)
        else:
            self.R = K.cast(loadmat('Sensor_25_new3.mat').get('R'), K.float32)
            self.G = 1*K.cast(loadmat('Sensor_25_new3.mat').get('G'), K.float32)
            self.B = K.cast(loadmat('Sensor_25_new3.mat').get('B'), K.float32)
        '''
        self.R = self.resize_sensor(K.cast(loadmat('optics/doe/Sensor_25_new3.mat').get('R'), K.float32))
        self.G = self.resize_sensor(K.cast(loadmat('optics/doe/Sensor_25_new3.mat').get('G'), K.float32))
        self.B = self.resize_sensor(K.cast(loadmat('optics/doe/Sensor_25_new3.mat').get('B'), K.float32))
        super(Sensing, self).__init__()
    def resize_sensor(self,x):
        x = K.squeeze(x)
        x = K.expand_dims(K.expand_dims(x,-1),-1)
        x = K.squeeze(K.image.resize(x, [self.wl, 1]))
        x = K.expand_dims(x,0)
        return x

    def build(self, input_shape):
        super(Sensing, self).build(input_shape)

    def call(self, input, **kwargs):
        Kernel = np.ones((1, 3, 3,1))
        for NLam in range(self.wl):
            if NLam > 0:
                y_med_r = y_med_r + K.math.abs(input[:, :, :, NLam]) * self.R[0, NLam]
                y_med_g = y_med_g + K.math.abs(input[:, :, :, NLam]) * self.G[0, NLam]
                y_med_b = y_med_b + K.math.abs(input[:, :, :, NLam]) * self.B[0, NLam]
            else:
                y_med_r = K.math.abs(input[:, :, :, NLam]) * self.R[0, NLam]
                y_med_g = K.math.abs(input[:, :, :, NLam]) * self.G[0, NLam]
                y_med_b = K.math.abs(input[:, :, :, NLam]) * self.B[0, NLam]
        y_med_r = K.expand_dims(y_med_r, 3)
        y_med_g = K.expand_dims(y_med_g, 3)
        y_med_b = K.expand_dims(y_med_b, 3)

        y_med_r = K.nn.conv2d(K.concat([y_med_r,y_med_r,y_med_r],axis=3), Kernel, strides=[1, 1, 1, 1],padding='SAME')
        y_med_g = K.nn.conv2d(K.concat([y_med_g,y_med_g,y_med_g],axis=3), Kernel, strides=[1, 1, 1, 1],padding='SAME')
        y_med_b = K.nn.conv2d(K.concat([y_med_b,y_med_b,y_med_b],axis=3), Kernel, strides=[1, 1, 1, 1],padding='SAME')

        y_med_r = K.nn.conv2d(K.concat([y_med_r,y_med_r,y_med_r],axis=3), Kernel, strides=[1, 1, 1, 1],padding='SAME')
        y_med_g = K.nn.conv2d(K.concat([y_med_g,y_med_g,y_med_g],axis=3), Kernel, strides=[1, 1, 1, 1],padding='SAME')
        y_med_b = K.nn.conv2d(K.concat([y_med_b,y_med_b,y_med_b],axis=3), Kernel, strides=[1, 1, 1, 1],padding='SAME')

        y_final = K.concat([y_med_r, y_med_g, y_med_b], axis=3)
        #y_final = K.concat([K.expand_dims(y_med_r, 3), K.expand_dims(y_med_g, 3), K.expand_dims(y_med_b, 3)], axis=3)

        #y_final = K.nn.conv2d(y_final, Kernel, strides=[1, 1, 1, 1], padding='SAME')
        #Kernel = np.zeros((1,150,150,1))
        #Kernel[0,70:80,70:80,0] = 1
        #Kernel =  K.cast(Kernel,K.complex64)
        #KF = K.signal.fft2d(Kernel)
        #Y_finalF = K.signal.fft2d(K.cast(y_final,K.complex64))
        #y_final = K.cast(K.math.abs(K.signal.ifft2d(K.math.multiply(Y_finalF, KF))),K.float32)
        #y_final = K.math.square(K.concat([K.expand_dims(y_med_r, 3), K.expand_dims(y_med_g, 3), K.expand_dims(y_med_b, 3)], axis=3))


        y_final = y_final / K.reduce_max(y_final)
        ## falta el cuadrado
        return y_final

class Propagation(Layer):

    def __init__(self, Mp=300, L=1.0, wl =15, wave_lengths=None, zi=2.0, Trai=True, **kwargs):

        # self.z = z
        self.Mpi = Mp
        #self.Mdoei = Mdoe
        self.Li = L
        self.zi = zi
        self.Trai = Trai
        self.wl = wl
        #self.r = Mdoe/Mp
        if wave_lengths is not None:
            self.wave_lengths = wave_lengths
        else:
            self.wave_lengths = np.linspace(420, 660, wl) * 1e-9

        super(Propagation, self).__init__()

    def build(self, input_shape):
        initializerC = K.constant_initializer(self.zi)
        self.z = self.add_weight("Distance", shape=[1], constraint=NonNeg(), initializer=initializerC, trainable=self.Trai)
        super(Propagation, self).build(input_shape)

    def call(self, inputs, **kwargs):
        L = self.Li
        Mp = self.Mpi
        Lambda = self.wave_lengths
        dx = L / Mp
        Ns = np.int(L * 2 / (2 * dx))
        # This need to be do it for all spectral bands
        fx = K.linspace(-1 / (2 * dx), 1 / (2 * dx) - 1 / L, Ns)
        [FFx, FFy] = K.meshgrid(fx, fx)
        #H = K.zeros([Mp, Mp, 25])
        for NLam in range(self.wl):
            Aux = -1j * m.pi * Lambda[NLam] * K.cast(self.z, K.complex64)
            Aux2 = K.cast(FFx ** 2 + FFy ** 2, K.complex64)
            Ha=K.math.exp(Aux * Aux2)
            Ha = K.expand_dims(K.signal.fftshift(Ha, axes=[0,1]), 2)
            if NLam > 0 :
                H = K.concat([H, Ha], axis=2, name='stack')
            else:
                H = Ha
        Aux3 = K.signal.fftshift(K.cast(inputs, K.complex64),axes=[1,2])
        u1f = K.signal.fft2d(Aux3)
        H = K.expand_dims(H, 0)
        u2f = K.math.multiply(u1f, H)
        u2 = K.signal.ifftshift(K.signal.ifft2d(u2f),axes=[1,2])


        return u2