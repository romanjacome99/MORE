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



def Forward_DM_Spiral(input_size=(128, 128,25), DOE_typeA='New',Nterms = 15,name='Forward_Model'):

    # define the model input
    MSS = 128
    Minput = 128
    inputs = Input(shape=input_size)
    In_DOE1a = Propagation(Mp=Minput, L=0.01, zi=0.06, Trai=False)(inputs) #AcÃ¡ hay que revisar dimensiones de la proyeccion y tamaÃ±os Hay que ajustar porque no está propagando
    Out_DOE1a = DOE(Mdoe=MSS, Mesce=Minput,  DOE_type=DOE_typeA, Trai=True, Nterms = Nterms)(In_DOE1a)
    In_IPa = Propagation(Mp=MSS, L=0.006, zi=0.05, Trai=False)(Out_DOE1a)
    Measurement = Sensing(Ms=MSS, Trai=False)(In_IPa)
    model = Model(inputs, Measurement,name=name)

    return model


class DOE(Layer):

    def __init__(self, Mdoe=128, Mesce=128, Nterms=15, DOE_type='New', Train=True, **kwargs):

        self.Mdoei = Mdoe
        self.Mesce = Mesce
        self.DOE_type = DOE_type
        self.Train = Train
        self.wave_lengths = np.linspace(420, 660, 25) * 1e-9
        if Train:

            Nterms_1 = Nterms
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
        if self.Train:
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
        else:
            Hm = K.cast(self.Hm_DOE, K.complex64)
        for NLam in range(25):
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
        return u2
        # return K.math.multiply(input, self.kernel)


class Sensing(Layer):

    def __init__(self, Ms=1000, wave_lengths=None, bgr_response=None, Trai=False, **kwargs):
        self.M = Ms
        '''
        if bgr_response is not None:
            self.bgr_response = K.cast(bgr_response, dtype=K.float32)
        else:
            self.R = K.cast(loadmat('Sensor_25_new3.mat').get('R'), K.float32)
            self.G = 1*K.cast(loadmat('Sensor_25_new3.mat').get('G'), K.float32)
            self.B = K.cast(loadmat('Sensor_25_new3.mat').get('B'), K.float32)
        '''    
        self.R = K.cast(loadmat('optics/doe/Sensor_25_new3.mat').get('R'), K.float32)
        self.G = 1*K.cast(loadmat('optics/doe/Sensor_25_new3.mat').get('G'), K.float32)
        self.B = K.cast(loadmat('optics/doe/Sensor_25_new3.mat').get('B'), K.float32)
        super(Sensing, self).__init__()

    def build(self, input_shape):
        super(Sensing, self).build(input_shape)

    def call(self, input, **kwargs):
        Kernel = np.ones((1, 3, 3,1))
        for NLam in range(25):
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

    def __init__(self, Mp=300, L=1, wave_lengths=None, zi=2, Trai=True, **kwargs):

        # self.z = z
        self.Mpi = Mp
        #self.Mdoei = Mdoe
        self.Li = L
        self.zi = zi
        self.Trai = Trai
        #self.r = Mdoe/Mp
        if wave_lengths is not None:
            self.wave_lengths = wave_lengths
        else:
            self.wave_lengths = np.linspace(420, 660, 25) * 1e-9

        super(Propagation, self).__init__()

    def build(self, input_shape):
        initializerC = K.constant_initializer(self.zi)
        self.z = self.add_weight("Distance", shape=[1], constraint=NonNeg(), initializer=initializerC, trainable=self.Trai)
        super(Propagation, self).build(input_shape)

    def call(self, input, **kwargs):
        L = self.Li
        Mp = self.Mpi
        Lambda = self.wave_lengths
        dx = L / Mp
        Ns = np.int(L * 2 / (2 * dx))
        # This need to be do it for all spectral bands
        fx = K.linspace(-1 / (2 * dx), 1 / (2 * dx) - 1 / L, Ns)
        [FFx, FFy] = K.meshgrid(fx, fx)
        #H = K.zeros([Mp, Mp, 25])
        for NLam in range(25):
            Aux = -1j * m.pi * Lambda[NLam] * K.cast(self.z, K.complex64)
            Aux2 = K.cast(FFx ** 2 + FFy ** 2, K.complex64)
            Ha=K.math.exp(Aux * Aux2)
            Ha = K.expand_dims(K.signal.fftshift(Ha, axes=[0,1]), 2)
            if NLam > 0 :
                H = K.concat([H, Ha], axis=2, name='stack')
            else:
                H = Ha
        Aux3 = K.signal.fftshift(K.cast(input, K.complex64),axes=[1,2])
        u1f = K.signal.fft2d(Aux3)
        H = K.expand_dims(H, 0)
        u2f = K.math.multiply(u1f, H)
        u2 = K.signal.ifftshift(K.signal.ifft2d(u2f),axes=[1,2])


        return u2