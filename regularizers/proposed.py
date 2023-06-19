import numpy as np
from matplotlib import patches
import tensorflow as tf
# from wavetf import WaveTFFactory
from utils.wavelet import DWT
from wavetf import WaveTFFactory

kernel = 'haar'
w = WaveTFFactory().build(kernel, dim=1)
w2 = DWT(wave='haar')


def variational(y, stddev, mean, model):
    z_mean = tf.reduce_mean(y, 0)
    z_log_var = tf.math.log(tf.math.reduce_std(y, 0))
    kl_loss = -0.5 * tf.math.reduce_mean(
        z_log_var - tf.math.log(stddev) - (tf.exp(z_log_var) + tf.pow(z_mean - mean, 2)) / (stddev ** 2) + 1)
    model.add_loss(kl_loss)
    model.add_metric(kl_loss, name='kl_loss', aggregation='mean')


def sparsity(y, param, model, batch_size):
    if len(y.shape) > 2:
        p_size = 64
        patches = tf.image.extract_patches(images=y,
                                           sizes=[1, p_size, p_size, 1],
                                           strides=[1, p_size, p_size, 1],
                                           rates=[1, 1, 1, 1],
                                           padding='VALID')
        a = tf.reshape(patches, shape=(batch_size * patches.shape[1] * patches.shape[1], p_size, p_size, 1))
        wavelet = w2.dwt(a, lvl=3)
        loss_sparsity = tf.reduce_mean(tf.norm(wavelet, ord=1))

    else:
        ys = tf.expand_dims(tf.expand_dims(tf.math.reduce_std(y, 0), 0), -1)
        wavelet = w.call(ys)
        wavelet = tf.split(wavelet, wavelet.shape[-1], axis=-1)
        wavelet = tf.concat([x for x in wavelet], 1)
        loss_sparsity = tf.norm(wavelet, ord=1)

    model.add_loss(loss_sparsity * param)
    model.add_metric(loss_sparsity, name='loss_sparsity', aggregation='mean')


class LowRank(tf.keras.layers.Layer):
    def __init__(self, param=1e-2, batch_size=15):
        super(LowRank, self).__init__()
        self.param = param
        self.batch_size = batch_size

    def call(self, y):
        n = np.prod(y.shape[1:])

        y = tf.reshape(y, [-1, n])
        svd = tf.linalg.svd(tf.squeeze(y), full_matrices=False
                            , compute_uv=False, name=None)
        loss_svd = tf.norm(svd[0], 1)
        self.add_loss(loss_svd * self.param)
        self.add_metric(loss_svd, name='loss_svd', aggregation='mean')


class MinVariance(tf.keras.layers.Layer):
    def __init__(self, param=1e-2):
        super(MinVariance, self).__init__()
        self.param = param

    def call(self, y):
        var_loss = tf.norm(tf.math.reduce_std(y, 0), 2)
        self.add_loss(var_loss * self.param)
        self.add_metric(var_loss, name='loss_min_var', aggregation='mean')


class MinVarVariance(tf.keras.layers.Layer):
    def __init__(self, param=1e-2):
        super(MinVarVariance, self).__init__()
        self.param = param

    def call(self, y):
        var_loss = tf.math.reduce_std(tf.math.reduce_std(y, 0))
        self.add_loss(var_loss * self.param)
        self.add_metric(var_loss, name='loss_min_var', aggregation='mean')


class Sparsity(tf.keras.layers.Layer):
    def __init__(self, param=1e-2):
        super(Sparsity, self).__init__()
        self.param = param
        self.w = WaveTFFactory().build(kernel, dim=1)
        self.w2 = DWT(wave='haar')

    def call(self, y):
        ys = tf.expand_dims(tf.expand_dims(tf.math.reduce_std(y, 0), 0), -1)
        if len(ys.shape) > 2:
            ys = tf.expand_dims(tf.expand_dims(tf.reshape(ys, [-1]), 0), -1)
        print(ys.shape)

        wavelet = self.w.call(ys)
        wavelet = tf.split(wavelet, wavelet.shape[-1], axis=-1)
        wavelet = tf.concat([x for x in wavelet], 1)
        loss_sparsity = 1 / ys.shape[1] * tf.norm(wavelet, ord=1)

        self.add_loss(loss_sparsity * self.param)
        self.add_metric(loss_sparsity, name='loss_sparsity', aggregation='mean')


class MaxVariance(tf.keras.layers.Layer):
    def __init__(self, param=1e-2, max_var=2.0):
        super(MaxVariance, self).__init__()
        self.param = param
        self.max_var = max_var

    def call(self, y):
        var_loss = tf.norm(tf.math.reduce_std(y, 0), 2)
        self.add_loss(-(var_loss) ** 2 * self.param)
        self.add_metric(var_loss, name='loss_max_var', aggregation='mean')


class KLGaussian(tf.keras.layers.Layer):
    def __init__(self, mean=1e-2, stddev=2.0):
        super(KLGaussian, self).__init__()
        self.mean = mean
        self.stddev = stddev

    def call(self, y):
        z_mean = tf.reduce_mean(y, 0)
        z_log_var = tf.math.log(tf.math.reduce_std(y, 0))
        print(self.stddev, self.mean)
        kl_loss = -0.5 * tf.math.reduce_mean(
            z_log_var - tf.math.log(self.stddev) - (tf.exp(z_log_var) + tf.pow(z_mean - self.mean, 2)) / (
                    self.stddev ** 2) + 1)
        self.add_loss(kl_loss)
        self.add_metric(kl_loss, name='kl_gauss', aggregation='mean')

class Correlation(tf.keras.layers.Layer):
    def __init__(self, batch_size=128,param=0.001):
        super(Correlation, self).__init__()
        self.batch_size = batch_size
        self.param = param

    def call(self, inputs):
        x,y= inputs
        alfa = 1.01
        Cxx = tf.matmul(tf.reshape(x, [self.batch_size, -1]), tf.transpose(tf.reshape(x, [self.batch_size, -1]))) / self.batch_size
        Cyy = tf.matmul(tf.reshape(y, [self.batch_size, -1]),
                        tf.transpose(tf.reshape(y, [self.batch_size, -1]))) / self.batch_size

        loss = tf.norm(Cxx-Cyy,2)
        # loss = tf.experimental.numpy.log2(
        #     tf.math.reduce_sum(tf.math.pow(tf.math.abs(tf.linalg.diag_part(C) - 1), 2 * alfa)))

        self.add_loss(self.param*loss)
        self.add_metric(loss, name='correlation', aggregation='mean')


class KLLaplacian(tf.keras.layers.Layer):
    def __init__(self, mean=1e-2, stddev=2.0):
        super(KLLaplacian, self).__init__()
        self.mean = mean
        self.stddev = stddev

    def call(self, y):
        z_mean = tf.reduce_mean(y, 0)
        z_var = tf.math.reduce_std(y, 0)
        lp_kl = tf.reduce_mean(
            (z_var * tf.math.exp(-tf.abs(z_mean - self.mean) / z_var) + tf.abs(
                z_mean - self.mean)) / self.stddev + tf.math.log(
                self.stddev / (z_var)) - 1)
        self.add_loss(lp_kl)
        self.add_metric(lp_kl, name='kl_laplace', aggregation='mean')


class GraphRegularized(tf.keras.layers.Layer):
    def __init__(self, sigma=0.2, epsilon=1.0, batch_size = 128, type='heat'):
        super(GraphRegularized, self).__init__()
        self.sigma = sigma
        self.epsilon = tf.constant(epsilon)
        self.type = type
        self.batch_size = batch_size

    def call(self, inputs, *args, **kwargs):
        b = self.batch_size

        loss = 0.0
        for i in range(b):
            for j in range(b):
                if i != j:
                    res = tf.cast(tf.norm(inputs[0][i,] - inputs[0][j,], 2),self.epsilon.dtype)

                    # print(res)
                    if res <= self.epsilon:
                        w = tf.cast(tf.exp(-res / self.sigma),tf.float32)

                        res_y = tf.norm(inputs[1][i,] - inputs[1][j,], 2)



                        loss += w * res_y

        self.add_loss(loss)
        self.add_metric(loss, name='graph_loss', aggregation='mean')




def low_rank(y, param, model):
    svd = tf.linalg.svd(tf.squeeze(y), full_matrices=False, compute_uv=True, name=None)
    loss_svd = tf.norm(svd[0], 1)
    model.add_loss(loss_svd * param)
    model.add_metric(loss_svd, name='loss_svd', aggregation='mean')


def min_variance(y, param, model):
    var_loss = tf.math.reduce_std(tf.math.reduce_std(y, 0))
    model.add_loss(var_loss * param)
    model.add_metric(var_loss, 'var_loss')


def max_variance(y, param, model, max_var):
    var = tf.math.reduce_std(tf.math.reduce_std(y, 0))
    var_loss = (max_var - var) ** 2
    model.add_loss(var_loss * param)
    model.add_metric(var_loss, 'var_loss')
