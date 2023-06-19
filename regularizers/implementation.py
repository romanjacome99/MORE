import tensorflow as tf

class Reg_Binary_0_1(tf.keras.regularizers.Regularizer):
    def __init__(self, parameter=10):
        self.parameter = tf.keras.backend.variable(parameter, name='parameter')

    def __call__(self, x):
        regularization = self.parameter * (tf.reduce_sum(tf.multiply(tf.square(x), tf.square(1 - x))))
        return regularization

    def get_config(self):
        return {'parameter': float(tf.keras.backend.get_value(self.parameter))}


  

class Reg_Binary_1_1(tf.keras.regularizers.Regularizer):
    def __init__(self, parameter=10):
        self.parameter = tf.keras.backend.variable(parameter,name='parameter')
    def __call__(self, x):
        regularization = self.parameter*(tf.reduce_sum(tf.math.multiply(tf.math.pow(1.+x,2),tf.math.pow(1.-x,2))))
        return regularization

    def get_config(self):
        return {'parameter': float(tf.keras.backend.get_value(self.parameter))}
