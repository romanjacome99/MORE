import tensorflow as tf
import numpy as np  
import scipy.io as sio
import h5py
from matplotlib import pyplot as plt
from wavetf import WaveTFFactory

class DWT():
  def __init__(self, wave='haar'):

      self.forward = WaveTFFactory().build(wave, dim=2)
      self.inverse = WaveTFFactory().build(wave, dim=2, inverse=True)

  def wavelet(self, inputs, lvl):
      inputs = tf.unstack(inputs, inputs.shape[0], 0)
      outputs = tf.stack( [self.forward(i) for i in inputs ] , 0 )
      outputs = tf.split(outputs, int(outputs.shape[-1]), -1)    

      if lvl > 1:
          outputs[0] = self.wavelet(outputs[0], lvl-1)

      outputs = tf.concat(outputs, -1)
      outputs = tf.transpose(outputs, perm=[0, 1, 4, 2, 3])
      _ , c, _, nrows, ncols = outputs.shape
      h = w = nrows*2
      
      outputs = tf.reshape(outputs, [-1, c, h//nrows, w//nrows , nrows, ncols])
      outputs = tf.transpose(outputs, perm=[0, 1, 2, 4, 3, 5])
      outputs = tf.reshape(outputs, [-1, c, h, w])
      outputs = tf.expand_dims(outputs, -1)
      return outputs

  def dwt(self, inputs, lvl=1): 
      inputs = tf.expand_dims(inputs, 1)
      inputs = tf.split(inputs, [1]*int(inputs.shape.dims[4]), 4)
      inputs = tf.concat([x for x in inputs], 1)

      inputs = self.wavelet(inputs, lvl)

      inputs = tf.reduce_sum(inputs,-1)
      inputs = tf.transpose(inputs, perm=[0, 2, 3, 1])
      
      return inputs

  def iwavelet(self, y , lvl):
    _ , _ , h , w , _ = y.shape
    nh , nw = int(h/2) , int(w/2)
    y = tf.split( y , y.shape[-1] ,axis=-1)

    if lvl > 0:
      temp = tf.unstack( y[0] , y[0].shape[0] ,axis=0)
      temp = [tf.image.extract_patches(i , [1, 2, 2, 1], [1, 1, 1, 1], [1, nh, nh, 1],  "VALID" ) for i in temp]
      temp = tf.stack(temp,0) 

      y[0] = self.iwavelet(temp, lvl-1)

    y = tf.concat(y ,axis=-1)  
    out_size = int(tf.shape(y)[2])

    y = tf.unstack( y, y.shape[0] , 0)
    y = tf.stack( [ self.inverse(i)  for i in y ] , 0 )    
    return y

  def idwt(self, y , lvl=1):
    _ , h , w , _ = y.shape
    nh , nw = int(h/2) , int(w/2)

    y = tf.transpose(y ,perm=[3,1,2,0])
    y = tf.split( y , y.shape[-1] ,axis=-1)
    y = [tf.image.extract_patches(i , [1, 2, 2, 1], [1, 1, 1, 1], [1, nh, nh, 1],  "VALID" ) for i in y]
    y = tf.stack(y,0)    
    y = self.iwavelet(y, lvl-1)

    y = tf.reduce_sum(y,-1)
    y = tf.transpose( y ,perm=[0,2,3,1])

    return y