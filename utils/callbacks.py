import tensorflow as tf
import os
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import LearningRateScheduler
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib import rc
from utils.spec2rgb import ColourSystem
import scipy.io as sio

import numpy as np
from utils.tensorboard import *
import h5py


class save_each_epoch(tf.keras.callbacks.Callback):
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir

    def on_epoch_end(self, epoch, logs=None):
        print('Model Saved at: ' + self.checkpoint_dir)
        self.model.save_weights(self.checkpoint_dir)

def lr_scheduler(epoch, lr):
    decay_step = 40
    if epoch % decay_step == 0 and epoch:
        lr = lr/2
        tf.print(' Learning rate ='+ str(lr))        
        return lr
    
    return lr
class Aument_parameters(tf.keras.callbacks.Callback):
    def __init__(self, p_aum,p_step):
        super().__init__()
        self.p_aum = p_aum
        self.p_step = p_step
        
    def on_epoch_end(self, epoch, logs=None):
        current_param=self.model.get_layer('optical_encoder').my_regularizer.parameter
        current_param = tf.keras.backend.get_value(current_param)
        print('\n regularizator ='+ str(current_param))
        
        if epoch%self.p_step==0 and epoch>50:
            
            new_param = current_param * self.p_aum
            self.model.get_layer('optical_encoder').my_regularizer.parameter.assign(new_param)
            print('\n regularizator updated to '+ str(new_param))

def load_callbacks(experiment='experiment',results_folder='results',arch ='spc'):
    
    try:
        os.mkdir(results_folder)
    except OSError as error:
        print(error)

    path =results_folder +'/' + experiment + "/"
    try:
        os.mkdir(path)
    except OSError as error:
        print(error)
    csv_file = path + 'results' + ".csv"

    
    model_path = path + 'best_model' + ".tf"

    

    check_point = tf.keras.callbacks.ModelCheckpoint(
        model_path,
        monitor="val_loss",
        save_best_only=False,
        save_weights_only=True,
        mode="min",
        save_freq="epoch",
        verbose=1)
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=path+"tensorboard/", histogram_freq=1, write_graph = False)

    lr_s = LearningRateScheduler(lr_scheduler, verbose=1)
    dynamic_param = Aument_parameters(p_aum=10,p_step=10)

    if arch=='cassi' or arch=='spc':
        callbacks = [tf.keras.callbacks.CSVLogger(csv_file, separator=',', append=False),
                    lr_s,
                    dynamic_param,
                    check_point,
                    tensorboard_callback]
    else:
        callbacks = [tf.keras.callbacks.CSVLogger(csv_file, separator=',', append=False),
                    lr_s,
                    check_point,
                    tensorboard_callback]
    return callbacks, path


def plot_xnca(Xgt,Xr):

    """Return a 5x3 grid of the validation images as a matplotlib figure."""

    color_space = "sRGB"
    start, end = 400, 700 # VariSpec VIS
    number_bands = 31

    cs = ColourSystem(cs=color_space, start=start, end=end, num=number_bands)

    figure, axs_t = plt.subplots(1,2, figsize=(20,5))
    img_rgb = cs.spec_to_rgb(Xgt)
    img_rgb_rec = cs.spec_to_rgb(Xr)
    axs_t[0].imshow(tf.squeeze(img_rgb).numpy())
    axs_t[0].axis('off')
    axs_t[0].set_title('GT')
    axs_t[1].imshow(tf.squeeze(img_rgb_rec).numpy())
    axs_t[1].axis('off')
    axs_t[1].set_title('Rec')
    figure.tight_layout()
    return figure


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png',bbox_inches='tight')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image

def log_xnca(epoch, logs, model,Xgt, inputs, fw_results, name):
    # Use the model to predict the values from the validation dataset.
    y,H = inputs
    Xt= model([y,H],isinference=True)
    # Log the results images as an image summary.
    figure = plot_xnca(Xgt=Xgt,Xr=Xt)
    image_resutls = plot_to_image(figure)

    # Log the results images as an image summary.
    with fw_results.as_default():
        tf.summary.image(name, image_resutls, step=epoch)


def log_xncax(epoch, logs, model,Xgt, fw_results, name):
    # Use the model to predict the values from the validation dataset.
    Xt= model(Xgt)
    # Log the results images as an image summary.
    figure = plot_xnca(Xgt=Xgt,Xr=Xt)
    image_resutls = plot_to_image(figure)

    # Log the results images as an image summary.
    with fw_results.as_default():
        tf.summary.image(name, image_resutls, step=epoch)

