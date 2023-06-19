import os
import random

import h5py
import numpy as np
from scipy.io import loadmat
from skimage.transform import resize

import tensorflow as tf
from tensorflow.keras import layers


# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
# simulated scene
# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#

def load_scene(scene_path, spectral_bands):
    scene_mat = loadmat(scene_path)
    spectral_scene = np.double(scene_mat['hyperimg'])[np.newaxis, ...]
    spectral_scene = spectral_scene[..., ::int(spectral_scene.shape[-1] / spectral_bands + 1)]
    spectral_scene = spectral_scene / np.max(spectral_scene)

    spectral_scene = tf.image.central_crop(spectral_scene, 256 / spectral_scene.shape[1])
    spectral_scene = spectral_scene.numpy()  # [..., ::int(31 / 7 + 1)]

    rgb_colors = (15, 13, 6) if spectral_bands <= 16 else (25, 22, 11)
    scene_rgb = spectral_scene[..., rgb_colors]
    input_size = spectral_scene.shape

    return input_size, spectral_scene, scene_rgb


def get_scene_data(sensing_name, model, spectral_scene, only_measure=False, only_transpose=False):
    try:
        sensing_model = model.get_layer(sensing_name)
        data = sensing_model(spectral_scene, only_measure=only_measure, only_transpose=only_transpose)

    except:
        raise 'sensing was not in the model, please check the code implementation'

    return data


# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
# real data
# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#

def load_real_data(real_path, H_path, y_path):
    H_real = loadmat(os.path.join(real_path, H_path))['H'][..., 0]
    y_real = loadmat(os.path.join(real_path, y_path))['Y'][np.newaxis, ..., 0]

    return H_real, y_real


# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
# dataset
# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#

def load_dataset_spectral(name, path, batch_size=1, input_size=(512, 512, 31),img=5):
    train_path = os.path.join(path, 'train.h5')
    test_path = os.path.join(path, 'test.h5')

    train_dataset = get_arad_dataset(train_path, batch_size, input_size, train=True,augment=False)
    test_dataset = get_arad_dataset(test_path, batch_size, input_size,train=False,augment=False)

    return train_dataset, test_dataset



def get_arad_dataset(path, batch_size, input_size, train, augment=False):
    dataset = ARADDataset(path, input_size, train)

    augmentation = []
    if augment:
        augmentation.extend([
            layers.RandomRotation(0.3),
            layers.RandomZoom((-0.5, 0.5)),
            layers.RandomTranslation(0.0, 0.1),
            layers.RandomFlip('horizontal')
        ])
    augmentation = tf.keras.Sequential(augmentation)

    dataset_pipeline = (dataset
                        .cache()
                        #.shuffle(batch_size)
                        .batch(batch_size, drop_remainder=True)
                        .map(lambda x: augmentation(x, training=True), num_parallel_calls=tf.data.AUTOTUNE)
                        .map(lambda x: (x, x), num_parallel_calls=tf.data.AUTOTUNE)
                        .prefetch(tf.data.AUTOTUNE))

    return dataset_pipeline



class ARADDataset(tf.data.Dataset):


    def _generator(path,input_dim,train):
        with h5py.File(path, 'r') as hf:
            for X in hf['cube']:
                x = X.astype(np.float32)# / (2 ** 16 - 1)
                if train:
                    X = tf.image.random_crop(x,[input_dim[0],input_dim[1],31])
                else:
                    mo = x.shape[0]

                    X = tf.image.central_crop(x,input_dim[1]/mo)

                X = X/np.max([np.max(X),1e-6])

                # transformaciones
                

                yield X[...,1:-1:2]

    def __new__(cls, path, input_size=(512, 512, 31),train=True):
        print(f'input_size: {input_size}')
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_signature=tf.TensorSpec(shape=input_size, dtype=tf.float32),
            args=(path,input_size,train)
        )
