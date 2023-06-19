# -*- coding: utf-8 -*-
import numpy as np

from load_data.spectral_data import *
from load_data.spectral_data_2 import *
from load_data.rgb_grayscale_data import *
from network.e2e_network import E2E_Network
from utils.callbacks import load_callbacks
from utils.metrics import psnr, SSIM
import scipy.io

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

dataset = 'spectral'
M = 128
L = 15
path = 'C:\Roman\datasets\Arad1k\Arad1k'
train_path = os.path.join(path, 'train')
test_path = os.path.join(path, 'val')
dataset_path = os.path.join(path, 'val')
batch_size = 10
ct = 'recovery'
input_dim = (M, M, L)
noise = False
snr = 30
decoder = 'unet'
compression = 0.1
Nterms = 250
shots = 1
arch = 'cassi'
opt_H = True
type_unr = 'hqs'
mean = 1.1
stddev = 0.9
param = 0.00001
max_var = 8.0
n_stages = 12
params = [1e-5, 1e-4,1e-2,1e-1]
Nterms = 250
regs = ['sparsity','low_rank','min-variance','max-variance']
for reg in regs:
    for param in params:
        type_reg = reg
        if type_reg == 'no-reg':
            regularization = False
        else:
            regularization = True

        bin_param = 1e-3
        lr = 1e-3
        results_folder = 'results/doe'

        print('[Info] Loading ' + dataset + ' dataset')
        if dataset == 'spectral':
            x_train, x_test, _ = generate_dataset(batch_size, M, L, train_path, test_path, test_path, 'cube')

        else:

            x_train, x_test = load_dataset(dataset=dataset, M=M, ct=ct)

        print('[Info] Employing ' + arch + ' architecture')
        if decoder == 'unrolling':
            print('[Info] Decoder ' + decoder + ' type ' + type_unr)
        else:
            print('[Info] Decoder ' + decoder)
        print('[Info] Employing ' + type_reg + ' regularization')

        experiment = f'{arch}_{decoder}_{type_reg}_sigma_{sigma}'

        model = E2E_Network(arch=arch, ct=ct, input_dim=input_dim, decoder=decoder, compression=compression, Nterms=Nterms,
                            n_stages=n_stages, shots=shots, type_reg=type_reg, param=param, max_var=max_var, noise=noise,
                            bin_param=bin_param,opt_H=opt_H, batch_size=batch_size, snr=snr, type_unr=type_unr,
                            regularization=regularization, mean=mean, stddev=stddev)
        model.build([batch_size, M, M, L])
        optimizad = tf.keras.optimizers.Adam(learning_rate=lr)
        if ct == 'recovery':
            model.compile(optimizer=optimizad, loss='mean_squared_error', metrics=[psnr, SSIM])
        elif ct == 'classification':
            model.compile(optimizer=optimizad, loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

        callbacks, path = load_callbacks(results_folder=results_folder, arch=arch, experiment=experiment)

        model.summary()

        h = model.fit(x_train, validation_data=x_test, verbose=1, epochs=50, callbacks=callbacks,
                      batch_size=batch_size)
        H = np.array(model.get_weights()[0])
        scipy.io.savemat(path + '/H.mat', {'H': H})

        x = tf.image.resize(tf.expand_dims(sio.loadmat('C:\Roman\datasets\Arad1k\Arad1k/val\ARAD_1K_0944.mat')['cube'][:, :, 1:-1:2], 0),[128, 128])
        # model.load_weights('results\doe_icvl_reg_exps\doe_unrolling_no-reg_mean_1.1_stddev_0.9_param_0.0001/best_model.tf')
        model_new = E2E_Network(arch=arch, ct=ct, input_dim=input_dim, decoder=decoder, compression=compression, Nterms=Nterms,
                            n_stages=n_stages, shots=shots, type_reg='no-reg', param=param, max_var=max_var, noise=noise,
                            bin_param=bin_param, opt_H=opt_H, batch_size=batch_size, snr=snr, type_unr=type_unr,
                            regularization=False, mean=mean, stddev=stddev)
        model.build([batch_size, M, M, L])
        model_new.load_weights(f'{path}/best_model.tf')
        x_rec = model_new(x)
        xr = x_rec[-1]
        sio.savemat(f'{path}/rec.mat',{'xgt':np.array(x),'xr':np.array(xr)})
