import tensorflow as tf
import matplotlib.pyplot as plt
import io
import numpy as np

def get_metrics(img1, img2, max_val=1):
    def mse(x1, x2):
        return tf.reduce_mean(tf.math.square(x1-x2), axis=[1, 2, 3])
    #img1 = img1[:,]
    #img2 = img2[:,]
    img1 = tf.cast(img1, img2.dtype)
    ssim = tf.image.ssim(img1, img2, max_val).numpy().tolist()
    psnr = tf.image.psnr(img1, img2, max_val).numpy().tolist()
    mse = mse(img1, img2).numpy().tolist()

    return mse, ssim, psnr

def plot_test_images_spectral(x_spectral_gt, x_spectral_pred, y,H, bands_to_show = [25, 15, 5]):
    metrics_spectral = get_metrics(x_spectral_gt, x_spectral_pred)
    p1 = np.random.randint(0,x_spectral_gt.shape[1])
    p2 = np.random.randint(0,x_spectral_gt.shape[1])
    x_pred_sig = np.squeeze(x_spectral_pred[:,p1,p2,:])
    gt_sig = np.squeeze(x_spectral_gt[:,p1,p2,:])
    
    
    figure = plt.figure(figsize=(35,5))
    plt.subplot(1, 5, 1); plt.imshow(tf.squeeze(H[:,:,:,1]), cmap="gray"); plt.colorbar(); plt.clim(0, 1); plt.xticks([]); plt.yticks([]); plt.title("H")

    plt.subplot(1, 5, 2); plt.imshow(tf.squeeze(y), cmap="gray"); plt.colorbar(); plt.clim(0, 1); plt.xticks([]); plt.yticks([]); plt.title("Y")
    
    plt.subplot(1, 5, 3); plt.imshow(tf.squeeze(x_spectral_gt.numpy()[...,bands_to_show]), cmap="gray"); plt.colorbar(); plt.clim(0, 1); plt.xticks([]); plt.yticks([]); plt.title("GT")

    plt.subplot(1, 5, 4); plt.imshow(tf.squeeze(x_spectral_pred.numpy()[...,bands_to_show]));plt.clim(0, 1); plt.xticks([]); plt.yticks([]); plt.title("X Pred mse "+ str(round(metrics_spectral[0][0],2)) + " ssim "+ str(round(metrics_spectral[1][0],2)) + " psnr "+ str(round(metrics_spectral[2][0],2)))

    plt.subplot(1, 5, 5); plt.plot(x_pred_sig, label='pred'), plt.plot(gt_sig, label='gt'),plt.legend(), plt.title("Spectral Reconstruction")
    plt.show()
    
    return figure

def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image

def log_images_spectral(epoch, logs, model, val_img, fw_results, name):
    # Use the model to predict the values from the validation dataset.
    inputs = val_img

    x_spectral_pred = model(inputs)
    H = model.layers[1].get_weights()[0]
    y = model.layers[1].get_measurements(tf.cast(inputs,tf.float32))
    x_spectral_pred = x_spectral_pred[0:1, ...]
    # Log the results images as an image summary.
    figure = plot_test_images_spectral(x_spectral_gt=inputs,x_spectral_pred=x_spectral_pred, y=y, H=H)
    image_resutls = plot_to_image(figure)

    # Log the results images as an image summary.
    with fw_results.as_default():
        tf.summary.image(name, image_resutls, step=epoch)


def plot_test_images_spc(x_gt, x_pred, H, bands_to_show=[25, 15, 5]):
    metrics_spectral = get_metrics(x_gt, x_pred)



    figure = plt.figure(figsize=(35, 5))
    plt.subplot(1, 4, 1);
    plt.imshow(tf.squeeze(H[:,:,:,1]), cmap="gray");
    plt.colorbar();
    plt.clim(0, 1);
    plt.xticks([]);
    plt.yticks([]);
    plt.title("H shot 1")

    plt.subplot(1, 4, 2);
    plt.imshow(tf.squeeze(H[:, :, :,  -4]), cmap="gray");
    plt.colorbar();
    plt.clim(0, 1);
    plt.xticks([]);
    plt.yticks([]);
    plt.title("H shot 50")

    plt.subplot(1, 4, 3);
    plt.imshow(tf.squeeze(x_gt), cmap="gray");
    plt.colorbar();
    plt.clim(0, 1);
    plt.xticks([]);
    plt.yticks([]);
    plt.title("Y")

    plt.subplot(1, 4, 4);
    plt.imshow(tf.squeeze(x_pred), cmap="gray");
    plt.colorbar();
    plt.clim(0, 1);
    plt.xticks([]);
    plt.yticks([]);
    plt.title("X Pred mse " + str(round(metrics_spectral[0][0], 2)) + " ssim " + str(
        round(metrics_spectral[1][0], 2)) + " psnr " + str(round(metrics_spectral[2][0], 2)))


    return figure
def log_images_spc(epoch, logs, model, val_img, fw_results, name):
    # Use the model to predict the values from the validation dataset.
    inputs = val_img

    x_pred = model(inputs)
    H = model.layers[1].get_weights()[0]
    x_pred = x_pred[0:1, ...]
    # Log the results images as an image summary.
    figure = plot_test_images_spc(x_pred=x_pred, x_gt=inputs,  H=H)
    image_resutls = plot_to_image(figure)

    # Log the results images as an image summary.
    with fw_results.as_default():
        tf.summary.image(name, image_resutls, step=epoch)


