import tensorflow as tf

def load_dataset(dataset,M,ct):


    if dataset == 'cifar10':
        (x_train,y_train),(x_test,y_test) = tf.keras.datasets.cifar10.load_data()
        y_train,y_test = tf.keras.utils.to_categorical(y_train,10),tf.keras.utils.to_categorical(y_test,10)     
        x_train,x_test = x_train/255.0,x_test/255.0
        x_train,x_test = tf.image.rgb_to_grayscale(tf.image.resize(x_train,[M,M])),tf.image.rgb_to_grayscale(tf.image.resize(x_test,[M,M]))

    elif dataset == 'mnist':
        (x_train,y_train),(x_test,y_test) = tf.keras.datasets.mnist.load_data()
        y_train,y_test = tf.keras.utils.to_categorical(y_train,10),tf.keras.utils.to_categorical(y_test,10)     
        x_train,x_test = tf.expand_dims(x_train/255.0,-1),tf.expand_dims(x_test/255.0,-1)
        x_train,x_test = tf.image.resize(x_train,[M,M]),tf.image.resize(x_test,[M,M])
 
    elif dataset == 'fashion':
        (x_train,y_train),(x_test,y_test) = tf.keras.datasets.fashion_mnist.load_data()
        y_train,y_test = tf.keras.utils.to_categorical(y_train,10),tf.keras.utils.to_categorical(y_test,10)     
        x_train,x_test = tf.expand_dims(x_train/255.0,-1),tf.expand_dims(x_test/255.0,-1)
        x_train,x_test = tf.image.resize(x_train,[M,M]),tf.image.resize(x_test,[M,M])

    if ct=='recovery':
        return (x_train,x_train),(x_test,x_test)
    elif ct=='classification':
        return (x_train,y_train),(x_test,y_test)
        

    