import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
# Set up some global variables
USE_GPU = True

if USE_GPU:
    device = '/device:GPU:0'
else:
    device = '/cpu:0'

# Constant to control how often we print when training models
print_every = 100

print('Using device: ', device)

# MNIST dataset

'''

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

image_size = x_train.shape[1]
original_dim = image_size * image_size

x_train = x_train.astype('float32') / 255. # Nawid - Divides the values by 255
x_test = x_test.astype('float32') / 255.
x_train =np.reshape(x_train, [-1, image_size, image_size, 1])
x_test =np.reshape(x_test, [-1, image_size, image_size, 1])
'''
image_size = 512
original_dim = image_size * image_size
data_dir = r'F:\Shi'
origin = np.load(os.path.join(data_dir,'512_train_data.npy'))
origin_test = np.load(os.path.join(data_dir,'512_test_data.npy'))

#origin = np.load(os.path.join(data_dir,'shuffled_data.npy'))

x_train = origin
x_test = origin_test
# Nawid - Noisy version of the testing and training data for denoising purposes
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# network parameters
input_shape = (image_size,image_size,1)
intermediate_dim = 80
batch_size = 128
latent_dim = 32
epochs = 2
conv_filters = 16
conv_kernel_size = 3

# Nawid - Layers for the encoder architecture
x = tf.keras.layers.Input(shape=(image_size,image_size,1))  # Nawid - Input layer
# 28x28x1
x1 = tf.keras.layers.Conv2D(filters = conv_filters, kernel_size = conv_kernel_size, activation ='relu', strides = 2, padding = 'same')(x)
# 
x2 = tf.keras.layers.Conv2D(filters = 2*conv_filters, kernel_size = conv_kernel_size, activation ='relu', strides = 2, padding = 'same')(x1)

shape = tf.keras.backend.int_shape(x2) # Nawid - THIS OBTAINS THE SHAPE NEEDED TO BE SPECIFIED FOR THE DECONVOLUTION (CONVTRANPOSE IN THE DECODER)
flattened_x = tf.keras.layers.Flatten()(x2)

# Nawid - This is an additional layer to obtain an intermediate representation beforehand - need to change to h
h = tf.keras.layers.Dense(intermediate_dim, activation='relu')(flattened_x)
z_mean = tf.keras.layers.Dense(latent_dim)(h)  #
z_log_sigma = tf.keras.layers.Dense(latent_dim)(h)   #shape:[batch_size x latent_dim]

class Sampling(tf.keras.layers.Layer): # Nawid - Specifies a custom layer
  """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

  def call(self, inputs):
    z_mean, z_log_sigma = inputs
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim)) # Nawid - Initalised random values with a mean of 0 and a standard deviation of 1
    return z_mean + tf.exp(0.5 * z_log_sigma) * epsilon

z = Sampling()((z_mean, z_log_sigma)) # Nawid - Instantiates the custom layer and gets the output of the custom layer which is the value of z
# Nawid - Specifies the encoder model
encoder = tf.keras.Model(x, z)
encoder.summary()

# Nawid - Specifies the decoder layers
z_decoded1 = tf.keras.layers.Dense(intermediate_dim, activation ='relu')(z)
#correspond to layer h: Dense(512)
z_decoded2 = tf.keras.layers.Dense(shape[1]*shape[2]*shape[3], activation='relu')(z_decoded1) # Nawid - z_decoded
#correspond to flatten layer: Flatten(x2)
z_decoded3 = tf.keras.layers.Reshape((shape[1], shape[2], shape[3]))(z_decoded2)
#get ready to conv_transpose  7x7x32
z_decoded4 = tf.keras.layers.Conv2DTranspose(filters =conv_filters, kernel_size = conv_kernel_size, activation ='relu', strides=2, padding='same')(z_decoded3)
#output: 14x14x16
z_decoded5 = tf.keras.layers.Conv2DTranspose(filters =conv_filters//2, kernel_size = conv_kernel_size, activation ='relu', strides=2, padding='same')(z_decoded4)
#output: 28x28x8  Extra layer comparing to Encoder
x_decoded = tf.keras.layers.Conv2DTranspose(filters =1, kernel_size = conv_kernel_size, activation ='sigmoid',padding = 'same')(z_decoded5)
#output: 28x28x1

# end-to-end autoencoder
vae = tf.keras.Model(x, x_decoded)
vae.summary()
def vae_loss(x, x_decoded_mean):
    xent_loss = tf.keras.losses.binary_crossentropy(tf.keras.backend.flatten(x), tf.keras.backend.flatten(x_decoded_mean))
    xent_loss *= original_dim
    #xent_loss = tf.keras.backend.mean(xent_loss)
    kl_loss = - 0.5 * tf.keras.backend.mean(1 + z_log_sigma - tf.keras.backend.square(z_mean) - tf.keras.backend.exp(z_log_sigma), axis=-1)
    return xent_loss + kl_loss

vae.compile(optimizer='adam', loss=vae_loss,experimental_run_tf_function=False)
# Explanation for where experimental_run_tf comes from -  https://github.com/tensorflow/probability/issues/519

#vae.compile(optimizer='adam', loss='binary_crossentropy', experimental_run_tf_function=False)

vae.fit(x_train, x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, x_test))


decoded_imgs = vae.predict(x_test)
'''
vae.fit(x_train, x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1)
'''

n = 10  # how many digits we will display
plt.figure(figsize=(image_size, image_size/5))
for i in range(10):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(image_size, image_size))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(image_size, image_size))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
