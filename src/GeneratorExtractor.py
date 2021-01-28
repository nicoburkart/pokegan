from tensorflow.keras.layers import InputLayer, Dense, BatchNormalization, Reshape, Conv2DTranspose, ReLU, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.initializers import RandomNormal
import os
import tensorflowjs as tfjs

# This script is used to restore the generator model from a specific checkpoint and to save it in the required format to be able to use it in JavaScript code.

# The kernel initializer defines how the weights in a neural network layer should be initialized. This has a direct effect on convergence speed. A recommended Distribution for training DCGANs is the following:
generator_kernel_initializer = RandomNormal(stddev=0.02)

def create_generator(noise_dimensions):
    model = Sequential()

    model.add(InputLayer(input_shape=(noise_dimensions,)))

    # First layer has to be a dense layer which expands the noise vector. This has to be done to create the required 16.384 values which after being reshaped form the input feature maps for the first deconvolutional layer.
    model.add(Dense(units=4 * 4 * 1024, kernel_initializer=generator_kernel_initializer, use_bias=False))
    model.add(BatchNormalization())

    model.add(Reshape(target_shape=(4, 4, 1024)))

    # The output feature maps of the first deconvolutional layer have a shape of (8, 8, 512). This could be achieved by the stride set to 2 and padding set to 'same' which basically doubles the size of the image (i.e. the pixels).
    model.add(Conv2DTranspose(filters=512, kernel_size=(5, 5), kernel_initializer=generator_kernel_initializer, strides=(2, 2), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(ReLU())

    # After the second one there are 256 feature maps with a size of 16x16 left.
    model.add(Conv2DTranspose(filters=256, kernel_size=(5, 5), kernel_initializer=generator_kernel_initializer, strides=(2, 2), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(ReLU())

    # After the third block there are only 128 feature maps with a size of 32x32 left.
    model.add(Conv2DTranspose(filters=128, kernel_size=(5, 5), kernel_initializer=generator_kernel_initializer, strides=(2, 2), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(ReLU())

    # After the last deconvolutional layer the output has a size of 64x64x3 (RGB image with a width and height of 64px).
    model.add(Conv2DTranspose(filters=3, kernel_size=(5, 5), kernel_initializer=generator_kernel_initializer, strides=(2, 2), padding='same', use_bias=False))
    # Tanh activation function is used to scale the values between -1 and 1 (same range as the preprocessed real images).
    model.add(Activation(activation='tanh'))

    assert model.output_shape == (None, 64, 64, 3)

    return model

NOISE_DIMENSIONS = 100
checkpoint_weights_file_url = os.path.join('src', 'assets', 'checkpoints', 'generator', 'weights.epoch-472.loss-1.5988234281539917.h5')
trained_generator_model_dir_url = 'public'

generator = create_generator(NOISE_DIMENSIONS)
generator.load_weights(filepath=checkpoint_weights_file_url)
generator.save(filepath=os.path.join(trained_generator_model_dir_url, 'generator-model.h5'), overwrite=True)
tfjs.converters.save_keras_model(generator, trained_generator_model_dir_url)
