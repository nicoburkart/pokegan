from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import os
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import imageio
from tensorflow.keras import models


def load_all_images(image_dir_url: str):
    image_urls = get_image_urls(image_dir_url)
    images = []
    for image_url in image_urls:
        image = load_img(path=image_url)
        image = img_to_array(img=image)
        images.append(image)
    return np.asarray(images)


# Plots 16 random images inside the set image directory in a grid.
def plot_random_images(image_dir_url: str):
    image_urls = get_image_urls(image_dir_url)
    random_image_urls = np.random.choice(a=image_urls, size=(16,))

    _, axes = plt.subplots(nrows=4, ncols=4)

    for axis, random_image_url in zip(axes.flatten(), random_image_urls):
        random_image = mpimg.imread(fname=random_image_url)
        axis.imshow(random_image)
        axis.axis('off')


def get_image_urls(image_dir_url: str):
    image_urls = [os.path.join(image_dir_url, filename) for filename in os.listdir(image_dir_url)]
    return image_urls


def generate_and_plot_images(generator: models.Sequential, seed):
    generated_images_batch = generator(seed, training=False)

    _, axes = plt.subplots(nrows=4, ncols=4, figsize=(4, 4))

    for axis, generated_image in zip(axes.flatten(), generated_images_batch):
        # The image has to be rescaled. Right now the ouput of the generator is an image with values between -1 and 1.
        generated_image = generated_image.numpy() * 127.5 + 127.5
        generated_image = generated_image.astype('uint8')
        axis.imshow(generated_image)
        axis.axis('off')


def generate_and_save_images(generator: models.Sequential, seed, output_url: str):
    generated_images_batch = generator(seed, training=False)

    figure, axes = plt.subplots(nrows=4, ncols=4, figsize=(4, 4))

    for axis, generated_image in zip(axes.flatten(), generated_images_batch):
        # The image has to be rescaled. Right now the ouput of the generator is an image with values between -1 and 1.
        generated_image = generated_image.numpy() * 127.5 + 127.5
        generated_image = generated_image.astype('uint8')
        axis.imshow(generated_image)
        axis.axis('off')

    figure.savefig(output_url)
    plt.close(figure)


def create_and_save_gif(image_dir_url: str, output_url: str):
    with imageio.get_writer(uri=output_url, mode='I') as writer:
        filenames = os.listdir(image_dir_url)
        # Sorts the images based on their epoch in ascending order
        filenames = sorted(filenames, key=lambda filename: int(filename[:-4].split('-')[-1]))

        for filename in filenames:
            filename = os.path.join(image_dir_url, filename)
            image = imageio.imread(uri=filename)
            writer.append_data(image)

        print(f'Saved gif at: {output_url}')
