from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import os
import random
import math
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# Used to plot images while preprocessing when using images as data.
class ImagePreprocessor:
    def __init__(self, image_dir_url: str):
        self.image_dir_url = image_dir_url

    # Plots 16 random images inside the set image directory in a grid.
    def plot_random_images(self):
        image_urls = self.get_image_urls()
        random_image_urls = random.choices(population=image_urls, k=16)

        figure, axes = plt.subplots(nrows=4, ncols=4)

        for index, random_image_url in enumerate(random_image_urls):
            random_image = mpimg.imread(fname=random_image_url)
            axis = axes[math.floor(index / 4)][index % 4]
            axis.imshow(random_image)
            axis.axis('off')

        figure.show()

    def get_image_urls(self):
        image_urls = [os.path.join(self.image_dir_url, filename) for filename in os.listdir(self.image_dir_url)]
        return image_urls

    # Plots 16 images derived from one image. Images are derived depending on the parameters set int the ImageDataGenerator passed as a parameter.
    def plot_augmented_random_image(self, image_data_generator: ImageDataGenerator):
        image_urls = self.get_image_urls()
        random_image_url = random.choice(image_urls)
        random_image = load_img(path=random_image_url)
        x = img_to_array(img=random_image)
        x = x.reshape((1,) + x.shape)

        figure, axes = plt.subplots(nrows=4, ncols=4)

        for index, (augmented_x,) in enumerate(image_data_generator.flow(x, batch_size=1)):
            if index == 16:
                break
            axis = axes[math.floor(index / 4)][index % 4]
            augmented_random_image = array_to_img(x=augmented_x)
            axis.imshow(augmented_random_image)
            axis.axis('off')

        figure.show()


# Following lines can be used to see the class in action and to see how the methods can be used.
# image_preprocessor = ImagePreprocessor(image_dir_url=os.path.join('data', 'smd'))
# image_preprocessor.plot_random_images()
# image_data_generator = ImageDataGenerator(rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
# image_preprocessor.plot_augmented_random_image(image_data_generator)
