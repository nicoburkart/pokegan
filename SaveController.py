import os
from tensorflow.keras import models
import time


def create_checkpoint_dirs(generator_checkpoints_dir_url: str, discriminator_checkpoints_dir_url: str):
    os.makedirs(generator_checkpoints_dir_url, exist_ok=True)
    os.makedirs(discriminator_checkpoints_dir_url, exist_ok=True)


def get_new_generated_media_subdir_urls(generated_media_dir_url):
    current_date_time = time.strftime('%Y-%m-%dT%H-%M-%S%z')
    generated_images_dir_url = os.path.join(generated_media_dir_url, current_date_time, 'images')
    generated_gifs_dir_url = os.path.join(generated_media_dir_url, current_date_time, 'gifs')
    generated_figures_dir_url = os.path.join(generated_media_dir_url, current_date_time, 'figures')
    return generated_images_dir_url, generated_gifs_dir_url, generated_figures_dir_url


def create_generated_media_dirs(generated_images_dir_url: str, generated_gifs_dir_url: str, generated_figures_dir_url: str):
    os.makedirs(generated_images_dir_url, exist_ok=True)
    os.makedirs(generated_gifs_dir_url, exist_ok=True)
    os.makedirs(generated_figures_dir_url, exist_ok=True)


def delete_checkpoint_files(generator_checkpoints_dir_url: str, discriminator_checkpoints_dir_url: str):
    delete_dir_files(generator_checkpoints_dir_url)
    delete_dir_files(discriminator_checkpoints_dir_url)


def delete_dir_files(dir_url):
    file_names = os.listdir(dir_url)
    for file_name in file_names:
        file_url = os.path.join(dir_url, file_name)
        os.remove(file_url)


# Checks if for this specific model any checkpoint files have been saved before.
def do_checkpoint_files_exist(generator_checkpoints_dir_url: str, discriminator_checkpoints_dir_url: str):
    if not do_checkpoint_dirs_exist(generator_checkpoints_dir_url, discriminator_checkpoints_dir_url) or (len(os.listdir(path=generator_checkpoints_dir_url)) == 0 or len(os.listdir(path=discriminator_checkpoints_dir_url)) == 0):
        return False
    else:
        return True#


def do_checkpoint_dirs_exist(generator_checkpoints_dir_url: str, discriminator_checkpoints_dir_url: str):
    return os.path.isdir(generator_checkpoints_dir_url) and os.path.isdir(discriminator_checkpoints_dir_url)


# Loads latest checkpoint files at the given urls
def load_latest_checkpoint_files(generator: models.Sequential, discriminator: models.Sequential, generator_checkpoints_dir_url: str, discriminator_checkpoints_dir_url: str):
    latest_generator_checkpoint_file_url = get_last_created_file_url(dir_url=generator_checkpoints_dir_url)
    generator.load_weights(filepath=latest_generator_checkpoint_file_url)
    print(f'Loaded generator weights from the latest checkpoint file {latest_generator_checkpoint_file_url}')

    latest_discriminator_checkpoint_file_url = get_last_created_file_url(dir_url=discriminator_checkpoints_dir_url)
    discriminator.load_weights(filepath=latest_discriminator_checkpoint_file_url)
    print(f'Loaded discriminator weights from the latest checkpoint file {latest_discriminator_checkpoint_file_url}\n')

    generator_last_completed_epoch = int(latest_generator_checkpoint_file_url.split('.')[1][6:])
    discriminator_last_completed_epoch = int(latest_discriminator_checkpoint_file_url.split('.')[1][6:])

    assert generator_last_completed_epoch == discriminator_last_completed_epoch

    return generator_last_completed_epoch


def get_last_created_file_url(dir_url: str):
    file_names = os.listdir(path=dir_url)
    file_creation_times = []
    for file_name in file_names:
        file_creation_time = os.path.getctime(filename=os.path.join(dir_url, file_name))
        file_creation_times.append(file_creation_time)
    last_created_file_name = sorted(zip(file_names, file_creation_times), key=lambda file_name_file_creation_time_tuple: file_name_file_creation_time_tuple[1], reverse=True)[0][0]
    return os.path.join(dir_url, last_created_file_name)


def get_last_created_subdir_url(dir_url: str):
    subdir_urls = [x[0] for x in os.walk(dir_url)]
    subdir_creation_times = []
    for subdir_url in subdir_urls:
        subdir_creation_time = os.path.getctime(filename=subdir_url)
        subdir_creation_times.append(subdir_creation_time)
    last_created_subdir_url = sorted(zip(subdir_urls, subdir_creation_times), key=lambda subdir_url_subdir_creation_time_tuple: subdir_url_subdir_creation_time_tuple[1], reverse=True)[0][0]
    return last_created_subdir_url


def save_checkpoint_weights(epoch: int, generator: models.Sequential, discriminator: models.Sequential, generator_checkpoints_dir_url: str, discriminator_checkpoints_dir_url: str, generator_loss: float, discriminator_loss: float):
    generator_checkpoint_file_url = os.path.join(generator_checkpoints_dir_url, f'weights.epoch-{epoch}.loss-{generator_loss}.h5')
    generator.save_weights(filepath=generator_checkpoint_file_url)

    discriminator_checkpoint_file_url = os.path.join(discriminator_checkpoints_dir_url, f'weights.epoch-{epoch}.loss-{discriminator_loss}.h5')
    discriminator.save_weights(filepath=discriminator_checkpoint_file_url)


# # Loads an entire model from the url constructed with the help of the model name provided in the constructor.
# def load_model(self):
#     model = models.load_model(filepath=self.model_file_url)
#     print(f'Loaded model from: {self.model_file_url}')
#     return model
#
# # Saves an entire model to the url constructed with the help of the model name provided in the constructor.
# def save_model(self, model: models.Sequential):
#     model.save(filepath=self.model_file_url, save_format='h5')
#     print(f'Saved model to: {self.model_file_url}')
