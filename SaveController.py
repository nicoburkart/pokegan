import tensorflow as tf
import os
from tensorflow.keras import models


# Used to handle saving weights during training or loading weights and continuing training or saving/loading whole models. For each model a new instance has to be created.

# Checks if for this specific model any checkpoint weights have been saved before.
def are_checkpoint_weights_available(generator_checkpoints_dir_url: str, discriminator_checkpoints_dir_url: str):
    if len(os.listdir(path=generator_checkpoints_dir_url)) > 0 and len(os.listdir(path=discriminator_checkpoints_dir_url)) > 0:
        print('Found previous weights for the generator and discriminator\n')
        return True
    else:
        print('No previous weights found\n')
        return False


# Loads checkpoint weights provided at the passed url. If no url is passed it loads the latest saved checkpoints.
def load_checkpoint_weights(generator: models.Sequential, discriminator: models.Sequential, generator_checkpoints_dir_url: str, discriminator_checkpoints_dir_url: str):
    latest_generator_checkpoint_file_url = get_latest_created_file_url(dir_url=generator_checkpoints_dir_url)
    generator.load_weights(filepath=latest_generator_checkpoint_file_url)
    print(f'Loaded generator weights from the latest checkpoint file {latest_generator_checkpoint_file_url}')

    latest_discriminator_checkpoint_file_url = get_latest_created_file_url(dir_url=discriminator_checkpoints_dir_url)
    discriminator.load_weights(filepath=latest_discriminator_checkpoint_file_url)
    print(f'Loaded discriminator weights from the latest checkpoint file {latest_discriminator_checkpoint_file_url}\n')

    generator_last_completed_epoch = int(latest_generator_checkpoint_file_url.split('.')[1][6:])
    discriminator_last_completed_epoch = int(latest_discriminator_checkpoint_file_url.split('.')[1][6:])

    assert generator_last_completed_epoch == discriminator_last_completed_epoch

    return generator_last_completed_epoch


def get_latest_created_file_url(dir_url: str):
    file_names = os.listdir(path=dir_url)
    file_creation_times = []
    for file_name in file_names:
        file_creation_time = os.path.getctime(filename=os.path.join(dir_url, file_name))
        file_creation_times.append(file_creation_time)
    latest_created_file_name, _ = sorted(zip(file_names, file_creation_times), key=lambda tuple: tuple[1], reverse=True)[0]
    return os.path.join(dir_url, latest_created_file_name)

def save_checkpoint_weights(epoch: int, generator: models.Sequential, discriminator: models.Sequential, generator_checkpoints_dir_url: str, discriminator_checkpoints_dir_url: str, generator_loss: float, discriminator_loss: float):
    generator_checkpoint_file_url = os.path.join(generator_checkpoints_dir_url, f'weights.epoch-{epoch}.loss-{generator_loss}.h5')
    generator.save_weights(filepath=generator_checkpoint_file_url)
    # print(f'Saved generator weights to file {generator_checkpoint_file_url}')

    discriminator_checkpoint_file_url = os.path.join(discriminator_checkpoints_dir_url, f'weights.epoch-{epoch}.loss-{discriminator_loss}.h5')
    discriminator.save_weights(filepath=discriminator_checkpoint_file_url)
    # print(f'Saved discriminator weights to file {discriminator_checkpoint_file_url}')


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
