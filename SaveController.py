import tensorflow as tf
import os
from tensorflow.keras import models


# Used to handle saving weights during training or loading weights and continuing training or saving/loading whole models. For each model a new instance has to be created.
class SaveController:
    def __init__(self, model_name: str):
        self.checkpoint_dir_url = os.path.join('checkpoints', model_name)
        model_dir_url = os.path.join('models', model_name)

        self.create_directories(self.checkpoint_dir_url, model_dir_url)

        checkpoint_file_url = os.path.join(self.checkpoint_dir_url, 'weights.{epoch:02d}-{val_loss:.2f}.h5')
        self.model_file_url = os.path.join(model_dir_url, 'model.h5')

        self.save_checkpoint_weights = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_file_url, save_weights_only=True, verbose=1)

    def create_directories(self, checkpoint_dir_url: str, model_dir_url: str):
        if not os.path.isdir(checkpoint_dir_url):
            os.mkdir(path=checkpoint_dir_url)
        if not os.path.isdir(model_dir_url):
            os.mkdir(path=model_dir_url)

    # Checks if for this specific model any checkpoint weights have been saved before.
    def are_checkpoint_weights_available(self):
        if len(os.listdir(path=self.checkpoint_dir_url)) > 0:
            print('Found previous weights')
            return True
        else:
            print('No previous weights found')
            return False

    # Loads checkpoint weights provided at the passed url. If no url is passed it loads the latest saved checkpoints.
    def load_checkpoint_weights(self, model: models.Sequential, checkpoint_file_name: str = None):
        checkpoint_file_url = ''
        if checkpoint_file_name is None:
            latest_checkpoint_file_name = self.get_latest_checkpoint_file_name()
            checkpoint_file_url = os.path.join(self.checkpoint_dir_url, latest_checkpoint_file_name)
            print('No checkpoint_file_name provided. Loading in weights from the latest checkpoint file...')
        else:
            checkpoint_file_url = os.path.join(self.checkpoint_dir_url, checkpoint_file_name)

        model.load_weights(filepath=checkpoint_file_url)
        print(f'Loaded weights from: {checkpoint_file_url}')

    def get_latest_checkpoint_file_name(self):
        checkpoint_files = []
        for checkpoint_file_name in os.listdir(path=self.checkpoint_dir_url):
            creation_time = os.path.getctime(filename=os.path.join(self.checkpoint_dir_url, checkpoint_file_name))
            checkpoint_files.append({'name': checkpoint_file_name, 'creation_time': creation_time})
        checkpoint_files.sort(key=lambda checkpoint_file: checkpoint_file.get('creation_time'), reverse=True)
        return checkpoint_files[0].get('name')

    # Loads an entire model from the url constructed with the help of the model name provided in the constructor.
    def load_model(self):
        model = models.load_model(filepath=self.model_file_url)
        print(f'Loaded model from: {self.model_file_url}')
        return model

    # Saves an entire model to the url constructed with the help of the model name provided in the constructor.
    def save_model(self, model: models.Sequential):
        model.save(filepath=self.model_file_url, save_format='h5')
        print(f'Saved model to: {self.model_file_url}')