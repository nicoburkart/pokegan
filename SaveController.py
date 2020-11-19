import tensorflow as tf
import os
from tensorflow.keras import models


# A controller object which can handle saving of weights during training or loading weights and continuing training or saving/loading whole models. For each model a new instance has to be created.
class SaveController:
    def __init__(self, model_name: str):
        checkpoint_dir_url = os.path.join('checkpoints', model_name)
        model_dir_url = os.path.join('models', model_name)

        self.create_directories(checkpoint_dir_url, model_dir_url)

        self.checkpoint_file_url = os.path.join(checkpoint_dir_url, 'checkpoint.h5')
        self.model_file_url = os.path.join(model_dir_url, 'model.h5')

        self.save_checkpoint_weights = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_file_url, save_weights_only=True, verbose=1)

    def create_directories(self, checkpoint_dir_url: str, model_dir_url: str):
        if not os.path.isdir(checkpoint_dir_url):
            os.mkdir(path=checkpoint_dir_url)
        if not os.path.isdir(model_dir_url):
            os.mkdir(path=model_dir_url)

    def are_checkpoint_weights_available(self):
        if os.path.isfile(self.checkpoint_file_url):
            print('Found previous weights')
            return True
        else:
            print('No previous weights found')
            return False

    def load_previous_checkpoint_weights(self, model: models.Sequential):
        model.load_weights(filepath=self.checkpoint_file_url)
        print(f'Loaded previous weights from: {self.checkpoint_file_url}')

    def load_model(self):
        model = models.load_model(filepath=self.model_file_url)
        print(f'Loaded model from: {self.model_file_url}')
        return model

    def save_model(self, model: models.Sequential):
        model.save(filepath=self.model_file_url)
        print(f'Saved model to: {self.model_file_url}')