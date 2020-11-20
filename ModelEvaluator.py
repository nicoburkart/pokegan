import os
from tensorflow.keras import models
from matplotlib import pyplot as plt
import numpy as np


# Used to compare the metrics of different models.
class ModelEvaluator:


    # Plots losses of all models inside the model directory onto a bar graph.
    def compare_test_loss_of_all_models(self, test_features, test_labels):
        model_dir_names = os.listdir('models')

        fig, ax = plt.subplots()

        for model_dir_name in model_dir_names:
            model = models.load_model(filepath=os.path.join('models', model_dir_name, 'model.h5'))
            loss = model.evaluate(x=test_features, y=test_labels)
            ax.bar(model_dir_name, loss, 0.5)

        plt.show()


# Following lines can be used to see the class in action and to see how the methods can be used.
# test_feature = np.array([2, 8, 11, 16, 18])
# test_label = np.array([33, 34, 45, 48, 50])
#
# model_evaluator = ModelEvaluator()
# model_evaluator.compare_test_loss_of_all_models(test_feature, test_label)
