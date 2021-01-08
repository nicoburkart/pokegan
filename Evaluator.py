from matplotlib import pyplot as plt
import os


def plot_and_save_losses(generator_losses: list, discriminator_losses: list, epochs: list, generated_figures_dir_url: str):
    figure, axis = plt.subplots()
    axis.set_title('Loss')
    axis.set_xlabel('Epoch')
    axis.set_ylabel('Loss')
    axis.grid()
    axis.plot(epochs, generator_losses, label='Generator')
    axis.plot(epochs, discriminator_losses, label='Discriminator')
    axis.legend()
    generated_figure_file_url = os.path.join(generated_figures_dir_url, 'losses.png')
    figure.savefig(generated_figure_file_url)
    print(f'\nLosses figure saved at:\n{generated_figure_file_url}')


def plot_and_save_discriminator_accuracies(discriminator_accuracies: list, epochs: list, generated_figures_dir_url: str):
    figure, axis = plt.subplots()
    axis.set_title('Accuracy')
    axis.set_xlabel('Epoch')
    axis.set_ylabel('Accuracy')
    axis.grid()
    axis.plot(epochs, discriminator_accuracies)
    generated_figure_file_url = os.path.join(generated_figures_dir_url, 'disc-accuracy.png')
    figure.savefig(generated_figure_file_url)
    print(f'\nAccuracies figure saved at:\n{generated_figure_file_url}')
