from matplotlib import pyplot as plt
import os

# This module provides functions to evaluate the results after training.


def plot_and_save_losses(generator_losses: list, discriminator_real_losses: list, discriminator_fake_losses: list, discriminator_total_losses: list, epochs: list, generated_figures_dir_url: str):
    figure, axes = plt.subplots(2, 1)
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid()
    axes[0].plot(epochs, generator_losses, label='Generator loss')
    axes[0].plot(epochs, discriminator_real_losses, label='Discriminator real loss')
    axes[0].plot(epochs, discriminator_fake_losses, label='Discriminator fake loss')
    axes[0].legend()
    axes[1].set_title('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].grid()
    axes[1].plot(epochs, generator_losses, label='Generator loss')
    axes[1].plot(epochs, discriminator_total_losses, label='Discriminator total loss')
    axes[1].legend()
    generated_figure_file_url = os.path.join(generated_figures_dir_url, 'loss.png')
    figure.savefig(generated_figure_file_url)
    print(f'\nLoss figure saved at:\n{generated_figure_file_url}')


def plot_and_save_discriminator_accuracies(discriminator_real_accuracies: list, discriminator_fake_accuracies: list, discriminator_total_accuracies: list, epochs: list, generated_figures_dir_url: str):
    figure, axes = plt.subplots(2, 1)
    axes[0].set_title('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].grid()
    axes[0].plot(epochs, discriminator_real_accuracies, label='Discriminator real accuracy')
    axes[0].plot(epochs, discriminator_fake_accuracies, label='Discriminator fake accuracy')
    axes[0].legend()
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].grid()
    axes[1].plot(epochs, discriminator_total_accuracies, label='Discriminator total accuracy')
    axes[1].legend()
    generated_figure_file_url = os.path.join(generated_figures_dir_url, 'accuracy.png')
    figure.savefig(generated_figure_file_url)
    print(f'\nAccuracy figure saved at:\n{generated_figure_file_url}')
