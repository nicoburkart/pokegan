from matplotlib import pyplot as plt


def plot_losses(generator_losses: list, discriminator_losses: list, epochs: list):
    _, axis = plt.subplots()
    axis.set_title('Loss')
    axis.set_xlabel('Epoch')
    axis.set_ylabel('Loss')
    axis.grid()
    axis.plot(epochs, generator_losses, label='Generator')
    axis.plot(epochs, discriminator_losses, label='Discriminator')
    axis.legend()


def plot_discriminator_accuracies(discriminator_accuracies: list, epochs: list):
    _, axis = plt.subplots()
    axis.set_title('Accuracy')
    axis.set_xlabel('Epoch')
    axis.set_ylabel('Accuracy')
    axis.grid()
    axis.plot(epochs, discriminator_accuracies)
