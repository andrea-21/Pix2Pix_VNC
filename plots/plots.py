import matplotlib.pyplot as plt
import os
import numpy as np


def generate_images(model, test_input, tar, plot = False):
  prediction = model(test_input, training=True)
  if plot == True:
      plt.figure(figsize=(15, 15))
    
      display_list = [test_input[0], tar[0], prediction[0]]
      title = ['Input Image', 'Ground Truth', 'Predicted Image']
    
      for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        # Getting the pixel values in the [0, 1] range to plot.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
      plt.show()

# Funzione per craere plot delle loss a fine training
def save_loss_plot(losses, keys, title, filename, log_dir):
    plt.figure(figsize=(8, 5))
    
    for key in keys:
        plt.plot(losses[key], label=key)
    
    plt.legend()
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    
    filepath = os.path.join(log_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

    # Funzione per craere plot delle loss a fine training
def save_loss_plot_lambda(losses, keys, title, filename, log_dir, _lambda = 1):
    """
    Solo 2 key
    Mettere sempre prima la key della loss l1 se si vuole usare lambda
    """
    plt.figure(figsize=(8, 5))
    plt.plot([x*_lambda for x in losses[keys[0]]], label=keys[0])
    plt.plot(losses[keys[1]], label=keys[1])
    
    plt.legend()
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    
    filepath = os.path.join(log_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

def apply_window(image, window_center, window_width):
    img_min = window_center - window_width / 2
    img_max = window_center + window_width / 2
    windowed = np.clip(image, img_min, img_max)
    windowed = (windowed - img_min) / (img_max - img_min)  # Normalize to [0, 1]
    return windowed