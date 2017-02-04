import random
import os

import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from gan import GAN


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

###############################################################################

input_dim = 28*28
z_dim = 32
load_path = 'models/mnist_gan/m'

###############################################################################

model = GAN(
            input_dim=input_dim,
            latent_dim=z_dim,
            generator_architechture=[64, 128],
            discriminator_architechture=[128, 64],
            scope='GAN',
            num_samples=1,
            mode='inference')

model.load_model(load_path)

def to_image(x):
    return x.reshape(28, 28)

def sample():

    fig = plt.figure(figsize=(7, 7))
    for i in range(225):
        ax = fig.add_subplot(15, 15, i+1)
        x = model.sample()
        ax.imshow(to_image(x), cmap='gray', aspect='auto')
        ax.set_axis_off()

    #remove spacings between subplots
    fig.subplots_adjust(hspace=0, wspace=0)
    os.makedirs('pics', exist_ok=True)
    fig.savefig('pics/sample_from_latent_space.png', tight_layout=True)
    plt.close()
    print('Sample saved.')

    
if __name__ == '__main__':

    sample()
