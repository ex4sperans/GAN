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
z_dim = 64
load_path = 'models/mnist_gan/m'

###############################################################################

model = GAN(
            input_dim=input_dim,
            latent_dim=z_dim,
            generator_architechture=[512],
            discriminator_architechture=[512],
            scope='GAN',
            mode='inference')

model.load_model(load_path)

def to_image(x):
    return x.reshape(28, 28)

def sample():

    fig = plt.figure(figsize=(4, 4))
    for i, x in enumerate(model.sample(100)):
        ax = fig.add_subplot(10, 10, i+1) 
        ax.imshow(to_image(x), cmap='gray', aspect='auto')
        ax.set_axis_off()

    #remove spacings between subplots
    fig.subplots_adjust(hspace=0, wspace=0, left=0, bottom=0, right=1, top=1)
    os.makedirs('pics', exist_ok=True)
    fig.savefig('pics/sample_from_latent_space.png')
    plt.close()
    print('Sample saved.')

    
if __name__ == '__main__':

    sample()
