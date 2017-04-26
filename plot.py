import os
import json
import argparse

import numpy as np 
import matplotlib.pyplot as plt

from models.gan import GAN
from models.wgan import WGAN
from models.dcgan import DCGAN

models = {
          'GAN': GAN,
          'WGAN': WGAN,
          'DCGAN': DCGAN}

parser = argparse.ArgumentParser(
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
                    '--model', type=str, required=True,
                        choices=list(models.keys()), help='model type')
args = parser.parse_args()

with open('model_params.json', 'r') as f:
    model_params = json.load(f)[args.model]

kwargs = dict(
              input_dim=model_params['input_dim'],
              latent_dim=model_params['latent_dim'],
              generator_architechture=model_params['generator_architechture'],
              discriminator_architechture=model_params['discriminator_architechture'],
              scope=model_params['scope'],
              mode='inference') 

if args.model == 'DCGAN':
    kwargs['reshaped_z_shape'] = model_params['reshaped_z_shape']
    kwargs['reshaped_x_shape'] = model_params['reshaped_x_shape']   

model = models[args.model]
model = model(**kwargs)

load_path = model_params['model_path']
model.load_model(load_path)

def to_image(x):
    return x.reshape(28, 28)

def sample():

    fig = plt.figure(figsize=(4, 4))
    for i, x in enumerate(model.sample(100)):
        ax = fig.add_subplot(10, 10, i+1) 
        ax.imshow(to_image(x), cmap='gray', aspect='auto', interpolation='bicubic')
        ax.set_axis_off()

    #remove spacings between subplots
    fig.subplots_adjust(hspace=0, wspace=0, left=0, bottom=0, right=1, top=1)
    os.makedirs('pics', exist_ok=True)
    fig.savefig('pics/sample_{}.png'.format(model.scope))
    plt.close()
    print('Sample saved.')
    
if __name__ == '__main__':

    sample()
