import argparse
import os
import json

from models.gan import GAN
from models.wgan import WGAN
from models.dcgan import DCGAN

from plot import sample

# mapping from model names to the corresponding classes
models = {
          'GAN': GAN,
          'WGAN': WGAN,
          'DCGAN': DCGAN}

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
train_set = mnist.train
valid_set = mnist.validation

parser = argparse.ArgumentParser(
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
                    '--batch_size',  type=int,
                        default=64, help='batch size')
parser.add_argument(
                    '--num_epochs', type=int,
                        default=100, help='maximum number of training epochs')
parser.add_argument(
                    '--learning_rate', type=float,
                        default=0.0005, help='learning rate')
parser.add_argument(
                    '--model', type=str, required=True,
                        choices=list(models.keys()), help='model type')

load_parser = parser.add_mutually_exclusive_group(required=False)
load_parser.add_argument('--restore', dest='restore', action='store_true')
parser.set_defaults(restore=False)

args = parser.parse_args()

# all the parameters are contained in the JSON file, so load them
with open('model_params.json', 'r') as f:
    model_params = json.load(f)[args.model]

kwargs = dict(
              input_dim=model_params['input_dim'],
              latent_dim=model_params['latent_dim'],
              generator_architechture=model_params['generator_architechture'],
              discriminator_architechture=model_params['discriminator_architechture'],
              scope=model_params['scope'],
              mode='train')

if args.model == 'DCGAN':
    kwargs['reshaped_z_shape'] = model_params['reshaped_z_shape']
    kwargs['reshaped_x_shape'] = model_params['reshaped_x_shape']

model = models[args.model]
model = model(**kwargs)

save_path = model_params['model_path']
os.makedirs(os.path.dirname(save_path), exist_ok=True)

if args.restore:
      model.load_model(save_path)

print('\nTraining {} with the following parameters:'.format(model.scope))
print('Latent dim: {}'.format(model_params['latent_dim']))
print('Generator architechture: {}'.format(model.generator_architechture))
print('Discriminator architechture: {}'.format(model.discriminator_architechture))
print('Learning rate: {}'.format(args.learning_rate))
print('Minibatch size: {}'.format(args.batch_size))
print('Number of epochs: {}'.format(args.num_epochs))

for epoch in range(1, args.num_epochs + 1):

    print('\n' + '-'*30, 'Epoch {}'.format(epoch), '-'*30, '\n')
    model.train(train_set, args.learning_rate, args.batch_size, args.batch_size)
    model.predict(valid_set)
    model.save_model(save_path)
    sample(model, epoch)