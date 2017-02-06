import argparse
import os
import json

from gan import GAN

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
train_set = mnist.train
valid_set = mnist.train

parser = argparse.ArgumentParser(
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
                    '--batch_size',  type=int,
                        default=32, help='batch size')
parser.add_argument(
                    '--num_epochs', type=int,
                        default=1000, help='maximum number of training epochs')
parser.add_argument(
                    '--learning_rate', type=float,
                        default=0.001, help='learning rate')

load_parser = parser.add_mutually_exclusive_group(required=False)
load_parser.add_argument('--restore', dest='restore', action='store_false')
parser.set_defaults(restore=False)

args = parser.parse_args()

with open('model_params.json', 'r') as f:
    model_params = json.load(f)

save_path = model_params['model_path']
os.makedirs(os.path.dirname(save_path), exist_ok=True)

model = GAN(
            input_dim=model_params['input_dim'],
            latent_dim=model_params['latent_dim'],
            generator_architechture=model_params['generator_architechture'],
            discriminator_architechture=model_params['discriminator_architechture'],
            scope=model_params['scope'],
            mode='train')

if args.restore:
      model.load_model(save_path)

print('\nTraining {} with the following parameters:'.format(model.scope))
print('Latent dim: {}'.format(model_params['latent_dim']))
print('Generator architechture: {}'.format(model.generator_architechture))
print('Discriminator architechture: {}'.format(model.discriminator_architechture))
print('Learning rate: {}'.format(args.learning_rate))

for epoch in range(1, args.num_epochs + 1):

    print('\n', '-'*30, 'Epoch {}'.format(epoch), '-'*30, '\n')
    model.train(train_set, args.learning_rate, args.batch_size, args.batch_size)
    model.predict(valid_set)
    model.save_model(save_path)
