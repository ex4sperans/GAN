from gan import GAN
import os

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
train_set = mnist.train
valid_set = mnist.train

###############################################################################

batch_size = 64
input_dim = 28*28
z_dim = 32
learning_rate = 1e-4
n_epochs = 500
save_path = 'models/mnist_gan/m'
load_path = save_path
resume = False
os.makedirs('models/mnist_gan', exist_ok=True)

###############################################################################

model = GAN(
            input_dim=input_dim,
            latent_dim=z_dim,
            generator_architechture=[64, 128],
            discriminator_architechture=[128, 64],
            scope='GAN',
            num_samples=batch_size,
            mode='train')

if resume:
	model.load_model(load_path)

for epoch in range(1, n_epochs+1):
	print('\n', '-'*30, 'Epoch {}'.format(epoch), '-'*30, '\n')
	model.train(batch_size, learning_rate, train_set, epoch)
	model.predict(valid_set)
	model.save_model(save_path)