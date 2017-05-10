import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np

from models.gan import GAN

class DCGAN(GAN):

    """A class representing Deep Convolutional Generative Adversarial Network"""

    def __init__(self, input_dim, latent_dim, reshaped_z_shape, reshaped_x_shape,
                 scope, generator_architechture, discriminator_architechture,
                 mode='train'):

        self.reshaped_z_shape = reshaped_z_shape
        self.reshaped_x_shape = reshaped_x_shape
        super(DCGAN, self).__init__(input_dim, latent_dim, scope,
            generator_architechture, discriminator_architechture,
            mode=mode, activation=tf.nn.relu)

    def _leaky_relu(self, x, a):
        return tf.nn.relu(x) - a*tf.nn.relu(-x)

    def _discriminate(self, x):

        net = x
        # reshape x to be in form of an image
        net = tf.reshape(x, [-1] + self.reshaped_x_shape)
        net = self._batch_norm(net, 'input')

        initializer = tf.random_normal_initializer(stddev=0.02)

        for i, layer_params in enumerate(self.discriminator_architechture):

            depth = layer_params['depth']
            stride = layer_params['stride']
            padding = layer_params['padding']
            kernel_size = layer_params['kernel_size']

            net = slim.conv2d(
                              net,
                              depth,
                              stride=stride,
                              kernel_size=kernel_size,
                              activation_fn=None,
                              padding=padding,
                              weights_initializer=initializer,
                              scope='layer{}'.format(i+1))

            net = self._batch_norm(net, scope='batch_norm{}'.format(i+1))
            net = self._leaky_relu(net, 0.2)
        net = slim.flatten(net)

        d = slim.fully_connected(
                                 net,
                                 1,
                                 tf.nn.sigmoid,
                                 scope='layer_out')
        return d 

    def _generate(self):

        z = self._gaussian_sample()
        net = z
        # project and reshape
        net = slim.fully_connected(
                                   net,
                                   int(np.prod(self.reshaped_z_shape)),
                                   activation_fn=None)
        net = tf.reshape(net, [-1] + self.reshaped_z_shape)
        net = self._batch_norm(net, 'input')

        initializer = tf.random_normal_initializer(stddev=0.02)

        for i, layer_params in enumerate(self.generator_architechture):

            stride = layer_params['stride']
            depth = layer_params['depth']
            padding = layer_params['padding']
            kernel_size = layer_params['kernel_size']

            net = slim.conv2d_transpose(
                                        net,
                                        depth,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=padding,
                                        activation_fn=None,
                                        weights_initializer=initializer,
                                        scope='layer{}'.format(i+1))

            net = self._batch_norm(net, scope='batch_norm{}'.format(i+1))

            # no activation of the last layer
            if i+1 < len(self.generator_architechture):
                net = self.activation(net)

        net = slim.flatten(net)
        sample = tf.nn.sigmoid(net)

        return sample

    def _create_optimizer(self, vars_, loss, opt_scope, clip=None):

        with tf.variable_scope('Optimizer_' + opt_scope + '_' + self.scope):
            # according to the paper
            optimizer = tf.train.AdamOptimizer(
                                        learning_rate=self.learning_rate,
                                        beta1=0.5)
            grads_and_vars = optimizer.compute_gradients(
                                                         loss=loss,
                                                         var_list=vars_)
            if clip is not None:
                grads_and_vars = [(tf.clip_by_value(g, -clip, clip), v)
                                  for g, v in grads_and_vars]
            return optimizer.apply_gradients(grads_and_vars)
