import tensorflow as tf
from tensorflow.contrib import slim

from models.gan import GAN

class WGAN(GAN):

    """A class representing Generative Adversarial Network
    with Wasserstein distance as loss function"""

    def __init__(self, input_dim, latent_dim, scope,
                 generator_architechture, discriminator_architechture,
                 mode='train'):

        super(WGAN, self).__init__(input_dim, latent_dim, scope,
            generator_architechture, discriminator_architechture, mode)
        self.activation = tf.nn.relu

        self._create_clip_op()

    def _discriminate(self, x):

        net = x
        for i, layer_size in enumerate(self.discriminator_architechture):
            net = slim.fully_connected(
                                       net,
                                       layer_size,
                                       self.activation,
                                       scope='layer{}'.format(i+1))
            net = tf.nn.dropout(net, 0.9)

        d = slim.fully_connected(
                                 net,
                                 1,
                                 None,
                                 scope='layer_out')
        return d 

    def _create_loss(self):
        # Wasserstein distance
        self.discriminator_loss = tf.reduce_mean(self.data_d) - tf.reduce_mean(self.model_d)
        self.generator_loss = tf.reduce_mean(self.model_d)

    def _create_clip_op(self):
        self.clip_weights = list(v.assign(tf.clip_by_value(v, -0.1, 0.1))
                             for v in self.discriminator_vars)  

    def train(self, train_data, learning_rate, batch_size, num_samples):

        for step in range(train_data.num_examples//batch_size):

            batch, _ = train_data.next_batch(batch_size)
            discriminator_feed = {
                                  self.inputs: batch,
                                  self.learning_rate: learning_rate,
                                  self.num_samples: num_samples,
                                  self.is_training: True}
            generator_feed = {
                              self.learning_rate: learning_rate,
                              self.num_samples: num_samples}
            
            self.sess.run(self.train_discriminator, discriminator_feed)
            # according to the paper
            if step % 5 == 0:
                self.sess.run(self.train_generator, generator_feed)
            # crop weights to speed up discriminator convergence
            self.sess.run(self.clip_weights)