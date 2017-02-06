import tensorflow as tf
from tensorflow.contrib import slim

class GAN():
    """A class representing Generative Adversarial Network"""

    def __init__(self, input_dim, latent_dim, scope,
                 generator_architechture, discriminator_architechture,
                 mode='train'):

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.activation = tf.nn.tanh
        self.scope = scope
        self.generator_architechture = generator_architechture
        self.discriminator_architechture = discriminator_architechture

        self._create_placeholders()
        self._build_graph()
        
        self.generator_vars = self._get_vars_by_scope(
                                    self.scope + '/generator')
        self.discriminator_vars = self._get_vars_by_scope(
                                    self.scope + '/discriminator')
        self.own_vars = self._get_vars_by_scope(self.scope)
        self._create_saver(self.own_vars)
        self._create_loss()

        if mode == 'train':
            self._init_vars(self.own_vars)
            self.train_generator = self._create_optimizer(
                                                loss=self.generator_loss,
                                                vars_=self.generator_vars,
                                                opt_scope='generator',
                                                clip_value=1)
            self._init_optimizer('generator')

            self.train_discriminator = self._create_optimizer(
                                                loss=self.discriminator_loss,
                                                vars_=self.discriminator_vars,
                                                opt_scope='discriminator',
                                                clip_value=1)
            self._init_optimizer('discriminator')
        
    def _discriminate(self, x):

        net = x
        for i, layer_size in enumerate(self.discriminator_architechture):
            net = slim.fully_connected(
                                       net,
                                       layer_size,
                                       self.activation,
                                       scope='layer{}'.format(i+1))

            net = tf.nn.dropout(net, 0.7)

        d = slim.fully_connected(
                                 net,
                                 1,
                                 tf.nn.sigmoid,
                                 scope='layer_out')
        return d 

    def _generate(self):

        z = self._gaussian_sample()
        net = z
        for i, layer_size in enumerate(self.generator_architechture):
            net = slim.fully_connected(
                                       net,
                                       layer_size,
                                       self.activation,
                                       scope='layer{}'.format(i+1))

        sample = slim.fully_connected(
                                      net,
                                      self.input_dim,
                                      tf.nn.sigmoid,
                                      scope='layer_out')

        return sample

    def _create_placeholders(self):

        self.inputs = tf.placeholder(tf.float32, shape=[None, self.input_dim])
        self.learning_rate = tf.placeholder(tf.float32, [])
        self.num_samples = tf.placeholder(tf.int32, [])

    def _build_graph(self):

        with tf.variable_scope(self.scope):
            with tf.variable_scope('generator'):
                self.sampled = self._generate()

            with tf.variable_scope('discriminator'):

                self.model_d = self._discriminate(self.sampled)
                tf.get_variable_scope().reuse_variables()
                self.data_d = self._discriminate(self.inputs)

    def _create_saver(self, vars_):
        self.saver = tf.train.Saver(var_list=vars_, max_to_keep=1000)

    def _create_loss(self):

        #to avoid nans
        eps = 1e-7
        data_loss = -tf.log(self.data_d + eps)
        sample_loss = -tf.log(1 - self.model_d + eps)
        self.discriminator_loss = tf.reduce_mean(data_loss + sample_loss)

        self.generator_loss = tf.reduce_mean(-tf.log(self.model_d + eps))

    def _create_optimizer(self, vars_, loss, opt_scope, clip_value=None):

        with tf.variable_scope('Optimizer_' + opt_scope + '_' + self.scope):
            optimizer = tf.train.RMSPropOptimizer(
                                        learning_rate=self.learning_rate,
                                        momentum=0.05,
                                        decay=0.7)
            grads_and_vars = optimizer.compute_gradients(
                                                         loss=loss,
                                                         var_list=vars_)
            if clip_value is not None:
                grads_and_vars = [(tf.clip_by_norm(g, clip_value), v)
                                  for g, v in grads_and_vars]
            return optimizer.apply_gradients(grads_and_vars)

    def _init_optimizer(self, opt_scope):

        opt_vars = self._get_vars_by_scope('Optimizer_' + opt_scope + '_' + self.scope)
        self._init_vars(opt_vars)
        print('\nInitialized optimizer for {}`s {}.'.format(self.scope, opt_scope))

    def train(self, train_data, learning_rate, batch_size, num_samples):

        for _ in range(train_data.num_examples//batch_size):

            batch, _ = train_data.next_batch(batch_size)
            discriminator_feed = {
                                  self.inputs: batch,
                                  self.learning_rate: learning_rate,
                                  self.num_samples: num_samples}
            generator_feed = {
                              self.learning_rate: learning_rate,
                              self.num_samples: num_samples}
            
            self.sess.run(self.train_discriminator, feed_dict=discriminator_feed)
            self.sess.run(self.train_generator, feed_dict=generator_feed)

    def predict(self, validation_data):

        batch = validation_data.images
        fetches = [self.discriminator_loss, self. generator_loss]
        feed_dict = {
                     self.inputs: batch,
                     self.num_samples: validation_data.num_examples}
        dl, gl = self.sess.run(fetches, feed_dict=feed_dict)        
        print(' Discriminator loss: {}, Generator loss: {}'.format(dl, gl))

    @property
    def sess(self):
        if not hasattr(self, '_sess'):
            self._sess = tf.Session()
        return self._sess

    def save_model(self, path, global_step=None):
        self.saver.save(self.sess, path, global_step=global_step)

    def load_model(self, path):
        self.saver.restore(self.sess, path)
        print('\n{} restored.'.format(self.scope))

    def _gaussian_sample(self):
        return tf.random_normal(
                                shape=[self.num_samples, self.latent_dim],
                                dtype=tf.float32)

    def _init_vars(self, vars_):
        self.sess.run(tf.variables_initializer(vars_))
        print('\nFollowing vars for {} have been initialized:'.format(self.scope))
        for v in vars_:
            print(v.name)

    def _get_vars_by_scope(self, scope):
        return [
                v for v in tf.global_variables() 
                if v.name.startswith(scope)]

    def sample(self, num_samples):
        return self.sess.run(self.sampled, {self.num_samples: num_samples})
