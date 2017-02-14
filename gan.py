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
        self.mode = mode

        self._create_placeholders()
        self._build_graph()
        
        self.generator_vars = self._get_vars_by_scope(
                                    self.scope + '/generator',
                                    only_trainable=True)
        self.discriminator_vars = self._get_vars_by_scope(
                                    self.scope + '/discriminator',
                                    only_trainable=True)
        self.own_vars = self._get_vars_by_scope(self.scope)
        self._create_saver(self.own_vars)
        self._create_loss()

        if mode == 'train':
            self._init_vars(self.own_vars)
            self.train_generator = self._create_optimizer(
                                                loss=self.generator_loss,
                                                vars_=self.generator_vars,
                                                opt_scope='generator',
                                                clip=None)
            self._init_optimizer('generator')

            self.train_discriminator = self._create_optimizer(
                                                loss=self.discriminator_loss,
                                                vars_=self.discriminator_vars,
                                                opt_scope='discriminator',
                                                clip=None)
            self._init_optimizer('discriminator')

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

    def _create_optimizer(self, vars_, loss, opt_scope, clip=None):

        with tf.variable_scope('Optimizer_' + opt_scope + '_' + self.scope):
            optimizer = tf.train.RMSPropOptimizer(
                                        learning_rate=self.learning_rate,
                                        momentum=0,
                                        decay=0.9)
            grads_and_vars = optimizer.compute_gradients(
                                                         loss=loss,
                                                         var_list=vars_)
            if clip is not None:
                grads_and_vars = [(tf.clip_by_value(g, -clip, clip), v)
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
        fetches = [self.discriminator_loss, self.generator_loss]
        feed_dict = {
                     self.inputs: batch,
                     self.num_samples: validation_data.num_examples}
        dl, gl = self.sess.run(fetches, feed_dict=feed_dict)        
        print(' Discriminator loss: {}, Generator loss: {}'.format(dl, gl))

    @property
    def sess(self):
        if not hasattr(self, '_sess'):
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self._sess = tf.Session(config=config)
        return self._sess


    def save_model(self, path, global_step=None):
        self.saver.save(self.sess, path, global_step=global_step)

    def load_model(self, path):
        self.saver.restore(self.sess, path)
        print('\nModel for {} restored.'.format(self.scope))

    def _gaussian_sample(self):
        return tf.random_normal(
                                shape=[self.num_samples, self.latent_dim],
                                dtype=tf.float32)

    def _init_vars(self, vars_):
        self.sess.run(tf.variables_initializer(vars_))
        print('\nFollowing vars for {} have been initialized:'.format(self.scope))
        for v in vars_:
            print(v.name)

    def _get_vars_by_scope(self, scope, only_trainable=False):
        if only_trainable:
            vars_ = tf.trainable_variables()
        else:
            vars_ = tf.global_variables()

        return list(v for v in vars_ if v.name.startswith(scope))

    def sample(self, num_samples):
        return self.sess.run(self.sampled, {self.num_samples: num_samples})


class WGAN(GAN):

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
        #Wasserstein distance
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
                                  self.num_samples: num_samples}
            generator_feed = {
                              self.learning_rate: learning_rate,
                              self.num_samples: num_samples}
            
            self.sess.run(self.train_discriminator, discriminator_feed)
            #according to the paper
            if step % 5 == 0:
                self.sess.run(self.train_generator, generator_feed)
            #crop weights to speed up discriminator convergence
            self.sess.run(self.clip_weights)