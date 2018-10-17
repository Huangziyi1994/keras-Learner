# the code for cnn 
import os
import tensorflow as tf, numpy as np
import traceback 

class Network():

    def __init__(self, root, exp_dir):

        self.root = root
        self.exp_dir = exp_dir

        # --- Set default classes (overload to customize) 
        self.Client = None
        self.Model = None 

    def initialize(self):
        """
        Method to initialize CNN

          (1) Init batch, placeholders 
          (2) Init graph

        """
        # --- Create objects
        self.model = self.Model()
        self.client = self.Client(root=self.root, exp_dir=self.exp_dir, fold=self.model.params['fold'])

        # --- Initialize
        tf.reset_default_graph()
        self.batch = self.init_batch()
        self.placeholders = self.init_placeholders()
        self.graph = self.init_graph()
        self.stats = self.init_stats()

    def init_batch(self, one_hot=True):
        """
        Method to return batched data

        """
        
        try: 
            def generator_train():
                while True:
                    dat, lbl = self.client.get(cohort='train')
                    yield (dat, lbl) 
        except Exception: 
            traceback.print_exc()

        def generator_valid():
            while True:
                dat, lbl = self.client.get(cohort='valid')
                yield (dat, lbl) 

        batch = {}
        for mode, generator in zip(['train', 'valid'], [generator_train, generator_valid]): 

            ds = tf.data.Dataset.from_generator(generator, 
                output_types=(tf.float32, tf.int32),
                output_shapes=([256, 200, 1], [1, 200]))
            ds = ds.batch(self.model.params['batch_size'])
            ds = ds.prefetch(self.model.params['batch_size'] * 5)
            its = ds.make_one_shot_iterator()
            it = its.get_next()
            y = tf.one_hot(it[1] - 1, depth=self.model.params['num_classes']) if one_hot else it[1]
            y = tf.cast(y, tf.float32)

            batch[mode] = {'X': it[0], 'y': y}

        return batch 

    def init_placeholders(self):

        return {
            'X': tf.placeholder(tf.float32, shape=[None, 256, 200, 1], name='X'),
            'y': tf.placeholder(tf.float32, shape=[None, 1, 200, 2], name='y'),
            'mode': tf.placeholder(tf.bool, name='mode')}

    def init_graph(self):
        """
        Method to create graph

          (1) self.model.create_classifier
          (2) self.model.create_dice
          (3) self.model.create_optimizers 
          (4) tf.add_to_collection
          (5) tf.summary operations (TensorBoard)

        """
        self.ops = {}

        # --- Define classifier 
        print('Initializing graph')
        self.ops['pred'] = self.model.create_classifier(
            X=self.placeholders['X'],
            training=self.placeholders['mode'])

        # --- Define loss
        print('Initializing loss function')
        self.ops['losses'] = self.model.create_dice(
            y_pred=self.ops['pred'],
            y_true=self.placeholders['y'])

        # --- Define optimizers
        print('Initializing optimizer')
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            self.ops['global_step'] = tf.train.get_or_create_global_step()
            optimizer = tf.train.AdamOptimizer(learning_rate=self.model.params['learning_rate'])
            self.ops['train'] = optimizer.minimize(-self.ops['losses']['dice'], global_step=self.ops['global_step'])

        # --- Define collections
        for key in ['X', 'y', 'mode']:
            tf.add_to_collection('placeholders', self.placeholders[key])
        tf.add_to_collection('outputs', self.ops['pred'])

        # --- Define tf.summary operations
        print('Initializing summary operations')
        tf.summary.histogram('logits', self.ops['pred'])
        tf.summary.scalar('dice-all', self.ops['losses']['dice'])
        for key in self.ops['losses']:
            if key != 'dice':
                tf.summary.scalar('dice-class-%i' % key, self.ops['losses'][key])
        self.ops['summary'] = tf.summary.merge_all()

    # =================================================================
    # STATS | Training & Validation Statistics
    # =================================================================

    def init_stats(self):
        """
        Method to initialize a stats dictionary based on keys in losses

        """
        stats = {'train': None, 'valid': None}
        for mode, d in stats.items():
            stats[mode] = dict([(k, 0) for k in self.ops['losses']])

        return stats 

    def update_ema(self, stats, mode, iteration):
        """
        Method to update the self.stats dict with exponential moving average

        :params

          (dict) stats : dictionary with stats 
          (str) mode : 'train' or 'valid'
          (int) iteration : update iteration (to determine EMA vs average)

        """
        decay = 0.99 if mode == 'train' else 0.9
        d = decay if iteration > 10 else 0.5

        for key, value in stats.items():
            self.stats[mode][key] = self.stats[mode][key] * d + value * (1 - d)

    def print_status(self, step, stats_names=[]):
        """
        Method to print iteration and self.stats for train/valid

        :params

          (int) step
          (list) stats_names : list of keys within stats to print 

        """
        printf = {'train': '', 'valid': ''}
        values = {'train': [], 'valid': []} 

        for name in stats_names:
            for mode in ['train', 'valid']:
                printf[mode] += '- %s : %s ' % (name, '%0.4f')
                values[mode].append(self.stats[mode][name])

        printf = '%s | TRAIN %s | VALID %s' % ('%07i', printf['train'], printf['valid'])
        values = [step] + values['train'] + values['valid']

        print(printf % tuple(values), end='\r')

    # =================================================================
    # TRAINING | CNN trainin 
    # =================================================================

    def train(self, iterations=100):
        """
        Method to perform CNN training

          (1) Initialize session
          (2) Run training

        """
        with tf.Session() as sess:

            sess, saver, writer_train, writer_valid = self.init_session(sess)

            try:
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord)
                for i in range(iterations):
                    X_batch, y_batch = sess.run([self.batch['valid']['X'], self.batch['valid']['y']])
                    #if i==0:
                        #print('\nFirst iteration\n')
                        #np.savez('scaledTrainingData',X=X_batch, Y=y_batch)
                        #print('\nBatch 1 data saved\n\n')

                    _, losses, summary, step  = sess.run(
                        [self.ops['train'], self.ops['losses'], self.ops['summary'], self.ops['global_step']],
                        feed_dict={
                            self.placeholders['X']: X_batch, 
                            self.placeholders['y']: y_batch, 
                            self.placeholders['mode']: True})

                    writer_train.add_summary(summary, step)
                    self.update_ema(stats=losses, mode='train', iteration=i)
                    self.print_status(step, stats_names=['dice', 0, 1])

                    # --- Every 10th iteration run a single validation batch
                    if not i % 10:

                        X_batch, y_batch = sess.run([self.batch['valid']['X'], self.batch['valid']['y']])
                        losses, summary = sess.run(
                            [self.ops['losses'], self.ops['summary']],
                            feed_dict={
                                self.placeholders['X']: X_batch, 
                                self.placeholders['y']: y_batch, 
                                self.placeholders['mode']: False})

                        writer_valid.add_summary(summary, step)
                        self.update_ema(stats=losses, mode='valid', iteration=i)
                        self.print_status(step, stats_names=['dice', 0, 1])

                saver.save(sess, '%s/checkpoint/model.ckpy' % self.exp_dir)

            finally:
                coord.request_stop()
                coord.join(threads)
                saver.save(sess, '%s/checkpoint/model.ckpy' % self.exp_dir)

    def init_session(self, sess):
        """
        Method to initialize generic Tensorflow objects

        :params
          
          (tf.Session) sess

        """
        writer_train = tf.summary.FileWriter('%s/logs/train' % self.exp_dir, sess.graph)
        writer_valid = tf.summary.FileWriter('%s/logs/valid' % self.exp_dir)

        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()

        # --- Restore checkpoints if available
        latest_check_point = tf.train.latest_checkpoint('%s/checkpoint' % self.exp_dir)
        if latest_check_point is not None:
            saver.restore(sess, latest_check_point)
        else:
            os.makedirs('%s/checkpoint' % self.exp_dir, exist_ok=True)

        return sess, saver, writer_train, writer_valid

class Model():

    def __init__(self):
        """
        Default initialization

        """
        self.params = {}
        self.params['learning_rate'] = 1e-3
        self.params['batch_size'] = 32 #initially was 16
        self.params['l2_reg'] = 1e-1

        self.init_hyperparams_custom()

    def conv_block(self, layer, fsize, training, name):
        """
        Method to perform basic CNN convolution block pattern
        
          [ CONV --> BN --> RELU ] --> [CONV s2 --> BN --> RELU] 
        """
        # --- Regular convolution layer
        layer = self.conv_layer(layer, fsize, strides=(1, 1), training=training, name=name, iteration=1)

        # --- Strided convolution layer
        layer = self.conv_layer(layer, fsize, strides=(2, 1), training=training, name=name, iteration=2)

        return layer

    def conv_layer(self, layer, fsize, strides, training, name, iteration=1):
        """
        Method to perform basic CNN convlution pattern

          [ CONV --> BN --> RELU  ]

        :params
          
          (tf.Tensor) layer : input layer
          (int) fsize : output filter size
          (tf.Tensor) training : boolean value regarding train/valid cohort
          (str) name : name of block 

        :return

          (tf.Tensor) layer : output layer 

        """
        with tf.variable_scope(name):

            layer = tf.layers.conv2d(layer, filters=fsize, kernel_size=(3, 3), strides=strides, padding='same',
                kernel_regularizer=self.l2_reg(self.params['l2_reg']), name='conv-%i' % iteration)
            layer = tf.layers.batch_normalization(layer, training=training, name='norm-%s' % iteration)
            layer = tf.nn.relu(layer, name='relu-%i' % iteration)

        return layer

    def l2_reg(self, scale):

        return tf.contrib.layers.l2_regularizer(scale)

    def create_dice(self, y_pred, y_true):
        """
        Method to approximate Dice score loss function

          Dice (formal) = 2 x (y_pred UNION y_true) 
                          -------------------------
                           | y_pred | + | y_true | 

          Dice (approx) = 2 x (y_pred * y_true) + d 
                          -------------------------
                         | y_pred | + | y_true | + d 

          where d is small delta == 1e-7 added both to numerator/denominator to
          prevent division by zero.

        :params

            (tf.Tensor) y_pred : predictions 
            (tf.Tensor) y_true : ground-truth 

        :return

            (dict) scores : {
              'final': final weighted Dice score,
              0: score for class 1,
              1: score for class 2, ...
            }

        """
        # --- Loop over the channels
        channels = y_pred.shape.as_list()[-1]
        losses = {}
        for ch in range(channels):
            num = 2 * tf.reduce_sum(y_pred[..., ch] * y_true[..., ch])
            den = tf.reduce_sum(y_pred[..., ch]) + tf.reduce_sum(y_true[..., ch])
            losses[ch] = (num + 1e-7) / (den + 1e-7) 

        # --- Calculate weighted dice score loss function
        weight = lambda ch : (tf.reduce_sum(y_true[..., ch]) + 1e-7) / (tf.reduce_sum(y_true) + 1e-7)
        losses['dice'] = tf.reduce_sum([losses[ch] * weight(ch) for ch in range(channels)])

        return losses 

def debug(X, y, training):

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    #config.gpu_options.allow_growth = True
    #session = tf.Session(config=config, ...)
    
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    sess.run(init)
    feed_dict = {
        X: np.random.rand(1, 256, 200, 1).astype('float32'), 
        y: np.ones((1, 1, 200, 2)).astype('float32'),
        training: True}

    return sess, feed_dict    

