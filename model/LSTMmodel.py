import tensorflow as tf


class LSTMmodel:
    def __init__(self, embedding_size, num_hidden, wordvector):
        self.embed_size = embedding_size
        self.num_hidden = num_hidden
        self.wv = wordvector

        if FLAGS.trainable_vectors:
            self.words = tf.Variable(self.wv, name='words')
        else:
            self.words = tf.constant(self.wv, name='words')

        with tf.variable_scope('Softmax') as scope:
            self.W = tf.get_variable(shape=[num_hidden*2, 4], initializer=tf.truncated_normal_initializer(stddev=0.01),
                                     name='weights', regularizer=tf.contrib.layers.l2_regularizer(0.001))
            self.b = tf.Variable(tf.zeros([4]), name='bias')

        self.trains_params = None
        self.inp = tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.max_sentence_len], name='input_placeholder')

    def inference(self, X, reuse=None):
        word_vec = tf.nn.embedding_lookup(self.words, X)
        length = get_length(word_vec, reuse)
        len64 = tf.cast(length, tf.int64)

        # BiLSTM
        with tf.variable_scope('rnn_fwbw', reuse=reuse) as scope:
            forward_out, _ = tf.nn.dynamic_rnn(tf.nn.rnn_cell.LSTMCell(self.num_hidden), word_vec,
                                               dtype=tf.float32, sequence_length=length, scope='RNN_forward')
            backward_out_, _ = tf.nn.dynamic_rnn(tf.nn.rnn_cell.LSTMCell(self.num_hidden),
                                                 inputs=tf.reverse_sequence(word_vec, len64, seq_axis=1),
                                                 dtype=tf.float32, sequence_length=length, scope='RNN_backward')
        backward_out = tf.reverse_sequence(backward_out_, len64, seq_axis=1, name='RNN_backward_reversed')
        output = tf.concat([forward_out, backward_out], 2)
        output = tf.reshape(output, [-1, self.num_hidden*2])

        # Unregularized CRF output
        matricized_unary_scores = tf.matmul(output, self.W)
        unary_score = tf.reshape(matricized_unary_scores, [-1, FLAGS.max_sentence_len, 4],
                                 name=('unary_score' if reuse else None))
        return unary_score, length

    def loss(self, X, Y):
        P, seq_length = self.inference(X)
        log_likelihood, self.trans_params = tf.contrib.crf.crf_log_likelihood(P, Y, seq_length)
        return tf.reduce_mean(-log_likelihood)

    def test_unary_score(self):
        return self.inference(self.inp, reuse=True)

    def train(total_loss, global_step):
        return tf.train.AdamOptimizer(FLAGS.alpha).minimize(total_loss, global_step=global_step)

