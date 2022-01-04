import tensorflow as tf
import numpy as np
L2 = 0.01

class RewardNetwork:
    def __init__(self, sess, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        # define placeholder for inputs to network
        self.state_input = tf.placeholder(tf.float32, [None, state_dim])
        self.action_input = tf.placeholder(tf.float32, [None, action_dim])
        self.ys = tf.placeholder(tf.float32, [None, 1])

        xs = tf.concat([self.state_input, self.action_input], 1)
        # add hidden layer
        l1, wb1 = self.add_layer(xs, state_dim + action_dim, 15, activation_function=tf.nn.sigmoid)
        # add output layer
        self.prediction, wb2 = self.add_layer(l1, 15, 1, activation_function=None)
        nets = wb1 + wb2
        weight_decay = tf.add_n([L2 * tf.nn.l2_loss(var) for var in nets])

        # the error between prediction and real data
        self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.ys - self.prediction) + 0.01*weight_decay, reduction_indices=[1]))
        self.train = tf.train.AdamOptimizer(0.01).minimize(self.loss)
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())

    def add_layer(self, inputs, in_size, out_size, activation_function=None):
        Weights = tf.Variable(tf.random_normal([in_size, out_size]))
        biases = tf.Variable(tf.zeros([1, out_size]))
        Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs, [Weights, biases]

    def update(self, state_batch, action_batch, rewards):
        _, loss = self.sess.run([self.train, self.loss], feed_dict={self.state_input:state_batch,
            self.action_input:action_batch, self.ys: rewards})
        return loss

    def predict(self, x, u):
        y = self.sess.run(self.prediction,
                          feed_dict={self.state_input: np.array([x]), self.action_input: np.array([u])})
        return y[0]

    def predict_batch(self, xs, us):
        ys = self.sess.run(self.prediction, feed_dict={self.state_input: xs, self.action_input: us})
        return ys
