import tensorflow as tf
import numpy as np
L2 = 0.01

class IdentifierNetwork:
    def __init__(self, sess, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        # define placeholder for inputs to network
        self.state_input = tf.placeholder(tf.float32, [None, state_dim])
        self.action_input = tf.placeholder(tf.float32, [None, action_dim])
        self.ys = tf.placeholder(tf.float32, [None, state_dim])

        xs = tf.concat([self.state_input, self.action_input], 1)
        # add hidden layer
        l1, wb1 = self.add_layer(xs, state_dim + action_dim, 15, activation_function=tf.nn.sigmoid)
        # add output layer
        self.prediction, wb2 = self.add_layer(l1, 15, state_dim, activation_function=None)
        nets = wb1 + wb2
        weight_decay = tf.add_n([L2 * tf.nn.l2_loss(var) for var in nets])
        lc = self.Lyapunov(self.prediction) - self.Lyapunov(self.state_input)
        self.dlda = tf.gradients(lc, self.action_input)
        # the error between prediction and real data
        self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.ys - self.prediction) + 0.01*weight_decay, reduction_indices=[1]))
        self.train = tf.train.AdamOptimizer(0.01).minimize(self.loss)
        self.sess = sess
        self.dL = tf.reduce_mean(lc)
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

    def update(self, state_batch, action_batch, label_batch):
        _, loss = self.sess.run([self.train, self.loss], feed_dict={self.state_input:state_batch,
            self.action_input:action_batch, self.ys:label_batch})
        return loss
        #print('id_loss: ', loss)

    def Lyapunov(self, state):
        return tf.square(30*(1 - state[:, 0]) + tf.square(state[:, 2])) * 0.01

    def gradients(self, state_batch,action_batch):
        return self.sess.run(self.dlda, feed_dict={self.state_input: state_batch, self.action_input: action_batch})[0]

    def predict(self, x, u):
        y = self.sess.run(self.prediction,
                          feed_dict={self.state_input: np.array([x]), self.action_input: np.array([u])})
        return y[0]

    def predict_batch(self, xs, us):
        ys = self.sess.run(self.prediction, feed_dict={self.state_input: xs, self.action_input: us})
        return ys

    def analysis(self, x, u):
        return self.sess.run(self.dL, feed_dict={self.state_input:np.array([x]), self.action_input:np.array([u])})
