import tensorflow as tf 
import numpy as np
import math


# Hyper Parameters
LAYER1_SIZE = 400
LAYER2_SIZE = 300
LEARNING_RATE = 1e-4
TAU = 0.001
BATCH_SIZE = 64

class ActorNetwork:
	"""docstring for ActorNetwork"""
	def __init__(self,sess,state_dim,action_dim, actor_n):

		self.sess = sess
		self.state_dim = state_dim
		self.action_dim = action_dim
		# create actor network
		self.state_input,self.feature_output,self.net = self.create_network(state_dim)
		self.actors = [self.create_actor(action_dim) for i in range(actor_n)]

		# create target actor network
		self.target_state_input,self.target_feature_output,self.target_update,self.target_net = self.create_target_network(state_dim,self.net)
		self.target_actors = [self.create_target_actor(self.actors[i][1]) for i in range(actor_n)]

		# define training rules
		self.create_training_method(actor_n)

		self.sess.run(tf.global_variables_initializer())

		self.sess.run(self.target_update)
		for i in range(actor_n):
			self.sess.run(self.target_actors[i][1])
		#self.load_network()

	def create_training_method(self, actor_n):
		self.q_gradient_input = tf.placeholder("float", [None,self.action_dim])
		self.l_gradient_input = tf.placeholder("float", [None, self.action_dim])
		self.optimizers = []
		for i in range(actor_n):
			weights = self.net + self.actors[i][1]
			parameters_gradients = tf.gradients(self.actors[i][0], weights, -self.q_gradient_input)
			stability_gradients = tf.gradients(self.actors[i][0], weights, self.l_gradient_input)
			opt1 = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(zip(parameters_gradients, weights))
			opt2 = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(zip(stability_gradients, weights))
			self.optimizers.append([opt1, opt2])

	def create_network(self, state_dim):
		layer1_size = LAYER1_SIZE
		layer2_size = LAYER2_SIZE

		state_input = tf.placeholder("float",[None,state_dim])

		W1 = self.variable([state_dim,layer1_size],state_dim)
		b1 = self.variable([layer1_size],state_dim)
		W2 = self.variable([layer1_size,layer2_size],layer1_size)
		b2 = self.variable([layer2_size],layer1_size)
		# W3 = tf.Variable(tf.random_uniform([layer2_size,action_dim],-3e-3,3e-3))
		# b3 = tf.Variable(tf.random_uniform([action_dim],-3e-3,3e-3))

		layer1 = tf.nn.relu(tf.matmul(state_input,W1) + b1)
		layer2 = tf.nn.relu(tf.matmul(layer1,W2) + b2)
		# action_output = tf.tanh(tf.matmul(layer2,W3) + b3)

		return state_input, layer2, [W1,b1,W2,b2]

	def create_actor(self, action_dim):
		W3 = tf.Variable(tf.random_uniform([LAYER2_SIZE, action_dim], -0.2, 0.2))
		b3 = tf.Variable(tf.random_uniform([action_dim],-3e-3,3e-3))
		action_output = tf.tanh(tf.matmul(self.feature_output, W3) + b3)
		return action_output, [W3, b3]

	def create_target_network(self,state_dim,net):
		state_input = tf.placeholder("float",[None,state_dim])
		ema = tf.train.ExponentialMovingAverage(decay=1-TAU)
		target_update = ema.apply(net)
		target_net = [ema.average(x) for x in net]

		layer1 = tf.nn.relu(tf.matmul(state_input,target_net[0]) + target_net[1])
		layer2 = tf.nn.relu(tf.matmul(layer1,target_net[2]) + target_net[3])
		# action_output = tf.tanh(tf.matmul(layer2,target_net[4]) + target_net[5])

		return state_input, layer2, target_update,target_net

	def create_target_actor(self, net):
		ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)
		target_update = ema.apply(net)
		target_net = [ema.average(x) for x in net]

		action_output = tf.tanh(tf.matmul(self.target_feature_output, target_net[0]) + target_net[1])

		return action_output, target_update, target_net

	def update_target(self, idx):
		self.sess.run([self.target_update, self.target_actors[idx][1]])

	def train(self, idx, state_batch, q_gradient_batch):
		fd_dict = {self.state_input: state_batch,
				   self.q_gradient_input: q_gradient_batch}

		self.sess.run(self.optimizers[idx][0],feed_dict=fd_dict)

	def stabilize(self, idx, state_batch, l_gradient_batch):
		fd_dict = {self.state_input: state_batch,
				   self.l_gradient_input: l_gradient_batch}
		self.sess.run(self.optimizers[idx][1], feed_dict=fd_dict)

	def actions(self,state_batch, idx):
		return self.sess.run(self.actors[idx][0],feed_dict={
			self.state_input:state_batch
			})

	def action(self, state, idx):
		return self.sess.run(self.actors[idx][0],feed_dict={
			self.state_input:[state]
			})[0]

	def target_actions(self,state_batch, idx):
		return self.sess.run(self.target_actors[idx][0],feed_dict={
			self.target_state_input:state_batch
			})

	# f fan-in size
	def variable(self,shape,f):
		return tf.Variable(tf.random_uniform(shape,-1/math.sqrt(f),1/math.sqrt(f)))
'''
	def load_network(self):
		self.saver = tf.train.Saver()
		checkpoint = tf.train.get_checkpoint_state("saved_actor_networks")
		if checkpoint and checkpoint.model_checkpoint_path:
			self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
			print "Successfully loaded:", checkpoint.model_checkpoint_path
		else:
			print "Could not find old network weights"
	def save_network(self,time_step):
		print 'save actor-network...',time_step
		self.saver.save(self.sess, 'saved_actor_networks/' + 'actor-network', global_step = time_step)

'''

		
