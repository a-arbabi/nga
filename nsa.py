import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
#import matplotlib.pyplot as plt
import read_gen
#import h5py
import sys
#import gpu_access


def linear(name, x, shape):
	w = weight_variable(name + 'W', shape)
	b = weight_variable(name + 'B',(shape[1]))
	return tf.matmul(x,w) + b

def kernel_variable(name, shape):
	return tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer_conv2d())

def weight_variable(name, shape):
	return tf.get_variable(name, shape, initializer=tf.truncated_normal_initializer(stddev = 0.02))

class Config:
	hidden_size = 200
	fc_size = 100
	read_size = 100
	projected_size = 200
	z_dim = 100
	#z_dim = 20
	learning_rate = 0.001
	lr_decay = 0.9
	batch_size = 512
	phi = 0.5

class NGA:
	def __init__(self, config):
		self.config = config
		self.x = tf.placeholder(tf.float32, shape = [None, config.read_size, 4])

		self.eps = tf.random_normal((config.batch_size, config.z_dim), 0, 1, 
				dtype=tf.float32)

		self.z_mean, self.z_log_sigma_sq = self.encoder()

		self.z = tf.add(self.z_mean, 
				tf.mul(tf.sqrt(tf.exp(self.z_log_sigma_sq)), self.eps))

		self.x_recon_theta = self.decoder()
		self.phi = tf.constant(config.phi)

		self.create_loss()

		self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
		self.sess.run(tf.initialize_all_variables())


	def encoder(self):
		# conv1
		# x.shape = [config.batch_size, 1, config.read_size, 4]
		inputs = tf.unpack(self.x, axis = 1)
		inputs.reverse()

		with tf.variable_scope("enc_rnn") as scope:
			single_cell = tf.nn.rnn_cell.GRUCell(self.config.hidden_size)
			self.cell = single_cell #tf.nn.rnn_cell.MultiRNNCell([single_cell]*config.num_layers, state_is_tuple=False)

			outputs, state = tf.nn.rnn(self.cell, inputs, dtype=tf.float32, \
					scope=scope) #, initial_state=self._initial_state)



		with tf.variable_scope('enc_fc') as scope:
			fc_layer1 = tf.nn.tanh(linear('w1', state, [self.config.hidden_size, self.config.hidden_size]))
			fc_layer2 = tf.nn.tanh(linear('w2', state, [self.config.hidden_size, self.config.fc_size]))
			z_mean = linear('mean', fc_layer2, (self.config.fc_size, self.config.z_dim))
			z_log_sigma_sq = linear('sigma', fc_layer2, (self.config.fc_size, self.config.z_dim))

		return (z_mean, z_log_sigma_sq)


	def hidden2logit(self, h, reuse=False):
		with tf.variable_scope("dec_rnn") as scope:
			if reuse:
				scope.reuse_variables()
			layer1 = tf.nn.tanh(linear('layer1', h, [self.config.hidden_size, 200]))
			#layer2 = tf.nn.tanh(linear('layer2', layer1, [200, 200]))
			layer3 = linear('layer3', layer1, [200, 4])
		return layer3


	def decoder(self):
		with tf.variable_scope('dec_fc') as scope:
			init_state_fc1 = tf.nn.tanh(linear('w1', self.z, [self.config.z_dim, self.config.fc_size]))
			init_state_fc2 = tf.nn.tanh(linear('w2', self.z, [self.config.z_dim, self.config.hidden_size]))
			init_state = tf.nn.tanh(linear('w3', init_state_fc2, [self.config.hidden_size, self.config.hidden_size]))

		with tf.variable_scope("dec_rnn") as scope:
			single_cell = tf.nn.rnn_cell.GRUCell(self.config.hidden_size)
			self.cell_dec = single_cell #tf.nn.rnn_cell.MultiRNNCell([single_cell]*config.num_layers, state_is_tuple=False)

		state = init_state

		predictions = []
		for i in range(self.config.read_size):
			reuse = (i > 0)
			logits = self.hidden2logit(state, reuse)
			last_pred = tf.nn.softmax(logits)
			predictions.append(last_pred)
			with tf.variable_scope("dec_rnn") as scope:
				if i>0:
					scope.reuse_variables()
				(output, state) = self.cell_dec(last_pred, state, scope)
	

		packed_predictions = tf.pack(predictions, axis=1)
		return packed_predictions

	def reconstr_loss(self):
		self.x_recon_theta_log = tf.log(1e-10 + self.x_recon_theta)
		return -tf.reduce_sum(self.x_recon_theta_log*self.x, [1,2])

	def reconstr_loss_slide(self):
		# reshaped -> [1, self.config.projected_size, 4, batch_size]
		self.x_recon_theta_tran = tf.transpose(tf.log(1e-10 + self.x_recon_theta), [1,2,3,0])
		# x_reshape -> [self.config.read_size, 4, batch_size, 1]
		self.x_tran = tf.transpose(self.x, [2,3,0,1])

		reconstr_conv = tf.nn.depthwise_conv2d(self.x_recon_theta_tran, self.x_tran, [1, 1, 1, 1], 'VALID')
		reconstr_conv = tf.squeeze(tf.transpose(reconstr_conv, [3,1,0,2]))
	
		p_range = np.abs(1.0*np.array(range(0, self.config.projected_size - self.config.read_size + 1))\
				- (self.config.projected_size - self.config.read_size)/2)
		p_range = tf.constant(p_range, tf.float32)

		reconstr_prior = tf.log(1e-10 + 1.0-self.phi) * p_range + tf.log(1e-10 + self.phi) - tf.log(2.0)
		reconstr_prior += tf.constant(np.array([0.0 if i!=(self.config.projected_size - self.config.read_size)/2 else np.log(2.0)\
				for i in range(0,(self.config.projected_size - self.config.read_size+1))]), tf.float32) 
		

		return  -tf.reduce_logsumexp(reconstr_prior + reconstr_conv ,[1])

	def create_loss(self):
		'''
		self.reconstr_loss = \
				-tf.reduce_sum(self.x * tf.log(1e-10 + self.x_recon_theta) ,[1,2,3])
		'''
		self.recon_loss = self.reconstr_loss()
		#self.reconstr_loss = self.reconstr_loss_slide()

		self.latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq 
				- tf.square(self.z_mean) 
				- tf.exp(self.z_log_sigma_sq), 1)
		self.cost = tf.reduce_mean(self.recon_loss + self.latent_loss)
#		self.lr = tf.Variable(self.config.learning_rate, False)
		self.optimizer = \
				tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.cost)
	
	def partial_fit(self, X):
		"""Train model based on mini-batch of input data.

		Return cost of mini-batch.
		"""

		#X = np.expand_dims(X, 1)
		'''
		print np.sum(self.sess.run(self.z, feed_dict={self.x: X}))
		print np.min(self.sess.run(self.x_recon_theta, feed_dict={self.x: X}))
		print np.sum(self.sess.run(self.x_recon_theta_log, feed_dict={self.x: X}))
		'''
		opt, cost = self.sess.run((self.optimizer, self.cost), 
								  feed_dict={self.x: X})
		return cost

	def transform(self, X):
		"""Transform data by mapping it into the latent space."""
		# Note: This maps to mean of distribution, we could alternatively
		# sample from Gaussian distribution
		X = np.expand_dims(X, 1)
		return self.sess.run(self.z_mean, feed_dict={self.x: X})

	def generate(self, z_mu=None):
		""" Generate data by sampling from latent space.

		If z_mu is not None, data for this point in latent space is
		generated. Otherwise, z_mu is drawn from prior in latent 
		space.        
		"""
		if z_mu is None:
			z_mu = np.random.normal(size=self.network_architecture["n_z"])
		# Note: This maps to mean of distribution, we could alternatively
		# sample from Gaussian distribution
		return self.sess.run(self.x_recon_theta, 
							 feed_dict={self.z: z_mu})

	def reconstruct(self, X):
		""" Use VAE to reconstruct given data. """
		return self.sess.run(self.x_recon_theta, 
				feed_dict={self.x: X})

	def save(self, rep_dir):
		saver = tf.train.Saver()
		saver.save(self.sess, rep_dir+'/training.ckpt')

	def load(self, rep_dir):
		saver = tf.train.Saver()
		saver.restore(self.sess, rep_dir + '/training.ckpt')

def train(nga, gen):
	# Training cycle
	display_step = 5
	for epoch in range(500):
		gen.reset_counter()
		total_cost = 0.
		count = 0
		batch_size = nga.config.batch_size
		# Loop over all batches

#		lr_new = nga.config.learning_rate * (nga.config.lr_decay ** int(epoch/100))
#		nga.sess.run(tf.assign(nga.lr, lr_new))

		while True:
			batch_xs, _ = gen.read_batch(batch_size) #mnist.train.next_batch(batch_size)
			if batch_xs is None:
				break

			# Fit training using batch data
			cost = nga.partial_fit(batch_xs)

			total_cost += cost 
			count += 1

		# Display logs per epoch step

		if  epoch % display_step == 0:
			print "Epoch:", '%04d' % (epoch), \
				"cost=", total_cost/count
			sys.stdout.flush()

	nga.save('checkpoints')

def create_sample_for_plot(nga, gen):
	gen.reset_counter()
	x_sample, y_sample = gen.read_batch(1000)
	y_sample = np.float64(y_sample) / len(gen.seq)

	z_mu = nga.transform(x_sample)

	h5f = h5py.File('plot_data.h5', 'w')
	h5f.create_dataset('z', data=z_mu)
	h5f.create_dataset('y', data=y_sample)
	h5f.close()

def main():
	print 'hello'
	gen = read_gen.ReadGen('dna_10k.fa', 20, Config.read_size)

	#board = gpu_access.get_gpu()
	nga = NGA(Config)
	#with tf.device('/gpu:'+board):
	train(nga, gen)
	exit()
	#nga.load('checkpoints')
	create_sample_for_plot(nga, gen)


if __name__ == '__main__':
	main()
