import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
#import matplotlib.pyplot as plt
import read_gen
import h5py
import sys


def linear(name, x, shape):
	w = weight_variable(name + 'W', shape)
	b = weight_variable(name + 'B',(shape[1]))
	return tf.matmul(x,w) + b

def kernel_variable(name, shape):
	return tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer_conv2d())

def weight_variable(name, shape):
	return tf.get_variable(name, shape, initializer=tf.truncated_normal_initializer(stddev = 0.02))

class Config:
	read_size = 100
	projected_size = 200
	z_dim = 10
	#z_dim = 20
	learning_rate = 0.0002
	lr_decay = 0.9
	batch_size = 512
	phi = 0.5

class NGA:
	def __init__(self, config):
		self.config = config
		self.x = tf.placeholder(tf.float32, shape = [None, 1, config.read_size, 4])

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
		with tf.variable_scope('conv1'):
			# k=10, hid=128
			kernel = kernel_variable('w', [1, 50, 4, 128])
			bias = weight_variable('b', [128])
			conv = tf.nn.conv2d(self.x, kernel, [1,1,1,1], padding='SAME')
			conv1 = tf.nn.relu(conv + bias)

		# pool1
		# conv1.shape = [config.batch_size, 1, config.read_size, 128]
		with tf.variable_scope('pool1'):
			pool1 = tf.nn.max_pool(conv1, ksize=[1, 1, 5, 1], strides=[1, 1, 2, 1], padding='SAME')

		# conv2
		# pool1.shape = [config.batch_size, 1, config.read_size/2, 128]
		with tf.variable_scope('conv2'):
			# k=10, hid=128
			kernel = kernel_variable('w', [1, 10, 128, 128])
			bias = weight_variable('b', [128])
			conv = tf.nn.conv2d(conv1, kernel, [1,1,1,1], padding='SAME')
			conv2 = tf.nn.relu(conv + bias)

		# pool2
		# conv2.shape = [config.batch_size, 1, config.read_size/2, 128]
		with tf.variable_scope('pool'):
			pool2 = tf.nn.max_pool(conv2, ksize=[1, 1, 5, 1], strides=[1, 1, 2, 1], padding='SAME')

		# pool2.shape = [config.batch_size, 1, config.read_size/2, 128]
		with tf.variable_scope('local3') as scope:
			dim = self.config.read_size * 128 / 2
			reshape = tf.reshape(pool2, [-1, dim])
			local3 = tf.nn.relu(linear('w', reshape, [dim, 400]))

		# local3
		# local3.shape = [config.batch_size, 400]
		with tf.variable_scope('z_theta') as scope:
			z_mean = linear('mean', local3, (400, self.config.z_dim))
			z_log_sigma_sq = linear('sigma', local3, (400, self.config.z_dim))

		return (z_mean, z_log_sigma_sq)


	def decoder(self):
		# local1
		# z.shape = [config.batch_size, config.z_dim]
		with tf.variable_scope('local1') as scope:
			local1 = tf.nn.relu(linear('w', self.z, (self.config.z_dim, 400)))

		# local2
		# local1.shape = [config.batch_size, 400]
		with tf.variable_scope('local2') as scope:
			local2 = tf.nn.relu(linear('w', local1, (400, 128 * self.config.read_size / 4)))

		# deconv3
		# local2.shape = [config.batch_size, 128*config.read_size/4]
		with tf.variable_scope('deconv3') as scope:
			reshape = tf.reshape(local2, (self.config.batch_size, 1, self.config.read_size/4, 128))
			kernel = kernel_variable('w', [1, 10, 128, 128])
			bias = weight_variable('b', [128])
			deconv = tf.nn.conv2d_transpose(reshape, kernel, (self.config.batch_size, 1, self.config.read_size/2, 128), [1, 1, 2, 1]) 
			deconv3 = tf.nn.relu(deconv + bias)

		# deconv4
		# deconv3.shape = [config.batch_size, 1, config.read_size/2, 128]
		with tf.variable_scope('deconv4') as scope:
			kernel = kernel_variable('w', [1, 10, 64, 128])
			bias = weight_variable('b', [64])
			deconv = tf.nn.conv2d_transpose(deconv3, kernel, (self.config.batch_size, 1, self.config.read_size, 64), [1, 1, 2, 1]) 
			deconv4 = tf.nn.relu(deconv + bias)

		with tf.variable_scope('deconv5') as scope:
			kernel = kernel_variable('w', [1, 50, 4, 64])
			bias = weight_variable('b', [4])
			deconv = tf.nn.conv2d_transpose(deconv4, kernel, (self.config.batch_size, 1, self.config.projected_size, 4), [1, 1, self.config.projected_size/self.config.read_size, 1]) 
			deconv5 = tf.nn.softmax(deconv + bias, dim=-1)

		#phi = tf.nn.sigmoid(linear('phi', self.z, [self.config.z_dim, 1]))

		return deconv5 #, phi

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
		self.reconstr_loss = self.reconstr_loss_slide()

		self.latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq 
				- tf.square(self.z_mean) 
				- tf.exp(self.z_log_sigma_sq), 1)
		self.cost = tf.reduce_mean(self.reconstr_loss + self.latent_loss)
#		self.lr = tf.Variable(self.config.learning_rate, False)
		self.optimizer = \
				tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.cost)
	
	def partial_fit(self, X):
		"""Train model based on mini-batch of input data.

		Return cost of mini-batch.
		"""

		X = np.expand_dims(X, 1)

		'''
		print X.shape
		print self.sess.run(self.x_recon_theta_tran, feed_dict={self.x: X}).shape
		print self.sess.run(self.x_tran, feed_dict={self.x: X}).shape
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
	gen = read_gen.ReadGen('dna_10k.fa', 100, Config.read_size)
	nga = NGA(Config)
	train(nga, gen)
	#nga.load('checkpoints')
	create_sample_for_plot(nga, gen)


if __name__ == '__main__':
	main()
