import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import read_gen

def linear(x, shape):
	w = weight_variable(shape)
	b = weight_variable((shape[1]))
	return tf.matmul(x,w) + b

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

class Config:
	read_size = 100
	z_dim = 2
	#z_dim = 20
	learning_rate = 0.001
	batch_size =50

def conv2d(x, name, shape):
	W = tf.get_variable(name, shape, initializer = tf.contrib.layers.xavier_initializer_conv2d)
	B = tf.zeros((shape[-1]))
	return tf.nn.relu(tf.nn.conv2d(x, W, [1, 1, 1, 1], padding='SAME')+B)

def max_pool(x):
	return tf.nn.max_pool(x, [1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')

def conv2d_transpose(x, name, filter_shape, output_shape):
	W = tf.get_variable(name, filter_shape, initializer = tf.contrib.layers.xavier_initializer_conv2d)
	B = tf.zeros((shape[-1]))
	return tf.nn.relu(tf.nn.conv2d_transpose(x, W, [1, 1, 1, 1], padding='SAME')+B)



class NGA:
	def __init__(self, config):
		self.config = config
		self.x = tf.placeholder(tf.float32, shape = [None, config.read_size, 4])
#		self.eps = tf.placeholder(tf.float32, shape = [None, config.z_dim])


		self.eps = tf.random_normal((config.batch_size, config.z_dim), 0, 1, 
				dtype=tf.float32)

		self.z_mean, self.z_log_sigma_sq = self.encoder()

		self.z = tf.add(self.z_mean, 
				tf.mul(tf.sqrt(tf.exp(self.z_log_sigma_sq)), self.eps))

		self.x_recon_theta = self.decoder()

		self.create_loss()

		self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
		self.sess.run(tf.initialize_all_variables())


	def encoder(self):
		# conv1
		# x.shape = [config.batch_size, 1, config.read_size, 4]
		with tf.variable_scope('conv1'):
			# k=10, hid=128
			kernel = weight_variable([1, 10, 4, 128])
			bias = weight_variable([128])
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
			kernel = weight_variable([1, 10, 128, 128])
			bias = weight_variable([128])
			conv = tf.nn.conv2d(self.x, kernel, [1,1,1,1], padding='SAME')
			conv2 = tf.nn.relu(conv + bias)

		# pool2
		# conv2.shape = [config.batch_size, 1, config.read_size/2, 128]
		with tf.variable_scope('pool'):
			pool2 = tf.nn.max_pool(conv2, ksize=[1, 1, 5, 1], strides=[1, 1, 2, 1], padding='SAME')

		# pool2.shape = [config.batch_size, 1, config.read_size/4, 128]
		with tf.variable_scope('local3') as scope:
			reshape = tf.reshape(pool2, [self.config.batch_size, -1])
			dim = reshape.get_shape()[1].value
			local3 = tf.nn.relu(linear(reshape, [dim, 400]))

		# local3
		# local3.shape = [config.batch_size, 400]
		with tf.variable_scope('local3') as scope:
			z_mean = linear(local3, (400, self.config.z_dim))
			z_log_sigma_sq = linear(local3, (400, self.config.z_dim))

		return (z_mean, z_log_sigma_sq)

	def decoder(self):
		# local1
		# z.shape = [config.batch_size, config.z_dim]
		with tf.variable_scope('local1') as scope:
			local1 = tf.relu(linear(self.z, (self.config.z_dim, 400)))

		# local2
		# local1.shape = [config.batch_size, 400]
		with tf.variable_scope('local2') as scope:
			local2 = tf.relu(linear(local1, (400, 128*config.read_size/4)))

		# deconv3
		# local2.shape = [config.batch_size, 128*config.read_size/4]
		with tf.variable_scope('deconv3') as scope:
			reshape = tf.reshape(local2, (config.batch_size, 1, config.read_size/4, 128))
			kernel = weight_variable([1, 10, 128, 128])
			bias = weight_variable([128])
			deconv = tf.nn.conv2d_transpose(reshape, kernel, (config.batch_size, 1, config.read_size/2, 128) [1, 1, 1, 1]) 
			deconv3 = tf.relu(deconv + bias)

		with tf.variable_scope('deconv4') as scope:
			kernel = weight_variable([1, 10, 4, 128])
			bias = weight_variable([128])
			deconv = tf.nn.conv2d_transpose(deconv3, kernel, (config.batch_size, 1, config.read_size, 128) [1, 1, 1, 1]) 
			deconv3 = tf.relu(deconv + bias)

		return deconv3

	def create_loss(self):
		self.reconstr_loss = \
				-tf.reduce_sum(self.x * tf.log(1e-10 + self.x_recon_theta)
						+ (1-self.x) * tf.log(1e-10 + 1 - self.x_recon_theta),1)
		reconstr_loss = self.reconstr_loss

		self.latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq 
				- tf.square(self.z_mean) 
				- tf.exp(self.z_log_sigma_sq), 1)
		latent_loss = self.latent_loss
		self.cost = tf.reduce_mean(reconstr_loss + latent_loss)
		self.optimizer = \
				tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.cost)
	
	def partial_fit(self, X):
		"""Train model based on mini-batch of input data.

		Return cost of mini-batch.
		"""

		'''
		print self.sess.run( self.latent_loss, feed_dict={self.x: X})
		print self.sess.run( self.reconstr_loss, feed_dict={self.x: X})
		print self.sess.run( self.x_recon_theta, feed_dict={self.x: X})
		exit()
		'''
		opt, cost = self.sess.run((self.optimizer, self.cost), 
								  feed_dict={self.x: X})
		return cost

	def transform(self, X):
		"""Transform data by mapping it into the latent space."""
		# Note: This maps to mean of distribution, we could alternatively
		# sample from Gaussian distribution
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
	for epoch in range(20):
		gen.reset_counter()
		avg_cost = 0.
		batch_size = nga.config.batch_size
		# Loop over all batches
		while True:
			batch_xs = gen.read_batch(batch_size) #mnist.train.next_batch(batch_size)
			if batch_xs == None:
				break

			'''
			plt.plot()
			plt.imshow(batch_xs[0].reshape(28, 28), vmin=0, vmax=1)
			plt.title("Test input")
			plt.show()
			'''

			# Fit training using batch data
			cost = nga.partial_fit(batch_xs)
			# Compute average loss
			avg_cost += cost / n_samples * batch_size

		# Display logs per epoch step
		if epoch % display_step == 0:
			print "Epoch:", '%04d' % (epoch+1), \
					"cost=", "{:.9f}".format(avg_cost)
	nga.save('checkpoints')

def plot_latent(nga, mnist):
	x_sample, y_sample = mnist.test.next_batch(5000)
	z_mu = nga.transform(x_sample)
	plt.figure(figsize=(8, 6)) 
	plt.scatter(z_mu[:, 0], z_mu[:, 1], c=np.argmax(y_sample, 1))
	plt.colorbar()
	plt.show()

def main():
	print 'hello'
	gen = read_gen.ReadGen('dna_10k.fa', 40, Config.read_size)
	nga = NGA(Config)
	train(nga, gen)
	return
	nga.load('checkpoints')
	plot_latent(nga, mnist)





	'''
	x_sample = mnist.test.next_batch(100)[0]
	for i in range(5):
		plt.plot()
		plt.imshow(x_sample[i].reshape(28, 28), vmin=0, vmax=1)
		plt.title("Test input")
		plt.show()
	'''



if __name__ == '__main__':
	main()
