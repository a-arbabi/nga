import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
#import matplotlib.pyplot as plt
import read_gen

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
	z_dim = 100
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
		self.x = tf.placeholder(tf.float32, shape = [None, 1, config.read_size, 4])
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
		self.tmp0 = self.x
		with tf.variable_scope('conv1'):
			# k=10, hid=128
			kernel = kernel_variable('w', [1, 10, 4, 128])
			self.tmp_kernel = kernel
			bias = weight_variable('b', [128])
			conv = tf.nn.conv2d(self.x, kernel, [1,1,1,1], padding='SAME')
			conv1 = tf.nn.relu(conv + bias)
		self.tmp1 = conv1

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
		self.tmp2 = conv2

		# pool2
		# conv2.shape = [config.batch_size, 1, config.read_size/2, 128]
		with tf.variable_scope('pool'):
			pool2 = tf.nn.max_pool(conv2, ksize=[1, 1, 5, 1], strides=[1, 1, 2, 1], padding='SAME')

		# pool2.shape = [config.batch_size, 1, config.read_size/2, 128]
		with tf.variable_scope('local3') as scope:
			reshape = tf.reshape(pool2, [self.config.batch_size, -1])
			dim = self.config.read_size * 128 / 2
			#dim = reshape.get_shape()[1].value
			local3 = tf.nn.relu(linear('w', reshape, [dim, 400]))
		self.tmp3 = local3

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
			kernel = kernel_variable('w', [1, 10, 4, 128])
			bias = weight_variable('b', [4])
			deconv = tf.nn.conv2d_transpose(deconv3, kernel, (self.config.batch_size, 1, self.config.read_size, 4), [1, 1, 2, 1]) 
			deconv4 = tf.nn.softmax(deconv + bias, dim=-1)

		return deconv4

	def create_loss(self):
		self.reconstr_loss = \
				-tf.reduce_sum(self.x * tf.log(1e-10 + self.x_recon_theta) ,[1,2,3])
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

		X = np.expand_dims(X, 1)
		print "___"
		'''
		print self.sess.run( self.z_log_sigma_sq, feed_dict={self.x: X})
		print np.max(self.sess.run( self.z_log_sigma_sq, feed_dict={self.x: X}))
		print np.min(self.sess.run( self.z_log_sigma_sq, feed_dict={self.x: X}))
		print self.sess.run( self.z_mean, feed_dict={self.x: X})
		print np.max(self.sess.run( self.z_mean, feed_dict={self.x: X}))
		print np.min(self.sess.run( self.z_mean, feed_dict={self.x: X}))
		print self.sess.run( self.latent_loss, feed_dict={self.x: X})
		'''
		'''
		print self.sess.run( self.reconstr_loss, feed_dict={self.x: X})
		print self.sess.run( self.x_recon_theta, feed_dict={self.x: X})
		'''
		'''
		print np.sum(self.sess.run( self.tmp0, feed_dict={self.x: X}))
		print np.sum(self.sess.run( self.tmp_kernel, feed_dict={self.x: X}))
		print np.max(self.sess.run( self.tmp_kernel, feed_dict={self.x: X}))
		print np.min(self.sess.run( self.tmp_kernel, feed_dict={self.x: X}))
		print (self.sess.run( self.tmp_kernel, feed_dict={self.x: X}))
		print np.min(self.sess.run( self.x_recon_theta, feed_dict={self.x: X}))
		print np.sum(self.sess.run( self.x_recon_theta, feed_dict={self.x: X}))
		print np.sum(self.sess.run( self.reconstr_loss, feed_dict={self.x: X}))
		print "-"
		print np.sum(self.sess.run( self.tmp1, feed_dict={self.x: X}))
		print np.sum(self.sess.run( self.tmp2, feed_dict={self.x: X}))
		print 'tmp3 :: ' , np.sum(self.sess.run( self.tmp3, feed_dict={self.x: X}))
		print 'zmean :: ' , np.sum(self.sess.run( self.z_mean, feed_dict={self.x: X}))
		print 'zlogsigma :: ' , np.sum(self.sess.run( self.z_log_sigma_sq, feed_dict={self.x: X}))
		print self.sess.run( self.cost, feed_dict={self.x: X})
		'''

		opt, cost = self.sess.run((self.optimizer, self.cost), 
								  feed_dict={self.x: X})
		'''
		cost = self.sess.run(self.x_recon_theta, 
								  feed_dict={self.x: X})
		'''
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
	display_step = 20
	for epoch in range(20):
		print "Epoch: " + str(epoch)
		gen.reset_counter()
		total_cost = 0.
		count = 0
		batch_size = nga.config.batch_size
		# Loop over all batches
		while True:
			batch_xs = gen.read_batch(batch_size) #mnist.train.next_batch(batch_size)
			if batch_xs is None:
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
			total_cost += cost 
			count += 1

		# Display logs per epoch step
			if  count % display_step == 0:
				print "Step:", '%04d' % (count), \
					"cost=", "{:.9f}".format(total_cost/count)
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
