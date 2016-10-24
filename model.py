import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

def linear(x, name, shape):
	w = tf.get_variable(name+'_w', shape, initializer = tf.contrib.layers.xavier_initializer(),)
	b = tf.zeros((shape[1]))
	return tf.matmul(x,w) + b

class Config:
	x_dim = 28*28
	z_dim = 2
	#z_dim = 20
	encode_layer1_size = 500
	encode_layer2_size = 500
	decode_layer1_size = 500
	decode_layer2_size = 500
	learning_rate = 0.001
	batch_size =50

class NGA:
	def __init__(self, config):
		self.config = config
		self.x = tf.placeholder(tf.float32, shape = [None, config.x_dim])
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
		layer_1 = tf.nn.softplus(linear(self.x, 'encode_layer1',\
				(self.config.x_dim, self.config.encode_layer1_size)))

		layer_2 = tf.nn.softplus(linear(layer_1, 'encode_layer2',\
				(self.config.encode_layer1_size, self.config.encode_layer2_size)))

		z_mean = linear(layer_2, 'z_mean_layer', \
				(self.config.encode_layer2_size, self.config.z_dim))
		z_log_sigma_sq = linear(layer_2, 'z_sigma_layer', \
				(self.config.encode_layer2_size, self.config.z_dim))

		return (z_mean, z_log_sigma_sq)

	def decoder(self):
		layer_1 = tf.nn.softplus(linear(self.z, 'decode_layer1',\
				(self.config.z_dim, self.config.decode_layer1_size)))

		layer_2 = tf.nn.softplus(linear(layer_1, 'decode_layer2',\
				(self.config.decode_layer1_size, self.config.decode_layer2_size)))
		x_recon_theta = tf.nn.sigmoid(linear(layer_2, 'x_recon_layer', \
				(self.config.decode_layer2_size, self.config.x_dim)))

		return x_recon_theta

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

def train(nga, mnist):
	# Training cycle
	display_step = 5
	for epoch in range(20):
		avg_cost = 0.
		batch_size = nga.config.batch_size
		n_samples = mnist.train.num_examples
		total_batch = int( n_samples/ batch_size)
		# Loop over all batches
		for i in range(total_batch):
			batch_xs, _ = mnist.train.next_batch(batch_size)
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
	mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
	nga = NGA(Config)
	#train(nga, mnist)
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
