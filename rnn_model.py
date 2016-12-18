import tensorflow as tf
import numpy as np
import read_gen
import sys


def linear(name, x, shape):
	w = weight_variable(name + 'W', shape)
	b = weight_variable(name + 'B',(shape[1]))
	return tf.matmul(x,w) + b

def kernel_variable(name, shape):
	return tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer_conv2d())

def weight_variable(name, shape):
	return tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer())
#return tf.get_variable(name, shape, initializer=tf.truncated_normal_initializer(stddev = 0.01))

class Config:
	read_size = 100
	hidden_size = 100
	num_layers = 1
	learning_rate = 0.01
	lr_decay = 0.9
	batch_size = 256
	test_size = 500

class NGA:
	def __init__(self, config):
		self.config = config
		self.lr = tf.Variable(config.learning_rate, False)
		self.x = tf.placeholder(tf.float32, shape = [None, config.read_size, 4])
		inputs = tf.unpack(self.x, axis = 1)

		targets = tf.slice(self.x, [0,1,0], [-1,-1,-1])

		with tf.variable_scope("NGA") as scope:
			single_cell = tf.nn.rnn_cell.GRUCell(config.hidden_size)
			self.cell = tf.nn.rnn_cell.MultiRNNCell([single_cell]*config.num_layers, state_is_tuple=False)

		outputs, state = tf.nn.rnn(self.cell, inputs, dtype=tf.float32, \
				scope=scope) #, initial_state=self._initial_state)
		logits = [self.hidden2logit(out, reuse = (i>0)) for i, out in enumerate(outputs)]

		self.rnn_output = outputs[-1]
		self.rnn_state = state

		logits_packed = tf.pack(logits[:-1], axis=1)

		self.loss = tf.reduce_mean(tf.reduce_sum(\
				tf.nn.softmax_cross_entropy_with_logits(\
						logits_packed, targets), [1]))
		self.optimizer = tf.train.AdamOptimizer(\
				learning_rate=self.lr).minimize(self.loss)

		self.start_state = tf.placeholder(tf.float32, shape=[None, self.config.num_layers*self.config.hidden_size])
		self.start_output = tf.placeholder(tf.float32, shape=[None, self.config.hidden_size])

		self.prediction_probs, self.prediction_final_state,\
				self.prediction_final_output = self.prediction_model(\
						self.config.test_size, self.start_state, self.start_output)

		self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
		self.sess.run(tf.initialize_all_variables())



	def hidden2logit(self, h, reuse=False):
		with tf.variable_scope("NGA") as scope:
			if reuse:
				scope.reuse_variables()
			layer1 = tf.nn.relu(linear('layer1', h, [self.config.hidden_size, 200]))
			layer2 = tf.nn.relu(linear('layer2', layer1, [200, 200]))
			layer3 = linear('layer3', layer2, [200, 4])
		return layer3



	def prediction_model(self, out_length, state, output):
		state = self.start_state
		output = self.start_output

		predictions = []
		for i in range(out_length):
			logits = self.hidden2logit(output, reuse=True)
			predictions.append(tf.nn.softmax(logits))
			with tf.variable_scope("NGA") as scope:
				scope.reuse_variables()
				(output, state) = self.cell(predictions[-1], state)
		
		packed_predictions = tf.pack(predictions, axis=1)
		return packed_predictions, state, output

#				self.outputs.append(rnn_output)


	def partial_fit(self, X):
		"""Train model based on mini-batch of input data.

		Return cost of mini-batch.
		"""

		'''
		print X.shape
		print self.sess.run(self.x_recon_theta_tran, feed_dict={self.x: X}).shape
		print self.sess.run(self.x_tran, feed_dict={self.x: X}).shape
		'''
		opt, loss = self.sess.run((self.optimizer, self.loss), 
								  feed_dict={self.x: X})
		return loss

	def generate_next(self, X, out_length):
		state, output = self.sess.run([self.rnn_state, self.rnn_output],
								  feed_dict={self.x: X})
		probs_list = []
		for i in range(out_length/self.config.test_size):
			state, output, probs = self.sess.run([self.prediction_final_state,
				self.prediction_final_output, self.prediction_probs],
				feed_dict={self.start_state: state, self.start_output: output})
			probs_list.append(probs)

		final_probs = np.concatenate(probs_list, axis=1)

		return final_probs
		for out in output:
			out_dna += nuc2id[np.argmax(out[0])]
		print probs.shape
		'''
		for out in output:
			out_dna += nuc2id[np.argmax(out[0])]
		return out_dna
		'''

	def get_hiddens(self, X):
		return  self.sess.run(self.outputs, 
								  feed_dict={self.x: X})

	def save(self, rep_dir):
		saver = tf.train.Saver()
		saver.save(self.sess, rep_dir+'/training.ckpt')

	def load(self, rep_dir):
		saver = tf.train.Saver()
		saver.restore(self.sess, rep_dir + '/training.ckpt')

def train(nga, gen):
	test_length = 8000
	total_dna = gen.get_long_seq(test_length+100, 0)
	dna_test = total_dna[:100]
	rest_dna = total_dna[100:]
	# Training cycle
	display_step = 50
	for epoch in range(501):
		gen.reset_counter()
		total_cost = 0.
		count = 0
		batch_size = nga.config.batch_size
		# Loop over all batches

		lr_new = nga.config.learning_rate * (nga.config.lr_decay ** int(epoch/50))
		nga.sess.run(tf.assign(nga.lr, lr_new))

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
				"cost=", total_cost/count,
			final_probs = nga.generate_next(gen.read2vec(dna_test), test_length) 

			predict = probs2str(final_probs)[0]
		
			evaluation = map(int,[x==y for (x,y) in zip(predict, rest_dna)])
			good_size = (evaluation+[0]).index(0)
			print good_size, '/', len(predict)

			sys.stdout.flush()

			 
#	create_sample_for_plot(nga.get_hiddens(gen.read2vec(dna_test)))
	nga.save('checkpoints')

def probs2str(probs):
	final_args = np.argmax(probs, 2)
	nuc2id = ['A', 'C', 'T', 'G']
	final_dnas = []
	for row in final_args:
		final_dnas.append(''.join([nuc2id[x] for x in row]))
	return final_dnas
def test(nga, gen, ind):
	test_length = 1000
	total_dna = gen.get_long_seq(test_length+100, 5000+ind*10)
	dna_test = total_dna[:100]
	rest_dna = total_dna[100:]

	probs = nga.generate_next(gen.read2vec(dna_test), test_length)
	predict = probs2str(probs)[0]
	for i,r in enumerate(probs[0]):
		print np.max(r)*100, predict[i], rest_dna[i]
		if i==200:
			break
	exit()

	evaluation = map(int,[x==y for (x,y) in zip(predict, rest_dna)])
	good_size = (evaluation+[0]).index(0)
	print good_size, '/', len(predict)

def main():
	print 'hello'
	gen = read_gen.ReadGen('dna_10k.fa', 20, Config.read_size)
	nga = NGA(Config)
	#nga.load('checkpoints')
	#test(nga, gen, 0)
	train(nga, gen)
	exit()
	for i in range(50):
		test(nga, gen, i)
	#create_sample_for_plot(nga, gen)


if __name__ == '__main__':
	main()
