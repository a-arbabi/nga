import argparse
import numpy as np


class ReadGen:
	def __init__(self, fa_file, cov, read_len):
		self.seq = open(fa_file).read().strip()
		self.total_reads = len(self.seq)*cov/read_len
		self.read_len = read_len


		self.reset_counter()


	def read2vec(self, s):
		nuc2id = {'A':0, 'C':1, 'T':2, 'G':3}
		v = np.zeros((1, len(s), 4))
		for i in range(len(s)):
			v[0][i][nuc2id[s[i]]] = 1
		return v

	def reset_counter(self):
		self.counter = 0
		self.starts = np.random.randint(0,len(self.seq)-100, self.total_reads)

		self.read_vectors = []

		for i in self.starts:
			self.read_vectors.append(self.read2vec(self.seq[i:i+self.read_len]))
		self.all_reads_vector = np.vstack(self.read_vectors)


	def get_long_seq(self, size, start=None):
		if start == None:
			start = np.random.randint(0,len(self.seq)-size, 1)
		return self.seq[start : start + size]


	def read_batch(self, batch_size):
		if self.counter + batch_size > self.all_reads_vector.shape[0]:
			return None, None
		batch = self.all_reads_vector[self.counter:self.counter + batch_size]
		starts = self.starts[self.counter:self.counter + batch_size]
		self.counter += batch_size
		return batch, starts


def main():
	parser = argparse.ArgumentParser(description='Hello!')
	parser.add_argument('fa_file') #, help="The location where the checkpoints are stored, default is \'checkpoints/\'", default="sent_checkpoints/")
	args = parser.parse_args()

	cov = 40
	read_len = 100

	read_gen = ReadGen(args.fa_file, cov, read_len)
	print read_gen.all_reads_vector.shape
	return
	print read_gen.read_batch(3)[0]
	print read_gen.read_batch(3)[0]




if __name__ == '__main__':
	main()
