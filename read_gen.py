import argparse
import numpy as np


class ReadGen:
	def __init__(self, fa_file, cov, read_len):
		seq = open(fa_file).read().strip()
		total_reads = len(seq)*cov/read_len

		starts = np.random.randint(0,len(seq)-100, total_reads)

		self.reads = []
		self.read_vectors = []

		for i in starts:
			self.reads.append(seq[i:i+read_len])
			self.read_vectors.append(self.read2vec(seq[i:i+read_len]))

		self.all_reads_vector = np.vstack(self.read_vectors)
		self.reset_counter()


	def read2vec(self, s):
		nuc2id = {'A':0, 'C':1, 'T':2, 'G':3}
		v = np.zeros((1, len(s), 4))
		for i in range(len(s)):
			v[0][i][nuc2id[s[i]]] = 1
		return v

	def reset_counter(self):
		self.counter = 0

	def read_batch(self, batch_size):
		if self.counter + batch_size > self.all_reads_vector.shape[0]:
			return None
		batch = self.all_reads_vector[self.counter:self.counter + batch_size]
		self.counter += batch_size
		return batch



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
