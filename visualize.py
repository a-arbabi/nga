from tsne import bh_sne
import matplotlib.pyplot as plt
import h5py
import numpy as np

def plot():
	h5f = h5py.File('plot_data.h5', 'r')
	z = np.array(h5f['z'])
	y = np.array(h5f['y'])
	h5f.close()

	vis_data = bh_sne(np.float64(z))

	vis_x = vis_data[:10, 0]
	vis_y = vis_data[:10, 1]
	y = y[:10]

	plt.scatter(vis_x, vis_y, c=y) #, cmap=plt.cm.get_cmap("jet", 10))
	plt.plot(vis_x, vis_y, lw=0.1) #, c=y)
	plt.colorbar() #ticks=range(10))
	plt.clim(0.0, 1.0)
	plt.show()

def main():
	plot()

if __name__ == '__main__':
	main()
