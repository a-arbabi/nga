import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

def plot():

	epochs = []
	for line in open('epoch2'):
		tmp = line.strip().split(' ')
		epochs.append([int(tmp[0]), float(tmp[1])])
	epochs = np.array(epochs)


	vis_x = epochs[:, 0]
	vis_y = epochs[:, 1]

	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')

	print vis_y
	#plt.scatter(vis_x, vis_y, c=y) #, cmap=plt.cm.get_cmap("jet", 10))
	plt.plot(vis_x, vis_y, lw=1) #, c=y)
	plt.xlabel(r'Epoch', fontsize=12)
	plt.ylabel(r'Loss', fontsize=12)
	plt.show()


plot()
