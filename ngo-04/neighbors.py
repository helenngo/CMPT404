import numpy as np
from matplotlib import pyplot as plt
from sklearn import neighbors
from sklearn.model_selection import KFold

class Neighbors:
	def __init__(self,N):
		# Generate Dataset
		self.X , self.Y = self.genDataSet(N)

	def genDataSet(self,N):
		x = np.random.normal(0, 1, N) # input values
		ytrue = (np.cos(x) + 2) / (np.cos(x*1.4) +2)
		noise = np.random.normal(0, 0.2, N)
		y = ytrue + noise # target output value
		# Plot data
		# plt.plot(x,y,'.',label="Noisy Y")
		# plt.plot(x,ytrue,'rx',label="Original Y")
		# plt.legend(loc=1)
		# plt.xlabel(r'$x$')
		# plt.ylabel(r'$y$')
		# plt.title(r'Neighbors and Noise')
		# plt.savefig('neighbors.pdf', bbox_inches='tight')
		# plt.show()
		# Return testing and training dataset
		return x, y

	def bestK(self,folds = 10):
		# Reshape dataset for k-neighbors regression
		x = (self.X).reshape(-1, 1)
		y = (self.Y).reshape(-1, 1)
		# Housekeeping
		Eout = []
		N = len(x)
		maxk = int(N*(folds-1)/folds)
		# Output Eout list for k-neighbor range
		for n_neighbors in range(1, maxk, 2):
			kf = KFold(n_splits = folds)
			kscore = []
			for train, test in kf.split(x):
				x_train, x_test = x[train], x[test]
				y_train, y_test = y[train], y[test]
				reg = neighbors.KNeighborsRegressor(
					n_neighbors, weights = 'distance')
				reg.fit(x_train, y_train)
				kscore.append(abs(reg.score(x_test, 
					y_test)))
			Eout.append(sum(kscore)/len(kscore))
		return Eout


def main():
	kvalues = np.empty([300,1]) # vector of best k's
	# Conducting 100 trials
	for trial in range(0,100):
		print(trial)
		d = Neighbors(N=1000) # generate data
		Eout = d.bestK() # Eout array
		# Determining best k
		for j in range(0,3):
			i = Eout.index(max(Eout))
			k = 2*i+1
			kvalues[trial*3+j] = k
			print('k = ', k)
			print('Eout = ', max(Eout))
			del Eout[i]
	# Plotting histogram
	plt.hist(kvalues)
	plt.title("Best k-neighbors Values")
	plt.xlabel("k-neighbors")
	plt.ylabel("Frequency")
	plt.savefig('khistogram.pdf', bbox_inches='tight')
	plt.show()

main()
