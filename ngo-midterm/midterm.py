import numpy as np
import random
import os, subprocess
import matplotlib.pyplot as plt
import copy
from sklearn.datasets.samples_generator import make_blobs
import time
from numpy import genfromtxt
 
class Pocket:
    def __init__(self):
        # Random linearly separated data
        self.V = (np.random.rand(3)*2)-1
        self.X = self.generate_points()

    def generate_points(self):
      #read digits data & split it into X (training input) and y (target output)
      dataset = genfromtxt('features.csv', delimiter=' ') 
      y=dataset[:,0]
      X=dataset[:,1:]
      y[y!=1] = -1 #rest of numbers are negative class
      y[y==1] = +1 #number zero is the positive class
      # Define x1, x2 limits
      self.x1min= np.floor(np.amin(X[:,0]))
      self.x1max = np.ceil(np.amax(X[:,0]))
      self.x2min = np.floor(np.amin(X[:,1]))
      self.x2max = np.ceil(np.amax(X[:,1]))
      # Plots data
      c0 = plt.scatter(X[y==-1,0],X[y==-1,1], 
        s=20, color='r', marker='x')
      c1 = plt.scatter(X[y==1,0],X[y==1,1], 
        s=20, color='b', marker='o')
      # c2 = plt.scatter(X[y==1,0][[0,1,2,3,4]],
      #   X[y==1,1][[0,1,2,3,4]], 
      #   s=20, color='c', marker='o')
      # Display legend
      plt.legend((c0,c1), ('All Other Numbers −1', 'Number One +1'), 
        loc='upper right',scatterpoints=1, fontsize=11)
      # Dispay axis legends and title
      plt.xlabel(r'$x_1$')
      plt.ylabel(r'$x_2$')
      plt.title(r'Intensity and Symmetry of Digits')
      # saves the figure into a .pdf file (desired!)
      plt.savefig('midterm.plot.pdf', bbox_inches='tight')
      # plt .show()
      # Create x matrix
      N = len(X)
      x0 = np.ones((N,1),dtype=np.int)
      x = np.hstack((x0,X))
      X = []
      for i in range(N):
        X.append((x[i], y[i]))
      return X
 
    def plot(self, mispts=None, vec=None, vec0=None, lr_vec=None, save=False, file='plot.pdf'):
    	plt.figure()
    	# Define x1, x2 limits
    	plt.axis([self.x1min, self.x1max,self.x2min, self.x2max])
    	l = np.linspace(self.x1min,self.x1max)
    	# Ploting data
    	cols = {1: 'bo', -1: 'rx'}
    	for x,y in self.X:
            plt.plot(x[1], x[2], cols[y])
    	if mispts:
        	for x,y in mispts:
        		plt.plot(x[1], x[2], cols[y])
        # Plot functiions
    	if vec != None:
            aa, bb = -vec[1]/vec[2], -vec[0]/vec[2]
            plt.plot(l, aa*l+bb, 'g-', lw=2)
    	if vec0 != None:
            aa, bb = -vec0[1]/vec0[2], -vec0[0]/vec0[2]
            plt.plot(l, aa*l+bb, 'g:', lw=1)
    	if lr_vec != None:
            aa, bb = -lr_vec[1]/lr_vec[2], -lr_vec[0]/lr_vec[2]
            plt.plot(l, aa*l+bb, 'k-', lw=2)
    	if save:
            if not mispts:
                plt.title('N = %s' % (str(len(self.X))))
            else:
                plt.title('N = %s with %s test points' \
                          % (str(len(self.X)),str(len(mispts))))
            plt.savefig(file, dpi=200, bbox_inches='tight')
 
    def classification_error(self, vec, pts=None):
        # Error defined as fraction of misclassified points
        if not pts:
            pts = self.X
        M = len(pts)
        n_mispts = 0
        for x,s in pts:
            if int(np.sign(vec.T.dot(x))) != s:
                n_mispts += 1
        error = n_mispts / float(M)
        return error
 
    def choose_miscl_point(self, vec):
        pts = self.X
        mispts = []
        for x,s in pts:
            if int(np.sign(vec.T.dot(x))) != s:
                mispts.append((x, s))
        return mispts[random.randrange(0,len(mispts))]
 
    def pla(self, save=False, linear_regression=False, 
    	file='pocket_learning.pdf'):
        # Initialize the weights to solution of linear regression
        if linear_regression:
        	lr_vec = self.linear_regression()
        	w0 = copy.deepcopy(lr_vec)
        	w = copy.deepcopy(w0)
        # Initialize the weigths to zeros
        else:
        	w = np.zeros(3)
        	w0 = np.zeros(3)
        # Reassign variables
        X, N = self.X, len(self.X)
        # Initialize convergence and iteration counters
        count = 0
        it = 0
        # Iterate until all points are correctly classified
        while self.classification_error(w) != 0:
            it += 1
            # Pick random misclassified point
            x, y = self.choose_miscl_point(w0)
            # Update weights
            w0 += y*x
            # Update if new weights are better
            if self.classification_error(w)>self.classification_error(w0):
            	w = copy.deepcopy(w0)
            	count = 0
            else:
                count += 1
            # Converge after 500 iterations with the same wieghts
            if count > 500:
            	break
            # Converge after 10000 iterations overall 
            if it > 10000:
                break
        self.w = w
        # Plot and save the last iteration
        if save:
            if linear_regression: 
                self.plot(vec=w, vec0=w0, lr_vec=lr_vec, 
                	file=file, save=True)
            else:
                self.plot(vec=w, vec0=w0, file=file, save=True)
        # Return number of iterations
        return it, self.classification_error(w)
 
    def check_error(self, M, vec):
        check_pts = self.generate_points(M)
        return self.classification_error(vec, pts=check_pts)

    def linear_regression(self, save=False, file='linear_regression.pdf'):
        # Reshape X matrix in order to calculate solution
        X = []
        for x,y in self.X:
            X.extend(x)
            X.extend([y])
        X = np.array(X).reshape(-1, 4)
        N = len(X)
        x = X[:,[0,1,2]]
        y = X[:,3]
        # Compute weights using equation
        w = np.linalg.pinv(x.T.dot(x)).dot(x.T).dot(y)
        print(self.classification_error(w))
        # Plot and save the solution of linear regression
        if save:
        	self.plot(vec=None, vec0=None, lr_vec=w, file=file, save=True)
        # Return weights
        return w

        
def main():
    p = Pocket()
    # pocket algorithm
    # localtime = time.asctime( time.localtime(time.time()) )
    # print ("Start time for pocket algorithm:", localtime)
    # random.seed(6)
    # iterations = p.pla(save=True)
    # print(iterations)
    # linear regression
    localtime = time.asctime( time.localtime(time.time()) )
    print ("Start time for linear regression:", localtime)
    p.linear_regression(save=True)
    localtime = time.asctime( time.localtime(time.time()) )
    # pocket with linear regression start
    # print ("Start time for pocket algorithm with linear regression solution:", localtime)
    # random.seed(6)
    # iterations = p.pla(save=True,linear_regression=True, file='pocket_with_lr.pdf')
    # print(iterations)
    # localtime = time.asctime( time.localtime(time.time()) )
    print ("End:", localtime)
main()
