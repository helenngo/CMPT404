import numpy as np
import random
import os, subprocess
import matplotlib.pyplot as plt
 
class Perceptron:
    def __init__(self,n_samples=100, thk=5, rad=10, sep=5.0):
        # Random linearly separated data
        self.V = (np.random.rand(3)*2)-1
        self.X = self.make_semi_circles(n_samples,thk,rad,sep)

    def make_semi_circles(self,n_samples=2000, thk=5, rad=10, sep=5, plot=False):
        noisey = np.random.uniform(low=-thk/100.0, high=thk/100.0, size=(n_samples // 2))
        noisex = np.random.uniform(low=-rad/100.0, high=rad/100.0, size=(n_samples // 2))
        separation = np.ones(n_samples // 2)*((-sep*0.1)-0.6)
        n_samples_out = n_samples // 2
        n_samples_in = n_samples - n_samples_out
    
        # generator = check_random_state(random_state)
        outer_circ_x = np.cos(np.linspace(0, np.pi, n_samples_out)) + noisex
        outer_circ_y = np.sin(np.linspace(0, np.pi, n_samples_out)) + noisey
        inner_circ_x = (1 - np.cos(np.linspace(0, np.pi, n_samples_in))) + noisex
        inner_circ_y = (1 - np.sin(np.linspace(0, np.pi, n_samples_in)) - .5) + noisey + separation
        
        t = np.ones((n_samples,1),dtype=np.int)
        x = np.vstack((np.append(outer_circ_x, inner_circ_x),
                    np.append(outer_circ_y, inner_circ_y))).T
        s = np.vstack([np.ones((n_samples_in,1), dtype=np.int)*-1,
                np.ones((n_samples_out,1), dtype=np.int)])
        
        x = np.hstack((t,x))
        X = []
        for i in range(n_samples):
            X.append((x[i], s[i]))
        return X
 
    def plot(self, mispts=None, vec=None, save=False,line=False):
        fig = plt.figure(figsize=(5,5))
        plt.xlim(-2.5,2.5)
        plt.ylim(-2.5,2.5)
        l = np.linspace(-2.5,2.5)

        if line: 
            w = self.linear_regression()
            m, b = -w[1]/w[2], -w[0]/w[2]
            plt.plot(l, m*l+b, 'k-')
        
        cols = {1: 'r', -1: 'b'}
        for x,s in self.X:
            plt.plot(x[1], x[2], cols[s[0]]+'o')
        if mispts:
            for x,s in mispts:
                plt.plot(x[1], x[2], cols[s[0]]+'.')
        if vec != None:
            aa, bb = -vec[1]/vec[2], -vec[0]/vec[2]
            plt.plot(l, aa*l+bb, 'g-', lw=2)
        if save:
            if not mispts:
                plt.title('N = %s' % (str(len(self.X))))
            else:
                plt.title('N = %s with %s test points' \
                          % (str(len(self.X)),str(len(mispts))))
            plt.savefig('p_N%s' % (str(len(self.X))), \
                        dpi=200, bbox_inches='tight')
        if line:
            return w
 
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
 
    def pla(self, save=False,line=False):
        # Initialize the weigths to zeros
        w = np.zeros(3)
        X, N = self.X, len(self.X)
        it = 0
        # Iterate until all points are correctly classified
        while self.classification_error(w) != 0:
            it += 1
            # Pick random misclassified point
            x, s = self.choose_miscl_point(w)
            # Update weights
            w += s*x
            if it > 10000:
                break
        self.w = w
        # Plot and save the last iteration.
        if save:
            if line: 
                w = self.plot(vec=w,line=True)
            else:
                self.plot(vec=w)
            plt.title('N = %s, Iteration %s\n' \
                    % (str(N),str(it)))
            plt.xlabel('x-axis')
            plt.ylabel('y-axis')
            plt.savefig('p_N%s_it%s' % (str(N),str(it)), \
                        dpi=200, bbox_inches='tight') 
        if line:
            return it, w
        else:
            return it
 
    def check_error(self, M, vec):
        check_pts = self.generate_points(M)
        return self.classification_error(vec, pts=check_pts)

    def linear_regression(self,save=False):

        X = []
        for x,s in self.X:
            X.extend(x)
            X.extend(s)

        X = np.array(X).reshape(-1, 4)
        N = len(X)
        x = X[:,[0,1,2]]
        y = X[:,3]
        w = np.linalg.pinv(x.T.dot(x)).dot(x.T).dot(y)

        if save:
            fig = plt.figure(figsize=(5,5))
            plt.xlim(-2.5,2.5)
            plt.ylim(-2.5,2.5)
            cols = {1: 'r', -1: 'b'}

            for x,s in self.X:
                plt.plot(x[1], x[2], cols[s[0]]+'o')

            l = np.linspace(-2.5,2.5)
            m, b = -w[1]/w[2], -w[0]/w[2]
            plt.plot(l, m*l+b, 'k-')
            plt.xlabel('x-axis')
            plt.ylabel('y-axis')
            plt.savefig('p_N%s_linear_regression' % (str(N)), \
                        dpi=200, bbox_inches='tight')

        return w

        
def main():
    #w = p.linear_regression(save=True)
    #print(w)
    iterations = np.empty([25,1])
    for k in range(1,26):
        p = Perceptron(n_samples=2000,sep=k*0.2)
        iterations[k-1] = p.pla()

    l = np.linspace(0.2, 5, 25, endpoint=True)
    plt.plot(l, iterations, 'b-')
    plt.xlabel('Separation (sep)')
    plt.ylabel('Number of Iterations')
    plt.savefig('Problem_3_2', \
      	         dpi=200, bbox_inches='tight')
    print(iterations)

main()
