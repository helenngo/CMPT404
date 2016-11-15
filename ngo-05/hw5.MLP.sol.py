import hw5gendata
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold

# number of samples
N = 10000

# generate data & split it into X (training input) and y (target output)
X, y = hw5gendata.genDataSet(N)

# linear regression solution
w=np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)


#neurons  <- number of neurons in the hidden layer
#eta  <- the learning rate parameter

bestNeurons=0
bestEta=0
bestScore=float('-inf')
score=0
for neurons in range(1,100,1):
  for eta in range(1,11,1):
    eta=eta/10.0
    kf = KFold(n_splits=10)
    cvscore=[]
    for train, validation in kf.split(X):
      X_train, X_validation, y_train, y_validation = X[train, :], X[validation, :], y[train], y[validation]
      # here we create the MLP regressor
      mlp =  MLPRegressor(hidden_layer_sizes=(neurons,), verbose=False, learning_rate_init=eta)
      # here we train the MLP
      mlp.fit(X_train, y_train)
      # now we get E_out for validation set
      score=mlp.score(X_validation, y_validation)
      cvscore.append(score)

    # average CV score
    score=sum(cvscore)/len(cvscore)
    if (score > bestScore):
      bestScore=score
      bestNeurons=neurons
      bestEta=eta
      print("Neurons " + str(neurons) + ", eta " + str(eta) + ". Testing set CV score: %f" % score)

# here we get a new training dataset
X, y = hw5gendata.genDataSet(N)
# here we create the final MLP regressor
mlp =  MLPRegressor(hidden_layer_sizes=(bestNeurons,), verbose=True, learning_rate_init=bestEta)
# here we train the final MLP
mlp.fit(X, y)
# E_out in training
print("Training set score: %f" % mlp.score(X, y)) 
# here we get a new testing dataset
X, y = hw5gendata.genDataSet(N)
# here test the final MLP regressor and get E_out for testing set
ypred=mlp.predict(X)
score=mlp.score(X, y)
print("Testing set score: %f" % score)
plt.plot(X[:, 0], X[:, 1], '.')
plt.plot(X[:, 0], y, 'rx')
plt.plot(X[:, 0], ypred, '-k')
ypredLR=X.dot(w)
plt.plot(X[:, 0], ypredLR, '--g')
plt.show()
