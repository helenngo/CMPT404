import hw5gendata
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold


# generates data & split it into X (training input) and y (target output)
X, y = hw5gendata.genDataSet(1000)

neurons = 3  # <- number of neurons in the hidden layer
eta = 0.1       # <- the learning rate parameter

# here we create the MLP regressor
mlp =  MLPRegressor(hidden_layer_sizes=(neurons,), verbose=True, learning_rate_init=eta)
# here we train the MLP
mlp.fit(X, y)
# E_out in training
print("Training set score: %f" % mlp.score(X, y))

# now we generate new data as testing set and get E_out for testing set
X, y = hw5gendata.genDataSet(1000)
print("Testing set score: %f" % mlp.score(X, y))
ypred = mlp.predict(X)

plt.plot(X[:, 0], X[:, 1], '.')
plt.plot(X[:, 0], y, 'rx')
plt.plot(X[:, 0], ypred, '-k')
plt.show()

