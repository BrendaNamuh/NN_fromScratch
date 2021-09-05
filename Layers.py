import random
# random.uniform(0, 1)
import nnfs as nnfs
from nnfs.datasets import spiral_data
import numpy as np
import matplotlib.pyplot as plt
import ActivationFunctions as func

nnfs.init()

'''X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]
'''

# 4 input nodes
np.random.seed(0)


class LayerDense:
    def __init__(self, nmbr_inputs, nmbr_nodes):
        self.weights = 0.1 * np.random.randn(nmbr_inputs, nmbr_nodes)  # To avoid having to transpose weights on line 23
        self.biases = np.zeros((1, nmbr_nodes))
        self.output = []

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


# 10 examples, 3 classes
X, y = spiral_data(100, 3)
print(X.size,y.size)
#plt.scatter(X[0])
plt.show()


# 100 examples, 3 classes
X, y = spiral_data(100, 3) #2,5 --- 5,3


dense1 = LayerDense(2,3)
activation1=func.Relu()

dense2 = LayerDense(3,3)
activation2 = func.Softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(np.array(activation1.output))   #layer only takes np array??!!

activation2.forward(dense2.output)

print(activation2.output)


