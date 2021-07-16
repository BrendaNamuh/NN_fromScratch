import numpy as np
import math


class Relu:
    def __init__(self):
        self.output = []

    def forward(self, inputs):
        for i in inputs:
            self.output.append(np.maximum(0, i))
        return self.output


class Softmax:

    def forward(self, inputs):
        values_exp = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))  # Avoids overflow from exponentiation
        base = np.sum(values_exp, axis=1, keepdims=True)  # is a column
        self.output = values_exp / base  #probabilites
        return self.output
