import numpy as np

class Softmax:
    # A standard fully-connected layer with softmax activation

    def __init__(self, input_len, nodes):
        # We divide by input_len to reduce the variance of initial values
        self.weights = np.random.randn(input_len, nodes) / input_len
        self.biases = np.zeros(nodes)

    def forward(self, input):
        '''
        Performs a forward pass of the softmax layer using the given input.
        Returns a 1D numpy array containing respective probability values
        '''
        input = input.flatten()

        input_len, nodes = self.weights.shape

        totals = input @ self.weights + self.biases
        exp = np.exp(totals)
        return exp / np.sum(exp, axis=0) # Softmax value