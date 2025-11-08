import numpy as np

class Conv3x3:
    # Convolution layer that uses 3x3 filters

    def __init__(self, num_filters):
        self.num_filters = num_filters

        # filters is a 3D array with dim (num_filters, 3, 3)
        # divided by 9 to reduce the variance of initial values
        # If the initial values are too large/small, training the network will be ineffective
        self.filters = np.random.randn(num_filters, 3, 3) / 9 # Where actual weights of the filters are stored

    # Helper generator function (iterator)
    def iterate_regions(self, image):
        '''
        Generates all possible 3x3 image regions using "valid padding".
        - image is a 2D numpy array (assumes it is a single channel image, not RGB)
        '''
        self.last_input = input

        h, w = image.shape

        for i in range(h - 2):
            for j in range(w - 2):
                im_region = image[i:(i+3), j:(j+3)]
                yield im_region, i, j

    def forward(self, input):
        '''
        Performs a forward pass of the conv layer using the given input.
        Returns a 3D numpy array (feature map volume) with dim (h - 2, w - 2, num_filters).
        - input is a 2D numpy array
        '''
        h, w = input.shape
        output = np.zeros((h - 2, w - 2, self.num_filters))

        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))

        return output
    
    def backprop(self, d_L_d_out, learn_rate):
        '''
        Performs a backward pass of the conv layer.
        '''
        d_L_d_filters = np.zeros(self.filters.shape)

        for im_region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                d_L_d_filters[f] += d_L_d_out[i, j, f] * im_region

        # Update filters
        self.filters -= learn_rate * d_L_d_filters

        # Return nothing because this is used as the first layer
        return None