import numpy as np
from tensorflow.keras.datasets import mnist # type: ignore
from conv import Conv3x3
from maxpool import MaxPool2

# Using the MNIST dataset to test the CNN
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

conv = Conv3x3(8) # Conv layer with 8 filters
pool = MaxPool2()

output = conv.forward(train_images[0])
output = pool.forward(output)

print(output.shape)