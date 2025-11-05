import numpy as np
from tensorflow.keras.datasets import mnist # type: ignore
from conv import Conv3x3
from maxpool import MaxPool2
from softmax import Softmax

# Using the MNIST dataset to test the CNN
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

conv = Conv3x3(8) # Conv layer with 8 filters
pool = MaxPool2()
softmax = Softmax(13 * 13 * 8, 10) # 13x13x8 -> 10

def forward(image, label):
    '''
    Completes a forward pass of the CNN and calculates the accuracy and cross-entropy loss
    '''
    out = conv.forward((image / 255) - 0.5) # Transform image from [0, 255] to [-0.5, 0.5]
    out = pool.forward(out)
    out = softmax.forward(out)

    # Calculate cross-entropy loss and accuracy
    loss = -np.log(out[label])
    acc = 1 if np.argmax(out) == label else 0

    return out, loss, acc

print("MNIST CNN initialized")

loss = 0
num_correct = 0

# With random weight initialization (and no backpropagation),
# the CNN should be as good as random guessing, so around 10% accuracy
for i, (im, label) in enumerate(zip(test_images, test_labels)):
    _, l, acc = forward(im, label)
    loss += l
    num_correct += acc

    if i % 100 == 99:
        print(
        '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
        (i + 1, loss / 100, num_correct)
        )
        loss = 0
        num_correct = 0