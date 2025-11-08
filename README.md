# Convolutional Neural Network from Scratch in NumPy

This project implements a simple **Convolutional Neural Network** from scratch using only **NumPy**, no TensorFlow, no PyTorch.  
The goal is to clearly show how **convolution**, **pooling**, **softmax classification**, and **backpropagation** work at the lowest level.

The implementation is intentionally straightforward and educational rather than optimized for performance.  
It’s designed to help understand the math and data flow behind CNNs.

## Features

-   Implements all layers manually:
    -   **3×3 Convolution layer**
    -   **2×2 Max Pooling layer**
    -   **Fully-connected Softmax output layer**
-   **Manual forward and backward propagation**
-   **Cross-entropy loss**
-   **Gradient descent weight updates**
-   **MNIST dataset** used for training and evaluation
-   Entirely **NumPy-based**, no deep learning frameworks

## Network Architecture

**Input:** 28×28 grayscale image (MNIST digit)

**Layers:**

1. **Conv3x3** – 8 filters of size 3×3  
   → Output shape: 26×26×8
2. **MaxPool2** – 2×2 pooling  
   → Output shape: 13×13×8
3. **Softmax** – Fully connected layer  
   → Input: 13×13×8 = 1352  
   → Output: 10 (digit classes 0–9)
