# MNIST-Classification-using-LibTorch

Custom CNN Model for MNIST Digit Recognition (Python & C++)

This repository compares implementations of a custom Convolutional Neural Network (CNN) model for recognizing handwritten digits in the MNIST dataset, written in both Python and C++.

MNIST Dataset:

The MNIST dataset is a widely used benchmark for image classification tasks. It consists of 60,000 images of handwritten digits (0-9) for training and 10,000 images for testing.

Model Architecture:

Both Python and C++ implementations use the same CNN architecture with the following layers:

    Convolutional Layer 1: 1 input channel (grayscale images), 10 output channels, kernel size 5x5
    Convolutional Layer 2: 10 input channels, 20 output channels, kernel size 5x5
    Convolutional Layer 3: 20 input channels, 30 output channels, kernel size 3x3
    Fully Connected Layer 1: 120 input neurons, 50 output neurons
    Fully Connected Layer 2: 50 input neurons, 10 output neurons (one for each digit)

The model uses activation functions (GELU) and dropout layers for regularization.

Running the Code:

Prerequisites:

    Python: Ensure you have Python 3.x installed with PyTorch and torchvision libraries.
    C++: A C++ compiler with support for Torch library is required.

Download MNIST data:

Both implementations assume the MNIST data is downloaded in the ./data directory. You can download it manually or use tools provided by PyTorch or Torch.

Run the code:

    Python: Execute python main.py.
    C++: Compile and run the C++ code following the instructions for your specific compiler and Torch setup.

Note: Training times may vary depending on your hardware.
