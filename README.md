# MyTorch: A Lightweight Educational Deep Learning Library

**MyTorch** is a minimalistic, educational deep learning framework designed for students and developers who wish to explore the fundamental concepts behind neural networks. While established libraries like PyTorch or TensorFlow abstract many of the complexities, MyTorch takes a hands-on approach by allowing users to dive deep into building and understanding neural networks from scratch. It’s the perfect tool for learning backpropagation, optimizers, loss functions, layers, and more—without relying on external black-box implementations.

## Why MyTorch?

Unlike traditional deep learning libraries, MyTorch emphasizes manual interaction with core concepts. This forces a deeper understanding of how neural networks operate. The project is built in a modular fashion, allowing you to easily modify and extend its components to experiment with various architectures and learning algorithms.

## Key Features

### Activation Functions (`activation/`)
Activation functions introduce non-linearity into a neural network, allowing it to model more complex patterns. **MyTorch** includes several essential activation functions:
- **ReLU** - The Rectified Linear Unit, commonly used due to its simplicity and efficiency.
- **Leaky ReLU** - A variant of ReLU that allows a small gradient for negative inputs.
- **Sigmoid** - Maps values into the range [0, 1]. Often used in binary classification tasks.
- **Tanh** - Maps values between -1 and 1, used when zero-centered output is needed.
- **Softmax** - Applied to the output of multi-class classification tasks to transform logits into probabilities.
- **Step** - A step function, mainly used for binary decisions.

### Layers (`layer/`)
The building blocks of any neural network are its layers. **MyTorch** comes with essential layers for constructing feedforward and convolutional neural networks:
- **Linear Layer** - Fully connected layer where each input neuron connects to every output neuron.
- **Conv2D** - A 2D convolutional layer used to capture spatial dependencies in data, such as images.
- **MaxPool2D & AvgPool2D** - Pooling layers for down-sampling input by taking the maximum or average value in a local region.

### Loss Functions (`loss/`)
During training, the neural network uses a loss function to evaluate how far the prediction deviates from the actual target value:
- **Cross-Entropy (CE)** - For categorization tasks, ensuring higher penalties for incorrect predictions.
- **Mean Squared Error (MSE)** - Often used in regression tasks where reducing the difference between predicted and actual values is the goal.

### Optimizers (`optimizer/`)
Optimizers update the model’s weights by minimizing the loss function’s output. **MyTorch** comes with several well-known optimizers:
- **SGD (Stochastic Gradient Descent)** - A simple but effective optimizer.
- **Adam (Adaptive Moment Estimation)** - A popular choice that adjusts learning rates during training.
- **RMSProp** - Reduces the learning rate by dividing it by an exponentially decaying average of squared gradients.
- **Momentum** - A modification of SGD that accelerates learning in relevant directions by using a moving average of gradients.

### Utilities (`util/`)
Helper functions provide common utilities to simplify network design and training:
- **Data Loader** - Efficiently batches data and helps during the training process.
- **Flatten** - Utility for flattening multi-dimensional data for input into fully connected layers.
- **Initializer** - Functions for initializing weights and biases.

### Tensor (`tensor.py`)
**MyTorch** includes a custom implementation of tensor operations, mimicking core functionalities from well-known libraries like PyTorch or NumPy. This allows for element-wise operations, broadcasting, and more, making it easier to work with multidimensional arrays in your neural networks.

### Model (`model.py`)
This file acts as the backbone for running neural models in MyTorch. It defines the training loop, forward passes (to make predictions), and backward passes (to update weights based on gradients).

---

## Example Notebooks

### Simple Network (`simple_network.ipynb`)
This notebook guides you through building and training a basic feedforward neural network (or fully connected network) using **MyTorch**. 

**Architecture:**
- The network consists of several linear (fully connected) layers with activation functions like ReLU or Sigmoid between them.
- A typical example might involve a two-layer network with a hidden ReLU-activated layer and a Softmax output layer for multi-class classification.

**Training Process:**
- The notebook demonstrates how to load data using the DataLoader utility.
- You'll specify a loss function such as Cross-Entropy or MSE, depending on the task.
- Choose an optimizer (e.g., SGD, Adam) to adjust model parameters and minimize the loss during training.

### MNIST CNN (`MNIST-cnn.ipynb`)
This notebook shows how to implement a Convolutional Neural Network (CNN) using the MyTorch framework for classifying MNIST digits.

**Architecture:**
- The CNN model includes 2D convolutional layers to extract spatial features from inputs, followed by pooling layers to down-sample.
- Final layers are fully connected, mapping features to class probabilities.
- ReLU activations are used after convolution layers, and Softmax is applied to the output for classification.

**Dataset:**
- The MNIST dataset consists of 28x28 grayscale handwritten digits (0-9). The notebook walks through loading and preprocessing MNIST images using MyTorch’s DataLoader function.

**Training:**
- Using a loss function like Cross-Entropy and an optimizer like Adam, the CNN learns visual features useful for identifying digits.

### MNIST MLP (`MNIST-mlp.ipynb`)
This notebook explains how to implement a Multi-Layer Perceptron (MLP) for classifying MNIST data using MyTorch.

**Architecture:**
- The MLP is built using multiple fully connected (Linear) layers, with activation functions like Sigmoid or ReLU applied between them.
- Input images are flattened into 1D vectors before being passed through the network.
- A Softmax activation is used in the output layer for multi-class classification.

**Training:**
- Similar to the CNN, the MLP is trained using Cross-Entropy loss and optimizers like SGD or Adam, making updates through forward and backward passes.

---

## Getting Started
 
1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/MyTorch.git
   cd MyTorch
   ```

2. Explore different notebooks to understand how each model architecture works and train on example datasets.

3. Extend the library by adding new activation functions, layers, or optimizers to experiment with your own neural network architectures!
