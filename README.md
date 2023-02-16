# Vectorized Neural Network

<div style='float:left;'>
  <img href='' src='https://img.shields.io/badge/Maintained%3F-Yes-brightgreen.svg' style='margin-right:10px;'>
  <img href='https://www.python.org/' src='https://img.shields.io/badge/Made with-Python-blue' style='margin-right:10px;'>
  <img href='https://choosealicense.com/licenses/mit/' src='https://img.shields.io/badge/LICENSE-MIT-green' style='margin-right:10px'>
  <img href='https://github.com/MatthewJansen/Vectorized-Neural-Network' src='https://img.shields.io/github/stars/MatthewJansen/Vectorized-Neural-Network?style=social' style='margin-right:10px'>
  <img href='https://github.com/MatthewJansen/Vectorized-Neural-Network/fork' src='https://img.shields.io/github/forks/MatthewJansen/Vectorized-Neural-Network?style=social' style='margin-right:10px'>
  <img href='https://github.com/MatthewJansen/Vectorized-Neural-Network' src='https://img.shields.io/github/watchers/MatthewJansen/Vectorized-Neural-Network?style=social' style='margin-right:10px'>
  <img href='https://twitter.com/intent/follow?screen_name=matthewjansen_' src='https://img.shields.io/twitter/follow/matthewjansen_?style=social' style='margin-right:10px;'>
</div>

<br><br>

<center>
  <img src='https://i.postimg.cc/W1wr7ctR/Neural-Net.png' alt='NeuralNetwork'>
</center>

## Overview

Python implementation for a fully connected, vectorized neural network which I wrote for the purpose of displaying and challenging my understanding of neural networks.

Neural networks computational graph structure which consists of layers of nodes called ’neurons’ which have biases and connecting weighted edges called ’synapses’.

**The process of training a neural network consists of three steps:**

  1. Forward Propagation:
  Here the inputs are fed to the network in order to obtain an output similar to the expected output from the dataset.
  2. Backpropagation:
  This step covers the process through which the relevance of each neuron’s activation values is computed with respect to the cost of the neural network’s output.
  3. Update Network:
  Finally, based on the results from backpropagating the output, the new weights and biases of the neural network are computed and updated.

> NOTE: Regression and Classification is possible with the current implementation of the Neural Network. However, it should be noted that all classification tasks are treated as ordinal regression problems.
## Requirements

- **Recommended python version:** Python 3.8.10 64-bit
- Unzip and to add the raw MNIST data to the same directory as the code before using the project's notebook and code.

[Note: This section will be updated in due course.]

## Project Structure

```
.
├── data
│   └── mnist_data.zip
├── src
│   ├── ActivationFunctions.py
│   ├── ClassificationMetrics.py
│   ├── DataHandler.py
│   ├── MNIST_Classification.ipynb
│   ├── NetworkConfigHandler.py
│   ├── NeuralNetwork.py
│   └── PreProcessor.py
├── LICENSE
├── NeuralNetwork.pdf
├── README.md
```

- **data/mnist_data.zip** | zip file containing mnist train and test data
- **src/ActivationFunctions.py** | class for neural network activation functions
- **src/ClassificationMetrics.py** | class for classification metrics
- **src/DataHandler.py** | dataset handler for loading data
- **src/MNIST_Classification.ipynb** | notebook for displaying project usage
- **src/NetworkConfigHandler.py** | class for loading and saving neural network data
- **src/NeuralNetwork.py** | neural network implementation
- **src/PreProcessor.py** | data preprocessor implementation
- **NeuralNetwork.pdf** | document for project explanation

## To Be Added

- Loss Function API [Status: Incomplete]
- Softmax compatibility [Status: Incomplete]
- Optimizer API [Status: Incomplete]
- Batching feature to DataHandler [Status: Incomplete]
- (Maybe) Callbacks API [Status: Incomplete]


## Usage
`See src/MNIST_Classification.ipynb (will update this section in due course...)`

## License
This project is licensed under the terms and conditions of the [MIT license](https://choosealicense.com/licenses/mit/).