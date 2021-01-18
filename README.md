# Neural Networks Compression

[![Python](https://img.shields.io/badge/python-3.8-informational)](https://docs.python.org/3/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

Implementation of neural network compression from [Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding](https://arxiv.org/abs/1510.00149) and [Learning both Weights and Connections for Efficient Neural Networks](https://arxiv.org/abs/1506.02626).

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [Abstract](#abstract)
* [Compression](#compression)
* [References](#references)
* [Contributing](#contributing)


### Abstract

Neural networks are both computationally and memory intensive.
This is why it is difficult to deploy then on embedded systems with limited hardware resources.

The aim of this project is to compress a neural network with pruning and quantization without accuracy degradation.

The experiments are executed on the [MNIST classification problem](https://en.wikipedia.org/wiki/MNIST_database), with the following neural networks: `LeNet300-100` and `LeNet5`.  

### Compression
### References

- [Learning both Weights and Connections for Efficient Neural Networks - Song Han, Jeff Pool, John Tran, William J. Dally](https://arxiv.org/abs/1506.02626). 

- [Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding - Song Han, Huizi Mao, John Tran, J.Dally](https://arxiv.org/abs/1510.00149).

### Contributing

To contribute, please open a PR based on the `master` branch.

We follow the [Conventional Commits specification](https://www.conventionalcommits.org/en/v1.0.0/).
