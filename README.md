pycnn
=====

Convolutional Neural Networks in Python.

For my own education, I wanted to implement a CNN framework from scratch in Python. Right now this is half-baked and not really useable for anything, but this served as a starting point for the CNN that we use in the assignments for [CS 231n](http://vision.stanford.edu/teaching/cs231n/).

When NVIDIA released [cuDNN](https://developer.nvidia.com/cuDNN) I was also curious to see if I could expose the CUDNN functionality to Python, with the goal of eventually implementing a GPU / CPU hybrid CNN implementation in Python. As a proof of concept, I implemented a Tensor4D object in C++ that manages CPU and GPU pointers to data, and several Layer objects that can operate on Tensor4D objects by calling into cuDNN functions. The CPU data of the Tensor4D objects are wrapped in numpy arrays, and exposed as Python classes using Boost-python. The implementation of the Tensor4D object can be found in the file `src/pycudnn.cpp` and example usage in Python can be found in the file `examples/pycudnn_example.py`. 

One day I want to revisit this and make it all actually work, but for now it will have to live a cursed, half-baked life.
