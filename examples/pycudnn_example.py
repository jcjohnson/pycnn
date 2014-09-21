import numpy as np
from pycnn.pycudnn import *

in_data = Tensor4D(1, 1, 5, 5) # num, channels, width, height
out_data = Tensor4D(1, 1, 5, 5) 
filters = Tensor4D(1, 1, 3, 3) # output channels, input channels, width, height

in_data.data[0, 0, :, :] = -1 * np.eye(5)
print in_data.data
in_data.toGpu()

filters.data[0, 0, :, :] = [[0, 0.1, 0], [0.1, 0.6, 0.1], [0, 0.1, 0]]
filters.toGpu()
print filters.data

conv = Convolution(1, 1, 1, 1) # pad_x, pad_y, stride_x, stride_y
conv.forward(in_data, filters, out_data, False)
out_data.fromGpu()
print out_data.data

relu = ReLu()
relu_data = Tensor4D(1, 1, 5, 5)
relu.forward(out_data, relu_data)
relu_data.fromGpu()
print relu_data.data

probs = Tensor4D(1, 1, 5, 5)
softmax = Softmax()
softmax.forward(out_data, probs)
probs.fromGpu()
print probs.data

