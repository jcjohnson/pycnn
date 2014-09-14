import numpy as np
from pycudnn import *

in_data = Tensor4D(1, 1, 5, 5)
out_data = Tensor4D(1, 1, 5, 5)
filters = Tensor4D(1, 1, 3, 3)

in_data.data[0, 0, :, :] = np.eye(5)
print in_data.data
in_data.toGpu()

filters.data[0, 0, :, :] = [[0, 0.1, 0], [0.1, 0.6, 0.1], [0, 0.1, 0]]
filters.toGpu()
print filters.data

conv = Convolution(1, 1, 1, 1)
conv.forward(in_data, filters, out_data, False)
out_data.fromGpu()
print out_data.data

probs = Tensor4D(1, 1, 5, 5)
softmax = Softmax()
softmax.forward(out_data, probs)
probs.fromGpu()
print probs.data
