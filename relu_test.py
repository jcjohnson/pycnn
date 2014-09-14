import numpy as np
from pycudnn import *

vals = Tensor4D(1, 1, 5, 5)
diffs = Tensor4D(1, 1, 5, 5)

vals.data[:] = np.random.randn(*vals.data.shape)
vals.toGpu()
print vals.data

relu = ReLu()
relu.forward(vals, vals)
vals.fromGpu()
print vals.data

diffs.data[:] = np.random.randn(*diffs.data.shape)
diffs.toGpu()
print diffs.data
relu.backward(vals, diffs, vals, diffs)
diffs.fromGpu()
print diffs.data
