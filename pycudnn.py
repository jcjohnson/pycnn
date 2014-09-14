import numpy as np
from pycudnn import *

t1 = Tensor4D(2,2,2,2)
t2 = Tensor4D(2,2,2,2)

t1.data[:] = np.random.randint(0, 5, size=t1.data.shape)
print t1.data
t1.toGpu()

softmax = Softmax()
softmax.forward(t1, t2)
t2.fromGpu()
print t2.data
print np.sum(t2.data[0, :, :, :])
print np.sum(t2.data[1, :, :, :])
