import numpy as np
from pycnn.pycudnn import *

t = Tensor4D(1, 2, 3, 4)
with t.data() as d:
  d[:] = np.random.randn(*d.shape)

print t._data
relu = ReLu()
relu.forward(t, t)

with t.data() as d:
  print d
  d *= -1

print t._data
relu.forward(t, t)
with t.data() as d:
  print d
