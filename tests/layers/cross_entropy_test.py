import unittest, random 

import numpy as np

from pycnn import Blob
from pycnn.layers import CrossEntropyLossLayer
from pycnn.layers.gradcheck import gradient_check_helper

class CrossEntropyLossLayerTest(unittest.TestCase):
  def forward_test(self):
    q = Blob((2, 2), vals=np.array([[1.0, 0.0], [0.0, 1.0]]))
    p = Blob((2, 2), vals=np.array([[0.5, 0.5], [0.5, 0.5]]))
    loss = Blob(())

    layer = CrossEntropyLossLayer(2)
    layer.forward([q, p], [loss])
    
    diff = loss.vals + 2.0 * np.log(0.5)
    self.assertTrue(diff < 10e-2)

  def get_random_layer(self):
    dim = np.random.randint(2, 5)
    return CrossEntropyLossLayer(dim)

  def rand_fn(self, s):
    # Since cross-entropy loss is really over discrete probability distrubtions
    # the elements must be in the range [0, 1]. Numeric derivatives will also be
    # unstable as p gets close to zero.
    low = 0.1
    high = 1.0
    return low + (high - low) * np.random.random(s)

  def q_gradient_numeric_test(self):
    passed = gradient_check_helper(self.get_random_layer,
                                   param_name=0,
                                   rand_fn=self.rand_fn)
    self.assertTrue(passed)

  def p_gradient_numeric_test(self):
    passed = gradient_check_helper(self.get_random_layer,
                                   param_name=1,
                                   rand_fn=self.rand_fn)
    self.assertTrue(passed)

