import unittest, random

import numpy as np

from pycnn import Blob
from pycnn.layers import L2LossLayer
from pycnn.layers.gradcheck import gradient_check_helper

class L2LossLayerTest(unittest.TestCase):
  def forward_test(self):
    x1 = np.array([[1, 2, 3], [2, 3, 4]], dtype=np.float32).T
    x1_blob = Blob(x1.shape, vals=x1)
    x2 = np.array([[2, 3, 4], [-2, 1, 5]], dtype=np.float32).T
    x2_blob = Blob(x2.shape, vals=x2)
    loss = Blob(())
    expected_loss = 24

    layer = L2LossLayer(3)
    layer.forward([x1_blob, x2_blob], [loss])
    self.assertTrue(np.abs(loss.vals - expected_loss) < 10e-5)

  def get_random_layer(self):
    dim = np.random.randint(2, 5)
    return L2LossLayer(dim)

  def x1_gradient_numeric_test(self):
    passed = gradient_check_helper(self.get_random_layer, param_name=0)
    self.assertTrue(passed)

  def x2_gradient_numeric_test(self):
    passed = gradient_check_helper(self.get_random_layer, param_name=1)
    self.assertTrue(passed)
