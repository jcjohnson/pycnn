import unittest, random

import numpy as np

from pycnn import Blob
from pycnn.layers import SigmoidLayer
from pycnn.layers.gradcheck import gradient_check_helper

class SigmoidLayerTest(unittest.TestCase):
  def forward_test(self):
    shape = (2, 3)
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    y = 1.0 / (1.0 + np.exp(-x))
    bottom_blob = Blob(shape, vals=x)
    top_blob = Blob(shape)

    layer = SigmoidLayer(shape)
    layer.forward([bottom_blob], [top_blob])
    diff = np.linalg.norm((y - top_blob.vals)[:])
    self.assertTrue(diff < 10e-2)

  def get_random_layer(self):
    dim = np.random.randint(2, 5)
    shape = tuple(np.random.randint(2, 3, size=dim))
    return SigmoidLayer(shape)

  def bottom_gradient_numeric_test(self):
    passed = gradient_check_helper(self.get_random_layer)
    self.assertTrue(passed)
