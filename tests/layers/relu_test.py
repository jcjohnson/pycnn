import unittest, random

import numpy as np

from pycnn import Blob
from pycnn.layers import ReLuLayer
from pycnn.layers.gradcheck import gradient_check_helper

class ReLuLayerTest(unittest.TestCase):
  def forward_test(self):
    shape = (2, 3)
    in_data = np.array([[-1, 2, 1], [0, -10, 5]])
    out_data = np.array([[0, 2, 1], [0, 0, 5]])
    bottom_blob = Blob(shape, vals=in_data)
    top_blob = Blob(shape)

    layer = ReLuLayer(shape)
    layer.forward([bottom_blob], [top_blob])
    self.assertTrue(np.all(top_blob.vals == out_data))

  def get_random_layer(self):
    dim = np.random.randint(2, 5)
    shape = tuple(np.random.randint(2, 3, size=dim))
    return ReLuLayer(shape)

  def bottom_gradient_numeric_test(self):
    passed = gradient_check_helper(self.get_random_layer)
    self.assertTrue(passed)
