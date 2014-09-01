import unittest, random

import numpy as np

from pycnn import Blob
from pycnn.layers import SoftmaxLayer
from pycnn.layers.gradcheck import gradient_check_helper

class SoftmaxLayerTest(unittest.TestCase):
  def forward_test(self):
    dim = 3
    x = np.array([[1,4],[2,5],[3,6]])
    e = np.exp(1)
    d1 = e + e ** 2 + e ** 3
    d2 = e ** 4 + e ** 5 + e ** 6
    y = np.array([[e / d1, e ** 4 / d2],
                  [e ** 2 / d1, e ** 5 / d2],
                  [e ** 3 / d1, e ** 6 / d2]])
    bottom_blob = Blob((3, 2), vals=x)
    top_blob = Blob((3, 2))

    layer = SoftmaxLayer(dim)

    layer.forward([bottom_blob], [top_blob])
    diff = top_blob.vals - y
    self.assertTrue(np.linalg.norm(diff[:]) < 10e-2)
