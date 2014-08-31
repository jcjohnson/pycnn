import unittest, random

import numpy as np

from pycnn import Blob
from pycnn.layers import FullyConnectedLayer
from pycnn.layers.gradcheck import gradient_check_helper

class FullyConnectedLayerTest(unittest.TestCase):
  def simple_forward_test(self):
    in_dim = 2
    out_dim = 3
    weights = np.array([[1,2],[3,4],[5,6]])
    weights_blob = Blob(weights.shape, vals=weights)
    bias = np.array([[1],[2],[3]])
    bias_blob = Blob(bias.shape, vals=bias)

    input_data = np.array([[10], [20]])
    output_data = np.array([[51], [112], [173]])
    input_blob = Blob((in_dim, 1), vals=input_data)
    output_blob = Blob((out_dim, 1))

    fcl = FullyConnectedLayer(in_dim, out_dim, weights=weights_blob,
                                               bias=bias_blob)
    fcl.forward([input_blob], [output_blob])

    self.assertTrue((output_blob.vals == output_data).all())
  
  def get_random_layer(self):
    input_dim = random.randint(2, 10)
    output_dim = random.randint(2, 10)
    weights = np.random.randn(output_dim, input_dim)
    weights_blob = Blob((output_dim, input_dim), vals=weights)
    bias = np.random.randn(output_dim, 1)
    bias_blob = Blob((output_dim, 1), vals=bias)
    return FullyConnectedLayer(input_dim, output_dim, weights=weights_blob,
                               bias=bias_blob)

  def bottom_gradient_numeric_test(self):
    gradient_check_helper(self.get_random_layer)

  def weights_gradient_numeric_test(self):
    gradient_check_helper(self.get_random_layer, param_name='weights')

  def bias_gradient_numeric_test(self):
    gradient_check_helper(self.get_random_layer, param_name='bias')
