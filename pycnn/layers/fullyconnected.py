import numpy as np
from pycnn import Blob
from pycnn.layers import BaseLayer

class FullyConnectedLayer(BaseLayer):
  def __init__(self, input_dim, output_dim, weights=None, bias=None):
    self.input_dim = input_dim
    self.output_dim = output_dim
    
    weight_shape = (output_dim, input_dim)
    if weights is not None:
      if not hasattr(weights, 'vals') or not hasattr(weights, 'diffs'):
        raise ValueError('weights must be a Blob')
      if weights.shape != weight_shape:
        raise ValueError('weights do not have the correct shape')
      self.weights = weights
    else:
      self.weights = Blob(weight_shape)

    bias_shape = (output_dim, 1)
    if bias is not None:
      if not hasattr(bias, 'vals') or not hasattr(bias, 'diffs'):
        raise ValueError('bias must be a Blob')
      if bias.shape != bias_shape:
        raise ValueError('bias does not have the correct shape')
      self.bias = bias
    else:
      self.bias = Blob(bias_shape)
    
  def get_bottom_shapes(self):
    return [(self.input_dim,)]

  def get_top_shapes(self):
    return [(self.output_dim,)]

  def forward(self, bottom_blobs, top_blobs):
    top_blobs[0].vals = self.weights.vals.dot(bottom_blobs[0].vals)
    top_blobs[0].vals += self.bias.vals

  def backward(self, bottom_blobs, top_blobs):
    delta_y = top_blobs[0].diffs

    # Compute derivative of objective with respect to input
    bottom_blobs[0].diffs = np.dot(self.weights.vals.T, delta_y)

    # Compute derivative of objective with respect to parameters
    self.weights.diffs = np.einsum('ik,jk', delta_y, bottom_blobs[0].vals)
    self.bias.diffs = np.sum(delta_y, axis=1, keepdims=True)
