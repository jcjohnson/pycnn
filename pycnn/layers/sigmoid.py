import numpy as np
from pycnn import Blob
from pycnn.layers import BaseLayer

class SigmoidLayer(BaseLayer):
  def __init__(self, shape):
    self.shape = shape

  def get_bottom_shapes(self):
    return [self.shape]

  def get_top_shapes(self):
    return [self.shape]

  def forward(self, bottom_blobs, top_blobs):
    x = bottom_blobs[0].vals
    y = top_blobs[0].vals
    # Compute y = 1 / (1 + exp(-x))
    np.multiply(x, -1.0, out=y)
    np.exp(y, out=y)
    np.add(y, 1, out=y)
    np.reciprocal(y, out=y)

  def backward(self, bottom_blobs, top_blobs):
    x = bottom_blobs[0].vals
    dx = bottom_blobs[0].diffs
    y = top_blobs[0].vals
    dy = top_blobs[0].diffs

    # Compute dx = exp(-x) / (1 + exp(-x))^2
    np.multiply(x, -1.0, out=dx)
    np.exp(dx, out=dx)
    np.multiply(dx, y, out=dx)
    np.multiply(dx, y, out=dx)

    # Multiply elementwise by dy
    np.multiply(dx, dy, out=dx)

