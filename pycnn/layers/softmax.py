import numpy as np
from pycnn import Blob
from pycnn.layers import BaseLayer

class SoftmaxLayer(BaseLayer):
  def __init__(self, dim):
    self.dim = dim

  def get_bottom_shapes(self):
    return [(self.dim,)]

  def get_top_shapes(self):
    return [(self.dim,)]

  def forward(self, bottom_blobs, top_blobs):
    x = bottom_blobs[0].vals
    y = top_blobs[0].vals
    np.subtract(x, np.amax(x, 0), out=y)
    np.exp(y, out=y)
    np.divide(y, np.sum(y, axis=0), out=y)

  def backward(self, bottom_blobs, top_blobs):
    y = top_blobs[0].vals
    dy = top_blobs[0].diffs
    dx = bottom_blobs[0].diffs

    np.multiply(y, dy, out=dx)
    np.subtract(dy, np.sum(dx, axis=0), out=dx)
    np.multiply(y, dx, out=dx)
