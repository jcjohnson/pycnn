import numpy as np

from pycnn import Blob
from pycnn.layers import BaseLayer

class L2LossLayer(BaseLayer):
  def __init__(self, dim, **kwargs):
    super(L2LossLayer, self).__init__(**kwargs)
    self.dim = dim

  def get_bottom_shapes(self):
    return [(self.dim,), (self.dim,)]

  def get_top_shapes(self):
    return [()]

  def forward(self, bottom_blobs, top_blobs):
    x1 = bottom_blobs[0].vals
    x2 = bottom_blobs[1].vals
    y = top_blobs[0].vals
    np.sum(np.square(x1 - x2), out=y)

  def backward(self, bottom_blobs, top_blobs):
    d = top_blobs[0].diffs
    x1, dx1 = bottom_blobs[0].vals, bottom_blobs[0].diffs
    x2, dx2 = bottom_blobs[1].vals, bottom_blobs[1].diffs
    np.subtract(x1, x2, out=dx1)
    np.subtract(x2, x1, out=dx2)
    dx1 *= d * 2.0
    dx2 *= d * 2.0

