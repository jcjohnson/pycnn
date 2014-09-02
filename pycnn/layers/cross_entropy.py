import numpy as np

from pycnn import Blob
from pycnn.layers import BaseLayer

class CrossEntropyLossLayer(BaseLayer):
  def __init__(self, dim):
    self.dim = dim

  def get_bottom_shapes(self):
    return [(self.dim,), (self.dim,)]

  def get_top_shapes(self):
    return [()]

  def forward(self, bottom_blobs, top_blobs):
    q = bottom_blobs[0].vals
    p = bottom_blobs[1].vals
    loss = -np.sum(q * np.log(p))
    top_blobs[0].vals = loss

  def backward(self, bottom_blobs, top_blobs):
    d = top_blobs[0].diffs
    q, dq = bottom_blobs[0].vals, bottom_blobs[0].diffs
    p, dp = bottom_blobs[1].vals, bottom_blobs[1].diffs
    np.log(p, out=dq)
    dq *= -d
    np.divide(q, p, out=dp)
    dp *= -d


