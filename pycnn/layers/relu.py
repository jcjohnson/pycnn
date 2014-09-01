import numpy as np
from pycnn import Blob
from pycnn.layers import BaseLayer

class ReLuLayer(BaseLayer):
  def __init__(self, shape):
    self.shape = shape

  def get_bottom_shapes(self):
    return [self.shape]

  def get_top_shapes(self):
    return [self.shape]

  def forward(self, bottom_blobs, top_blobs):
    np.maximum(bottom_blobs[0].vals, 0, out=top_blobs[0].vals)
    # top_blobs[0].vals = np.maximum(bottom_blobs[0].vals, 0)

  def backward(self, bottom_blobs, top_blobs):
    # bottom_blobs[0].diffs = (bottom_blobs[0].vals > 0) * top_blobs[0].diffs
    np.multiply(bottom_blobs[0].vals > 0, top_blobs[0].diffs,
                out=bottom_blobs[0].diffs)

