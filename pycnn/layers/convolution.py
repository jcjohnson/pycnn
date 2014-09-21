from pycnn.layers import BaseLayer
from pycnn import pycudnn

# TODO: Make this work with a bias term
class ConvolutionalLayer(BaseLayer):
  def __init__(self, name, pad_x=0, pad_y=0, stride_x=1, stride_y=1):
    self.name = name
    self.num_bottom_blobs = 2
    self.num_top_blobs = 1
    self.conv = pycudnn.Convolution(pad_x, pad_y, stride_x, stride_y)

  def forward(self, bottom_blobs, top_blobs):
    """
    This expects [data, filters] as bottom blobs
    """
    self.check_blobs(bottom_blobs, top_blobs)
    data = bottom_blobs[0]
    filters = bottom_blobs[1]
    top = top_blobs[0]
    self.conv.forward(data.vals, filters.vals, top.vals)

  def backward(self, bottom_blobs, top_blobs):
    self.check_blobs(bottom_blobs, top_blobs)
    data = bottom_blobs[0]
    filters = bottom_blobs[1]
    top = top_blobs[0]
    self.conv.backward(top.vals, top.diffs, filters.vals, filters.diffs,
                       data.diffs)

