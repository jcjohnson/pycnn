from pycnn.layers import BaseLayer
from pycnn import pycudnn

class PoolingLayer(BaseLayer):
  def __init__(self, name, pool_type='max', width=2, height=2,
               stride_x=2, stride_y=2):
    self.name = name
    self.num_bottom_blobs = 1
    self.num_top_blobs = 1
    if pool_type == 'max':
      pool_class = pycudnn.MaxPooling
    elif pool_type == 'avg':
      pool_class= pycudnn.AvgPooling
    else:
      raise ValueError('Unrecognized pool_type "%s"')
    self.pool = pool_class(width, height, stride_x, stride_y)

  def forward(self, bottom_blobs, top_blobs):
    self.check_blobs(bottom_blobs, top_blobs)
    self.pool.forward(bottom_blobs[0].vals, top_blobs[0].vals)

  def backward(self, bottom_blobs, top_blobs):
    self.check_blobs(bottom_blobs, top_blobs)
    bottom = bottom_blobs[0]
    top = top_blobs[0]
    self.pool.backward(top.vals, top.diffs, bottom.vals, bottom.diffs)
