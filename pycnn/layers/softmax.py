from pycnn.layers import BaseLayer
from pycnn import pycudnn

class SoftmaxLayer(BaseLayer):
  def __init__(self, name):
    self.name = name
    self.num_bottom_blobs = 1
    self.num_top_blobs = 1
    self.softmax = pycudnn.Softmax()

  def forward(self, bottom_blobs, top_blobs):
    self.check_blobs(bottom_blobs, top_blobs)
    self.softmax.forward(bottom_blobs[0].vals, top_blobs[0].vals)

  def backward(self, bottom_blobs, top_blobs):
    self.check_blobs(bottom_blobs, top_blobs)
    top = top_blobs[0]
    self.softmax.backward(top.vals, top.diffs, bottom_blobs[0].diffs)
