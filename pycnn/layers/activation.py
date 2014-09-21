from pycnn.layers import BaseLayer
from pycnn import pycudnn

class ReLuLayer(BaseLayer):
  def __init__(self, name):
    self.name = name
    self.num_bottom_blobs = 1
    self.num_top_blobs = 1
    self.relu = pycudnn.ReLu()

  def forward(self, bottom_blobs, top_blobs):
    self.check_blobs(bottom_blobs, top_blobs)
    self.relu.forward(bottom_blobs[0].vals, top_blobs[0].vals)

  def backward(self, bottom_blobs, top_blobs):
    self.check_blobs(bottom_blobs, top_blobs)
    bottom = bottom_blobs[0]
    top = top_blobs[0]
    self.relu.backward(top.vals, top.diffs, bottom.vals, bottom.diffs)


class SigmoidLayer(BaseLayer):
  def __init__(self, name):
    self.name = name
    self.num_bottom_blobs = 1
    self.num_top_blobs = 1
    self.sigmoid = pycudnn.Sigmoid()

  def forward(self, bottom_blobs, top_blobs):
    self.check_blobs(bottom_blobs, top_blobs)
    self.sigmoid.forward(bottom_blobs[0].vals, top_blobs[0].diffs)

  def backward(self, bottom_blobs, top_blobs):
    self.check_blobs(bottom_blobs, top_blobs)
    bottom = bottom_blobs[0]
    top = top_blobs[0]
    self.sigmoid.backward(top.vals, top.diffs, bottom.vals, bottom.diffs)


class TanhLayer(BaseLayer):
  def __init__(self, name):
    self.name = name
    self.num_bottom_blobs = 1
    self.num_top_blobs = 1
    self.tanh = pycudnn.Tanh()

  def forward(self, bottom_blobs, top_blobs):
    self.check_blobs(bottom_blobs, top_blobs)
    self.tanh.forward(bottom_blobs[0].vals, top_blobs[0].diffs)

  def backward(self, bottom_blobs, top_blobs):
    self.check_blobs(bottom_blobs, top_blobs)
    bottom = bottom_blobs[0]
    top = top_blobs[0]
    self.tanh.backward(top.vals, top.diffs, bottom.vals, bottom.diffs)
