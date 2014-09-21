from pycnn.pycudnn import Tensor4D

class Blob(object):
  """
  A Blob holds data and derivatives.
  """
  def __init__(self, name, shape):
    """
    Create a new Blob.
    
    name - String.
    shape - (num, channels, height, width)
    """
    if len(shape) != 4:
      raise ValueError("Shape must be length 4")

    self.vals = Tensor4D(*shape)
    self.diffs = Tensor4D(*shape)
