from pycudnn import Tensor4D

"""
We attach additional methods to the Tensor4D class here.
"""

class _Tensor4DDataContext(object):
  def __init__(self, tensor):
    self.tensor = tensor

  def __enter__(self):
    self.tensor.fromGpu()
    return self.tensor._data

  def __exit__(self, type, value, traceback):
    self.tensor.toGpu()

def data(self):
  if not hasattr(self, '_data_context'):
    self._data_context = _Tensor4DDataContext(self)
  return self._data_context

Tensor4D.data = data
