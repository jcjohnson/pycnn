import numpy as np

class Blob(object):
  def __init__(self, shape, name=None, vals=None, diffs=None):
    """
    name: String name of blob
    shape: tuple of integers
    """
    self.name = name
    self.shape = shape

    if vals is not None:
      self.vals = vals
    else:
      self.vals = np.zeros(shape)

    if diffs is not None:
      self.diffs = diffs
    else:
      self.diffs = np.zeros(shape)

  @property
  def vals(self):
    return self._vals

  @vals.setter
  def vals(self, new_vals):
    if new_vals.shape != self.shape:
      raise ValueError('Cannot set vals; shape mismatch')
    self._vals = new_vals

  @property
  def diffs(self):
    return self._diffs

  @diffs.setter
  def diffs(self, new_diffs):
    if new_diffs.shape != self.shape:
      raise ValueError('Cannot set diffs; shape mismatch')
    self._diffs = new_diffs

