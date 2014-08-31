import unittest
import numpy as np
from pycnn import Blob

class BlobTest(unittest.TestCase):
  def init_test(self):
    shape = (4, 5)
    blob = Blob(shape)

    self.assertTrue(np.all(blob.vals == np.zeros(shape)))
    self.assertTrue(np.all(blob.diffs == np.zeros(shape)))

  def init_vals_test(self):
    shape = (2, 3)
    vals = np.array([[1, 2, 3], [4, 5, 6]])

    blob = Blob(shape, vals=vals)
    self.assertTrue(np.all(blob.vals == vals))
    self.assertTrue(np.all(blob.diffs == np.zeros(shape)))

  def init_diffs_test(self):
    shape = (3, 2)
    diffs = np.array([[1, 2], [3, 4], [5, 6]])

    blob = Blob(shape, diffs=diffs)
    self.assertTrue(np.all(blob.vals == np.zeros(shape)))
    self.assertTrue(np.all(blob.diffs == diffs))

  def init_bad_vals_test(self):
    shape = (2, 3)
    vals = np.array([[1, 2], [3, 4]])
    with self.assertRaises(ValueError):
      blob = Blob(shape, vals=vals)

  def init_bad_diffs_test(self):
    shape = (3, 3)
    diffs = np.array([[1, 2], [3, 4]])
    with self.assertRaises(ValueError):
      blob = Blob(shape, diffs=diffs)

  def set_vals_test(self):
    shape = (2, 3)
    zero = np.zeros(shape)
    vals = np.array([[1, 2, 3], [4, 5, 6]])

    blob = Blob(shape)
    self.assertTrue(np.all(blob.vals == zero))
    self.assertTrue(np.all(blob.diffs == zero))

    blob.vals = vals
    self.assertTrue(np.all(blob.vals == vals))
    self.assertTrue(np.all(blob.diffs == zero))

  def set_bad_vals_test(self):
    shape = (5, 5)
    blob = Blob(shape)
    with self.assertRaises(ValueError):
      blob.vals = np.array([[1, 2], [4, 5]])
