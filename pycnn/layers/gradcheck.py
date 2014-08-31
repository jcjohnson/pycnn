import random
import numpy as np
from pycnn import Blob

def numeric_derivative(layer, blob, bottom_blob, top_blob):
  """
  Numerically compute the derivative of blob using the forward pass of the
  layer.
  """
  EPSILON = 0.01
  it = np.nditer(blob.diffs, op_flags=['readwrite'], flags=['multi_index'])
  start_vals = blob.vals.copy()
  while not it.finished:
    idx = it.multi_index

    blob.vals[idx] += EPSILON
    layer.forward([bottom_blob], [top_blob])
    pos = top_blob.vals.copy()
    blob.vals = start_vals.copy()

    blob.vals[idx] -= EPSILON
    layer.forward([bottom_blob], [top_blob])
    neg = top_blob.vals.copy()
    blob.vals = start_vals.copy()

    blob.diffs[idx] = np.sum((pos - neg) * top_blob.diffs)
    blob.diffs[idx] /= 2.0 * EPSILON

    it.iternext()

def gradient_check(layer, param_name=None, batch_size=None):
  """
  Compare numeric gradients with layer-computed gradients for either an input
  blob to the layer or an internal parameter of the layer. Returns the
  Frobenius norm of the difference between the two.
  
  If param_name is None, the input to the layer is checked. Otherwise,
  param_name is used to fetch a Blob from the layer.
  """
  if len(layer.get_bottom_shapes()) > 1 or len(layer.get_top_shapes()) > 1:
    raise ValueError('Gradient checking is only implemented for layers with '
                     'one input and one output')

  if batch_size is None:
    batch_size = random.randint(2, 10)

  bottom_shape = layer.get_bottom_shapes()[0] + (batch_size,)
  bottom_blob = Blob(bottom_shape, vals=np.random.randn(*bottom_shape)) 
  top_shape = layer.get_top_shapes()[0] + (batch_size,)
  top_blob = Blob(top_shape, diffs=np.random.randn(*top_shape))
  
  if param_name is None:
    blob = bottom_blob
  else:
    blob = getattr(layer, param_name)

  # This should only modify top_blob.vals
  layer.forward([bottom_blob], [top_blob])

  # This should only modify blob.diffs and bottom_blob.diffs
  layer.backward([bottom_blob], [top_blob])
  layer_diffs = blob.diffs.copy()

  numeric_derivative(layer, blob, bottom_blob, top_blob)
  numeric_diffs = blob.diffs.copy()

  return np.linalg.norm(layer_diffs - numeric_diffs, ord='fro')

def gradient_check_helper(layer_factory, param_name=None, num_tests=10,
                               threshold=0.01):
  """
  Run gradient checking num_tests times, and reports whether the difference
  was less than threshold for all tests.
  layer_factory: function that returns layers
  """
  for _ in xrange(num_tests):
    layer = layer_factory()
    diff = gradient_check(layer, param_name=param_name)
    if diff > threshold:
      return False
  return True

