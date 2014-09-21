import random
import numpy as np
from pycnn import Blob

def numeric_derivative(layer, blob, bottom_blobs, top_blob):
  """
  Numerically compute the derivative of blob using the forward pass of the
  layer.
  """
  EPSILON = 0.001
  it = np.nditer(blob.diffs, op_flags=['readwrite'], flags=['multi_index'])
  start_vals = blob.vals.copy()
  while not it.finished:
    idx = it.multi_index

    blob.vals[idx] += EPSILON
    layer.forward(bottom_blobs, [top_blob])
    pos = top_blob.vals.copy()
    blob.vals = start_vals.copy()

    blob.vals[idx] -= EPSILON
    layer.forward(bottom_blobs, [top_blob])
    neg = top_blob.vals.copy()
    blob.vals = start_vals.copy()

    blob.diffs[idx] = np.sum((pos - neg) * top_blob.diffs)
    blob.diffs[idx] /= 2.0 * EPSILON

    it.iternext()

def gradient_check(layer, param_name=0, batch_size=None, rand_fn=None):
  """
  Compare numeric gradients with layer-computed gradients for either an input
  blob to the layer or an internal parameter of the layer. Returns the
  Frobenius norm of the difference between the two.
  
  If param_name is None, the input to the layer is checked. Otherwise,
  param_name is used to fetch a Blob from the layer.
  """
  if len(layer.get_top_shapes()) > 1:
    raise ValueError('Gradient checking is only implemented for layers with '
                     'one output')

  if rand_fn is None:
    rand_fn = lambda s: 10.0 * np.random.standard_normal(s)

  if batch_size is None:
    batch_size = random.randint(2, 10)

  bottom_shapes = layer.get_bottom_shapes()
  bottom_shapes = [s + (batch_size,) for s in bottom_shapes]
  bottom_blobs = [Blob(s, vals=rand_fn(s)) for s in bottom_shapes]
  
  for b in bottom_blobs:
    print b.vals

  top_shape = layer.get_top_shapes()[0]
  if len(top_shape) > 0:
    top_shape = top_shape + (batch_size,)
  top_diffs = rand_fn(top_shape)
  top_blob = Blob(top_shape, diffs=rand_fn(top_shape))

  if type(param_name) == int:
    blob = bottom_blobs[param_name]
  else:
    blob = getattr(layer, param_name)

  # This should only modify top_blob.vals
  layer.forward(bottom_blobs, [top_blob])

  # This should only modify blob.diffs and bottom_blob.diffs
  layer.backward(bottom_blobs, [top_blob])
  layer_diffs = blob.diffs.copy()

  numeric_derivative(layer, blob, bottom_blobs, top_blob)
  numeric_diffs = blob.diffs.copy()

  diff = layer_diffs - numeric_diffs
  # diff = np.max(np.abs(diff))
  diff = np.linalg.norm(diff[:])
  return diff

def gradient_check_helper(layer_factory, num_tests=10, threshold=10e-3,
                          **kwargs):
  """
  Run gradient checking num_tests times, and reports whether the difference
  was less than threshold for all tests.
  layer_factory: function that returns layers
  """
  print threshold
  for _ in xrange(num_tests):
    layer = layer_factory()
    diff = gradient_check(layer, **kwargs)
    if diff > threshold:
      return False
  return True

