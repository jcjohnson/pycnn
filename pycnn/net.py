import numpy as np
from pycnn import Blob

class Net(object):
  def __init__(self, layers, batch_size):
    """
    Create a new Net.

    layers: An iterable of Layer objects. Must be sorted in topological order.
    batch_size: Integer.
    """
    for layer in layers:
      if layer.name is None:
        raise ValueError('All layers in a net must be named')
    self.layers = layers
    self.blobs = {}
    self.batch_size = batch_size
    self.input_names = set()
    self.output_names = set()
    self._init_blobs()
    self._check_layers_sorted()

  def _check_layers_sorted(self):
    seen_inputs = set()
    seen_outputs = set()
    for layer in self.layers:
      for blob_name in layer.input_names:
        seen_inputs.add(blob_name)
      for blob_name in layer.output_names:
        if blob_name in seen_inputs and blob_name not in seen_outputs:
          raise ValueError('Layers are not sorted; blob "%s" first seen as '
                           'input and later seen as output' % blob_name)
        seen_outputs.add(blob_name)

    self.input_names = seen_inputs - seen_outputs
    self.output_names = seen_outputs - seen_inputs

  def _init_blobs(self):
    def helper(shape, name):
      if len(shape) > 0:
        shape = shape + (self.batch_size,)
      if name not in self.blobs:
        self.blobs[name] = Blob(shape, name=name)
      elif name in self.blobs and shape != self.blobs[name].shape:
        raise ValueError('Blob "%s" referenced with inconsistent shapes %s, %s'
                         % (name, shape, self.blobs[name].shape))

    for layer in self.layers:
      for shape, name in zip(layer.get_bottom_shapes(), layer.input_names):
        helper(shape, name)
      for shape, name in zip(layer.get_top_shapes(), layer.output_names):
        helper(shape, name)
  
  def forward(self, **input_vals):
    if set(input_vals.keys()) != self.input_names:
      raise ValueError('Called forward with incorrect inputs')
    for name, vals in input_vals.iteritems():
      self.blobs[name].vals = vals
    for layer in self.layers:
      input_blobs = [self.blobs[name] for name in layer.input_names]
      output_blobs = [self.blobs[name] for name in layer.output_names]
      layer.forward(input_blobs, output_blobs)
    return {name: self.blobs[name].vals for name in self.output_names}

  def backward(self):
    # TODO: Implement this
    pass
