import unittest

import numpy as np

from pycnn import Net, Blob
from pycnn.layers import FakeLayer, FullyConnectedLayer, ReLuLayer

class NetTest(unittest.TestCase):
  def layer_no_name_test(self):
    with self.assertRaises(ValueError):
      net = Net([FakeLayer()], batch_size=2)
    
  def inconsistent_blob_size_test(self):
    layer1 = FakeLayer(name='layer1',
                       input_shapes=[(2, 3)], input_names=['data'],
                       output_shapes=[(2, 2)], output_names=['hidden'])
    layer2 = FakeLayer(name='layer2',
                       input_shapes=[(3, 3)], input_names=['hidden'],
                       output_shapes=[()], output_names=['loss'])

    with self.assertRaises(ValueError):
      net = Net([layer1, layer2], batch_size=2)
  
  def init_net_test(self):
    layer1 = FakeLayer(name='layer1',
                       input_shapes=[(2, 2)], input_names=['data'],
                       output_shapes=[(4, 4)], output_names=['hidden'])
    layer2 = FakeLayer(name='layer2',
                       input_shapes=[(4, 4)], input_names=['hidden'],
                       output_shapes=[()], output_names=['loss'])
    net = Net([layer1, layer2], batch_size=7)
    self.assertEquals(len(net.blobs), 3)
    blob_to_shape = {name: blob.shape for name, blob in net.blobs.iteritems()}
    expected_blob_to_shape = {
      'data': (2, 2, 7),
      'hidden': (4, 4, 7),
      'loss': (),
    }
    self.assertEquals(blob_to_shape, expected_blob_to_shape)
    self.assertEquals(net.input_names, {'data'})
    self.assertEquals(net.output_names, {'loss'})

  def layers_out_of_order_test(self):
    layer1 = FakeLayer(name='layer1',
                       input_shapes=[(2, 2)], input_names=['data'],
                       output_shapes=[(4, 4)], output_names=['hidden'])
    layer2 = FakeLayer(name='layer2',
                       input_shapes=[(4, 4)], input_names=['hidden'],
                       output_shapes=[()], output_names=['loss'])
    with self.assertRaises(ValueError):
      net = Net([layer2, layer1], batch_size=7)

  def forward_wrong_inputs_test(self):
    layer1 = FakeLayer(name='layer1',
                       input_shapes=[(2, 2)], input_names=['data'],
                       output_shapes=[(4, 4)], output_names=['hidden'])
    layer2 = FakeLayer(name='layer2',
                       input_shapes=[(4, 4)], input_names=['hidden'],
                       output_shapes=[()], output_names=['loss'])
    net = Net([layer1, layer2], batch_size=7)

    with self.assertRaises(ValueError):
      net.forward(data=10, other_data=20)

    with self.assertRaises(ValueError):
      net.forward(wrong_name=10)

  def forward_test(self):
    in_dim = 3
    out_dim = 2
    w = np.array([[1, -1, 0], [0, 1, -1]], dtype=np.float32)
    b = np.array([[2], [3]], dtype=np.float32)
    weights = Blob((out_dim, in_dim), vals=w)
    bias = Blob((out_dim, 1), vals=b)
    layer1 = FullyConnectedLayer(in_dim, out_dim, name='fc',
                                 input_names=['data'], output_names=['hidden'],
                                 weights=weights, bias=bias)
    layer2 = ReLuLayer((out_dim,), name='relu',
                       input_names=['hidden'], output_names=['output'])
    net = Net([layer1, layer2], batch_size=2)
    inputs = np.array([[1, 2, 3], [2, 5, 2]]).T
    expected_output = np.array([[1, 2], [0, 6]]).T
    outputs = net.forward(data=inputs)
    
    self.assertEqual(1, len(outputs))
    self.assertTrue('output' in outputs)
    self.assertTrue(np.all(outputs['output'] == expected_output))
