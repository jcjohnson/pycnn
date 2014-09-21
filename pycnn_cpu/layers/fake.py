from pycnn.layers import BaseLayer

class FakeLayer(BaseLayer):
  """
  A fake layer that doesn't do anything but parrot input and output shapes.
  Used in tests for Net.
  """
  def __init__(self, input_shapes=None, output_shapes=None, **kwargs):
    super(FakeLayer, self).__init__(**kwargs)
    self.input_shapes = input_shapes
    self.output_shapes = output_shapes

  def get_bottom_shapes(self):
    return self.input_shapes

  def get_top_shapes(self):
    return self.output_shapes
