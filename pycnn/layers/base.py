
class BaseLayer(object):

  def __init__(self, name=None, input_names=None, output_names=None):
    self.name = name
    self.input_names = input_names
    self.output_names = output_names

  def get_bottom_shapes(self):
    """
    Return a list of tuples of the expected shapes for the inputs to this layer.
    The actual inputs will have one extra dimension corresponding to the batch
    size.
    """
    raise NotImplementedError()

  def get_top_shapes(self):
    """
    Return a list of tuples of the shapes of the outputs of this layer.
    The actual outputs will have one extra dimension corresponding to the batch
    size.

    A top shape of () indicates that this output is a loss, and will not have
    extra dimensions at runtime.
    """
    raise NotImplementedError()

  def forward(self, bottom_blobs, top_blobs):
    """
    Use the vals of the bottom blobs to compute the vals of the top blobs.

    bottom_blobs: A list of Blobs
    top_blobs: A list of Blobs
    """
    raise NotImplementedError()

  def backward(self, bottom_blobs, top_blobs):
    """
    Use the diffs of the top blobs to compute the diffs of the bottom blobs.

    bottom_blobs: A list of Blobs
    top_blobs: A list of Blobs
    """
    raise NotImplementedError()

