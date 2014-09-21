class BaseLayer(object):

  def __init__(self, name=None):
    self.name = name
    self.num_bottom_blobs = 0
    self.num_top_blobs = 0

  def forward(self, bottom_blobs, top_blobs):
    """
    Run the layer forward, using the vals of the bottom blobs to compute the
    diffs of the top blobs.

    bottom_blobs - A list of Blobs
    top_blobs - A list of Blobs
    """
    raise NotImplementedError()

  def backward(self, bottom_blobs, top_blobs):
    """
    Run the layer backward, using the vals of the bottom blobs and the vals and
    diffs of the top blobs to compute the diffs of the bottom blobs.
    """
    raise NotImplementedError();

  def check_blobs(self, bottom_blobs, top_blobs):
    if len(bottom_blobs) != self.num_bottom_blobs:
      raise ValueError('Expected %d bottom blobs but got %d'
                       % (self.num_bottom_blobs, len(bottom_blobs)))

    if len(top_blobs) != self.num_top_blobs:
      raise ValueError('Expected %d bottom blobs but got %d'
                       % (self.num_top_blobs, len(top_blobs)))
