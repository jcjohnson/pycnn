import struct, sys, argparse
import numpy as np
import matplotlib.pyplot as plt

def read_image_file(f):
  """
  Read a file of images in MNIST format.

  Inputs:
  f - A file opened for binary reading

  Outputs:
  images - H x W x N numpy array of uint8
  """
  magic, num, height, width = struct.unpack('>iiii', f.read(16))
  if magic != 2051:
    raise ValueError('Magic number is not correct')
  images = np.zeros((height, width, num), dtype=np.uint8)
  row_string = '>' + width * 'B'
  row_size = struct.calcsize(row_string)
  for n in xrange(num):
    for i in xrange(height):
      row = struct.unpack(row_string, f.read(row_size))
      images[i, :, n] = row
  return images


def read_label_file(f):
  """
  Read a file of labels in MNIST format

  Inputs:
  f - A file opened for binary reading

  Outputs:
  labels - uint8 numpy vector of labels
  """
  magic_number, num_items = struct.unpack('>ii', f.read(8))
  if magic_number != 2049:
    raise ValueError('Magic number is not correct')
  labels = np.zeros((num_items), dtype=np.uint8)
  for i in xrange(num_items):
    label, = struct.unpack('>B', f.read(1))
    labels[i] = label
  return labels


def show_mnist_data(images, labels, grid_size=4):
  """
  Show MNIST images and labels

  Inputs:
  images - H x W x N numpy array of uint8 image data
  labels - N length numpy array of uint8 labels
  grid_size - Images are shown grid_size * grid_size at a time
  """
  idx = 0
  while idx < labels.shape[0]:
    for i in xrange(grid_size ** 2):
      plt.subplot(grid_size, grid_size, i)
      img_plt = plt.imshow(images[:, :, idx])
      img_plt.set_interpolation('nearest')
      img_plt.set_cmap('gray')
      plt.title(labels[idx])
      frame = plt.gca()
      frame.axes.get_xaxis().set_ticks([])
      frame.axes.get_yaxis().set_ticks([])
      idx += 1
    plt.show()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--input_images',
                      help='Input file of images in MNIST format')
  parser.add_argument('--input_labels',
                      help='Input file of labels in MNIST format')
  parser.add_argument('--output_images',
                      help='Output file for images in .npy format')
  parser.add_argument('--output_labels',
                      help='Output file for labels in .npy format')
  parser.add_argument('--show_images', action='store_true',
                      help='Show some of the loaded images and labels')
  args = parser.parse_args()

  if args.input_images:
    with open(args.input_images, 'rb') as f:
      images = read_image_file(f)
    if args.output_images:
      with open(args.output_images, 'wb') as f:
        np.save(f, images)

  if args.input_labels:
    with open(args.input_labels, 'rb') as f:
      labels = read_label_file(f)
    if args.output_labels:
      with open(args.output_labels, 'wb') as f:
        np.save(f, labels)
  
  if args.input_images and args.input_labels and args.show_images:
    show_mnist_data(images, labels)

