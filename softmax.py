import sys
import numpy as np
import matplotlib.pyplot as plt

def train_softmax(x, y, x_test, y_test):
  K = np.unique(y).shape[0]
  M, N = x.shape
  theta = np.random.rand(K, M)
  alpha = 0.002

  objective_values = []
  train_errors = []
  test_errors = []

  for t in xrange(101):
    obj, dTheta, thetaX = objective(x, y, theta, returnThetaX=True)
    theta -= (alpha / N) * dTheta

    predictions = softmax_predict(x, theta)
    train_error = 1.0 - np.sum(predictions == y) / float(N)

    test_predictions = softmax_predict(x_test, theta)
    test_error = 1.0 - np.sum(test_predictions == y_test) / float(y_test.size)

    objective_values.append(obj)
    train_errors.append(train_error)
    test_errors.append(test_error)

    print t

    if t > 0 and t % 50 == 0:
      plt.plot(train_errors, label='training error')
      plt.plot(test_errors, label='test error')
      plt.legend(loc=3)
      plt.show()

  return theta


def softmax_predict(x, theta, thetaX=None):
  if thetaX is None:
    thetaX = theta.dot(x)
  return np.argmax(thetaX, axis=0)


def objective(x, y, theta, returnThetaX=False):
  thetaX = theta.dot(x)
  alphas = np.max(thetaX, axis=0)
  thetaX -= alphas
  denoms = np.sum(np.exp(thetaX), axis=0)
  obj = np.sum(np.log(denoms) - np.choose(y, thetaX))

  N = y.shape[0]
  c = 0
  obj += c * N * 0.5 * np.linalg.norm(theta, ord='fro') ** 2.0

  dTheta = np.zeros(theta.shape)
  K = theta.shape[0]
  for k in xrange(K):
    probs = np.exp(thetaX[k, :]) / denoms
    probs -= (y == k).astype(np.float32)
    dTheta[k, :] = np.sum(probs * x, axis=1)

  dTheta += c * N * theta

  if returnThetaX:
    return obj, dTheta, thetaX
  else:
    return obj, dTheta


def objective_slow(x, y, theta):
  """
  x - M x N
  y - M
  theta - K x M
  """
  M, N = x.shape
  K = theta.shape[0]
  obj = 0
  for i in xrange(N):
    s = 0
    alpha = max(theta[j, :].dot(x[:, i]) for j in xrange(K))
    for j in xrange(K):
      s += np.exp(theta[j, :].dot(x[:, i]) - alpha)
    obj += np.log(s)
    obj -= theta[y[i], :].dot(x[:, i]) - alpha

  dTheta = np.zeros(theta.shape)
  for k in xrange(K):
    for i in xrange(N):
      s = 0
      alpha = max(theta[j, :].dot(x[:, i]) for j in xrange(K))
      for j in xrange(K):
        s += np.exp(theta[j, :].dot(x[:, i]) - alpha)
      s = np.exp(theta[k, :].dot(x[:, i]) - alpha) / s
      dTheta[k, :] += s * x[:, i]
      if y[i] == k:
        dTheta[k, :] -= x[:, i]

  return obj, dTheta

def compare_objectives():
  K = 5
  N = 10
  M = 17
  y = np.random.randint(0, high=K, size=N)
  x = np.random.rand(M, N)
  theta = np.random.rand(K, M)

  obj1, dTheta1 = objective_slow(x, y, theta)
  obj2, dTheta2 = objective(x, y, theta)
  print obj1 - obj2
  print np.linalg.norm(dTheta1 - dTheta2, ord='fro')

def gradient_check():
  K = 5
  N = 10
  M = 5
  y = np.random.randint(0, high=K, size=N)
  x = np.random.rand(M, N)
  theta = np.random.rand(K, M)

  _, dTheta = objective(x, y, theta)
  
  dTheta_numeric = np.zeros(theta.shape)
  epsilon = 0.001
  for k in xrange(K):
    for m in xrange(M):
      theta[k, m] += epsilon / 2.0
      obj1, _ = objective(x, y, theta)
      theta[k, m] -= epsilon
      obj2, _ = objective(x, y, theta)
      theta[k, m] += epsilon / 2.0
      dTheta_numeric[k, m] = (obj1 - obj2) / epsilon

  print np.linalg.norm(dTheta - dTheta_numeric, ord='fro')

def visualize_weights(theta):
  theta_img = np.reshape(theta.T, (28, 28, 10), order='F')
  for i in xrange(10):
    plt.subplot(3, 4, i)
    img_plt = plt.imshow(theta_img[:, :, i])
    img_plt.set_interpolation('nearest')
    img_plt.set_cmap('gray')
    frame = plt.gca()
    frame.axes.get_xaxis().set_ticks([])
    frame.axes.get_yaxis().set_ticks([])
  plt.show()

if __name__ == '__main__':
  if False:
    gradient_check()
    # compare_objectives()
    sys.exit(0)

  with open('data/train-images.npy', 'rb') as f:
    train_images = np.load(f)
  with open('data/train-labels.npy', 'rb') as f:
    train_labels = np.load(f)
  with open('data/test-images.npy', 'rb') as f:
    test_images = np.load(f)
  with open('data/test-labels.npy', 'rb') as f:
    test_labels = np.load(f)

  height, width, num = train_images.shape
  train_images_vec = np.reshape(train_images, (height * width, num), 'F')

  test_num = test_images.shape[2]
  test_images_vec = np.reshape(test_images, (height * width, test_num), 'F')

  val_mask = np.random.rand(num) < 0.2
  all_train_images = train_images_vec
  all_train_labels = train_labels

  train_images = all_train_images[:, np.logical_not(val_mask)]
  train_labels = all_train_labels[np.logical_not(val_mask)]
  val_images = all_train_images[:, val_mask]
  val_labels = all_train_labels[val_mask]

  
  # K = 10
  # avg_imgs = np.zeros((K, height * width))
  # for k in xrange(K):
  #   samples = train_images_vec[:, train_labels == k]
  #   avg_imgs[k, :] = np.mean(samples, axis=1)
  # with open('data/avg-images.npy', 'wb') as f:
  #   np.save(f, avg_imgs)
  # sys.exit(0)

  theta = train_softmax(train_images, train_labels, val_images, val_labels)
  visualize_weights(theta)
