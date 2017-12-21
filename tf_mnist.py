from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

x = tf.placeholder(tf.float32, [None, 784])
W_n = tf.Variable(tf.zeros([784, 10]))
b_n = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W_n) + b_n)

y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# Train
for i in range(1000):
  print("Minibatch {}...".format(i))
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Test
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

classify = tf.argmax(y, 1)


import imageio
import matplotlib.pyplot as plt
import numpy as np

## Converts a 28x28, greyscale (8-bit) png image into a MNIST data point
def mnistify(img):
    img_flat = img.reshape((1, 28**2)) #, order='F')
    img_flat = img_flat / 255.0  # 8-bit png to 0..1
    img_flat = 1 - img_flat  # MNIST uses 0 for white

def demnistify(img):
    img = 1 - img
    img = img * 255
    return img.reshape((28, 28))

    return np.array(img_flat, dtype=np.float32)
# 'MNIST_data/my_test/2/a.png'
def classify_demo(path):
    img = imageio.imread(path)   
    img_flat = mnistify(img)

    label = sess.run(classify, feed_dict={x: img_flat})

    plt.imshow(img, cmap='gray')
    plt.title(label)
    plt.show()
