import os
import sys

def usage(exe_name):
  print('''Usage: {} [FLAG] PATH
FLAG:
-t, --train    If present, will train model and save it to PATH
PATH           Path to load model from otherwise
  '''.format(os.path.basename(exe_name)))

# Simple command line argument parse
if len(sys.argv) > 2 and sys.argv[1] in ["--train", "-t"]:
  train_model = True
  model_path = sys.argv[2]
elif len(sys.argv) > 1:
  train_model = False
  model_path = sys.argv[1]
else:
  print(usage(sys.argv[0]))
  sys.exit(2)


# These are slow, doing them after arg parsing
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

x = tf.placeholder(tf.float32, [None, 784])
x_image = tf.reshape(x, [-1, 28, 28, 1])

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


saver = tf.train.Saver()


with tf.Session() as sess:
    if train_model:
        tf.global_variables_initializer().run()
        
        # Train
        for i in range(2200):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            if(i % 100 == 0):
                train_accuracy = accuracy.eval(feed_dict={
                    x: batch_xs, y_: batch_ys, keep_prob: 1.0})
                print("step {}, accuracy {}".format(i, train_accuracy))
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
        
        # Save first, just in cases something goes wrong
        saver.save(sess, model_path)
        
        # Test
        print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

    else:
        saver.restore(sess, model_path)


import imageio
import matplotlib.pyplot as plt
import numpy as np

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

classify = tf.argmax(y, 1)

## Converts a 28x28, greyscale (8-bit) png image into a MNIST data point
def mnistify(img):
    img_flat = img.reshape((1, 28**2)) #, order='F')
    img_flat = img_flat / 255.0  # 8-bit png to 0..1
    img_flat = 1 - img_flat  # MNIST uses 0 for white

    return np.array(img_flat, dtype=np.float32)

def demnistify(img):
    img = 1 - img
    img = img * 255
    return img.reshape((28, 28))

# 'MNIST_data/my_test/2/a.png'
def classify_demo(path):
    img = imageio.imread(path)   
    img_flat = mnistify(img)

    label = sess.run(classify, feed_dict={x: img_flat, keep_prob: 1.0})

    plt.imshow(img, cmap='gray')
    plt.title(label)
    plt.show()
