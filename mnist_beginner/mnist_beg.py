# Import data from mnist to get images
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

# Define our tensor to hold images. 784 = pixels, None indicates
# we can hold any number of images
x = tf.placeholder(tf.float32, [None, 784])

# Create the Variables for the weights and biases
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# This is our prediction output
y = tf.nn.softmax(tf.matmul(x, W) + b)

# This will be our correct answers/output
y_ = tf.placeholder(tf.float32, [None, 10])

# Calculate cross entropy by taking the negative sum of our correct values
# multiplied by the log of our predictions
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

# This is training the NN with backpropagation
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# Initializes all the variables we made above
init = tf.initialize_all_variables()

# Create and launch the session
sess = tf.Session()
sess.run(init)

# Train. In this case, 1000 times
for i in range(1000):
  # Feeds in 100 random images each time for training (stochastic training)
  # batch_xs are our images (pixels), batch_ys are our correct outputs
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  
# tf.argmax gives the index of the highest entry in a tensor along some axis
# argmax (y, 1) gives the label our model said was right, argmax(y_, 1) is the
# correct label. tf.equal sees if these two are the same
# This returns a list of booleans saying if it's true or false (predicted or not)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

# False is 0 and True is 1, what was our average?
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# How well did we do?
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

# Close the session (self documenting code)
sess.close()