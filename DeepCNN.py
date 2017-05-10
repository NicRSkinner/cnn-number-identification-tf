import tensorflow as tf
import numpy as np


class DeepCNN:
    def __init__(self, savefile):
        self.savefile = savefile

    @property
    def savefile(self):
        return self.savefile

    @savefile.setter
    def savefile(self, value):
        self.savefile = value

    @property
    def prediction_op(self):
        return self.prediction_op

    @prediction_op.setter
    def prediction_op(self, value):
        self.prediction_op = value

    def build(self, x):
        x_image = tf.reshape(x, [-1, 28, 28, 1])

        W_conv1 = self.weight_variable([5, 5, 1, 32])
        b_conv1 = self.bias_variable([32])
        h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)

        h_pool1 = self.max_pool_2x2(h_conv1)

        # Second convolutional layer -- maps 32 feature maps to 64.
        W_conv2 = self.weight_variable([5, 5, 32, 64])
        b_conv2 = self.bias_variable([64])
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)

        # Second pooling layer.
        h_pool2 = self.max_pool_2x2(h_conv2)

        # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
        # is down to 7x7x64 feature maps -- maps this to 1024 features.
        W_fc1 = self.weight_variable([7 * 7 * 64, 1024])
        b_fc1 = self.bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # Dropout - controls the complexity of the model, prevents co-adaptation of
        # features.
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # Map the 1024 features to 10 classes, one for each digit
        W_fc2 = self.weight_variable([1024, 10])
        b_fc2 = self.bias_variable([10])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        self.prediction_op = tf.argmax(y_conv, 1)
        return y_conv, keep_prob

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

    def fit(self, x, y, xtest, ytest):
        # Create the model
        inputs = tf.placeholder(tf.float32, [None, 784])

        # Define loss and optimizer
        y_ = tf.placeholder(tf.float32, [None, 10])

        # Build the graph for the deep net
        y_conv, keep_prob = self.build(inputs)

        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for i in range(300):
                xbatch = x[0:50]
                ybatch = y[0:50]

                xtestbatch = xtest[0:50]
                ytestbatch = ytest[0:50]

                if i % 100 == 0:
                    train_accuracy = accuracy.eval(feed_dict={
                        inputs: xbatch, y_: ybatch, keep_prob: 1.0})
                    print('step %d, training accuracy %g' % (i, train_accuracy))

                train_step.run(feed_dict={inputs: xbatch, y_: ybatch, keep_prob: 0.5})

            print('test accuracy %g' % accuracy.eval(feed_dict={
                inputs: xtestbatch, y_: ytestbatch, keep_prob: 1.0}))
