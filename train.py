"""
A very simple MNIST classifier.
See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import pickle

from DeepCNN import DeepCNN

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    parser.add_argument('--network_save_dir', type=str,
                        default='data/',
                        help='Directory for storing the neural network')
    FLAGS, unparsed = parser.parse_known_args()

    train_labels, train_images, test_labels, test_images = pickle.load(open("MNIST/data.pkl", 'rb'))

    model = DeepCNN('tf.model')

    model.fit(x=train_images, y=train_labels, xtest=test_images, ytest=test_labels)
