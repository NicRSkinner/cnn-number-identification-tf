import numpy as np
import tensorflow as tf
import pickle
from PIL import Image

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('test_labels', 't10k-labels-idx1-ubyte',
                    "Test Labels file")

flags.DEFINE_string('test_images', 't10k-images-idx3-ubyte',
                    "Test Images file")

flags.DEFINE_string('train_labels', 'train-labels-idx1-ubyte',
                    "Training Labels file")

flags.DEFINE_string('train_images', 'train-images-idx3-ubyte',
                    "Training Images file")


def bytes_from_file(filename, chunksize=8192, byteoffset=8):
    with open(filename, 'rb') as byte_file:
        byte_file.read(byteoffset)
        while True:
            chunk = byte_file.read(chunksize)

            if chunk:
                for b in chunk:
                    yield b
            else:
                break


def get_arr_images(fbytes):
    fdata = []

    for fbyte in fbytes:
        fdata.append(fbyte)

    fdata = np.array(fdata)

    item_amt = int(fdata.size / (28 * 28))

    fdata = np.reshape(fdata, (item_amt, 28, 28))

    return fdata


def get_arr_labels(fbytes):
    fdata = []

    for fbyte in fbytes:
        fdata.append(fbyte)

    fdata = np.array(fdata)

    return fdata


def pickle_files(filenames):
    file_data = []

    for file in filenames:
        if file == filenames[1] or file == filenames[3]:
            file_bytes = bytes_from_file(file, 1, 16)
            file_data.append(get_arr_images(file_bytes))
        else:
            file_bytes = bytes_from_file(file, 1)
            file_data.append(get_arr_labels(file_bytes))

    with open('data.pkl', 'wb') as data_file:
        pickle.dump(file_data, data_file)


def de_pickle():
    train_labels, train_images, test_labels, test_images = pickle.load(open('data.pkl', 'rb'))

    print(np.array(test_labels)[2])
    test_image = np.array(test_images)[2]

    img = Image.new('L', (28, 28), "black")
    pixels = img.load()

    for i in range(img.size[0]):
        for j in range(img.size[1]):
            pixels[i, j] = (test_image[i][j],)

    img.show()


if __name__ == '__main__':
    filenames = [
        FLAGS.train_labels,
        FLAGS.train_images,
        FLAGS.test_labels,
        FLAGS.test_images
    ]

    #pickle_files(filenames)
    de_pickle()
