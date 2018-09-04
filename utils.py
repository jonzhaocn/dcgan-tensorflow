import os
import tensorflow as tf
import numpy as np
import scipy.misc

batch_size = 64
z_dim = 128


def read_and_decode(tfrecords_dir, height, width, channel=3):
    tfrecords_list = os.listdir(tfrecords_dir)
    tfrecords_names = [os.path.join(tfrecords_dir, name) for name in tfrecords_list]
    tfrecords_queue = tf.train.string_input_producer(tfrecords_names, shuffle=False)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(tfrecords_queue)
    features = tf.parse_single_example(serialized_example, features={'image': tf.FixedLenFeature([], tf.string)})
    img = tf.decode_raw(features['image'], tf.uint8)
    img = tf.reshape(img, [height, width, channel])
    img = (tf.cast(img, tf.float32) - 128.0) / 128.0
    return img


def save_sample(images, size, path):

    pardir, _ = os.path.split(path)
    if not os.path.exists(pardir):
        os.mkdir(pardir)

    h, w = images.shape[1], images.shape[2]

    # create a large array for storing images
    merge_img = np.zeros((h * size[0], w * size[1], 3))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        if j >= size[0]:
            break
        merge_img[j * h:j * h + h, i * w:i * w + w, :] = image

    # save the array
    return scipy.misc.imsave(path, merge_img)

