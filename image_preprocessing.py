import os
import time
from PIL import Image
import math
import tensorflow as tf


crop_width = 128
crop_height = 128
resize_width = 64
resize_height = 64
channel = 3
images_count_each_tfrecords = 20000
images_dir = '../data/img_align_celeba'
tfrecords_dir = '../data/celeba_tfrecords'


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(images_dir, tfrecords_dir):
    """
        read images and then convert them to tfrecords
        tfrecords is a file format that will use in training as input
        divide the all images into several parts
        :param data_dir:
        :param tfrecords_dir:
        :return:
    """
    if not os.path.exists(tfrecords_dir):
        os.mkdir(tfrecords_dir)

    if len(os.listdir(tfrecords_dir)) != 0:
        print('the %s is not empty' % tfrecords_dir)
        exit()

    # set the crop size
    file_list = os.listdir(images_dir)
    list_len = len(file_list)
    tfrecords_count = int(math.ceil(list_len / images_count_each_tfrecords))
    total_count = 0
    print('there are %d images and it will create %d tfrecords' % (list_len, tfrecords_count))
    for i in range(tfrecords_count):
        print('write the %dth tfrecords' % (i + 1))

        # -----write to tfrecords
        writer = tf.python_io.TFRecordWriter(os.path.join(tfrecords_dir, str(i + 1) + '.tfrecords'))
        os.listdir(images_dir)
        curr_file_list = []
        if i < tfrecords_count - 1:
            curr_file_list = file_list[i * images_count_each_tfrecords:
                                       (i + 1) * images_count_each_tfrecords]
        elif i == tfrecords_count - 1:
            curr_file_list = file_list[i * images_count_each_tfrecords:]

        for j in range(len(curr_file_list)):
            if j % 1000 == 0:
                print("current index of image is %d in the %dth tfrecords" % (j + 1, i + 1))
            img_name = curr_file_list[j]
            img_path = os.path.join(images_dir, img_name)
            img = Image.open(img_path)

            # crop the center of images
            w, h = img.size[:2]
            j = (w - crop_width) // 2
            k = (h - crop_height) // 2
            # box ask for two points
            box = (j, k, j + crop_width, k + crop_height)
            img = img.crop(box=box)
            img = img.resize((resize_width, resize_height))
            # convert to bytes
            img_raw = img.tobytes()
            if len(img_raw) != resize_width * resize_height * channel:
                continue
            total_count += 1
            example = tf.train.Example(features=tf.train.Features(feature={
                'image': _bytes_feature(img_raw)}))
            writer.write(example.SerializeToString())
        writer.close()
    print("total count:", total_count)


if __name__ == '__main__':
    start_time = time.time()
    print('Convert start')
    convert_to(images_dir, tfrecords_dir)
    print('Convert done, take %.2f seconds' % (time.time() - start_time))