import tensorflow as tf
from model import generator
from utils import save_sample

batch_size = 64
z_dim = 128
device = '/gpu:0'
ckpt_dir = './ckpt_dcgan'


def generate_from_ckpt():
    with tf.device(device):
        noises = tf.random_normal([batch_size, z_dim], mean=0.0, stddev=1.0)
        generate_images = generator(noises)
    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    session_config.gpu_options.per_process_gpu_memory_fraction = 0.6

    if ckpt_dir is not None:
        with tf.Session(config=session_config) as sess:
            saver = tf.train.Saver()
            lasted_checkpoint = tf.train.latest_checkpoint(ckpt_dir)
            if lasted_checkpoint is not None:
                saver.restore(sess, lasted_checkpoint)
                images = sess.run(generate_images)
                save_sample(images, [8, 8], './generated_image.jpg')
                print('generate a image:', './generated_image.jpg')
            else:
                print('there is not checkpoint file in:', ckpt_dir)


if __name__ == '__main__':
    generate_from_ckpt()