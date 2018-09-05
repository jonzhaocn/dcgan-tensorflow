import tensorflow as tf
import tensorflow.contrib.layers as layers
import os
from utils import read_and_decode, save_sample

max_iter = 400000
batch_size = 64
z_dim = 100
image_height = 64
image_width = 64
channel = 3
device = '/gpu:0'
tfrecords_dir = '../data/celeba_tfrecords'
sample_dir = './sample'
ckpt_dir = './ckpt'
log_dir = './log'
load_model = True
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def generator(z):
    first_feature_h = 4
    first_feature_w = 4
    with tf.variable_scope('generator'):
        # tf.contrib.layers.fully_connected(inputs, num_outputs, activation_fn=tf.nn.relu, normalizer_fn=None)
        train = layers.fully_connected(z, first_feature_h*first_feature_w*512,
                                       weights_initializer=tf.random_normal_initializer(0, 0.02))
        # tf.contrib.layers.conv2d_transpose(inputs, num_outputs, kernel_size,stride=1, padding='SAME')

        train = layers.batch_norm(tf.reshape(train, [-1, first_feature_h, first_feature_w, 512]),
                                  activation_fn=tf.nn.relu)

        train = layers.conv2d_transpose(train, 256, 3, stride=2, activation_fn=tf.nn.relu,
                                        normalizer_fn=layers.batch_norm, padding='SAME',
                                        weights_initializer=tf.random_normal_initializer(0, 0.02))

        train = layers.conv2d_transpose(train, 128, 3, stride=2, activation_fn=tf.nn.relu,
                                        normalizer_fn=layers.batch_norm, padding='SAME',
                                        weights_initializer=tf.random_normal_initializer(0, 0.02))

        train = layers.conv2d_transpose(train, 64, 3, stride=2, activation_fn=tf.nn.relu,
                                        normalizer_fn=layers.batch_norm, padding='SAME',
                                        weights_initializer=tf.random_normal_initializer(0, 0.02))

        train = layers.conv2d_transpose(train, channel, 3, stride=2, activation_fn=tf.nn.tanh, padding="SAME",
                                        weights_initializer=tf.random_normal_initializer(0, 0.02))

        return train


def discriminator(image, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        image = layers.conv2d(image, 64, kernel_size=3, stride=2, activation_fn=tf.nn.leaky_relu,
                              weights_initializer=tf.random_normal_initializer(0, 0.02))

        image = layers.conv2d(image, 128, kernel_size=3, stride=2, activation_fn=tf.nn.leaky_relu,
                              normalizer_fn=layers.batch_norm,
                              weights_initializer=tf.random_normal_initializer(0, 0.02))

        image = layers.conv2d(image, 256, kernel_size=3, stride=2, activation_fn=tf.nn.leaky_relu,
                              normalizer_fn=layers.batch_norm,
                              weights_initializer=tf.random_normal_initializer(0, 0.02))

        image = layers.conv2d(image, 512, kernel_size=3, stride=2, activation_fn=tf.nn.leaky_relu,
                              normalizer_fn=layers.batch_norm,
                              weights_initializer=tf.random_normal_initializer(0, 0.02))

        image = tf.reshape(image, [batch_size, -1])

        logit = layers.fully_connected(image, 1, activation_fn=None,
                                       weights_initializer=tf.random_normal_initializer(0, 0.02))

    return logit


def build_graph():
    images_data_set = read_and_decode(tfrecords_dir, height=image_height, width=image_width)
    true_images = tf.train.shuffle_batch([images_data_set], batch_size=batch_size, capacity=5000,
                                         min_after_dequeue=2500, num_threads=2)

    noises = tf.random_uniform([batch_size, z_dim], minval=-1.0, maxval=1.0)
    fake_images = generator(noises)
    true_logits = discriminator(true_images)
    fake_logits = discriminator(fake_images, reuse=True)

    dis_loss_true = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=true_logits, labels=tf.ones_like(true_logits)))
    dis_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=fake_logits, labels=tf.zeros_like(fake_logits)))
    dis_loss_true_summary = tf.summary.scalar('discriminator_loss_true', dis_loss_true)
    dis_loss_fake_summary = tf.summary.scalar('discriminator_loss_fake', dis_loss_fake)

    dis_loss = dis_loss_true + dis_loss_fake

    gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=fake_logits, labels=tf.ones_like(fake_logits)))

    gen_loss_summary = tf.summary.scalar("generator_loss", gen_loss)
    dis_loss_summary = tf.summary.scalar("discriminator_loss", dis_loss)

    gen_loss_summary_merge = tf.summary.merge([gen_loss_summary])
    dis_loss_summary_merge = tf.summary.merge([dis_loss_summary, dis_loss_true_summary, dis_loss_fake_summary])

    gen_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    dis_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

    gen_counter = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
    dis_counter = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
    # ------------------------
    # set the learning rate to 1e-4, different from paper
    # if the learning_rate = 2e-4, it's easy to be model collapse
    gen_opt = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5,).\
        minimize(gen_loss, var_list=gen_params, global_step=gen_counter)
    dis_opt = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5,).\
        minimize(dis_loss, var_list=dis_params, global_step=dis_counter)
    return gen_opt, dis_opt, fake_images, gen_loss_summary_merge, dis_loss_summary_merge


def train():
    with tf.device(device):
        gen_opt, dis_opt, fake_images, gen_loss_summary, dis_loss_summary = build_graph()
    saver = tf.train.Saver()
    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    session_config.gpu_options.per_process_gpu_memory_fraction = 0.8

    with tf.Session(config=session_config) as sess:
        # add tf.train.start_queue_runners, it's important to start queue for tf.train.shuffle_batch
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        iter_start = 0
        if load_model:
            lasted_checkpoint = tf.train.latest_checkpoint(ckpt_dir)
            if lasted_checkpoint is not None:
                saver.restore(sess, lasted_checkpoint)
                print('load model:', lasted_checkpoint)
                iter_start = int(lasted_checkpoint.split('/')[-1].split('-')[-1])+1
            else:
                print('init global variables')
                sess.run(tf.global_variables_initializer())
        for iter_count in range(iter_start, max_iter):

            # train discriminator
            if iter_count % 100 == 99:
                _, summary = sess.run([dis_opt, dis_loss_summary])
                summary_writer.add_summary(summary, iter_count)
            else:
                sess.run(dis_opt)

            # Run gen_opt twice to make sure that d_loss does not go to zero (different from paper)
            sess.run(gen_opt)
            # train generator
            if iter_count % 100 == 99:
                _, summary = sess.run([gen_opt, gen_loss_summary])
                summary_writer.add_summary(summary, iter_count)
            else:
                sess.run(gen_opt)

            # save sample
            if iter_count % 1000 == 999:
                sample_path = os.path.join(sample_dir, '%d.jpg' % iter_count)
                sample = sess.run(fake_images)
                sample = (sample+1.0)/2.0
                save_sample(sample, [4, 4], sample_path)
                print('save sample:', sample_path)

            # save model
            if iter_count % 1000 == 999:
                if not os.path.exists(ckpt_dir):
                    os.mkdir(ckpt_dir)
                ckpt_path = os.path.join(ckpt_dir, "model.ckpt")
                saver.save(sess, ckpt_path, global_step=iter_count)
                print('save ckpt:', ckpt_dir)
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    train()