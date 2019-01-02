import tensorflow as tf


def linear_layer(name, input_data, input_size, output_size):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        w = tf.get_variable('w',
                            shape=[input_size, output_size],
                            trainable=True,
                            initializer=tf.random_uniform_initializer(-0.1, 0.1))
        b = tf.get_variable('b',
                            shape=[output_size],
                            trainable=True,
                            initializer=tf.random_uniform_initializer(-0.1, 0.1))
    return tf.matmul(input_data, w) + b
