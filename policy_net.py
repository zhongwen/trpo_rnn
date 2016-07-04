import tensorflow as tf
import numpy as np
import prettytensor as pt


def construct_policy_net(obs, action_dim):
    with tf.variable_scope('policy_net') as scope:
        mean = (
            pt.wrap(obs).fully_connected(action_dim, activation_fn=None, stddev=0.01))
        # fully_connected(64, activation_fn=tf.nn.relu, stddev=0.01).
        # fully_connected(action_dim, activation_fn=None, stddev=0.01))
        action_dist_logstd_param = tf.get_variable(
            'logstd', [1, action_dim],
            initializer=tf.random_normal_initializer(0.0, 0.1))
        action_dist_logstd = tf.tile(
            action_dist_logstd_param, tf.pack([obs.get_shape()[0], 1]))
        std = tf.exp(action_dist_logstd)
    return tf.concat(1, [mean, std])
