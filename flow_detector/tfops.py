import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope, arg_scope


def default_initial_value(shape, std=0.05):
    return tf.random_normal(shape, 0., std)


def default_initializer(std=0.05):
    return tf.random_normal_initializer(0., std)


def int_shape(x):
    if str(x.get_shape()[0]) != '?':
        return list(map(int, x.get_shape()))
    return [-1] + list(map(int, x.get_shape()[1:]))


@add_arg_scope
def conv1d(name, x, out_channel, filter_size=1, stride=1, do_actnorm=True):
    with tf.variable_scope(name):
        n_in = int(x.get_shape()[2])
        w = tf.get_variable("W", [filter_size, n_in, out_channel], tf.float32, initializer=default_initializer())
        x = tf.nn.conv1d(x, filters=w, stride=stride, padding="SAME", data_format='NWC')

        if do_actnorm:
            x = actnorm("actnorm", x)
        else:
            x += tf.get_variable("b", [1, 1, out_channel], initializer=tf.zeros_initializer())

    return x


@add_arg_scope
def conv1d_zeros(name, x, out_channel, filter_size=1, stride=1, logscale_factor=3):
    """Both W and B are initialized to zeros"""
    with tf.variable_scope(name):
        n_in = int(x.get_shape()[2])
        w = tf.get_variable("W", [filter_size, n_in, out_channel], tf.float32, initializer=tf.zeros_initializer())
        x = tf.nn.conv1d(x, w, stride=stride, padding="SAME", data_format='NWC')
        b = tf.get_variable("b", [1, 1, out_channel], initializer=tf.zeros_initializer())
        logs = tf.get_variable("logs", [1, 1, out_channel], initializer=tf.zeros_initializer()) * logscale_factor
    return b + x * tf.exp(logs)


def flatten_sum(logps):
    if len(logps.get_shape()) == 2:
        return tf.reduce_sum(logps, [1])
    elif len(logps.get_shape()) == 3:
        return tf.reduce_sum(logps, [1, 2])
    else:
        raise Exception()


# wrapper tf.get_variable, augmented with 'init' functionality
# Get variable with data dependent init
@add_arg_scope
def get_variable_ddi(name, shape, initial_value, dtype=tf.float32, init=False, trainable=True):
    w = tf.get_variable(name, shape, dtype, None, trainable=trainable)
    if init:
        assign_w = w.assign(initial_value)
        with tf.control_dependencies([assign_w]):
            return w
    return w


def squeeze1d(x, factor=2):
    assert factor >= 1
    if factor == 1:
        return x

    shape = x.get_shape()
    width = int(shape[1])
    n_channels = int(shape[2])
    assert width % factor == 0

    x = tf.reshape(x, [-1, width // factor, factor, n_channels])
    x = tf.transpose(x, [0, 1, 3, 2])
    x = tf.reshape(x, [-1, width // factor, n_channels * factor])
    return x


# Activation normalization
# Convenience function that does centering+scaling
@add_arg_scope
def actnorm(name, x, scale=1., logdet=None, batch_variance=False, trainable=True):
    if arg_scope([get_variable_ddi], trainable=trainable):
        x = actnorm_center(name + "_center", x)
        x = actnorm_scale(name + "_scale", x, scale, logdet, batch_variance, trainable)
        if logdet is not None:
            x, logdet = x

        if logdet is not None:
            return x, logdet
        return x


# Activation normalization
@add_arg_scope
def actnorm_center(name, x):
    shape = x.get_shape()
    with tf.variable_scope(name):
        assert len(shape) == 2 or len(shape) == 3
        if len(shape) == 2:
            x_mean = tf.reduce_mean(x, [0], keepdims=True)
            b = get_variable_ddi("b", (1, int_shape(x)[1]), initial_value=-x_mean)
        elif len(shape) == 3:
            x_mean = tf.reduce_mean(x, [0, 1], keepdims=True)
            b = get_variable_ddi("b", (1, 1, int_shape(x)[2]), initial_value=-x_mean)

        x += b

        return x


# Activation normalization
@add_arg_scope
def actnorm_scale(name, x, scale=1., logdet=None, batch_variance=False, trainable=True):
    shape = x.get_shape()
    with tf.variable_scope(name), arg_scope([get_variable_ddi], trainable=trainable):
        assert len(shape) == 2 or len(shape) == 3
        if len(shape) == 2:
            x_var = tf.reduce_mean(x ** 2, [0], keepdims=True)
            logdet_factor = 1
            _shape = (1, int_shape(x)[1])
        elif len(shape) == 3:
            x_var = tf.reduce_mean(x ** 2, [0, 1], keepdims=True)
            logdet_factor = int(shape[1])
            _shape = (1, 1, int_shape(x)[2])

        if batch_variance:
            x_var = tf.reduce_mean(x ** 2, keepdims=True)

        logs = get_variable_ddi("logs", _shape, initial_value=tf.log(scale / (tf.sqrt(x_var) + 1e-6)))

        x = x * tf.exp(logs)

        if logdet is not None:
            dlogdet = tf.reduce_sum(logs) * logdet_factor
            with tf.control_dependencies([tf.check_numerics(dlogdet, "actnorm_scale logdet not a number")]):
                return x, logdet + dlogdet

        return x


# Invertible unit conv
@add_arg_scope
def unit_conv(name, z, logdet):
    # LU-decomposed version
    shape = int_shape(z)
    with tf.variable_scope(name):
        dtype = 'float64'

        # Random orthogonal matrix:
        import scipy
        np_w = scipy.linalg.qr(np.random.randn(shape[2], shape[2]))[0].astype('float32')

        np_p, np_l, np_u = scipy.linalg.lu(np_w)
        np_s = np.diag(np_u)
        np_sign_s = np.sign(np_s)
        np_log_s = np.log(abs(np_s))
        np_u = np.triu(np_u, k=1)

        p = tf.get_variable("P", initializer=np_p, trainable=False)
        l = tf.get_variable("L", initializer=np_l)
        sign_s = tf.get_variable("sign_S", initializer=np_sign_s, trainable=False)
        log_s = tf.get_variable("log_S", initializer=np_log_s)
        # S = tf.get_variable("S", initializer=np_s)
        u = tf.get_variable("U", initializer=np_u)

        p = tf.cast(p, dtype)
        l = tf.cast(l, dtype)
        sign_s = tf.cast(sign_s, dtype)
        log_s = tf.cast(log_s, dtype)
        u = tf.cast(u, dtype)

        w_shape = [shape[2], shape[2]]

        l_mask = np.tril(np.ones(w_shape, dtype=dtype), -1)
        l = l * l_mask + tf.eye(*w_shape, dtype=dtype)
        u = u * np.transpose(l_mask) + tf.diag(sign_s * tf.exp(log_s))
        w = tf.matmul(p, tf.matmul(l, u))

        w = tf.cast(w, tf.float32)
        log_s = tf.cast(log_s, tf.float32)

        w = tf.reshape(w, [1] + w_shape)
        z = tf.nn.conv1d(z, w, stride=1, padding='SAME', data_format='NWC')
        dlogdet = tf.reduce_sum(log_s) * shape[1]
        with tf.control_dependencies([tf.check_numerics(dlogdet, "uni_conv logdet not a number")]):
            return z, logdet + dlogdet


def f_nn(name, x, hidden_channel, out_channel=None):
    out_channel = out_channel or int(x.get_shape()[2])
    with tf.variable_scope(name):
        x = tf.nn.relu(conv1d("layer_1", x, hidden_channel, filter_size=3))
        x = tf.nn.relu(conv1d("layer_2", x, hidden_channel, filter_size=1))
        x = conv1d_zeros("l_last", x, out_channel)
    return x


@add_arg_scope
def affine_coupling(name, z, logdet):
    with tf.variable_scope(name):
        z_shape = int_shape(z)
        logs = tf.get_variable("logs", [1, z_shape[1], z_shape[2]], initializer=tf.zeros_initializer())
        b = tf.get_variable("b", [1, z_shape[1], z_shape[2]], initializer=tf.zeros_initializer())
        z = z * tf.exp(logs) + b
        logdet += tf.reduce_sum(logs, [1, 2])

        return z, logdet


# Normalizing flow
@add_arg_scope
def flow(name, z, logdet, depth):
    with tf.variable_scope(name):
        for i in range(depth):
            z, logdet = flow_step(str(i), z, logdet)
    return z, logdet


# Simpler, new version
@add_arg_scope
def flow_step(name, z, logdet):
    with tf.variable_scope(name):
        z, logdet = actnorm("act_norm", z, logdet=logdet)
        z, logdet = unit_conv("inv_conv", z, logdet)
        z, logdet = affine_coupling("affine_coupling", z, logdet)

    return z, logdet
