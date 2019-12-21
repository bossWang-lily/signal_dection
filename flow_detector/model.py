import tensorflow as tf

import tfops
from data import *


def standard_gaussian(shape):
    return GaussianDiag(tf.zeros(shape), tf.zeros(shape))


class GaussianDiag:
    def __init__(self, mean, logsd):
        self.mean = mean
        self.logsd = logsd

    def sample_from(self, eps):
        return self.mean + tf.exp(self.logsd) * eps

    def logp(self, x):
        result = -0.5 * (np.log(2 * np.pi) + 2. * self.logsd + (x - self.mean) ** 2 / tf.exp(2. * self.logsd))
        with tf.control_dependencies([tf.assert_non_positive(result)]):
            return result


class Detector:
    """Maximum likelihood detector with normalizing flow based density estimation"""

    def __init__(self, detpth):
        self.detpth = detpth
        self.unique_name = "DensityEstimator{}".format(self.detpth)
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.variable_scope('model'):
                self.w = tf.placeholder(tf.float32, [None, 2 * NUM_ANT, 1], name="w")
                self.sum_logdets = tf.zeros_like(self.w, dtype='float32')[:, 0, 0]

                # Encoding
                self.z = tfops.squeeze1d(self.w, factor=2)
                self.z, self.sum_logdets = tfops.flow("flows", self.z, self.sum_logdets, self.detpth)

                # Prior
                z_shape = tfops.int_shape(self.z)
                h = tfops.conv1d_zeros('p', tf.zeros([1, z_shape[1], z_shape[2]]), 2 * z_shape[2])
                mu = h[:, :, :z_shape[2]]
                logs = h[:, :, z_shape[2]:]
                self.gauss = standard_gaussian([1, z_shape[1], z_shape[2]])
                self.logpz = self.gauss.logp(self.z)

                # Objective function
                self.logpw = tf.reduce_sum(self.logpz, [1, 2]) + self.sum_logdets

                # Optimization
                self.nll = tf.reduce_mean(-self.logpw)
                self.train_op = tf.train.AdamOptimizer().minimize(self.nll)

                self.init_variables = tf.global_variables_initializer()

        gpu_options = tf.GPUOptions()
        gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=tf.ConfigProto(gpu_options=gpu_options))

    def logp(self, w):
        return self.sess.run(self.logpw, feed_dict={self.w: w})  # [batch_size]

    def load(self):
        with self.graph.as_default():
            saver = tf.train.Saver(tf.global_variables())
            path = "savedModel/{}/".format(self.unique_name)
            saver.restore(self.sess, path)
            print("Model \"{}\" loaded".format(self.unique_name))

    def save(self):
        with self.graph.as_default():
            saver = tf.train.Saver(tf.global_variables())
            path = "savedModel/{}/".format(self.unique_name)
            folder = os.path.dirname(path)
            if not os.path.exists(folder):
                os.makedirs(folder)
            prefix = saver.save(self.sess, path)
            print("Model saved at \"{}\"".format(prefix))

    def close(self):
        self.sess.close()

    def train(self, train_set, valid_set, max_flip=5, max_epoch=1000):
        print("Initializing model {}".format(self.unique_name))
        self.sess.run(self.init_variables)

        flip_count = 0
        best_ber = None
        epoch = 0
        while epoch < max_epoch:
            batch_idx = 0
            for y, h, s, one_hot, w, s_mld, w_mld in train_set.fetch():
                _, nll, logpz, sum_logdets = self.sess.run(
                    [self.train_op, self.nll, self.logpz, self.sum_logdets],
                    feed_dict={
                        self.w: w
                    }
                )
                batch_idx += 1
                print("Training model, epoch {}, batch {}, nll={:e}, logpz={:e} sum_logdets={:e}".format(
                    epoch + 1, batch_idx, nll, logpz.mean(), sum_logdets.mean()))

            print()

            if (epoch+1) % 5 == 0:
                new_ber = self.__valid(valid_set)
                if best_ber is None or new_ber < best_ber:
                    best_ber = new_ber
                    self.save()
                else:
                    flip_count += 1
                    if flip_count >= max_flip:
                        break

            epoch += 1
        print("Model \"{}\" train over".format(self.unique_name))

    def __valid(self, valid_set):
        mld_err_bits = 0
        err_bits = 0
        total_bits = 0
        idx = 0

        for y, h, s, one_hot, w, s_mld, w_mld in valid_set.fetch():
            bits = get_bits(s)
            total_bits += bits.size

            mld_bits = get_bits(s_mld)
            mld_err_bits += len(np.argwhere(mld_bits != bits))
            mld_ber = mld_err_bits / total_bits

            bits_estimated = self.detect(y, h)
            err_bits += len(np.argwhere(bits_estimated != bits))
            ber = err_bits / total_bits

            print("Validating, batch {}, ber={:e} mld_ber={:e}".format(idx + 1, ber, mld_ber),
                  end="\r")
            idx += 1

        print()
        print("Model validated, ber={:e} mld_ber={:e}".format(ber, mld_ber))
        return ber

    def detect(self, y, h):
        s = None

        # Log-likelihoods for all candidates
        ll_candidates = None
        for k in range(QPSK_CANDIDATE_SIZE):
            s_cand = QPSK_CANDIDATES[:, k:k + 1]
            logpw = self.logp(y - h @ s_cand)
            logpw = np.reshape(logpw, [-1, 1])
            if ll_candidates is None:
                ll_candidates = logpw
            else:
                ll_candidates = np.concatenate((ll_candidates, logpw), axis=1)

        # Select the maximum likelihood candidates
        maximum_ll = np.argmax(ll_candidates, axis=1)
        for k in maximum_ll.flatten():
            s_cand = QPSK_CANDIDATES[:, k:k + 1].reshape([-1, 2 * NUM_ANT, 1])
            s = concatenate(s, s_cand)

        return get_bits(s)
