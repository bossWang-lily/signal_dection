import tensorflow as tf

from qpsk import *


def count_error_bits(pred, true):
    assert pred.shape == true.shape
    return len(np.argwhere(pred != true))


class DNN:
    def __init__(self, sir):
        self.sir = sir
        self.unique_name = "DNN_sir{}".format(NUM_ANT, sir)

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.y = tf.placeholder(tf.float32, [TIME_SLOTS_PER_BATCH, 2 * NUM_ANT], "y")
            self.x_one_hot = tf.placeholder(tf.float32, [TIME_SLOTS_PER_BATCH, QPSK_CANDIDATE_SIZE], "one_hot")

            with tf.variable_scope("dnn"):
                # fc + relu
                h1 = tf.layers.dense(inputs=self.y, units=128, use_bias=True, activation=tf.nn.relu)
                # fc + relu
                h2 = tf.layers.dense(inputs=h1, units=128, use_bias=True, activation=tf.nn.relu)
                # fc + relu
                h3 = tf.layers.dense(inputs=h2, units=128, use_bias=True, activation=tf.nn.relu)
                # fc
                logits = tf.layers.dense(inputs=h3, units=QPSK_CANDIDATE_SIZE, use_bias=True, activation=None)

                # softmax  -> X_{one_hot} -> Cross Encropy
                self.loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.x_one_hot, logits=logits))

                # softmax -> X_{one_hot}
                self.hat_x_one_hot = tf.nn.softmax(logits)

            self.train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)
            self.init_variables = tf.global_variables_initializer()

        gpu_options = tf.GPUOptions()
        gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=tf.ConfigProto(gpu_options=gpu_options))

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

    def train(self):
        print("Initializing model {}".format(self.unique_name))
        self.sess.run(self.init_variables)

        train_set = DataSet(sir=self.sir, flag="train")

        flip_count = 0
        best_ber = None
        epoch = 0
        while epoch < MAX_EPOCHS:
            batch_idx = 0
            for y, s, x_one_hot in train_set.fetch():
                _, loss, = self.sess.run(
                    [self.train_op, self.loss],
                    feed_dict={
                        self.y: y,
                        self.x_one_hot: x_one_hot,
                    }
                )
                batch_idx += 1
                print("Training model \"{}\", epoch {}/{}, batch {}/{}, loss={:e}".format(
                    self.unique_name, epoch + 1, MAX_EPOCHS, batch_idx, TRAIN_TOTAL_BATCH, loss), end="\r")
            print()

            epoch += 1

            if epoch > 0 and epoch % VALID_MODEL_EVERY_EPOCHES == 0:
                new_ber = self.__valid_online()
                if best_ber is None or new_ber < best_ber:
                    best_ber = new_ber
                    self.save()
                else:
                    flip_count += 1
                    if flip_count >= MAX_FLIP:
                        break

        print("Model \"{}\" train over".format(self.unique_name))

    def __valid_online(self):
        err_count = 0
        bits_count = 0
        valid_set = DataSet(sir=self.sir, flag="valid")
        idx = 0
        for y, s, x_one_hot in valid_set.fetch():
            bits = get_bits(s)
            bits_count += bits.size

            # print("true_class=\n", np.unravel_index(x_one_hot.argmax(1), x_one_hot.shape)[1])
            bits_detected = self.detect_bits(y)

            err_count += count_error_bits(pred=bits_detected, true=bits)

            print("Validating model \"{}\", batch {}/{}, err_count={}".format(
                self.unique_name, idx + 1, VALID_TOTAL_BATCH, err_count), end="\r")

            idx += 1
        print()
        ber = err_count / bits_count
        print("Model validated, BER={:e}".format(ber))
        return ber

    def detect_bits(self, y):
        hat_x_one_hot = self.sess.run(self.hat_x_one_hot, feed_dict={self.y: y})

        s_batch = None
        max_indexes_on_axis_1 = np.unravel_index(hat_x_one_hot.argmax(1), hat_x_one_hot.shape)[1]
        # print("pred_class=\n", max_indexes_on_axis_1)

        for t in range(TIME_SLOTS_PER_BATCH):
            index = max_indexes_on_axis_1[t]
            s = QPSK_CANDIDATES[:, index]
            s = s.reshape([1, 2 * NUM_ANT, 1])
            s_batch = concatenate(s_batch, s)
        return get_bits(s_batch)
