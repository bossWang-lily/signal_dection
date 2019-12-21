import itertools
import multiprocessing
import os
import pathlib

import numpy as np

from global_settings import *

QPSK_CANDIDATE_SIZE = 2 ** (2 * NUM_ANT)
QPSK_CANDIDATES = np.array([x for x in itertools.product([1, -1], repeat=2 * NUM_ANT)]).T / np.sqrt(2)


def get_bits(s):
    return np.where(s < 0, 0, 1)


def mkdir(file_path):
    folder = os.path.dirname(file_path)
    if not os.path.exists(folder):
        os.makedirs(folder)


def mkfile(file_path):
    mkdir(file_path)
    filename = pathlib.Path(file_path)
    filename.touch(exist_ok=True)


def concatenate(total, part):
    return part if total is None else np.concatenate((total, part))


def random_h_batch():
    h_batch = None
    for _ in range(PACKETS_PER_BATCH):
        real = np.random.randn(NUM_ANT, NUM_ANT)
        imag = np.random.randn(NUM_ANT, NUM_ANT)
        h = np.row_stack(
            (
                np.column_stack((real, -imag)),
                np.column_stack((imag, real)),
            )
        )
        h = h.reshape([1, 2 * NUM_ANT, 2 * NUM_ANT])
        for t in range(TIME_SLOTS_PER_PACKET):
            h_batch = concatenate(h_batch, h)
    return h_batch


def random_s_batch():
    s_batch = None
    one_hot_batch = np.zeros([TIME_SLOTS_PER_BATCH, QPSK_CANDIDATE_SIZE])
    random_indexes = np.random.uniform(low=0, high=QPSK_CANDIDATE_SIZE, size=TIME_SLOTS_PER_BATCH)
    for t in range(TIME_SLOTS_PER_BATCH):
        i = int(random_indexes[t])
        one_hot_batch[t, i] = 1
        s = QPSK_CANDIDATES[:, i:i + 1]
        s = s.reshape([1, 2 * NUM_ANT, 1])
        s_batch = concatenate(s_batch, s)
    return s_batch, one_hot_batch


def gen_awgn_received_y(sir):
    power = 10 ** (sir / 10)
    h = np.sqrt(power / NUM_ANT) * random_h_batch()
    s, s_one_hot = random_s_batch()
    w = np.random.randn(1, 2 * NUM_ANT, 1)

    # [ R      [ R   -I    [ R       [ R
    #       =            @        +
    #  I ]       I   R ]     I ]       I ]
    y = h @ s + w
    return y, s, s_one_hot


class DataSet:
    flags = ["train", "test", "valid"]

    def __init__(self, sir: float, flag="train"):
        assert flag in DataSet.flags
        self.flag = flag
        self.sir = sir

    def __open_file(self, name, mode):
        file_name = "savedData/sir{}/{}/{}".format(self.sir, self.flag, name)
        mkfile(file_name)
        return open(file_name, mode)

    def gen_func(self, _idx):
        return gen_awgn_received_y(self.sir)

    def __open_all(self, mode):
        file_y = self.__open_file("y", mode)
        file_s = self.__open_file("s", mode)
        file_one_hot = self.__open_file("one_hot", mode)
        return file_y, file_s, file_one_hot

    def generate(self):
        file_y, file_s, file_one_hot = self.__open_all("wb")

        if self.flag == "train":
            total_batch = TRAIN_TOTAL_BATCH
        elif self.flag == "valid":
            total_batch = VALID_TOTAL_BATCH
        else:
            total_batch = TEST_TOTAL_BATCH

        if NUM_WORKERS > 0:
            pool = multiprocessing.pool.Pool(NUM_WORKERS, maxtasksperchild=MAX_TASKS_PER_CHILD)
        else:
            pool = multiprocessing.pool.Pool(maxtasksperchild=MAX_TASKS_PER_CHILD)
        idx = 0
        for ret_value in pool.imap(self.gen_func, range(total_batch)):
            print("{} set，batch {}/{}".format(self.flag, idx + 1, total_batch), end="\r")
            ret_value[0].astype(np.float32).tofile(file_y)
            ret_value[1].astype(np.float32).tofile(file_s)
            ret_value[2].astype(np.float32).tofile(file_one_hot)

            file_y.flush()
            file_s.flush()
            file_one_hot.flush()

            idx += 1
        pool.close()

        file_y.close()
        file_s.close()
        file_one_hot.close()

        print()

    def fetch(self):
        file_y, file_s, file_one_hot = self.__open_all("rb")
        if self.flag == "train":
            total_batch = TRAIN_TOTAL_BATCH
        elif self.flag == "test":
            total_batch = VALID_TOTAL_BATCH
        else:
            total_batch = TEST_TOTAL_BATCH

        for i in range(total_batch):
            file_y.seek(i * TIME_SLOTS_PER_BATCH * 2 * NUM_ANT)
            file_s.seek(i * TIME_SLOTS_PER_BATCH * 2 * NUM_ANT)
            file_one_hot.seek(i * TIME_SLOTS_PER_BATCH * QPSK_CANDIDATE_SIZE)

            y = np.fromfile(
                file_y,
                dtype=np.float32,
                count=TIME_SLOTS_PER_BATCH * 2 * NUM_ANT
            ).reshape([-1, 2 * NUM_ANT])

            # [RRRRIIII] -> [RIRIRIRI]
            y = y.reshape([-1, NUM_ANT, 2], order="F").reshape([-1, 2 * NUM_ANT])

            s = np.fromfile(
                file_s,
                dtype=np.float32,
                count=TIME_SLOTS_PER_BATCH * 2 * NUM_ANT
            ).reshape([-1, 2 * NUM_ANT, 1])

            one_hot = np.fromfile(
                file_one_hot,
                dtype=np.float32,
                count=TIME_SLOTS_PER_BATCH * QPSK_CANDIDATE_SIZE
            ).reshape([-1, QPSK_CANDIDATE_SIZE])

            yield y, s, one_hot

        file_y.close()
        file_s.close()
        file_one_hot.close()
