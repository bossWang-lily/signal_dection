import itertools
import os
import pathlib

from multiprocessing import pool

import numpy as np

NUM_ANT = 4  # number of antennas at both transmitter and receiver

PACKET_SIZE = 576  # number of bits per packet
TIME_SLOTS_PER_PACKET = int(PACKET_SIZE / (2 * NUM_ANT))
PACKETS_PER_BATCH = 10
TIMES_SLOTS_PER_BATCH = PACKETS_PER_BATCH * TIME_SLOTS_PER_PACKET

TRAIN_TOTAL_BATCH = 10000
VALID_TOTAL_BATCH = 2000
TEST_TOTAL_BATCH = 500

IO_FLAG_TRAIN = 0
IO_FLAG_VALID = 1
IO_FLAG_TEST = 2

DATA_FLAG_GAUSS = 0
DATA_FLAG_POISSON_FIELD = 1
DATA_FLAG_ALPHA_STABLE = 2
DATA_LFAG_GAUSS_MIXTURE = 3

NUM_WORKERS = 20
MAX_TASKS_PER_CHILD = 100

LATENT_DIM = 64
NUM_SAMPLES = 1
MAX_EPOCHS = 100
VALID_EVERY_EPOCHES = 1

'''
QPSK SIGNALS
00->np.exp(np.pi*1/4j)
01->np.exp(np.pi*3/4j)
11->np.exp(np.pi*5/4j)
10->np.exp(np.pi*7/4j)
'''
QPSK_CANDIDATE_SIZE = 2 ** (2 * NUM_ANT)
QPSK_CANDIDATES = np.array([x for x in itertools.product([1, -1], repeat=2 * NUM_ANT)]).T / np.sqrt(2)


def get_power(w, sir):
    rate_now = 1 / np.mean(w ** 2)
    expected_rate = 10 ** (sir / 10)
    fix_rate = expected_rate / rate_now
    fix_db = 10.0 * np.log10(fix_rate)
    power = 10.0 ** (fix_db / 10.0)
    return power


def get_bits(x):
    return np.where(x < 0, 0, 1)


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


def csi():
    real = np.random.randn(NUM_ANT, NUM_ANT)
    imag = np.random.randn(NUM_ANT, NUM_ANT)
    h = np.row_stack(
        (
            np.column_stack((real, -imag)),
            np.column_stack((imag, real)),
        )
    )
    return h


def csi_batch():
    h_batch = None
    for _ in range(PACKETS_PER_BATCH):
        h = csi().reshape([1, 2 * NUM_ANT, 2 * NUM_ANT])
        for _ in range(TIME_SLOTS_PER_PACKET):
            h_batch = concatenate(h_batch, h)
    return h_batch


def signal_batch():
    s_batch = None
    one_hot = np.zeros([TIMES_SLOTS_PER_BATCH, QPSK_CANDIDATE_SIZE])
    random_indexes = np.random.uniform(low=0, high=QPSK_CANDIDATE_SIZE, size=TIMES_SLOTS_PER_BATCH)
    for t in range(TIMES_SLOTS_PER_BATCH):
        i = int(random_indexes[t])
        one_hot[t, i] = 1
        s = QPSK_CANDIDATES[:, i:i + 1].reshape([1, 2 * NUM_ANT, 1])
        s_batch = concatenate(s_batch, s)
    return s_batch, one_hot


def random_distance(n, length):
    x = np.random.uniform(-1, 1, [n, 1, 1]) * length / 2
    y = np.random.uniform(-1, 1, [n, 1, 1]) * length / 2
    return np.sqrt(x ** 2 + y ** 2)


def stb_generator(alpha, beta, mu, sigma, size):
    """
    SaS稳定分布随机变量发生器：该函数产生任意的1维稳定过程
    :param size: 生成数据的尺寸
    :param alpha: (1,2]，特征指数，它决定了稳定分布的概率密度函数拖尾厚度。它的值越小，分布的拖尾也就越厚，所以分布的冲击性越强，即偏离中值的样本个数越多。
    alpha=2退化为高斯, alpha=1且beta=0时退化为柯西。
    :param beta: [-1,1]，偏斜参数(Skewness parameter)，它决定了分布的对称程度，β>0和好β<0分别对应分布的左偏和右偏
    :param sigma: 大于0，噪声强度，尺度参数。它是关于分布样本偏离其均值的一种度量，其意义类似于高斯分布时的方差。实际上，在高斯分布情况下γ为方差的两倍。
    :param mu: 任意实数，位置参数，表征了概率密度函数在X轴的偏移，表示分布的均值(1<α≤2)或中值(0<α≤1)。
    :return:
    """
    if alpha > 2:
        alpha = 2
    elif alpha <= 0:
        alpha = 0.01

    if beta > 1:
        beta = 1
    elif beta < -1:
        beta = -1

    if sigma <= 0:
        sigma = 0.01

    # 产生(-pi / 2, pi / 2) 的均匀分布随机变量
    rand_uni = (np.random.uniform(size=size) - 0.5) * np.pi
    # 产生均值为1的指数分布随机变量
    rand_exp = np.random.exponential(size=size)

    # 分布变换，产生S2(alpha,1,beta2,0)的基本分布
    if alpha != 1:
        ki = alpha if alpha < 1 else alpha - 2
        if alpha != 2:
            beta2 = 2 * np.arctan(beta * np.tan(np.pi * alpha / 2)) / (np.pi * ki)
        else:
            beta2 = beta
        # 求gama0
        gama0 = -(np.pi / 2) * beta2 * ki / alpha

        a = np.sin(alpha * (rand_uni - gama0))
        b = (np.cos(rand_uni - alpha * (rand_uni - gama0)) / rand_exp) ** ((1 - alpha) / alpha)
        c = np.cos(rand_uni) ** (1 / alpha)
        sta_proce = (a * b) / c
    else:
        beta2 = beta
        a = np.pi / 2 + beta2 * rand_uni
        b = rand_exp * np.cos(rand_uni) / (np.pi / 2 + beta2 * rand_uni)
        sta_proce = a * np.arctan(rand_uni) - beta2 * np.log(b)

    # 分布变换，产生任意分布
    # 设定1维任意稳定过程数据
    if alpha != 1:
        # sigma的变换，变换成S2参数系
        sigma2 = sigma * (1 + (beta * np.tan(np.pi * alpha / 2)) ** 2) ** (1 / (2 * alpha))
        return sta_proce * sigma2 + mu
    else:
        # sigma的变换
        sigma2 = sigma * 2 / np.pi
        # 修正mu
        mc = beta * sigma2 * np.log(sigma2)
        return sta_proce * sigma2 + mu + mc


def ppp_batch(lam, length, alpha):
    w_batch = None
    for _ in range(TIMES_SLOTS_PER_BATCH):
        # number of interfering sources \sim poisson(lam) distribution
        n_sources = np.sum(np.random.poisson(lam, 1))
        if n_sources == 0:
            w = np.zeros([1, 2 * NUM_ANT, 1])
        else:
            # Channel fading factors of interfering sources ∼ Nakagami-m
            h = np.random.randn(n_sources, 1, 1)
            # Distances to interfering sources
            r = random_distance(n_sources, length)
            # Signals of interfering sources
            s = np.cos(np.random.rand(n_sources, 2 * NUM_ANT, 1) * 2 * np.pi)
            w = np.sum(h * (r ** (-alpha / 2)) * s, axis=0, keepdims=True)
            w = w.reshape([1, 2 * NUM_ANT, 1])
        w_batch = concatenate(w_batch, w)
    return w_batch


def gauss_mixture_batch():
    w_batch = None
    for _ in range(TIMES_SLOTS_PER_BATCH):
        v = np.random.rand()
        if v < 0.33:
            w = np.random.normal(loc=4, scale=4, size=[1, 2 * NUM_ANT, 1])
        elif v < 0.66:
            w = np.random.normal(loc=8, scale=4, size=[1, 2 * NUM_ANT, 1])
        else:
            w = np.random.normal(loc=12, scale=4, size=[1, 2 * NUM_ANT, 1])
        w_batch = concatenate(w_batch, w)
    return w_batch


def make_all_batch_gauss(sir):
    w = np.random.randn(TIMES_SLOTS_PER_BATCH, 2 * NUM_ANT, 1)

    power = get_power(w, sir)
    h = np.sqrt(power / NUM_ANT) * csi_batch()
    s, one_hot = signal_batch()

    y = h @ s + w

    s_mld = mld_detect_batch(y, h)
    w_mld = y - h @ s_mld

    return y, h, s, one_hot, w, s_mld, w_mld


def make_all_batch_gauss_mixture(sir):
    w = gauss_mixture_batch()

    power = get_power(w, sir)
    h = np.sqrt(power / NUM_ANT) * csi_batch()
    s, one_hot = signal_batch()

    y = h @ s + w

    s_mld = mld_detect_batch(y, h)
    w_mld = y - h @ s_mld

    return y, h, s, one_hot, w, s_mld, w_mld


def make_all_batch_alpha_stable(sir, alpha, beta, mu, sigma):
    w = stb_generator(alpha=alpha, beta=beta, mu=mu, sigma=sigma, size=[TIMES_SLOTS_PER_BATCH, 2 * NUM_ANT, 1])

    power = get_power(w, sir)
    h = np.sqrt(power / NUM_ANT) * csi_batch()
    s, one_hot = signal_batch()

    y = h @ s + w

    s_mld = mld_detect_batch(y, h)
    w_mld = y - h @ s_mld

    return y, h, s, one_hot, w, s_mld, w_mld


def make_all_batch_poisson_field(sir, lam, length, alpha):
    w = ppp_batch(lam, length, alpha)

    power = get_power(w, sir)
    h = np.sqrt(power / NUM_ANT) * csi_batch()
    s, one_hot = signal_batch()

    y = h @ s + w

    s_mld = mld_detect_batch(y, h)
    w_mld = y - h @ s_mld

    return y, h, s, one_hot, w, s_mld, w_mld


def mld_detect_batch(y, h):
    distances = np.sum(np.square(y - h @ QPSK_CANDIDATES), axis=1)
    min_indexes = np.unravel_index(distances.argmin(1), distances.shape)[1]

    x_mld = None
    for t in range(TIMES_SLOTS_PER_BATCH):
        index = min_indexes[t]
        s = QPSK_CANDIDATES[:, index]
        s = s.reshape([1, 2 * NUM_ANT, 1])
        x_mld = concatenate(x_mld, s)
    return x_mld


def test_conventional_mld_gauss(sir):
    err_mld = 0.0
    total = 0.0
    idx = 0
    while idx < 1000:
        y, h, x, one_hot, w, x_mld, w_mld = make_all_batch_gauss(sir)
        bits = get_bits(x)
        bits_mld = get_bits(x_mld)
        err_mld += len(np.argwhere(bits_mld != bits))
        total += bits.size
        print("error_bits={} batch={}".format(err_mld, idx + 1), end="\r")
        idx += 1

    ber = err_mld / total
    print("")
    print("Gauss MLD BER={:e}".format(ber))


def test_conventional_mld_ppp(sir, lam, length, alpha):
    err_mld = 0.0
    total = 0.0
    idx = 0
    while idx < 1000:
        y, h, x, one_hot, w, x_mld, w_mld = make_all_batch_poisson_field(sir, lam, length, alpha)
        bits = get_bits(x)
        bits_mld = get_bits(x_mld)
        err_mld += len(np.argwhere(bits_mld != bits))
        total += bits.size
        print("error_bits={} batch={}".format(err_mld, idx + 1), end="\r")
        idx += 1

    ber = err_mld / total
    print("")
    print("PPP MLD BER={:e}".format(ber))


def test_conventional_mld_stb(sir, alpha, beta, mu, sigma):
    print("sir={} alpha={} beta={} mu={} sigma={}".format(sir, alpha, beta, mu, sigma))
    err_mld = 0.0
    total = 0.0
    idx = 0
    while idx < 1000:
        y, h, x, one_hot, w, x_mld, w_mld = make_all_batch_alpha_stable(sir, alpha, beta, mu, sigma)
        bits = get_bits(x)
        bits_mld = get_bits(x_mld)
        err_mld += len(np.argwhere(bits_mld != bits))
        total += bits.size
        print("error_bits={} batch={}".format(err_mld, idx + 1), end="\r")
        idx += 1

    ber = err_mld / total
    print("")
    print("alpha-stable MLD BER={:e}".format(ber))


def get_data_description(data_flag, params):
    if data_flag == DATA_FLAG_GAUSS:
        description = "gauss"
    elif data_flag == DATA_FLAG_POISSON_FIELD:
        description = "ppp"
    elif data_flag == DATA_FLAG_ALPHA_STABLE:
        description = "alpha_stable"
    elif data_flag == DATA_LFAG_GAUSS_MIXTURE:
        description = "gauss_mixture_model"
    else:
        raise NotImplementedError("Unknown data_flag={}".format(data_flag))
    for k in params:
        description += "_{}{}".format(k, params[k])
    return description


class DataSet:
    def __init__(self, io_flag, data_flag, **params):
        self.io_flag = io_flag
        self.data_flag = data_flag
        self.params = params

    def __get_file_name(self, name):
        description = get_data_description(self.data_flag, self.params)
        if self.io_flag == IO_FLAG_TRAIN:
            file_name = "savedData/ant{}/{}/train/{}".format(NUM_ANT, description, name)
        elif self.io_flag == IO_FLAG_VALID:
            file_name = "savedData/ant{}/{}/valid/{}".format(NUM_ANT, description, name)
        elif self.io_flag == IO_FLAG_TEST:
            file_name = "savedData/ant{}/{}/test/{}".format(NUM_ANT, description, name)
        else:
            raise NotImplementedError("Unknown io_flag={}".format(self.io_flag))

        return file_name

    def __open_file(self, name, mode):
        file_name = self.__get_file_name(name)
        mkfile(file_name)
        return open(file_name, mode)

    def __open_files(self, mode):
        file_y = self.__open_file("y", mode)
        file_h = self.__open_file("h", mode)
        file_s = self.__open_file("s", mode)
        file_one_hot = self.__open_file("one_hot", mode)
        file_w = self.__open_file("w", mode)
        file_s_mld = self.__open_file("s_mld", mode)
        file_w_mld = self.__open_file("w_mld", mode)
        return file_y, file_h, file_s, file_one_hot, file_w, file_s_mld, file_w_mld

    def __delete_file(self, name):
        file_name = self.__get_file_name(name)
        if os.path.exists(file_name):
            os.remove(file_name)

    def delete_all(self):
        self.__delete_file("y")
        self.__delete_file("h")
        self.__delete_file("s")
        self.__delete_file("one_hot")
        self.__delete_file("w")
        self.__delete_file("s_mld")
        self.__delete_file("w_mld")

    def produce_func(self, _idx):
        if self.data_flag == DATA_FLAG_GAUSS:
            return make_all_batch_gauss(**self.params)
        elif self.data_flag == DATA_FLAG_POISSON_FIELD:
            return make_all_batch_poisson_field(**self.params)
        elif self.data_flag == DATA_FLAG_ALPHA_STABLE:
            return make_all_batch_alpha_stable(**self.params)
        elif self.data_flag == DATA_LFAG_GAUSS_MIXTURE:
            return make_all_batch_gauss_mixture(**self.params)
        else:
            raise NotImplementedError("Unknown data_flag={}".format(self.data_flag))

    def produce_all(self):
        f_y, f_h, f_s, f_one_hot, f_w, f_s_mld, f_w_mld = self.__open_files("wb")

        if self.io_flag == 0:
            total_batch = TRAIN_TOTAL_BATCH
        elif self.io_flag == 1:
            total_batch = VALID_TOTAL_BATCH
        else:
            total_batch = TEST_TOTAL_BATCH

        proc_pool = pool.Pool(NUM_WORKERS, maxtasksperchild=MAX_TASKS_PER_CHILD)
        idx = 0
        for ret_value in proc_pool.imap(self.produce_func, range(total_batch)):
            print("Producing batch {}/{}".format(idx + 1, total_batch), end="\r")
            ret_value[0].astype(np.float32).tofile(f_y)
            ret_value[1].astype(np.float32).tofile(f_h)
            ret_value[2].astype(np.float32).tofile(f_s)
            ret_value[3].astype(np.float32).tofile(f_one_hot)
            ret_value[4].astype(np.float32).tofile(f_w)
            ret_value[5].astype(np.float32).tofile(f_s_mld)
            ret_value[6].astype(np.float32).tofile(f_w_mld)

            f_y.flush()
            f_h.flush()
            f_s.flush()
            f_one_hot.flush()
            f_w.flush()
            f_s_mld.flush()
            f_w_mld.flush()

            idx += 1
        print("")

        proc_pool.close()

        f_y.close()
        f_h.close()
        f_s.close()
        f_one_hot.close()
        f_w.close()
        f_s_mld.close()
        f_w_mld.close()

    def fetch(self):
        f_y, f_h, f_s, f_one_hot, f_w, f_s_mld, f_w_mld = self.__open_files(
            "rb")
        if self.io_flag == 0:
            total_batch = TRAIN_TOTAL_BATCH
        elif self.io_flag == 1:
            total_batch = VALID_TOTAL_BATCH
        else:
            total_batch = TEST_TOTAL_BATCH

        for i in range(total_batch):
            f_y.seek(i * TIMES_SLOTS_PER_BATCH * 2 * NUM_ANT * 1)
            f_h.seek(i * TIMES_SLOTS_PER_BATCH * 2 * NUM_ANT * 2 * NUM_ANT)
            f_s.seek(i * TIMES_SLOTS_PER_BATCH * 2 * NUM_ANT * 1)
            f_one_hot.seek(i * TIMES_SLOTS_PER_BATCH * QPSK_CANDIDATE_SIZE)
            f_w.seek(i * TIMES_SLOTS_PER_BATCH * 2 * NUM_ANT * 1)
            f_s_mld.seek(i * TIMES_SLOTS_PER_BATCH * 2 * NUM_ANT * 1)
            f_w_mld.seek(i * TIMES_SLOTS_PER_BATCH * 2 * NUM_ANT * 1)

            y = np.fromfile(
                f_y,
                dtype=np.float32,
                count=TIMES_SLOTS_PER_BATCH * 2 * NUM_ANT * 1
            ).reshape([-1, 2 * NUM_ANT, 1])

            h = np.fromfile(
                f_h,
                dtype=np.float32,
                count=TIMES_SLOTS_PER_BATCH * 2 * NUM_ANT * 2 * NUM_ANT
            ).reshape([-1, 2 * NUM_ANT, 2 * NUM_ANT])

            s = np.fromfile(
                f_s,
                dtype=np.float32,
                count=TIMES_SLOTS_PER_BATCH * 2 * NUM_ANT * 1
            ).reshape([-1, 2 * NUM_ANT, 1])

            one_hot = np.fromfile(
                f_one_hot,
                dtype=np.float32,
                count=TIMES_SLOTS_PER_BATCH * QPSK_CANDIDATE_SIZE
            ).reshape([-1, QPSK_CANDIDATE_SIZE])

            w = np.fromfile(
                f_w,
                dtype=np.float32,
                count=TIMES_SLOTS_PER_BATCH * 2 * NUM_ANT * 1
            ).reshape([-1, 2 * NUM_ANT, 1])

            s_mld = np.fromfile(
                f_s_mld,
                dtype=np.float32,
                count=TIMES_SLOTS_PER_BATCH * 2 * NUM_ANT * 1
            ).reshape([-1, 2 * NUM_ANT, 1])

            w_mld = np.fromfile(
                f_w_mld,
                dtype=np.float32,
                count=TIMES_SLOTS_PER_BATCH * 2 * NUM_ANT * 1
            ).reshape([-1, 2 * NUM_ANT, 1])

            yield y, h, s, one_hot, w, s_mld, w_mld

        f_y.close()
        f_h.close()
        f_s.close()
        f_one_hot.close()
        f_w.close()
        f_s_mld.close()
        f_w_mld.close()
