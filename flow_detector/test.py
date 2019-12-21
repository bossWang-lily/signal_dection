from model import *


def generate_data(data_flag, **params):
    train_set = DataSet(io_flag=IO_FLAG_TRAIN, data_flag=data_flag, **params)
    train_set.produce_all()

    valid_set = DataSet(io_flag=IO_FLAG_VALID, data_flag=data_flag, **params)
    valid_set.produce_all()

    test_set = DataSet(io_flag=IO_FLAG_TEST, data_flag=data_flag, **params)
    test_set.produce_all()


def train_model(data_flag, **params):
    train_set = DataSet(io_flag=IO_FLAG_TRAIN, data_flag=data_flag, **params)
    valid_set = DataSet(io_flag=IO_FLAG_VALID, data_flag=data_flag, **params)

    model = Detector(detpth=16)
    model.train(train_set, valid_set)
    model.close()


if __name__ == "__main__":
    # generate_data(data_flag=DATA_LFAG_GAUSS_MIXTURE, sir=10)
    train_model(data_flag=DATA_FLAG_POISSON_FIELD, sir=10, lam=10, length=100, alpha=2)
    # train_model(data_flag=DATA_FLAG_ALPHA_STABLE, sir=10, alpha=1.5, beta=0, mu=0, sigma=1)
