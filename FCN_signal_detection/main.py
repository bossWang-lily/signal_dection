from nn import *


def generate_data(sir):
    train_set = DataSet(sir=sir, flag="train")
    train_set.generate()

    valid_set = DataSet(sir=sir, flag="valid")
    valid_set.generate()

    test_set = DataSet(sir=sir, flag="test")
    test_set.generate()


def train_model(sir):
    model = DNN(sir=sir)
    model.train()
    model.close()


if __name__ == "__main__":
    # generate_data(sir=10)
    train_model(sir=10)
