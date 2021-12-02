from utils.utils import DATA_DIR


class Dataloader:
    DATA_DIR = DATA_DIR

    def __init__(self):
        pass

    def get_train_epoch(self):
        raise NotImplementedError()

    def get_test_epoch(self):
        raise NotImplementedError()

    def get_train_len(self):
        raise NotImplementedError()

    def get_test_len(self):
        raise NotImplementedError()
