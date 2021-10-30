from torchvision.datasets import DatasetFolder
from torchvision.datasets import MNIST


support_list = (
    DatasetFolder,
    MNIST
)


def check(dataset):
    return isinstance(dataset, support_list)


class Base(object):
    def __init__(self, train_dataset, test_dataset, model, loss, schedule=None):
        assert isinstance(train_dataset, support_list), 'train_dataset is an unsupported dataset type, train_dataset should be a subclass of our support list.'
        self.train_dataset = train_dataset

        assert isinstance(test_dataset, support_list), 'test_dataset is an unsupported dataset type, test_dataset should be a subclass of our support list.'
        self.test_dataset = test_dataset
        self.model = model
        self.loss = loss
        self.schedule = schedule
    
    def train(self, schedule=None):
        if schedule is None:
            schedule = self.schedule
        if schedule is None:
            raise AttributeError("Training schedule is None, please check your schedule setting.")
        

