import torch
from torch import nn as nn
from torch.nn import Conv2d, Conv3d, Dropout
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
import copy


class Conv2(nn.Module):
    def __init__(self):
        super(Conv2, self).__init__()
        self.net = nn.Sequential(
            Conv2d(in_channels=10, out_channels=10, kernel_size=5, padding=2),
            nn.BatchNorm2d(10),
            nn.ReLU(True),
            Conv2d(in_channels=10, out_channels=10, kernel_size=3, padding=1))

    def forward(self, x):
        return self.net(x)


class Conv1(nn.Module):
    def __init__(self):
        super(Conv1, self).__init__()
        self.conv1 = Conv2d(in_channels=10, out_channels=10, kernel_size=5, padding=2, bias=False)

    def forward(self, x):
        conv1_output = self.conv1(x)
        return conv1_output


class TaskSolver:
    def __init__(self, logger, base_net=None):
        self.net = None
        self.logger = logger
        self.base_net = base_net

    def train(self, task_train, n_epoch=100, debug=False):
        if self.base_net:
            self.net = nn.Sequential(
                copy.deepcopy(self.base_net),
                Conv1()).cuda()
        else:
            self.net = Conv2().cuda()

        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self.net.parameters(), lr=0.1)

        for epoch in range(n_epoch):
            epoch_loss = 0
            num_examples = 0
            for sample in task_train:
                inputs = torch.FloatTensor(inp2img(sample['input'])).unsqueeze(dim=0).cuda()
                labels = torch.LongTensor(sample['output']).unsqueeze(dim=0).cuda()

                # self.logger.debug(f'inputs {inputs.shape}, labels {labels.shape}')

                optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_examples += len(sample['output'])

            if debug and epoch % 5 == 0:
                self.logger.debug('epoch {}, loss {}'.format(epoch, round(epoch_loss / num_examples, 6)))

        return self

    def predict(self, task_test):
        predictions = []
        with torch.no_grad():
            for sample in task_test:
                inputs = torch.FloatTensor(inp2img(sample['input'])).unsqueeze(dim=0).cuda()
                outputs = self.net(inputs)
                pred = outputs.squeeze(dim=0).cpu().numpy().argmax(0)
                predictions.append(pred)

                # self.logger.debug(f'prediction inputs {inputs.shape}, outputs {outputs.shape}, pred {pred.shape}')

        return predictions


def calc_score(task_test, predict):
    return [int(np.equal(sample['output'], pred).all()) for sample, pred in zip(task_test, predict)]


def input_output_shape_is_same(task):
    return all([np.array(el['input']).shape == np.array(el['output']).shape for el in task['train']])


def evaluate(tasks, logger, base_net=None):
    ts = TaskSolver(logger, base_net)
    result = []
    predictions = []
    for i, task in enumerate(tqdm(tasks)):
        debug = False
        if i % 100 == 0:
            logger.debug(f'task {i}')
            debug = True

        if input_output_shape_is_same(task):
            ts.train(task['train'], debug=debug)
            pred = ts.predict(task['test'])
            score = calc_score(task['test'], pred)
        else:
            pred = [el['input'] for el in task['test']]
            score = [0] * len(task['test'])

        predictions.append(pred)
        result.append(score)

    return result, predictions


def build_base_net(tasks, logger):
    net = Conv2d(in_channels=10, out_channels=10, kernel_size=7, padding=3).cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(net.parameters(), lr=0.1)

    for i, task in enumerate(tqdm(tasks)):
        if not input_output_shape_is_same(task):
            continue

        for epoch in range(10):
            for sample in task['train']:
                inputs = torch.FloatTensor(inp2img(sample['input'])).unsqueeze(dim=0).cuda()
                labels = torch.LongTensor(sample['output']).unsqueeze(dim=0).cuda()

                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
            optimizer.step()

    return net


def inp2img(inp):
    inp = np.array(inp)
    img = np.full((10, inp.shape[0], inp.shape[1]), 0, dtype=np.uint8)
    for i in range(10):
        img[i] = (inp == i)
    return img
