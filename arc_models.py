import torch
from torch import nn as nn
from torch.nn import Conv2d, Conv3d
from torch.optim import Adam
from tqdm import tqdm
import numpy as np


class Conv2(nn.Module):
    def __init__(self):
        super(Conv2, self).__init__()
        self.conv1 = Conv2d(in_channels=10, out_channels=10, kernel_size=5, padding=2)
        self.conv2 = Conv2d(in_channels=10, out_channels=10, kernel_size=5, padding=2)

    def forward(self, x):
        conv1_output = self.conv1(x)
        conv2_output = self.conv2(conv1_output)
        return conv2_output


class Conv1(nn.Module):
    def __init__(self):
        super(Conv1, self).__init__()
        self.conv1 = Conv2d(in_channels=10, out_channels=10, kernel_size=5, padding=2)

    def forward(self, x):
        conv1_output = self.conv1(x)
        return conv1_output


class TaskSolver:

    def __init__(self, logger):
        self.net = None
        self.logger = logger

    def train(self, task_train, n_epoch=30, debug=False):
        self.net = Conv1().cuda()

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

            if debug:
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


def evaluate(tasks, logger):
    ts = TaskSolver(logger)
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


def inp2img(inp):
    inp = np.array(inp)
    img = np.full((10, inp.shape[0], inp.shape[1]), 0, dtype=np.uint8)
    for i in range(10):
        img[i] = (inp == i)
    return img
