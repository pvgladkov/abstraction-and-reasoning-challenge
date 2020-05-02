import torch
from torch import nn as nn
from torch.nn import Conv2d
from torch.optim import Adam
from tqdm import tqdm
import numpy as np


class BasicCNNModel(nn.Module):
    def __init__(self, conv_out_1, conv_out_2, inp_dim=(10, 10), outp_dim=(10, 10)):
        super(BasicCNNModel, self).__init__()

        self.conv_in = 3
        self.kernel_size = 3
        self.dense_in = conv_out_2

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.dense_1 = nn.Linear(self.dense_in, outp_dim[0] * outp_dim[1] * 10)

        if inp_dim[0] < 5 or inp_dim[1] < 5:
            self.kernel_size = 1

        self.conv2d_1 = nn.Conv2d(self.conv_in, conv_out_1, kernel_size=self.kernel_size)
        self.conv2d_2 = nn.Conv2d(conv_out_1, conv_out_2, kernel_size=self.kernel_size)

    def forward(self, x, outp_dim):
        x = torch.cat([x.unsqueeze(0)] * 3)
        x = x.permute((1, 0, 2, 3)).float()
        self.conv2d_1.in_features = x.shape[1]
        conv_1_out = self.relu(self.conv2d_1(x))
        self.conv2d_2.in_features = conv_1_out.shape[1]
        conv_2_out = self.relu(self.conv2d_2(conv_1_out))

        self.dense_1.out_features = outp_dim
        feature_vector, _ = torch.max(conv_2_out, 2)
        feature_vector, _ = torch.max(feature_vector, 2)
        logit_outputs = self.dense_1(feature_vector)

        out = []
        for idx in range(logit_outputs.shape[1] // 10):
            out.append(self.softmax(logit_outputs[:, idx * 10: (idx + 1) * 10]))
        return torch.cat(out, axis=1)


class TaskSolver:

    def __init__(self):
        self.net = None

    def train(self, task_train, n_epoch=30):
        self.net = Conv2d(in_channels=10, out_channels=10, kernel_size=5, padding=2).cuda()

        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self.net.parameters(), lr=0.1)

        for epoch in range(n_epoch):
            for sample in task_train:
                inputs = torch.FloatTensor(inp2img(sample['input'])).unsqueeze(dim=0).cuda()
                labels = torch.LongTensor(sample['output']).unsqueeze(dim=0).cuda()

                optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        return self

    def predict(self, task_test):
        predictions = []
        with torch.no_grad():
            for sample in task_test:
                inputs = torch.FloatTensor(inp2img(sample['input'])).unsqueeze(dim=0).cuda()
                outputs = self.net(inputs)
                pred = outputs.squeeze(dim=0).cpu().numpy().argmax(0)
                predictions.append(pred)

        return predictions


def calc_score(task_test, predict):
    return [int(np.equal(sample['output'], pred).all()) for sample, pred in zip(task_test, predict)]


def input_output_shape_is_same(task):
    return all([np.array(el['input']).shape == np.array(el['output']).shape for el in task['train']])


def evaluate(tasks):
    ts = TaskSolver()
    result = []
    predictions = []
    for task in tqdm(tasks):
        if input_output_shape_is_same(task):
            ts.train(task['train'])
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
