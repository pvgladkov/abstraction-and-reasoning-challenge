import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch import nn
from torch.optim import Adam
from numpy.random import randint

from arc_utils import load_data, flattener, dump_json, get_logger
from arc_models import TaskSolver, evaluate, input_output_shape_is_same
import pickle
from arc_models import Conv1
from arc_img_utils import inp2img
import torch.nn.functional as F
from torch.utils.data import TensorDataset, RandomSampler, DataLoader, SequentialSampler


BASE_PATH = '/data/arc'
# BASE_PATH = '../input/abstraction-and-reasoning-challenge/'

DEBUG = True
N_EPOCH = 100
BUTCH_SIZE = 24
N_CHANNELS = 10
IM_H = 30
IM_W = 30


def pad(array):
    r = np.zeros((IM_H, IM_W), dtype=np.float)
    np_array = np.array(array)
    r[:np_array.shape[0], :np_array.shape[1]] = np_array
    return r


if __name__ == '__main__':

    logger = get_logger()

    train_tasks = load_data(BASE_PATH + '/training')
    evaluation_tasks = load_data(BASE_PATH + '/evaluation')
    test_tasks = load_data(BASE_PATH + '/test')

    ch = 10

    net = Conv1()
    net.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(net.parameters(), lr=0.1)

    all_inputs = []
    all_outputs = []

    for task in train_tasks:
        for sample in task['train']:
            all_inputs.append(inp2img(pad(sample['input'])))
            all_outputs.append(pad(sample['input']))

    x = torch.tensor(all_inputs, dtype=torch.float)
    y = torch.tensor(all_outputs, dtype=torch.long)
    dataset = TensorDataset(x, y)
    train_sampler = RandomSampler(dataset)
    data_loader = DataLoader(dataset, sampler=train_sampler, batch_size=BUTCH_SIZE, drop_last=True)

    for epoch in range(N_EPOCH):
        epoch_loss = 0
        num_examples = 0
        for i, (inputs, labels) in enumerate(tqdm(data_loader)):

            mask = np.ones((BUTCH_SIZE, N_CHANNELS, IM_H, IM_W), dtype=np.float)
            mask[:, :, randint(0, IM_H), randint(0, IM_W)] = 0
            mask[:, :, randint(0, IM_H), randint(0, IM_W)] = 0
            mask[:, :, randint(0, IM_H), randint(0, IM_W)] = 0
            mask[:, :, randint(0, IM_H), randint(0, IM_W)] = 0

            mask[:, randint(0, N_CHANNELS), randint(0, IM_H), randint(0, IM_W)] = 1
            mask[:, randint(0, N_CHANNELS), randint(0, IM_H), randint(0, IM_W)] = 1
            mask[:, randint(0, N_CHANNELS), randint(0, IM_H), randint(0, IM_W)] = 1
            mask[:, randint(0, N_CHANNELS), randint(0, IM_H), randint(0, IM_W)] = 1

            optimizer.zero_grad()
            outputs = net(inputs.cuda() * torch.tensor(mask, dtype=torch.float).cuda())
            loss = F.cross_entropy(outputs, labels.cuda())
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_examples += len(labels)
        logger.debug('epoch {}, loss {}'.format(epoch, round(epoch_loss / num_examples, 6)))

    torch.save(net.state_dict(), 'models_weights/structure_model.pkl')
