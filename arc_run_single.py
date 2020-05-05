import pickle
from argparse import ArgumentParser

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from arc_img_utils import rotations, inp2img
from arc_models import Conv2
from arc_utils import load_data, get_logger

BASE_PATH = '/data/arc'

# BASE_PATH = '../input/abstraction-and-reasoning-challenge/'

DEBUG = True

logger = get_logger()


if __name__ == '__main__':
    train_tasks = load_data(BASE_PATH + '/training')
    evaluation_tasks = load_data(BASE_PATH + '/evaluation')

    parser = ArgumentParser()
    parser.add_argument('i_task', type=int)
    args = parser.parse_args()

    i_task = args.i_task

    n_epoch = 200
    task = evaluation_tasks[i_task]

    net = Conv2().cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(net.parameters(), lr=0.1)

    epoch_predictions = {}

    for i, epoch in enumerate(range(n_epoch)):
        epoch_loss = 0
        num_examples = 0

        for sample in task['train']:

            input_rotations = rotations(sample['input'])
            output_rotations = rotations(sample['output'])

            inputs = torch.FloatTensor([inp2img(j) for j in input_rotations]).cuda()
            labels = torch.LongTensor([inp2img(j) for j in output_rotations]).cuda()

            logger.debug(f'inputs {inputs.shape}, labels {labels.shape}')

            optimizer.zero_grad()
            outputs = net(inputs)

            loss = criterion(outputs, labels)

            epoch_loss += loss.item()
            num_examples += len(labels)

            loss.backward()
            optimizer.step()

        logger.debug('epoch {}, loss {}'.format(epoch, round(epoch_loss / num_examples, 6)))

        if i % 1 == 0:
            predictions = []
            with torch.no_grad():
                for sample in task['test']:
                    inputs = torch.FloatTensor(inp2img(sample['input'])).unsqueeze(dim=0).cuda()
                    outputs = net(inputs)
                    pred = outputs.squeeze(dim=0).cpu().numpy().argmax(0).argmax(0)

                    assert pred.shape == np.array(sample['input']).shape
                    predictions.append(pred)

            epoch_predictions[i] = predictions

        with open('pkl/single_run/epoch_predictions.pkl', 'w+b') as f:
            pickle.dump(epoch_predictions, f)

        with open('pkl/single_run/task.pkl', 'w+b') as f:
            pickle.dump(task, f)