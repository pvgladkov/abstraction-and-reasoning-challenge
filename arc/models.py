import numpy as np
import torch
from torch import nn
from torch.nn import Conv2d, Conv3d
from torch.optim import Adam

from arc.img_utils import inp2img, rotations2, flips


class Conv3(nn.Module):
    def __init__(self):
        super(Conv3, self).__init__()
        self.conv2 = Conv3d(in_channels=10, out_channels=10, kernel_size=5, padding=2, bias=True)

    @staticmethod
    def mask(index, shape):
        z = np.full(shape, 0, dtype=np.uint8)
        z[:, index, :, :] = 1
        return torch.tensor(z, dtype=torch.float).cuda()

    def forward(self, x):
        t = []
        for idx in range(10):
            t.append(x * self.mask(idx, x.shape))

        stack_x = torch.stack(t, dim=1)
        return self.conv2(stack_x)


class Conv2(nn.Module):
    def __init__(self):
        super(Conv2, self).__init__()
        self.conv1 = Conv2d(in_channels=10, out_channels=2, kernel_size=7, padding=3, bias=True)
        self.conv2 = Conv3d(in_channels=2, out_channels=10, kernel_size=7, padding=3, bias=True)

    def forward(self, x):
        grey_x = self.conv1(x)
        grey_xx = torch.stack([grey_x[:, 0, :, :]] + 9*[grey_x[:, 1, :, :]], dim=1)
        assert grey_xx.shape[1] == 10
        stack_x = torch.stack([x, x-grey_xx], dim=1)
        return self.conv2(stack_x)


class TaskSolver:

    def __init__(self, logger):
        self.logger = logger

    def train(self, task_train, params):
        raise NotImplementedError

    def predict(self, task_test):
        raise NotImplementedError


class TaskSolverConv1(TaskSolver):
    def __init__(self, logger):
        super(TaskSolverConv1, self).__init__(logger)
        self.net = None

    def train(self, task_train, n_epoch=30, debug=False):

        self.net = Conv2d(in_channels=10, out_channels=10, kernel_size=3, padding=1, bias=True).cuda()

        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self.net.parameters(), lr=0.1)

        sample_inputs = []
        sample_outputs = []

        for sample in task_train:

            input_rotations = flips(sample['input'])
            output_rotations = flips(sample['output'])

            if len(sample['input']) == len(sample['input'][0]):
                input_rotations += rotations2(sample['input'])
                output_rotations += rotations2(sample['output'])

            sample_inputs.append([inp2img(j) for j in input_rotations])
            sample_outputs.append([j for j in output_rotations])

        max_loss = None
        losses_tries = 0
        patient = 2

        for epoch in range(n_epoch):
            epoch_loss = 0
            num_examples = 0
            for sample_input, sample_output in zip(sample_inputs, sample_outputs):
                inputs = torch.FloatTensor(sample_input).cuda()
                labels = torch.LongTensor(sample_output).cuda()

                # self.logger.debug(f'inputs {inputs.shape}, labels {labels.shape}')

                optimizer.zero_grad()
                outputs = self.net(inputs)

                # self.logger.debug(f'outputs {outputs.shape}')

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_examples += len(sample_output)

            if max_loss is None:
                max_loss = epoch_loss

            if debug and epoch % 5 == 0:
                self.logger.debug('epoch {}, loss {}'.format(epoch, round(epoch_loss / num_examples, 6)))

            if epoch_loss > max_loss:
                max_loss = epoch_loss
                losses_tries += 1

            if losses_tries > patient:
                self.logger.debug('early stopping')
                break

        return True

    def predict(self, task_test):
        predictions = []
        with torch.no_grad():
            for sample in task_test:
                inputs = torch.FloatTensor(inp2img(sample['input'])).unsqueeze(dim=0).cuda()
                outputs = self.net(inputs)
                pred = outputs.squeeze(dim=0).cpu().numpy().argmax(0)

                assert pred.shape == np.array(sample['input']).shape
                predictions.append(pred)

                # self.logger.debug(f'prediction inputs {inputs.shape}, outputs {outputs.shape}, pred {pred.shape}')

        return predictions


def calc_score(task_test, predict):
    return [int(np.equal(sample['output'], pred).all()) for sample, pred in zip(task_test, predict)]


def input_output_shape_is_same(task):
    return all([np.array(el['input']).shape == np.array(el['output']).shape for el in task['train']])
