import numpy as np
import torch
from torch import nn
from torch.nn import Conv2d, Conv3d
from torch.optim import Adam
from arc.unet import UNet

from arc.utils import inp2img, flips, rotations2, TaskSolver, input_output_shape_is_same


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
        self.conv1 = Conv2d(in_channels=10, out_channels=2, kernel_size=5, padding=2, bias=True)
        self.conv2 = Conv3d(in_channels=2, out_channels=10, kernel_size=5, padding=2, bias=True)

    def forward(self, x):
        grey_x = self.conv1(x)
        grey_xx = torch.stack([grey_x[:, 0, :, :]] + 9*[grey_x[:, 1, :, :]], dim=1)
        assert grey_xx.shape[1] == 10
        stack_x = torch.stack([x, x-grey_xx], dim=1)
        return self.conv2(stack_x)


class TaskSolverConv1(TaskSolver):
    def __init__(self, logger, n_epoch=30):
        super(TaskSolverConv1, self).__init__(logger)
        self.net = None
        self.n_epoch = n_epoch

    def _net(self):
        return Conv2d(in_channels=10, out_channels=10, kernel_size=5, padding=2, bias=True).cuda()

    def convert_outputs(self, outputs):
        return outputs.squeeze(dim=0).cpu().numpy().argmax(0)

    def target(self, t):
        return t

    def train(self, task_train, debug=False):

        if not input_output_shape_is_same(task_train):
            return False

        self.net = self._net()

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
            sample_outputs.append([self.target(j) for j in output_rotations])

        max_loss = None
        losses_tries = 0
        patient = 2

        for epoch in range(self.n_epoch):
            epoch_loss = 0
            num_examples = 0
            for sample_input, sample_output in zip(sample_inputs, sample_outputs):
                inputs = torch.FloatTensor(sample_input).cuda()
                labels = torch.LongTensor(sample_output).cuda()

                # self.logger.debug(f'inputs {inputs.shape}, labels {labels.shape}')

                optimizer.zero_grad()
                try:
                    outputs = self.net(inputs)
                except Exception as e:
                    self.logger.debug(e)
                    return False

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
            self.net.eval()
            for sample in task_test:
                inputs = torch.FloatTensor(inp2img(sample['input'])).unsqueeze(dim=0).cuda()
                try:
                    outputs = self.net(inputs)
                except Exception as e:
                    self.logger.debug(e)
                    return [el['input'] for el in task_test]
                pred = self.convert_outputs(outputs)

                assert pred.shape == np.array(sample['input']).shape
                predictions.append(pred)

                # self.logger.debug(f'prediction inputs {inputs.shape}, outputs {outputs.shape}, pred {pred.shape}')

        return predictions


class TaskSolverConv2(TaskSolverConv1):

    def _net(self):
        return Conv2().cuda()

    def convert_outputs(self, outputs):
        return outputs.squeeze(dim=0).cpu().numpy().argmax(0).argmax(0)

    def target(self, t):
        return inp2img(t)


class TaskSolverUNet(TaskSolverConv1):

    def _net(self):
        return UNet(10, 10, bilinear=False).cuda()