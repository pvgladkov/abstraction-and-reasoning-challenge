import torch
from torch import nn as nn


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