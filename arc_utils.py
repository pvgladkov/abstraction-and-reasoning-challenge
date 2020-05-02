import json
import os

import cv2
import numpy as np
from keras.utils import to_categorical
from torch.utils.data import Dataset
import logging
import sys
import pandas as pd
from os.path import join as path_join


def replace_values(a, d):
    return np.array([d.get(i, -1) for i in range(a.min(), a.max() + 1)])[a - a.min()]


def repeat_matrix(a, size):
    return np.concatenate([a]*((size // len(a)) + 1))[:size]


def get_new_matrix(X):
    if len(set([np.array(x).shape for x in X])) > 1:
        X = np.array([X[0]])
    return X


def get_outp(outp, dictionary=None, replace=True):
    if replace:
        outp = replace_values(outp, dictionary)

    outp_matrix_dims = outp.shape
    outp_probs_len = outp.shape[0]*outp.shape[1]*10
    outp = to_categorical(outp.flatten(),
                          num_classes=10).flatten()

    return outp, outp_probs_len, outp_matrix_dims


def transform_dim(inp_dim, outp_dim, test_dim):
    return (test_dim[0]*outp_dim[0]/inp_dim[0],
            test_dim[1]*outp_dim[1]/inp_dim[1])


def resize(x, test_dim, inp_dim):
    if inp_dim == test_dim:
        return x
    else:
        return cv2.resize(flt(x), inp_dim, interpolation=cv2.INTER_AREA)


def flt(x):
    return np.float32(x)


def npy(x):
    return x.cpu().detach().numpy()


def itg(x):
    return np.int32(np.round(x))


def flattener(pred):
    str_pred = str([row for row in pred])
    str_pred = str_pred.replace(', ', '')
    str_pred = str_pred.replace('[[', '|')
    str_pred = str_pred.replace('][', '|')
    str_pred = str_pred.replace(']]', '|')
    return str_pred


class ARCDataset(Dataset):
    def __init__(self, X, y, size, stage="train"):
        self.size = size

        self.X = get_new_matrix(X)
        self.X = repeat_matrix(self.X, self.size)

        self.stage = stage
        if self.stage == "train":
            self.y = get_new_matrix(y)
            self.y = repeat_matrix(self.y, self.size)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        inp = self.X[idx]
        if self.stage == "train":
            outp = self.y[idx]

        if idx != 0:
            rep = np.arange(10)
            orig = np.arange(10)
            np.random.shuffle(rep)
            dictionary = dict(zip(orig, rep))
            inp = replace_values(inp, dictionary)
            if self.stage == "train":
                outp, outp_probs_len, outp_matrix_dims = get_outp(outp, dictionary)

        if idx == 0:
            if self.stage == "train":
                outp, outp_probs_len, outp_matrix_dims = get_outp(outp, None, False)

        return inp, outp, outp_probs_len, outp_matrix_dims, self.y


def debug_json(names, x_test, y_test):
    for t_name, x, y in zip(names, x_test, y_test):
        dump_json(t_name, x, y)


def dump_json(name, x, y):
    data = {'train': [{'input': x, 'output': [list(map(int, list(a))) for a in y]}],
            'test': [{'input': x}]}

    with open('debug/{}.json'.format(name), 'w') as f:
        f.write(json.dumps(data))


def get_test_tasks(test_path):
    test_task_files = sorted(os.listdir(test_path))
    test_tasks = []
    names = []
    for task_file in test_task_files:
        name = task_file.split('.')[0]
        names.append(name)
        with open(str(test_path / task_file), 'r') as f:
            task = json.load(f)
            test_tasks.append(task)

    return test_tasks, names


def get_logger():
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root.addHandler(handler)
    return root


def inp2img(inp):
    inp = np.array(inp)
    img = np.full((10, inp.shape[0], inp.shape[1]), 0, dtype=np.uint8)
    for i in range(10):
        img[i] = (inp == i)
    return img


def load_data(path):
    tasks = pd.Series()
    for file_path in os.listdir(path):
        task_file = path_join(path, file_path)

        with open(task_file, 'r') as f:
            task = json.load(f)

        tasks[file_path[:-5]] = task
    return tasks
