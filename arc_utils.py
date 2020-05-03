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


def load_data(path):
    tasks = pd.Series()
    for file_path in os.listdir(path):
        task_file = path_join(path, file_path)

        with open(task_file, 'r') as f:
            task = json.load(f)

        tasks[file_path[:-5]] = task
    return tasks
