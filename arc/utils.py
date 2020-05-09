import json
import os

import cv2
import numpy as np
import logging
import sys
import pandas as pd
from os.path import join


def flattener(pred):
    str_pred = str([row for row in pred])
    str_pred = str_pred.replace(', ', '')
    str_pred = str_pred.replace('[[', '|')
    str_pred = str_pred.replace('][', '|')
    str_pred = str_pred.replace(']]', '|')
    return str_pred


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
        task_file = join(path, file_path)

        with open(task_file, 'r') as f:
            task = json.load(f)

        tasks[file_path[:-5]] = task
    return tasks


def pad(array, high, width):
    r = np.zeros((high, width), dtype=np.float)
    np_array = np.array(array)
    r[:np_array.shape[0], :np_array.shape[1]] = np_array
    return r


def inp2grey(inp, exp=True):
    inp = np.array(inp)
    b_inp = (inp > 0).astype(np.int)
    if exp:
        return np.expand_dims(b_inp, 0)
    return b_inp


def inp2img(inp):
    inp = np.array(inp)
    img = np.full((10, inp.shape[0], inp.shape[1]), 0, dtype=np.uint8)
    for i in range(10):
        img[i] = (inp == i)
    return img


def flips(matrix):
    matrix = np.array(matrix)
    return [matrix, np.fliplr(matrix), np.flipud(matrix)]


def rotations2(matrix):
    matrix = np.array(matrix)
    return [matrix, np.rot90(matrix, 1), np.rot90(matrix, 2), np.rot90(matrix, 3)]


class TaskSolver:

    def __init__(self, logger):
        self.logger = logger

    def train(self, task_train, params):
        raise NotImplementedError

    def predict(self, task_test):
        raise NotImplementedError


def input_output_shape_is_same(task):
    return all([np.array(el['input']).shape == np.array(el['output']).shape for el in task])