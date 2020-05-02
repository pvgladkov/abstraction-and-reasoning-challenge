import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from arc_utils import load_data, flattener, dump_json
from arc_models import TaskSolver, evaluate, input_output_shape_is_same

BASE_PATH = '/data/arc/'

# BASE_PATH = '../input/abstraction-and-reasoning-challenge/'

DEBUG = True


def make_prediction(tasks, debug=False):
    ts = TaskSolver()
    result = pd.Series()
    for idx, task in tqdm(tasks.iteritems()):
        if input_output_shape_is_same(task):
            ts.train(task['train'])
            pred = ts.predict(task['test'])
        else:
            pred = [el['input'] for el in task['test']]

        for i, p in enumerate(pred):
            result[f'{idx}_{i}'] = flattener(np.array(p).tolist())
            if debug:
                dump_json(f'{idx}_{i}', task['test'][i]['input'], p)

    return result


if __name__ == '__main__':
    train_tasks = load_data(BASE_PATH + '/training')
    evaluation_tasks = load_data(BASE_PATH + 'evaluation')
    test_tasks = load_data(BASE_PATH + '/test')

    train_result, train_predictions = evaluate(train_tasks)
    train_solved = [any(score) for score in train_result]

    total = sum([len(score) for score in train_result])
    print(f"solved : {sum(train_solved)} from {total} ({sum(train_solved) / total})")

    evaluation_result, evaluation_predictions = evaluate(evaluation_tasks)
    evaluation_solved = [any(score) for score in evaluation_result]

    total = sum([len(score) for score in evaluation_result])
    print(f"solved : {sum(evaluation_solved)} from {total} ({sum(evaluation_solved) / total})")

    submission = make_prediction(test_tasks, DEBUG)
    submission = submission.reset_index()
    submission.columns = ['output_id', 'output']
    submission.to_csv('submission.csv', index=False)
