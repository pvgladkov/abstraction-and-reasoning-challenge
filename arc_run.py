import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from arc_utils import load_data, flattener, dump_json, get_logger
from arc_models import TaskSolver, evaluate, input_output_shape_is_same, build_base_net
import pickle

BASE_PATH = '/data/arc'

# BASE_PATH = '../input/abstraction-and-reasoning-challenge/'

DEBUG = True

logger = get_logger()


def make_prediction(tasks, logger, base_net=None, debug=False):
    ts = TaskSolver(logger, base_net)
    result = pd.Series()
    for idx, task in tqdm(tasks.items()):
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
    evaluation_tasks = load_data(BASE_PATH + '/evaluation')
    test_tasks = load_data(BASE_PATH + '/test')

    # base_net = build_base_net(train_tasks, logger)
    base_net = None

    train_result, train_predictions = evaluate(train_tasks, logger, base_net)
    train_solved = [any(score) for score in train_result]

    total = sum([len(score) for score in train_result])
    logger.info(f"solved : {sum(train_solved)} from {total} ({sum(train_solved) / total})")

    evaluation_result, evaluation_predictions = evaluate(evaluation_tasks, logger, base_net)
    evaluation_solved = [any(score) for score in evaluation_result]

    if DEBUG:
        with open('pkl/evaluation_tasks.pkl', 'w+b') as f:
            pickle.dump(evaluation_tasks, f)
        with open('pkl/evaluation_result.pkl', 'w+b') as f:
            pickle.dump(evaluation_result, f)
        with open('pkl/evaluation_predictions.pkl', 'w+b') as f:
            pickle.dump(evaluation_predictions, f)
        with open('pkl/evaluation_solved.pkl', 'w+b') as f:
            pickle.dump(evaluation_solved, f)

    total = sum([len(score) for score in evaluation_result])
    logger.info(f"solved : {sum(evaluation_solved)} from {total} ({sum(evaluation_solved) / total})")

    submission = make_prediction(test_tasks, logger, base_net, DEBUG)
    submission = submission.reset_index()
    submission.columns = ['output_id', 'output']
    submission.to_csv('submission.csv', index=False)