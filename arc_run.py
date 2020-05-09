import numpy as np
import pandas as pd
from tqdm import tqdm

from arc.utils import load_data, flattener, get_logger
from arc.models import TaskSolverConv1, input_output_shape_is_same, calc_score
from arc.trees import TaskSolverTree
import pickle

BASE_PATH = '/data/arc'

# BASE_PATH = '../input/abstraction-and-reasoning-challenge/'

DEBUG = True

logger = get_logger()


def make_prediction(tasks, solver):
    result = pd.Series()
    for idx, task in tqdm(tasks.items()):
        if input_output_shape_is_same(task):
            task_result = solver.train(task['train'])
            if task_result:
                pred = solver.predict(task['test'])
            else:
                pred = [el['input'] for el in task['test']]
        else:
            pred = [el['input'] for el in task['test']]

        for i, p in enumerate(pred):
            result[f'{idx}_{i}'] = flattener(np.array(p).tolist())

    return result


def evaluate(tasks, solver):
    result = []
    predictions = []
    for i, task in enumerate(tqdm(tasks)):

        if input_output_shape_is_same(task):
            task_result = solver.train(task['train'])
            if task_result:
                pred = solver.predict(task['test'])
                score = calc_score(task['test'], pred)
            else:
                pred = [el['input'] for el in task['test']]
                score = [0] * len(task['test'])
        else:
            pred = [el['input'] for el in task['test']]
            score = [0] * len(task['test'])

        predictions.append(pred)
        result.append(score)

    return result, predictions


if __name__ == '__main__':

    # task_solver = TaskSolverConv1(logger)
    task_solver = TaskSolverTree(logger)

    train_tasks = load_data(BASE_PATH + '/training')
    evaluation_tasks = load_data(BASE_PATH + '/evaluation')
    test_tasks = load_data(BASE_PATH + '/test')

    train_result, train_predictions = evaluate(train_tasks, task_solver)
    train_solved = [any(score) for score in train_result]

    total = sum([len(score) for score in train_result])
    logger.info(f"train solved : {sum(train_solved)} from {total} ({sum(train_solved) / total})")

    evaluation_result, evaluation_predictions = evaluate(evaluation_tasks, task_solver)
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
    logger.info(f"evaluation solved : {sum(evaluation_solved)} from {total} ({sum(evaluation_solved) / total})")

    submission = make_prediction(test_tasks, task_solver)
    submission = submission.reset_index()
    submission.columns = ['output_id', 'output']
    submission.to_csv('submission.csv', index=False)
