import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

from arc.colors import TaskSolverColor
from arc.sophisticated_trees import StackedTaskSolver
from arc.trees import TaskSolverTree
from arc.utils import load_data, flattener, get_logger

BASE_PATH = '/data/arc'

# BASE_PATH = '../input/abstraction-and-reasoning-challenge/'

DEBUG = True

logger = get_logger()


def make_prediction(tasks, solver):
    result = pd.Series()
    for idx, task in tqdm(tasks.items()):
        task_result = solver.train(task['train'])
        if task_result:
            pred = solver.predict(task['test'])
        else:
            pred = [el['input'] for el in task['test']]

        for i, p in enumerate(pred):
            result[f'{idx}_{i}'] = flattener(np.array(p).tolist())

    return result


def calc_score(task_test, predict):
    def comp(out, pred):
        try:
            return int(np.equal(out, pred).all())
        except:
            return 0

    return [comp(sample['output'], pred) for sample, pred in zip(task_test, predict)]


def evaluate(tasks, solver):
    result = []
    predictions = []
    for i, task in enumerate(tqdm(tasks)):
        task_result = solver.train(task['train'])
        if task_result:
            pred = solver.predict(task['test'])
            score = calc_score(task['test'], pred)
        else:
            pred = [el['input'] for el in task['test']]
            score = [0] * len(task['test'])

        predictions.append(pred)
        result.append(score)

    return result, predictions


if __name__ == '__main__':

    task_solver_1 = StackedTaskSolver(logger)
    task_solver_2 = TaskSolverTree(logger)
    task_solver_3 = TaskSolverColor(logger)

    solvers = [task_solver_2]

    train_tasks = load_data(BASE_PATH + '/training')
    evaluation_tasks = load_data(BASE_PATH + '/evaluation')
    test_tasks = load_data(BASE_PATH + '/test')

    submissions = []

    for i, task_solver in enumerate(solvers):

        logger.info(f'task solver {i}')

        if DEBUG:
            train_result, train_predictions = evaluate(train_tasks, task_solver)
            train_solved = [any(score) for score in train_result]

            total = sum([len(score) for score in train_result])
            logger.info(f"train solved : {sum(train_solved)} from {total} ({sum(train_solved) / total})")

            evaluation_result, evaluation_predictions = evaluate(evaluation_tasks, task_solver)
            evaluation_solved = [any(score) for score in evaluation_result]

            total = sum([len(score) for score in evaluation_result])
            logger.info(f"evaluation solved : {sum(evaluation_solved)} from {total} ({sum(evaluation_solved) / total})")

            with open('pkl/evaluation_tasks.pkl', 'w+b') as f:
                pickle.dump(evaluation_tasks, f)
            with open('pkl/evaluation_result.pkl', 'w+b') as f:
                pickle.dump(evaluation_result, f)
            with open('pkl/evaluation_predictions.pkl', 'w+b') as f:
                pickle.dump(evaluation_predictions, f)
            with open('pkl/evaluation_solved.pkl', 'w+b') as f:
                pickle.dump(evaluation_solved, f)

        submission = make_prediction(test_tasks, task_solver)
        submission = submission.reset_index()
        submission.columns = ['output_id', f'output_{i}']
        submission = submission.sort_values(by="output_id")
        submissions.append(submission)

    submission = pd.merge(submissions[0], submissions[1], on='output_id')
    submission = pd.merge(submission, submissions[2], on='output_id')

    def merge_cols(row):
        c1 = row[1].strip().split(" ")[:1]
        c2 = row[2].strip().split(" ")[:1]
        c3 = row[3].strip().split(" ")[:1]
        return ' '.join(c1 + c2 + c3)

    submission['output'] = submission.apply(merge_cols, axis=1)

    submission[['output_id', 'output']].to_csv('submission.csv', index=False)
