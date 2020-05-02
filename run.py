import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from arc_models import BasicCNNModel
from arc_utils import transform_dim, resize, flt, npy, itg, \
    flattener, ARCDataset, debug_json, get_test_tasks, get_logger

T = torch.Tensor

SIZE = 1000
EPOCHS = 50
CONV_OUT_1 = 50
CONV_OUT_2 = 100
BATCH_SIZE = 128

TEST_PATH = Path('/data/arc/')
SUBMISSION_PATH = Path('/data/arc/')

# TEST_PATH = Path('../input/abstraction-and-reasoning-challenge/')
# SUBMISSION_PATH = Path('../input/abstraction-and-reasoning-challenge/')

TEST_PATH = TEST_PATH / 'test'
SUBMISSION_PATH = SUBMISSION_PATH / 'sample_submission.csv'

DEBUG = True

logger = get_logger()


if __name__ == '__main__':

    test_tasks, task_names = get_test_tasks(TEST_PATH)

    Xs_test, Xs_train, ys_train = [], [], []
    y_task_names = []

    for task, name in zip(test_tasks, task_names):
        X_test, X_train, y_train = [], [], []

        for i, pair in enumerate(task["test"]):
            X_test.append(pair["input"])
            y_task_names.append('{}_{}'.format(name, i))

        for pair in task["train"]:
            X_train.append(pair["input"])
            y_train.append(pair["output"])

        Xs_test.append(X_test)
        Xs_train.append(X_train)
        ys_train.append(y_train)

    idx = 0
    start = time.time()
    test_predictions = []

    for X_train, y_train in zip(Xs_train, ys_train):

        logger.info("TASK " + str(idx + 1))

        train_set = ARCDataset(X_train, y_train, SIZE, stage="train")
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

        inp_dim = np.array(X_train[0]).shape
        outp_dim = np.array(y_train[0]).shape
        network = BasicCNNModel(CONV_OUT_1, CONV_OUT_2, inp_dim, outp_dim).cuda()
        optimizer = Adam(network.parameters(), lr=0.01)

        for epoch in range(EPOCHS):
            epoch_loss = 0
            num_examples = 0
            for train_batch in train_loader:
                train_X, train_y, out_d, d, out = train_batch
                train_preds = network.forward(train_X.cuda(), out_d.cuda())
                train_loss = nn.MSELoss()(train_preds, train_y.cuda())

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                epoch_loss += train_loss.item()
                num_examples += len(train_y)

            logger.debug('task {}, epoch {}, loss {}'.format(idx+1, epoch, round(epoch_loss/num_examples, 6)))

        end = time.time()
        logger.info("Total time: " + str(np.round(end - start, 1)) + " s")

        X_test = np.array([resize(flt(X), np.shape(X), inp_dim) for X in Xs_test[idx]])

        for X in X_test:
            test_dim = np.array(T(X)).shape
            test_preds = npy(network.forward(T(X).unsqueeze(0).cuda(), out_d.cuda()))
            test_preds = np.argmax(test_preds.reshape((10, *outp_dim)), axis=0)
            test_predictions.append(itg(resize(test_preds, np.shape(test_preds),
                                               tuple(itg(transform_dim(inp_dim,
                                                                       outp_dim,
                                                                       test_dim))))))
        idx += 1

    test_predictions = [[list(pred) for pred in test_pred] for test_pred in test_predictions]

    if DEBUG:
        Xs_test_flat = [item for t in Xs_test for item in t]
        debug_json(y_task_names, Xs_test_flat, test_predictions)

    for idx, pred in enumerate(test_predictions):
        test_predictions[idx] = flattener(pred)

    submission = pd.DataFrame({'output_id': y_task_names, 'output': test_predictions})

    submission.to_csv("submission.csv", index=False)
