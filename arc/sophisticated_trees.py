import numpy as np
from xgboost import XGBClassifier

from arc.nn import TaskSolverUNet
from arc.trees import TaskSolverTree
from arc.utils import TaskSolver


class StackedTaskSolver(TaskSolver):

    def __init__(self, logger):
        super(StackedTaskSolver, self).__init__(logger)
        self.net = TaskSolverUNet(logger, n_epoch=50)
        self.tree = TaskSolverTree(logger)
        self.xgb = None

    def train(self, task_train, params=None):

        net_is_trained = self.net.train(task_train[:1])
        if not net_is_trained:
            return False

        task_train_net = self.net.predict_proba(task_train[1:])

        feat, target = [], []
        for orig, transformed in zip(task_train[1:], task_train_net):
            nrows, ncols = len(orig['input']), len(orig['input'][0])
            target_rows, target_cols = len(orig['output']), len(orig['output'][0])
            if (target_rows != nrows) or (target_cols != ncols):
                return False

            orig_f = self.tree.make_features(np.array(orig['input']))

            target_f = np.array(np.array(orig['output'])).reshape(-1, )
            full_f = np.zeros((nrows*ncols, orig_f.shape[1] + 1))
            idx = 0
            for i in range(nrows):
                for j in range(ncols):
                    full_f[idx, 0:orig_f.shape[1]] = orig_f[idx]
                    full_f[idx, orig_f.shape[1]] = transformed[:, i, j].argmax(0) > 0
                    idx += 1

            full_target = target_f

            feat.extend(full_f)
            target.extend(full_target)

        self.xgb = XGBClassifier(n_estimators=10, n_jobs=-1)
        self.xgb.fit(np.array(feat), np.array(target), verbose=-1)
        return True

    def predict(self, task_test):

        task_train_net = self.net.predict_proba(task_test)

        predictions = []

        for task_num in range(len(task_test)):
            nrows, ncols = len(task_test[task_num]['input']), len(task_test[task_num]['input'][0])

            orig_f = self.tree.make_features(np.array(task_test[task_num]['input']))

            full_f = np.zeros((nrows * ncols, orig_f.shape[1] + 1))

            idx = 0
            for i in range(nrows):
                for j in range(ncols):
                    full_f[idx, 0:orig_f.shape[1]] = orig_f[idx]
                    full_f[idx, orig_f.shape[1]] = task_train_net[task_num][:, i, j].argmax(0) > 0
                    idx += 1

            preds = self.xgb.predict(full_f).reshape(nrows, ncols)
            preds = preds.astype(int).tolist()
            predictions.append(preds)
        return predictions
