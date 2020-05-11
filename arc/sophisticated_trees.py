from arc.utils import TaskSolver
from arc.nn import TaskSolverUNet, TaskSolverConv1
from arc.trees import TaskSolverTree
import numpy as np
from xgboost import XGBClassifier


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

        task_train_net = self.net.predict(task_train[1:])

        feat, target = [], []
        for orig, transformed in zip(task_train[1:], task_train_net):
            nrows, ncols = len(orig['input']), len(orig['input'][0])
            target_rows, target_cols = len(orig['output']), len(orig['output'][0])
            if (target_rows != nrows) or (target_cols != ncols):
                return False

            orig_f = self.tree.make_features(np.array(orig['input']))
            # transformed_f = self.tree.make_features(np.array(transformed))

            target_f = np.array(np.array(orig['output'])).reshape(-1, )

            full_f = np.hstack((orig_f, np.array(transformed).reshape(orig_f.shape[0], 1)))
            full_target = target_f

            feat.extend(full_f)
            target.extend(full_target)

        self.xgb = XGBClassifier(n_estimators=10, n_jobs=-1)
        self.xgb.fit(np.array(feat), np.array(target), verbose=-1)
        return True

    def predict(self, task_test):

        task_train_net = self.net.predict(task_test)

        predictions = []

        for task_num in range(len(task_test)):
            nrows, ncols = len(task_train_net[task_num]), len(task_train_net[task_num][0])

            orig_f = self.tree.make_features(np.array(task_test[task_num]['input']))
            # transformed_f = self.tree.make_features(np.array(task_train_net[task_num]))

            full_f = np.hstack((orig_f, np.array(task_train_net[task_num]).reshape(orig_f.shape[0], 1)))

            preds = self.xgb.predict(full_f).reshape(nrows, ncols)
            preds = preds.astype(int).tolist()
            predictions.append(preds)
        return predictions
