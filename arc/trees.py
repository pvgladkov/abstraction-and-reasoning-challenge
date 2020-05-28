from collections import defaultdict
from itertools import product, combinations, permutations
from math import floor

import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from arc.utils import TaskSolver, input_output_shape_is_same, flips, rotations2


class TaskSolverTree(TaskSolver):
    def __init__(self, logger):
        super(TaskSolverTree, self).__init__(logger)
        self.xgb = None

    def train(self, task_train, params=None):
        feat, target, not_valid = self.features(task_train)
        if not_valid:
            return False
        self.xgb = XGBClassifier(n_estimators=10, n_jobs=-1)
        self.xgb.fit(feat, target, verbose=-1)
        return True

    def predict(self, task_test):
        predictions = []

        for task_num in range(len(task_test)):
            input_color = np.array(task_test[task_num]['input'])
            nrows, ncols = len(task_test[task_num]['input']), len(task_test[task_num]['input'][0])
            feat = self.make_features(input_color)
            preds = self.xgb.predict(feat).reshape(nrows, ncols)
            preds = preds.astype(int).tolist()
            predictions.append(preds)
        return predictions

    @staticmethod
    def get_moore_neighbours(color, cur_row, cur_col, nrows, ncols):

        if (cur_row <= 0) or (cur_col > ncols - 1):
            top = -1
        else:
            top = color[cur_row - 1][cur_col]

        if (cur_row >= nrows - 1) or (cur_col > ncols - 1):
            bottom = -1
        else:
            bottom = color[cur_row + 1][cur_col]

        if (cur_col <= 0) or (cur_row > nrows - 1):
            left = -1
        else:
            left = color[cur_row][cur_col - 1]

        if (cur_col >= ncols - 1) or (cur_row > nrows - 1):
            right = -1
        else:
            right = color[cur_row][cur_col + 1]

        return top, bottom, left, right

    @staticmethod
    def get_tl_tr(color, cur_row, cur_col, nrows, ncols):

        if cur_row == 0:
            top_left = -1
            top_right = -1
        else:
            if cur_col == 0:
                top_left = -1
            else:
                top_left = color[cur_row - 1][cur_col - 1]
            if cur_col == ncols - 1:
                top_right = -1
            else:
                top_right = color[cur_row - 1][cur_col + 1]

        return top_left, top_right

    def make_features(self, input_color, nfeat=30, local_neighb=5):
        nrows, ncols = input_color.shape
        feat = np.zeros((nrows * ncols, nfeat))
        cur_idx = 0
        for i in range(nrows):
            for j in range(ncols):
                feat[cur_idx, 0] = i
                feat[cur_idx, 1] = j
                feat[cur_idx, 2] = input_color[i][j]
                feat[cur_idx, 3:7] = self.get_moore_neighbours(input_color, i, j, nrows, ncols)
                feat[cur_idx, 7:9] = self.get_tl_tr(input_color, i, j, nrows, ncols)
                feat[cur_idx, 9] = len(np.unique(input_color[i, :]))
                feat[cur_idx, 10] = len(np.unique(input_color[:, j]))
                feat[cur_idx, 11] = (i + j)
                feat[cur_idx, 12] = len(np.unique(input_color[i - local_neighb:i + local_neighb,
                                                  j - local_neighb:j + local_neighb]))
                feat[cur_idx, 13:17] = self.get_moore_neighbours(input_color, i + 1, j, nrows, ncols)
                feat[cur_idx, 17:21] = self.get_moore_neighbours(input_color, i - 1, j, nrows, ncols)

                feat[cur_idx, 21:25] = self.get_moore_neighbours(input_color, i, j + 1, nrows, ncols)
                feat[cur_idx, 25:29] = self.get_moore_neighbours(input_color, i, j - 1, nrows, ncols)

                cur_idx += 1

        return feat

    def features(self, task):
        feat, target = [], []

        for sample in task:
            nrows, ncols = len(sample['input']), len(sample['input'][0])

            target_rows, target_cols = len(sample['output']), len(sample['output'][0])

            if (target_rows != nrows) or (target_cols != ncols):
                return None, None, 1

            for input_fl, output_fl in zip(flips(sample['input']), flips(sample['output'])):
                feat.extend(self.make_features(input_fl))
                target.extend(np.array(output_fl).reshape(-1, ))

            for input_rot, output_rot in zip(rotations2(sample['input']), rotations2(sample['output'])):
                feat.extend(self.make_features(input_rot))
                target.extend(np.array(output_rot).reshape(-1, ))

        return np.array(feat), np.array(target), 0


class TaskSolverTree2(TaskSolver):

    def __init__(self, logger):
        super(TaskSolverTree2, self).__init__(logger)
        self.model = None
        self.size = 1

    def train(self, task_train, params=None):

        if not input_output_shape_is_same(task_train):
            return False

        bl_cols = self.get_bl_cols(task_train)
        isflip = False
        X1, Y1 = self.get_task_xy(task_train, True, self.size, bl_cols, isflip)
        self.model = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=100)
        self.model.fit(X1, Y1)
        return True

    def _predict(self, inp, model, size):
        r, c = len(inp), len(inp[0])
        oup = np.zeros([r, c], dtype=int)
        for i in range(r):
            for j in range(c):
                x = self.getX(inp, i, j, size)
                o = int(model.predict([x]))
                o = 0 if o < 0 else o
                oup[i][j] = o
        return oup

    def predict(self, task_test):
        predictions = []
        for pair in task_test:
            inp = pair["input"]
            oup = self._predict(inp, self.model, self.size)
            predictions.append(oup.tolist())
        return predictions

    @staticmethod
    def getiorc(pair):
        inp = pair["input"]
        return pair["input"], pair["output"], len(inp), len(inp[0])

    def getBkgColor(self, task_json):
        color_dict = defaultdict(int)

        for pair in task_json:
            inp, oup, r, c = self.getiorc(pair)
            for i in range(r):
                for j in range(c):
                    color_dict[inp[i][j]] += 1
        color = -1
        max_count = 0
        for col, cnt in color_dict.items():
            if (cnt > max_count):
                color = col
                max_count = cnt
        return color

    def get_bl_cols(self, task_json):
        result = []
        bkg_col = self.getBkgColor(task_json)
        result.append(bkg_col)
        # num_input,input_cnt,num_output,output_cnt
        met_map = {}
        for i in range(10):
            met_map[i] = [0, 0, 0, 0]

        total_ex = 0
        for pair in task_json:
            inp, oup = pair["input"], pair["output"]
            u, uc = np.unique(inp, return_counts=True)
            inp_cnt_map = dict(zip(u, uc))
            u, uc = np.unique(oup, return_counts=True)
            oup_cnt_map = dict(zip(u, uc))

            for col, cnt in inp_cnt_map.items():
                met_map[col][0] = met_map[col][0] + 1
                met_map[col][1] = met_map[col][1] + cnt
            for col, cnt in oup_cnt_map.items():
                met_map[col][2] = met_map[col][2] + 1
                met_map[col][3] = met_map[col][3] + cnt
            total_ex += 1

        for col, met in met_map.items():
            num_input, input_cnt, num_output, output_cnt = met
            if (num_input == total_ex or num_output == total_ex):
                result.append(col)
            elif (num_input == 0 and num_output > 0):
                result.append(col)

        result = np.unique(result).tolist()
        if (len(result) == 10):
            result.append(bkg_col)
        return np.unique(result).tolist()

    @staticmethod
    def getAround(i, j, inp, size=1):
        # v = [-1,-1,-1,-1,-1,-1,-1,-1,-1]
        r, c = len(inp), len(inp[0])
        v = []
        sc = [0]
        for q in range(size):
            sc.append(q + 1)
            sc.append(-(q + 1))
        for idx, (x, y) in enumerate(product(sc, sc)):
            ii = (i + x)
            jj = (j + y)
            v.append(-1)
            if ((0 <= ii < r) and (0 <= jj < c)):
                v[idx] = (inp[ii][jj])
        return v

    def getX(self, inp, i, j, size):
        z = []
        n_inp = np.array(inp)
        z.append(i)
        z.append(j)
        r, c = len(inp), len(inp[0])
        for m in range(5):
            z.append(i % (m + 1))
            z.append(j % (m + 1))
        z.append(i + j)
        z.append(i * j)
        #     z.append(i%j)
        #     z.append(j%i)
        z.append((i + 1) / (j + 1))
        z.append((j + 1) / (i + 1))
        z.append(r)
        z.append(c)
        z.append(len(np.unique(n_inp[i, :])))
        z.append(len(np.unique(n_inp[:, j])))
        arnd = self.getAround(i, j, inp, size)
        z.append(len(np.unique(arnd)))
        z.extend(arnd)
        return z

    def getXy(self, inp, oup, size):
        x = []
        y = []
        r, c = len(inp), len(inp[0])
        for i in range(r):
            for j in range(c):
                x.append(self.getX(inp, i, j, size))
                y.append(oup[i][j])
        return x, y

    @staticmethod
    def get_flips(inp, oup):
        result = []
        result.append((np.fliplr(inp).tolist(), np.fliplr(oup).tolist()))
        result.append((np.rot90(np.fliplr(inp), 1).tolist(), np.rot90(np.fliplr(oup), 1).tolist()))
        result.append((np.rot90(np.fliplr(inp), 2).tolist(), np.rot90(np.fliplr(oup), 2).tolist()))
        result.append((np.rot90(np.fliplr(inp), 3).tolist(), np.rot90(np.fliplr(oup), 3).tolist()))
        result.append((np.flipud(inp).tolist(), np.flipud(oup).tolist()))
        result.append((np.rot90(np.flipud(inp), 1).tolist(), np.rot90(np.flipud(oup), 1).tolist()))
        result.append((np.rot90(np.flipud(inp), 2).tolist(), np.rot90(np.flipud(oup), 2).tolist()))
        result.append((np.rot90(np.flipud(inp), 3).tolist(), np.rot90(np.flipud(oup), 3).tolist()))
        result.append((np.fliplr(np.flipud(inp)).tolist(), np.fliplr(np.flipud(oup)).tolist()))
        result.append((np.flipud(np.fliplr(inp)).tolist(), np.flipud(np.fliplr(oup)).tolist()))
        return result

    @staticmethod
    def replace(inp, uni, perm):
        # uni = '234' perm = ['5','7','9']
        # print(uni,perm)
        r_map = {int(c): int(s) for c, s in zip(uni, perm)}
        r, c = len(inp), len(inp[0])
        rp = np.array(inp).tolist()
        # print(rp)
        for i in range(r):
            for j in range(c):
                if (rp[i][j] in r_map):
                    rp[i][j] = r_map[rp[i][j]]
        return rp

    def augment(self, inp, oup, bl_cols):
        cols = "0123456789"
        npr_map = [1, 9, 72, 3024, 15120, 60480, 181440, 362880, 362880]
        uni = "".join([str(x) for x in np.unique(inp).tolist()])
        for c in bl_cols:
            cols = cols.replace(str(c), "")
            uni = uni.replace(str(c), "")

        exp_size = len(inp) * len(inp[0]) * npr_map[len(uni)]

        mod = floor(exp_size / 120000)
        mod = 1 if mod == 0 else mod

        # print(exp_size,mod,len(uni))
        result = []
        count = 0
        for comb in combinations(cols, len(uni)):
            for perm in permutations(comb):
                count += 1
                if (count % mod == 0):
                    result.append((self.replace(inp, uni, perm), self.replace(oup, uni, perm)))
        return result

    def get_task_xy(self, task_json, aug, around_size, bl_cols, flip=True):
        X = []
        Y = []
        for pair in task_json:
            inp, oup = pair["input"], pair["output"]
            tx, ty = self.getXy(inp, oup, around_size)
            X.extend(tx)
            Y.extend(ty)
            if (flip):
                for ainp, aoup in self.get_flips(inp, oup):
                    tx, ty = self.getXy(ainp, aoup, around_size)
                    X.extend(tx)
                    Y.extend(ty)
                    if (aug):
                        augs = self.augment(ainp, aoup, bl_cols)
                        for ainp, aoup in augs:
                            tx, ty = self.getXy(ainp, aoup, around_size)
                            X.extend(tx)
                            Y.extend(ty)
            if (aug):
                augs = self.augment(inp, oup, bl_cols)
                for ainp, aoup in augs:
                    tx, ty = self.getXy(ainp, aoup, around_size)
                    X.extend(tx)
                    Y.extend(ty)
        return X, Y