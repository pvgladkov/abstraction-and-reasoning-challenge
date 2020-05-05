import numpy as np


def transpose(matrix: [[]]) -> [[]]:
    matrix[:] = [list(x) for x in zip(*matrix)]
    return matrix


def reverse_row(matrix: [[]]) -> [[]]:
    matrix[:] = matrix[::-1]
    return matrix


def reverse_column(matrix: [[]]) -> [[]]:
    matrix[:] = [x[::-1] for x in matrix]
    return matrix


def rotate_90(matrix: [[]]) -> [[]]:
    """
    >>> rotate_90(make_matrix())
    [[4, 8, 12, 16], [3, 7, 11, 15], [2, 6, 10, 14], [1, 5, 9, 13]]
    >>> rotate_90(make_matrix()) == transpose(reverse_column(make_matrix()))
    True
    """
    return reverse_row(transpose(matrix))


def rotate_180(matrix: [[]]) -> [[]]:
    """
    >>> rotate_180(make_matrix())
    [[16, 15, 14, 13], [12, 11, 10, 9], [8, 7, 6, 5], [4, 3, 2, 1]]
    >>> rotate_180(make_matrix()) == reverse_column(reverse_row(make_matrix()))
    True
    """
    return reverse_row(reverse_column(matrix))


def rotate_270(matrix: [[]]) -> [[]]:
    """
    >>> rotate_270(make_matrix())
    [[13, 9, 5, 1], [14, 10, 6, 2], [15, 11, 7, 3], [16, 12, 8, 4]]
    >>> rotate_270(make_matrix()) == transpose(reverse_row(make_matrix()))
    True
    """
    return reverse_column(transpose(matrix))


def rotations(matrix):
    return [matrix, rotate_90(matrix), rotate_180(matrix), rotate_270(matrix)]


def inp2img(inp):
    inp = np.array(inp)
    img = np.full((10, inp.shape[0], inp.shape[1]), 0, dtype=np.uint8)
    for i in range(10):
        img[i] = (inp == i)
    return img


def inp2grey(inp, exp=True):
    inp = np.array(inp)
    b_inp = (inp > 0).astype(np.int)
    if exp:
        return np.expand_dims(b_inp, 0)
    return b_inp