import numpy as np
import scipy.io as sio


def rnd_tupel(scale, i, j):
    while True:
        x = int(abs(np.random.normal(scale=scale))) + i
        if x != i and x <= scale:
            break

    while True:
        y = int(abs(np.random.normal(scale=scale))) + j
        if y != j and y <= scale:
            break
    return (x, y)




def indx(square_range, i=None, j=None, sigma=1):
    i_indexes = []
    j_indexes = []
    for a in range(square_range):

        i_indexes.append([])
        j_indexes.append([])
        for b in range(square_range):
            # i_indexes[a].append([])
            # for c in range(square_range**2):
            #     indexes[a][b].append(rnd_tupel(square_range, a, b))
            x = np.random.normal(scale=square_range, size=square_range)
            # x = x * square_range
            x1 = map(int, x)
            x2 = np.asarray(list(x1)) + a
            x3 = np.delete(x2, np.where((x2 == a) | (x2 >= square_range) | (x2 < 0)))

            i_indexes[a].append(x3)
            y = np.random.normal(scale=square_range, size=square_range)
            y1 = map(int, y)
            y2 = np.asarray(list(y1)) + b
            y3 = np.delete(y2, np.where((y2 == a) | (y2 >= square_range) | (y2 < 0)))
            j_indexes[a].append(y3)

    return i_indexes, j_indexes


a, b = indx(100)

sio.savemat('indexes.mat', mdict={'i': a, 'j':b})
