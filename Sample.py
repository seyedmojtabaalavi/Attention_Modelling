import numpy as np
import pylab as plt
import Neurons
import Networking
import Simulation
import Recorder
from scipy.optimize import curve_fit
import scipy.io as sio

t = 700
res = 1
stim_start = 100
stim_end = 450


def guss(x, a, b, c):
    return a * np.exp(-(x - b ** 2) / (2 * c ** 2))


base = np.arange(0, t, res)

net = Networking.network()
fr = Recorder.Voltage()

pos = 1

Nrns = np.empty((100, 100), dtype=type(net))
Nrns[0, 0] = net.create('FEF')
for i in range(np.size(Nrns, 0)):
    for j in range(np.size(Nrns, 1)):
        Nrns[i, j] = net.create('FEF')
        input = np.zeros(len(base))
        if pos == 1:
            pos_coef = guss(np.sqrt(i ** 2 + (np.size(Nrns, 1) - j) ** 2), 1, 0, 1) * 1
        elif pos == 2:
            pos_coef = guss(np.sqrt((np.size(Nrns, 0)/2 - i) ** 2 + j ** 2), 1, 0, 1) * 1
        elif pos == 3:
            pos_coef = guss(np.sqrt(i ** 2 + j ** 2), 1, 0, 1) * 1
        elif pos == 4:
            pos_coef = guss(np.sqrt(i ** 2 + (np.size(Nrns, 1) / 2 - j) ** 2), 1, 0, 1) * 1
        elif pos == 5:
            pos_coef = guss(np.sqrt(i ** 2 + (np.size(Nrns, 1) - j) ** 2), 1, 0, 1) * 1
        elif pos == 6:
            pos_coef = guss(np.sqrt((np.size(Nrns, 0)/2 - i) ** 2 + (np.size(Nrns, 1)/2 - j) ** 2), 1, 0, 1) * 1
        elif pos == 7:
            pos_coef = guss(np.sqrt((np.size(Nrns, 0) - i) ** 2 + (np.size(Nrns, 1) - j) ** 2), 1, 0, 1) * 1
        elif pos == 8:
            pos_coef = guss(np.sqrt(i ** 2 + (np.size(Nrns, 1) / 2 - j) ** 2), 1, 0, 1) * 1

        input[int(np.where(base == stim_start)[0]):int(np.where(base == stim_end)[0])] = np.hanning(
            len(input[int(np.where(base == stim_start)[0]):int(np.where(base == stim_end)[0])])) * pos_coef
        Nrns[i, j].set_input_current(np.copy(input))
Nrns = np.asmatrix(Nrns)

n_conn = 0
d = []
for i in range(np.size(Nrns, 0)):
    for j in range(np.size(Nrns, 1)):
        d.append(np.sqrt(i ** 2 + j ** 2))

d_list = np.unique(d)
n_trials = 1
distances = np.zeros((n_trials, len(d_list)))
weits = np.zeros((n_trials, len(d_list)))

# for x in range(n_trials):
#     distance = []
#     for i in range(np.size(Nrns, 0)):
#         print(i)
#         for j in range(np.size(Nrns, 1)):
#
#             for k in range(np.size(Nrns, 0)):
#                 for l in range(np.size(Nrns, 1)):
#                     if i != k or j != l:
#                         dist = np.sqrt((i - k) ** 2 + (j - l) ** 2)
#                         prob = dist / np.sqrt(np.size(Nrns, 0) ** 2 + np.size(Nrns, 1) ** 2)
#                         rand = np.random.normal()
#                         if rand >= prob and rand >= 0:
#                             w = (np.random.normal() + 0.75) * 0.35 * (1 - prob)
#                             net.connect(Nrns[i, j], Nrns[k, l], w)
#                             distances[x, np.where(d_list == dist)[0]] += 1
#                             weits[x, np.where(d_list == dist)[0]] = w

    # distances.append(distance)

indexes = sio.loadmat('indexes.mat')
i_indx = indexes['i']
j_indx = indexes['j']

for x in range(n_trials):
    distance = []
    for i in range(np.size(Nrns, 0)):
        print(i)
        for j in range(np.size(Nrns, 1)):

            for l in range(min(len(i_indx[i][j][0]), len(j_indx[i][j][0]))):

                net.connect(Nrns[i, j], Nrns[i_indx[i][j][0][l], j_indx[i][j][0][l]], 1)

m_distances = np.mean(distances, 0)
m_weits = np.mean(weits, 0)
params, pcov = curve_fit(guss, d_list[1:], m_distances[1:])

params_w, p_w = curve_fit(guss, d_list[1:], m_weits[1:])


fr.record_from(Nrns)

simul = Simulation.simulation()
simul.set_resolution(res)
simul.run(net, t, [fr])

v, t_1 = fr.get_result()
plt.figure()

plt.plot(t_1, v[0, 0], label='Neuron')# + str(i+1))
plt.plot(t_1, v[-1, -1], label='Neuron')

plt.xlim([0, t])
plt.legend()
plt.xlabel('time (ms)')
plt.ylabel('Firing rate (Hz)')

plt.show()
