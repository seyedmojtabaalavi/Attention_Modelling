import numpy as np
from scipy.integrate import odeint
from mpi4py import MPI

if not MPI.COMM_WORLD.Get_rank():
    import progressbar

resolution = 0.1
iteration = 0


class simulation:
    def __init__(self):
        global resolution
        self.last_t = 0.0
        self.result = []
        self.recorder = []

    def set_resolution(self, set_resolution):
        global resolution
        resolution = set_resolution

    def run(self, network, time_duration_ms, recorder=None, start_time=0):
        global resolution
        global iteration

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        nodes_size = comm.Get_size()

        if rank == 0:
            if recorder is None:
                print('Warning: There is no recorder. You wont have result !')

        if nodes_size > 1:
            ranklist = np.arange(1, nodes_size)
            size = nodes_size - 1
        else:
            ranklist = np.arange(0, nodes_size + 1)
            size = nodes_size

        neuron_lists = network.neuron_lists
        connection_list = network.connection_list
        t = np.arange(start_time + self.last_t, self.last_t + time_duration_ms + resolution, resolution)

        if not isinstance(recorder, list):
            recorder = [recorder]
        self.recorder = recorder

        time_size = len(t)
        list_neuron_size = len(neuron_lists)
        list_network_size = len(connection_list)
        for i in range(list_neuron_size):
            v_init = neuron_lists[i].v_init

            temp_init = [v_init]
            self.result.append(temp_init)
            del temp_init
        ###
        for j in range(list_network_size):
            for k in range(1, len(connection_list[j])):
                if connection_list[j][k].get_type() == 'FEF':
                    connection_list[j][k].set_inputs(connection_list[j][0].get_v())

        self.kernel_random()

        if not rank:
            # print '\n  Running simulation for', time_duration_ms, ' ms ...'
            bar = progressbar.ProgressBar(maxval=time_size, widgets=[' Simulating: ', progressbar.Percentage(), ' ',
                                                                     progressbar.Bar('=', '[', ']'),
                                                                     ', ', progressbar.ETA()])
            bar.start()
        for i in range(1, time_size):
            if not rank:
                # print '\r', '\r%.2f' % (i * 100.0 / (time_size - 1)), '%', ''
                bar.update(i)
            pkg = np.zeros(2)
            for j in range(list_neuron_size):
                if rank == ranklist[j % size]:
                    # self.result[j] = odeint(neuron_lists[j].diff_equation, self.result[j], t[i - 1:i + 1])[1]
                    # self.result[j] = np.multiply(resolution, self.result[j]) + neuron_lists[j].diff_equation(
                    #     self.result[j], t[i])
                    self.result[j] += np.multiply(resolution, neuron_lists[j].diff_equation(self.result[j], t[i]))

                    if nodes_size > 1 and rank:
                        pkg[:-1] = self.result[j]
                        pkg[-1] = j
                        comm.Send(pkg, dest=0, tag=0)
                    else:
                        neuron_lists[j].clear_inputs()
                        neuron_lists[j].set_self_v(self.result[j][0])
                        for k in range(len(recorder)):
                            recorder[k].record(neuron_lists[j], self.result[j], t[i])

            if nodes_size > 1 and not rank:
                get_pkg = np.zeros((list_neuron_size, 1))
                temp_counter = 0
                while temp_counter < list_neuron_size:
                    comm.Recv(pkg, source=MPI.ANY_SOURCE, tag=0)
                    temp_counter += 1
                    get_pkg[int(pkg[-1]), 0] = pkg[0]
                    for j in range(len(recorder)):
                        recorder[j].record(neuron_lists[int(pkg[-1])], pkg[:-1], t[i])
                for k in range(size):
                    comm.Send(get_pkg, dest=ranklist[k], tag=1)

            if nodes_size > 1 and rank:
                get_pkg = np.zeros((list_neuron_size, 2))
                p = np.zeros(1)
                comm.Recv(get_pkg, source=0, tag=1)
                for j in range(list_neuron_size):
                    neuron_lists[j].set_self_v(get_pkg[j, 0])
                    neuron_lists[j].clear_inputs()
                comm.Send(p, dest=0, tag=2)

            for j in range(list_network_size):
                for k in range(1, len(connection_list[j])):
                    if connection_list[j][0].get_type() == 'FEF':
                        connection_list[j][k].set_inputs(connection_list[j][0].get_v())
                        #
                        # if connection_list[j][k].get_id() == 2:
                        #     print(connection_list[j][k].inputs)

            if nodes_size > 1 and not rank:
                p = np.zeros(1)
                for j in range(size):
                    comm.Recv(p, source=MPI.ANY_SOURCE, tag=2)
            self.kernel_random()
            iteration += 1

        self.last_t = t[i]

        if rank == 0:
            bar.finish()
            for i in range(len(recorder)):
                recorder[i].finish()

    # ------------------------------------------------------------------------------------------------------------------

    def clear(self):
        self.last_t = 0.0
        self.result = []
        for i in range(len(self.recorder)):
            self.recorder[i].clear()

    # ------------------------------------------------------------------------------------------------------------------

    def kernel_random(self):
        global counter
        global random_list
        global random_types
        global random_arguments

        comm = MPI.COMM_WORLD
        pkg = np.zeros(2)
        if comm.Get_rank() == 0:
            for i in range(counter):
                if random_types[i] == 0:
                    random_list[i] = np.random.normal(random_arguments[i][0], random_arguments[i][1],
                                                      random_arguments[i][2])
                elif random_types[i] == 1:
                    random_list[i] = np.random.normal(random_arguments[i][0], random_arguments[i][1],
                                                      random_arguments[i][2])
                elif random_types[i] == 2:
                    random_list[i] = np.random.poisson(random_arguments[i][0], random_arguments[i][1])
                elif random_types[i] == 3:
                    random_list[i] = np.random.uniform(random_arguments[i][0], random_arguments[i][1],
                                                       random_arguments[i][2])
                else:
                    print('It seems there is something wrong about random generation. Please check your code.')

                pkg[0] = random_list[i]
                pkg[1] = i
                for j in range(1, comm.Get_size()):
                    comm.Send(pkg, dest=j, tag=3)
        else:
            for i in range(counter):
                comm.Recv(pkg, source=0, tag=3)
                random_list[int(pkg[1])] = pkg[0]


# ======================================================================================================================

random_list = []
random_types = []
counter = 0
random_arguments = []


class Kernel_random:
    def __init__(self, distribution='normal', arguments=[0, 1, 1]):
        global counter
        global random_list
        global random_types
        global random_arguments

        self.number = counter
        counter += 1
        random_list.append(0)
        if distribution == 'normal':
            random_types.append(0)
        elif distribution == 'gamma':
            random_types.append(1)
        elif distribution == 'poisson':
            random_types.append(2)
        elif distribution == 'uniform':
            random_types.append(3)
        else:
            print("there is no distribution by name, '", distribution, "'.")
        random_arguments.append(arguments)

    def get_value(self):
        global random_list
        if not isinstance(random_list[self.number], list):
            return random_list[self.number]
        else:
            return random_list[self.number][0]
