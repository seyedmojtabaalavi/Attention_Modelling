import numpy as np
import Neurons
from mpi4py import MPI


class network:
    def __init__(self):
        self.connection_list = []
        self.neuron_lists = []
        self.weight_list = []
        self.id = 0

    # ------------------------------------------------------------------------------------------------------------------
    # Connection

    def connect(self, presynapse, postsynapse, w=1):
        length_list = len(self.connection_list)
        flag = False
        if length_list != 0:
            for i in range(length_list):
                if self.connection_list[i][0].get_id() == presynapse.get_id():
                    flag1 = True
                    for j in range(1, len(self.connection_list[i])):
                        if self.connection_list[i][j].get_id() == postsynapse.get_id():
                            flag1 = False
                            break

                    if flag1:
                        self.connection_list[i].append(postsynapse)
                        postsynapse.set_weights(w)
                    flag = True
            if not flag:
                self.connection_list.append([presynapse, postsynapse])
                postsynapse.set_weights(w)
        else:
            self.connection_list.append([presynapse, postsynapse])
            postsynapse.set_weights(w)

    # ------------------------------------------------------------------------------------------------------------------
    # Creation

    def create(self, neuron_type, number=1):
        if number == 1:
            if neuron_type == 'FEF':
                FEF = Neurons.FEF()
                FEF.set_id(self.id)
                self.id += 1
                self.neuron_lists.append(FEF)
                return FEF
        elif number > 1:
            neurons_list = []
            if neuron_type == 'FEF':
                for i in range(number):
                    FEF = Neurons.FEF()
                    FEF.set_id(self.id)
                    self.id += 1
                    self.neuron_lists.append(FEF)
                    neurons_list.append(FEF)
                    del FEF
            return neurons_list
        print('wrong usage')
        return
