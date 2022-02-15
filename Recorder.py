import numpy as np
import scipy.signal as signal
import Simulation


class Voltage:
    def __init__(self):
        self.voltage = []
        self.list_cell = []
        self.t = []
        self.voltages = []
        self.temp = []
        self.records_from = 0.0
        self.until = np.inf
        self.eliminate = True

    # ------------------------------------------------------------------------------------------------------------------

    def record_from(self, cells, records_from=0.0, until=np.inf, eliminate=True):
        self.voltage = []
        if isinstance(cells, list) or isinstance(cells, np.ndarray) or isinstance(cells, np.matrix):
            self.list_cell = cells
        else:
            self.list_cell.append(cells)
        if not isinstance(cells, np.matrix):
            for i in self.list_cell:
                self.voltage.append([])
                self.temp.append([])
        else:
            for i in range(np.size(self.list_cell[0])):
                self.voltage.append([])
                for j in range(np.size(self.list_cell[1])):
                    self.voltage[i].append([])
        self.list_cell = np.asanyarray(self.list_cell)
        self.records_from = records_from
        self.until = until
        self.eliminate = eliminate

    # ------------------------------------------------------------------------------------------------------------------

    def record(self, cell, data, t):
        if cell in self.list_cell and self.records_from <= t < self.until:
            v = data[0]
            if not isinstance(self.list_cell, np.matrix):
                self.voltage[np.where(self.list_cell == cell)[0][0]].append(v)
            else:
                self.voltage[np.where(self.list_cell == cell)[0][0]][np.where(self.list_cell == cell)[1][0]].append(v)
            if not isinstance(self.list_cell, np.matrix):
                if cell == self.list_cell[0]:
                    if self.eliminate:
                        self.t.append(t - self.records_from)
                    else:
                        self.t.append(t)
            else:
                if cell == self.list_cell[0, 0]:
                    if self.eliminate:
                        self.t.append(t - self.records_from)
                    else:
                        self.t.append(t)

    # ------------------------------------------------------------------------------------------------------------------

    def finish(self):
        if not isinstance(self.list_cell, np.matrix):
            self.voltages = np.empty((len(self.list_cell), len(self.voltage[0])))
            for i in range(len(self.list_cell)):
                self.voltages[i, :] = self.voltage[i]
        else:
            self.voltages = np.empty((np.size(self.list_cell, 0), np.size(self.list_cell, 1), len(self.t)))
            for i in range(np.size(self.list_cell, 0)):
                for j in range(np.size(self.list_cell, 1)):
                    self.voltages[i, j, :] = self.voltage[i][j]
        self.t = np.asanyarray(self.t)

    # ------------------------------------------------------------------------------------------------------------------

    def get_result(self):
        return self.voltages, self.t

    # ------------------------------------------------------------------------------------------------------------------

    def clear(self, all=False):
        self.t = []
        self.voltages = []
        if all:
            self.voltage = []
            self.list_cell = []
            self.records_from = 0.0
            self.until = np.inf
            self.eliminate = True