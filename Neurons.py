import numpy as np


class FEF:
    def __init__(self):
        self.type = 'FEF'
        self.id = 0
        self.input_weight = 0.8
        self.a = 0.5
        self.theta = 1.6
        self.inputs = []
        self.weights = []
        self.noise_coef = 0.02

        self.external_current = 0
        self.external_input = 0
        self.bg_current = 0

        self.v_init = 0
        self.v = self.v_init

        self.t_indx = 0

    def set_inputs(self, x):
        self.inputs.append(x)

    def set_self_v(self, v):
        self.v = v

    def clear_inputs(self):
        self.inputs = []

    def set_weights(self, x):
        self.weights.append(x)

    def get_v(self):
        return self.v

    def get_id(self):
        return self.id

    def set_id(self, id):
        self.id = id

    def get_type(self):
        return self.type

    def set_current(self, f):
        self.external_current = f

    def set_input_current(self, f):
        self.external_input = f

    def current(self, t):
        if callable(self.external_current):
            return self.external_current(t)  # * (t >= 0)
        elif isinstance(self.external_current, list) or isinstance(self.external_current, np.ndarray):
            a = self.external_current[self.t_indx]
            self.t_indx += 1
            return a

        else:
            return self.external_current * (t >= 0)

    def input_current(self, t):
        if callable(self.external_input):
            return self.external_input(t)  # * (t >= 0)
        elif isinstance(self.external_input, list) or isinstance(self.external_input, np.ndarray):
            a = self.external_input[self.t_indx]
            self.t_indx += 1
            return a

        else:
            return self.external_input * (t >= 0)

    def diff_equation(self, v, t):
        V = v[0]

        if len(self.inputs) == 1:
            out = -V + 1 / (1 + np.exp(
                -self.a * (self.inputs[0] * self.weights[0] + self.input_current(t) - self.theta)) - 1 / (1 + np.exp(
                self.a * self.theta))) + np.random.rand() * self.noise_coef + self.current(t) + self.bg_current

        elif len(self.inputs) > 1:
            out = -V + 1 / (1 + np.exp(
                -self.a * (np.matmul(self.inputs, self.weights) + self.input_current(t) - self.theta)) - 1 / (
                                        1 + np.exp(
                                    self.a * self.theta))) + np.random.rand() * self.noise_coef + self.current(
                t) + self.bg_current

        else:
            out = -V + 1 / (1 + np.exp(-self.a * self.v_init + self.input_current(t) - self.theta) - 1 / (
                    1 + np.exp(self.a * self.theta))) + np.random.rand() * self.noise_coef + self.current(
                t) + self.bg_current

        return out
