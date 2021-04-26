import numpy as np
import pandas as pd

from function.single_pendulum import single_pendulum
from function.double_pendulum import double_pendulum

class Sampling():
    init_file = './/data//single_init_27.csv'
    test_file = './/data//single_pendulum_test.csv'

    samples = []
    train = pd.DataFrame()
    test = pd.DataFrame()

    def __init__(self, n_var, var_ranges, n_samples, function="single_pendulum"):
        self.n_var = n_var
        self.var_ranges = var_ranges
        self.n_samples = n_samples
        self.function = function
        if self.function == "single_pendulum":
            self.init_file = './/data//single_init_27.csv'
            self.test_file = './/data//single_pendulum_test.csv'
        elif self.function == "double_pendulum":
            self.init_file = './/data//double_init_27.csv'
            self.test_file = './/data/double_pendulum_test.csv'

    def sampling(self):
        for i in range(self.n_samples):
            sample = []
            for v in range(self.n_var):
                val = [0.]  # need to implement
                sample.append(val)
            self.samples.append(sample)

    def show_samples(self):
        print("\n", type(self).__name__)
        for i in range(len(self.samples)):
            print(i, self.samples[i])

    def generate_train_data(self):
        if self.function == "single_pendulum":
            for i in range(len(self.samples)):
                # L = self.samples[i][0]
                # v0 = self.samples[i][1]
                # c = self.samples[i][2]
                L, v0, c = (self.samples[i][0], self.samples[i][1], self.samples[i][2])

                # t, angle, vel, accel = single_pendulum(L, v0, c)
                df = single_pendulum(L, v0, c)
                # print(df)
                # print(len(t), len(vel), len(accel))
                # ys = [t, angle, vel, accel]
                # print(i, " => ", self.samples[i], ys)  # 4개 * 101시점
                self.train = self.train.append(df)

        elif self.function == "double_pendulum":
            for i in range(len(self.samples)):
                # L1 = self.samples[i][0]
                # L2 = self.samples[i][1]
                # v0 = self.samples[i][2]
                L = np.array([self.samples[i][0], self.samples[i][1]])
                v0 = np.array([self.samples[i][2], self.samples[i][3]])

                # t, angle1, angle2, vel1, vel2 = double_pendulum(L, v0)
                df = double_pendulum(L, v0)
                # print(df)
                # print(len(t), len(vel), len(accel))
                # ys = [t, angle, vel, accel]
                # print(i, " => ", self.samples[i], ys)  # 4개 * 101시점
                self.train = self.train.append(df)

        return self.train

    def read_train_file(self):
        self.train = pd.read_csv(self.init_file)

    def read_test_file(self):
        self.test = pd.read_csv(self.test_file)
        # test_x = self.test[['L', 'v0', 'C', 't']]
        # test_y = self.test['angle']
        return self.test

