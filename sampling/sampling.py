import numpy as np
import pandas as pd
import csv

from function.single_pendulum import single_pendulum
from function.double_pendulum import double_pendulum

class Sampling():
    # init_file = './/data//single_init_27.csv'
    # test_file = './/data//single_pendulum_test.csv'

    def __init__(self, n_var, var_ranges, n_samples, function="single_pendulum"):
        self.samples = []
        self.train = pd.DataFrame()
        self.test = pd.DataFrame()

        self.n_var = n_var
        self.var_ranges = var_ranges
        self.n_samples = n_samples
        self.function = function

    def sampling(self):
        # for i in range(self.n_samples):
        #     sample = []
        #     for v in range(self.n_var):
        #         val = [0.]  # need to implement
        #         sample.append(val)
        #     self.samples.append(sample)
        self.samples = [[0. for v in range(self.n_var)] for i in range(self.n_samples)]

    def show_samples(self):
        print("\n", type(self).__name__)
        # cnt = 0
        # for sample in self.samples:
        #     print(cnt, sample)
        #     cnt += 1
        [print(x) for x in self.samples]

    def write_samples_to_file(self, out_file=None):
        with open('data\\init_file_test', 'w') as f:
            # using csv.writer method from CSV package
            write = csv.writer(f)

            # write.writerow(fields)
            write.writerows(self.samples)

    def generate_train_data(self):
        def append(x):
            self.train = self.train.append(x)

        if self.function == "single_pendulum":
            # for sample in self.samples:
            #     # # L = self.samples[i][0]
            #     # # v0 = self.samples[i][1]
            #     # # c = self.samples[i][2]
            #     #
            #     # # L, v0, c = (self.samples[i][0], self.samples[i][1], self.samples[i][2])
            #     # # # t, angle, vel, accel = single_pendulum(L, v0, c)
            #     # # df = single_pendulum(L, v0, c)
            #     # df = single_pendulum(sample[0], sample[1], sample[2])
            #     # # print(df)
            #     # # print(len(t), len(vel), len(accel))
            #     # # ys = [t, angle, vel, accel]
            #     # # print(i, " => ", self.samples[i], ys)  # 4개 * 101시점
            #     # self.train = self.train.append(df)
            #     # del df
            #     append(single_pendulum(sample[0], sample[1], sample[2]))
            [append(single_pendulum(sample[0], sample[1], sample[2])) for sample in self.samples]

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

    # def read_train_file(self, init_file):
    #     self.train = pd.read_csv(init_file)
    #
    # def read_test_file(self, test_file):
    #     self.test = pd.read_csv(test_file)
    #     return self.test

