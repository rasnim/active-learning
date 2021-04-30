import random

from sampling.sampling import Sampling

class RandomSampling(Sampling):
    # random.seed(10)

    def sampling(self):
        for i in range(self.n_samples):
            sample = []
            for v in range(self.n_var):
                val = random.uniform(self.var_ranges[v][0], self.var_ranges[v][1])
                sample.append(val)
            self.samples.append(sample)


# if __name__ == "__main__":
#     # Min_L = 0.1   Max_L = 0.3
#     # Min_v = 0     Max_v = 5
#     # Min_c = 0     Max_c = 0.2
#     n_var = 3
#     var_ranges = [[0.1, 0.3], [0., 5.], [0., 0.2]]
#     n_samples = 27
#
#     rd_s = RandomSampling(n_var, var_ranges, n_samples)
#     rd_s.sampling()
#     rd_s.show_samples()
