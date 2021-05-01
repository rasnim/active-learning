from sampling.sampling import Sampling

class GridSampling(Sampling):
    samples = []

    def __init__(self, n_var, var_ranges, n_samples=None, n_pts_axis=2, function="single_pendulum"):
        # n_pts_axis is the number of points in each axis (variable)
        # if n_pts_axis = 2, only use the min and max values
        super().__init__(n_var, var_ranges, n_samples)
        self.n_pts_axis = n_pts_axis
        self.function = function

    def sampling(self):
        self.recursive_bubbling(self.n_var-1)

    def recursive_bubbling(self, var_idx):
        # vals = []
        min = self.var_ranges[self.n_var - 1 - var_idx][0]
        max = self.var_ranges[self.n_var - 1 - var_idx][1]

        # if n_pts_axis = 2, only use the min and max values
        gap = (max - min) / (self.n_pts_axis - 1)

        # for i in range(self.n_pts_axis):
        #     # if n_divide=0, add only the max. Otherwise, add each in-between values
        #     vals.append(min + gap * i)
        vals = [min + gap * i for i in range(self.n_pts_axis)]

        if var_idx == 0:
            self.samples = [[x] for x in vals]
            # for val in (vals):
            #     sample = []
            #     sample.append(val)
            #     self.samples.append(sample)
        else:
            self.recursive_bubbling(var_idx - 1)
            pre_samples = self.samples
            new_samples = []
            for val in vals:
                test = [[val]+x for x in pre_samples]
                new_samples = new_samples + test
            self.samples = new_samples
            # self.samples = [[[[val] + x] for val in vals] for x in self.samples]
            # self.samples = [[[[val] + x] for x in self.samples] for val in vals]

    def show_samples(self):
        super().show_samples()
        # for i in range(len(self.samples)):
        #     print(i, self.samples[i])


if __name__ == "__main__":
    # Min_L = 0.1   Max_L = 0.3
    # Min_v = 0     Max_v = 5
    # Min_c = 0     Max_c = 0.2
    n_var = 3
    var_ranges = [[0.1, 0.3], [0., 5.], [0., 0.2]]
    n_samples = 8

    grid_s = GridSampling(n_var, var_ranges, n_samples=None, n_pts_axis=2)
        # n_pts_axis=3 selects three points in each axis
    grid_s.sampling()
    grid_s.show_samples()
