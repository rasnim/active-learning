from sampling.sampling import Sampling
from sampling.grid_sampling import GridSampling

class SmartSampling(Sampling):
    init_samples = []
    # n_pts_axis = 2
    # n_init_samples = 8

    def __init__(self, n_var, var_ranges, n_samples=None, n_pts_axis=2):
        # n_pts_axis is the number of points in each axis (variable)
        # if n_pts_axis = 2, only use the min and max values
        super().__init__(n_var, var_ranges, n_samples)
        self.n_pts_axis = n_pts_axis
        self.n_init_samples = pow(n_pts_axis, n_var)

    def init_sampling(self):
        grid_s = GridSampling(self.n_var, self.var_ranges, n_pts_axis=self.n_pts_axis)
            # n_pts_axis=3 selects three points in each axis
        grid_s.sampling()
        grid_s.show_samples()
        self.init_samples = grid_s.samples
        self.samples = grid_s.samples

    def show_init_samples(self):
        print("\nInitial Samles : n_init_samples=", self.n_init_samples)
        for i in range(len(self.init_samples)):
            print(i, self.init_samples[i])


if __name__ == "__main__":
    # Min_L = 0.1   Max_L = 0.3
    # Min_v = 0     Max_v = 5
    # Min_c = 0     Max_c = 0.2
    n_var = 3
    var_ranges = [[0.1, 0.3], [0., 5.], [0., 0.2]]
    n_samples = 27

    ss = SmartSampling(n_var, var_ranges, n_pts_axis=3)
    ss.init_sampling()
    ss.get_y()
    ss.show_samples_y()
