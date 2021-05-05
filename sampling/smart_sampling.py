import pandas as pd
import numpy as np
import random
import math

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, r2_score

from sampling.sampling import Sampling
from sampling.grid_sampling import GridSampling
from meta_model import MetaModel


def select_new_pt(selected_pt, num):
    # x1 : [0.1, 0.3]
    # v : [0, 5]
    # c : [0, 0.2]
    # var_ranges = [[0.1, 0.2], [0., 5.], [0., 0.15]]

    Max_L = 0.2
    Min_L = 0.1
    Max_v = 5
    Min_v = 0
    Max_c = 0.15
    Min_c = 0

    div_thres = 10
    dist_L = (Max_L - Min_L) / div_thres
    dist_v = (Max_v - Min_v) / div_thres
    dist_c = (Max_c - Min_c) / div_thres

    L_range_min = selected_pt[0] - dist_L
    L_range_max = selected_pt[0] + dist_L
    v_range_min = selected_pt[1] - dist_v
    v_range_max = selected_pt[1] + dist_v
    c_range_min = selected_pt[1] - dist_c
    c_range_max = selected_pt[1] + dist_c

    if L_range_min < Min_L:
        L_range_min = Min_L
    if L_range_max > Max_L:
        L_range_max = Max_L

    if v_range_min < Min_v:
        v_range_min = Min_v
    if v_range_max > Max_v:
        v_range_max = Max_v

    if c_range_min < Min_c:
        c_range_min = Min_c
    if c_range_max > Max_c:
        c_range_max = Max_c

    new_pt = pd.DataFrame()
    for i in range(num):
        new_l = random.uniform(L_range_min, L_range_max)
        new_v0 = random.uniform(v_range_min, v_range_max)
        new_c = random.uniform(c_range_min, c_range_max)
        new_pt = new_pt.append([[new_l, new_v0, new_c]], ignore_index=True)

    new_pt.columns = ['L', 'v0', 'C']
    return new_pt


def committee(data, n_tree, m_depth):
    qbc = data.sample(n=len(data), replace=True)

    x_qbc = qbc[['L', 'v0', 'C', 't']]
    y_qbc = qbc['angle']

    regressor = ExtraTreesRegressor(n_estimators=n_tree, max_depth=m_depth)
    regressor.fit(x_qbc, y_qbc)
    return regressor


def r2_rmse(train, test):
    train_x = train[['L', 'v0', 'C', 't']]
    train_y = train['angle']
    test_x = test[['L', 'v0', 'C', 't']]
    test_y = test['angle']

    regressor = ExtraTreesRegressor(n_estimators=100, random_state=42)
    regressor.fit(train_x, train_y)
    y_pred = regressor.predict(test_x)
    rmse = mean_squared_error(test_y, y_pred, squared=False)
    r2 = r2_score(test_y, y_pred)
    return r2, rmse


class SmartSampling(Sampling):
    # init_samples = []
    # n_pts_axis = 2
    # n_init_samples = 8

    def __init__(self, n_var, var_ranges, n_samples=None, n_pts_axis=2, function="single_pendulum"):
        # n_pts_axis is the number of points in each axis (variable)
        # if n_pts_axis = 2, only use the min and max values
        super().__init__(n_var, var_ranges, n_samples)
        self.n_var = n_var
        self.var_ranges = var_ranges
        self.n_samples = n_samples
        self.n_pts_axis = n_pts_axis
        self.function = function
        self.n_init_samples = n_pts_axis**n_var
        self.time_range = 201

    def sampling(self):
        # 초기 샘플 개수 설정
        self.init_n_pts_axis = 3
        self.init_n_samples = self.init_n_pts_axis ** self.n_var

        # innital sampling
        self.init_sampling()

        # sequential sampling
        time_range = 201
        for _ in range(self.init_n_samples, self.n_samples):
            self.sequential_sampling()
            if int(len(self.train)/time_range) >= self.n_samples:
                break

    def init_sampling(self):
        # inital sampling
        grid_s = GridSampling(self.n_var, self.var_ranges, self.init_n_samples, self.init_n_pts_axis, self.function)
        grid_s.sampling()
        grid_s.show_samples()
        # grid_s.write_samples_to_file()
        self.init_samples = grid_s.samples
        self.samples = grid_s.samples

        self.train = self.generate_train_data()
        self.test()

    def sequential_sampling(self):
        # LOO, time_range 고정
        var_mean = self.LOO(self.train, self.time_range)
        print('\nResult of LOO :')
        # display(var_mean)

        # 박스칠 init 선택, rmse prob으로 선택, 0.2=박스칠 갯수, rate
        n_add_samples = min(
            int (math.ceil(len(var_mean) * 0.2)),
            int (self.n_samples - len(self.train)/self.time_range)
        )
        select_init = np.random.choice(var_mean.index, n_add_samples, [var_mean['probability']])
        select_init = sorted(set(select_init))

        candidates = pd.DataFrame()
        for pt in select_init:
            print(pt)

            # 박스내 후보군 한개 생성 = select_new_pt , I=1 ,(L,V0,C)
            new_pt = select_new_pt(pt, 1)
            candidates = candidates.append(new_pt, ignore_index=True)

        # L,V,C,->T,ACNGLE,VELOCITY,ACCEL->L,V,C,T
        ss = Sampling(
            self.n_var, self.var_ranges, self.n_samples,
            # function=function
        )
        # ss.sampling()
        # ss.show_samples()
        ss.samples = candidates.values.tolist()  # DataFrame to List
        candidates_time = ss.generate_train_data()

        # candidates_time = get_ready_for_trn_df(candidates)
        candi_x = candidates_time[['L', 'v0', 'C', 't']]
        # print('박스내 후보군')
        # display(candi_x)

        # qbc committee 3개 생성 , extra tree , 파라미터 변경
        # init_samples_df = pd.DataFrame(self.init_samples)
        qbc1 = committee(self.train, 100, 7)
        qbc2 = committee(self.train, 100, 8)
        qbc3 = committee(self.train, 100, 9)

        # QBC
        varaince = []
        n = 0

        candidates = candi_x.drop_duplicates(['L', 'v0', 'C'])
        candidates.reset_index(drop=True, inplace=True)
        for i in range(len(candidates)):
            x_loo = candi_x[n:n + self.time_range]
            pred1 = qbc1.predict(x_loo)
            pred2 = qbc2.predict(x_loo)
            pred3 = qbc3.predict(x_loo)
            p = [pred1, pred2, pred3]
            v = np.var(p)

            varaince.append(v)
            n += self.time_range

        Var = pd.DataFrame(data=varaince, columns=['var'])
        can_prior = pd.merge(candidates, Var, left_index=True, right_index=True)

        can_prior['probability'] = can_prior['var'] / sum(can_prior['var'])
        can_prior['p_root'] = (can_prior['var'] / sum(can_prior['var'])) ** 0.5
        can_prior['p_square'] = (can_prior['var'] / sum(can_prior['var'])) ** 2

        can_prior.sort_values(by=['var'], ascending=False, inplace=True)
        can_prior.reset_index(drop=True, inplace=True)

        # print('qbc 결과')
        # display(can_prior)

        # 후보군 중에서 분산이 가장 큰 top1 한개 선택 후 * time
        # top_1=list(can_prior.iloc[0][['L','v0','C']])
        # top_1=pd.DataFrame([top_1],columns=['L','v0','C'])
        #     new_points = get_ready_for_trn_df(top_1)
        #     init=init.append(new_points,ignore_index=True)
        #     NEW_POINTS=NEW_POINTS.append(new_points,ignore_index=True)

        # 후보군 중에서 prob으로 한개 선택 후 * time (probability, p_root,p_square)
        select_index = np.random.choice(can_prior.index, 1, [can_prior['p_root']])
        top_1 = list(can_prior.loc[select_index[0]][['L', 'v0', 'C']])
        top_1 = pd.DataFrame([top_1], columns=['L', 'v0', 'C'])

        # new_points = get_ready_for_trn_df(top_1)
        ss.samples = top_1.values.tolist()
        new_points = ss.generate_train_data()

        self.train = self.train.append(new_points) #, ignore_index=True)
        # NEW_POINTS = NEW_POINTS.append(new_points, ignore_index=True)
        # display(new_points)

        self.test(ss)

    # def show_init_samples(self):
    #     print("\nInitial Samles : n_init_samples=", self.n_init_samples)
    #     for i in range(len(self.init_samples)):
    #         print(i, self.init_samples[i])

    def test(self):
        test_file = 'data//single_test1331.csv'
        test = pd.read_csv(test_file)
        r2, rmse = r2_rmse(self.train, test)

        print('num of experiments ', int(len(self.train)/self.time_range), len(self.train), 'r2 : ', r2, 'rmse: ', rmse)

    def model_ET(self, train_x, train_y, test_x, test_y):
        regressor = ExtraTreesRegressor(n_estimators=100, random_state=42)
        regressor.fit(train_x, train_y)
        y_pred = regressor.predict(test_x)
        rmse = mean_squared_error(test_y, y_pred, squared=False)
        r2 = r2_score(test_y, y_pred)

        return r2, rmse

    def LOO(self, init, time_range):
        n=0
        RMSE = pd.DataFrame()
        init.reset_index(inplace=True, drop=True)
        n_exps = int(init.shape[0] / time_range)
        for i in range(n_exps):
            x_true = init[['L', 'v0', 'C', 't']]
            y_true = init['angle']

            sub_x = x_true.drop(x_true.index[n:n + time_range])
            sub_y = y_true.drop(y_true.index[n:n + time_range])

            # sub를 제외한 나머지 labeled data로 모델 생성
            r2, rmse = self.model_ET(sub_x, sub_y, x_true, y_true)
            RMSE = RMSE.append([rmse], ignore_index=True)
            n += time_range

            if n >= len(init):
                break

        RMSE.columns = ['rmse']
        # init_diff = init.drop_duplicates()
        init_diff = init.drop_duplicates(['L', 'v0', 'C'])
        init_diff3 = init_diff[['L', 'v0', 'C']]
        init_diff3.reset_index(inplace=True, drop=True)
        # var_df = pd.merge(init_diff, MSE, left_index=True, right_index=True)
        var_df = pd.concat([init_diff3, RMSE], axis=1)
        var_df['probability'] = var_df['rmse'] / sum(var_df['rmse'])
        var_df.sort_values(by=['rmse'], ascending=False, inplace=True)
        var_df.reset_index(drop=True, inplace=True)
        var_mean = var_df.groupby(['L','v0','C']).mean()

        return var_mean


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
