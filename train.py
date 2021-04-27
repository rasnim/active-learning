import pandas as pd
from datetime import datetime

from sampling.sampling import Sampling
from sampling.grid_sampling import GridSampling
from sampling.random_sampling import RandomSampling
from sampling.smart_sampling import SmartSampling
from meta_model import MetaModel

# experimental design
# n_pts_axis = 2
# n_samples = 8
n_var = 4
# function = "single_pendulum"
function = "double_pendulum"

exp_grid = True
exp_rand = False
exp_smart = False

# evaluation metrics
# r2 = 0.
# rmse = 0.

def main(n_pts_axis_, n_samples_, n_est_=100, model_num=1):
    global r2, mae, rmse, mape, n_pts_axis, n_samples, n_var
    n_pts_axis = n_pts_axis_
    n_samples = n_samples_

    if function == "single_pendulum":
        # L = [0.1, 0.3]    : length of the messless rod
        # v0 = [0, 5]       : initial angle velocity
        # c = [0, 0.2]      : damping coefficient => 논문에는 [0, 1] (Fig.3 캡션 참고)
        # t = [0, 2]        : 0.02 간격으로 101개 => 논문에는 0.01 간격으로 201개 (Table 5 참고)
        # test sample 개수 = 7^3 * 101 = 343 * 101 = 34,643
        n_var = 3
        var_ranges = [[0.1, 0.3], [0., 5.], [0., 0.2]]
    elif function == "double_pendulum":
        # L1 = [1, 2],  L2 = [2, 3]
        # v1 = [0, .1], v2 = [.3, .5]
        # t = [0, 5]        : 0.05 간격으로 101개 => 논문에는 0.01 간격으로 501개 (Table 9 참고)
        # test sample 개수 = 7^4 * 101 = 2401 * 101 = 242,501
        n_var = 4
        var_ranges = [[1, 2], [2, 3], [0., .1], [.3, .5]]

    train_data = pd.DataFrame()

    ss = Sampling(n_var, var_ranges, n_samples, function=function)

    if exp_grid:
        grid_s = GridSampling(
            n_var, var_ranges, n_samples,
            n_pts_axis=n_pts_axis,  # n_pts_axis=3 selects three points in each axis
            function=function
        )
        grid_s.sampling()
        # grid_s.show_samples()
        train = grid_s.generate_train_data()
        n_samples = len(train)/101

    elif exp_rand:
        rd_s = RandomSampling(
            n_var, var_ranges, n_samples,
            function=function
        )
        rd_s.sampling()
        # rd_s.show_samples()
        train = rd_s.generate_train_data()

        if function == "single_pendulum":
            init_file = './/data//single_init_27.csv'
            test_file = './/data//single_pendulum_test.csv'
        elif function == "double_pendulum":
            init_file = './/data//double_init_27.csv'
            test_file = './/data/double_pendulum_test.csv'
        # train.to_csv(test_file, index=False, header=True)  # Test file 만들 떄 사용하는 코드

    elif exp_smart:
        ss = SmartSampling(
            n_var, var_ranges, n_samples,
            n_pts_axis=n_pts_axis,
            function=function
        )
        # 초기 샘플 생성
        # ss.init_sampling()
        # ss.get_y()

        # 초기 샘플 생성 대신에 읽기
        ss.read_init_sample_file()
        # print(ss.train)

    # 테스트 데이터 읽기
    test = ss.read_test_file()
    # print(ss.test)

    # 학습 및 평가
    mm = MetaModel(train, test, n_est_, function=function)

    # 학습모델 선택
    if model_num == 0:
        mm.model_random_forest()
    elif model_num == 1:
        mm.model_extra_tree()
    elif model_num == 2:
        mm.model_xgboost()
    elif model_num == 3:
        mm.model_lightgbm()
    elif model_num == 4:
        mm.model_gaussian_process()
    elif model_num == 5:
        mm.model_dnn()

    r2, rmse, mae, mape = mm.evaluate()
    # print("r2 = ", r2, ", rmse = ", rmse)


if __name__ == "__main__":
    start_time0 = datetime.now()

    print(function)
    # 0:random_forest, 1:extra_tree [Bagging]
    # 2:xgboost, 3:lightgbm         [Boosting]
    # 4:gaussian_process, 5:dnn     [others]
    for model_num in [1]:
        for n_est in [100]: #, 200, 1000, 2000]:
            print("\nmodel=", model_num, "n_est=", n_est)
            print("n_exps, n_pts_axis, r2, rmse, mae, mape, exe_time")
            for n_pts_axis_ in [2,3,4,5,6,7]:
                start_time = datetime.now()
                main(n_pts_axis_, n_pts_axis_**n_var, n_est, model_num)
                end_time = datetime.now()
                exe_time = end_time - start_time
                print(n_samples, n_pts_axis, r2, rmse, mae, mape, exe_time)

    end_time = datetime.now()
    exe_time = end_time - start_time0
    print()
    print(start_time0)
    print(end_time)
    print(exe_time)
