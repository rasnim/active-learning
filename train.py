import pandas as pd
import os.path
from datetime import datetime

from sampling.grid_sampling import GridSampling
from sampling.random_sampling import RandomSampling
from sampling.smart_sampling import SmartSampling
from meta_model import MetaModel

# experimental design
# function = "single_pendulum"
function = "double_pendulum"

exp_grid = True
exp_rand = False
exp_smart = False

if function == "single_pendulum":
    # L = [0.1, 0.2]    : length of the messless rod
    # v0 = [0, 5]       : initial angle velocity
    # c = [0, 0.15]      : damping coefficient => 논문에는 [0, 1] (Fig.3 캡션 참고)
    # t = [0, 2]        : 0.02 간격으로 101개 => 논문에는 0.01 간격으로 201개 (Table 5 참고)
    # test sample 개수 = 7^3 * 101 = 343 * 101 = 34,643
    n_var = 3
    var_ranges = [[0.1, 0.2], [0., 5.], [0., 0.15]]
    init_file = 'data//single_init_27.csv'
    test_file = 'data//single_test1331.csv'
    result_file = 'result//single_pendulum_result.csv'

elif function == "double_pendulum":
    # L1 = [1, 2],  L2 = [2, 3]
    # v1 = [0, .1], v2 = [.3, .5]
    # t = [0, 5]        : 0.05 간격으로 101개 => 논문에는 0.01 간격으로 501개 (Table 9 참고)
    # test sample 개수 = 7^4 * 101 = 2401 * 101 = 242,501
    n_var = 4
    var_ranges = [[1, 2], [2, 3], [0., .1], [.3, .5]]
    init_file = 'data//double_init_27.csv'
    test_file = 'data/double_pendulum_test.csv'
    result_file = 'result//double_pendulum_result.csv'


def main(n_pts_axis_, n_samples_, n_est_=100, model_num=1, dnn_param=['64-2', 'relu', 2000]):
    global r2, rmse, mae, mape, n_samples, exp
    n_samples = n_samples_

    train_data = pd.DataFrame()

    # ss = Sampling(n_var, var_ranges, n_samples, function=function)

    if exp_grid:
        exp = 'grid'
        ss = GridSampling(
            n_var, var_ranges, n_samples,
            n_pts_axis=n_pts_axis_,  # n_pts_axis=3 selects three points in each axis
            function=function
        )
        ss.sampling()
        # grid_s.show_samples()
        train = ss.generate_train_data()
        n_samples = len(train)

    elif exp_rand:
        exp = 'rand'
        ss = RandomSampling(
            n_var, var_ranges, n_samples,
            function=function
        )
        ss.sampling()
        # ss.show_samples()
        train = ss.generate_train_data()
        n_samples = len(train)

        # Test file 만들 떄 사용하는 코드
        # file_name = test_file.split(".")[0] +"-"+ str(n_pts_axis_)+".csv"
        # if os.path.exists(file_name):
        #     os.remove(file_name)
        # train.to_csv(file_name, index=False, header=True)

    elif exp_smart:
        exp = 'rand_box'
        ss = SmartSampling(
            n_var, var_ranges, n_samples,
            n_pts_axis=n_pts_axis_,
            function=function
        )
        ss.sampling()
        # ss.show_samples()
        train = ss.generate_train_data()
        n_samples = len(train)

    # 테스트 데이터 읽기
    test = pd.read_csv(test_file)
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
        mm.model_dnn(dnn_param=dnn_param)

    r2, rmse, mae, mape = mm.evaluate()
    # print("r2 = ", r2, ", rmse = ", rmse)


def switch_model(model_num):
    model_dic = {
        0: "RandomForest", 1: "ExtraTree", # [Bagging]
        2: "XGBoost", 3: "LightGBM",         # [Boosting]
        4: "GaussianProcess", 5: "DNN"      # [others]
    }
    return model_dic.get(model_num, "Invalid model")


if __name__ == "__main__":
    start_time0 = datetime.now()
    print(start_time0)

    header = "exp, model, n_samples, n_pts_axis, n_exps, r2, rmse, mae, mape, exe_time, param"
    if os.path.exists(result_file):
        f = open(result_file, "a+")
    else:
        f = open(result_file, "a+")
        f.write(header + "\n")

    print(function)

    # for random or grid sampling
    model_nums = [3] # [0,1,2,3]
    n_ests = [200]
    n_pts_axis_set = [7]

    # # for random or grid sampling (only for deep learning)
    # model_nums = [5]
    # n_ests = [0]
    # n_pts_axis_set = [2] #, 3, 4, 5, 6, 7]
    #
    # act_fnts = ['relu']  # 'leaky',
    # net_strs = ['128-2'] #, '128-4']
    # epochs = [2000] #, 2000]

    # for smart sampling
    # model_nums = [1]    # for QBC model
    # n_ests = [100]      # for QBC model
    # n_pts_axis_set = [4] # num of sequential samples

    # Smart Sampling으로 Single Pendulum에서 4^3 및 5^3개 n_exps를 수행하여,
    # ExtraTree의 정확도를 98%까지 향상시키는 것을 목표로 함!
    for model_num in model_nums: # [0,1,2,3,5]:
        cnt = 0
        model = switch_model(model_num)
        for n_est in n_ests:
            # header = "exp, model, n_samples, n_pts_axis, n_exps, r2, rmse, mae, mape, exe_time, param"
            # print("\nmodel=", model, ", n_est=", n_est, "\n", header)
            for n_pts_axis in n_pts_axis_set: #[2,3,4,5,6,7]:
                # if os.path.exists(result_file):
                #     f = open(result_file, "a+")
                # else:
                #     f = open(result_file, "a+")
                #     f.write(header + "\n")

                # print(start_time)
                n_exps = n_pts_axis ** n_var
                if model_num != 5:
                    start_time = datetime.now()
                    main(n_pts_axis, n_exps, n_est, model_num)  # 핵심 구문

                    end_time = datetime.now()
                    exe_time = end_time - start_time
                    res = [exp, model, n_pts_axis, n_exps, n_samples, r2, rmse, mae, mape, exe_time,
                           "n_est=" + str(n_est)]
                    print(','.join(map(str, res)))
                    f.write(','.join(map(str, res)) + "\n")
                    f.flush()

                else:  # dnn의 경우
                    for activate in act_fnts:  # activation function
                        for structure in net_strs:   # network structure
                            for epoch in epochs:     # num of epoches
                                start_time = datetime.now()

                                dnn_param = [structure, activate, epoch]
                                main(n_pts_axis, n_exps, n_est, model_num, dnn_param=dnn_param)

                                end_time = datetime.now()
                                exe_time = end_time - start_time
                                res = [exp, model, n_pts_axis, n_exps, n_samples, r2, rmse, mae, mape, exe_time,
                                       "dnn_param="+str(dnn_param).replace(',','|').replace(' ','')]
                                print(','.join(map(str, res)))
                                f.write(','.join(map(str, res)) + "\n")
                                f.flush()

            if model_num == 5 and cnt > 0:
                break
            cnt += 1

        if model_num == 5 and cnt > 0:
            break

    f.close()
    end_time = datetime.now()
    exe_time = end_time - start_time0
    print()
    print(start_time0)
    print(end_time)
    print(exe_time)
