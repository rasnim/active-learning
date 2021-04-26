import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.datasets import make_friedman2
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from pykrige.rk import RegressionKriging
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from matplotlib import pyplot
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, LeakyReLU


class MetaModel():
    init_file = './/data//single_init_27.csv'
    test_file = './/data//single_pendulum_test.csv'

    def __init__(self, train, test, n_est = 100, function="single_pendulum"):
        self.n_est = n_est
        if function == "single_pendulum":
            self.n_var = 4
            self.train_x = train[['L', 'v0', 'C', 't']]
            self.train_y = train['angle']
            self.test_x = test[['L', 'v0', 'C', 't']]
            self.test_y = test['angle']
            self.init_file = './/data//single_init_27.csv'
            self.test_file = './/data//single_pendulum_test.csv'
        elif function == "double_pendulum":
            self.n_var = 5
            self.train_x = train[['L1', 'L2', 'v1', 'v2', 't']]
            self.train_y = train['angle1']
            self.test_x = test[['L1', 'L2', 'v1', 'v2', 't']]
            self.test_y = test['angle1']
            self.init_file = './/data//double_init_27.csv'
            self.test_file = './/data/double_pendulum_test.csv'

    def model_random_forest(self):
        model = RandomForestRegressor(n_estimators=self.n_est) #, random_state=42)
        model.fit(self.train_x, self.train_y)

        self.y_pred = model.predict(self.test_x)

    def model_extra_tree(self):
        model = ExtraTreesRegressor(n_estimators=self.n_est) #, random_state=42)
        model.fit(self.train_x, self.train_y)

        self.y_pred = model.predict(self.test_x)

    def model_xgboost(self):
        model = XGBRegressor(
            n_estimators=self.n_est, random_state=4,
            # max_depth=8, learning_rate=0.01,
            # eta=0.1, subsample=0.7, colsample_bytree=0.8
        )

        model.fit(self.train_x, self.train_y)
        model.score(self.train_x, self.train_y)
        self.y_pred = model.predict(self.test_x)

    def model_lightgbm(self):
        hyper_params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': ['rmse'], #['l2', 'auc'],
            'learning_rate': 0.05,
            # 'feature_fraction': 0.9,
            # 'bagging_fraction': 0.7,
            # 'bagging_freq': 10,
            # 'verbose': 0,
            # "max_depth": 8,
            # "num_leaves": 128,
            # "max_bin": 512,
            # "num_iterations": 100000,
            "n_estimators": self.n_est
        }
        model = LGBMRegressor(**hyper_params)

        model.fit(self.train_x, self.train_y)
        model.score(self.train_x, self.train_y)
        self.y_pred = model.predict(self.test_x)

    def model_gaussian_process(self):
        # kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
        kernel = DotProduct() + WhiteKernel()
        # gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
        model = GaussianProcessRegressor(kernel=kernel) #, random_state=0)

        model.fit(self.train_x, self.train_y)
        model.score(self.train_x, self.train_y)
        self.y_pred, sigma = model.predict(self.test_x, return_std=True)

        # plt.figure()
        # # plt.plot(self.test_x, f(x), 'r:', label=r'$f(x) = x\,\sin(x)$')
        # plt.plot(self.test_x, self.test_y, 'r.', markersize=10, label='Observations')
        # plt.plot(self.test_x, self.y_pred, 'b-', label='Prediction')
        # plt.fill(np.concatenate([self.test_x, self.test_x[::-1]]),
        #          np.concatenate([self.y_pred - 1.9600 * sigma,
        #                          (self.y_pred + 1.9600 * sigma)[::-1]]),
        #          alpha=.5, fc='b', ec='None', label='95% confidence interval')
        # plt.xlabel('$x$')
        # plt.ylabel('$f(x)$')
        # plt.ylim(-10, 20)
        # plt.legend(loc='upper left')

    def model_dnn(self):
        # define the layers
        x_in = Input(shape=(self.n_var,))

        epochs = 2000
        # act_ftn = 'relu'
        act_ftn = LeakyReLU(alpha=0.1)
        x = Dense(128, activation=act_ftn)(x_in)
        x = Dense(128, activation=act_ftn)(x)
        x = Dense(64, activation=act_ftn)(x)
        x = Dense(64, activation=act_ftn)(x)

        # x = Dense(64, activation=act_ftn)(x_in)
        # x = Dense(64, activation=act_ftn)(x)

        x_out = Dense(1)(x)

        # define the model
        model = Model(inputs=x_in, outputs=x_out)
        model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae', 'mape'])

        # fit the model
        history = model.fit(self.train_x, self.train_y, epochs=epochs, batch_size=len(self.train_x), verbose=0)

        # plot metrics
        pyplot.plot(history.history['mse'])
        pyplot.plot(history.history['mae'])
        # pyplot.plot(history.history['mape'])
        pyplot.show()

        self.y_pred = model.predict(self.test_x)

    def evaluate(self):
        mse = mean_squared_error(self.test_y, self.y_pred, squared=True)
        rmse = mse**0.5
        mae = mean_absolute_error(self.test_y, self.y_pred)
        mape = mean_absolute_percentage_error(self.test_y, self.y_pred)
        r2 = r2_score(self.test_y, self.y_pred)

        return r2, rmse, mae, mape
