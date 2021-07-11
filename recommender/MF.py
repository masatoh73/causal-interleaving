import numpy as np
from recommender import Recommender
from numpy.random.mtrand import RandomState
import random

class MF(Recommender):
    def __init__(self, num_users, num_items,
                 metric='RMSE',
                 dim_factor=200, with_bias=False,
                 learn_rate = 0.01, reg_factor = 0.01, reg_bias = 0.01, sd_init = 0.1,
                 colname_user='idx_user', colname_item='idx_item',
                 colname_outcome='outcome', colname_prediction='pred',
                 colname_treatment='treated', colname_propensity='propensity'):
        super().__init__(num_users=num_users, num_items=num_items,
                         colname_user=colname_user, colname_item=colname_item,
                         colname_outcome=colname_outcome, colname_prediction=colname_prediction,
                         colname_treatment=colname_treatment, colname_propensity=colname_propensity)
        self.metric = metric
        self.dim_factor = dim_factor
        self.rng = RandomState(seed=None)
        self.with_bias = with_bias

        self.learn_rate = learn_rate
        self.reg_bias = reg_bias
        self.reg_factor = reg_factor
        self.sd_init = sd_init

        self.flag_prepared = False

        self.user_factors = self.rng.normal(loc=0, scale=self.sd_init, size=(self.num_users, self.dim_factor))
        self.item_factors = self.rng.normal(loc=0, scale=self.sd_init, size=(self.num_items, self.dim_factor))
        if self.with_bias:
            self.user_biases = np.zeros(self.num_users)
            self.item_biases = np.zeros(self.num_items)
            self.global_bias = 0.0

    # def prepare_dictionary(self, df, colname_time='idx_time'):
    #     print("start prepare dictionary")
    #     self.colname_time = colname_time
    #     self.num_times = np.max(df.loc[:, self.colname_time]) + 1
    #     self.dict_positive_sets = dict()
    #
    #     df_posi = df.loc[df.loc[:, self.colname_outcome] > 0]
    #
    #     for t in np.arange(self.num_times):
    #         df_t = df_posi.loc[df_posi.loc[:, self.colname_time] == t]
    #         self.dict_positive_sets[t] = dict()
    #         for u in np.unique(df_t.loc[:, self.colname_user]):
    #             self.dict_positive_sets[t][u] = \
    #                 np.unique(df_t.loc[df_t.loc[:, self.colname_user] == u, self.colname_item].values)
    #
    #     self.flag_prepared = True
    #     print("prepared dictionary!")


    def train(self, df, iter = 100):

        # by default, rating prediction
        # outcome = rating
        df_train = df.loc[~np.isnan(df.loc[:, self.colname_outcome]), :]

        # # in case of binary implicit feedback
        # if self.metric == 'logloss':
        #     df_train = df.loc[df.loc[:, self.colname_outcome] > 0, :]  # need only positive outcomes
        #     if not self.flag_prepared: # prepare dictionary
        #         self.prepare_dictionary(df)
        # else:
        #     df_train = df.loc[~np.isnan(df.loc[:, self.colname_outcome]), :]

        err = 0
        current_iter = 0
        while True:
            if self.metric == 'RMSE':
                df_train = df_train.sample(frac=1)
                users = df_train.loc[:, self.colname_user].values
                items = df_train.loc[:, self.colname_item].values
                outcomes = df_train.loc[:, self.colname_outcome].values

                for n in np.arange(len(df_train)):
                    u = users[n]
                    i = items[n]
                    r = outcomes[n]

                    u_factor = self.user_factors[u, :]
                    i_factor = self.item_factors[i, :]

                    rating = np.sum(u_factor * i_factor)
                    if self.with_bias:
                        rating += self.item_biases[i] + self.user_biases[u] + self.global_bias

                    coeff = r - rating
                    err += np.abs(coeff)

                    self.user_factors[u, :] += \
                        self.learn_rate * (coeff * i_factor - self.reg_factor * u_factor)
                    self.item_factors[i, :] += \
                        self.learn_rate * (coeff * u_factor - self.reg_factor * i_factor)

                    if self.with_bias:
                        self.item_biases[i] += \
                            self.learn_rate * (coeff - self.reg_bias * self.item_biases[i])
                        self.user_biases[u] += \
                            self.learn_rate * (coeff - self.reg_bias * self.user_biases[u])
                        self.global_bias += \
                            self.learn_rate * (coeff)

                    current_iter += 1
                    if current_iter >= iter:
                        return err / iter

            elif self.metric == 'logloss': # logistic matrix factorization
                df_train = df_train.sample(frac=1)
                users = df_train.loc[:, self.colname_user].values
                items = df_train.loc[:, self.colname_item].values
                outcomes = df_train.loc[:, self.colname_outcome].values

                for n in np.arange(len(df_train)):
                    u = users[n]
                    i = items[n]
                    r = outcomes[n]

                    u_factor = self.user_factors[u, :]
                    i_factor = self.item_factors[i, :]

                    rating = np.sum(u_factor * i_factor)
                    if self.with_bias:
                        rating += self.item_biases[i] + self.user_biases[u] + self.global_bias

                    if r > 0:
                        coeff = self.func_sigmoid(-rating)
                    else:
                        coeff = - self.func_sigmoid(rating)

                    self.user_factors[u, :] += \
                        self.learn_rate * (coeff * i_factor - self.reg_factor * u_factor)
                    self.item_factors[i, :] += \
                        self.learn_rate * (coeff * u_factor - self.reg_factor * i_factor)

                    if self.with_bias:
                        self.item_biases[i] += \
                            self.learn_rate * (coeff - self.reg_bias * self.item_biases[i])
                        self.user_biases[u] += \
                            self.learn_rate * (coeff - self.reg_bias * self.user_biases[u])
                        self.global_bias += \
                            self.learn_rate * (coeff)

                    current_iter += 1
                    if current_iter >= iter:
                        return err / iter



    def predict(self, df):
        users = df[self.colname_user].values
        items = df[self.colname_item].values
        pred = np.zeros(len(df))
        for n in np.arange(len(df)):
            pred[n] = np.inner(self.user_factors[users[n], :], self.item_factors[items[n], :])
            if self.with_bias:
                pred[n] += self.item_biases[items[n]]
                pred[n] += self.user_biases[users[n]]
                pred[n] += self.global_bias

        if self.metric == 'logloss':
            pred = 1 / (1 + np.exp(-pred))
        return pred
