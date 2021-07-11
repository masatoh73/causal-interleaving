import numpy as np
import random

class Recommender(object):

    def __init__(self, num_users, num_items,
                 colname_user = 'idx_user', colname_item = 'idx_item',
                 colname_outcome = 'outcome', colname_prediction='pred',
                 colname_treatment='treated', colname_propensity='propensity'):
        super().__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.colname_user = colname_user
        self.colname_item = colname_item
        self.colname_outcome = colname_outcome
        self.colname_prediction = colname_prediction
        self.colname_treatment = colname_treatment
        self.colname_propensity = colname_propensity


    def train(self, df, iter=100):
        pass

    def predict(self, df):
        pass

    def recommend(self, df, num_rec=10):
        pass

    def func_sigmoid(self, x):
        if x >= 0:
            return 1.0 / (1.0 + np.exp(-x))
        else:
            return np.exp(x) / (1.0 + np.exp(x))

    


