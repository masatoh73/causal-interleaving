
import numpy as np
import pandas as pd
from recommender import MF

# data generation from movielens
class DataGeneratorML():
    def __init__(self, 
                 colname_user='idx_user', colname_item='idx_item',
                 colname_outcome='outcome',
                 colname_outcome_treated='outcome_T', colname_outcome_control='outcome_C',
                 colname_treatment='treated', colname_propensity='propensity',
                 colname_effect='causal_effect', colname_expectation='causal_effect_expectation',
                 colname_prediction='pred',
                 random_seed=1):

        self.colname_user = colname_user
        self.colname_item = colname_item
        self.colname_outcome = colname_outcome
        self.colname_outcome_treated = colname_outcome_treated
        self.colname_outcome_control = colname_outcome_control
        self.colname_effect = colname_effect
        self.colname_expectation = colname_expectation
        self.colname_treatment = colname_treatment
        self.colname_propensity = colname_propensity
        self.colname_prediction = colname_prediction
        self.random_seed = random_seed
        self.optimal_power = None


    def load_data(self, version_of_movielens):

        if version_of_movielens == '100k':
            self.df_raw = pd.read_table('data_raw/u.data',
                                   names=('idx_user', 'idx_item', 'rating', 'timestamp'),
                                        sep='\t')
            self.df_raw = self.df_raw.drop(columns='timestamp')

        elif version_of_movielens == '1m':
            self.df_raw = pd.read_table('data_raw/ratings.dat',
                                   names=('idx_user', 'idx_item', 'rating', 'timestamp'),
                                        sep='::')
            self.df_raw = self.df_raw.drop(columns='timestamp')
            
        # force the index to start from 0
        if np.min(self.df_raw.loc[:, 'idx_user']) == 1:
            self.df_raw.loc[:, 'idx_user'] = self.df_raw.loc[:, 'idx_user'] - 1
        if np.min(self.df_raw.loc[:, 'idx_item']) == 1:
            self.df_raw.loc[:, 'idx_item'] = self.df_raw.loc[:, 'idx_item'] - 1

        self.num_data_raw = self.df_raw.shape[0]
        self.num_users = np.max(self.df_raw.loc[:, self.colname_user].values) + 1
        self.num_items = np.max(self.df_raw.loc[:, self.colname_item].values) + 1
        self.df_raw.loc[:, 'watch'] = 1
        self.df_raw.loc[:, 'idx_time'] = 0

        df_data = pd.DataFrame(
            {'idx_user': np.repeat(np.arange(self.num_users), self.num_items),
             'idx_item': np.tile(np.arange(self.num_items), self.num_users)})

        df_data = pd.merge(df_data, self.df_raw, on=[self.colname_user, self.colname_item], how='left')
        df_data = df_data.fillna({'watch': 0})
        df_data = df_data.fillna({'idx_time': 0})
        
        self.df_data = df_data
        self.num_data = self.df_data.shape[0]
        

    def predict_rating(self, iter, dim_factor, learn_rate, reg_factor, reg_bias=0.0):
        recommender = MF(num_users=self.num_users, num_items=self.num_items,
                         colname_user=self.colname_user, colname_item=self.colname_item,
                         colname_outcome='rating', colname_prediction=self.colname_prediction,
                         dim_factor=dim_factor, with_bias=False,
                         learn_rate=learn_rate,
                         sd_init=0.1 / np.sqrt(dim_factor),
                         reg_factor=reg_factor, reg_bias=reg_bias,
                         metric='RMSE')

        recommender.train(self.df_raw, iter=iter)
        self.df_data.loc[:, 'pred_rating'] = recommender.predict(self.df_data)


    def predict_watch(self, iter, dim_factor, learn_rate, reg_factor, reg_bias=0.0):
        recommender = MF(num_users=self.num_users, num_items=self.num_items,
                         colname_user=self.colname_user, colname_item=self.colname_item,
                         colname_outcome='watch', colname_prediction=self.colname_prediction,
                         dim_factor=dim_factor, with_bias=False,
                         learn_rate=learn_rate,
                         sd_init=0.1 / np.sqrt(dim_factor),
                         reg_factor=reg_factor, reg_bias=reg_bias,
                         metric='logloss')

        recommender.train(self.df_data, iter=iter)
        self.df_data.loc[:, 'pred_watch'] = recommender.predict(self.df_data)


    # prob_outcome_treated, prob_outcome_control
    def set_prob_outcome_treated(self, steepness=1.0, offset=5.0):
        self.df_data.loc[:, 'prob_outcome_treated'] = 1.0/(1.0 + np.exp(- steepness * (self.df_data.loc[:, 'pred_rating'] - offset)))

    def set_prob_outcome_control(self, scaling_outcome=1.0):
        self.df_data.loc[:, 'prob_outcome_control'] = np.power(self.df_data.loc[:, 'pred_watch'], scaling_outcome)

    # set propensity
    def assign_propensity(self, capping = 0.01, mode='uniform', scaling_propensity=1.0, num_rec=100, df_train=None):

        if mode == 'uniform':
            self.df_data.loc[:, self.colname_propensity] = num_rec/self.num_items

        else:
            df = self.df_data
            df.loc[:, self.colname_prediction] = df.loc[:, 'prob_outcome_control']/2 + df.loc[:, 'prob_outcome_treated']/2
            # get ranking
            df = df.sort_values(by=[self.colname_user, self.colname_prediction], ascending=False)
            df.loc[:, 'rank'] = np.tile(np.arange(self.num_items) + 1, self.num_users)
            
            # scaling
            if mode == 'rank':
                df.loc[:, self.colname_propensity] = 1.0 / np.power(df.loc[:, 'rank'], scaling_propensity)
                sum_propensity = np.sum(1.0 / np.power(np.arange(self.num_items) + 1, scaling_propensity))
            elif mode == 'logrank':
                df.loc[:, self.colname_propensity] = 1.0 / np.power(np.log2(df.loc[:, 'rank'] + 1), scaling_propensity)
                sum_propensity = np.sum(1.0 / np.power(np.log2(np.arange(self.num_items) + 2), scaling_propensity))

            df.loc[:, self.colname_propensity] /= sum_propensity
            df.loc[:, self.colname_propensity] *= num_rec

            while True:
                df.loc[df.loc[:, self.colname_propensity] > 1, self.colname_propensity] = 1.0
                total_num_rec = np.sum(df.loc[:, self.colname_propensity])
                avg_num_rec = total_num_rec/self.num_users
                
                if round(avg_num_rec) < num_rec:
                    df.loc[:, self.colname_propensity] = df.loc[:, self.colname_propensity] * num_rec/avg_num_rec
                else:
                    break
            self.df_data = df

        if capping is not None:
            self.df_data.loc[self.df_data.loc[:, self.colname_propensity] < capping, self.colname_propensity] = capping
            self.df_data.loc[self.df_data.loc[:, self.colname_propensity] > 1 - capping, self.colname_propensity] = 1 - capping


    def assign_treatment(self):
        self.df_data.loc[:, self.colname_treatment] = 0
        bool_treatment = self.df_data.loc[:, self.colname_propensity] > np.random.rand(self.num_data)
        self.df_data.loc[bool_treatment, self.colname_treatment] = 1

    def assign_outcome(self):
        self.df_data.loc[:, self.colname_outcome] = 0
        prob = np.random.rand(self.num_data)
        self.df_data.loc[:, self.colname_outcome_treated] = (self.df_data.loc[:, 'prob_outcome_treated'] >= prob) * 1.0
        prob = np.random.rand(self.num_data)
        self.df_data.loc[:, self.colname_outcome_control] = (self.df_data.loc[:, 'prob_outcome_control'] >= prob) * 1.0

        self.df_data.loc[:, self.colname_outcome] = \
            self.df_data.loc[:, self.colname_treatment] * self.df_data.loc[:, self.colname_outcome_treated] + \
            (1 - self.df_data.loc[:, self.colname_treatment]) * self.df_data.loc[:, self.colname_outcome_control]
        self.df_data.loc[:, self.colname_effect] = \
            self.df_data.loc[:, self.colname_outcome_treated] - self.df_data.loc[:,self.colname_outcome_control]

    def get_observation(self): # only observable variables
        return self.df_data.loc[:, [self.colname_user, self.colname_item, self.colname_treatment, self.colname_outcome, self.colname_propensity]]

    def get_ground_truth(self): # include unobservable ground truth
        return self.df_data.loc[:, [self.colname_user, self.colname_item, self.colname_treatment, self.colname_outcome, self.colname_propensity, self.colname_effect, self.colname_outcome_treated, self.colname_outcome_control]]
