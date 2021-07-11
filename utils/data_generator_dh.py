import numpy as np
import pandas as pd

class DataGeneratorDH():
    def __init__(self, df_cnt, rate_prior=0.4, 
                 colname_user='idx_user', colname_item='idx_item',
                 colname_outcome='outcome',
                 colname_outcome_treated='outcome_T', colname_outcome_control='outcome_C',
                 colname_treatment='treated', colname_propensity='propensity',
                 colname_effect='causal_effect', 
                 colname_prediction='pred',
                 random_seed=1):
        self.df_cnt = df_cnt
        self.rate_prior = rate_prior
        self.colname_user = colname_user
        self.colname_item = colname_item
        self.colname_outcome = colname_outcome
        self.colname_outcome_treated = colname_outcome_treated
        self.colname_outcome_control = colname_outcome_control
        self.colname_effect = colname_effect
        self.colname_treatment = colname_treatment
        self.colname_propensity = colname_propensity
        self.colname_prediction = colname_prediction
        self.random_seed = random_seed
        self.optimal_power = None

    def prepare_prob(self, capping=0.01):
        
        self.df_data = self.df_cnt
        self.df_data.loc[:, 'num_control'] = self.df_data.loc[:, 'num_visit'] - self.df_data.loc[:, 'num_treatment']
        self.df_data.loc[:, 'num_control_outcome'] = self.df_data.loc[:, 'num_outcome'] - self.df_data.loc[:, 'num_treated_outcome']

        # get item means
        df_mean = self.df_data.loc[:, [self.colname_item, 'num_treated_outcome', 'num_control_outcome',
                                       'num_treatment', 'num_control', 'num_outcome', 'num_visit']]
        df_mean = df_mean.groupby(self.colname_item, as_index=False).mean()
        df_mean = df_mean.rename(columns={'num_treated_outcome': 'num_treated_outcome_mean',
                                          'num_control_outcome': 'num_control_outcome_mean',
                                          'num_treatment': 'num_treatment_mean',
                                          'num_control': 'num_control_mean',
                                          'num_outcome': 'num_outcome_mean',
                                          'num_visit': 'num_visit_mean'})
        # merge
        self.df_data = pd.merge(self.df_data, df_mean, on=[self.colname_item], how='left')

        self.df_data.loc[:, 'prob_outcome_treated'] = \
            (self.df_data.loc[:, 'num_treated_outcome'] + self.rate_prior * self.df_data.loc[:, 'num_treated_outcome_mean']) / \
            (self.df_data.loc[:, 'num_treatment'] + self.rate_prior * self.df_data.loc[:, 'num_treatment_mean'])
        self.df_data.loc[:, 'prob_outcome_control'] = \
            (self.df_data.loc[:, 'num_control_outcome'] + self.rate_prior * self.df_data.loc[:,'num_control_outcome_mean']) / \
            (self.df_data.loc[:, 'num_control'] + self.rate_prior * self.df_data.loc[:, 'num_control_mean'])
        self.df_data.loc[:, 'prob_outcome'] = \
            (self.df_data.loc[:, 'num_outcome'] + self.rate_prior * self.df_data.loc[:, 'num_outcome_mean']) / \
            (self.df_data.loc[:, 'num_visit'] + self.rate_prior * self.df_data.loc[:, 'num_visit_mean'])

        self.num_data = self.df_data.shape[0]
        self.num_users = np.max(self.df_data.loc[:, self.colname_user].values) + 1
        self.num_items = np.max(self.df_data.loc[:, self.colname_item].values) + 1

        self.df_data.loc[:, self.colname_propensity] = \
                (self.df_data.loc[:, 'num_treatment'] + self.rate_prior * self.df_data.loc[:,'num_treatment_mean']) / \
                (self.df_data.loc[:, 'num_visit'] + self.rate_prior * self.df_data.loc[:, 'num_visit_mean'])
        
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

 