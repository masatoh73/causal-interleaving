import numpy as np
import pandas as pd
from datetime import datetime
from recommender import NeighborBase, CausalNeighborBase, LMF, ULMF

class Ranker():
    def __init__(self, colname_user='idx_user', colname_item='idx_item',
                 colname_outcome='outcome', colname_prediction='pred',
                 colname_outcome_treated='outcome_T', colname_outcome_control='outcome_C',
                 colname_effect='causal_effect', 
                 colname_treatment='treated', colname_propensity='propensity'):
        self.colname_user = colname_user
        self.colname_item = colname_item
        self.colname_outcome = colname_outcome
        self.colname_outcome_treated =  colname_outcome_treated
        self.colname_outcome_control =  colname_outcome_control
        self.colname_effect =  colname_effect
        self.colname_prediction = colname_prediction
        self.colname_treatment = colname_treatment
        self.colname_propensity = colname_propensity

        self.default_params = {
            # model condition
            'recommender': 'ULMF',
            'dim_factor': 200,
            'with_bias': False, 

            # train condition
            'reg_common': 0.01, 
            'sd_init': 0.1,
            'reg_factor': -1.,
            'reg_bias': -1.,
            'reg_factor_j': -1.,
            'reg_bias_j': -1.,
            'learn_rate': 0.01,
            'train_metric': 'AUC',
            'ratio_nega': 0.8,
            'ratio_treatment': 0.5,
            'alpha': 0.0, # alpha for ULBPR and ULRMF

            # kNN condition 
            'measure_simil': 'cosine',
            'way_simil': 'treatment',
            'way_neighbor': 'user',
            'num_neighbor': 100,
            'way_self': 'exclude',
            'scale_similarity': 1.0,
            'normalize_similarity': False,
            'convert_count': 'log2',
            'shrinkage': 1.0,
            'shrinkage_T': -1.0,
            'shrinkage_C': -1.0,

            # eval condition
            'iter': 1
        }

    def fill_defaults(self, params):
        for k, v in self.default_params.items():
            if k in params.keys():
                if not type(self.default_params[k]) == type(params[k]):
                    if type(self.default_params[k]) == int:
                        params[k] = int(params[k])
                    elif type(self.default_params[k]) == float:
                        params[k] = float(params[k])
                    elif type(self.default_params[k]) == bool:
                        if type(params[k]) == str:
                            if params[k][0].upper() == 'T':
                                params[k] = True
                            else:
                                params[k] = False
                        elif type(params[k]) == int or type(params[k]) == float:
                            if params[k] > 0:
                                params[k] = True
                            else:
                                params[k] = False
            else:
                params[k] = v

        return params

    def get_ranking(self, df, num_rec=10):
        df = df.sort_values(by=[self.colname_user, self.colname_prediction], ascending=False)
        df_ranking = df.groupby(self.colname_user).head(num_rec)
        return df_ranking

    def get_sorted(self, df):
        df = df.sort_values(by=[self.colname_user, self.colname_prediction], ascending=False)
        return df

    def set_recommender(self, params, num_users, num_items):
        if  params['recommender'] in ['NeighborBase']:
            recommender = NeighborBase(num_users=num_users, num_items=num_items,
                                       colname_user=self.colname_user, colname_item=self.colname_item,
                                       colname_outcome=self.colname_outcome,
                                       colname_prediction=self.colname_prediction,
                                       measure_simil=params['measure_simil'],
                                       way_neighbor=params['way_neighbor'],
                                       num_neighbor=params['num_neighbor'],
                                       scale_similarity=params['scale_similarity'],
                                       shrinkage=params['shrinkage'],
                                       normalize_similarity=params['normalize_similarity'])
        elif params['recommender'] in ['CausalNeighborBase']:
            recommender = CausalNeighborBase(num_users=num_users, num_items=num_items,
                                             colname_user=self.colname_user, colname_item=self.colname_item,
                                             colname_outcome=self.colname_outcome,
                                             colname_prediction=self.colname_prediction,
                                             way_simil=params['way_simil'],
                                             measure_simil=params['measure_simil'],
                                             way_neighbor=params['way_neighbor'],
                                             num_neighbor=params['num_neighbor'],
                                             way_self=params['way_self'],
                                             shrinkage_T=params['shrinkage_T'],
                                             shrinkage_C=params['shrinkage_C'],
                                             scale_similarity=params['scale_similarity'],
                                             normalize_similarity=params['normalize_similarity'])
        elif params['recommender'] == 'LMF':
            recommender = LMF(num_users=num_users, num_items=num_items,
                              colname_user=self.colname_user, colname_item=self.colname_item,
                              colname_outcome=self.colname_outcome, colname_prediction=self.colname_prediction,
                              dim_factor=params['dim_factor'], with_bias=params['with_bias'],
                              learn_rate=params['learn_rate'],
                              sd_init=params['sd_init'] / np.sqrt(params['dim_factor']),
                              reg_factor=params['reg_factor'], reg_bias=params['reg_bias'],
                              metric=params['train_metric'], ratio_nega=params['ratio_nega'])
        elif params['recommender'] == 'ULMF':
            recommender = ULMF(num_users=num_users, num_items=num_items,
                               colname_user=self.colname_user, colname_item=self.colname_item,
                               colname_outcome=self.colname_outcome, colname_prediction=self.colname_prediction,
                               dim_factor=params['dim_factor'], with_bias=params['with_bias'],
                               learn_rate=params['learn_rate'],
                               sd_init=params['sd_init'] / np.sqrt(params['dim_factor']),
                               reg_factor=params['reg_factor'], reg_bias=params['reg_bias'],
                               metric=params['train_metric'], ratio_nega=params['ratio_nega'],
                               alpha=params['alpha'])
        else:
            pass
        print("set_recommender: {}".format(params['recommender']))
        
        return recommender

    def set_params(self, cond):
        cond_params = cond.split('+')
        params = dict()
        for cond_param in cond_params:
            conds = cond_param.split(':')
            params[conds[0]] = conds[1]

        return params

    def rank(self, params, df_train, df_test, num_users=-1, num_items=-1, num_rec=100):
        if num_users < 0:
            num_users = np.max(df_train.loc[:, self.colname_user].values) + 1
            num_users_vali = np.max(df_test.loc[:, self.colname_user].values) + 1
            if num_users_vali > num_users:
                num_users = num_users_vali
        if num_items < 0:
            num_items = np.max(df_train.loc[:, self.colname_item].values) + 1
            num_items_vali = np.max(df_test.loc[:, self.colname_item].values) + 1
            if num_items_vali > num_items:
                num_items = num_items_vali

        params = self.fill_defaults(params)
        params = self.set_common_reg(params)
        params = self.set_common_shrinkage(params)

        recommender = self.set_recommender(params, num_users, num_items)

        print("start training.")
        t_init = datetime.now()
        recommender.train(df_train, iter = params['iter'])
        t_diff = (datetime.now() - t_init)
        t_train = t_diff.seconds / 60
        print("t_train: {} min".format(t_train))

        t_init = datetime.now()
        df_test.loc[:, self.colname_prediction] = recommender.predict(df_test)
        t_diff = (datetime.now() - t_init)
        t_pred = t_diff.seconds / 60
        print("t_pred: {} min".format(t_pred))

        df_rank = df_test.loc[:, [self.colname_user,self.colname_item,self.colname_prediction, self.colname_outcome_treated, self.colname_effect]]
        if num_rec < num_items:
            df_rank = self.get_ranking(df_rank, num_rec)
        else:
            df_rank = self.get_sorted(df_rank)
        return df_rank

    def set_common_params(self, list_params, common_params):
        for n in np.arange(len(list_params)):
            for k, v in common_params.items():
                list_params[n][k] = v
        return list_params

    def set_common_shrinkage(self, params):
        if params['shrinkage_T'] < 0:
            params['shrinkage_T'] = params['shrinkage']
            print("shrinkage_T is set to {}".format(params['shrinkage']))
        if params['shrinkage_C'] < 0:
            params['shrinkage_C'] = params['shrinkage']
            print("shrinkage_C is set to {}".format(params['shrinkage']))
        return params
        
    def set_common_reg(self, params):
        if params['reg_bias'] < 0:
            params['reg_bias'] = params['reg_common']
            print("reg_bias is set to {}".format(params['reg_common']))
        if params['reg_factor'] < 0:
            params['reg_factor'] = params['reg_common']
            print("reg_factor is set to {}".format(params['reg_common']))
        if params['reg_bias_j'] < 0:
            params['reg_bias_j'] = params['reg_bias']
            print("reg_bias_j is set to {}".format(params['reg_bias']))
        if params['reg_factor_j'] < 0:
            params['reg_factor_j'] = params['reg_factor']
            print("reg_factor_j is set to {}".format(params['reg_factor']))

        return params