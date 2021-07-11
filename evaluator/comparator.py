import numpy as np
import pandas as pd
from scipy import stats
import random
from datetime import datetime

class Comparator():

    def __init__(self, 
                 colname_user='idx_user', colname_item='idx_item', colname_time='idx_time',
                 colname_outcome='outcome', colname_prediction='pred',
                 colname_outcome_T='outcome_T', colname_outcome_C='outcome_C', 
                 colname_treatment='treated', colname_propensity='propensity',
                 colname_effect='causal_effect', colname_estimate='causal_effect_estimate',
                 p_common=np.array([]), p_noncommon=np.array([])):

        self.colname_user = colname_user
        self.colname_item = colname_item
        self.colname_time = colname_time
        self.colname_outcome = colname_outcome
        self.colname_outcome_T = colname_outcome_T
        self.colname_outcome_C = colname_outcome_C
        self.colname_prediction = colname_prediction
        self.colname_treatment = colname_treatment
        self.colname_propensity = colname_propensity
        self.colname_effect = colname_effect
        self.colname_estimate = colname_estimate
        self.p_common = p_common
        self.p_noncommon = p_noncommon

    
    def observe_all(self, df_rank, df_po, users, num_rec):
        df_rank_ = df_rank.loc[np.isin(df_rank.loc[:, self.colname_user], users),:]
        df_rank_ = self.get_ranking(df_rank_, num_rec)
        df_rank_ = df_rank_.loc[:, [self.colname_user, self.colname_item]]
        df_rank_.loc[:, self.colname_treatment] = 1.0
        
        df_po_ = df_po.loc[np.isin(df_po.loc[:, self.colname_user], users), [self.colname_user, self.colname_item, self.colname_outcome_T, self.colname_outcome_C, self.colname_effect]]
        df_obs = pd.merge(df_po_, df_rank_, on=[self.colname_user, self.colname_item], how='left')
        
        df_obs = df_obs.fillna(0.0)
        
        df_obs.loc[:, self.colname_outcome] = df_obs.loc[:, self.colname_outcome_T] * df_obs.loc[:, self.colname_treatment] + \
                df_obs.loc[:, self.colname_outcome_C] * (1.0 - df_obs.loc[:, self.colname_treatment])
        obs = df_obs.groupby(self.colname_user).agg({self.colname_outcome: np.sum}).loc[:, self.colname_outcome].values
        effect = df_obs.loc[df_obs.loc[:, self.colname_treatment] == 1.0,:].groupby(self.colname_user).agg({self.colname_effect: np.mean}).loc[:, self.colname_effect].values
        
        return obs/num_rec, effect


    def observe_list_only(self, df_rank, df_po, users, num_rec):
        df_rank_ = df_rank.loc[np.isin(df_rank.loc[:, self.colname_user], users),:]
        df_rank_ = self.get_ranking(df_rank_, num_rec)
        df_rank_ = df_rank_.loc[:, [self.colname_user, self.colname_item]]
        df_rank_.loc[:, self.colname_treatment] = 1.0
        
        df_po_ = df_po.loc[np.isin(df_po.loc[:, self.colname_user], users), [self.colname_user, self.colname_item, self.colname_outcome_T, self.colname_outcome_C, self.colname_effect]]
        df_obs = pd.merge(df_po_, df_rank_, on=[self.colname_user, self.colname_item], how='left')
        
        df_obs = df_obs.fillna(0.0)
        
        df_obs.loc[:, self.colname_outcome] = df_obs.loc[:, self.colname_outcome_T] * df_obs.loc[:, self.colname_treatment] 
        obs = df_obs.groupby(self.colname_user).agg({self.colname_outcome: np.sum}).loc[:, self.colname_outcome].values
        effect = df_obs.loc[df_obs.loc[:, self.colname_treatment] == 1.0,:].groupby(self.colname_user).agg({self.colname_effect: np.mean}).loc[:, self.colname_effect].values
        
        return obs/num_rec, effect


    def observe_IL_simple(self, df_A, df_B, df_po, users, num_rec, num_add, verbose=False):
        # trimming to necessary rankings
        df_po_ = df_po.loc[np.isin(df_po.loc[:, self.colname_user], users), [self.colname_user, self.colname_item, self.colname_outcome_T, self.colname_outcome_C, self.colname_effect]]
        df_A_ = df_A.loc[np.isin(df_A.loc[:, self.colname_user], users),:]
        df_A_ = self.get_ranking(df_A_, num_rec+num_add)
        df_B_ = df_B.loc[np.isin(df_B.loc[:, self.colname_user], users),:]
        df_B_ = self.get_ranking(df_B_, num_rec+num_add)
        # construct comparison for each user
        list_users = np.array([])
        list_items = np.array([])
        list_bool_rec = np.array([])
        list_bool_A = np.array([])
        list_bool_B = np.array([])
        users_not_comparable = np.array([])

        for u in users:
            ranking_A = df_A_.loc[df_A_.loc[:, self.colname_user] == u, self.colname_item].values
            ranking_B = df_B_.loc[df_B_.loc[:, self.colname_user] == u, self.colname_item].values
            items_unique = np.unique(np.concatenate([ranking_A,ranking_B],0))

            items_recommended = np.random.choice(items_unique, num_rec)

            bool_A = np.isin(items_unique, ranking_A)
            bool_B = np.isin(items_unique, ranking_B)
            bool_rec = np.isin(items_unique, items_recommended)
            list_users = np.append(list_users, np.repeat(u, len(items_unique)))
            list_items = np.append(list_items, items_unique)
            list_bool_rec = np.append(list_bool_rec, bool_rec)
            list_bool_A = np.append(list_bool_A, bool_A)
            list_bool_B = np.append(list_bool_B, bool_B)
            
            if np.sum(bool_A & ~bool_rec) == 0 or np.sum(bool_B & ~bool_rec) == 0 or \
                np.sum(bool_A & bool_rec) == 0 or np.sum(bool_B & bool_rec) == 0:
                users_not_comparable = np.append(users_not_comparable, u)

        df_obs = pd.DataFrame(
            {self.colname_user: list_users,
             self.colname_item: list_items,
             'bool_rec': list_bool_rec,
             'bool_A': list_bool_A,
             'bool_B': list_bool_B})
        df_obs = pd.merge(df_obs, df_po_, on=[self.colname_user, self.colname_item], how='left')


        df_obsA = df_obs.loc[df_obs.loc[:,'bool_A']==1.0, :]
        df_obsB = df_obs.loc[df_obs.loc[:,'bool_B']==1.0, :]
        if len(users_not_comparable) > 0: 
            df_obsA = df_obsA.loc[~np.isin(df_obsA.loc[:,self.colname_user].values, users_not_comparable), :]
            df_obsB = df_obsB.loc[~np.isin(df_obsB.loc[:,self.colname_user].values, users_not_comparable), :]

        obsA = df_obsA.loc[df_obsA.loc[:,'bool_rec']==1.0, :].groupby(self.colname_user).agg({self.colname_outcome_T: np.mean}).loc[:,self.colname_outcome_T].values
        obsA = obsA - df_obsA.loc[df_obsA.loc[:,'bool_rec']==0.0, :].groupby(self.colname_user).agg({self.colname_outcome_C: np.mean}).loc[:,self.colname_outcome_C].values

        obsB = df_obsB.loc[df_obsB.loc[:,'bool_rec']==1.0, :].groupby(self.colname_user).agg({self.colname_outcome_T: np.mean}).loc[:,self.colname_outcome_T].values
        obsB = obsB - df_obsB.loc[df_obsB.loc[:,'bool_rec']==0.0, :].groupby(self.colname_user).agg({self.colname_outcome_C: np.mean}).loc[:,self.colname_outcome_C].values
    
        effect = df_obs.loc[df_obs.loc[:,'bool_rec']==1.0, :].groupby(self.colname_user).agg({self.colname_effect: np.mean}).loc[:, self.colname_effect].values
        return obsA,obsB,effect
            

    def calc_propensity(self, num_rec, num_add, num_repeat):

        p_common = np.zeros(num_rec+num_add)
        p_noncommon = np.zeros(num_rec+num_add)

        for nc in np.arange(num_rec + num_add):
            num_common = nc + 1
            if num_common <= 10:
                num_repeat_ = num_repeat * int(10/num_common)
            else:
                num_repeat_ = num_repeat
            
            ranking_A = np.arange(num_rec + num_add)
            ranking_B = np.arange(num_rec + num_add) + num_rec + num_add - num_common
            
            items_unique = np.unique(np.concatenate([ranking_A,ranking_B],0))
            items_common = ranking_A[np.isin(ranking_A, ranking_B)]

            cnt_common = len(items_common) * num_repeat_
            cnt_noncommon = (num_rec + num_add - num_common) * 2 * num_repeat_
            cnt_common_rec = 0

            for n in np.arange(num_repeat_):
                items_recommended = np.array([])
                next_choice_A = random.random() >= 0.5
                while len(items_recommended) < num_rec:
                    if next_choice_A:
                        items_recommended = np.append(items_recommended, np.random.choice(ranking_A[~np.isin(ranking_A, items_recommended)], 1))
                        next_choice_A = False
                    else:
                        items_recommended = np.append(items_recommended, np.random.choice(ranking_B[~np.isin(ranking_B, items_recommended)], 1))
                        next_choice_A = True
                
                cnt_common_rec += np.sum(np.isin(items_common, items_recommended))
            
            cnt_noncommon_rec = num_repeat_ * num_rec - cnt_common_rec
            p_common[nc] = cnt_common_rec/cnt_common
            if cnt_noncommon > 0:
                p_noncommon[nc] = cnt_noncommon_rec/cnt_noncommon
        
        return p_common, p_noncommon


    def observe_IL_balanced(self, df_A, df_B, df_po, users, num_rec, num_add, adjust="IPS", verbose=False):
        if len(self.p_common) != num_rec + num_add:
            self.p_common, self.p_noncommon = self.calc_propensity(num_rec, num_add, num_repeat=100)
            print("calculated propensity.")
            print(self.p_common)
            print(self.p_noncommon)
            print(datetime.now())

        # trimming to necessary rankings
        df_po_ = df_po.loc[np.isin(df_po.loc[:, self.colname_user], users), [self.colname_user, self.colname_item, self.colname_outcome_T, self.colname_outcome_C, self.colname_effect]]
        df_A_ = df_A.loc[np.isin(df_A.loc[:, self.colname_user], users),:]
        df_A_ = self.get_ranking(df_A_, num_rec+num_add)
        df_B_ = df_B.loc[np.isin(df_B.loc[:, self.colname_user], users),:]
        df_B_ = self.get_ranking(df_B_, num_rec+num_add)
        
        list_users = np.array([])
        list_items = np.array([])
        list_bool_rec = np.array([])
        list_bool_A = np.array([])
        list_bool_B = np.array([])
        list_propensity = np.array([])
        users_not_comparable = np.array([])
        list_num_samples = np.array([])
        list_num_common = np.array([])
        
        for u in users:
            ranking_A = df_A_.loc[df_A_.loc[:, self.colname_user] == u, self.colname_item].values
            ranking_B = df_B_.loc[df_B_.loc[:, self.colname_user] == u, self.colname_item].values
            items_unique = np.unique(np.concatenate([ranking_A,ranking_B],0))
            list_num_samples = np.append(list_num_samples, len(items_unique))
            items_recommended = np.array([])
            next_choice_A = random.random() >= 0.5

            while len(items_recommended) < num_rec:
                if next_choice_A:
                    items_recommended = np.append(items_recommended, np.random.choice(ranking_A[~np.isin(ranking_A, items_recommended)], 1))
                    next_choice_A = False
                else:
                    items_recommended = np.append(items_recommended, np.random.choice(ranking_B[~np.isin(ranking_B, items_recommended)], 1))
                    next_choice_A = True

            bool_A = np.isin(items_unique, ranking_A)
            bool_B = np.isin(items_unique, ranking_B)
            bool_rec = np.isin(items_unique, items_recommended)
            list_users = np.append(list_users, np.repeat(u, len(items_unique)))
            list_items = np.append(list_items, items_unique)
            list_bool_rec = np.append(list_bool_rec, bool_rec)
            list_bool_A = np.append(list_bool_A, bool_A)
            list_bool_B = np.append(list_bool_B, bool_B)

            bool_common = np.isin(ranking_A, ranking_B)
            num_common = np.sum(bool_common)
            list_num_common = np.append(list_num_common, num_common)
            if num_common > 0:
                items_common = ranking_A[bool_common]
                propensity = np.repeat(self.p_noncommon[num_common-1], len(items_unique))
                propensity[np.isin(items_unique, items_common)] = self.p_common[num_common-1]
            else:
                p = num_rec/len(items_unique)
                propensity = np.repeat(p, len(items_unique))

            
            list_propensity = np.append(list_propensity, propensity)
            if num_add == 0:
                if np.sum(bool_A & ~bool_rec) == 0 or np.sum(bool_B & ~bool_rec) == 0:
                    users_not_comparable = np.append(users_not_comparable, u)

        df_obs = pd.DataFrame(
            {self.colname_user: list_users,
             self.colname_item: list_items,
             'bool_rec': list_bool_rec,
             'bool_A': list_bool_A,
             'bool_B': list_bool_B,
             'propensity': list_propensity})
        if verbose:
            print("mean_num_common",np.mean(list_num_common))
        
        df_obs = pd.merge(df_obs, df_po_, on=[self.colname_user, self.colname_item], how='left')

        if adjust in ["IPS"]:
            df_obs.loc[:, 'IPS'] = 1.0/df_obs.loc[:, 'propensity']
            df_obs.loc[df_obs.loc[:,'bool_rec']==0.0, 'IPS'] = 1.0/(1.0-df_obs.loc[df_obs.loc[:,'bool_rec']==0.0, 'propensity'])
            # only outcome_T is used for Z=1 and only outcome_C is used for Z=0 in the following observation. Hence wrong values for the unused ones do not harm.
            df_obs.loc[:, self.colname_outcome_T] = df_obs.loc[:, self.colname_outcome_T] * df_obs.loc[:, 'IPS']
            df_obs.loc[:, self.colname_outcome_C] = df_obs.loc[:, self.colname_outcome_C] * df_obs.loc[:, 'IPS']
            

        df_obsA = df_obs.loc[df_obs.loc[:,'bool_A']==1.0, :]
        df_obsB = df_obs.loc[df_obs.loc[:,'bool_B']==1.0, :]
        if num_add == 0 and len(users_not_comparable) > 0: # remove users without control outcomes
            df_obsA = df_obsA.loc[~np.isin(df_obsA.loc[:,self.colname_user].values, users_not_comparable), :]
            df_obsB = df_obsB.loc[~np.isin(df_obsB.loc[:,self.colname_user].values, users_not_comparable), :]
            
        if adjust in ["IPS"]:
            temp_df = df_obsA.loc[df_obsA.loc[:,'bool_rec']==1.0, :].groupby(self.colname_user).agg({self.colname_outcome_T: np.sum})
            obsA = temp_df.loc[:, self.colname_outcome_T].values/num_rec
            temp_df = df_obsA.loc[df_obsA.loc[:,'bool_rec']==0.0, :].groupby(self.colname_user).agg({self.colname_outcome_C: np.sum})
            obsA = obsA - temp_df.loc[:, self.colname_outcome_C].values/num_rec

            temp_df = df_obsB.loc[df_obsB.loc[:,'bool_rec']==1.0, :].groupby(self.colname_user).agg({self.colname_outcome_T: np.sum})
            obsB = temp_df.loc[:, self.colname_outcome_T].values/num_rec
            temp_df = df_obsB.loc[df_obsB.loc[:,'bool_rec']==0.0, :].groupby(self.colname_user).agg({self.colname_outcome_C: np.sum})
            obsB = obsB - temp_df.loc[:, self.colname_outcome_C].values/num_rec
        else:
            obsA = df_obsA.loc[df_obsA.loc[:,'bool_rec']==1.0, :].groupby(self.colname_user).agg({self.colname_outcome_T: np.mean}).loc[:,self.colname_outcome_T].values
            obsA = obsA - df_obsA.loc[df_obsA.loc[:,'bool_rec']==0.0, :].groupby(self.colname_user).agg({self.colname_outcome_C: np.mean}).loc[:,self.colname_outcome_C].values
            obsB = df_obsB.loc[df_obsB.loc[:,'bool_rec']==1.0, :].groupby(self.colname_user).agg({self.colname_outcome_T: np.mean}).loc[:,self.colname_outcome_T].values
            obsB = obsB - df_obsB.loc[df_obsB.loc[:,'bool_rec']==0.0, :].groupby(self.colname_user).agg({self.colname_outcome_C: np.mean}).loc[:,self.colname_outcome_C].values

        effect = df_obs.loc[df_obs.loc[:,'bool_rec']==1.0, :].groupby(self.colname_user).agg({self.colname_effect: np.mean}).loc[:, self.colname_effect].values
        return obsA,obsB,effect


    def compare_AB(self, df_A, df_B, df_po, users, method_comparison, num_rec=10, num_add=0, verbose=False):
        usersA = np.random.choice(users, int(len(users)/2), replace=False)
        usersB = users[~np.isin(users, usersA)]
        
        if method_comparison == "AB_total":
            obsA, effectA = self.observe_all(df_A, df_po, usersA, num_rec)
            obsB, effectB = self.observe_all(df_B, df_po, usersB, num_rec)
            effect = np.concatenate([effectA,effectB],0)
        elif method_comparison == "AB_list":
            obsA, effectA = self.observe_list_only(df_A, df_po, usersA, num_rec)
            obsB, effectB = self.observe_list_only(df_B, df_po, usersB, num_rec)
            effect = np.concatenate([effectA,effectB],0)  
        elif method_comparison == "EPI_RCT":
            obsA, obsB, effect = self.observe_IL_simple(df_A, df_B, df_po, users, num_rec, num_add, verbose=verbose)    
        elif method_comparison[:3] == "CBI":
            wc = method_comparison.split("_")
            obsA, obsB, effect = self.observe_IL_balanced(df_A, df_B, df_po, users, num_rec, num_add, adjust=wc[1], verbose=verbose)    
        else:
            return np.nan, np.nan, np.nan
        
        diff_estimate = np.mean(obsA) - np.mean(obsB)
        mean_causal_effect = np.mean(effect)

        return diff_estimate, mean_causal_effect


    def repeat_comparison(self, df_A, df_B, df_po, method_comparison, num_repeat, num_users, num_rec=10, num_add=0):
        cand_users = np.unique(df_po.loc[:,'idx_user'])
        if num_users > len(cand_users):
            num_users = len(cand_users)

        method_comparison = method_comparison.replace("-","_")
        df_result = pd.DataFrame({'idx_sim': np.arange(num_repeat)})
        df_result.loc[:, 'diff_cnt'] = 0.0
        df_result.loc[:, 'diff_mean'] = 0.0
        df_result.loc[:, 'pval_ttest'] = 0.0
        df_result.loc[:, 'pval_wilcox'] = 0.0
        df_result.loc[:, 'mean_effect'] = 0.0
        for n in np.arange(num_repeat):
            users = np.random.choice(cand_users, num_users, replace=False)
            verbose = (n == 0)
            diff_estimate, mean_causal_effect = self.compare_AB(df_A, df_B, df_po, users, method_comparison, num_rec, num_add, verbose)
            df_result.loc[n, 'diff_estimate'] = diff_estimate
            df_result.loc[n, 'mean_causal_effect'] = mean_causal_effect

        print("failure ratio", np.mean(df_result.loc[:, 'diff_estimate'] < 0))
        print("mean of mean_causal_effect",np.mean(df_result.loc[:, 'mean_causal_effect']))

        return df_result

    def get_ranking(self, df, num_rec=10, resorting=True):
        if self.colname_prediction in df.columns and resorting:
            df = df.sort_values(by=[self.colname_user, self.colname_prediction], ascending=False)
        df_ranking = df.groupby(self.colname_user).head(num_rec)
        return df_ranking

    def get_sorted(self, df):
        df = df.sort_values(by=[self.colname_user, self.colname_prediction], ascending=False)
        return df
