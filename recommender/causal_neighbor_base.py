import numpy as np
from recommender import Recommender
from datetime import datetime


class CausalNeighborBase(Recommender):

    def __init__(self, num_users, num_items,
                 colname_user='idx_user', colname_item='idx_item',
                 colname_outcome='outcome', colname_prediction='pred',
                 measure_simil='cosine', way_simil='treatment',
                 way_neighbor='user', num_neighbor=100,
                 way_self='exclude',
                 weight_treated_outcome=0.5,
                 shrinkage_T=1.0, shrinkage_C=1.0,
                 scale_similarity=1.0, normalize_similarity=False):

        super().__init__(num_users=num_users, num_items=num_items,
                         colname_user=colname_user, colname_item=colname_item,
                         colname_outcome=colname_outcome, colname_prediction=colname_prediction)
        self.measure_simil = measure_simil
        self.way_simil = way_simil
        self.way_neighbor = way_neighbor
        self.num_neighbor = num_neighbor
        self.scale_similarity = scale_similarity
        self.normalize_similarity = normalize_similarity
        self.weight_treated_outcome = weight_treated_outcome
        self.shrinkage_T = shrinkage_T
        self.shrinkage_C = shrinkage_C
        self.way_self = way_self # exclude/include/only


    def simil(self, set1, set2, measure_simil):
        if measure_simil == "jaccard":
            return self.simil_jaccard(set1, set2)
        elif measure_simil == "cosine":
            return self.simil_cosine(set1, set2)

    def train(self, df, iter=1):
        df_posi = df.loc[df.loc[:, self.colname_outcome] > 0]
        print("len(df_posi): {}".format(len(df_posi)))

        dict_items2users = dict() # map an item to users who consumed the item
        for i in np.arange(self.num_items):
            dict_items2users[i] = np.unique(df_posi.loc[df_posi.loc[:, self.colname_item] == i, self.colname_user].values)
        self.dict_items2users = dict_items2users
        print("prepared dict_items2users")

        dict_users2items = dict()  # map an user to items which are consumed by the user
        for u in np.arange(self.num_users):
            dict_users2items[u] = np.unique(df_posi.loc[df_posi.loc[:, self.colname_user] == u, self.colname_item].values)
        self.dict_users2items = dict_users2items
        print("prepared dict_users2items")

        df_treated = df.loc[df.loc[:, self.colname_treatment] > 0]  # calc similarity by treatment assignment
        print("len(df_treated): {}".format(len(df_treated)))

        dict_items2users_treated = dict() # map an item to users who get treatment of the item
        for i in np.arange(self.num_items):
            dict_items2users_treated[i] = np.unique(df_treated.loc[df_treated.loc[:, self.colname_item] == i, self.colname_user].values)
        self.dict_items2users_treated = dict_items2users_treated
        print("prepared dict_items2users_treated")

        dict_users2items_treated = dict()  # map an user to items which are treated to the user
        for u in np.arange(self.num_users):
            dict_users2items_treated[u] = np.unique(df_treated.loc[df_treated.loc[:, self.colname_user] == u, self.colname_item].values)
        self.dict_users2items_treated = dict_users2items_treated
        print("prepared dict_users2items_treated")

        if self.way_simil == 'treatment':
            if self.way_neighbor == 'user':
                dict_simil_users = {}
                sum_simil = np.zeros(self.num_users)
                for u1 in np.arange(self.num_users):
                    if u1 % round(self.num_users/10) == 0:
                        print("progress of similarity computation: {:.1f} %".format(100 * u1/self.num_users))

                    items_u1 = self.dict_users2items_treated[u1]
                    dict_neighbor = {}
                    if len(items_u1) > 0:
                        cand_u2 = np.unique(df_treated.loc[np.isin(df_treated.loc[:, self.colname_item], items_u1), self.colname_user].values)
                        for u2 in cand_u2:
                            if u2 != u1:
                                items_u2 = self.dict_users2items_treated[u2]
                                dict_neighbor[u2] = self.simil(items_u1, items_u2, self.measure_simil)

                        if len(dict_neighbor) > self.num_neighbor:
                            dict_neighbor = self.trim_neighbor(dict_neighbor, self.num_neighbor)
                        if self.scale_similarity != 1.0:
                            dict_neighbor = self.rescale_neighbor(dict_neighbor, self.scale_similarity)
                        if self.normalize_similarity:
                            dict_neighbor = self.normalize_neighbor(dict_neighbor)
                        dict_simil_users[u1] = dict_neighbor
                        sum_simil[u1] = np.sum(np.array(list(dict_neighbor.values())))
                    else:
                        dict_simil_users[u1] = dict_neighbor
                self.dict_simil_users = dict_simil_users
                self.sum_simil = sum_simil

            elif self.way_neighbor == 'item':
                dict_simil_items = {}
                sum_simil = np.zeros(self.num_items)
                for i1 in np.arange(self.num_items):
                    if i1 % round(self.num_items/10) == 0:
                        print("progress of similarity computation: {:.1f} %".format(100 * i1 / self.num_items))

                    users_i1 = self.dict_items2users_treated[i1]
                    dict_neighbor = {}
                    if len(users_i1) > 0:
                        cand_i2 = np.unique(
                            df_treated.loc[np.isin(df_treated.loc[:, self.colname_user], users_i1), self.colname_item].values)
                        for i2 in cand_i2:
                            if i2 != i1:
                                users_i2 = self.dict_items2users_treated[i2]
                                dict_neighbor[i2] = self.simil(users_i1, users_i2, self.measure_simil)

                        if len(dict_neighbor) > self.num_neighbor:
                            dict_neighbor = self.trim_neighbor(dict_neighbor, self.num_neighbor)
                        if self.scale_similarity != 1.0:
                            dict_neighbor = self.rescale_neighbor(dict_neighbor, self.scale_similarity)
                        if self.normalize_similarity:
                            dict_neighbor = self.normalize_neighbor(dict_neighbor)
                        dict_simil_items[i1] = dict_neighbor
                        sum_simil[i1] = np.sum(np.array(list(dict_neighbor.values())))
                    else:
                        dict_simil_items[i1] = dict_neighbor
                self.dict_simil_items = dict_simil_items
                self.sum_simil = sum_simil
        else:
            if self.way_neighbor == 'user':
                dict_simil_users = {}
                sum_simil = np.zeros(self.num_users)
                for u1 in np.arange(self.num_users):
                    if u1 % round(self.num_users/10) == 0:
                        print("progress of similarity computation: {:.1f} %".format(100 * u1 / self.num_users))

                    items_u1 = self.dict_users2items[u1]
                    dict_neighbor = {}
                    if len(items_u1) > 0:
                        cand_u2 = np.unique(
                            df_posi.loc[np.isin(df_posi.loc[:, self.colname_item], items_u1), self.colname_user].values)
                        for u2 in cand_u2:
                            if u2 != u1:
                                items_u2 = self.dict_users2items[u2]
                                dict_neighbor[u2] = self.simil(items_u1, items_u2, self.measure_simil)

                        if len(dict_neighbor) > self.num_neighbor:
                            dict_neighbor = self.trim_neighbor(dict_neighbor, self.num_neighbor)
                        if self.scale_similarity != 1.0:
                            dict_neighbor = self.rescale_neighbor(dict_neighbor, self.scale_similarity)
                        if self.normalize_similarity:
                            dict_neighbor = self.normalize_neighbor(dict_neighbor)
                        dict_simil_users[u1] = dict_neighbor
                        sum_simil[u1] = np.sum(np.array(list(dict_neighbor.values())))
                    else:
                        dict_simil_users[u1] = dict_neighbor
                self.dict_simil_users = dict_simil_users
                self.sum_simil = sum_simil

            elif self.way_neighbor == 'item':
                dict_simil_items = {}
                sum_simil = np.zeros(self.num_items)
                for i1 in np.arange(self.num_items):
                    if i1 % round(self.num_items/10) == 0:
                        print("progress of similarity computation: {:.1f} %".format(100 * i1 / self.num_items))

                    users_i1 = self.dict_items2users[i1]
                    dict_neighbor = {}
                    if len(users_i1) > 0:
                        cand_i2 = np.unique(
                            df_posi.loc[np.isin(df_posi.loc[:, self.colname_user], users_i1), self.colname_item].values)
                        for i2 in cand_i2:
                            if i2 != i1:
                                users_i2 = self.dict_items2users[i2]
                                dict_neighbor[i2] = self.simil(users_i1, users_i2, self.measure_simil)

                        if len(dict_neighbor) > self.num_neighbor:
                            dict_neighbor = self.trim_neighbor(dict_neighbor, self.num_neighbor)
                        if self.scale_similarity != 1.0:
                            dict_neighbor = self.rescale_neighbor(dict_neighbor, self.scale_similarity)
                        if self.normalize_similarity:
                            dict_neighbor = self.normalize_neighbor(dict_neighbor)
                        dict_simil_items[i1] = dict_neighbor
                        sum_simil[i1] = np.sum(np.array(list(dict_neighbor.values())))
                    else:
                        dict_simil_items[i1] = dict_neighbor
                self.dict_simil_items = dict_simil_items
                self.sum_simil = sum_simil


    def trim_neighbor(self, dict_neighbor, num_neighbor):
        return dict(sorted(dict_neighbor.items(), key=lambda x:x[1], reverse = True)[:num_neighbor])

    def normalize_neighbor(self, dict_neighbor):
        sum_simil = 0.0
        for v in dict_neighbor.values():
            sum_simil += v
        for k, v in dict_neighbor.items():
            dict_neighbor[k] = v/sum_simil
        return dict_neighbor

    def rescale_neighbor(self, dict_neighbor, scaling_similarity=1.0):
        for k, v in dict_neighbor.items():
            dict_neighbor[k] = np.power(v, scaling_similarity)
        return dict_neighbor


    def predict(self, df):
        users = df[self.colname_user].values
        items = df[self.colname_item].values
        pred = np.zeros(len(df))
        if self.way_neighbor == 'user':
            for n in np.arange(len(df)):
                u1 = users[n]
                simil_users = np.fromiter(self.dict_simil_users[u1].keys(), dtype=int)
                i_users_posi = self.dict_items2users[items[n]]  # users who consumed i=items[n]
                i_users_treated = self.dict_items2users_treated[items[n]]  # users who are treated i=items[n]
                if n % round(len(df)/10) == 0:
                    print(datetime.now())
                    print("progress of prediction computation: {:.1f} %".format(100 * n / len(df)))
             
                # initialize 
                value_T = 0.0
                denom_T = 0.0
                value_C = 0.0
                denom_C = 0.0

                if np.any(np.isin(simil_users, i_users_posi)):
                    simil_users = simil_users[np.isin(simil_users, np.unique(np.append(i_users_treated,i_users_posi)))]
                    for u2 in simil_users:
                        if u2 in i_users_treated:
                            denom_T += self.dict_simil_users[u1][u2]
                            if u2 in i_users_posi:
                                value_T += self.dict_simil_users[u1][u2]
                        else:
                            value_C += self.dict_simil_users[u1][u2]
                            
                    denom_C = self.sum_simil[u1] - denom_T # denom_T + denom_C = sum_simil

                if self.way_self == 'include': # add data of self u-i
                    if u1 in i_users_treated:
                        denom_T += 1.0
                        if u1 in i_users_posi:
                            value_T += 1.0
                    else:
                        denom_C += 1.0
                        if u1 in i_users_posi:
                            value_C += 1.0

                if self.way_self == 'only': # force data to self u-i
                    if u1 in i_users_treated:
                        denom_T = 1.0
                        if u1 in i_users_posi:
                            value_T = 1.0
                        else:
                            value_T = 0.0
                    else:
                        denom_C = 1.0
                        if u1 in i_users_posi:
                            value_C = 1.0
                        else:
                            value_C = 0.0

                if value_T > 0:
                    pred[n] += 2 * self.weight_treated_outcome * value_T / (self.shrinkage_T + denom_T)
                if value_C > 0:
                    pred[n] -= 2 * (1 - self.weight_treated_outcome) * value_C / (self.shrinkage_C + denom_C)
            

        elif self.way_neighbor == 'item':
            for n in np.arange(len(df)):
                i1 = items[n]
                simil_items = np.fromiter(self.dict_simil_items[i1].keys(), dtype=int)
                u_items_posi = self.dict_users2items[users[n]]  # items that is consumed by u=users[n]
                u_items_treated = self.dict_users2items_treated[users[n]] # items that is treated for u=users[n]
                if n % round(len(df)/10) == 0:
                    print(datetime.now())
                    print("progress of prediction computation: {:.1f} %".format(100 * n / len(df)))

                # initialize 
                value_T = 0.0
                denom_T = 0.0
                value_C = 0.0
                denom_C = 0.0

                if np.any(np.isin(simil_items, u_items_posi)):
                    simil_items = simil_items[np.isin(simil_items, np.unique(np.append(u_items_posi, u_items_treated)))]
                    for i2 in simil_items:
                        if i2 in u_items_treated: # we assume that treated items are less than untreated items
                            denom_T += self.dict_simil_items[i1][i2]
                            if i2 in u_items_posi:
                                value_T += self.dict_simil_items[i1][i2]
                        else:
                            value_C += self.dict_simil_items[i1][i2]
                    denom_C = self.sum_simil[i1] - denom_T  # denom_T + denom_C = sum_simil

                if self.way_self == 'include': # add data of self u-i
                    if i1 in u_items_treated:
                        denom_T += 1.0
                        if i1 in u_items_posi:
                            value_T += 1.0
                    else:
                        denom_C += 1.0
                        if i1 in u_items_posi:
                            value_C += 1.0

                if self.way_self == 'only': # force data to self u-i
                    if i1 in u_items_treated:
                        denom_T = 1.0
                        if i1 in u_items_posi:
                            value_T = 1.0
                        else:
                            value_T = 0.0
                    else:
                        denom_C = 1.0
                        if i1 in u_items_posi:
                            value_C = 1.0
                        else:
                            value_C = 0.0

                if value_T > 0:
                    pred[n] += 2 * self.weight_treated_outcome * value_T / (self.shrinkage_T + denom_T)
                if value_C > 0:
                    pred[n] -= 2 * (1 - self.weight_treated_outcome) * value_C / (self.shrinkage_C + denom_C)

        return pred


    def simil_jaccard(self, x, y):
        return len(np.intersect1d(x, y))/len(np.union1d(x, y))

    def simil_cosine(self, x, y):
        return len(np.intersect1d(x, y))/np.sqrt(len(x)*len(y))

