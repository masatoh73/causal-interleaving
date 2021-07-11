
from datetime import datetime
import numpy as np
import pandas as pd

import random
import os
random.seed(10)
np.random.seed(10)

from utils import Ranker

import argparse

def setup_arg_parser():
    parser = argparse.ArgumentParser(
        prog='param_search.py',
        usage='generate rankings',
        description='',
        add_help=True)

    parser.add_argument('-d', '--dir_data', type=str, default='data_dho',
                        help='path of data directory', required=False)

    parser.add_argument('-tm', '--type_model', type=str,
                        help='type of model',
                        required=True)
    parser.add_argument('-cs', '--conds', type=str, default='dim_factor:100+reg_common:0.0001+learn_rate:0.001',
                        help='condition',
                        required=True)

    parser.add_argument('-nu', '--num_users', type=int, default=-1,
                        help='number of users',
                        required=False)
    parser.add_argument('-ni', '--num_items', type=int, default=-1,
                        help='number of items',
                        required=False)
    parser.add_argument('-ssr', '--set_seed_random', type=int, default=1,
                        help='set seed for randomness',
                        required=False)
    parser.add_argument('-ne', '--name_ranking', type=str, default='exp',
                        help='abbreviated name to express the ranking',
                        required=False)
    parser.add_argument('-ls', '--list_size', type=int, default=10,
                        help='list size of recommendation',
                        required=False)
 
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = setup_arg_parser()

    random.seed(args.set_seed_random)
    np.random.seed(args.set_seed_random)

    dir_data = args.dir_data
    if dir_data[-1] != '/':
        dir_data = dir_data + '/'

    # load data
    df_train = pd.read_csv(dir_data + 'data_train.csv')
    if 'idx_time' not in df_train.columns:
        df_train.loc[:,'idx_time'] = 0

    df_train = df_train.loc[:, ['idx_time', 'idx_user', 'idx_item', 'propensity', 'treated', 'outcome']]
    # to reduce memory usage
    df_train = df_train.astype({'idx_time': 'int8', 'idx_user': 'uint32', 'idx_item': 'uint32', 'outcome': 'int8', 'treated': 'int8'})
    df_test = pd.read_csv(dir_data + 'data_test.csv')
    df_test = df_test.loc[:, ['idx_user', 'idx_item', 'outcome_T', 'outcome_C', 'causal_effect']]
    df_test = df_test.astype({'idx_user': 'uint32', 'idx_item': 'uint32', 'outcome_T': 'int8', 'outcome_C': 'int8', 'causal_effect': 'int8'})
    print('Data loaded.')

    ranker = Ranker()
    params = ranker.set_params(args.conds)
    params['recommender'] = args.type_model
    print("args.type_model: {}".format(args.type_model))

    print('Start training.')
    t_init = datetime.now()
    df_rank = ranker.rank(params, df_train, df_test, num_users=-1, num_items=-1, num_rec=args.list_size)
    print('Average causal effect of recommendations: {}'.format(np.mean(df_rank.loc[:, 'causal_effect'])))
    print('Average outcomes with recommendation: {}'.format(np.mean(df_rank.loc[:, 'outcome_T'])))

    df_rank = df_rank.loc[:, ['idx_user', 'idx_item', 'pred']]
    df_rank.to_csv(dir_data + args.name_ranking + '.csv')

    t_end = datetime.now()
    t_diff = t_end - t_init
    hours = t_diff.days * 24 + t_diff.seconds/(60 * 60)
    print('Completed in {:.2f} hours.'.format(hours))
