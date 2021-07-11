
from datetime import datetime
import numpy as np
import pandas as pd

import random
import os
random.seed(10)

from utils import DataGeneratorDH

import argparse

def setup_arg_parser():
    parser = argparse.ArgumentParser(
        prog='generate_dataset_dh.py',
        usage='prepare semi-synthetic data from Dunnhumby',
        description='',
        add_help=True)
    parser.add_argument('-d', '--dir_data_output', type=str, default='data_dho/',
                        help='path of output data directory', required=False)
    parser.add_argument('-rp', '--rate_prior', type=float,
                        help='rate of prior obtained from mean values', default=0.4,
                        required=False)
    parser.add_argument('-nr', '--num_rec', type=int,
                        help='expected number of recommendation', default=210,
                        required=False)
    parser.add_argument('-cap', '--capping', type=float,
                        help='capping', default=0.000001,
                        required=False)
    parser.add_argument('-ssr', '--set_seed_random', type=int, default=1,
                        help='set seed for randomness',
                        required=False)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = setup_arg_parser()
    dir_data_output = args.dir_data_output
    if dir_data_output[-1] != '/':
        dir_data_output = dir_data_output + '/'
    if not os.path.exists(dir_data_output):
        os.mkdir(dir_data_output)

    t_init = datetime.now()
    df_cnt = pd.read_csv('data_raw/cnt_logs.csv')
    data_generator = DataGeneratorDH(df_cnt=df_cnt, rate_prior=args.rate_prior)
    np.random.seed(seed=args.set_seed_random)
    capping = args.capping

    data_generator.prepare_prob(capping=capping)


    data_generator.assign_treatment()
    data_generator.assign_outcome()
    df_train = data_generator.get_observation()
    df_train.to_csv(dir_data_output + 'data_train.csv', index=False)

    data_generator.assign_treatment()
    data_generator.assign_outcome()
    df_vali = data_generator.get_ground_truth()
    df_vali.to_csv(dir_data_output + 'data_vali.csv', index=False)

    data_generator.assign_treatment()
    data_generator.assign_outcome()
    df_test = data_generator.get_ground_truth()
    df_test.to_csv(dir_data_output + 'data_test.csv', index=False)

    print('Data prepared.')
    print('Max propensity: {}'.format(np.max(df_test.loc[:, 'propensity'])))
    print('Min propensity: {}'.format(np.min(df_test.loc[:, 'propensity'])))
    print('Average propensity: {}'.format(np.mean(df_test.loc[:, 'propensity'])))
    print('Average number of recommendations: {}'.format(np.mean(df_test.loc[:, 'treated'])*data_generator.num_items))
    print('Ratio of positive outcomes: {}'.format(np.mean(df_test.loc[:, 'outcome'])))
    print('Ratio of positive treatment effect: {}'.format(np.mean(df_test.loc[:, 'causal_effect'] > 0)))
    print('Ratio of negative treatment effect: {}'.format(np.mean(df_test.loc[:, 'causal_effect'] < 0)))
    print('Average treatment effect: {}'.format(np.mean(df_test.loc[:, 'causal_effect'])))

    t_end = datetime.now()
    t_diff = t_end - t_init

    hours = (t_diff.days * 24) + (t_diff.seconds / 3600)

    print('Completed in {:.2f} hours.'.format(hours))


