
from datetime import datetime
import numpy as np
import pandas as pd

import random
import os
random.seed(10)
np.random.seed(10)

from evaluator import Comparator
import warnings
import argparse

def setup_arg_parser():
    parser = argparse.ArgumentParser(
        prog='eval_sim_online.py',
        usage='evaluation by simulated online experiments',
        description='',
        add_help=True)

    parser.add_argument('-d', '--dir_data', type=str, default='data_dho/',
                        help='path of data directory', required=False)
    # conditions for the experiment
    parser.add_argument('-a', '--rankingA', type=str,
                        help='compared ranking A',
                        required=True)
    parser.add_argument('-b', '--rankingB', type=str,
                        help='compared ranking B',
                        required=True)
    parser.add_argument('-mc', '--methods_comparison', type=str,
                        help='methods of ranking comparison',
                        required=True)
    parser.add_argument('-cnu', '--cond_num_users', type=str, default='50:100:200',
                        help='conditions about the numbers of experimented users',
                        required=False)
    parser.add_argument('-cnr', '--cond_num_rec', type=str, default='10:100',
                        help='conditions about the numbers of recommendations (= the sizes of recommendation lists)',
                        required=False)
    parser.add_argument('-cna', '--cond_num_add', type=str, default='0',
                        help='conditions about the size addition to the original lists (not used in the paper)',
                        required=False)
    parser.add_argument('-nr', '--num_repeat', type=int, default=100,
                        help='number of repeated simulations',
                        required=False)
    parser.add_argument('-ne', '--name_experiment', type=str, default='exp',
                        help='abbreviated name to denote the experiment',
                        required=False)
    parser.add_argument('-ssr', '--set_seed_random', type=int, default=1,
                        help='set seed for randomness',
                        required=False)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = setup_arg_parser()
    t_init = datetime.now()

    random.seed(args.set_seed_random)
    np.random.seed(args.set_seed_random)

    if args.dir_data[-1] != '/':
        args.dir_data = args.dir_data + '/'
    dir_data = args.dir_data

    # load potential outcomes
    df_po = pd.read_csv(dir_data +  'data_test.csv')
    
    # load two rankings
    df_rankA = pd.read_csv(dir_data + args.rankingA + '.csv')
    df_rankB = pd.read_csv(dir_data + args.rankingB + '.csv')

    warnings.filterwarnings("ignore")

    if not os.path.exists(dir_data + 'comparisons/'):
        os.mkdir(dir_data + 'comparisons/')
    dir_data_comparisons = dir_data + 'comparisons/' + args.rankingA + '_' + args.rankingB + '_' + args.name_experiment + '/'
    if not os.path.exists(dir_data_comparisons):
        os.mkdir(dir_data_comparisons)

    comp = Comparator()
        
    adjust = ''
    for num_rec in args.cond_num_rec.split(":"):
        print("num_rec: {}".format(num_rec))
        num_rec = int(num_rec)
        for num_add in args.cond_num_add.split(":"):
            print("num_add: {}".format(num_add))
            num_add = int(num_add)
            for num_users in args.cond_num_users.split(":"):
                print("num_users: {}".format(num_users))
                num_users = int(num_users)
                for method_comparison in args.methods_comparison.split(":"):
                    name_condition = "{}_nrep{}_nuser{}_nrec{}_nadd{}".format(method_comparison, args.num_repeat, num_users, num_rec, num_add)
                    print(name_condition)
                    save_result_file = dir_data_comparisons + name_condition + ".csv"
                    print(datetime.now())
                    df_result = comp.repeat_comparison(df_rankA, df_rankB, df_po, method_comparison, args.num_repeat, num_users, num_rec, num_add)
                    df_result.to_csv(save_result_file)

    t_end = datetime.now()
    t_diff = t_end - t_init

    hours = t_diff.days * 24 + t_diff.seconds/(60 * 60)
    print('Completed in {:.2f} hours.'.format(hours))
