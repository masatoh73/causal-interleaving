# Online Evaluation Methods for the Causal Effect of Recommendation

This is our implementation for the paper ["Online Evaluation Methods for the Causal Effect of Recommendation"](https://doi.org/10.1145/3460231.3474235) presented in RecSys '21.

It includes online evaluation methods experimetend in the paper: AB-total, AB-list, EPI-RCT, CBI-RCT, CBI-IPS.

It also covers the codes for generating semi-synthetic datasets and preparing rankings for the experiments.

## Requirement
The source codes are mostly written in Python.
Getting the statistics of Dunnhumby data is written in R.
The libraries requred are listed below.
The versions in parenthesis are our environment for the experiment.

* Python (3.8.5)
* numpy (1.19.2)
* pandas (1.1.3)
* R (3.4.4)
* data.table (1.14.0)

## Usage
Simulated online experiments are conducted as follows. 

```bash:sample
cd ./CausalInterleaving
python eval_sim_online.py -d data_dho -a CUBNT -b ULRMF -cnu 100 -cnr 10 -nr 100 -mc AB_total:AB_list:EPI_RCT:CBI_RCT:CBI_IPS -ne method_comparison
```
- **-d (dir_data)** specifies the path of data directory. Put the rankings and ground truth data here.
- **-a (rankingA)** specifies the compared ranking A. The file name should be *rankingA*.csv.
- **-b (rankingB)** specifies the compared ranking B. The file name should be *rankingB*.csv.
- **-cnu (cond_num_users)** specifies the conditions of the number of users. You can input multiple conditions, 100:1000:10000, to conduct multiple experiments in different conditions.
- **-cnr (cond_num_rec)** specifies the conditions of the number of recommendation ( = the size of interleaved list = the sizes of original lists). 
- **-nr (num_repeat)** specifies the number of repeats for the experiment.
- **-mc (methods_comparison)** specifies the methods for comparison. You can input multiple methods, EPI_RCT:CBI_RCT:CBI_IPS, to conduct multiple experiments in different conditions. Note that the method name can use both hyphen (e.g., CBI-IPS) and underscore (e.g., CBI_IPS).
- **-ne (name_experiment)** specifies the name of experiments used for the result output folder.

It outputs experiment results on the folder *{dir_data}*/comparisons/*{rankingA}*\_*{rankingB}*\_*{name_experiment}* 


## Preparation
Before the experiment, we need to prepare semi-synthetic datasets and rankings to be compared. 

### For Dunnhumby dataset
1. Download ***The Complete Journey*** dataset provided by [Dunnhumby](https://www.dunnhumby.com/careers/engineering/sourcefiles). Put the data on the folder **CausalInterleaving/data_raw**.
1. Get the statistics of Dunnhumby data with **get_statistics_dunnhumby.R**. It outputs **cnt_logs.csv** that includes numbers of recommendations, numbers of recommended purchases, etc. for each user-item pair.
    ```bash:sample
    Rscript get_statistics_dunnhumby.R
    ```
1. Generate a semi-synthetic dataset with **generate_dataset_dh.py**.
    ```bash:sample
    python generate_dataset_dh.py -d data_dho
    ```
1. Prepare rankings to be prepared.
    ```bash:sample
    python prepare_ranking.py -d data_dho -tm LMF -cs iter:5000000+train_metric:AUC+dim_factor:200+learn_rate:0.003+reg_factor:0.1 -ne BPR

    python prepare_ranking.py -d data_dho -tm ULMF -cs iter:95000000+train_metric:logloss+dim_factor:200+learn_rate:0.01+alpha:0.0+reg_factor:0.01 -ne ULRMF

    python prepare_ranking.py -d data_dho -tm CausalNeighborBase -cs iter:1+way_simil:treatment+measure_simil:cosine+way_neighbor:user+scale_similarity:3.0+shrinkage:1.0+num_neighbor:3000 -ne CUBNT
    ```

### For MovieLens dataset
1. Download the MovieLens-1M dataset provided by [GroupLens](https://grouplens.org/datasets/movielens). Put the data on the folder **CausalInterleaving/data_raw**.
1. Generate a semi-synthetic dataset with *generate_dataset_ml.py*.
    ```bash:sample
    python generate_dataset_ml.py -d data_ml1m
    ```
1. Prepare rankings to be prepared.
    ```bash:sample
    python prepare_ranking.py -d data_ml1m -tm NeighborBase -cs iter:1+way_simil:outcome+measure_simil:cosine+way_neighbor:user+scale_similarity:0.33+shrinkage:0.3+num_neighbor:10000 -ne UBN

    python prepare_ranking.py -d data_ml1m -tm ULMF -cs iter:75000000+train_metric:AUC+dim_factor:100+learn_rate:0.003+alpha:0.8+reg_factor:0.1 -ne ULBPR

    python prepare_ranking.py -d data_ml1m -tm CausalNeighborBase -cs iter:1+way_simil:outcome+measure_simil:cosine+way_neighbor:user+scale_similarity:0.5+shrinkage:30.0+num_neighbor:10000 -ne CUBNO
    ```

