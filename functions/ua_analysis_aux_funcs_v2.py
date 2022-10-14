import sys
sys.path.append("/data/Fmubang/CP4-ORGANIZED-V3-FIX-DROPNA/functions")

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

import pandas as pd
import os,sys
from scipy import stats
import numpy as np
from pnnl_metric_funcs_v2 import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import basic_utils as bu
# import train_data_funcs as tdf
# import fixed_volume_func as fvf
# from infoIDs18 import get_cp4_challenge_18_infoIDs
# import model_param_utils as mp
# import baseline_utils as base_u
# from SampleGenerator import SampleGenerator
# from ModelHandler import ModelHandler,model_type_to_param_name_order_dict,xgboost_param_name_order,model_type_to_model_func_dict
# from ft_importance_funcs import save_ft_info_v2_simpler
# import model_eval_funcs as mef
import joblib
from basic_utils import create_output_dir

def get_seed_to_model_to_dir_dict(SEEDS, platform, main_input_dir, vol_tag):

    model_tags = set()

    seed_to_model_to_dir_dict = {}
    for SEED in SEEDS:
        seed_to_model_to_dir_dict[SEED] = {}
        #make seed dir
        cur_seed_input_dir = main_input_dir + "SEED-" + str(SEED) + "/"
        cur_seed_input_dir = cur_seed_input_dir + platform + "/"

        #get subdirs
        all_subdirs = os.listdir(cur_seed_input_dir)
        full_dirs = []
        for s in all_subdirs:
            s = cur_seed_input_dir + s + "/"
            full_dirs.append(s)
            print(s)
        all_subdirs = full_dirs

        #extract model tags
        print("\nModel tags:")
        for s in all_subdirs:
            LB = s.split("-")[-2]
            LB = int(LB)
            model_tag = vol_tag + "V-" + str(LB) + "U"

            if "VAM-" not in model_tag:
                model_tag = "VAM-" + model_tag

            model_tags.add(model_tag)
            print(model_tag)
            seed_to_model_to_dir_dict[SEED][model_tag] = s

    return seed_to_model_to_dir_dict, list(model_tags)


def modify_ua_metric_df(df, metric, vol_tag):

    baseline_col = "Persistence Baseline Avg. %s"%metric
    model_col =  "%s Avg. %s"%(vol_tag, metric)
    pim_col = "VAM Percent Improvement From Baseline (%)"

    df = df.rename(columns={
        "Persistence_Baseline_avg_%s"%(metric.lower()): baseline_col,
     "infoID":"Topic",
        "best_non_pb_model_avg_%s"%(metric.lower()): model_col,
        })

    print(df)

    df = df[["Topic", model_col, baseline_col]]
    df[pim_col] = 100.0 * (df[baseline_col] - df[model_col])/(df[baseline_col])
    df[pim_col] = np.round(df[pim_col], 2)

    round_cols = [baseline_col, model_col]
    for r in round_cols:
        df[r] = np.round(df[r], 6)

    print("\nfinal df")
    print(df)


    return df, model_col, baseline_col, pim_col

def plot_ua_comp_results(df, metric, model_output_dir,model_col, baseline_col, topic_col = "Topic"):

    #plot it
    plot_cols = [topic_col, model_col, baseline_col]
    plot_df = df[plot_cols]
    error_sum_series = plot_df[model_col] + plot_df[baseline_col]
    plot_df[model_col] = plot_df[model_col]/error_sum_series
    plot_df[baseline_col] = plot_df[baseline_col]/error_sum_series
    norm_fp = model_output_dir +"/Normalized-Final-Results.csv"
    plot_df.to_csv(norm_fp, index=False)

    plot_df = plot_df.rename(columns={model_col: "VAM"})
    plot_df = plot_df.rename(columns={baseline_col: "Persistence Baseline"})
    # plot_df = plot_df.rename(columns={"infoID":"Topic"})
    plot_df = plot_df.set_index("Topic")

    sns.set()
    plot_df.plot(kind="bar")
    fig = plt.gcf()
    fig.set_size_inches(5,5)
    ax = plt.gca()

    ax.set_title("%s Results"%metric)
    ax.tick_params(labelsize=7)
    plt.tight_layout()
    fig.savefig(model_output_dir + "%s-Bar-plots.png"%metric)
    fig.savefig(model_output_dir + "%s-Bar-plots.svg"%metric)
    plt.close()

def plot_ua_comp_results_v2_better(df, metric, model_output_dir,model_col, baseline_col, topic_col = "Topic"):

    #plot it
    plot_cols = [topic_col, model_col, baseline_col]
    plot_df = df[plot_cols]
    error_sum_series = plot_df[model_col] + plot_df[baseline_col]
    plot_df[model_col] = plot_df[model_col]/error_sum_series
    plot_df[baseline_col] = plot_df[baseline_col]/error_sum_series
    norm_fp = model_output_dir +"/Normalized-Final-Results.csv"
    plot_df.to_csv(norm_fp, index=False)

    plot_df = plot_df.rename(columns={model_col: "VAM"})
    plot_df = plot_df.rename(columns={baseline_col: "Persistence Baseline"})
    # plot_df = plot_df.rename(columns={"infoID":"Topic"})
    plot_df = plot_df.set_index("Topic")

    LSIZE = 9

    sns.set()
    plot_df.plot(kind="bar")
    fig = plt.gcf()
    #fig.set_size_inches(5,5)

    fig.set_size_inches(12, 8)
    ax = plt.gca()

    if metric == "EMD":
        metric_name = "Earth Mover's Distance"
    else:
        metric_name = "Relative Hausdorff Distance"

    ax.set_title("%s Results\n%s vs. Persistence Baseline"%(metric_name, model_col),fontdict={'fontsize':LSIZE+15})

    ax.tick_params(labelsize=LSIZE+2)

    plt.xticks(rotation=70)
    plt.tight_layout()
    fig.savefig(model_output_dir + "%s-Bar-plots.png"%metric)
    fig.savefig(model_output_dir + "%s-Bar-plots.svg"%metric)
    plt.close()