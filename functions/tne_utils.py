import sys
# sys.path.append("/data/Fmubang/cp4-code-clean/functions")
# sys.path.append("/data/Fmubang/cp4-VAM-write-up-exps/functions")
sys.path.append("/storage2-mnt/data/fmubang/CP5-VAM-Paper-Stuff-3-3/functions")
import pandas as pd
import os,sys
from scipy import stats
import numpy as np
# from cascade_ft_funcs import *
import pickle
from random import seed
from random import random
from random import randrange

import joblib
# from ft_categories import *
import multiprocessing as mp

# from sample_gen_aux_funcs import *
from basic_utils import create_output_dir,gzip_save,gzip_load
from multiprocessing import Pool
import basic_utils as bu
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

#might have to come back to this here: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html
# def split_train_into_train_and_val(x_train, y_val):
#     y_val_cat = np.sum(y_val, axis=1).astype("int32")
#     print()
#     print(y_val_cat.shape)

#     # x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,stratify=Y,test_size=0.2) # before model building

#     return x_train, x_val, y_train, y_val

def plot_models(cur_output_dir,pred_dict, gt_dict, SIM_PERIODS,infoIDs,desired_output_types,LOG_LIST=[True]):

    print("\nMaking plots...")
    for LOG in LOG_LIST:
        plot_output_dir = cur_output_dir + "Plots-LOG-%s/"%LOG
        create_output_dir(plot_output_dir)

        for SIM_PERIOD in range(1, SIM_PERIODS + 1):
            for output_type in desired_output_types:
                cur_plot_output_dir = plot_output_dir + "SIM_PERIOD-" + str(SIM_PERIOD) + "/" + str(output_type) + "/"
                create_output_dir(cur_plot_output_dir)

                for infoID in infoIDs:

                    hyp_infoID = infoID.replace("/", "_")

                    if LOG == True:
                        #get time series
                        pred_ts = pred_dict[infoID][SIM_PERIOD][output_type] +1
                        gt_ts = gt_dict[infoID][SIM_PERIOD][output_type] + 1

                    #plot params
                    fig, ax = plt.subplots(figsize=(15,10))

                    title =  infoID.capitalize() + "\n" + output_type + "\n" + "SIM_PERIOD-%d"%(SIM_PERIOD)
                    fig.suptitle(title, fontsize=25)
                    # fig.suptitle(suptitle)

                    # #plot gt
                    ax.plot(gt_ts,label='GT',color='black',lw=4)

                    # #plot sims
                    ax.plot(pred_ts,label="Pred",color='red',linestyle='-',lw=4)

                    #set labels
                    plt.xlabel("Time (Hours)",fontSize=25)
                    if "nunique_old" in output_type:
                        task = "Old Users"
                    if "actions" in output_type:
                        task = "Actions"
                    if "new_users" in output_type:
                        task = "New Users"

                    plt.ylabel("# %s"%task,fontSize=25)
                    plt.xticks(fontsize=20)
                    plt.yticks(fontsize=20)
                    plt.legend(fontsize=20)

                    if LOG == True:
                        plt.yscale('log',basey=10)
                    plt.tight_layout()

                    #png results
                    output_fp = cur_plot_output_dir+ hyp_infoID + ".png"
                    plt.savefig(output_fp)
                    print(output_fp)

                    plt.close()



    return

def plot_models_v2(cur_output_dir,pred_dict, gt_dict, SIM_PERIOD_LIST,infoIDs,desired_output_types,LOG_LIST=[True]):

    print("\nMaking plots...")
    for LOG in LOG_LIST:
        plot_output_dir = cur_output_dir + "Plots-LOG-%s/"%LOG
        create_output_dir(plot_output_dir)

        for SIM_PERIOD in SIM_PERIOD_LIST:
            for output_type in desired_output_types:
                cur_plot_output_dir = plot_output_dir + "SIM_PERIOD-" + str(SIM_PERIOD) + "/" + str(output_type) + "/"
                create_output_dir(cur_plot_output_dir)

                for infoID in infoIDs:

                    hyp_infoID = infoID.replace("/", "_")

                    if LOG == True:
                        #get time series
                        pred_ts = pred_dict[infoID][SIM_PERIOD][output_type] +1
                        gt_ts = gt_dict[infoID][SIM_PERIOD][output_type] + 1

                    #plot params
                    fig, ax = plt.subplots(figsize=(15,10))

                    title =  infoID.capitalize() + "\n" + output_type + "\n" + "SIM_PERIOD-%d"%(SIM_PERIOD)
                    fig.suptitle(title, fontsize=25)
                    # fig.suptitle(suptitle)

                    # #plot gt
                    ax.plot(gt_ts,label='GT',color='black',lw=4)

                    # #plot sims
                    ax.plot(pred_ts,label="Pred",color='red',linestyle='-',lw=4)

                    #set labels
                    plt.xlabel("Time (Hours)",fontSize=25)
                    if "nunique_old" in output_type:
                        task = "Old Users"
                    if "actions" in output_type:
                        task = "Actions"
                    if "new_users" in output_type:
                        task = "New Users"

                    plt.ylabel("# %s"%task,fontSize=25)
                    plt.xticks(fontsize=20)
                    plt.yticks(fontsize=20)
                    plt.legend(fontsize=20)

                    if LOG == True:
                        plt.yscale('log',basey=10)
                    plt.tight_layout()

                    #png results
                    output_fp = cur_plot_output_dir+ hyp_infoID + ".png"
                    plt.savefig(output_fp)
                    print(output_fp)

                    plt.close()



    return



def get_tne_metric_results(y_test_pred_infoID_dict, y_test_gt_infoID_dict, DESIRED_METRICS, metric_to_func_dict, pred_output_tag,
    cur_output_dir, infoIDs, DESIRED_EVALUATION_TIMESTEP, desired_output_types):

    #=============================== get metric results ========================================
    y_test_pred_full_metric_df = get_para_full_metric_df(y_test_pred_infoID_dict, y_test_gt_infoID_dict, DESIRED_METRICS, metric_to_func_dict,infoIDs)

    #save
    pred_output_dir = cur_output_dir + pred_output_tag + "/"
    create_output_dir(pred_output_dir)

    #preds
    gzip_save(y_test_gt_infoID_dict, pred_output_dir + "y_test_gt_infoID_dict")
    gzip_save(y_test_pred_infoID_dict , pred_output_dir + "y_test_pred_infoID_dict")

    #metrics
    y_test_pred_full_metric_df.to_csv(pred_output_dir + "y_test_pred_full_metric_df.csv", index=False)

    #==================== avg up metric results and save the infoID dfs =========================

    print()
    print(y_test_pred_full_metric_df)

    #kick out
    y_test_mean_pred_metric_df = y_test_pred_full_metric_df[["infoID", "metric", "metric_result"]]
    y_test_mean_pred_metric_df["metric_result"] = y_test_mean_pred_metric_df.groupby(["infoID", "metric"])["metric_result"].transform("mean")
    print()
    y_test_mean_pred_metric_df=y_test_mean_pred_metric_df.drop_duplicates().reset_index(drop=True)
    print(y_test_mean_pred_metric_df )

    #save
    y_test_mean_pred_metric_df.to_csv(pred_output_dir + "y_test_mean_pred_metric_df.csv")

    #get simplified metrics
    y_test_simple_metric_df = y_test_mean_pred_metric_df[["metric", "metric_result"]]
    y_test_simple_metric_df["metric_result"] = y_test_simple_metric_df.groupby(["metric"])["metric_result"].transform("mean")
    y_test_simple_metric_df = y_test_simple_metric_df.drop_duplicates().reset_index(drop=True)
    print()
    print(y_test_simple_metric_df)

    output_fp = pred_output_dir + "y_test_simple-summ-metric-results.csv"
    y_test_simple_metric_df.to_csv(output_fp, index=False)
    print(output_fp)

    #plots

    plot_models_v2(pred_output_dir,y_test_pred_infoID_dict, y_test_gt_infoID_dict, [DESIRED_EVALUATION_TIMESTEP],infoIDs,desired_output_types)


    return

def get_remaining_configs(model_param_main_input_dir,tracker_input_dir):

    main_combo_df = pd.read_csv(model_param_main_input_dir + "eval-combos.csv")
    print()
    print(main_combo_df)
    NUM_COMBOS= main_combo_df.shape[0]

    #check what you've done so far
    comp_so_far = os.listdir(tracker_input_dir)
    combo_idx_done_list = []
    for c in comp_so_far:
        # print(c)
        c_split = c.split("Combo")

        # print(c_split)
        c_split2 = c_split[1].split("-")
        combo_idx_done_list.append(int(c_split2[1]))

    #kickout completed nums
    main_combo_df = main_combo_df[~main_combo_df["combo_idx"].isin(combo_idx_done_list)].reset_index(drop=True)
    print()
    print(main_combo_df)

    #get combo dicts
    combo_dicts = main_combo_df.to_dict('records')

    return combo_dicts


def aggregate_edge_preds_to_volume_preds_v2_from_df(pred_df, infoIDs,  NEW_USER_DIR, eval_tag,DESIRED_EVALUATION_TIMESTEP, NUM_SIM_PERIODS, TS_PER_PERIOD=24):

    pred_df["SIM_PERIOD"] = DESIRED_EVALUATION_TIMESTEP
    pred_df = pred_df.rename(columns={"child":"user"})
    remove_cols = ["parent"]
    for c in remove_cols:
        if c in list(pred_df):
            pred_df = pred_df.drop(c, axis=1)

    print()
    print(pred_df)

    # sys.exit(0)

    #setup output types
    output_types = [
        "twitter_platform_infoID_pair_nunique_new_users",
        "twitter_platform_infoID_pair_nunique_old_users",
        "twitter_platform_infoID_pair_num_actions"
    ]

    #name cols
    new_user_col =  "twitter_platform_infoID_pair_nunique_new_users"
    old_user_col = "twitter_platform_infoID_pair_nunique_old_users"
    action_col = "twitter_platform_infoID_pair_num_actions"

    #make pred dict
    pred_dict = {}


    #fix time cols
    ts_cols = ["ts_%d"%(idx+1) for idx in range(TS_PER_PERIOD)]


    #make action initial df
    action_agg_df = pred_df.copy()
    action_agg_df =action_agg_df.drop("user", axis=1)
    for ts_col in ts_cols:
        action_agg_df[ts_col ] = action_agg_df.groupby(["infoID","SIM_PERIOD"])[ts_col].transform("sum")
    action_agg_df = action_agg_df.drop_duplicates().reset_index(drop=True)
    print()
    print(action_agg_df)

    #make old user initial df
    old_user_agg_df = pred_df.copy()
    print()
    print(old_user_agg_df)

    #add up actions user did in each ts
    for ts_col in ts_cols:
        old_user_agg_df[ts_col ] = old_user_agg_df.groupby(["infoID","SIM_PERIOD","user"])[ts_col].transform("sum")

    #with total actions of each old user, but with dupes
    print("\nWith summed hourly ts with dupes")
    print(old_user_agg_df)

    print("\nNo dupe user records")
    old_user_agg_df = old_user_agg_df.drop_duplicates("user").reset_index(drop=True)
    print(old_user_agg_df)

    #we only need to know if user was active or not
    old_user_agg_df[ts_cols] = old_user_agg_df[ts_cols].clip(upper=1)
    print("\nBinary old df")
    print(old_user_agg_df)

    #now kick out user col
    old_user_agg_df = old_user_agg_df.drop("user", axis=1)

    #now just get active users per ts
    for ts_col in ts_cols:
        old_user_agg_df[ts_col ] = old_user_agg_df.groupby(["infoID","SIM_PERIOD"])[ts_col].transform("sum")
    old_user_agg_df = old_user_agg_df.drop_duplicates().reset_index(drop=True)
    print("\nold_user_agg_df")
    print(old_user_agg_df)

    print("\naction_agg_df")
    print(action_agg_df)

    #now make full dfs
    for infoID in infoIDs:
        pred_dict[infoID] = {}
        SIM_PERIODS = old_user_agg_df["SIM_PERIOD"].unique()
        for SIM_PERIOD in SIM_PERIODS:

            old_user_temp_sim_period_df = old_user_agg_df[(old_user_agg_df["infoID"]==infoID) & (old_user_agg_df["SIM_PERIOD"]==SIM_PERIOD)].reset_index(drop=True)
            action_temp_sim_period_df = action_agg_df[(action_agg_df["infoID"]==infoID) & (action_agg_df["SIM_PERIOD"]==SIM_PERIOD)].reset_index(drop=True)
            print()
            print(old_user_temp_sim_period_df)
            print()
            print(action_temp_sim_period_df)

            #get arrays
            actions = action_temp_sim_period_df[ts_cols].values.flatten()
            old_users = old_user_temp_sim_period_df[ts_cols].values.flatten()
            print()
            print(actions)
            print(old_users)

            if NEW_USER_DIR == None:
                new_users = [0 for t in ts_cols]
            else:
                hyp_infoID = infoID.replace("/","_")
                cur_new_user_input_fp = NEW_USER_DIR + hyp_infoID + "/%s-day-%d-of-%d.csv"%(eval_tag, SIM_PERIOD, NUM_SIM_PERIODS)
                print(cur_new_user_input_fp)
                new_user_df = pd.read_csv(cur_new_user_input_fp)
                new_user_df = new_user_df.sort_values("timestep").reset_index(drop=True)
                print()
                print(new_user_df)
                new_users = new_user_df["num_new_users"].values
                new_user_actions = new_user_df["num_new_user_actions"].values
                actions = actions + new_user_actions


            # print(new_users)

            #make df
            data={new_user_col:new_users, old_user_col:old_users, action_col:actions}
            cur_df = pd.DataFrame(data=data)
            cur_df["timestep"] = [i+1 for i in range(cur_df.shape[0])]
            cur_df = cur_df[["timestep",new_user_col, old_user_col, action_col]]
            pred_dict[infoID][SIM_PERIOD] = cur_df

            print()
            print(cur_df)

                # sys.exit(0)

    return pred_dict

def get_full_gt_dict(full_gt_input_dir,DESIRED_EVALUATION_TIMESTEP, eval_tag,NUM_EVAL_PERIODS,infoIDs):

    new_user_col =  "twitter_platform_infoID_pair_nunique_new_users"
    old_user_col = "twitter_platform_infoID_pair_nunique_old_users"
    action_col = "twitter_platform_infoID_pair_num_actions"

    gt_dict = {}
    SIM_PERIODS = [DESIRED_EVALUATION_TIMESTEP]
    for infoID in infoIDs:
        hyp_infoID = infoID.replace("/","_")
        gt_dict[infoID] = {}
        for SIM_PERIOD in SIM_PERIODS:
            cur_gt_input_fp = full_gt_input_dir + hyp_infoID + "/%s-day-%d-of-%d.csv"%(eval_tag, SIM_PERIOD, NUM_EVAL_PERIODS)
            cur_gdf = pd.read_csv(cur_gt_input_fp)
            print()
            print(cur_gdf)

            cur_gdf[new_user_col] = cur_gdf["num_new_users"].copy()
            cur_gdf[old_user_col] = cur_gdf["num_old_users"].copy()
            cur_gdf[action_col] = cur_gdf["num_new_user_actions"] + cur_gdf["num_old_user_actions"]
            cur_gdf = cur_gdf[["timestep", new_user_col, old_user_col, action_col]]
            gt_dict[infoID][SIM_PERIOD] = cur_gdf


    return gt_dict

def get_para_full_metric_df( y_pred_df_dict, y_true_df_dict, DESIRED_METRICS, metric_to_func_dict,infoIDs, NJOBS=32):

    output_types= [
    "twitter_platform_infoID_pair_nunique_new_users",
    "twitter_platform_infoID_pair_nunique_old_users" ,
    "twitter_platform_infoID_pair_num_actions"]

    metric_result_dict = {}

    tuple_idx = 1
    all_arg_tuples = []

    for METRIC in DESIRED_METRICS:

        # infoID_df_list = []
        # output_type_df_list = []
        # sim_period_df_list = []
        # metric_result_df_list = []

        for infoID in infoIDs:
            metric_result_dict[infoID] = {}

            y_pred_infoID_dict = y_pred_df_dict[infoID]
            y_true_infoID_dict = y_true_df_dict[infoID]

            num_tuples = len(DESIRED_METRICS)*len(infoIDs)*len(y_pred_infoID_dict) * len(output_types)

            for SIM_PERIOD, y_pred_df in y_pred_infoID_dict.items():

                # print()
                # print(SIM_PERIOD)
                # print(y_pred_df)

                y_true_df = y_true_infoID_dict[SIM_PERIOD]

                for output_type in output_types:
                    y_pred = y_pred_df[output_type]
                    y_true = y_true_df[output_type]
                    metric_func = metric_to_func_dict[METRIC]

                    arg_tuple = (METRIC, infoID, SIM_PERIOD, output_type, y_true, y_pred, metric_func, tuple_idx, num_tuples)
                    all_arg_tuples.append(arg_tuple)
                    tuple_idx+=1

    if len(all_arg_tuples) != num_tuples:
        print("\nError! len(all_arg_tuples) != num_tuples")
        print(len(all_arg_tuples))
        print(num_tuples)
        sys.exit(0)

    #multi proc
    global pool
    pool = Pool(NJOBS)
    print("\nStarting multiproc...")
    results = pool.map(get_mini_metric_df, all_arg_tuples)

    #added 2/17/22 -> prevents memory leak
    pool.close()
    pool.join()


    full_metric_df = pd.concat(results)
    print("\nfull_metric df after multiproc")
    print(full_metric_df)

    return full_metric_df

def get_mini_metric_df(arg_tuple):

    METRIC, infoID, SIM_PERIOD, output_type, y_true, y_pred, metric_func, tuple_idx, num_tuples = arg_tuple
    metric_result = metric_func(y_true, y_pred)

    metric_df = pd.DataFrame(data={"infoID":[infoID], "output_type":[output_type], "SIM_PERIOD":[SIM_PERIOD], "metric":[METRIC],
        "metric_result":[metric_result]})

    MOD = 300
    if tuple_idx%MOD == 0:
        print("Done with mini metric df %d of %d"%(tuple_idx, num_tuples))

    return metric_df