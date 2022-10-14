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
import xgboost as xgb
import joblib
# from ft_categories import *
import multiprocessing as mp

from sample_gen_aux_funcs import *
from basic_utils import *
from multiprocessing import Pool
import basic_utils as bu
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

output_type_to_task_dict = {
    "twitter_platform_infoID_pair_nunique_new_users" : "New_Users",
    "twitter_platform_infoID_pair_num_actions" : "Activities",
    "twitter_platform_infoID_pair_nunique_old_users" : "Old_Users",
        "youtube_platform_infoID_pair_nunique_new_users" : "New_Users",
    "youtube_platform_infoID_pair_num_actions" : "Activities",
    "youtube_platform_infoID_pair_nunique_old_users" : "Old_Users",

}

def load_xgb_pred_dict(xgb_pred_count_dir, infoIDs, output_types, NUM_SIM_PERIODS, DESIRED_VAM_XG_MODEL):

    pred_dict = {}

    for infoID in infoIDs:
        pred_dict[infoID] = {}
        hyp_infoID = infoID.replace("/", "_")

        for SIM_PERIOD in range(1, NUM_SIM_PERIODS+1):


            ot_to_series_dict = {}
            for output_type in output_types:
                fp = xgb_pred_count_dir + DESIRED_VAM_XG_MODEL+"/" + hyp_infoID + "/" + output_type + "/SIM-PERIOD-" + str(SIM_PERIOD) + "-of-" + str(NUM_SIM_PERIODS) + ".csv"
                df = pd.read_csv(fp)
                # print()
                # print(df)

                forecast = df["forecast"]
                ot_to_series_dict[output_type] = forecast
            df = pd.DataFrame(data=ot_to_series_dict)
            print()
            print(df)

            pred_dict[infoID][SIM_PERIOD]=df

    return pred_dict

def xg_vam_metric_df(input_dir, infoIDs, DESIRED_METRICS, output_types, DESIRED_MODEL, result_type, platform="twitter"):

    all_metric_dfs = []
    for infoID in infoIDs:
        hyp_infoID = infoID.replace("/", "_")
        for output_type in output_types:
            for METRIC in DESIRED_METRICS:

                metric_dir = input_dir + hyp_infoID + "/" + output_type + "/" + METRIC + "/"
                metric_fp = metric_dir + "%s-%s-%s-%s-%s-results.csv"%(platform, hyp_infoID, output_type, METRIC, result_type)
                metric_df = pd.read_csv(metric_fp)
                metric_df = metric_df[metric_df["model"]==DESIRED_MODEL].reset_index(drop=True)
                all_metric_dfs.append(metric_df)

    metric_df = pd.concat(all_metric_dfs)
    print()
    print(metric_df)

    return metric_df


def get_combined_xgb_and_lstm_metric_df(raw_lstm_test_pred_metric_df,raw_xgb_test_pred_metric_df,infoIDs,output_types,VAM_LSTM_NEW_TAG,VAM_XGB_NEW_TAG, DEBUG):
    raw_lstm_test_pred_metric_df = raw_lstm_test_pred_metric_df[["infoID", "output_type", "SIM_PERIOD", "metric", "metric_result"]]
    raw_xgb_test_pred_metric_df=raw_xgb_test_pred_metric_df[["infoID", "output_type", "SIM_PERIOD", "metric", "metric_result"]]
    merge_cols  = ["infoID", "output_type", "SIM_PERIOD", "metric"]

    if DEBUG == True:
        raw_lstm_test_pred_metric_df = raw_lstm_test_pred_metric_df[raw_lstm_test_pred_metric_df["output_type"].isin(output_types)]
        raw_lstm_test_pred_metric_df = raw_lstm_test_pred_metric_df[raw_lstm_test_pred_metric_df["infoID"].isin(infoIDs)].reset_index(drop=True)

    lstm_size = raw_lstm_test_pred_metric_df.shape[0]
    xgb_size = raw_xgb_test_pred_metric_df.shape[0]
    if lstm_size != xgb_size:
        print("\nError! lstm_size!= xgb_size")
        print(lstm_size)
        print(xgb_size)
        sys.exit(0)

    raw_lstm_test_pred_metric_df = raw_lstm_test_pred_metric_df.rename(columns={"metric_result":VAM_LSTM_NEW_TAG})
    raw_xgb_test_pred_metric_df = raw_xgb_test_pred_metric_df.rename(columns={"metric_result":VAM_XGB_NEW_TAG, "metric_name":"metric"})
    combined_test_metric_df = pd.merge(raw_lstm_test_pred_metric_df, raw_xgb_test_pred_metric_df, on=merge_cols, how="inner").reset_index(drop=True)
    print()
    lstm_errors = list(combined_test_metric_df[VAM_LSTM_NEW_TAG])
    xgb_errors = list(combined_test_metric_df[VAM_XGB_NEW_TAG])
    combined_test_metric_df["LSTM_is_winner"] = [1 if le < xe else 1 if ((le==xe) and le==0 ) else 0 for le,xe in zip(lstm_errors, xgb_errors) ]
    combined_test_metric_df["LSTM_PIFB"] = 100.0 * (combined_test_metric_df[VAM_XGB_NEW_TAG] - combined_test_metric_df[VAM_LSTM_NEW_TAG])/combined_test_metric_df[VAM_XGB_NEW_TAG]
    combined_test_metric_df = combined_test_metric_df.sort_values("LSTM_PIFB", ascending=False).reset_index(drop=True)
    print(combined_test_metric_df)

    return combined_test_metric_df

def plot_top_models(rank_df, best_results_output_dir, main_model_pred_dict, bl_model_pred_dict, gt_dict ,rank_col ,main_tag, bl_tag ,TOP=20):

    bu.create_output_dir(best_results_output_dir)

    rank_df = rank_df.sort_values(rank_col, ascending=False).reset_index(drop=True)
    top_df = rank_df.head(TOP)

    print("\ntop_df")
    print(top_df)

    for LOG in [True, False]:
        for idx, row in top_df.iterrows():

            rank = idx+1

            infoID = row["infoID"]
            output_type = row["output_type"]
            SIM_PERIOD = row["SIM_PERIOD"]
            rank_val = row[rank_col]

            main_model_ts = main_model_pred_dict[infoID][SIM_PERIOD][output_type]
            bl_model_ts = bl_model_pred_dict[infoID][SIM_PERIOD][output_type]
            gt_ts = gt_dict[infoID][SIM_PERIOD][output_type]

            if LOG == True:
                main_model_ts = main_model_ts +1
                bl_model_ts = bl_model_ts + 1
                gt_ts = gt_ts + 1

            #plot params
            fig, ax = plt.subplots(figsize=(15,10))


            hyp_infoID = infoID.replace("/", "_")
            title =  infoID.capitalize() + "\n" + output_type + "\n" + "SIM_PERIOD-%d"%(SIM_PERIOD) + "\n" + rank_col + ": " + str(rank_val)
            fig.suptitle(title, fontsize=25)
            # fig.suptitle(suptitle)

            # #plot gt
            ax.plot(gt_ts,label='GT',color='black',lw=4)

            # #plot sims
            ax.plot(main_model_ts,label=main_tag,color='red',linestyle='-',lw=4)
            ax.plot(bl_model_ts,label=bl_tag,color='green',linestyle="-",lw=4)

            #set labels
            plt.xlabel("Time (Hours)",fontSize=25)
            task = output_type_to_task_dict[output_type]
            plt.ylabel("# %s"%task,fontSize=25)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.legend(fontsize=20)

            if LOG == True:
                plt.yscale('log',basey=10)
            plt.tight_layout()

            # #svg results
            # svg_dir = best_results_output_dir +"/SVGs/LOG-%s/"%LOG
            # bu.create_output_dir(svg_dir)
            # output_fp =  "Rank-%d-LOG-%s.svg"%(rank, LOG)
            # plt.savefig(output_fp)

            #png results
            png_dir = best_results_output_dir +"LOG-%s/"%LOG
            bu.create_output_dir(png_dir)
            output_fp = png_dir+ "Rank-%d-LOG-%s.png"%(rank, LOG)
            plt.savefig(output_fp)

            plt.close()

    return