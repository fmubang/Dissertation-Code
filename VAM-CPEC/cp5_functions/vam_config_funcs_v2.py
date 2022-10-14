import sys
# sys.path.append("/data/Fmubang/cp4-code-clean/functions")
sys.path.append("/beegfs-mnt/data/gaivi2/data/Fmubang/cp4-VAM-write-up-exps/functions")
import pandas as pd
import os,sys
from scipy import stats
import numpy as np
from cascade_ft_funcs import *
import pickle
from random import seed
from random import random
from random import randrange
import xgboost as xgb
import joblib
from ft_categories import *
from ast import literal_eval
# print("\nft_to_ft_cat_dict")
# print(ft_to_ft_cat_dict)
from ft_categories_v2 import ft_to_ft_cat_dict_v2_sep_aux_infoID_fts
# from keras.layers import Bidirectional
from pnnl_metric_funcs_v2 import *
# from p3_SampleGenerator import SampleGenerator
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
# from p3_SampleGenerator_v2_corrs import SampleGenerator_V2

import seaborn as sns
from textwrap  import wrap
import matplotlib.dates as mdates
from sklearn.svm import SVR
from sklearn.metrics import ndcg_score
from bisect import bisect

def get_ndcg_score(y_test, y_pred):

    df = pd.DataFrame(data={"y_test":y_test, "y_pred":y_pred})
    df = df.sort_values("y_test", ascending=False)
    df = df.reset_index(drop=True)

    y_test = [df["y_test"].values]
    y_pred = [df["y_pred"].values]

    ndcg = ndcg_score(y_test,y_pred)

    return ndcg


def get_time_series_df(df, start, end, GRAN, token_col, tag, main_output_dir):

    create_output_dir(main_output_dir)

    df = config_df_by_dates(df, start, end)
    df["nodeTime"] = df["nodeTime"].dt.floor(GRAN)

    tokens = list(df[token_col].unique())
    print(tokens)

    #get dates
    start = df["nodeTime"].iloc[0]
    end = df["nodeTime"].iloc[-1]
    dates = pd.date_range(start, end, freq=GRAN)
    date_df = pd.DataFrame(data={"nodeTime":dates})
    date_df["nodeTime"] = pd.to_datetime(date_df["nodeTime"], utc=True)


    for token in tokens:
        df[token] = df[token_col].isin([token]).astype("int32")
    print(df)

    df = df[["nodeTime"]+tokens]
    df = df.merge(date_df, on="nodeTime", how="outer").fillna(0)

    for token in tokens:
        df[token] = df.groupby(["nodeTime"])[token].transform("sum")

    df = df.drop_duplicates().reset_index(drop=True)
    print(df)

    output_fp = main_output_dir + "%s-Time-Series-%s-to-%s-GRAN-%s.csv"%(tag, start,end, GRAN)
    df.to_csv(output_fp,index=False)
    print(output_fp)

    return df


def plot_basic(df, x_col, y_col, title, output_tag, kind):

    time_df[x_col] = minmax_scale_series(time_df[x_col])
    time_df[y_col] = minmax_scale_series(time_df[y_col])
    time_df.plot(x = x_col, y=y_col, kind=kind)
    fig = plt.gcf()
    ax = plt.gca()
    ax.set_title(title)
    output_fp = output_dir + "%s.png"%output_tag
    fig.savefig(output_fp)
    print(output_fp)
    output_fp = output_dir + "%s.svg"%output_tag
    fig.savefig(output_fp)
    print(output_fp)
    plt.close()


def minmax_scale_series(series):

    smax = series.max()
    smin = series.min()
    series = (series - smin)/(smax - smin)

    return series

def rename_model_tag_by_day_v2_with_ignore(model_tags, ignore_tags):

    model_tags = list(model_tags)

    new_tags = []
    for model_tag in model_tags:

        if model_tag in ignore_tags:
            new_tags.append(model_tag)
            continue

        split_tag_list = model_tag.split("-")
        num = int(split_tag_list[2])
        new_tag = split_tag_list[0] + "-" + split_tag_list[1] + "-" + str(int(num/24))
        new_tags.append(new_tag)

    return new_tags


def get_all_users_edge_weight_dist_table(infoIDs,edge_weight_df):
    all_users_edge_weight_dist_table = edge_weight_df.copy()
    parent_user_val_counts = all_users_edge_weight_dist_table["parentUserID"].value_counts()
    print("\nparent_user_val_counts")
    print(parent_user_val_counts)

    infoID_to_all_users_edge_weight_dist_table_dict = {}
    all_users_edge_weight_dist_table["edge_weight"] = all_users_edge_weight_dist_table.groupby(["nodeUserID", "informationID", "parentUserID"])["edge_weight_this_timestep"].transform("sum")
    for infoID in infoIDs:
        cur_all_users_edge_weight_dist_table = all_users_edge_weight_dist_table[all_users_edge_weight_dist_table["informationID"]==infoID]
        cur_all_users_edge_weight_dist_table = cur_all_users_edge_weight_dist_table[["nodeUserID", "parentUserID", "edge_weight", "informationID"]].drop_duplicates().reset_index(drop=True)
        cur_all_users_edge_weight_dist_table["edge_weight"]=cur_all_users_edge_weight_dist_table["edge_weight"]/cur_all_users_edge_weight_dist_table["edge_weight"].sum()
        infoID_to_all_users_edge_weight_dist_table_dict[infoID] = cur_all_users_edge_weight_dist_table
        print("\ncur_all_users_edge_weight_dist_table")
        print(cur_all_users_edge_weight_dist_table)
    return infoID_to_all_users_edge_weight_dist_table_dict

def get_pnnl_baseline_dates(sample_gen_param_dict):

    #get num days in test period
    test_start = sample_gen_param_dict["test_start"]
    test_end = sample_gen_param_dict["test_end"]
    train_start = sample_gen_param_dict["train_start"]
    train_end = sample_gen_param_dict["train_end"]
    val_start = sample_gen_param_dict["val_start"]
    val_end = sample_gen_param_dict["val_end"]

    train_dates = pd.date_range(train_start, train_end, freq="S")
    val_dates = pd.date_range(val_start, val_end, freq="S")
    test_dates = pd.date_range(test_start, test_end, freq="S")

    train_date_series = pd.Series(train_dates)
    val_date_series = pd.Series(val_dates)
    test_date_series = pd.Series(test_dates)

    num_test_dates = len(test_date_series)
    new_test_dates =test_date_series - pd.Timedelta(seconds=num_test_dates)
    new_train_dates =train_date_series - pd.Timedelta(seconds=num_test_dates)
    new_val_dates =val_date_series - pd.Timedelta(seconds=num_test_dates)

    print("\nnum_test_dates: %d"%num_test_dates)
    print("\nnew_train_dates")
    print(new_train_dates)
    print("\nnew_val_dates")
    print(new_val_dates)
    print("\nnew_test_dates")
    print(new_test_dates)

    train_end = new_train_dates.iloc[-1]
    val_start = new_val_dates.iloc[0]
    val_end = new_val_dates.iloc[-1]
    test_start = new_test_dates.iloc[0]
    test_end = new_test_dates.iloc[-1]

    sample_gen_param_dict["train_start"]=pd.to_datetime(train_start, utc=True)
    sample_gen_param_dict["train_end"]=pd.to_datetime(train_end, utc=True)
    sample_gen_param_dict["val_start"]=pd.to_datetime(val_start, utc=True)
    sample_gen_param_dict["val_end"]=pd.to_datetime(val_end, utc=True)
    sample_gen_param_dict["test_start"]=pd.to_datetime(test_start, utc=True)
    sample_gen_param_dict["test_end"]=pd.to_datetime(test_end, utc=True)

    key_prints = ["train_start", "train_end","val_start", "val_end","test_start", "test_end"]

    print("\nnew dates")
    for date_key in key_prints:
        print("%s: %s"%(date_key, str(sample_gen_param_dict[date_key])))
    # sys.exit(0)



    return sample_gen_param_dict


def get_pnnl_baseline_data(sample_gen_param_dict,SampleGenerator):

    baseline_dates_sample_gen_param_dict = dict(get_pnnl_baseline_dates(sample_gen_param_dict))

    sample_generator = SampleGenerator(**baseline_dates_sample_gen_param_dict)
    print("\nsample_generator")
    print(sample_generator)



    infoIDs = list(sample_gen_param_dict["infoIDs"])
    #==================== put corrs in here ====================

    #get array dict
    infoID_train_and_test_array_dict = sample_generator.configure_features_and_get_infoID_train_and_test_array_dict()

    print("\nMaking baselines...")
    y_pred_baseline_test_data_dict = {}
    y_pred_baseline_val_data_dict = {}
    y_pred_baseline_test_sliding_window_data_dict = {}
    y_pred_baseline_val_sliding_window_data_dict = {}
    for infoID in infoIDs:
        y_pred_baseline_test_data_dict[infoID] = infoID_train_and_test_array_dict[infoID]["y_test"]
        y_pred_baseline_val_data_dict[infoID] = infoID_train_and_test_array_dict[infoID]["y_val"]
        y_pred_baseline_test_sliding_window_data_dict[infoID] = infoID_train_and_test_array_dict[infoID]["y_test_sliding_window"]
        y_pred_baseline_val_sliding_window_data_dict[infoID] = infoID_train_and_test_array_dict[infoID]["y_val_sliding_window"]

    # # #baselines
    # print("\nMaking baselines...")
    # y_pred_baseline_test_data_dict = sample_generator.get_test_data_baseline_pred_dict().copy()
    # y_pred_baseline_val_data_dict = sample_generator.get_val_data_baseline_pred_dict().copy()
    # y_pred_baseline_test_sliding_window_data_dict = sample_generator.get_test_sliding_window_data_baseline_pred_dict().copy()
    # y_pred_baseline_val_sliding_window_data_dict = sample_generator.get_val_sliding_window_data_baseline_pred_dict().copy()

    return y_pred_baseline_test_data_dict,y_pred_baseline_val_data_dict,y_pred_baseline_test_sliding_window_data_dict,y_pred_baseline_val_sliding_window_data_dict


def get_baseline_data(sample_gen_param_dict,SampleGenerator):
    sample_generator = SampleGenerator(**sample_gen_param_dict)
    print("\nsample_generator")
    print(sample_generator)

    #==================== put corrs in here ====================

    #get array dict
    infoID_train_and_test_array_dict = sample_generator.configure_features_and_get_infoID_train_and_test_array_dict()

    # sys.exit(0)

    target_ft_category_to_list_of_temporal_indices_dict = sample_generator.get_target_ft_category_to_list_of_temporal_indices()
    for target_ft_cat,target_ft_cat_dict in target_ft_category_to_list_of_temporal_indices_dict.items():
        target_ft_idx_list = target_ft_cat_dict["indices_list"]
        print("%s: %s"%(target_ft_cat, target_ft_idx_list))


    input_ft_category_to_list_of_temporal_indices_dict = sample_generator.get_input_ft_category_to_list_of_temporal_indices()
    for input_ft_cat,input_ft_cat_dict in input_ft_category_to_list_of_temporal_indices_dict.items():
        print()
        input_ft_idx_list = input_ft_cat_dict["indices_list"]
        print("%s: %s"%(input_ft_cat, input_ft_idx_list))

    #============================================================

    #train data
    x_train,y_train = sample_generator.configure_training_data()


    # #get the other arrays
    x_val,y_val = sample_generator.configure_val_data()
    x_test,y_test = sample_generator.configure_test_data()
    x_val_sliding_window,y_val_sliding_window = sample_generator.configure_val_sliding_window_data()
    x_test_sliding_window,y_test_sliding_window = sample_generator.configure_test_sliding_window_data()

    # #baselines
    print("\nMaking baselines...")
    y_pred_baseline_test_data_dict = sample_generator.get_test_data_baseline_pred_dict().copy()
    y_pred_baseline_val_data_dict = sample_generator.get_val_data_baseline_pred_dict().copy()
    y_pred_baseline_test_sliding_window_data_dict = sample_generator.get_test_sliding_window_data_baseline_pred_dict().copy()
    y_pred_baseline_val_sliding_window_data_dict = sample_generator.get_val_sliding_window_data_baseline_pred_dict().copy()



    return y_pred_baseline_test_data_dict,y_pred_baseline_val_data_dict,y_pred_baseline_test_sliding_window_data_dict,y_pred_baseline_val_sliding_window_data_dict

def get_summary_info_of_baseline_comp_df(comp_df, proper_baseline_dir, y_eval_tag, metric_tag, model_tag, baseline_tag):

    model_error_tag = model_tag + "_" + metric_tag
    baseline_error_tag = baseline_tag + "_" + metric_tag

    mean_model_error_tag =  "mean_" + model_tag + "_" + metric_tag
    mean_baseline_error_tag ="mean_" + baseline_tag + "_" + metric_tag

    nunique_timesteps = comp_df["timestep"].nunique()

    #get infoID view
    infoID_comp_df = comp_df.copy()
    infoID_comp_df[mean_model_error_tag] = infoID_comp_df.groupby(["infoID"])[model_error_tag].transform("mean")
    infoID_comp_df[mean_baseline_error_tag] = infoID_comp_df.groupby(["infoID"])[baseline_error_tag].transform("mean")
    infoID_comp_df["num_trials"] = infoID_comp_df.groupby(["infoID"])["is_winner"].transform("count")
    infoID_comp_df = infoID_comp_df[infoID_comp_df["is_winner"]!="tie"]
    infoID_comp_df["num_trials_no_ties"] = infoID_comp_df.groupby(["infoID"])["is_winner"].transform("count")
    infoID_comp_df["num_infoID_wins"] = infoID_comp_df.groupby(["infoID"])["is_winner"].transform("sum")
    infoID_comp_df["win_freq"] = infoID_comp_df["num_infoID_wins"]/(1.0 * infoID_comp_df["num_trials_no_ties"])
    infoID_keep_cols = ["infoID", mean_model_error_tag,mean_baseline_error_tag,"num_infoID_wins","num_trials" ,"num_trials_no_ties","win_freq"]
    infoID_comp_df= infoID_comp_df[infoID_keep_cols]
    infoID_comp_df = infoID_comp_df.drop_duplicates().reset_index(drop=True)

    print("\ninfoID_comp_df")
    print(infoID_comp_df)

    output_fp = proper_baseline_dir + y_eval_tag + "-" + "InfoID-Summary-Comparisons.csv"
    infoID_comp_df.to_csv(output_fp,index=False)
    print(output_fp)

    #get infoID view
    target_ft_comp_df = comp_df.copy()
    target_ft_comp_df[mean_model_error_tag] = target_ft_comp_df.groupby(["target_ft"])[model_error_tag].transform("mean")
    target_ft_comp_df[mean_baseline_error_tag] = target_ft_comp_df.groupby(["target_ft"])[baseline_error_tag].transform("mean")
    target_ft_comp_df["num_trials"] = target_ft_comp_df.groupby(["target_ft"])["is_winner"].transform("count")
    target_ft_comp_df = target_ft_comp_df[target_ft_comp_df["is_winner"]!="tie"]
    target_ft_comp_df["num_trials_no_ties"] = target_ft_comp_df.groupby(["target_ft"])["is_winner"].transform("count")
    target_ft_comp_df["num_target_ft_wins"] = target_ft_comp_df.groupby(["target_ft"])["is_winner"].transform("sum")
    target_ft_comp_df["win_freq"] = target_ft_comp_df["num_target_ft_wins"]/(1.0 * target_ft_comp_df["num_trials_no_ties"])
    target_ft_keep_cols = ["target_ft", mean_model_error_tag,mean_baseline_error_tag,"num_target_ft_wins","num_trials" ,"num_trials_no_ties","win_freq"]
    target_ft_comp_df= target_ft_comp_df[target_ft_keep_cols]
    target_ft_comp_df = target_ft_comp_df.drop_duplicates().reset_index(drop=True)

    print("\ntarget_ft_comp_df")
    print(target_ft_comp_df)

    output_fp = proper_baseline_dir + y_eval_tag + "-" + "target_ft-Summary-Comparisons.csv"
    target_ft_comp_df.to_csv(output_fp,index=False)
    print(output_fp)

    overall_summary_dict = {}
    overall_summary_dict["target_ft_" + mean_model_error_tag] = [target_ft_comp_df[mean_model_error_tag].mean()]
    overall_summary_dict["infoID_" + mean_model_error_tag] = [infoID_comp_df[mean_model_error_tag].mean()]
    overall_summary_dict["target_ft_" + mean_baseline_error_tag] = [target_ft_comp_df[mean_baseline_error_tag].mean()]
    overall_summary_dict["infoID_" + mean_baseline_error_tag] = [infoID_comp_df[mean_baseline_error_tag].mean()]

    overall_summary_dict["target_ft_mean_model_win_freq"] = [target_ft_comp_df["win_freq"].mean()]
    overall_summary_dict["infoID_mean_model_win_freq"] = [infoID_comp_df["win_freq"].mean()]

    col_order = [
    "target_ft_" + mean_model_error_tag,
    "infoID_" + mean_model_error_tag,
    "target_ft_" + mean_baseline_error_tag,
    "infoID_" + mean_baseline_error_tag,
    "target_ft_mean_model_win_freq",
    "infoID_mean_model_win_freq"
    ]

    overall_summary_df = pd.DataFrame(data=overall_summary_dict)
    overall_summary_df=overall_summary_df[col_order]

    print("\noverall_summary_df")
    print(overall_summary_df)

    output_fp = proper_baseline_dir + y_eval_tag + "-Overall-Summary.csv"
    overall_summary_df.to_csv(output_fp, index=False)
    print(output_fp)


    return infoID_comp_df, target_ft_comp_df,overall_summary_df


def agg_timesteps_to_higher_gran(agg_timestep_amount ,pred_df, pred_col, acutal_col,groupby_cols, timestep_col="timestep" ):

    timestep_group = []
    num_unique_timesteps = pred_df[timestep_col].nunique()
    print("\nnum_unique_timesteps")
    print(num_unique_timesteps)
    cur_group_num = 1
    timestep_to_group_num_dict = {}
    for t in range(1, num_unique_timesteps+1):
        timestep_to_group_num_dict[t] = cur_group_num

        if t%agg_timestep_amount == 0:
            cur_group_num+=1
    pred_df["timestep_group"] = pred_df[timestep_col].map(timestep_to_group_num_dict)

    print(pred_df)

    agg_cols = [pred_col, acutal_col]
    for agg_col in agg_cols:
        pred_df[agg_col] = pred_df.groupby(["timestep_group"]+list(groupby_cols))[agg_col].transform("sum")
        pred_df[agg_col] = np.round(pred_df[agg_col], 0)

    pred_df = pred_df.drop(timestep_col, axis=1)
    pred_df[timestep_col] = pred_df["timestep_group"]
    pred_df = pred_df.drop_duplicates().reset_index(drop=True)



    return pred_df

def get_nc_rmse(ground_truth, simulation):
    return pnnl_rmse(ground_truth, simulation, cumulative=True, normed=True)

def get_regular_rmse(ground_truth, simulation):
    return pnnl_rmse(ground_truth, simulation, cumulative=False, normed=False)

def get_ape(ground_truth, simulation):

    result = np.abs(ground_truth.sum()-simulation.sum())
    result = 100.0 * result / np.abs(float(ground_truth.sum()))

    return result


metric_str_to_function_dict={
    "rmse":get_regular_rmse,
    "nc_rmse":get_nc_rmse,
    "ape" : get_ape

}

model_type_to_model_func_dict={
    "XGBoost_Regressor":xgb.XGBRegressor,
    "KNN_Regressor":KNeighborsRegressor,
    "SVR_RBF":SVR
}

xgboost_param_name_order = [
"objective",
"colsample_bytree",
"learning_rate",
"max_depth",
"n_estimators",
"subsample",
"gamma",
"alpha",
"reg_lambda",
"random_state",
"n_jobs"]

knn_param_name_order = ["weights", "algorithm", "n_neighbors", "n_jobs"]

svr_rbf_name_order = ["kernel", "C", "epsilon", "gamma"]

model_type_to_param_name_order_dict={"XGBoost_Regressor":xgboost_param_name_order, "KNN_Regressor":knn_param_name_order, "SVR_RBF":svr_rbf_name_order}

class ModelHandler:

    def __init__(self, param_dict, model_type,model_type_to_param_name_order_dict):
        self.param_dict=param_dict
        self.model_type=model_type
        self.model_type_to_model_func_dict=model_type_to_model_func_dict
        self.model_type_to_param_name_order_dict=model_type_to_param_name_order_dict

        self.indiv_model_jobs = 1
        self.wrapper_model_n_jobs = int(param_dict["n_jobs"])
        self.param_dict["n_jobs"] = self.indiv_model_jobs


    def initialize_model(self):
        model_func = self.model_type_to_model_func_dict[self.model_type]
        print(model_func)

        if (self.model_type == "XGBoost_Regressor") or ("SVR" in self.model_type):
            self.model = MultiOutputRegressor(model_func(**self.param_dict), n_jobs=self.wrapper_model_n_jobs)
        else:
            self.model = model_func(**self.param_dict)
        print(self.model)
        return self.model

    def get_model_param_tag(self):

        param_name_order = model_type_to_param_name_order_dict[self.model_type]
        param_vals = []
        for param_name in param_name_order:
            param_val = self.param_dict[param_name]
            param_vals.append(param_val)
        self.param_tag = make_param_tag(param_vals)
        print(self.param_tag)
        self.model_param_df = pd.DataFrame(data={"param":param_name_order, "value":param_vals})
        print(self.model_param_df)
        return self.model_param_df,self.param_tag

def get_idx_to_model_param_dict(DEBUG,objective_list,
    colsample_bytree_list,
    learning_rate_list,
    max_depth_list,
    n_estimators_list,
    subsample_list,
    gamma_list,
    alpha_list,
    reg_lambda_list,
    random_state_list,
    n_jobs_list):

    print("\nGetting model param dict...")
    #============================= debug =============================
    if DEBUG == True:
        objective_list= objective_list[:1]
        colsample_bytree_list= colsample_bytree_list[:1]
        learning_rate_list= learning_rate_list[:1]
        max_depth_list= max_depth_list[:1]
        n_estimators_list= n_estimators_list[:1]
        subsample_list=subsample_list[:1]
        gamma_list= gamma_list[:1]
        alpha_list= alpha_list[:1]

        #l2
        reg_lambda_list= reg_lambda_list[:1]
        random_state_list= random_state_list[:1]
        n_jobs_list= n_jobs_list[:1]

    #============================= combo list =============================
    combo_list=[
    objective_list,
    colsample_bytree_list,
    learning_rate_list,
    max_depth_list,
    n_estimators_list,
    subsample_list,
    gamma_list,
    alpha_list,
    reg_lambda_list,
    random_state_list,
    n_jobs_list
    ]

    num_combos=1
    for cur_list in combo_list:
        num_combos = num_combos*len(cur_list)
    print("\nnum combos: %d"%num_combos)

    idx_to_model_param_dict = {}

    i=0
    for objective in objective_list:
        for colsample_bytree in colsample_bytree_list:
            for learning_rate in learning_rate_list:
                for max_depth in max_depth_list:
                    for n_estimators in n_estimators_list:
                        for subsample in subsample_list:
                            for gamma in gamma_list:
                                for alpha in alpha_list:
                                    for reg_lambda in reg_lambda_list:
                                        for random_state in random_state_list:
                                            for n_jobs in n_jobs_list:
                                                param_dict = {
                                                    "objective":objective,
                                                    "colsample_bytree":colsample_bytree,
                                                    "learning_rate":learning_rate,
                                                    "max_depth":max_depth,
                                                    "n_estimators":n_estimators,
                                                    "subsample":subsample,
                                                    "gamma":gamma,
                                                    "alpha":alpha,
                                                    "reg_lambda":reg_lambda,
                                                    "random_state":random_state,
                                                    "n_jobs":n_jobs
                                                    }
                                                idx_to_model_param_dict[i]=param_dict
                                                i+=1
    return idx_to_model_param_dict,num_combos

def get_idx_to_sample_generator_param_dict( DEBUG,train_start,train_end,val_start,
    val_end,test_start,test_end,n_jobs,GRAN,
    main_input_dir,infoIDs,platforms,user_statuses,LOGNORM_IGNORE_TAGS,LOGNORM_DEBUG_PRINT,
    RESCALE_LIST,SCALER_TYPE_LIST,FEATURE_RANGE_LIST,RESCALE_TARGET_LIST,INPUT_TWITTER_LOGNORM_LIST,OUTPUT_TWITTER_LOGNORM_LIST,
    INPUT_OTHER_LOGNORM_LIST,OUTPUT_OTHER_LOGNORM_LIST,TARGET_PLATFORM_LIST,INPUT_HOURS_LIST,OUTPUT_HOURS_LIST,GET_GDELT_FEATURES_LIST,
    GET_REDDIT_FEATURES_LIST,GET_1HOT_INFO_ID_FTS_LIST, GET_EXTERNAL_PLATFORM_FTS_LIST,main_output_dir):

    print("\nGetting sample gen params...")
    idx_to_sample_generator_param_dict = {}



    if DEBUG==True:
        INPUT_HOURS_LIST=INPUT_HOURS_LIST[:1]
        OUTPUT_HOURS_LIST=OUTPUT_HOURS_LIST[:1]
        GET_GDELT_FEATURES_LIST=GET_GDELT_FEATURES_LIST[:1]
        GET_REDDIT_FEATURES_LIST=GET_REDDIT_FEATURES_LIST[:1]
        GET_1HOT_INFO_ID_FTS_LIST=GET_1HOT_INFO_ID_FTS_LIST[:1]
        GET_EXTERNAL_PLATFORM_FTS_LIST=GET_EXTERNAL_PLATFORM_FTS_LIST[:1]
        TARGET_PLATFORM_LIST=TARGET_PLATFORM_LIST[:1]
        RESCALE_LIST=RESCALE_LIST[:1]
        SCALER_TYPE_LIST= SCALER_TYPE_LIST[:1]
        FEATURE_RANGE_LIST= FEATURE_RANGE_LIST[:1]
        RESCALE_TARGET_LIST= RESCALE_TARGET_LIST[:1]
        INPUT_TWITTER_LOGNORM_LIST =INPUT_TWITTER_LOGNORM_LIST[:1]
        OUTPUT_TWITTER_LOGNORM_LIST = OUTPUT_TWITTER_LOGNORM_LIST[:1]
        INPUT_OTHER_LOGNORM_LIST = INPUT_OTHER_LOGNORM_LIST[:1]
        OUTPUT_OTHER_LOGNORM_LIST = OUTPUT_OTHER_LOGNORM_LIST[:1]


    num_combos=0
    for INPUT_HOURS in INPUT_HOURS_LIST:
        for OUTPUT_HOURS in OUTPUT_HOURS_LIST:
            for GET_GDELT_FEATURES in GET_GDELT_FEATURES_LIST:
                for GET_REDDIT_FEATURES in GET_REDDIT_FEATURES_LIST:
                    for GET_1HOT_INFO_ID_FTS in GET_1HOT_INFO_ID_FTS_LIST:
                        for GET_EXTERNAL_PLATFORM_FTS in GET_EXTERNAL_PLATFORM_FTS_LIST:

                            for TARGET_PLATFORM in TARGET_PLATFORM_LIST:
                                if (GET_EXTERNAL_PLATFORM_FTS==False) and (TARGET_PLATFORM =="all"):
                                    break
                                else:
                                    # print("continuing loop...")
                                    for RESCALE in RESCALE_LIST:
                                        for SCALER_TYPE in SCALER_TYPE_LIST:
                                            for FEATURE_RANGE in FEATURE_RANGE_LIST:
                                                for RESCALE_TARGET in RESCALE_TARGET_LIST:
                                                    for INPUT_TWITTER_LOGNORM in INPUT_TWITTER_LOGNORM_LIST:
                                                        for OUTPUT_TWITTER_LOGNORM in OUTPUT_TWITTER_LOGNORM_LIST:
                                                            for INPUT_OTHER_LOGNORM in INPUT_OTHER_LOGNORM_LIST:
                                                                for OUTPUT_OTHER_LOGNORM in OUTPUT_OTHER_LOGNORM_LIST:
                                                                    cur_sample_gen_param_dict = {}
                                                                    # basic_sample_gen_params = {}
                                                                    # cur_sample_gen_param_dict["DEBUG"]=DEBUG
                                                                    cur_sample_gen_param_dict["train_start"]=train_start
                                                                    cur_sample_gen_param_dict["train_end"]=train_end
                                                                    cur_sample_gen_param_dict["val_start"]=val_start
                                                                    cur_sample_gen_param_dict["val_end"]=val_end
                                                                    cur_sample_gen_param_dict["test_start"]=test_start
                                                                    cur_sample_gen_param_dict["test_end"]=test_end
                                                                    # cur_sample_gen_param_dict["sum_fts"]=sum_fts
                                                                    # cur_sample_gen_param_dict["avg_fts"]=avg_fts
                                                                    cur_sample_gen_param_dict["n_jobs"]=n_jobs
                                                                    cur_sample_gen_param_dict["GRAN"]=GRAN
                                                                    cur_sample_gen_param_dict["main_input_dir"]=main_input_dir
                                                                    cur_sample_gen_param_dict["main_output_dir"]=main_output_dir
                                                                    cur_sample_gen_param_dict["infoIDs"]=infoIDs
                                                                    cur_sample_gen_param_dict["platforms"]=platforms
                                                                    cur_sample_gen_param_dict["user_statuses"]=user_statuses
                                                                    cur_sample_gen_param_dict["LOGNORM_IGNORE_TAGS"]=LOGNORM_IGNORE_TAGS
                                                                    cur_sample_gen_param_dict["LOGNORM_DEBUG_PRINT"]=LOGNORM_DEBUG_PRINT
                                                                    cur_sample_gen_param_dict["INPUT_HOURS"]=INPUT_HOURS
                                                                    cur_sample_gen_param_dict["OUTPUT_HOURS"]=OUTPUT_HOURS
                                                                    cur_sample_gen_param_dict["GET_GDELT_FEATURES"]=GET_GDELT_FEATURES
                                                                    cur_sample_gen_param_dict["GET_REDDIT_FEATURES"]=GET_REDDIT_FEATURES
                                                                    cur_sample_gen_param_dict["GET_1HOT_INFO_ID_FTS"]=GET_1HOT_INFO_ID_FTS
                                                                    cur_sample_gen_param_dict["GET_EXTERNAL_PLATFORM_FTS"]=GET_EXTERNAL_PLATFORM_FTS
                                                                    cur_sample_gen_param_dict["TARGET_PLATFORM"]=TARGET_PLATFORM
                                                                    cur_sample_gen_param_dict["RESCALE"]=RESCALE
                                                                    cur_sample_gen_param_dict["SCALER_TYPE"]=SCALER_TYPE
                                                                    cur_sample_gen_param_dict["FEATURE_RANGE"]=FEATURE_RANGE
                                                                    cur_sample_gen_param_dict["RESCALE_TARGET"]=RESCALE_TARGET
                                                                    cur_sample_gen_param_dict["INPUT_TWITTER_LOGNORM"]=INPUT_TWITTER_LOGNORM
                                                                    cur_sample_gen_param_dict["OUTPUT_TWITTER_LOGNORM"]=OUTPUT_TWITTER_LOGNORM
                                                                    cur_sample_gen_param_dict["INPUT_OTHER_LOGNORM"]=INPUT_OTHER_LOGNORM
                                                                    cur_sample_gen_param_dict["OUTPUT_OTHER_LOGNORM"]=OUTPUT_OTHER_LOGNORM
                                                                    idx_to_sample_generator_param_dict[num_combos]=dict(cur_sample_gen_param_dict)
                                                                    num_combos+=1
    print("\nnum_combos: %d"%(num_combos+1))


    return idx_to_sample_generator_param_dict, num_combos

def get_idx_to_sample_generator_param_dict_v2_sample_weights( DEBUG,train_start,train_end,val_start,
    val_end,test_start,test_end,n_jobs,GRAN,
    main_input_dir,infoIDs,platforms,user_statuses,LOGNORM_IGNORE_TAGS,LOGNORM_DEBUG_PRINT,
    RESCALE_LIST,SCALER_TYPE_LIST,FEATURE_RANGE_LIST,RESCALE_TARGET_LIST,INPUT_TWITTER_LOGNORM_LIST,OUTPUT_TWITTER_LOGNORM_LIST,
    INPUT_OTHER_LOGNORM_LIST,OUTPUT_OTHER_LOGNORM_LIST,TARGET_PLATFORM_LIST,INPUT_HOURS_LIST,OUTPUT_HOURS_LIST,GET_GDELT_FEATURES_LIST,
    GET_REDDIT_FEATURES_LIST,GET_1HOT_INFO_ID_FTS_LIST, GET_EXTERNAL_PLATFORM_FTS_LIST,main_output_dir, sample_weight_decay_list):

    print("\nGetting sample gen params...")
    idx_to_sample_generator_param_dict = {}



    if DEBUG==True:
        INPUT_HOURS_LIST=INPUT_HOURS_LIST[:1]
        OUTPUT_HOURS_LIST=OUTPUT_HOURS_LIST[:1]
        GET_GDELT_FEATURES_LIST=GET_GDELT_FEATURES_LIST[:1]
        GET_REDDIT_FEATURES_LIST=GET_REDDIT_FEATURES_LIST[:1]
        GET_1HOT_INFO_ID_FTS_LIST=GET_1HOT_INFO_ID_FTS_LIST[:1]
        GET_EXTERNAL_PLATFORM_FTS_LIST=GET_EXTERNAL_PLATFORM_FTS_LIST[:1]
        TARGET_PLATFORM_LIST=TARGET_PLATFORM_LIST[:1]
        RESCALE_LIST=RESCALE_LIST[:1]
        SCALER_TYPE_LIST= SCALER_TYPE_LIST[:1]
        FEATURE_RANGE_LIST= FEATURE_RANGE_LIST[:1]
        RESCALE_TARGET_LIST= RESCALE_TARGET_LIST[:1]
        INPUT_TWITTER_LOGNORM_LIST =INPUT_TWITTER_LOGNORM_LIST[:1]
        OUTPUT_TWITTER_LOGNORM_LIST = OUTPUT_TWITTER_LOGNORM_LIST[:1]
        INPUT_OTHER_LOGNORM_LIST = INPUT_OTHER_LOGNORM_LIST[:1]
        OUTPUT_OTHER_LOGNORM_LIST = OUTPUT_OTHER_LOGNORM_LIST[:1]


    num_combos=0
    for INPUT_HOURS in INPUT_HOURS_LIST:
        for OUTPUT_HOURS in OUTPUT_HOURS_LIST:
            for GET_GDELT_FEATURES in GET_GDELT_FEATURES_LIST:
                for GET_REDDIT_FEATURES in GET_REDDIT_FEATURES_LIST:
                    for GET_1HOT_INFO_ID_FTS in GET_1HOT_INFO_ID_FTS_LIST:
                        for GET_EXTERNAL_PLATFORM_FTS in GET_EXTERNAL_PLATFORM_FTS_LIST:

                            for TARGET_PLATFORM in TARGET_PLATFORM_LIST:
                                if (GET_EXTERNAL_PLATFORM_FTS==False) and (TARGET_PLATFORM =="all"):
                                    break
                                else:
                                    # print("continuing loop...")
                                    for RESCALE in RESCALE_LIST:
                                        for SCALER_TYPE in SCALER_TYPE_LIST:
                                            for FEATURE_RANGE in FEATURE_RANGE_LIST:
                                                for RESCALE_TARGET in RESCALE_TARGET_LIST:
                                                    for INPUT_TWITTER_LOGNORM in INPUT_TWITTER_LOGNORM_LIST:
                                                        for OUTPUT_TWITTER_LOGNORM in OUTPUT_TWITTER_LOGNORM_LIST:
                                                            for INPUT_OTHER_LOGNORM in INPUT_OTHER_LOGNORM_LIST:
                                                                for OUTPUT_OTHER_LOGNORM in OUTPUT_OTHER_LOGNORM_LIST:
                                                                    for sample_weight_decay in sample_weight_decay_list:
                                                                        cur_sample_gen_param_dict = {}

                                                                        cur_sample_gen_param_dict["sample_weight_decay"] = sample_weight_decay
                                                                        # basic_sample_gen_params = {}
                                                                        # cur_sample_gen_param_dict["DEBUG"]=DEBUG
                                                                        cur_sample_gen_param_dict["train_start"]=train_start
                                                                        cur_sample_gen_param_dict["train_end"]=train_end
                                                                        cur_sample_gen_param_dict["val_start"]=val_start
                                                                        cur_sample_gen_param_dict["val_end"]=val_end
                                                                        cur_sample_gen_param_dict["test_start"]=test_start
                                                                        cur_sample_gen_param_dict["test_end"]=test_end
                                                                        # cur_sample_gen_param_dict["sum_fts"]=sum_fts
                                                                        # cur_sample_gen_param_dict["avg_fts"]=avg_fts
                                                                        cur_sample_gen_param_dict["n_jobs"]=n_jobs
                                                                        cur_sample_gen_param_dict["GRAN"]=GRAN
                                                                        cur_sample_gen_param_dict["main_input_dir"]=main_input_dir
                                                                        cur_sample_gen_param_dict["main_output_dir"]=main_output_dir
                                                                        cur_sample_gen_param_dict["infoIDs"]=infoIDs
                                                                        cur_sample_gen_param_dict["platforms"]=platforms
                                                                        cur_sample_gen_param_dict["user_statuses"]=user_statuses
                                                                        cur_sample_gen_param_dict["LOGNORM_IGNORE_TAGS"]=LOGNORM_IGNORE_TAGS
                                                                        cur_sample_gen_param_dict["LOGNORM_DEBUG_PRINT"]=LOGNORM_DEBUG_PRINT
                                                                        cur_sample_gen_param_dict["INPUT_HOURS"]=INPUT_HOURS
                                                                        cur_sample_gen_param_dict["OUTPUT_HOURS"]=OUTPUT_HOURS
                                                                        cur_sample_gen_param_dict["GET_GDELT_FEATURES"]=GET_GDELT_FEATURES
                                                                        cur_sample_gen_param_dict["GET_REDDIT_FEATURES"]=GET_REDDIT_FEATURES
                                                                        cur_sample_gen_param_dict["GET_1HOT_INFO_ID_FTS"]=GET_1HOT_INFO_ID_FTS
                                                                        cur_sample_gen_param_dict["GET_EXTERNAL_PLATFORM_FTS"]=GET_EXTERNAL_PLATFORM_FTS
                                                                        cur_sample_gen_param_dict["TARGET_PLATFORM"]=TARGET_PLATFORM
                                                                        cur_sample_gen_param_dict["RESCALE"]=RESCALE
                                                                        cur_sample_gen_param_dict["SCALER_TYPE"]=SCALER_TYPE
                                                                        cur_sample_gen_param_dict["FEATURE_RANGE"]=FEATURE_RANGE
                                                                        cur_sample_gen_param_dict["RESCALE_TARGET"]=RESCALE_TARGET
                                                                        cur_sample_gen_param_dict["INPUT_TWITTER_LOGNORM"]=INPUT_TWITTER_LOGNORM
                                                                        cur_sample_gen_param_dict["OUTPUT_TWITTER_LOGNORM"]=OUTPUT_TWITTER_LOGNORM
                                                                        cur_sample_gen_param_dict["INPUT_OTHER_LOGNORM"]=INPUT_OTHER_LOGNORM
                                                                        cur_sample_gen_param_dict["OUTPUT_OTHER_LOGNORM"]=OUTPUT_OTHER_LOGNORM
                                                                        idx_to_sample_generator_param_dict[num_combos]=dict(cur_sample_gen_param_dict)
                                                                        num_combos+=1
    print("\nnum_combos: %d"%(num_combos+1))


    return idx_to_sample_generator_param_dict, num_combos


def insert_metrics_into_pred_df_v2_TEMPORAL(pred_df, infoIDs, platforms, output_dir,target_fts,metrics_to_get,metric_str_to_function_dict):
    #get model tag
    model_tag = pred_df["model_tag"].unique()[0]
    y_tag = pred_df["y_tag"].unique()[0]

    cur_output_dir = output_dir + model_tag + "/"
    create_output_dir(cur_output_dir)

    metric_to_result_list_dict = {}
    for metric in metrics_to_get:
        metric_to_result_list_dict[metric]=[]


    infoID_df_list = []
    target_df_list = []
    metric_df_list = []
    timestep_df_list = []
    pred_df_list = []
    actual_df_list = []

    platform = pred_df["platform"].unique()[0]


    for infoID in infoIDs:
        for target_ft in target_fts:




            temp_df = pred_df[(pred_df["infoID"]==infoID) & (pred_df["target_ft"]==target_ft)]
            print(temp_df)

            cur_timestep_list = [t+1 for t in range(temp_df.shape[0])]

            actuals = list(temp_df["actual"])
            preds = list(temp_df["pred"])

            for metric in metrics_to_get:
                metric_func = metric_str_to_function_dict[metric]

                for actual,pred in zip(actuals,preds):
                    result = metric_func(pd.Series([actual]), pd.Series([pred]))
                    metric_to_result_list_dict[metric].append(result)

            for i in range(temp_df.shape[0]):
                infoID_df_list.append(infoID)
                target_df_list.append(target_ft)

            timestep_df_list = timestep_df_list + list(cur_timestep_list)

            pred_df_list = pred_df_list + list(preds)
            actual_df_list = actual_df_list + list(actuals)

    df = pd.DataFrame(data={"infoID":infoID_df_list, "target_ft":target_df_list})
    for metric,metric_list in metric_to_result_list_dict.items():
        df[metric] = metric_list

    df["y_tag"]=y_tag
    df["model_tag"]=model_tag
    df["timestep"]=timestep_df_list
    df["pred"] = pred_df_list
    df["actual"] = actual_df_list
    df["platform"]=platform


    print(df)

    output_fp = cur_output_dir + y_tag + "-" + model_tag + "-metrics.csv"
    df.to_csv(output_fp, index=False)
    print(output_fp)

    # sys.exit(0)

    return df

def insert_metrics_into_pred_df(pred_df, infoIDs, platforms, output_dir,target_fts,metrics_to_get,metric_str_to_function_dict):
    #get model tag
    model_tag = pred_df["model_tag"].unique()[0]
    y_tag = pred_df["y_tag"].unique()[0]

    cur_output_dir = output_dir + model_tag + "/"
    create_output_dir(cur_output_dir)

    metric_to_result_list_dict = {}
    for metric in metrics_to_get:
        metric_to_result_list_dict[metric]=[]


    infoID_df_list = []
    target_df_list = []
    metric_df_list = []
    for infoID in infoIDs:
        for target_ft in target_fts:
            temp_df = pred_df[(pred_df["infoID"]==infoID) & (pred_df["target_ft"]==target_ft)]
            print(temp_df)

            for metric in metrics_to_get:
                metric_func = metric_str_to_function_dict[metric]
                result = metric_func(temp_df["actual"], temp_df["pred"])
                metric_to_result_list_dict[metric].append(result)

            infoID_df_list.append(infoID)
            target_df_list.append(target_ft)

    df = pd.DataFrame(data={"infoID":infoID_df_list, "target_ft":target_df_list})
    for metric,metric_list in metric_to_result_list_dict.items():
        df[metric] = metric_list

    df["y_tag"]=y_tag
    df["model_tag"]=model_tag

    print(df)

    output_fp = cur_output_dir + y_tag + "-" + model_tag + "-metrics.csv"
    df.to_csv(output_fp, index=False)
    print(output_fp)

    return df

def get_metric_df_from_pred_df(pred_df,metrics_to_get,metric_str_to_function_dict):

    metric_to_result_list_dict = {}
    for metric in metrics_to_get:
        metric_to_result_list_dict[metric]=[]

    infoID_df_list = []
    target_df_list = []
    metric_df_list = []
    y_tag = pred_df["y_tag"].unique()[0]
    model_tag = pred_df["model_tag"].unique()[0]

    target_fts = list(pred_df["target_ft"].unique())
    infoIDs = list(pred_df["infoID"].unique())
    for infoID in infoIDs:
        for target_ft in target_fts:
            temp_df = pred_df[(pred_df["infoID"]==infoID) & (pred_df["target_ft"]==target_ft)]
            print(temp_df)

            for metric in metrics_to_get:
                metric_func = metric_str_to_function_dict[metric]
                result = metric_func(temp_df["actual"], temp_df["pred"])
                metric_to_result_list_dict[metric].append(result)

            infoID_df_list.append(infoID)
            target_df_list.append(target_ft)

    df = pd.DataFrame(data={"infoID":infoID_df_list, "target_ft":target_df_list})
    for metric,metric_list in metric_to_result_list_dict.items():
        df[metric] = metric_list

    df["y_tag"] = y_tag
    df["model_tag"]=model_tag

    return df

def get_metric_df_from_pred_df_v2_simple(pred_df,metrics_to_get,metric_str_to_function_dict,model_type):

    metric_to_result_list_dict = {}
    for metric in metrics_to_get:
        metric_to_result_list_dict[metric]=[]

    infoID_df_list = []
    target_df_list = []
    metric_df_list = []
    y_tag = pred_df["y_tag"].unique()[0]
    # model_tag = pred_df["model_tag"].unique()[0]

    target_fts = list(pred_df["target_ft"].unique())
    infoIDs = list(pred_df["infoID"].unique())
    for infoID in infoIDs:
        for target_ft in target_fts:
            temp_df = pred_df[(pred_df["infoID"]==infoID) & (pred_df["target_ft"]==target_ft)]
            print(temp_df)

            for metric in metrics_to_get:
                metric_func = metric_str_to_function_dict[metric]
                result = metric_func(temp_df["actual"], temp_df["pred"])
                metric_to_result_list_dict[metric].append(result)

            infoID_df_list.append(infoID)
            target_df_list.append(target_ft)

    df = pd.DataFrame(data={"infoID":infoID_df_list, "target_ft":target_df_list})
    for metric,metric_list in metric_to_result_list_dict.items():
        df[metric] = metric_list

    df["y_tag"] = y_tag
    df["model_tag"]=model_type

    return df


def create_prediction_df(platforms,y_pred_dict, infoID_train_and_test_array_dict,infoIDs,target_fts,y_tag,model_tag, output_dir,SAVE_PREDS=True):

    if SAVE_PREDS == True:
        cur_output_dir = output_dir + model_tag + "/"
        create_output_dir(cur_output_dir)


    all_dfs = []


    for infoID in infoIDs:
        y_pred =  y_pred_dict[infoID]
        y_ground_truth = infoID_train_and_test_array_dict[infoID][y_tag]

        print("\ny_pred.shape")
        print(y_pred.shape)
        print("\ny_ground_truth.shape")
        print(y_ground_truth.shape)

        for target_ft in target_fts:

            pred_df = pd.DataFrame(data=y_pred, columns=target_fts)
            pred_df=pred_df[[target_ft]]

            ground_truth_df = pd.DataFrame(data=y_ground_truth, columns=target_fts)
            ground_truth_df=ground_truth_df[[target_ft]]

            pred_df = pred_df.rename(columns={target_ft:"pred"})
            ground_truth_df = ground_truth_df.rename(columns={target_ft:"actual"})
            df = pd.concat([pred_df, ground_truth_df], axis=1)
            print(df)

            for platform in platforms:
                if platform in target_ft:
                    df["platform"]=platform
                    break
            df["infoID"] = infoID
            # print(df)
            df["timestep"] = [(i+1) for i in range(df.shape[0])]
            df["target_ft"] = target_ft
            col_order = ["timestep", "pred", "actual", "platform", "infoID","target_ft"]
            df = df[col_order]
            print(df)
            all_dfs.append(df)
    df = pd.concat(all_dfs)
    df["y_tag"]=y_tag
    df["model_tag"]=model_tag
    print(df)

    if SAVE_PREDS == True:
        output_fp = cur_output_dir + y_tag + "-" + model_tag + "-predictions.csv"
        df.to_csv(output_fp, index=False)
        print(output_fp)


    return df

def create_prediction_df_with_corr_data(CORRELATION_FUNCS_TO_USE_LIST,array_type_to_correlation_materials_dict,platforms,y_pred_dict, infoID_train_and_test_array_dict,infoIDs,target_fts,array_tag,model_tag, output_dir,SAVE_PREDS=True):

    if SAVE_PREDS == True:
        cur_output_dir = output_dir + model_tag + "/"
        create_output_dir(cur_output_dir)

    y_tag = "y_" + array_tag
    x_tag = "x_" + array_tag


    all_dfs = []

    try:
        cur_array_type_correlation_materials_dict = array_type_to_correlation_materials_dict[array_tag]
        infoID_to_overall_timestep_to_corr_func_to_target_ft_dict=cur_array_type_correlation_materials_dict["infoID_to_overall_timestep_to_corr_func_to_target_ft_dict"]
        num_overall_timesteps = cur_array_type_correlation_materials_dict["num_overall_timesteps"]
    except:
        print("no corr data, continuing")



    for infoID in infoIDs:


        y_pred =  y_pred_dict[infoID]
        y_ground_truth = infoID_train_and_test_array_dict[infoID][y_tag]

        print("\ny_pred.shape")
        print(y_pred.shape)
        print("\ny_ground_truth.shape")
        print(y_ground_truth.shape)

        for target_ft in target_fts:

            pred_df = pd.DataFrame(data=y_pred, columns=target_fts)
            pred_df=pred_df[[target_ft]]

            ground_truth_df = pd.DataFrame(data=y_ground_truth, columns=target_fts)
            ground_truth_df=ground_truth_df[[target_ft]]

            pred_df = pred_df.rename(columns={target_ft:"pred"})
            ground_truth_df = ground_truth_df.rename(columns={target_ft:"actual"})
            df = pd.concat([pred_df, ground_truth_df], axis=1)
            print(df)

            for platform in platforms:
                if platform in target_ft:
                    df["platform"]=platform
                    break
            df["infoID"] = infoID
            # print(df)
            df["timestep"] = [(i+1) for i in range(df.shape[0])]
            df["target_ft"] = target_ft
            col_order = ["timestep", "pred", "actual", "platform", "infoID","target_ft"]

            for corr_func_str in CORRELATION_FUNCS_TO_USE_LIST:
                col_order.append(corr_func_str)
                corr_vals = []
                for overall_timestep in range(num_overall_timesteps):
                    avg_corr = infoID_to_overall_timestep_to_corr_func_to_target_ft_dict[infoID][overall_timestep][corr_func_str][target_ft]
                    corr_vals.append(avg_corr)
                df[corr_func_str] = corr_vals

            df = df[col_order]
            print(df)
            all_dfs.append(df)
    df = pd.concat(all_dfs)
    df["y_tag"]=y_tag
    df["model_tag"]=model_tag
    print(df)

    if SAVE_PREDS == True:
        output_fp = cur_output_dir + y_tag + "-" + model_tag + "-predictions.csv"
        df.to_csv(output_fp, index=False)
        print(output_fp)


    return df

def graph_pred_vs_gt(y_pred_dict, infoID_train_and_test_array_dict, y_tag, infoIDs,target_fts,dates,output_dir):

    output_dir = output_dir+y_tag+ "/"
    create_output_dir(output_dir)

    print("\nGraphing %s"%y_tag)
    # y_ground_truth_dict = infoID_train_and_test_array_dict[y_tag]

    hyp_dict = hyphenate_infoID_dict(infoIDs)

    for infoID in infoIDs:

        hyp_infoID = hyp_dict[infoID]
        # print()
        # print(infoID)

        # cur_output_dir = output_dir + hyp_infoID + "/"
        # create_output_dir(cur_output_dir)

        print("\nCurrent infoID: %s"%infoID)
        y_pred = y_pred_dict[infoID]
        y_ground_truth = infoID_train_and_test_array_dict[infoID][y_tag]

        print("\ny_pred.shape")
        print(y_pred.shape)
        print("\ny_ground_truth.shape")
        print(y_ground_truth.shape)

        y_pred = y_pred.T
        y_ground_truth = y_ground_truth.T

        print("\ntransposed y_pred.shape")
        print(y_pred.shape)
        print("\ntransposed y_ground_truth.shape")
        print(y_ground_truth.shape)

        num_rows = 3
        num_cols = 2

        fig, axs = plt.subplots(3, 2,figsize=(8,8))

        coord_list = [(0,0), (0,1), (1,0), (1,1), (2,0), (2,1)]

        target_ft_to_axis_coordinates_dict = {}

        # num_coords = len(coord_list)
        idx=0
        for target_ft,coord in zip(target_fts,coord_list ):
            target_ft_to_axis_coordinates_dict[target_ft] = coord
            print("%s: %s"%(target_ft, str(coord)))

            # cur_df = pd.DataFrame(data={"Prediction":y_pred, "Ground_Truth":y_ground_truth})

            cur_y_pred = y_pred[idx]
            cur_y_ground_truth = y_ground_truth[idx]
            x_coor = coord[0]
            y_coor = coord[1]
            axs[x_coor,y_coor].plot(cur_y_pred,":r" ,label="Prediction")
            axs[x_coor,y_coor].plot(cur_y_ground_truth, "-k" ,label="Ground Truth")

            target_no_underscore = target_ft.replace("_"," ")
            title_tag = "%s %s"%(infoID, target_no_underscore)
            title = ("\n".join(wrap(title_tag, 20)))
            # title = '%s \n%s'%(infoID, target_ft)
            axs[x_coor,y_coor].set_title(title)
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))
            idx+=1

        for ax in axs.flat:
            ax.set(xlabel='days', ylabel='Volume')

        plt.tight_layout()
        output_fp = output_dir + "%s-%s.png"%( hyp_infoID,target_ft)
        fig.savefig(output_fp)
        plt.close()
        print(output_fp)

def graph_pred_vs_gt_options(y_pred_dict, infoID_train_and_test_array_dict, y_tag, infoIDs,target_fts,dates,output_dir):

    output_dir = output_dir+y_tag+ "/"
    create_output_dir(output_dir)

    print("\nGraphing %s"%y_tag)
    # y_ground_truth_dict = infoID_train_and_test_array_dict[y_tag]

    hyp_dict = hyphenate_infoID_dict(infoIDs)

    for infoID in infoIDs:

        hyp_infoID = hyp_dict[infoID]
        # print()
        # print(infoID)

        # cur_output_dir = output_dir + hyp_infoID + "/"
        # create_output_dir(cur_output_dir)

        print("\nCurrent infoID: %s"%infoID)
        y_pred = y_pred_dict[infoID]
        y_ground_truth = infoID_train_and_test_array_dict[infoID][y_tag]

        print("\ny_pred.shape")
        print(y_pred.shape)
        print("\ny_ground_truth.shape")
        print(y_ground_truth.shape)

        y_pred = y_pred.T
        y_ground_truth = y_ground_truth.T

        print("\ntransposed y_pred.shape")
        print(y_pred.shape)
        print("\ntransposed y_ground_truth.shape")
        print(y_ground_truth.shape)

        # num_rows = 3
        # num_cols = 2

        # num_rows = len(target_fts)/

        num_target_fts = len(target_fts)
        if num_target_fts == 3:
            coord_list = [0, 1, 2]
            fig, axs = plt.subplots(3,figsize=(8,8))
        else:
            coord_list = [(0,0), (0,1), (1,0), (1,1), (2,0), (2,1)]
            fig, axs = plt.subplots(3, 2,figsize=(8,8))

        target_ft_to_axis_coordinates_dict = {}

        # num_coords = len(coord_list)
        idx=0
        for target_ft,coord in zip(target_fts,coord_list ):
            target_ft_to_axis_coordinates_dict[target_ft] = coord
            print("%s: %s"%(target_ft, str(coord)))

            # cur_df = pd.DataFrame(data={"Prediction":y_pred, "Ground_Truth":y_ground_truth})

            cur_y_pred = y_pred[idx]
            cur_y_ground_truth = y_ground_truth[idx]

            target_no_underscore = target_ft.replace("_"," ")
            title_tag = "%s %s"%(infoID, target_no_underscore)
            title = ("\n".join(wrap(title_tag, 20)))
            # title = '%s \n%s'%(infoID, target_ft)

            if num_target_fts == 6:
                x_coor = coord[0]
                y_coor = coord[1]
                print("\ncur_y_pred shape")
                print(cur_y_pred.shape)
                print("\ncur_y_ground_truth shape")
                print(cur_y_ground_truth.shape)
                axs[x_coor,y_coor].plot(cur_y_pred,":r" ,label="Prediction")
                axs[x_coor,y_coor].plot(cur_y_ground_truth, "-k" ,label="Ground Truth")
                axs[x_coor,y_coor].set_title(title)
            else:
                # x_coor = coord[0]
                # y_coor = coord[1]
                print("\ncur_y_pred shape")
                print(cur_y_pred.shape)
                print("\ncur_y_ground_truth shape")
                print(cur_y_ground_truth.shape)
                axs[coord].plot(cur_y_pred,":r" ,label="Prediction")
                axs[coord].plot(cur_y_ground_truth, "-k" ,label="Ground Truth")
                axs[coord].set_title(title)

            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))
            idx+=1

        for ax in axs.flat:
            ax.set(xlabel='days', ylabel='Volume')

        plt.tight_layout()
        output_fp = output_dir + "%s.png"%( hyp_infoID)
        fig.savefig(output_fp)
        plt.close()
        print(output_fp)

def make_comparison_df(model_pred_metric_df, baseline_pred_metric_df,metrics_to_get ,baseline_comp_output_dir,merge_cols,y_tag):
    baseline_comp_output_dir = baseline_comp_output_dir + y_tag + "/"
    create_output_dir(baseline_comp_output_dir)

    print(model_pred_metric_df)
    # sys.exit(0)

    model_tag = model_pred_metric_df["model_tag"].iloc[0]
    baseline_tag = baseline_pred_metric_df["model_tag"].iloc[0]

    model_pred_metric_df = model_pred_metric_df.drop(["model_tag"], axis=1)
    baseline_pred_metric_df = baseline_pred_metric_df.drop(["model_tag"], axis=1)

    print(model_pred_metric_df )
    print(baseline_pred_metric_df)
    # sys.exit(0)

    for metric in metrics_to_get:
        model_pred_metric_df = model_pred_metric_df.rename(columns={metric:model_tag + "_" + metric})
        baseline_pred_metric_df = baseline_pred_metric_df.rename(columns={metric:baseline_tag + "_" + metric})

    metric_df_list = []
    num_wins_df_list = []
    num_ties_df_list = []
    num_trials_df_list = []
    win_freq_df_list = []
    num_trials_no_ties_list = []

    for metric in metrics_to_get:
        print()
        print(metric)

        cur_comp_df = pd.merge(model_pred_metric_df, baseline_pred_metric_df,on=merge_cols, how="inner")
        print(cur_comp_df)
        cur_comp_df = cur_comp_df[merge_cols+ [model_tag + "_" + metric, baseline_tag + "_" + metric]]
        print(cur_comp_df)

        model_errors = list(cur_comp_df[model_tag + "_" + metric])
        baseline_errors = list(cur_comp_df[baseline_tag + "_" + metric])
        cur_comp_df["is_winner"]=[1 if model_error < baseline_error else 0 if baseline_error<model_error else "tie" for model_error, baseline_error in zip(model_errors, baseline_errors)]
        print(cur_comp_df)
        # sys.exit(0)

        num_trials =cur_comp_df.shape[0]
        num_wins = cur_comp_df[cur_comp_df["is_winner"]==1].shape[0]
        num_ties = cur_comp_df[cur_comp_df["is_winner"]=="tie"].shape[0]
        win_freq = num_wins/(1.0 * num_trials)
        num_trials_no_ties = num_trials - num_ties

        metric_df_list.append(metric)
        num_wins_df_list.append(num_wins)
        num_ties_df_list.append(num_ties)
        num_trials_df_list.append(num_trials)
        win_freq_df_list.append(win_freq)
        num_trials_no_ties_list.append(num_trials_no_ties)


        cur_metric_output_fp = baseline_comp_output_dir + "%s-%s-model-baseline-comparisons.csv"%(y_tag ,metric )
        cur_comp_df.to_csv(cur_metric_output_fp, index=False)
        print(cur_metric_output_fp)

    data={
        "metric":metric_df_list,
        "num_trials":num_trials_df_list,
        "num_ties":num_ties_df_list,
        "num_trials":num_trials_df_list,
        "num_wins_freq":win_freq_df_list,
        "num_trials_no_ties":num_trials_no_ties_list,
        "num_wins":num_wins_df_list
        }

    col_order = ["metric","num_wins" ,"num_trials", "num_ties", "num_trials","num_trials_no_ties", "num_wins_freq", ]

    metric_comp_df = pd.DataFrame(data=data)
    metric_comp_df=metric_comp_df[col_order]
    print(metric_comp_df)

    output_fp = baseline_comp_output_dir + "%s-model-baseline-comp-summary.csv"%y_tag
    metric_comp_df.to_csv(output_fp,index=False)
    print(output_fp)

    return metric_comp_df

def get_idx_to_sample_generator_param_dict_v2_corr_fts( DEBUG,train_start,train_end,val_start,
    val_end,test_start,test_end,n_jobs,GRAN,
    main_input_dir,infoIDs,platforms,user_statuses,LOGNORM_IGNORE_TAGS,LOGNORM_DEBUG_PRINT,
    RESCALE_LIST,SCALER_TYPE_LIST,FEATURE_RANGE_LIST,RESCALE_TARGET_LIST,INPUT_TWITTER_LOGNORM_LIST,OUTPUT_TWITTER_LOGNORM_LIST,
    INPUT_OTHER_LOGNORM_LIST,OUTPUT_OTHER_LOGNORM_LIST,TARGET_PLATFORM_LIST,INPUT_HOURS_LIST,OUTPUT_HOURS_LIST,GET_GDELT_FEATURES_LIST,
    GET_REDDIT_FEATURES_LIST,GET_1HOT_INFO_ID_FTS_LIST, GET_EXTERNAL_PLATFORM_FTS_LIST,main_output_dir,CORRELATION_FUNCS_TO_USE_LIST_LIST,CORRELATION_STR_TO_FUNC_DICT):

    print("\nGetting sample gen params...")
    idx_to_sample_generator_param_dict = {}



    if DEBUG==True:
        INPUT_HOURS_LIST=INPUT_HOURS_LIST[:1]
        OUTPUT_HOURS_LIST=OUTPUT_HOURS_LIST[:1]
        GET_GDELT_FEATURES_LIST=GET_GDELT_FEATURES_LIST[:1]
        GET_REDDIT_FEATURES_LIST=GET_REDDIT_FEATURES_LIST[:1]
        GET_1HOT_INFO_ID_FTS_LIST=GET_1HOT_INFO_ID_FTS_LIST[:1]
        GET_EXTERNAL_PLATFORM_FTS_LIST=GET_EXTERNAL_PLATFORM_FTS_LIST[:1]
        TARGET_PLATFORM_LIST=TARGET_PLATFORM_LIST[:1]
        RESCALE_LIST=RESCALE_LIST[:1]
        SCALER_TYPE_LIST= SCALER_TYPE_LIST[:1]
        FEATURE_RANGE_LIST= FEATURE_RANGE_LIST[:1]
        RESCALE_TARGET_LIST= RESCALE_TARGET_LIST[:1]
        INPUT_TWITTER_LOGNORM_LIST =INPUT_TWITTER_LOGNORM_LIST[:1]
        OUTPUT_TWITTER_LOGNORM_LIST = OUTPUT_TWITTER_LOGNORM_LIST[:1]
        INPUT_OTHER_LOGNORM_LIST = INPUT_OTHER_LOGNORM_LIST[:1]
        OUTPUT_OTHER_LOGNORM_LIST = OUTPUT_OTHER_LOGNORM_LIST[:1]
        CORRELATION_FUNCS_TO_USE_LIST_LIST=CORRELATION_FUNCS_TO_USE_LIST_LIST[:1]


    num_combos=0
    for CORRELATION_FUNCS_TO_USE_LIST in CORRELATION_FUNCS_TO_USE_LIST_LIST:
        for INPUT_HOURS in INPUT_HOURS_LIST:
            for OUTPUT_HOURS in OUTPUT_HOURS_LIST:
                for GET_GDELT_FEATURES in GET_REDDIT_FEATURES_LIST:
                    for GET_REDDIT_FEATURES in GET_REDDIT_FEATURES_LIST:
                        for GET_1HOT_INFO_ID_FTS in GET_1HOT_INFO_ID_FTS_LIST:
                            for GET_EXTERNAL_PLATFORM_FTS in GET_EXTERNAL_PLATFORM_FTS_LIST:

                                for TARGET_PLATFORM in TARGET_PLATFORM_LIST:
                                    if (GET_EXTERNAL_PLATFORM_FTS==False) and (TARGET_PLATFORM =="all"):
                                        break
                                    else:
                                        # print("continuing loop...")
                                        for RESCALE in RESCALE_LIST:
                                            for SCALER_TYPE in SCALER_TYPE_LIST:
                                                for FEATURE_RANGE in FEATURE_RANGE_LIST:
                                                    for RESCALE_TARGET in RESCALE_TARGET_LIST:
                                                        for INPUT_TWITTER_LOGNORM in INPUT_TWITTER_LOGNORM_LIST:
                                                            for OUTPUT_TWITTER_LOGNORM in OUTPUT_TWITTER_LOGNORM_LIST:
                                                                for INPUT_OTHER_LOGNORM in INPUT_OTHER_LOGNORM_LIST:
                                                                    for OUTPUT_OTHER_LOGNORM in OUTPUT_OTHER_LOGNORM_LIST:
                                                                        cur_sample_gen_param_dict = {}
                                                                        # basic_sample_gen_params = {}
                                                                        # cur_sample_gen_param_dict["DEBUG"]=DEBUG
                                                                        cur_sample_gen_param_dict["train_start"]=train_start
                                                                        cur_sample_gen_param_dict["train_end"]=train_end
                                                                        cur_sample_gen_param_dict["val_start"]=val_start
                                                                        cur_sample_gen_param_dict["val_end"]=val_end
                                                                        cur_sample_gen_param_dict["test_start"]=test_start
                                                                        cur_sample_gen_param_dict["test_end"]=test_end
                                                                        # cur_sample_gen_param_dict["sum_fts"]=sum_fts
                                                                        # cur_sample_gen_param_dict["avg_fts"]=avg_fts
                                                                        cur_sample_gen_param_dict["n_jobs"]=n_jobs
                                                                        cur_sample_gen_param_dict["GRAN"]=GRAN
                                                                        cur_sample_gen_param_dict["main_input_dir"]=main_input_dir
                                                                        cur_sample_gen_param_dict["main_output_dir"]=main_output_dir
                                                                        cur_sample_gen_param_dict["infoIDs"]=infoIDs
                                                                        cur_sample_gen_param_dict["platforms"]=platforms
                                                                        cur_sample_gen_param_dict["user_statuses"]=user_statuses
                                                                        cur_sample_gen_param_dict["LOGNORM_IGNORE_TAGS"]=LOGNORM_IGNORE_TAGS
                                                                        cur_sample_gen_param_dict["LOGNORM_DEBUG_PRINT"]=LOGNORM_DEBUG_PRINT
                                                                        cur_sample_gen_param_dict["INPUT_HOURS"]=INPUT_HOURS
                                                                        cur_sample_gen_param_dict["OUTPUT_HOURS"]=OUTPUT_HOURS
                                                                        cur_sample_gen_param_dict["GET_GDELT_FEATURES"]=GET_GDELT_FEATURES
                                                                        cur_sample_gen_param_dict["GET_REDDIT_FEATURES"]=GET_REDDIT_FEATURES
                                                                        cur_sample_gen_param_dict["GET_1HOT_INFO_ID_FTS"]=GET_1HOT_INFO_ID_FTS
                                                                        cur_sample_gen_param_dict["GET_EXTERNAL_PLATFORM_FTS"]=GET_EXTERNAL_PLATFORM_FTS
                                                                        cur_sample_gen_param_dict["TARGET_PLATFORM"]=TARGET_PLATFORM
                                                                        cur_sample_gen_param_dict["RESCALE"]=RESCALE
                                                                        cur_sample_gen_param_dict["SCALER_TYPE"]=SCALER_TYPE
                                                                        cur_sample_gen_param_dict["FEATURE_RANGE"]=FEATURE_RANGE
                                                                        cur_sample_gen_param_dict["RESCALE_TARGET"]=RESCALE_TARGET
                                                                        cur_sample_gen_param_dict["INPUT_TWITTER_LOGNORM"]=INPUT_TWITTER_LOGNORM
                                                                        cur_sample_gen_param_dict["OUTPUT_TWITTER_LOGNORM"]=OUTPUT_TWITTER_LOGNORM
                                                                        cur_sample_gen_param_dict["INPUT_OTHER_LOGNORM"]=INPUT_OTHER_LOGNORM
                                                                        cur_sample_gen_param_dict["OUTPUT_OTHER_LOGNORM"]=OUTPUT_OTHER_LOGNORM
                                                                        cur_sample_gen_param_dict["CORRELATION_FUNCS_TO_USE_LIST"]=CORRELATION_FUNCS_TO_USE_LIST
                                                                        cur_sample_gen_param_dict["CORRELATION_STR_TO_FUNC_DICT"]=CORRELATION_STR_TO_FUNC_DICT
                                                                        idx_to_sample_generator_param_dict[num_combos]=dict(cur_sample_gen_param_dict)
                                                                        num_combos+=1
    print("\nnum_combos: %d"%(num_combos+1))


    return idx_to_sample_generator_param_dict, num_combos




def get_all_result_subdirs(main_input_dirs):

    all_subdirs = []

    for main_input_dir in main_input_dirs:

        level1_subdirs = os.listdir(main_input_dir)
        for l1_dir in level1_subdirs:
            cur_subdir = main_input_dir + l1_dir + "/"
            # print(cur_subdir)
            # if  "Combo-Counts" not in l1_dir:
            l2_subdirs = os.listdir(cur_subdir)

            # print("\nl2_subdirs")
            # print(l2_subdirs)

            for l2_dir in l2_subdirs:
                if "Combo-Counts" not in l2_dir:
                    cur_subdir = main_input_dir + l1_dir + "/" + l2_dir + "/"
                    # print(cur_subdir)

                    l3_dirs = os.listdir(cur_subdir)

                    for l3_dir in l3_dirs:
                        cur_subdir = main_input_dir + l1_dir + "/" + l2_dir + "/" + l3_dir + "/"
                        print(cur_subdir)
                        all_subdirs.append(cur_subdir)

    return all_subdirs

def get_candidate_dirs_per_tag(all_subdirs, platform, PLATFORM_TO_MODEL_SET_DICT,main_output_dir,SAVE_MODEL_DIRS,param_fp="full-params.csv"):

    # #make platform output dir
    # platform_output_dir = main_output_dir + platform + "/"
    # create_output_dir(platform_output_dir)

    #get platform dict
    MODEL_NAME_TO_PARAMS_DICT =PLATFORM_TO_MODEL_SET_DICT[platform]

    #make candidate dict
    model_tag_to_candidate_subdir_dict = {}
    model_tags = list(MODEL_NAME_TO_PARAMS_DICT.keys())
    print("\nmodel_tags")
    print(model_tags)

    for model_tag in model_tags:
        model_tag_to_candidate_subdir_dict[model_tag] = []

    print("\nChecking if params match...")
    num_subdirs = len(all_subdirs)
    for i,subdir in enumerate(all_subdirs):

        try:
            cur_param_df = pd.read_csv(subdir + param_fp )
            print("Got param df %d of %d"%(i+1, num_subdirs))
            # print(cur_param_df)
        except FileNotFoundError:
            print("%s does not exist. Moving on..."%(subdir + param_fp))

        param_dict = convert_df_2_cols_to_dict(cur_param_df, "param", "value")

        for model_tag in model_tags:

            cur_model_required_params_dict = MODEL_NAME_TO_PARAMS_DICT[model_tag]

            subdir_is_match=True
            for req_param, req_val in cur_model_required_params_dict.items():
                for check_param, check_val in param_dict.items():

                    if check_val=="False":
                        check_val=False
                    if check_val=="True":
                        check_val=True

                    if (req_param==check_param):
                        if (check_val==req_val):
                            print(req_param)
                            print(check_param)
                            print(req_val)
                            print(check_val)
                            print("%s matches with %s"%(model_tag, subdir))
                        else:
                            subdir_is_match=False
                            break
                if subdir_is_match==False:
                    break

            if subdir_is_match == True:
                model_tag_to_candidate_subdir_dict[model_tag].append(subdir)


    print("Done checking")

    for model_tag,subdir_cand_list in model_tag_to_candidate_subdir_dict.items():
        print("%s has %d candidates"%(model_tag, len(subdir_cand_list)))

    return model_tag_to_candidate_subdir_dict

def get_candidate_dirs_per_tag_v2_with_required_params(REQUIRED_PARAM_DICT,all_subdirs, platform, PLATFORM_TO_MODEL_SET_DICT,main_output_dir,SAVE_MODEL_DIRS,param_fp="full-params.csv"):

    # #make platform output dir
    # platform_output_dir = main_output_dir + platform + "/"
    # create_output_dir(platform_output_dir)

    #get platform dict
    MODEL_NAME_TO_PARAMS_DICT =PLATFORM_TO_MODEL_SET_DICT[platform]

    #make candidate dict
    model_tag_to_candidate_subdir_dict = {}
    model_tags = list(MODEL_NAME_TO_PARAMS_DICT.keys())
    print("\nmodel_tags")
    print(model_tags)

    for model_tag in model_tags:
        model_tag_to_candidate_subdir_dict[model_tag] = []

    print("\nChecking if params match...")
    num_subdirs = len(all_subdirs)
    for i,subdir in enumerate(all_subdirs):

        try:
            cur_param_df = pd.read_csv(subdir + param_fp )
            print("Got param df %d of %d"%(i+1, num_subdirs))
            # print(cur_param_df)
        except FileNotFoundError:
            print("%s does not exist. Moving on..."%(subdir + param_fp))

        param_dict = convert_df_2_cols_to_dict(cur_param_df, "param", "value")

        for model_tag in model_tags:

            cur_model_required_params_dict = MODEL_NAME_TO_PARAMS_DICT[model_tag]
            cur_model_required_params_dict.update(dict(REQUIRED_PARAM_DICT))
            # print("\ncur_model_required_params_dict")
            # print(cur_model_required_params_dict)

            # sys.exit(0)

            subdir_is_match=True
            for req_param, req_val in cur_model_required_params_dict.items():
                for check_param, check_val in param_dict.items():

                    if check_val=="False":
                        check_val=False
                    if check_val=="True":
                        check_val=True

                    if (req_param==check_param):
                        if (check_val==req_val):
                            print(req_param)
                            print(check_param)
                            print(req_val)
                            print(check_val)
                            print("%s matches with %s"%(model_tag, subdir))
                        else:
                            subdir_is_match=False
                            break
                if subdir_is_match==False:
                    break

            if subdir_is_match == True:
                model_tag_to_candidate_subdir_dict[model_tag].append(subdir)


    print("Done checking")

    for model_tag,subdir_cand_list in model_tag_to_candidate_subdir_dict.items():
        print("%s has %d candidates"%(model_tag, len(subdir_cand_list)))

    return model_tag_to_candidate_subdir_dict

def get_all_param_fps(all_subdirs, param_fp="full-params.csv"):

    all_param_fps = []
    for subdir in all_subdirs:
        cur_param_fp = subdir + param_fp
        all_param_fps.append(cur_param_fp)
        print(cur_param_fp)

    return all_param_fps


def get_model_tag_to_candidate_results_dict(model_tag_to_candidate_subdirs_dict, eval_tag,metric_tag,model_type,baseline_tag):

    model_tags = list(model_tag_to_candidate_subdirs_dict.keys())

    model_tag_to_candidate_results_dict = {}

    for model_tag in model_tags:

        cur_model_cand_subdirs = model_tag_to_candidate_subdirs_dict[model_tag]

        #cur_model_avg_error_list = []

        all_model_tag_subdir_dfs = []
        for subdir in cur_model_cand_subdirs:

            baseline_comp_fp = subdir + "Baseline-Comparisons/y_" + eval_tag +"/y_%s-%s-model-baseline-comparisons.csv"%(eval_tag, metric_tag)
            comp_df = pd.read_csv(baseline_comp_fp)

            num_trials = comp_df.shape[0]
            num_trials_no_ties = comp_df[comp_df["is_winner"]!="tie"].shape[0]
            num_wins = comp_df[comp_df["is_winner"]==1].shape[0]
            win_freq = num_wins/(num_trials_no_ties * 1.0)

            #get tags
            model_error_metric_col = model_type + "_" + metric_tag
            baseline_error_metric_col = baseline_tag + "_" + metric_tag

            model_median_error = comp_df[model_error_metric_col].median()
            model_mean_error = comp_df[model_error_metric_col].mean()

            baseline_median_error = comp_df[baseline_error_metric_col].median()
            baseline_mean_error = comp_df[baseline_error_metric_col].mean()

            data={
                "%s_median_%s"%(baseline_tag, metric_tag) : [baseline_median_error],
                "%s_median_%s"%(model_type, metric_tag) : [model_median_error],
                "%s_mean_%s"%(baseline_tag, metric_tag) :[ baseline_mean_error],
                "%s_mean_%s"%(model_type, metric_tag) : [model_mean_error],
                }

            col_order = ["%s_median_%s"%(model_type, metric_tag),"%s_median_%s"%(baseline_tag, metric_tag),"%s_mean_%s"%(model_type, metric_tag),"%s_mean_%s"%(baseline_tag, metric_tag)]


            cur_comp_df = pd.DataFrame(data=data)[col_order]
            cur_comp_df["score_against_baseline"]=np.round(win_freq,2 )
            all_model_tag_subdir_dfs.append(cur_comp_df)

        try:
            cur_model_df = pd.concat(all_model_tag_subdir_dfs).reset_index(drop=True)
            cur_model_df = cur_model_df.sort_values("%s_mean_%s"%(model_type, metric_tag)).reset_index(drop=True)
            cur_model_df["model_dir"]=cur_model_cand_subdirs
            print("\n%s"%model_tag)
            print(cur_model_df)
            model_tag_to_candidate_results_dict[model_tag]=cur_model_df
        except ValueError:
            model_tag_to_candidate_results_dict[model_tag]=None


    return model_tag_to_candidate_results_dict

def get_best_model_results_v2_with_extra_metrics(summary_stat_tag,model_tag_to_candidate_results_dict ,metric_tag,model_type,baseline_tag,platform,SAVE_MODEL_DIRS,eval_tag,metrics_to_get):

    other_metrics_to_get = []
    for m in metrics_to_get:
        if m != metric_tag:
            other_metrics_to_get.append(m)
    print("\nother_metrics_to_get")
    print(other_metrics_to_get)

    model_tags = list(model_tag_to_candidate_results_dict.keys())

    model_error_col = "%s_%s_%s"%(model_type, summary_stat_tag,metric_tag)
    baseline_error_col = "%s_%s_%s"%(baseline_tag, summary_stat_tag,metric_tag)

    keep_cols = [model_error_col, baseline_error_col, "score_against_baseline", "model_dir"]

    all_dfs = []
    for model_tag, result_df in model_tag_to_candidate_results_dict.items():
        try:
            print(result_df.shape[0])
        except AttributeError:
            continue
        result_df = result_df.sort_values(model_error_col).reset_index(drop=True)
        print("\nresult_df")
        print(result_df)
        # sys.exit(0)


        result_cols  = list(result_df)
        for col in result_cols:
            if col not in keep_cols:
                result_df = result_df.drop(col, axis=1)

        result_df = result_df.head(1)
        result_df["model_tag"]=model_tag
        all_dfs.append(result_df)

    result_df = pd.concat(all_dfs).reset_index(drop=True)
    result_df = result_df.sort_values(model_error_col).reset_index(drop=True)
    print(result_df)


    return result_df


def get_best_model_results(summary_stat_tag,model_tag_to_candidate_results_dict ,metric_tag,model_type,baseline_tag,platform,SAVE_MODEL_DIRS,eval_tag):

    model_tags = list(model_tag_to_candidate_results_dict.keys())

    model_error_col = "%s_%s_%s"%(model_type, summary_stat_tag,metric_tag)
    baseline_error_col = "%s_%s_%s"%(baseline_tag, summary_stat_tag,metric_tag)

    keep_cols = [model_error_col, baseline_error_col, "score_against_baseline", "model_dir"]

    all_dfs = []
    for model_tag, result_df in model_tag_to_candidate_results_dict.items():
        try:
            print(result_df.shape[0])
        except AttributeError:
            continue
        result_df = result_df.sort_values(model_error_col).reset_index(drop=True)
        print("\nresult_df")
        print(result_df)
        result_cols  = list(result_df)
        for col in result_cols:
            if col not in keep_cols:
                result_df = result_df.drop(col, axis=1)

        result_df = result_df.head(1)
        result_df["model_tag"]=model_tag
        all_dfs.append(result_df)

    result_df = pd.concat(all_dfs).reset_index(drop=True)
    result_df = result_df.sort_values(model_error_col).reset_index(drop=True)
    print(result_df)

    return result_df

# def save_model_dirs_in_b
def save_model_dirs(model_df,output_dir):

    model_tag_to_dir_dict = {}

    model_tags = list(model_df["model_tag"])
    model_dirs = list(model_df["model_dir"])

    num_model_dirs = len(model_dirs)
    for i,model_dir in enumerate(model_dirs):
        print("\nSaving model %d of %d"%(i+1, num_model_dirs))

        model_tag = model_tags[i]

        cur_output_dir = output_dir + model_tag + "/"
        create_output_dir(cur_output_dir)

        src = model_dir
        dest = cur_output_dir
        copytree(src, dest, symlinks = False, ignore = None)
        print("Got %s"%dest)

        model_tag_to_dir_dict[model_tag]=cur_output_dir
    return model_tag_to_dir_dict

def create_mean_model_ensemble(platform, best_result_input_dir,MEAN_MODEL_ENSEMBLE_TAG_TO_MODEL_TAGS_TO_GET_DICT ,model_type, ensemble_eval_tags, combined_df_drop_cols):



    print("\nbest_result_input_dir: ")
    print(best_result_input_dir)

    mean_models_output_dir = best_result_input_dir + "Mean-Models/"
    create_output_dir(mean_models_output_dir)

    MEAN_MODEL_ENSEMBLE_TAG_TO_RESULTS_DICT = {}

    metrics_to_get = ["rmse", "nc_rmse", "ape"]

    for MEAN_MODEL_ENSEMBLE_TAG, MODEL_TAGS_TO_GET_LIST in MEAN_MODEL_ENSEMBLE_TAG_TO_MODEL_TAGS_TO_GET_DICT.items():
        MEAN_MODEL_ENSEMBLE_TAG_TO_RESULTS_DICT[MEAN_MODEL_ENSEMBLE_TAG] = {}
        for ensemble_eval_tag in ensemble_eval_tags:

            print()
            print(MEAN_MODEL_ENSEMBLE_TAG)
            print(MODEL_TAGS_TO_GET_LIST)

            cur_df_list_for_means = []

            for cur_model_tag_for_mean in MODEL_TAGS_TO_GET_LIST:
                # cur_eval_result_fp = best_result_input_dir + cur_model_tag_for_mean + "/Metrics/" + model_type + "/" + "y_" + ensemble_eval_tag + "-" + model_type + "-metrics.csv"
                cur_pred_result_fp = best_result_input_dir + cur_model_tag_for_mean + "/"+ model_type + "/y_" + ensemble_eval_tag + "-" + model_type + "-predictions.csv"
                cur_pred_result_df = pd.read_csv(cur_pred_result_fp)
                print()
                print(cur_pred_result_df)
                cur_df_list_for_means.append(cur_pred_result_df)

                # sys.exit(0)
            cur_stacked_df = pd.concat(cur_df_list_for_means).reset_index(drop=True)

            for col in combined_df_drop_cols:
                if col in list(cur_stacked_df):
                    cur_stacked_df = cur_stacked_df.drop(col,axis=1)
            print("\ncur_stacked_df")
            print(cur_stacked_df)

            groupby_cols = ["infoID", "target_ft", "y_tag", "model_tag", "timestep"]
            cur_stacked_df["pred"] = cur_stacked_df.groupby(groupby_cols)["pred"].transform("mean")

            print("\ncur_stacked_df before drop dupes")
            print(cur_stacked_df)

            cur_stacked_df = cur_stacked_df.drop_duplicates().reset_index(drop=True)
            print("\ncur_stacked_df after drop dupes")
            print(cur_stacked_df)

            MEAN_MODEL_ENSEMBLE_TAG_TO_RESULTS_DICT[MEAN_MODEL_ENSEMBLE_TAG][ensemble_eval_tag] = cur_stacked_df

            cur_output_dir = mean_models_output_dir + MEAN_MODEL_ENSEMBLE_TAG + "/" + model_type + "/"
            create_output_dir(cur_output_dir)

            output_fp = cur_output_dir + "y_%s-%s-predictions.csv"%(ensemble_eval_tag, model_type)
            cur_stacked_df.to_csv(output_fp, index=False)
            print("Saved %s"%output_fp)

            #get metrics
            cur_output_dir = mean_models_output_dir + MEAN_MODEL_ENSEMBLE_TAG + "/Metrics/"
            create_output_dir(cur_output_dir)



    return MEAN_MODEL_ENSEMBLE_TAG_TO_RESULTS_DICT,mean_models_output_dir

def get_persistence_metric_model(platform, MEAN_MODEL_ENSEMBLE_TAG_TO_MODEL_TAGS_TO_GET_DICT,best_result_input_dir,ensemble_eval_tags):

    eval_tag_to_persistence_model_metric_df_dict = {}

    #just choose a model tag
    for model_ensemble_tag,model_tags in MEAN_MODEL_ENSEMBLE_TAG_TO_MODEL_TAGS_TO_GET_DICT.items():
        model_tag=model_tags[0]
        print("\nmodel_tag")
        print(model_tag)
        break

    for ensemble_eval_tag in ensemble_eval_tags:

        print()
        print(ensemble_eval_tag)

        cur_persistence_model_metrics_fp = best_result_input_dir + model_tag + "/Metrics/Persistence_Baseline/y_%s-Persistence_Baseline-metrics.csv"%ensemble_eval_tag
        cur_persistence_model_metrics_df = pd.read_csv(cur_persistence_model_metrics_fp)
        print("\ncur_persistence_model_metrics_df")
        print(cur_persistence_model_metrics_df)

        eval_tag_to_persistence_model_metric_df_dict[ensemble_eval_tag] = cur_persistence_model_metrics_df

    return eval_tag_to_persistence_model_metric_df_dict

def get_persistence_metric_model_v2_simple(platform, model_tag,input_dir,ensemble_eval_tags):

    eval_tag_to_persistence_model_metric_df_dict = {}


    # model_tag=model_tags[0]
    # print("\nmodel_tag")
    # print(model_tag)


    for ensemble_eval_tag in ensemble_eval_tags:

        print()
        print(ensemble_eval_tag)

        cur_persistence_model_metrics_fp = input_dir + "/Metrics/Persistence_Baseline/y_%s-Persistence_Baseline-metrics.csv"%ensemble_eval_tag
        cur_persistence_model_metrics_df = pd.read_csv(cur_persistence_model_metrics_fp)
        print("\ncur_persistence_model_metrics_df")
        print(cur_persistence_model_metrics_df)

        eval_tag_to_persistence_model_metric_df_dict[ensemble_eval_tag] = cur_persistence_model_metrics_df

    return eval_tag_to_persistence_model_metric_df_dict

def get_model_tag_to_subdir_dict_from_best_result_df(best_result_df):

    model_tag_to_subdir_dict = {}
    model_tags = list(best_result_df["model_tag"])
    model_dirs = list(best_result_df["model_dir"])

    for model_tag,model_dir in zip(model_tags, model_dirs):
        model_tag_to_subdir_dict[model_tag]=model_dir
    return model_tag_to_subdir_dict

def get_mean_ensemble_model(metric_tag, output_dir, PLATFORM_TO_MEAN_MODEL_ENSEMBLE_TAG_TO_MODEL_TAGS_TO_GET_DICT,platform,model_type,ensemble_eval_tags, combined_df_drop_cols,metrics_to_get,
    summary_stat_tag,baseline_tag,val_best_result_df,test_best_result_df, main_output_dir):
    val_best_result_input_dir = output_dir
    MEAN_MODEL_ENSEMBLE_TAG_TO_MODEL_TAGS_TO_GET_DICT = PLATFORM_TO_MEAN_MODEL_ENSEMBLE_TAG_TO_MODEL_TAGS_TO_GET_DICT[platform]
    MEAN_MODEL_ENSEMBLE_TAG_TO_RESULTS_DICT,mean_models_output_dir = create_mean_model_ensemble(platform, val_best_result_input_dir,MEAN_MODEL_ENSEMBLE_TAG_TO_MODEL_TAGS_TO_GET_DICT, model_type,ensemble_eval_tags, combined_df_drop_cols)

    # MEAN_MODEL_ENSEMBLE_TAG_TO_MODEL_TAGS_TO_GET_DICT = PLATFORM_TO_MEAN_MODEL_ENSEMBLE_TAG_TO_MODEL_TAGS_TO_GET_DICT[platform]
    eval_tag_to_persistence_model_metric_df_dict = get_persistence_metric_model(platform, MEAN_MODEL_ENSEMBLE_TAG_TO_MODEL_TAGS_TO_GET_DICT,val_best_result_input_dir,ensemble_eval_tags)

    #get dict for mean metrics
    mean_model_tag_to_eval_tag_to_metric_df_dict = {}

    for MEAN_MODEL_ENSEMBLE_TAG, MEAN_MODEL_ENSEMBLE_TAG_DICT in MEAN_MODEL_ENSEMBLE_TAG_TO_RESULTS_DICT.items():
        mean_model_tag_to_eval_tag_to_metric_df_dict[MEAN_MODEL_ENSEMBLE_TAG] = {}
        for ensemble_eval_tag, pred_df in MEAN_MODEL_ENSEMBLE_TAG_DICT.items():

            print()
            print(MEAN_MODEL_ENSEMBLE_TAG)
            print(ensemble_eval_tag)
            print(pred_df)

            # groupby_cols = ["infoID", "model_tag", "platform", "target_ft", "y_tag"]

            metric_df = get_metric_df_from_pred_df(pred_df,metrics_to_get,metric_str_to_function_dict)
            print("\nmetric_df")
            print(metric_df)

            cur_metric_output_dir = mean_models_output_dir + MEAN_MODEL_ENSEMBLE_TAG + "/Metrics/" + model_type + "/"
            create_output_dir(cur_metric_output_dir)

            output_fp = cur_metric_output_dir + "y_%s-%s-metrics.csv"%(ensemble_eval_tag, model_type)
            metric_df.to_csv(output_fp, index=False)
            print(output_fp)
            mean_model_tag_to_eval_tag_to_metric_df_dict[MEAN_MODEL_ENSEMBLE_TAG][ensemble_eval_tag] = metric_df


    #get baseline comparison files
    print("\nGetting model metrics...")
    mean_model_tag_to_eval_tag_to_comp_df = {}
    for MEAN_MODEL_ENSEMBLE_TAG,MEAN_MODEL_ENSEMBLE_TAG_DICT in mean_model_tag_to_eval_tag_to_metric_df_dict.items():
        mean_model_tag_to_eval_tag_to_comp_df[MEAN_MODEL_ENSEMBLE_TAG] = {}
        for ensemble_eval_tag, model_metric_df in MEAN_MODEL_ENSEMBLE_TAG_DICT.items():
            print()
            print(MEAN_MODEL_ENSEMBLE_TAG)
            print(ensemble_eval_tag)
            print(model_metric_df)

            y_eval_tag = "y_%s"%ensemble_eval_tag
            merge_cols = ["infoID", "target_ft","y_tag"]
            baseline_comp_output_dir = mean_models_output_dir + MEAN_MODEL_ENSEMBLE_TAG +"/Baseline-Comparisons/"
            create_output_dir(baseline_comp_output_dir)

            baseline_metric_df = eval_tag_to_persistence_model_metric_df_dict[ensemble_eval_tag]
            print("\nbaseline_metric_df")
            print(baseline_metric_df)

            # sys.exit(0)
            cur_comp_df = make_comparison_df(model_metric_df, baseline_metric_df,metrics_to_get ,baseline_comp_output_dir,merge_cols,y_eval_tag)
            mean_model_tag_to_eval_tag_to_comp_df[MEAN_MODEL_ENSEMBLE_TAG][y_eval_tag] = cur_comp_df

    #get mean model model_tag_to_candidate_subdirs_dict
    mean_model_tag_to_candidate_subdirs_dict = {}
    MEAN_MODEL_ENSEMBLE_TAGS = list(MEAN_MODEL_ENSEMBLE_TAG_TO_MODEL_TAGS_TO_GET_DICT.keys())
    print("\nMEAN_MODEL_ENSEMBLE_TAGS")
    print(MEAN_MODEL_ENSEMBLE_TAGS)

    for MEAN_MODEL_ENSEMBLE_TAG in MEAN_MODEL_ENSEMBLE_TAGS:
        cur_mm_subdir = mean_models_output_dir + MEAN_MODEL_ENSEMBLE_TAG + "/"
        mean_model_tag_to_candidate_subdirs_dict[MEAN_MODEL_ENSEMBLE_TAG] = [cur_mm_subdir]


    val_mean_model_tag_to_candidate_results_dict = get_model_tag_to_candidate_results_dict(mean_model_tag_to_candidate_subdirs_dict, "val",metric_tag,model_type,baseline_tag)
    mean_model_val_best_result_df = get_best_model_results(summary_stat_tag,val_mean_model_tag_to_candidate_results_dict ,metric_tag,model_type,baseline_tag,platform,False,"val")
    print("\nMean model mean_model_val_best_result_df")
    print(mean_model_val_best_result_df)

    test_mean_model_tag_to_candidate_results_dict = get_model_tag_to_candidate_results_dict(mean_model_tag_to_candidate_subdirs_dict, "test",metric_tag,model_type,baseline_tag)
    mean_model_test_best_result_df = get_best_model_results(summary_stat_tag,test_mean_model_tag_to_candidate_results_dict ,metric_tag,model_type,baseline_tag,platform,False,"test")
    print("\nMean model mean_model_test_best_result_df")
    print(mean_model_test_best_result_df)

    mm_output_fp = mean_models_output_dir + "%s-mean-ens-model-val-best-results-%s.csv"%(platform ,summary_stat_tag)
    mean_model_val_best_result_df.to_csv(mm_output_fp, index=False)
    print(mm_output_fp)

    mm_output_fp = mean_models_output_dir + "%s-mean-ens-model-test-best-results-%s.csv"%(platform ,summary_stat_tag)
    mean_model_test_best_result_df.to_csv(mm_output_fp, index=False)
    print(mm_output_fp)

    #combine with other
    full_test_best_result_df = pd.concat([mean_model_test_best_result_df, test_best_result_df]).reset_index(drop=True)
    full_test_best_result_df = full_test_best_result_df.sort_values("%s_%s_%s"%(model_type,summary_stat_tag ,metric_tag)).reset_index(drop=True)
    cur_output_dir = main_output_dir + platform + "/" + summary_stat_tag + "/"
    full_test_best_result_fp = cur_output_dir + "%s-full-test-best-results-%s.csv"%(platform ,summary_stat_tag)
    full_test_best_result_df.to_csv(full_test_best_result_fp, index=False)
    print(full_test_best_result_df)
    print(full_test_best_result_fp)

    full_val_best_result_df = pd.concat([mean_model_val_best_result_df, val_best_result_df]).reset_index(drop=True)
    full_val_best_result_df = full_val_best_result_df.sort_values("%s_%s_%s"%(model_type,summary_stat_tag ,metric_tag)).reset_index(drop=True)
    full_val_best_result_fp = cur_output_dir + "%s-full-val-best-results-%s.csv"%(platform ,summary_stat_tag)
    full_val_best_result_df.to_csv(full_val_best_result_fp, index=False)
    print(full_val_best_result_df)
    print(full_val_best_result_fp)

    return full_val_best_result_df, full_test_best_result_df, full_test_best_result_fp, full_val_best_result_fp, mean_model_tag_to_candidate_subdirs_dict


def bool_str_to_bool(bool_str):

    if bool_str == "True":
        return True
    if bool_str == "False":
        return False
    print("Error! String is not bool!")
    sys.exit(0)


#get params
def get_model_data_proc_params_from_param_df(param_df):

    if "Unnamed: 0" in list(param_df):
        param_df = param_df.drop("Unnamed: 0",axis=1)

    input_param_dict = convert_df_2_cols_to_dict(param_df, "param", "value")
    print("\ninput_param_dict")
    print(input_param_dict)
    output_param_dict = {}

    try:
        output_param_dict["INPUT_HOURS"] = int(input_param_dict["INPUT_HOURS"])
    except KeyError:
        output_param_dict["INPUT_HOURS"] = int(input_param_dict[" INPUT_HOURS"])


    output_param_dict["train_start"] = input_param_dict["train_start"]
    output_param_dict["train_end"] = input_param_dict["train_end"]

    output_param_dict["val_start"] = input_param_dict["val_start"]
    output_param_dict["val_end"] = input_param_dict["val_end"]

    output_param_dict["test_start"] = input_param_dict["test_start"]
    output_param_dict["test_end"] = input_param_dict["test_end"]

    output_param_dict["GRAN"] = input_param_dict["GRAN"]
    output_param_dict["OUTPUT_HOURS"] = int(input_param_dict["OUTPUT_HOURS"])
    output_param_dict["GET_1HOT_INFO_ID_FTS"] =  bool_str_to_bool(input_param_dict["GET_1HOT_INFO_ID_FTS"])
    output_param_dict["GET_GDELT_FEATURES"] =  bool_str_to_bool(input_param_dict["GET_GDELT_FEATURES"])
    output_param_dict["GET_REDDIT_FEATURES"] =  bool_str_to_bool(input_param_dict["GET_REDDIT_FEATURES"])
    output_param_dict["TARGET_PLATFORM"] = input_param_dict["TARGET_PLATFORM"]
    output_param_dict["GET_EXTERNAL_PLATFORM_FTS"] = bool_str_to_bool(input_param_dict["GET_EXTERNAL_PLATFORM_FTS"])
    output_param_dict["RESCALE"] = bool_str_to_bool(input_param_dict["RESCALE"])
    output_param_dict["RESCALE_TARGET"] = bool_str_to_bool(input_param_dict["RESCALE_TARGET"])
    output_param_dict["SCALER_TYPE"] = input_param_dict["SCALER_TYPE"]
    output_param_dict["FEATURE_RANGE"] = literal_eval(input_param_dict["FEATURE_RANGE"])
    output_param_dict["INPUT_TWITTER_LOGNORM"] = int(input_param_dict["INPUT_TWITTER_LOGNORM"])
    output_param_dict["OUTPUT_TWITTER_LOGNORM"] = int(input_param_dict["OUTPUT_TWITTER_LOGNORM"])
    output_param_dict["INPUT_OTHER_LOGNORM"] = int(input_param_dict["INPUT_OTHER_LOGNORM"])
    output_param_dict["OUTPUT_OTHER_LOGNORM"] = int(input_param_dict["OUTPUT_OTHER_LOGNORM"])
    output_param_dict["LOGNORM_IGNORE_TAGS"] = literal_eval(input_param_dict["LOGNORM_IGNORE_TAGS"])
    output_param_dict["CORRELATION_FUNCS_TO_USE_LIST"] = literal_eval(input_param_dict["CORRELATION_FUNCS_TO_USE_LIST"])
    output_param_dict["n_jobs"] = int(input_param_dict["n_jobs"])

    return output_param_dict

def create_gt_df_with_corr_data(CORRELATION_FUNCS_TO_USE_LIST,array_type_to_correlation_materials_dict,platforms,infoID_train_and_test_array_dict,infoIDs,target_fts,array_tag,model_tag, output_dir,SAVE_FILE=True):

    if SAVE_FILE == True:
        cur_output_dir = str(output_dir)
        create_output_dir(cur_output_dir)

    y_tag = "y_" + array_tag
    x_tag = "x_" + array_tag


    all_dfs = []

    try:
        cur_array_type_correlation_materials_dict = array_type_to_correlation_materials_dict[array_tag]
        infoID_to_overall_timestep_to_corr_func_to_target_ft_dict=cur_array_type_correlation_materials_dict["infoID_to_overall_timestep_to_corr_func_to_target_ft_dict"]
        num_overall_timesteps = cur_array_type_correlation_materials_dict["num_overall_timesteps"]
    except:
        print("no corr data, continuing")



    for infoID in infoIDs:

        y_ground_truth = infoID_train_and_test_array_dict[infoID][y_tag]

        print("\ny_ground_truth.shape")
        print(y_ground_truth.shape)

        for target_ft in target_fts:

            ground_truth_df = pd.DataFrame(data=y_ground_truth, columns=target_fts)
            ground_truth_df=ground_truth_df[[target_ft]]
            ground_truth_df = ground_truth_df.rename(columns={target_ft:"actual"})
            # df = pd.concat([pred_df, ground_truth_df], axis=1)
            # print(df)
            df = ground_truth_df.copy()
            print(df)

            for platform in platforms:
                if platform in target_ft:
                    df["platform"]=platform
                    break
            df["infoID"] = infoID
            # print(df)
            df["timestep"] = [(i+1) for i in range(df.shape[0])]
            df["target_ft"] = target_ft
            col_order = ["timestep", "actual", "platform", "infoID","target_ft"]

            for corr_func_str in CORRELATION_FUNCS_TO_USE_LIST:
                col_order.append(corr_func_str)
                corr_vals = []
                for overall_timestep in range(num_overall_timesteps):
                    avg_corr = infoID_to_overall_timestep_to_corr_func_to_target_ft_dict[infoID][overall_timestep][corr_func_str][target_ft]
                    corr_vals.append(avg_corr)
                df[corr_func_str] = corr_vals

            df = df[col_order]
            print(df)
            all_dfs.append(df)
    df = pd.concat(all_dfs)
    df["y_tag"]=y_tag
    df["model_tag"]=model_tag
    # df[""]
    print(df)

    if SAVE_FILE == True:
        output_fp = cur_output_dir + y_tag + "-" + model_tag + "-ground-truth-with-corr-data.csv"
        df.to_csv(output_fp, index=False)
        print(output_fp)


    return df

def create_gt_df_with_corr_data_v2_diff_filename(CORRELATION_FUNCS_TO_USE_LIST,array_type_to_correlation_materials_dict,platforms,infoID_train_and_test_array_dict,infoIDs,target_fts,array_tag,model_tag, output_dir, SAVE_FILE=True):
    y_tag = "y_" + array_tag
    x_tag = "x_" + array_tag

    corr_type = CORRELATION_FUNCS_TO_USE_LIST[0]

    if SAVE_FILE == True:
        cur_output_dir = str(output_dir) + y_tag + "-%s-corr-data/"%corr_type
        create_output_dir(cur_output_dir)




    all_dfs = []

    try:
        cur_array_type_correlation_materials_dict = array_type_to_correlation_materials_dict[array_tag]
        infoID_to_overall_timestep_to_corr_func_to_target_ft_dict=cur_array_type_correlation_materials_dict["infoID_to_overall_timestep_to_corr_func_to_target_ft_dict"]
        num_overall_timesteps = cur_array_type_correlation_materials_dict["num_overall_timesteps"]
    except:
        print("no corr data, continuing")



    for infoID in infoIDs:

        y_ground_truth = infoID_train_and_test_array_dict[infoID][y_tag]

        print("\ny_ground_truth.shape")
        print(y_ground_truth.shape)

        for target_ft in target_fts:

            ground_truth_df = pd.DataFrame(data=y_ground_truth, columns=target_fts)
            ground_truth_df=ground_truth_df[[target_ft]]
            ground_truth_df = ground_truth_df.rename(columns={target_ft:"actual"})
            # df = pd.concat([pred_df, ground_truth_df], axis=1)
            # print(df)
            df = ground_truth_df.copy()
            print(df)

            for platform in platforms:
                if platform in target_ft:
                    df["platform"]=platform
                    break
            df["infoID"] = infoID
            # print(df)
            df["timestep"] = [(i+1) for i in range(df.shape[0])]
            df["target_ft"] = target_ft
            col_order = ["timestep", "actual", "platform", "infoID","target_ft"]

            for corr_func_str in CORRELATION_FUNCS_TO_USE_LIST:
                col_order.append(corr_func_str)
                corr_vals = []
                for overall_timestep in range(num_overall_timesteps):
                    avg_corr = infoID_to_overall_timestep_to_corr_func_to_target_ft_dict[infoID][overall_timestep][corr_func_str][target_ft]
                    corr_vals.append(avg_corr)
                df[corr_func_str] = corr_vals

            df = df[col_order]
            print(df)
            all_dfs.append(df)
    df = pd.concat(all_dfs)
    df["y_tag"]=y_tag
    df["model_tag"]=model_tag
    # df[""]
    print(df)

    if SAVE_FILE == True:
        output_fp = cur_output_dir + model_tag + "-ground-truth-with-corr-data.csv"
        df.to_csv(output_fp, index=False)
        print(output_fp)


    return df

def get_corr_exp_pred_df(model_tags, model_tag_to_corr_dfs_dict,model_tag_to_subdir_dict, eval_tag, model_type,ignore_tags):

    all_pred_dfs = []

    for model_tag, model_subdir in model_tag_to_subdir_dict.items():

        # if model_tag in ignore_tags:
        #   continue
        print()
        print("%s: %s"%(model_tag, model_subdir))

        cur_pred_fp = model_subdir + model_type + "/y_" + eval_tag + "-" + model_type + "-predictions.csv"
        print(cur_pred_fp)
        cur_pred_df = pd.read_csv(cur_pred_fp)
        cur_pred_df["model_tag"] = model_tag
        print("\ncur_pred_df")
        print(cur_pred_df)


        all_pred_dfs.append(cur_pred_df)

    pred_df = pd.concat(all_pred_dfs).reset_index(drop=True)
    print("\npred_df")
    print(pred_df)

    return pred_df

def get_corr_exp_pred_df_v2_simple(model_tags, model_tag_to_subdir_dict, eval_tag, model_type):

    all_pred_dfs = []

    for model_tag, model_subdir in model_tag_to_subdir_dict.items():

        # if model_tag in ignore_tags:
        #   continue
        print()
        print("%s: %s"%(model_tag, model_subdir))

        cur_pred_fp = model_subdir + model_type + "/y_" + eval_tag + "-" + model_type + "-predictions.csv"
        print(cur_pred_fp)
        cur_pred_df = pd.read_csv(cur_pred_fp)
        cur_pred_df["model_tag"] = model_tag
        print("\ncur_pred_df")
        print(cur_pred_df)


        all_pred_dfs.append(cur_pred_df)

    pred_df = pd.concat(all_pred_dfs).reset_index(drop=True)
    print("\npred_df")
    print(pred_df)

    return pred_df

def get_corr_ensemble_threshold_table(thresholds_to_try ,main_regular_model_pred_df, main_pred_and_corr_df, threshold_output_dir,
    drop_dupe_cols,corr_type,eval_tag,metrics_to_get,model_type,metric_tag,summary_stat_tag,baseline_tag,platform,eval_tag_to_persistence_model_metric_df_dict):

    print("\nthresholds_to_try")
    print(thresholds_to_try)

    nunique_timesteps = main_pred_and_corr_df["timestep"].nunique()
    num_infoIDs = main_pred_and_corr_df["infoID"].nunique()
    num_target_fts = main_pred_and_corr_df["target_ft"].nunique()

    num_preds_there_should_be = nunique_timesteps*num_infoIDs*num_target_fts
    print("\nnum_preds_there_should_be")
    print(num_preds_there_should_be)

    corr_model_tag_to_candidate_subdirs_dict = {}


    for cur_corr_threshold in thresholds_to_try:

        corr_model_tag = "VAM-Corr-Ensemble-Model-Threshold-%s"%str(cur_corr_threshold)
        print("\ncorr_model_tag")
        print(corr_model_tag)

        regular_model_pred_df = main_regular_model_pred_df.copy()
        pred_and_corr_df = main_pred_and_corr_df.copy()
        print("\ncur_corr_threshold")
        print(cur_corr_threshold)

        threshold_pred_and_corr_df = pred_and_corr_df[pred_and_corr_df[corr_type]>=cur_corr_threshold]
        print("\nthreshold_pred_and_corr_df filtered by threshold: %.3f"%cur_corr_threshold)
        print(threshold_pred_and_corr_df)

        threshold_pred_and_corr_df = threshold_pred_and_corr_df.sort_values(corr_type, ascending=False).reset_index(drop=True)
        print("\nthreshold_pred_and_corr_df after sort")
        print(threshold_pred_and_corr_df)

        threshold_pred_and_corr_df = threshold_pred_and_corr_df.drop_duplicates(drop_dupe_cols).reset_index(drop=True)
        print("\nthreshold_pred_and_corr_df")
        print(threshold_pred_and_corr_df)

        threshold_pred_and_corr_df["sampleID"]=threshold_pred_and_corr_df["timestep"].astype("str") + "-"+ threshold_pred_and_corr_df["infoID"] + "-"+threshold_pred_and_corr_df["target_ft"]
        regular_model_pred_df["sampleID"]=regular_model_pred_df["timestep"].astype("str") + "-"+ regular_model_pred_df["infoID"] + "-"+regular_model_pred_df["target_ft"]
        threshold_pred_and_corr_df_sampleIDs = threshold_pred_and_corr_df["sampleID"]

        print("\nthreshold_pred_and_corr_df_sampleIDs")
        print(threshold_pred_and_corr_df_sampleIDs)
        print("\nthreshold_pred_and_corr_df")
        print(threshold_pred_and_corr_df)


        regular_model_pred_df = regular_model_pred_df[~regular_model_pred_df["sampleID"].isin(threshold_pred_and_corr_df_sampleIDs)].reset_index(drop=True)
        print("\nregular_model_pred_df")
        print(regular_model_pred_df)

        cur_corr_pred_ensemble_model_df = pd.concat([regular_model_pred_df,threshold_pred_and_corr_df]).reset_index(drop=True)
        cur_corr_pred_ensemble_model_df[corr_type] = cur_corr_pred_ensemble_model_df[corr_type].fillna("no_corr_score")
        print("\ncur_corr_pred_ensemble_model_df")
        print(cur_corr_pred_ensemble_model_df)

        cur_threshold_output_dir = threshold_output_dir + corr_model_tag + "/"
        create_output_dir(cur_threshold_output_dir)
        corr_model_tag_to_candidate_subdirs_dict[corr_model_tag]=[cur_threshold_output_dir]

        model_type_output_dir = cur_threshold_output_dir + model_type + "/"
        create_output_dir(model_type_output_dir)

        output_fp = model_type_output_dir + "y_%s-%s-predictions.csv"%(eval_tag,model_type)
        cur_corr_pred_ensemble_model_df.to_csv(output_fp, index=False)
        print(output_fp)

        val_count_df = cur_corr_pred_ensemble_model_df[["model_tag"]]
        val_count_df["model_tag_frequency"] = val_count_df.groupby(["model_tag"])["model_tag"].transform("count")
        val_count_df = val_count_df.drop_duplicates().reset_index(drop=True)

        # value_counts = cur_corr_pred_ensemble_model_df["model_tag"].value_counts()

        # val_count_df = value_counts.rename_axis('unique_values').reset_index(name='counts')
        print("\nval_count_df")
        print(val_count_df)

        val_count_fp = model_type_output_dir + "%s-%s-model-tag-val-counts.csv"%(eval_tag,str(cur_corr_threshold))
        val_count_df.to_csv(val_count_fp, index=False)
        print(val_count_fp)

        metric_df = get_metric_df_from_pred_df_v2_simple(cur_corr_pred_ensemble_model_df,metrics_to_get,metric_str_to_function_dict, model_type)
        print("\nmetric_df")
        print(metric_df)

        metric_output_dir = cur_threshold_output_dir + "Metrics/" + model_type + "/"
        create_output_dir(metric_output_dir)
        metric_fp = metric_output_dir + "y_%s-%s-metrics.csv"%(eval_tag,model_type)
        metric_df.to_csv(metric_fp, index=False)
        print(metric_fp)

        y_eval_tag = "y_%s"%eval_tag
        merge_cols = ["infoID", "target_ft","y_tag"]
        baseline_comp_output_dir = cur_threshold_output_dir +"Baseline-Comparisons/"
        create_output_dir(baseline_comp_output_dir)

        baseline_metric_df = eval_tag_to_persistence_model_metric_df_dict[eval_tag]
        print("\nbaseline_metric_df")
        print(baseline_metric_df)

        # sys.exit(0)
        cur_comp_df = make_comparison_df(metric_df, baseline_metric_df,metrics_to_get ,baseline_comp_output_dir,merge_cols,y_eval_tag)
        print("\ncur_comp_df")
        print(cur_comp_df)

    corr_model_tag_to_candidate_results_dict = get_model_tag_to_candidate_results_dict(corr_model_tag_to_candidate_subdirs_dict, eval_tag,metric_tag,model_type,baseline_tag)
    corr_model_result_df = get_best_model_results(summary_stat_tag,corr_model_tag_to_candidate_results_dict ,metric_tag,model_type,baseline_tag,platform,False,eval_tag)
    print("\ncorr_model_result_df")
    print(corr_model_result_df)

    output_fp = threshold_output_dir+ "%s-corr-ens-model-%s-best-results-%s.csv"%(platform, eval_tag,summary_stat_tag)


    return corr_model_tag_to_candidate_results_dict,corr_model_tag_to_candidate_subdirs_dict,corr_model_result_df







def make_comparison_df_no_save(model_pred_metric_df, baseline_pred_metric_df,metrics_to_get ,merge_cols,y_tag):
    # baseline_comp_output_dir = baseline_comp_output_dir + y_tag + "/"
    # create_output_dir(baseline_comp_output_dir)

    print(model_pred_metric_df)
    # sys.exit(0)

    model_tag = model_pred_metric_df["model_tag"].iloc[0]
    baseline_tag = baseline_pred_metric_df["model_tag"].iloc[0]

    model_pred_metric_df = model_pred_metric_df.drop(["model_tag"], axis=1)
    baseline_pred_metric_df = baseline_pred_metric_df.drop(["model_tag"], axis=1)

    print(model_pred_metric_df )
    print(baseline_pred_metric_df)
    # sys.exit(0)

    for metric in metrics_to_get:
        model_pred_metric_df = model_pred_metric_df.rename(columns={metric:model_tag + "_" + metric})
        baseline_pred_metric_df = baseline_pred_metric_df.rename(columns={metric:baseline_tag + "_" + metric})

    metric_df_list = []
    num_wins_df_list = []
    num_ties_df_list = []
    num_trials_df_list = []
    win_freq_df_list = []
    num_trials_no_ties_list = []

    for metric in metrics_to_get:
        print()
        print(metric)

        cur_comp_df = pd.merge(model_pred_metric_df, baseline_pred_metric_df,on=merge_cols, how="inner")
        print(cur_comp_df)
        cur_comp_df = cur_comp_df[merge_cols+ [model_tag + "_" + metric, baseline_tag + "_" + metric]]
        print(cur_comp_df)

        model_errors = list(cur_comp_df[model_tag + "_" + metric])
        baseline_errors = list(cur_comp_df[baseline_tag + "_" + metric])
        cur_comp_df["is_winner"]=[1 if model_error < baseline_error else 0 if baseline_error<model_error else "tie" for model_error, baseline_error in zip(model_errors, baseline_errors)]
        print(cur_comp_df)
        # sys.exit(0)

        num_trials =cur_comp_df.shape[0]
        num_wins = cur_comp_df[cur_comp_df["is_winner"]==1].shape[0]
        num_ties = cur_comp_df[cur_comp_df["is_winner"]=="tie"].shape[0]
        win_freq = num_wins/(1.0 * num_trials)
        num_trials_no_ties = num_trials - num_ties

        metric_df_list.append(metric)
        num_wins_df_list.append(num_wins)
        num_ties_df_list.append(num_ties)
        num_trials_df_list.append(num_trials)
        win_freq_df_list.append(win_freq)
        num_trials_no_ties_list.append(num_trials_no_ties)


        cur_metric_output_fp = baseline_comp_output_dir + "%s-%s-model-baseline-comparisons.csv"%(y_tag ,metric )
        cur_comp_df.to_csv(cur_metric_output_fp, index=False)
        print(cur_metric_output_fp)

    data={
        "metric":metric_df_list,
        "num_trials":num_trials_df_list,
        "num_ties":num_ties_df_list,
        "num_trials":num_trials_df_list,
        "num_wins_freq":win_freq_df_list,
        "num_trials_no_ties":num_trials_no_ties_list,
        "num_wins":num_wins_df_list
        }

    col_order = ["metric","num_wins" ,"num_trials", "num_ties", "num_trials","num_trials_no_ties", "num_wins_freq", ]

    metric_comp_df = pd.DataFrame(data=data)
    metric_comp_df=metric_comp_df[col_order]
    print(metric_comp_df)

    # output_fp = baseline_comp_output_dir + "%s-model-baseline-comp-summary.csv"%y_tag
    # metric_comp_df.to_csv(output_fp,index=False)
    # print(output_fp)

    return metric_comp_df

#compare these models
def compare_models_to_basic_model(main_output_dir ,full_model_tag_to_final_subdir_dict, REGULAR_MODEL_TAG, metrics_to_get, metric_tag, model_type, eval_tag):

    y_tag = "y_%s"%eval_tag

    # main_metric_comp_tag = "%s_%s"%()

    model_tag_to_metric_df_dict = {}
    for model_tag, model_dir in full_model_tag_to_final_subdir_dict.items():
        print("%s: %s"%(model_tag, model_dir))

        fp = model_dir + "Metrics/%s/y_%s-%s-metrics.csv"%( model_type,eval_tag, model_type)
        metric_df = pd.read_csv(fp)

        # try:
        #   fp = model_dir + "Metrics/%s/y_%s-%s-metrics.csv"%( model_type,eval_tag, model_type)
        #   metric_df = pd.read_csv(fp)
        # except FileNotFoundError:
        #   fp = model_dir + "Metrics/%s/y_%s-%s-metrics.csv"%( model_type,eval_tag, model_type)
        #   metric_df = pd.read_csv(fp)

        # metric_df["model_tag"] = model_tag
        metric_df = metric_df.drop("model_tag", axis=1)
        for col in metrics_to_get:
            metric_df = metric_df.rename(columns={col : col +"_" + model_tag})
        print("\nmetric_df")
        print(metric_df)
        model_tag_to_metric_df_dict[model_tag] = metric_df

    baseline_pred_metric_df = model_tag_to_metric_df_dict[REGULAR_MODEL_TAG]
    print("\nbaseline_pred_metric_df")
    # baseline_pred_metric_df["model_tag"] = REGULAR_MODEL_TAG
    print(baseline_pred_metric_df)

    merge_cols = ["infoID", "target_ft", "y_tag"]


    model_tag_df_list = []
    regular_model_tag_df_list = []
    win_df_list = []
    num_trials_df_list = []
    num_ties_df_list = []
    win_freq_df_list = []
    num_trials_no_ties_list = []
    model_tag_to_comp_df_dict = {}
    model_metric_to_error_list_dict = {}
    baseline_metric_to_error_list_dict = {}
    for cur_metric in metrics_to_get:
        model_metric_to_error_list_dict[cur_metric] = []
        baseline_metric_to_error_list_dict[cur_metric] = []
    for model_tag, metric_df in model_tag_to_metric_df_dict.items():

        if model_tag != REGULAR_MODEL_TAG:
            cur_comp_df = pd.merge(metric_df, baseline_pred_metric_df, on=merge_cols, how="inner")
            print("\ncur_comp_df")
            print(cur_comp_df)

            model_errors = list(cur_comp_df[metric_tag +"_" + model_tag])
            baseline_errors = list(cur_comp_df[metric_tag +"_" + REGULAR_MODEL_TAG])
            cur_comp_df["is_winner"]=[1 if model_error < baseline_error else 0 if baseline_error<model_error else "tie" for model_error, baseline_error in zip(model_errors, baseline_errors)]
            print(cur_comp_df)
            model_tag_to_comp_df_dict[model_tag] = cur_comp_df
            # sys.exit(0)

            num_trials =cur_comp_df.shape[0]
            num_wins = cur_comp_df[cur_comp_df["is_winner"]==1].shape[0]
            num_ties = cur_comp_df[cur_comp_df["is_winner"]=="tie"].shape[0]
            win_freq = num_wins/(1.0 * num_trials)
            num_trials_no_ties = num_trials - num_ties

            model_tag_df_list.append(model_tag)
            regular_model_tag_df_list.append(REGULAR_MODEL_TAG)
            win_df_list.append(num_wins)
            num_trials_df_list.append(num_trials)
            num_ties_df_list.append(num_ties)
            num_trials_no_ties_list.append(num_trials_no_ties)
            win_freq_df_list.append(win_freq)

            for cur_metric in metrics_to_get:
                model_errors = cur_comp_df[cur_metric +"_" + model_tag]
                baseline_errors = cur_comp_df[cur_metric +"_" + REGULAR_MODEL_TAG]

                model_metric_to_error_list_dict[cur_metric].append(model_errors.mean())
                baseline_metric_to_error_list_dict[cur_metric].append(baseline_errors.mean())





    data={
    "model_tag" : model_tag_df_list,
    "baseline_tag" : regular_model_tag_df_list,
    "model_win_freq" : win_freq_df_list ,
    "num_trials" : num_trials_df_list,
    "num_ties" : num_ties_df_list,
    "num_trials_no_ties" : num_trials_no_ties_list,
    "num_wins" : win_df_list
    }

    comp_df = pd.DataFrame(data=data)
    col_order = [ "model_tag","num_wins","num_ties","num_trials" ,"num_trials_no_ties","model_win_freq"]
    comp_df = comp_df[col_order]
    # print("\ncomp_df")
    # print(comp_df)

    for cur_metric in metrics_to_get:
        comp_df["mean_model_%s"%cur_metric] = model_metric_to_error_list_dict[cur_metric]
        comp_df["mean_baseline_%s"%cur_metric] = baseline_metric_to_error_list_dict[cur_metric]



    comp_df = comp_df.sort_values("model_win_freq", ascending=False).reset_index(drop=True)

    print("\ncomp_df")
    print(comp_df)

    return comp_df

def get_idx_to_model_param_dict_v2(DEBUG,objective_list,
    colsample_bytree_list,
    learning_rate_list,
    max_depth_list,
    n_estimators_list,
    subsample_list,
    gamma_list,
    alpha_list,
    reg_lambda_list,
    random_state_list,
    n_jobs_list):

    print("\nGetting model param dict...")
    #============================= debug =============================
    if DEBUG == True:
        objective_list= objective_list[:1]
        colsample_bytree_list= colsample_bytree_list[:1]
        learning_rate_list= learning_rate_list[:1]
        max_depth_list= max_depth_list[:1]
        n_estimators_list= n_estimators_list[:1]
        subsample_list=subsample_list[:1]
        gamma_list= gamma_list[:1]
        alpha_list= alpha_list[:1]

        #l2
        reg_lambda_list= reg_lambda_list[:1]
        random_state_list= random_state_list[:1]
        n_jobs_list= n_jobs_list[:1]

    #============================= combo list =============================
    combo_list=[
    objective_list,
    colsample_bytree_list,
    learning_rate_list,
    max_depth_list,
    n_estimators_list,
    subsample_list,
    gamma_list,
    alpha_list,
    reg_lambda_list,
    random_state_list,
    n_jobs_list
    ]

    num_combos=1
    for cur_list in combo_list:
        num_combos = num_combos*len(cur_list)
    print("\nnum combos: %d"%num_combos)

    idx_to_model_param_dict = {}

    i=0
    for objective in objective_list:
        for colsample_bytree in colsample_bytree_list:
            for learning_rate in learning_rate_list:
                for max_depth in max_depth_list:
                    for n_estimators in n_estimators_list:
                        for subsample in subsample_list:
                            for gamma in gamma_list:
                                for alpha in alpha_list:
                                    for reg_lambda in reg_lambda_list:
                                        for random_state in random_state_list:
                                            for n_jobs in n_jobs_list:
                                                param_dict = {
                                                    "objective":objective,
                                                    "colsample_bytree":colsample_bytree,
                                                    "learning_rate":learning_rate,
                                                    "max_depth":max_depth,
                                                    "n_estimators":n_estimators,
                                                    "subsample":subsample,
                                                    "gamma":gamma,
                                                    "alpha":alpha,
                                                    "reg_lambda":reg_lambda,
                                                    "random_state":random_state,
                                                    "n_jobs":n_jobs
                                                    }
                                                idx_to_model_param_dict[i]=param_dict
                                                i+=1
    return idx_to_model_param_dict,num_combos

def get_idx_to_model_param_dict(DEBUG,objective_list,
    colsample_bytree_list,
    learning_rate_list,
    max_depth_list,
    n_estimators_list,
    subsample_list,
    gamma_list,
    alpha_list,
    reg_lambda_list,
    random_state_list,
    n_jobs_list):

    print("\nGetting model param dict...")
    #============================= debug =============================
    if DEBUG == True:
        objective_list= objective_list[:1]
        colsample_bytree_list= colsample_bytree_list[:1]
        learning_rate_list= learning_rate_list[:1]
        max_depth_list= max_depth_list[:1]
        n_estimators_list= n_estimators_list[:1]
        subsample_list=subsample_list[:1]
        gamma_list= gamma_list[:1]
        alpha_list= alpha_list[:1]

        #l2
        reg_lambda_list= reg_lambda_list[:1]
        random_state_list= random_state_list[:1]
        n_jobs_list= n_jobs_list[:1]

    #============================= combo list =============================
    combo_list=[
    objective_list,
    colsample_bytree_list,
    learning_rate_list,
    max_depth_list,
    n_estimators_list,
    subsample_list,
    gamma_list,
    alpha_list,
    reg_lambda_list,
    random_state_list,
    n_jobs_list
    ]

    num_combos=1
    for cur_list in combo_list:
        num_combos = num_combos*len(cur_list)
    print("\nnum combos: %d"%num_combos)

    idx_to_model_param_dict = {}

    i=0
    for objective in objective_list:
        for colsample_bytree in colsample_bytree_list:
            for learning_rate in learning_rate_list:
                for max_depth in max_depth_list:
                    for n_estimators in n_estimators_list:
                        for subsample in subsample_list:
                            for gamma in gamma_list:
                                for alpha in alpha_list:
                                    for reg_lambda in reg_lambda_list:
                                        for random_state in random_state_list:
                                            for n_jobs in n_jobs_list:
                                                param_dict = {
                                                    "objective":objective,
                                                    "colsample_bytree":colsample_bytree,
                                                    "learning_rate":learning_rate,
                                                    "max_depth":max_depth,
                                                    "n_estimators":n_estimators,
                                                    "subsample":subsample,
                                                    "gamma":gamma,
                                                    "alpha":alpha,
                                                    "reg_lambda":reg_lambda,
                                                    "random_state":random_state,
                                                    "n_jobs":n_jobs
                                                    }
                                                idx_to_model_param_dict[i]=param_dict
                                                i+=1
    return idx_to_model_param_dict,num_combos

def KNN_get_idx_to_model_param_dict(DEBUG,knn_weight_types,
    algorithms,n_neighbor_list,n_jobs_list):

    print("\nGetting model param dict...")
    #============================= debug =============================
    if DEBUG == True:
        knn_weight_types = knn_weight_types[:1]
        algorithms=algorithms[:1]
        n_neighbor_list = n_neighbor_list[:1]
        n_jobs_list=n_jobs_list[:1]

    #============================= combo list =============================
    combo_list=[
        knn_weight_types,
        algorithms,
        n_neighbor_list,
        n_jobs_list
    ]

    num_combos=1
    for cur_list in combo_list:
        num_combos = num_combos*len(cur_list)
    print("\nnum combos: %d"%num_combos)

    idx_to_model_param_dict = {}

    i=0
    for weights in knn_weight_types:
        for algorithm in algorithms:
            for n_neighbors in n_neighbor_list:
                for n_jobs in n_jobs_list:
                    param_dict = {
                        "weights" : weights,
                        "n_neighbors" : n_neighbors,
                        "algorithm" : algorithm,
                        "n_jobs":n_jobs
                        }
                    idx_to_model_param_dict[i]=param_dict
                    i+=1
    return idx_to_model_param_dict,num_combos

def SVR_RBF_get_idx_to_model_param_dict(DEBUG,kernel_list,C_list,epsilon_list,gamma_list ):

    print("\nGetting model param dict...")
    #============================= debug =============================
    if DEBUG == True:
        kernel_list = kernel_list[:1]
        C_list=C_list[:1]
        epsilon_list = epsilon_list[:1]
        gamma_list=gamma_list[:1]

    #============================= combo list =============================
    combo_list=[
        kernel_list,
        C_list,
        epsilon_list,
        gamma_list
    ]

    num_combos=1
    for cur_list in combo_list:
        num_combos = num_combos*len(cur_list)
    print("\nnum combos: %d"%num_combos)

    idx_to_model_param_dict = {}

    i=0
    for kernel in kernel_list:
        for C in C_list:
            for epsilon in epsilon_list:
                for gamma in gamma_list:
                    param_dict = {
                        "kernel" : kernel,
                        "C" : C,
                        "epsilon" : epsilon,
                        "gamma":gamma
                        }
                    idx_to_model_param_dict[i]=param_dict
                    i+=1
    return idx_to_model_param_dict,num_combos

def get_proper_temporal_metrics(metrics_to_get, model_tag,model_subdir, model_type, eval_tag):

    metric_to_metric_df_dict = {}
    for metric_tag in metrics_to_get:
        metric_func = metric_str_to_function_dict[metric_tag]
        print(metric_func)

        pred_fp  = model_subdir + model_type + "/y_" + eval_tag + "-" + model_type + "-predictions.csv"
        pred_df = pd.read_csv(pred_fp)
        print(pred_df)

        metric_results= []
        preds = list(pred_df["pred"])
        actuals = list(pred_df["actual"])
        for pred,actual in zip(preds,actuals):
            metric_results.append(metric_func(pd.Series([actual]),pd.Series([pred])))
        pred_df[metric_tag] = metric_results

        print("\npred_df with metrics")
        print(pred_df)

        metric_to_metric_df_dict[metric_tag] = pred_df



    return pred_df

def get_proper_persistence_metric_model_dfs(metrics_to_get ,platform, arbitrary_model_input_dir,ensemble_eval_tags):

    proper_eval_tag_to_persistence_model_metric_df_dict = {}

    for ensemble_eval_tag in ensemble_eval_tags:

        print()
        print(ensemble_eval_tag)

        cur_persistence_model_pred_fp = arbitrary_model_input_dir + "Persistence_Baseline/y_%s-Persistence_Baseline-predictions.csv"%ensemble_eval_tag
        cur_persistence_model_pred_df = pd.read_csv(cur_persistence_model_pred_fp)
        print("\ncur_persistence_model_pred_df")
        print(cur_persistence_model_pred_df)

        for metric_tag in metrics_to_get:
            metric_func = metric_str_to_function_dict[metric_tag]
            print(metric_func)
            metric_results= []
            preds = list(cur_persistence_model_pred_df["pred"])
            actuals = list(cur_persistence_model_pred_df["actual"])
            for pred,actual in zip(preds,actuals):
                metric_results.append(metric_func(pd.Series([actual]),pd.Series([pred])))
            cur_persistence_model_pred_df[metric_tag] = metric_results

        print("\n%s cur_persistence_model_pred_df"%ensemble_eval_tag)
        print(cur_persistence_model_pred_df)


        proper_eval_tag_to_persistence_model_metric_df_dict[ensemble_eval_tag] = cur_persistence_model_pred_df

    return proper_eval_tag_to_persistence_model_metric_df_dict

def create_proper_metrics_dir_per_model(full_model_tag_to_final_subdir_dict, ensemble_eval_tags, eval_tag_to_model_tag_to_metric_df_dict,model_type):

    for model_tag,model_subdir in full_model_tag_to_final_subdir_dict.items():

        print()
        print(model_tag)
        print(model_subdir)

        metric_output_dir = model_subdir + "Proper-Metrics/"
        create_output_dir(metric_output_dir)

        for ensemble_eval_tag in ensemble_eval_tags:

            metric_df = eval_tag_to_model_tag_to_metric_df_dict[ensemble_eval_tag][model_tag]
            print("\nmetric_df")
            print(metric_df)

            cur_output_dir = metric_output_dir + model_type + "/"
            create_output_dir(cur_output_dir)
            output_fp =cur_output_dir  + "y_" + ensemble_eval_tag +"-"+ model_type + "-proper-temporal-metrics.csv"
            metric_df.to_csv(output_fp , index=False)
            print(output_fp)

            # sys.exit(0)

def get_cur_comp_df_win_info(cur_comp_df, metric, model_tag, baseline_tag):
    model_errors = list(cur_comp_df[model_tag + "_" + metric])
    baseline_errors = list(cur_comp_df[baseline_tag + "_" + metric])
    cur_comp_df["is_winner"]=[1 if model_error < baseline_error else 0 if baseline_error<model_error else "tie" for model_error, baseline_error in zip(model_errors, baseline_errors)]
    print(cur_comp_df)
    return cur_comp_df

def create_proper_baseline_comparison_df(baseline_model_metric_df, model_metric_df, rename_cols,comp_merge_cols,proper_baseline_dir,y_eval_tag, metric_tag="rmse"):

    baseline_tag = baseline_model_metric_df["model_tag"].unique()[0]
    model_tag = model_metric_df["model_tag"].unique()[0]

    for col in rename_cols:
        baseline_model_metric_df = baseline_model_metric_df.rename(columns={col: baseline_tag + "_" +col})

    for col in rename_cols:
        model_metric_df = model_metric_df.rename(columns={col: model_tag +"_" +col})

    cur_comp_df = pd.merge(model_metric_df, baseline_model_metric_df, on=comp_merge_cols, how="inner")

    print("\ncur_comp_df")
    print(cur_comp_df)

    cur_comp_df = get_cur_comp_df_win_info(cur_comp_df,metric_tag, model_tag, baseline_tag)

    print("\ncur_comp_df with win info")
    print(cur_comp_df)

    cur_output_dir =proper_baseline_dir + y_eval_tag + "/"
    create_output_dir(cur_output_dir)
    cur_output_fp = cur_output_dir + y_eval_tag + "-" + metric_tag + "-model-baseline-omparisons.csv"
    cur_comp_df.to_csv(cur_output_fp,index=False)
    print(cur_output_fp)

    return cur_comp_df




def create_proper_baseline_comparison_dir_per_model( full_model_tag_to_final_subdir_dict,rename_cols,comp_merge_cols,comp_drop_cols, eval_tag_to_model_tag_to_metric_df_dict,proper_eval_tag_to_persistence_model_metric_df_dict,ensemble_eval_tags,metric_tag):

    eval_tag_to_model_tag_to_comp_df_dict = {}
    for eval_tag in ensemble_eval_tags:
        eval_tag_to_model_tag_to_comp_df_dict[eval_tag] = {}

        #get baseline model
        baseline_model_metric_df = proper_eval_tag_to_persistence_model_metric_df_dict[eval_tag]
        print()
        print(eval_tag)
        print("\nbaseline_model_metric_df")
        print(baseline_model_metric_df)

        for col in rename_cols:
            baseline_model_metric_df = baseline_model_metric_df.rename(columns={col: "Persistence_Baseline_"+col})

        cur_eval_dict = eval_tag_to_model_tag_to_metric_df_dict[eval_tag]
        for model_tag,metric_df in cur_eval_dict.items():


            print()
            print(eval_tag)
            print(model_tag)
            print(metric_df)

            for col in rename_cols:
                metric_df = metric_df.rename(columns={col: model_tag +"_" +col})

            cur_comp_df = pd.merge(metric_df, baseline_model_metric_df, on=comp_merge_cols, how="inner")

            print("\ncur_comp_df")
            print(cur_comp_df)

            cur_comp_df = get_cur_comp_df_win_info(cur_comp_df, "rmse", model_tag, "Persistence_Baseline")

            print("\ncur_comp_df with win info")
            print(cur_comp_df)

            model_subdir = full_model_tag_to_final_subdir_dict[model_tag]
            cur_output_dir = model_subdir + "Proper-Baseline-Comparisons/y_%s/"%eval_tag
            create_output_dir(cur_output_dir)

            cur_output_fp = cur_output_dir + "y_" + eval_tag + "-" + metric_tag + "-model-baseline-comparisons.csv"
            cur_comp_df.to_csv(cur_output_fp,index=False)
            print(cur_output_fp)

            eval_tag_to_model_tag_to_comp_df_dict[eval_tag][model_tag] = cur_comp_df

    return eval_tag_to_model_tag_to_comp_df_dict

def make_comp_df_from_eval_tag_to_model_tag_to_comp_df_dict(eval_tag_to_model_tag_to_comp_df_dict, baseline_tag, metric_tag,output_dir ):

    for eval_tag,eval_tag_dict in eval_tag_to_model_tag_to_comp_df_dict.items():
        model_tag_df_list = []
        num_wins_df_list = []
        num_ties_df_list = []
        num_trials_df_list = []
        win_freq_df_list = []
        num_trials_no_ties_list = []
        avg_baseline_error_list = []
        avg_model_error_list = []

        for model_tag, cur_comp_df in eval_tag_dict.items():
            print()
            print(model_tag)
            print(cur_comp_df)






            num_trials =cur_comp_df.shape[0]
            num_wins = cur_comp_df[cur_comp_df["is_winner"]==1].shape[0]
            num_ties = cur_comp_df[cur_comp_df["is_winner"]=="tie"].shape[0]
            win_freq = num_wins/(1.0 * num_trials)
            num_trials_no_ties = num_trials - num_ties

            avg_model_error = cur_comp_df[model_tag + "_" + metric_tag].mean()
            avg_baseline_error = cur_comp_df[baseline_tag + "_" + metric_tag].mean()

            model_error_tag = "mean_model_" + metric_tag
            baseline_error_tag = "mean_" + baseline_tag + "_" + metric_tag

            model_tag_df_list.append(model_tag)
            num_wins_df_list.append(num_wins)
            num_ties_df_list.append(num_ties)
            num_trials_df_list.append(num_trials)
            win_freq_df_list.append(win_freq)
            num_trials_no_ties_list.append(num_trials_no_ties)
            avg_baseline_error_list.append(avg_baseline_error)
            avg_model_error_list.append(avg_model_error)

        data={
        "model_tag" : model_tag_df_list,
        "model_win_freq" : win_freq_df_list ,
        "num_trials" : num_trials_df_list,
        "num_ties" : num_ties_df_list,
        "num_trials_no_ties" : num_trials_no_ties_list,
        "num_wins" : num_wins_df_list,
        model_error_tag: avg_model_error_list,
        baseline_error_tag: avg_baseline_error_list,
        }

        comp_df = pd.DataFrame(data=data)
        comp_df["baseline_tag"] = baseline_tag
        col_order = [ "model_tag", "baseline_tag","num_wins","num_ties","num_trials" ,"num_trials_no_ties","model_win_freq",model_error_tag,baseline_error_tag]
        comp_df = comp_df[col_order]
        comp_df = comp_df.sort_values("model_win_freq", ascending=False).reset_index(drop=True)
        print("\ncomp_df")
        print(comp_df)

        output_fp = output_dir + "Y_%s-Proper-Temporal-Comparison-Results.csv"%eval_tag
        comp_df.to_csv(output_fp, index=False)
        print(output_fp)




    # sys.exit(0)
def get_corr_ensemble_model(model_tag_to_subdir_dict,CORRELATION_FUNCS_TO_USE_LIST,additional_params,output_dir,corr_model_tag, model_type,sample_generator_class_func,platforms,infoIDs,corr_type,n_jobs):

    corr_output_dir = output_dir+ corr_model_tag + "/"
    create_output_dir(corr_output_dir)

    #load best params
    # model_tag_to_sample_gen_obj_dict = {}
    model_tag_to_corr_dfs_dict = {}
    false_corr_flag_dict = {}
    #combine the dfs
    all_val_corr_dfs= []
    all_test_corr_dfs = []

    for model_tag,model_subdir in model_tag_to_subdir_dict.items():

        #get params
        param_fp = model_subdir + "full-params.csv"
        param_df = pd.read_csv(param_fp)
        print(param_df)

        data_proc_params_dict = get_model_data_proc_params_from_param_df(param_df)
        data_proc_params_dict.update(additional_params)
        # data_proc_params_dict[]
        for param, val in data_proc_params_dict.items():
            print("%s: %s"%(param, str(val)))

        data_proc_params_dict["CORRELATION_FUNCS_TO_USE_LIST"]=CORRELATION_FUNCS_TO_USE_LIST
        data_proc_params_dict["n_jobs"] = n_jobs
        sample_generator = sample_generator_class_func(**data_proc_params_dict)

        #get array dict
        infoID_train_and_test_array_dict = sample_generator.configure_features_and_get_infoID_train_and_test_array_dict()
        target_ft_category_to_list_of_temporal_indices_dict = sample_generator.get_target_ft_category_to_list_of_temporal_indices()
        input_ft_category_to_list_of_temporal_indices_dict = sample_generator.get_input_ft_category_to_list_of_temporal_indices()
        # sample_generator.get_corr_fts()

        #GOT_CORR_FTS_FLAG = sample_generator.get_main_ts_to_exo_ts_corrs()
        GOT_CORR_FTS_FLAG = sample_generator.get_full_corr_data_para()
        false_corr_flag_dict[model_tag] = GOT_CORR_FTS_FLAG

        if GOT_CORR_FTS_FLAG == True:
            target_fts = sample_generator.target_fts
            array_type_to_correlation_materials_dict=sample_generator.array_type_to_correlation_materials_dict
            val_gt_df_with_corr_data = create_gt_df_with_corr_data(CORRELATION_FUNCS_TO_USE_LIST,array_type_to_correlation_materials_dict,platforms,infoID_train_and_test_array_dict,infoIDs,target_fts,"val",model_tag, model_subdir,SAVE_FILE=True)
            test_gt_df_with_corr_data = create_gt_df_with_corr_data(CORRELATION_FUNCS_TO_USE_LIST,array_type_to_correlation_materials_dict,platforms,infoID_train_and_test_array_dict,infoIDs,target_fts,"test",model_tag, model_subdir,SAVE_FILE=True)
            model_tag_to_corr_dfs_dict["val"] = val_gt_df_with_corr_data
            model_tag_to_corr_dfs_dict["test"] = test_gt_df_with_corr_data

            all_val_corr_dfs.append(val_gt_df_with_corr_data)
            all_test_corr_dfs.append(test_gt_df_with_corr_data)

            print("\nmodel_tag_to_subdir_dict")
            print(model_tag_to_subdir_dict)

    print("\nfalse_corr_flag_dict:")
    for model_tag,got_corr_ft_flag in false_corr_flag_dict.items():
        print("%s: %s"%(model_tag, got_corr_ft_flag))

    model_tags = list(model_tag_to_subdir_dict.keys())
    print("\nmodel_tags")
    print(model_tags)

    # ignore_tags = [REGULAR_MODEL_TAG]
    val_full_pred_df = get_corr_exp_pred_df(model_tags, model_tag_to_corr_dfs_dict,model_tag_to_subdir_dict, "val", model_type,[])
    test_full_pred_df = get_corr_exp_pred_df(model_tags, model_tag_to_corr_dfs_dict,model_tag_to_subdir_dict, "test", model_type,[])

    # val_regular_model_pred_df = val_full_pred_df[val_full_pred_df["model_tag"]==REGULAR_MODEL_TAG].reset_index(drop=True)
    # test_regular_model_pred_df = test_full_pred_df[test_full_pred_df["model_tag"]==REGULAR_MODEL_TAG].reset_index(drop=True)

    val_full_corr_df = pd.concat(all_val_corr_dfs)
    test_full_corr_df = pd.concat(all_test_corr_dfs)

    print("\nval_full_pred_df")
    print(val_full_pred_df)

    print("\ntest_full_pred_df")
    print(test_full_pred_df)


    print("\nval_full_corr_df")
    print(val_full_corr_df)

    print("\ntest_full_corr_df")
    print(test_full_corr_df)

    merge_cols = ["timestep" ,"actual", "platform", "infoID", "target_ft", "model_tag", "y_tag"]

    val_pred_and_corr_df = pd.merge(val_full_corr_df,val_full_pred_df,on=merge_cols, how="outer")
    test_pred_and_corr_df = pd.merge(test_full_corr_df,test_full_pred_df,on=merge_cols, how="outer")

    val_pred_and_corr_df = val_pred_and_corr_df.dropna()
    test_pred_and_corr_df = test_pred_and_corr_df.dropna()

    print("\nval_pred_and_corr_df")
    print(val_pred_and_corr_df)

    print("\ntest_pred_and_corr_df")
    print(test_pred_and_corr_df)

    drop_dupe_cols = ["timestep", "target_ft", "platform", "infoID"]
    val_pred_and_corr_df = val_pred_and_corr_df.sort_values(corr_type, ascending=False)
    val_pred_and_corr_df = val_pred_and_corr_df.drop_duplicates(drop_dupe_cols).reset_index(drop=True)
    val_pred_and_corr_df = val_pred_and_corr_df.sort_values(["timestep", "infoID", "target_ft"]).reset_index(drop=True)
    print("\nval_pred_and_corr_df after getting preds by corr val")
    print(val_pred_and_corr_df)
    print("\nval_pred_and_corr_df val counts")
    print(val_pred_and_corr_df["model_tag"].value_counts())

    # drop_dupe_cols = ["timestep", "target_ft", "platform", "infoID"]
    test_pred_and_corr_df = test_pred_and_corr_df.sort_values(corr_type, ascending=False)
    test_pred_and_corr_df = test_pred_and_corr_df.drop_duplicates(drop_dupe_cols).reset_index(drop=True)
    test_pred_and_corr_df = test_pred_and_corr_df.sort_values(["timestep", "infoID", "target_ft"]).reset_index(drop=True)
    print("\ntest_pred_and_corr_df after getting preds by corr val")
    print(test_pred_and_corr_df)
    print("\ntest_pred_and_corr_df val counts")
    print(test_pred_and_corr_df["model_tag"].value_counts())

    cur_corr_output_dir = corr_output_dir + model_type + "/"
    create_output_dir(cur_corr_output_dir)
    output_fp = cur_corr_output_dir + "y_val-" + model_type + "-predictions.csv"
    val_pred_and_corr_df.to_csv(output_fp, index=False)
    print(output_fp)

    value_count_output_fp = cur_corr_output_dir + "y_val-" + model_type + "-value-counts.txt"
    value_counts = val_pred_and_corr_df["model_tag"].value_counts()
    with open(value_count_output_fp, "w") as f:
        f.write(str(value_counts))
        print(value_count_output_fp)

    # cur_corr_output_dir = corr_output_dir + model_type + "/"
    # create_output_dir(cur_corr_output_dir)
    output_fp = cur_corr_output_dir + "y_test-" + model_type + "-predictions.csv"
    test_pred_and_corr_df.to_csv(output_fp, index=False)
    print(output_fp)

    value_count_output_fp = cur_corr_output_dir + "y_test-" + model_type + "-value-counts.txt"
    value_counts = test_pred_and_corr_df["model_tag"].value_counts()
    with open(value_count_output_fp, "w") as f:
        f.write(str(value_counts))
        print(value_count_output_fp)

    return corr_output_dir

def get_idx_to_sample_generator_param_dict_v3_day_gran_option( DEBUG,train_start,train_end,val_start,val_end,test_start,test_end,n_jobs,
    main_input_dir,infoIDs,platforms,user_statuses,LOGNORM_IGNORE_TAGS,LOGNORM_DEBUG_PRINT,
    RESCALE_LIST,SCALER_TYPE_LIST,FEATURE_RANGE_LIST,RESCALE_TARGET_LIST,INPUT_TWITTER_LOGNORM_LIST,OUTPUT_TWITTER_LOGNORM_LIST,
    INPUT_OTHER_LOGNORM_LIST,OUTPUT_OTHER_LOGNORM_LIST,TARGET_PLATFORM_LIST,GET_GDELT_FEATURES_LIST,
    GET_REDDIT_FEATURES_LIST,GET_1HOT_INFO_ID_FTS_LIST, GET_EXTERNAL_PLATFORM_FTS_LIST,main_output_dir,CORRELATION_FUNCS_TO_USE_LIST_LIST,CORRELATION_STR_TO_FUNC_DICT,
    INPUT_GRAN_LIST ,
    OUTPUT_GRAN_LIST,
    INPUT_TIMESTEPS_LIST,
    OUTPUT_TIMESTEPS_LIST):

    print("\nGetting sample gen params...")
    idx_to_sample_generator_param_dict = {}



    if DEBUG==True:
        GET_GDELT_FEATURES_LIST=GET_GDELT_FEATURES_LIST[:1]
        GET_REDDIT_FEATURES_LIST=GET_REDDIT_FEATURES_LIST[:1]
        GET_1HOT_INFO_ID_FTS_LIST=GET_1HOT_INFO_ID_FTS_LIST[:1]
        GET_EXTERNAL_PLATFORM_FTS_LIST=GET_EXTERNAL_PLATFORM_FTS_LIST[:1]
        TARGET_PLATFORM_LIST=TARGET_PLATFORM_LIST[:1]
        RESCALE_LIST=RESCALE_LIST[:1]
        SCALER_TYPE_LIST= SCALER_TYPE_LIST[:1]
        FEATURE_RANGE_LIST= FEATURE_RANGE_LIST[:1]
        RESCALE_TARGET_LIST= RESCALE_TARGET_LIST[:1]
        INPUT_TWITTER_LOGNORM_LIST =INPUT_TWITTER_LOGNORM_LIST[:1]
        OUTPUT_TWITTER_LOGNORM_LIST = OUTPUT_TWITTER_LOGNORM_LIST[:1]
        INPUT_OTHER_LOGNORM_LIST = INPUT_OTHER_LOGNORM_LIST[:1]
        OUTPUT_OTHER_LOGNORM_LIST = OUTPUT_OTHER_LOGNORM_LIST[:1]
        CORRELATION_FUNCS_TO_USE_LIST_LIST=CORRELATION_FUNCS_TO_USE_LIST_LIST[:1]


        INPUT_GRAN_LIST = INPUT_GRAN_LIST[:1]
        OUTPUT_GRAN_LIST = OUTPUT_GRAN_LIST[:1]
        INPUT_TIMESTEPS_LIST = INPUT_TIMESTEPS_LIST[:1]
        OUTPUT_TIMESTEPS_LIST = OUTPUT_TIMESTEPS_LIST[:1]


    num_combos=0
    for CORRELATION_FUNCS_TO_USE_LIST in CORRELATION_FUNCS_TO_USE_LIST_LIST:
        for INPUT_GRAN in INPUT_GRAN_LIST:
            for OUTPUT_GRAN in OUTPUT_GRAN_LIST:
                for INPUT_TIMESTEPS in INPUT_TIMESTEPS_LIST:
                    for OUTPUT_TIMESTEPS in OUTPUT_TIMESTEPS_LIST:
                        for GET_GDELT_FEATURES in GET_REDDIT_FEATURES_LIST:
                            for GET_REDDIT_FEATURES in GET_REDDIT_FEATURES_LIST:
                                for GET_1HOT_INFO_ID_FTS in GET_1HOT_INFO_ID_FTS_LIST:
                                    for GET_EXTERNAL_PLATFORM_FTS in GET_EXTERNAL_PLATFORM_FTS_LIST:

                                        for TARGET_PLATFORM in TARGET_PLATFORM_LIST:
                                            if (GET_EXTERNAL_PLATFORM_FTS==False) and (TARGET_PLATFORM =="all"):
                                                break
                                            else:
                                                # print("continuing loop...")
                                                for RESCALE in RESCALE_LIST:
                                                    for SCALER_TYPE in SCALER_TYPE_LIST:
                                                        for FEATURE_RANGE in FEATURE_RANGE_LIST:
                                                            for RESCALE_TARGET in RESCALE_TARGET_LIST:
                                                                for INPUT_TWITTER_LOGNORM in INPUT_TWITTER_LOGNORM_LIST:
                                                                    for OUTPUT_TWITTER_LOGNORM in OUTPUT_TWITTER_LOGNORM_LIST:
                                                                        for INPUT_OTHER_LOGNORM in INPUT_OTHER_LOGNORM_LIST:
                                                                            for OUTPUT_OTHER_LOGNORM in OUTPUT_OTHER_LOGNORM_LIST:
                                                                                cur_sample_gen_param_dict = {}
                                                                                # basic_sample_gen_params = {}
                                                                                # cur_sample_gen_param_dict["DEBUG"]=DEBUG
                                                                                cur_sample_gen_param_dict["INPUT_GRAN"] = INPUT_GRAN
                                                                                cur_sample_gen_param_dict["OUTPUT_GRAN"] = OUTPUT_GRAN
                                                                                cur_sample_gen_param_dict["INPUT_TIMESTEPS"] = INPUT_TIMESTEPS
                                                                                cur_sample_gen_param_dict["OUTPUT_TIMESTEPS"] = OUTPUT_TIMESTEPS

                                                                                cur_sample_gen_param_dict["train_start"]=train_start
                                                                                cur_sample_gen_param_dict["train_end"]=train_end
                                                                                cur_sample_gen_param_dict["val_start"]=val_start
                                                                                cur_sample_gen_param_dict["val_end"]=val_end
                                                                                cur_sample_gen_param_dict["test_start"]=test_start
                                                                                cur_sample_gen_param_dict["test_end"]=test_end
                                                                                # cur_sample_gen_param_dict["sum_fts"]=sum_fts
                                                                                # cur_sample_gen_param_dict["avg_fts"]=avg_fts
                                                                                cur_sample_gen_param_dict["n_jobs"]=n_jobs
                                                                                # cur_sample_gen_param_dict["GRAN"]=GRAN
                                                                                cur_sample_gen_param_dict["main_input_dir"]=main_input_dir
                                                                                cur_sample_gen_param_dict["main_output_dir"]=main_output_dir
                                                                                cur_sample_gen_param_dict["infoIDs"]=infoIDs
                                                                                cur_sample_gen_param_dict["platforms"]=platforms
                                                                                cur_sample_gen_param_dict["user_statuses"]=user_statuses
                                                                                cur_sample_gen_param_dict["LOGNORM_IGNORE_TAGS"]=LOGNORM_IGNORE_TAGS
                                                                                cur_sample_gen_param_dict["LOGNORM_DEBUG_PRINT"]=LOGNORM_DEBUG_PRINT
                                                                                # cur_sample_gen_param_dict["INPUT_HOURS"]=INPUT_HOURS
                                                                                # cur_sample_gen_param_dict["OUTPUT_HOURS"]=OUTPUT_HOURS
                                                                                cur_sample_gen_param_dict["GET_GDELT_FEATURES"]=GET_GDELT_FEATURES
                                                                                cur_sample_gen_param_dict["GET_REDDIT_FEATURES"]=GET_REDDIT_FEATURES
                                                                                cur_sample_gen_param_dict["GET_1HOT_INFO_ID_FTS"]=GET_1HOT_INFO_ID_FTS
                                                                                cur_sample_gen_param_dict["GET_EXTERNAL_PLATFORM_FTS"]=GET_EXTERNAL_PLATFORM_FTS
                                                                                cur_sample_gen_param_dict["TARGET_PLATFORM"]=TARGET_PLATFORM
                                                                                cur_sample_gen_param_dict["RESCALE"]=RESCALE
                                                                                cur_sample_gen_param_dict["SCALER_TYPE"]=SCALER_TYPE
                                                                                cur_sample_gen_param_dict["FEATURE_RANGE"]=FEATURE_RANGE
                                                                                cur_sample_gen_param_dict["RESCALE_TARGET"]=RESCALE_TARGET
                                                                                cur_sample_gen_param_dict["INPUT_TWITTER_LOGNORM"]=INPUT_TWITTER_LOGNORM
                                                                                cur_sample_gen_param_dict["OUTPUT_TWITTER_LOGNORM"]=OUTPUT_TWITTER_LOGNORM
                                                                                cur_sample_gen_param_dict["INPUT_OTHER_LOGNORM"]=INPUT_OTHER_LOGNORM
                                                                                cur_sample_gen_param_dict["OUTPUT_OTHER_LOGNORM"]=OUTPUT_OTHER_LOGNORM
                                                                                cur_sample_gen_param_dict["CORRELATION_FUNCS_TO_USE_LIST"]=CORRELATION_FUNCS_TO_USE_LIST
                                                                                cur_sample_gen_param_dict["CORRELATION_STR_TO_FUNC_DICT"]=CORRELATION_STR_TO_FUNC_DICT
                                                                                idx_to_sample_generator_param_dict[num_combos]=dict(cur_sample_gen_param_dict)
                                                                                num_combos+=1
    print("\nnum_combos: %d"%(num_combos+1))


    return idx_to_sample_generator_param_dict, num_combos

def graph_pred_vs_gt_with_baseline_from_comp_df(proper_comp_df, model_tag, baseline_tag, metric_tag,platform,graph_output_dir):
    create_output_dir(graph_output_dir)


    model_pred_col = model_tag + "_pred"
    baseline_pred_col = baseline_tag + "_pred"
    infoIDs = list(proper_comp_df["infoID"].unique())
    num_timesteps = proper_comp_df["timestep"].nunique()
    target_fts = list(proper_comp_df["target_ft"].unique())

    hyp_dict = hyphenate_infoID_dict(infoIDs)

    for infoID in infoIDs:

        hyp_infoID = hyp_dict[infoID]
        #============ plot stuff ==============

        coord_list = [0, 1, 2]
        fig, axs = plt.subplots(3,figsize=(9,9))

        target_ft_to_axis_coordinates_dict = {}

        # num_coords = len(coord_list)
        idx=0
        for target_ft,coord in zip(target_fts,coord_list ):
            target_ft_to_axis_coordinates_dict[target_ft] = coord
            print("%s: %s"%(target_ft, str(coord)))

        for coord,target_ft in enumerate(target_fts):
            print("\nPlotting %s %s %s..."%(platform, infoID, target_ft))

            target_ft_to_axis_coordinates_dict[target_ft] = coord



            temp_df = proper_comp_df[(proper_comp_df["infoID"]==infoID) & (proper_comp_df["platform"]==platform) & (proper_comp_df["target_ft"]==target_ft)]
            temp_df = temp_df.reset_index(drop=True)
            print("\ntemp_df")
            print(temp_df)

            if temp_df.shape[0] != num_timesteps:
                print("\nError! temp_df.shape[0] != num_timesteps")
                print(temp_df.shape[0])
                print(num_timesteps)
                sys.exit(0)
            else:
                print("\nCounts are good...Plotting...")

            #======================= get series =======================
            model_pred_series = temp_df[model_pred_col]
            baseline_pred_series = temp_df[baseline_pred_col]
            gt_series = temp_df["actual"]

            #======================= score stuff =======================
            model_mean_error = temp_df[model_tag + "_" + metric_tag].mean()
            model_wins = temp_df[temp_df["is_winner"]==1].shape[0]
            num_trials_no_ties = temp_df[temp_df["is_winner"]!="tie"].shape[0]
            try:
                model_win_score = model_wins/num_trials_no_ties
            except ZeroDivisionError:
                model_win_score = 0
            model_error_result_tag = "%s %s: %.4f"%(model_tag, metric_tag, model_mean_error)
            # print()
            # print(model_error_result_tag)

            baseline_mean_error = temp_df[baseline_tag + "_" + metric_tag].mean()
            baseline_error_result_tag = "%s %s: %.4f"%(baseline_tag , metric_tag, baseline_mean_error)
            # print(baseline_error_result_tag)

            model_score_tag = "%s win freq: %.4f"%(model_tag,  model_win_score)
            # print(model_score_tag)

            full_report_tag = model_error_result_tag + "\n" + baseline_error_result_tag + "\n" + model_score_tag
            print()
            print(full_report_tag)

            # target_no_underscore = target_ft.replace("_"," ")
            # title_tag = "%s %s"%(infoID, target_no_underscore)
            # title_tag = title_tag + "\n"+ full_report_tag
            # title = ("\n".join(wrap(title_tag, 50)))

            target_no_underscore = target_ft.replace("_"," ")
            infoID_target_tag = "%s %s"%(infoID, target_no_underscore)
            title_tag = infoID_target_tag + "\n"+ full_report_tag
            title = title_tag

            # print("\ncur_y_pred shape")
            # print(cur_y_pred.shape)
            # print("\ncur_y_ground_truth shape")
            # print(cur_y_ground_truth.shape)
            axs[coord].plot(model_pred_series,"-r" ,label=model_tag)
            axs[coord].plot(baseline_pred_series,":g" ,label=baseline_tag)
            axs[coord].plot(gt_series, "-k" ,label="Ground Truth")
            axs[coord].set_title(title)

        for ax in axs.flat:
            ax.set(xlabel='days', ylabel='Volume')
            ax.legend()

        # plt.legend()
        plt.tight_layout()
        output_fp = graph_output_dir + "%s.png"%( hyp_infoID)
        fig.savefig(output_fp)
        plt.close()
        print(output_fp)


def get_idx_to_sample_generator_param_dict_v3( DEBUG,train_start,train_end,val_start,
    val_end,test_start,test_end,n_jobs,GRAN,
    main_input_dir,infoIDs,platforms,user_statuses,LOGNORM_IGNORE_TAGS,LOGNORM_DEBUG_PRINT,
    RESCALE_LIST,SCALER_TYPE_LIST,FEATURE_RANGE_LIST,RESCALE_TARGET_LIST,INPUT_TWITTER_LOGNORM_LIST,OUTPUT_TWITTER_LOGNORM_LIST,
    INPUT_OTHER_LOGNORM_LIST,OUTPUT_OTHER_LOGNORM_LIST,TARGET_PLATFORM_LIST,INPUT_HOURS_LIST,OUTPUT_HOURS_LIST,GET_GDELT_FEATURES_LIST,
    GET_REDDIT_FEATURES_LIST,GET_1HOT_INFO_ID_FTS_LIST, GET_EXTERNAL_PLATFORM_FTS_LIST,main_output_dir,CORRELATION_FUNCS_TO_USE_LIST_LIST,
    CORRELATION_STR_TO_FUNC_DICT,AGG_TIME_SERIES_LIST):

    print("\nGetting sample gen params...")
    idx_to_sample_generator_param_dict = {}



    if DEBUG==True:
        INPUT_HOURS_LIST=INPUT_HOURS_LIST[:1]
        OUTPUT_HOURS_LIST=OUTPUT_HOURS_LIST[:1]
        GET_GDELT_FEATURES_LIST=GET_GDELT_FEATURES_LIST[:1]
        GET_REDDIT_FEATURES_LIST=GET_REDDIT_FEATURES_LIST[:1]
        GET_1HOT_INFO_ID_FTS_LIST=GET_1HOT_INFO_ID_FTS_LIST[:1]
        GET_EXTERNAL_PLATFORM_FTS_LIST=GET_EXTERNAL_PLATFORM_FTS_LIST[:1]
        TARGET_PLATFORM_LIST=TARGET_PLATFORM_LIST[:1]
        RESCALE_LIST=RESCALE_LIST[:1]
        SCALER_TYPE_LIST= SCALER_TYPE_LIST[:1]
        FEATURE_RANGE_LIST= FEATURE_RANGE_LIST[:1]
        RESCALE_TARGET_LIST= RESCALE_TARGET_LIST[:1]
        INPUT_TWITTER_LOGNORM_LIST =INPUT_TWITTER_LOGNORM_LIST[:1]
        OUTPUT_TWITTER_LOGNORM_LIST = OUTPUT_TWITTER_LOGNORM_LIST[:1]
        INPUT_OTHER_LOGNORM_LIST = INPUT_OTHER_LOGNORM_LIST[:1]
        OUTPUT_OTHER_LOGNORM_LIST = OUTPUT_OTHER_LOGNORM_LIST[:1]
        CORRELATION_FUNCS_TO_USE_LIST_LIST=CORRELATION_FUNCS_TO_USE_LIST_LIST[:1]


    num_combos=0
    for CORRELATION_FUNCS_TO_USE_LIST in CORRELATION_FUNCS_TO_USE_LIST_LIST:
        for INPUT_HOURS in INPUT_HOURS_LIST:
            for OUTPUT_HOURS in OUTPUT_HOURS_LIST:
                for GET_GDELT_FEATURES in GET_REDDIT_FEATURES_LIST:
                    for GET_REDDIT_FEATURES in GET_REDDIT_FEATURES_LIST:
                        for GET_1HOT_INFO_ID_FTS in GET_1HOT_INFO_ID_FTS_LIST:
                            for GET_EXTERNAL_PLATFORM_FTS in GET_EXTERNAL_PLATFORM_FTS_LIST:

                                for TARGET_PLATFORM in TARGET_PLATFORM_LIST:
                                    if (GET_EXTERNAL_PLATFORM_FTS==False) and (TARGET_PLATFORM =="all"):
                                        break
                                    else:
                                        # print("continuing loop...")
                                        for RESCALE in RESCALE_LIST:
                                            for SCALER_TYPE in SCALER_TYPE_LIST:
                                                for FEATURE_RANGE in FEATURE_RANGE_LIST:
                                                    for RESCALE_TARGET in RESCALE_TARGET_LIST:
                                                        for INPUT_TWITTER_LOGNORM in INPUT_TWITTER_LOGNORM_LIST:
                                                            for OUTPUT_TWITTER_LOGNORM in OUTPUT_TWITTER_LOGNORM_LIST:
                                                                for INPUT_OTHER_LOGNORM in INPUT_OTHER_LOGNORM_LIST:
                                                                    for OUTPUT_OTHER_LOGNORM in OUTPUT_OTHER_LOGNORM_LIST:
                                                                        for AGG_TIME_SERIES in AGG_TIME_SERIES_LIST:
                                                                            cur_sample_gen_param_dict = {}
                                                                            # basic_sample_gen_params = {}
                                                                            # cur_sample_gen_param_dict["DEBUG"]=DEBUG
                                                                            cur_sample_gen_param_dict["train_start"]=train_start
                                                                            cur_sample_gen_param_dict["train_end"]=train_end
                                                                            cur_sample_gen_param_dict["val_start"]=val_start
                                                                            cur_sample_gen_param_dict["val_end"]=val_end
                                                                            cur_sample_gen_param_dict["test_start"]=test_start
                                                                            cur_sample_gen_param_dict["test_end"]=test_end
                                                                            # cur_sample_gen_param_dict["sum_fts"]=sum_fts
                                                                            # cur_sample_gen_param_dict["avg_fts"]=avg_fts
                                                                            cur_sample_gen_param_dict["n_jobs"]=n_jobs
                                                                            cur_sample_gen_param_dict["GRAN"]=GRAN
                                                                            cur_sample_gen_param_dict["main_input_dir"]=main_input_dir
                                                                            cur_sample_gen_param_dict["main_output_dir"]=main_output_dir
                                                                            cur_sample_gen_param_dict["infoIDs"]=infoIDs
                                                                            cur_sample_gen_param_dict["platforms"]=platforms
                                                                            cur_sample_gen_param_dict["user_statuses"]=user_statuses
                                                                            cur_sample_gen_param_dict["LOGNORM_IGNORE_TAGS"]=LOGNORM_IGNORE_TAGS
                                                                            cur_sample_gen_param_dict["LOGNORM_DEBUG_PRINT"]=LOGNORM_DEBUG_PRINT
                                                                            cur_sample_gen_param_dict["INPUT_HOURS"]=INPUT_HOURS
                                                                            cur_sample_gen_param_dict["OUTPUT_HOURS"]=OUTPUT_HOURS
                                                                            cur_sample_gen_param_dict["GET_GDELT_FEATURES"]=GET_GDELT_FEATURES
                                                                            cur_sample_gen_param_dict["GET_REDDIT_FEATURES"]=GET_REDDIT_FEATURES
                                                                            cur_sample_gen_param_dict["GET_1HOT_INFO_ID_FTS"]=GET_1HOT_INFO_ID_FTS
                                                                            cur_sample_gen_param_dict["GET_EXTERNAL_PLATFORM_FTS"]=GET_EXTERNAL_PLATFORM_FTS
                                                                            cur_sample_gen_param_dict["TARGET_PLATFORM"]=TARGET_PLATFORM
                                                                            cur_sample_gen_param_dict["RESCALE"]=RESCALE
                                                                            cur_sample_gen_param_dict["SCALER_TYPE"]=SCALER_TYPE
                                                                            cur_sample_gen_param_dict["FEATURE_RANGE"]=FEATURE_RANGE
                                                                            cur_sample_gen_param_dict["RESCALE_TARGET"]=RESCALE_TARGET
                                                                            cur_sample_gen_param_dict["INPUT_TWITTER_LOGNORM"]=INPUT_TWITTER_LOGNORM
                                                                            cur_sample_gen_param_dict["OUTPUT_TWITTER_LOGNORM"]=OUTPUT_TWITTER_LOGNORM
                                                                            cur_sample_gen_param_dict["INPUT_OTHER_LOGNORM"]=INPUT_OTHER_LOGNORM
                                                                            cur_sample_gen_param_dict["OUTPUT_OTHER_LOGNORM"]=OUTPUT_OTHER_LOGNORM
                                                                            cur_sample_gen_param_dict["CORRELATION_FUNCS_TO_USE_LIST"]=CORRELATION_FUNCS_TO_USE_LIST
                                                                            cur_sample_gen_param_dict["CORRELATION_STR_TO_FUNC_DICT"]=CORRELATION_STR_TO_FUNC_DICT
                                                                            cur_sample_gen_param_dict["AGG_TIME_SERIES"]=AGG_TIME_SERIES
                                                                            idx_to_sample_generator_param_dict[num_combos]=dict(cur_sample_gen_param_dict)
                                                                            num_combos+=1
    print("\nnum_combos: %d"%(num_combos+1))


    return idx_to_sample_generator_param_dict, num_combos





def create_prediction_df_with_corr_data_v2_hourly_option(CORRELATION_FUNCS_TO_USE_LIST,array_type_to_correlation_materials_dict,platforms,y_pred_dict, infoID_train_and_test_array_dict,infoIDs,target_ft_categories,array_tag,model_tag, output_dir,SAVE_PREDS=True):

    if SAVE_PREDS == True:
        cur_output_dir = output_dir + model_tag + "/"
        create_output_dir(cur_output_dir)

    y_tag = "y_" + array_tag
    x_tag = "x_" + array_tag


    all_dfs = []

    try:
        cur_array_type_correlation_materials_dict = array_type_to_correlation_materials_dict[array_tag]
        infoID_to_overall_timestep_to_corr_func_to_target_ft_dict=cur_array_type_correlation_materials_dict["infoID_to_overall_timestep_to_corr_func_to_target_ft_dict"]
        num_overall_timesteps = cur_array_type_correlation_materials_dict["num_overall_timesteps"]
    except:
        print("no corr data, continuing")



    for infoID in infoIDs:


        y_pred =  y_pred_dict[infoID]
        y_ground_truth = infoID_train_and_test_array_dict[infoID][y_tag]



        print("\ny_pred.shape")
        print(y_pred.shape)
        print("\ny_ground_truth.shape")
        print(y_ground_truth.shape)

        num_target_ft_categories = len(target_ft_categories)
        num_timesteps_per_sample = int(y_pred.shape[1]/num_target_ft_categories)
        print("\nnum_timesteps_per_sample")
        print(num_timesteps_per_sample)

        y_pred = y_pred.reshape((y_pred.shape[0] * num_timesteps_per_sample, num_target_ft_categories))
        y_ground_truth = y_ground_truth.reshape((y_ground_truth.shape[0] * num_timesteps_per_sample, num_target_ft_categories))

        print("\ny_pred.shape after timestep hack")
        print(y_pred.shape)
        print("\ny_ground_truth.shape after timestep hack")
        print(y_ground_truth.shape)

        # sys.exit(0)

        for target_ft in target_ft_categories:

            pred_df = pd.DataFrame(data=y_pred, columns=target_ft_categories)
            pred_df=pred_df[[target_ft]]

            ground_truth_df = pd.DataFrame(data=y_ground_truth, columns=target_ft_categories)
            ground_truth_df=ground_truth_df[[target_ft]]

            pred_df = pred_df.rename(columns={target_ft:"pred"})
            ground_truth_df = ground_truth_df.rename(columns={target_ft:"actual"})
            df = pd.concat([pred_df, ground_truth_df], axis=1)
            print(df)

            for platform in platforms:
                if platform in target_ft:
                    df["platform"]=platform
                    break
            df["infoID"] = infoID
            # print(df)
            df["timestep"] = [(i+1) for i in range(df.shape[0])]
            df["target_ft"] = target_ft
            col_order = ["timestep", "pred", "actual", "platform", "infoID","target_ft"]

            for corr_func_str in CORRELATION_FUNCS_TO_USE_LIST:
                col_order.append(corr_func_str)
                corr_vals = []
                for overall_timestep in range(num_overall_timesteps):
                    avg_corr = infoID_to_overall_timestep_to_corr_func_to_target_ft_dict[infoID][overall_timestep][corr_func_str][target_ft]
                    corr_vals.append(avg_corr)
                df[corr_func_str] = corr_vals

            df = df[col_order]
            print(df)
            all_dfs.append(df)
    df = pd.concat(all_dfs)
    df["y_tag"]=y_tag
    df["model_tag"]=model_tag
    print(df)

    if SAVE_PREDS == True:
        output_fp = cur_output_dir + y_tag + "-" + model_tag + "-predictions.csv"
        df.to_csv(output_fp, index=False)
        print(output_fp)


    return df


def graph_pred_vs_gt_options_v2_from_df(y_pred_df, output_dir):
    y_tag = y_pred_df["y_tag"].unique()[0]
    infoIDs = list(y_pred_df["infoID"].unique())

    output_dir = output_dir+y_tag+ "/"
    create_output_dir(output_dir)

    print("\nGraphing %s"%y_tag)
    # y_ground_truth_dict = infoID_train_and_test_array_dict[y_tag]

    hyp_dict = hyphenate_infoID_dict(infoIDs)
    target_fts = list(y_pred_df["target_ft"].unique())



    for infoID in infoIDs:

        hyp_infoID = hyp_dict[infoID]

        cur_infoID_df = y_pred_df[y_pred_df["infoID"]==infoID]
        print("\ncur_infoID_df")
        print(cur_infoID_df)



        num_target_fts = len(target_fts)
        if num_target_fts == 3:
            coord_list = [0, 1, 2]
            fig, axs = plt.subplots(3,figsize=(8,8))
        else:
            coord_list = [(0,0), (0,1), (1,0), (1,1), (2,0), (2,1)]
            fig, axs = plt.subplots(3, 2,figsize=(8,8))

        target_ft_to_axis_coordinates_dict = {}

        # num_coords = len(coord_list)
        idx=0
        for target_ft,coord in zip(target_fts,coord_list ):

            cur_target_ft_df = cur_infoID_df[cur_infoID_df["target_ft"]==target_ft].reset_index(drop=True)
            print("\ncur_target_ft_df")
            print(cur_target_ft_df)

            target_ft_to_axis_coordinates_dict[target_ft] = coord
            print("%s: %s"%(target_ft, str(coord)))

            # cur_df = pd.DataFrame(data={"Prediction":y_pred, "Ground_Truth":y_ground_truth})
            cur_target_ft_df=cur_target_ft_df.sort_values("timestep").reset_index(drop=True)

            cur_y_pred = cur_target_ft_df["pred"].values
            cur_y_ground_truth = cur_target_ft_df["actual"].values

            # cur_y_pred = y_pred[idx]
            # cur_y_ground_truth = y_ground_truth[idx]

            target_no_underscore = target_ft.replace("_"," ")
            title_tag = "%s %s"%(infoID, target_no_underscore)
            title = ("\n".join(wrap(title_tag, 20)))
            # title = '%s \n%s'%(infoID, target_ft)

            if num_target_fts == 6:
                x_coor = coord[0]
                y_coor = coord[1]
                print("\ncur_y_pred shape")
                print(cur_y_pred.shape)
                print("\ncur_y_ground_truth shape")
                print(cur_y_ground_truth.shape)
                axs[x_coor,y_coor].plot(cur_y_pred,":r" ,label="Prediction")
                axs[x_coor,y_coor].plot(cur_y_ground_truth, "-k" ,label="Ground Truth")
                axs[x_coor,y_coor].set_title(title)
            else:
                # x_coor = coord[0]
                # y_coor = coord[1]
                print("\ncur_y_pred shape")
                print(cur_y_pred.shape)
                print("\ncur_y_ground_truth shape")
                print(cur_y_ground_truth.shape)
                axs[coord].plot(cur_y_pred,":r" ,label="Prediction")
                axs[coord].plot(cur_y_ground_truth, "-k" ,label="Ground Truth")
                axs[coord].set_title(title)

            # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            # plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))
            idx+=1

        for ax in axs.flat:
            ax.set(xlabel='days', ylabel='Volume')

        plt.tight_layout()
        output_fp = output_dir + "%s.png"%( hyp_infoID)
        fig.savefig(output_fp)
        plt.close()
        print(output_fp)


def get_selected_models_param_dict(all_models_df,selected_models,overwrite_params_dict):

    cur_dict_idx = 0
    idx_to_sample_generator_param_dict = {}

    selected_models_df = all_models_df[all_models_df["model_tag"].isin(selected_models)].reset_index(drop=True)
    print("\nselected_models_df")
    print(selected_models_df)

    data_param_num_combos = selected_models_df.shape[0]
    print("\ndata_param_num_combos")
    print(data_param_num_combos)

    model_tag_to_dir_dict = convert_df_2_cols_to_dict(selected_models_df, "model_tag", "model_dir")
    for model_tag,model_dir in model_tag_to_dir_dict.items():
        print("%s: %s"%(model_tag , model_dir))

        #current param fp
        param_fp = model_dir + "full-params.csv"
        param_df = pd.read_csv(param_fp)
        print(param_df)

        if "Unnamed: 0" in list(param_df):
            param_df = param_df.drop("Unnamed: 0",axis=1)

        input_param_dict = convert_df_2_cols_to_dict(param_df, "param", "value")
        print("\ninput_param_dict")
        print(input_param_dict)
        output_param_dict = {}

        try:
            output_param_dict["INPUT_HOURS"] = int(input_param_dict["INPUT_HOURS"])
        except KeyError:
            output_param_dict["INPUT_HOURS"] = int(input_param_dict[" INPUT_HOURS"])


        output_param_dict["train_start"] = input_param_dict["train_start"]
        output_param_dict["train_end"] = input_param_dict["train_end"]

        output_param_dict["val_start"] = input_param_dict["val_start"]
        output_param_dict["val_end"] = input_param_dict["val_end"]

        output_param_dict["test_start"] = input_param_dict["test_start"]
        output_param_dict["test_end"] = input_param_dict["test_end"]

        output_param_dict["GRAN"] = input_param_dict["GRAN"]
        output_param_dict["OUTPUT_HOURS"] = int(input_param_dict["OUTPUT_HOURS"])
        output_param_dict["GET_1HOT_INFO_ID_FTS"] =  bool_str_to_bool(input_param_dict["GET_1HOT_INFO_ID_FTS"])
        output_param_dict["GET_GDELT_FEATURES"] =  bool_str_to_bool(input_param_dict["GET_GDELT_FEATURES"])
        output_param_dict["GET_REDDIT_FEATURES"] =  bool_str_to_bool(input_param_dict["GET_REDDIT_FEATURES"])
        output_param_dict["TARGET_PLATFORM"] = input_param_dict["TARGET_PLATFORM"]
        output_param_dict["GET_EXTERNAL_PLATFORM_FTS"] = bool_str_to_bool(input_param_dict["GET_EXTERNAL_PLATFORM_FTS"])
        output_param_dict["RESCALE"] = bool_str_to_bool(input_param_dict["RESCALE"])
        output_param_dict["RESCALE_TARGET"] = bool_str_to_bool(input_param_dict["RESCALE_TARGET"])
        output_param_dict["SCALER_TYPE"] = input_param_dict["SCALER_TYPE"]
        output_param_dict["FEATURE_RANGE"] = literal_eval(input_param_dict["FEATURE_RANGE"])
        output_param_dict["INPUT_TWITTER_LOGNORM"] = int(input_param_dict["INPUT_TWITTER_LOGNORM"])
        output_param_dict["OUTPUT_TWITTER_LOGNORM"] = int(input_param_dict["OUTPUT_TWITTER_LOGNORM"])
        output_param_dict["INPUT_OTHER_LOGNORM"] = int(input_param_dict["INPUT_OTHER_LOGNORM"])
        output_param_dict["OUTPUT_OTHER_LOGNORM"] = int(input_param_dict["OUTPUT_OTHER_LOGNORM"])
        output_param_dict["LOGNORM_IGNORE_TAGS"] = literal_eval(input_param_dict["LOGNORM_IGNORE_TAGS"])
        output_param_dict["CORRELATION_FUNCS_TO_USE_LIST"] = literal_eval(input_param_dict["CORRELATION_FUNCS_TO_USE_LIST"])
        output_param_dict["n_jobs"] = int(input_param_dict["n_jobs"])

        output_param_dict.update(overwrite_params_dict)
        idx_to_sample_generator_param_dict[cur_dict_idx]=output_param_dict
        cur_dict_idx+=1

    return idx_to_sample_generator_param_dict, data_param_num_combos

def get_selected_models_param_dict_v2(all_models_df,selected_models,overwrite_params_dict):

    cur_dict_idx = 0
    idx_to_data_and_model_materials_dict = {}

    selected_models_df = all_models_df[all_models_df["model_tag"].isin(selected_models)].reset_index(drop=True)
    print("\nselected_models_df")
    print(selected_models_df)

    data_param_num_combos = selected_models_df.shape[0]
    print("\ndata_param_num_combos")
    print(data_param_num_combos)

    model_tag_to_dir_dict = convert_df_2_cols_to_dict(selected_models_df, "model_tag", "model_dir")
    for model_tag,model_dir in model_tag_to_dir_dict.items():
        print("%s: %s"%(model_tag , model_dir))

        #current param fp
        param_fp = model_dir + "full-params.csv"
        param_df = pd.read_csv(param_fp)
        print(param_df)

        if "Unnamed: 0" in list(param_df):
            param_df = param_df.drop("Unnamed: 0",axis=1)

        input_param_dict = convert_df_2_cols_to_dict(param_df, "param", "value")
        print("\ninput_param_dict")
        print(input_param_dict)
        sample_generator_param_dict = {}

        try:
            sample_generator_param_dict["INPUT_HOURS"] = int(input_param_dict["INPUT_HOURS"])
        except KeyError:
            sample_generator_param_dict["INPUT_HOURS"] = int(input_param_dict[" INPUT_HOURS"])


        sample_generator_param_dict["train_start"] = input_param_dict["train_start"]
        sample_generator_param_dict["train_end"] = input_param_dict["train_end"]

        sample_generator_param_dict["val_start"] = input_param_dict["val_start"]
        sample_generator_param_dict["val_end"] = input_param_dict["val_end"]

        sample_generator_param_dict["test_start"] = input_param_dict["test_start"]
        sample_generator_param_dict["test_end"] = input_param_dict["test_end"]

        sample_generator_param_dict["GRAN"] = input_param_dict["GRAN"]
        sample_generator_param_dict["OUTPUT_HOURS"] = int(input_param_dict["OUTPUT_HOURS"])
        sample_generator_param_dict["GET_1HOT_INFO_ID_FTS"] =  bool_str_to_bool(input_param_dict["GET_1HOT_INFO_ID_FTS"])
        sample_generator_param_dict["GET_GDELT_FEATURES"] =  bool_str_to_bool(input_param_dict["GET_GDELT_FEATURES"])
        sample_generator_param_dict["GET_REDDIT_FEATURES"] =  bool_str_to_bool(input_param_dict["GET_REDDIT_FEATURES"])
        sample_generator_param_dict["TARGET_PLATFORM"] = input_param_dict["TARGET_PLATFORM"]
        sample_generator_param_dict["GET_EXTERNAL_PLATFORM_FTS"] = bool_str_to_bool(input_param_dict["GET_EXTERNAL_PLATFORM_FTS"])
        sample_generator_param_dict["RESCALE"] = bool_str_to_bool(input_param_dict["RESCALE"])
        sample_generator_param_dict["RESCALE_TARGET"] = bool_str_to_bool(input_param_dict["RESCALE_TARGET"])
        sample_generator_param_dict["SCALER_TYPE"] = input_param_dict["SCALER_TYPE"]
        sample_generator_param_dict["FEATURE_RANGE"] = literal_eval(input_param_dict["FEATURE_RANGE"])
        sample_generator_param_dict["INPUT_TWITTER_LOGNORM"] = int(input_param_dict["INPUT_TWITTER_LOGNORM"])
        sample_generator_param_dict["OUTPUT_TWITTER_LOGNORM"] = int(input_param_dict["OUTPUT_TWITTER_LOGNORM"])
        sample_generator_param_dict["INPUT_OTHER_LOGNORM"] = int(input_param_dict["INPUT_OTHER_LOGNORM"])
        sample_generator_param_dict["OUTPUT_OTHER_LOGNORM"] = int(input_param_dict["OUTPUT_OTHER_LOGNORM"])
        sample_generator_param_dict["LOGNORM_IGNORE_TAGS"] = literal_eval(input_param_dict["LOGNORM_IGNORE_TAGS"])
        sample_generator_param_dict["CORRELATION_FUNCS_TO_USE_LIST"] = literal_eval(input_param_dict["CORRELATION_FUNCS_TO_USE_LIST"])
        sample_generator_param_dict["n_jobs"] = int(input_param_dict["n_jobs"])

        sample_generator_param_dict.update(overwrite_params_dict)
        # idx_to_sample_generator_param_dict[cur_dict_idx]=sample_generator_param_dict

        # all_model_params = [
        # "objective","colsample_bytree","learning_rate","max_depth","n_estimators",
        #   "subsample","gamma","alpha","reg_lambda","random_state","n_jobs"
        #   ]

        int_params = ["max_depth","n_estimators","random_state","n_jobs"]
        float_params = ["colsample_bytree","learning_rate",
            "subsample","gamma","alpha","reg_lambda"]

        model_param_dict = {}
        model_param_dict["objective"] = input_param_dict["objective"]
        for param in int_params:
            model_param_dict[param] = int(input_param_dict[param])
        for param in float_params:
            model_param_dict[param] = float(input_param_dict[param])

        idx_to_data_and_model_materials_dict[cur_dict_idx] =( sample_generator_param_dict , model_param_dict ,model_tag)

        cur_dict_idx+=1

    return idx_to_data_and_model_materials_dict


def get_idx_to_sample_generator_param_dict_v4_darpa_format_option( DEBUG,train_start,train_end,val_start,
    val_end,test_start,test_end,n_jobs,GRAN,
    main_input_dir,infoIDs,platforms,user_statuses,LOGNORM_IGNORE_TAGS,LOGNORM_DEBUG_PRINT,
    RESCALE_LIST,SCALER_TYPE_LIST,FEATURE_RANGE_LIST,RESCALE_TARGET_LIST,INPUT_TWITTER_LOGNORM_LIST,OUTPUT_TWITTER_LOGNORM_LIST,
    INPUT_OTHER_LOGNORM_LIST,OUTPUT_OTHER_LOGNORM_LIST,TARGET_PLATFORM_LIST,INPUT_HOURS_LIST,OUTPUT_HOURS_LIST,GET_GDELT_FEATURES_LIST,
    GET_REDDIT_FEATURES_LIST,GET_1HOT_INFO_ID_FTS_LIST, GET_EXTERNAL_PLATFORM_FTS_LIST,main_output_dir,CORRELATION_FUNCS_TO_USE_LIST_LIST,
    CORRELATION_STR_TO_FUNC_DICT,AGG_TIME_SERIES_LIST, DARPA_INPUT_FORMAT_LIST):

    print("\nGetting sample gen params...")
    idx_to_sample_generator_param_dict = {}



    if DEBUG==True:
        INPUT_HOURS_LIST=INPUT_HOURS_LIST[:1]
        OUTPUT_HOURS_LIST=OUTPUT_HOURS_LIST[:1]
        GET_GDELT_FEATURES_LIST=GET_GDELT_FEATURES_LIST[:1]
        GET_REDDIT_FEATURES_LIST=GET_REDDIT_FEATURES_LIST[:1]
        GET_1HOT_INFO_ID_FTS_LIST=GET_1HOT_INFO_ID_FTS_LIST[:1]
        GET_EXTERNAL_PLATFORM_FTS_LIST=GET_EXTERNAL_PLATFORM_FTS_LIST[:1]
        TARGET_PLATFORM_LIST=TARGET_PLATFORM_LIST[:1]
        RESCALE_LIST=RESCALE_LIST[:1]
        SCALER_TYPE_LIST= SCALER_TYPE_LIST[:1]
        FEATURE_RANGE_LIST= FEATURE_RANGE_LIST[:1]
        RESCALE_TARGET_LIST= RESCALE_TARGET_LIST[:1]
        INPUT_TWITTER_LOGNORM_LIST =INPUT_TWITTER_LOGNORM_LIST[:1]
        OUTPUT_TWITTER_LOGNORM_LIST = OUTPUT_TWITTER_LOGNORM_LIST[:1]
        INPUT_OTHER_LOGNORM_LIST = INPUT_OTHER_LOGNORM_LIST[:1]
        OUTPUT_OTHER_LOGNORM_LIST = OUTPUT_OTHER_LOGNORM_LIST[:1]
        CORRELATION_FUNCS_TO_USE_LIST_LIST=CORRELATION_FUNCS_TO_USE_LIST_LIST[:1]


    num_combos=0
    for CORRELATION_FUNCS_TO_USE_LIST in CORRELATION_FUNCS_TO_USE_LIST_LIST:
        for INPUT_HOURS in INPUT_HOURS_LIST:
            for OUTPUT_HOURS in OUTPUT_HOURS_LIST:
                for GET_GDELT_FEATURES in GET_REDDIT_FEATURES_LIST:
                    for GET_REDDIT_FEATURES in GET_REDDIT_FEATURES_LIST:
                        for GET_1HOT_INFO_ID_FTS in GET_1HOT_INFO_ID_FTS_LIST:
                            for GET_EXTERNAL_PLATFORM_FTS in GET_EXTERNAL_PLATFORM_FTS_LIST:

                                for TARGET_PLATFORM in TARGET_PLATFORM_LIST:
                                    if (GET_EXTERNAL_PLATFORM_FTS==False) and (TARGET_PLATFORM =="all"):
                                        break
                                    else:
                                        # print("continuing loop...")
                                        for RESCALE in RESCALE_LIST:
                                            for SCALER_TYPE in SCALER_TYPE_LIST:
                                                for FEATURE_RANGE in FEATURE_RANGE_LIST:
                                                    for RESCALE_TARGET in RESCALE_TARGET_LIST:
                                                        for INPUT_TWITTER_LOGNORM in INPUT_TWITTER_LOGNORM_LIST:
                                                            for OUTPUT_TWITTER_LOGNORM in OUTPUT_TWITTER_LOGNORM_LIST:
                                                                for INPUT_OTHER_LOGNORM in INPUT_OTHER_LOGNORM_LIST:
                                                                    for OUTPUT_OTHER_LOGNORM in OUTPUT_OTHER_LOGNORM_LIST:
                                                                        for AGG_TIME_SERIES in AGG_TIME_SERIES_LIST:
                                                                            for DARPA_INPUT_FORMAT in DARPA_INPUT_FORMAT_LIST:
                                                                                cur_sample_gen_param_dict = {}
                                                                                # basic_sample_gen_params = {}
                                                                                # cur_sample_gen_param_dict["DEBUG"]=DEBUG
                                                                                cur_sample_gen_param_dict["train_start"]=train_start
                                                                                cur_sample_gen_param_dict["train_end"]=train_end
                                                                                cur_sample_gen_param_dict["val_start"]=val_start
                                                                                cur_sample_gen_param_dict["val_end"]=val_end
                                                                                cur_sample_gen_param_dict["test_start"]=test_start
                                                                                cur_sample_gen_param_dict["test_end"]=test_end
                                                                                # cur_sample_gen_param_dict["sum_fts"]=sum_fts
                                                                                # cur_sample_gen_param_dict["avg_fts"]=avg_fts
                                                                                cur_sample_gen_param_dict["n_jobs"]=n_jobs
                                                                                cur_sample_gen_param_dict["GRAN"]=GRAN
                                                                                cur_sample_gen_param_dict["main_input_dir"]=main_input_dir
                                                                                cur_sample_gen_param_dict["main_output_dir"]=main_output_dir
                                                                                cur_sample_gen_param_dict["infoIDs"]=infoIDs
                                                                                cur_sample_gen_param_dict["platforms"]=platforms
                                                                                cur_sample_gen_param_dict["user_statuses"]=user_statuses
                                                                                cur_sample_gen_param_dict["LOGNORM_IGNORE_TAGS"]=LOGNORM_IGNORE_TAGS
                                                                                cur_sample_gen_param_dict["LOGNORM_DEBUG_PRINT"]=LOGNORM_DEBUG_PRINT
                                                                                cur_sample_gen_param_dict["INPUT_HOURS"]=INPUT_HOURS
                                                                                cur_sample_gen_param_dict["OUTPUT_HOURS"]=OUTPUT_HOURS
                                                                                cur_sample_gen_param_dict["GET_GDELT_FEATURES"]=GET_GDELT_FEATURES
                                                                                cur_sample_gen_param_dict["GET_REDDIT_FEATURES"]=GET_REDDIT_FEATURES
                                                                                cur_sample_gen_param_dict["GET_1HOT_INFO_ID_FTS"]=GET_1HOT_INFO_ID_FTS
                                                                                cur_sample_gen_param_dict["GET_EXTERNAL_PLATFORM_FTS"]=GET_EXTERNAL_PLATFORM_FTS
                                                                                cur_sample_gen_param_dict["TARGET_PLATFORM"]=TARGET_PLATFORM
                                                                                cur_sample_gen_param_dict["RESCALE"]=RESCALE
                                                                                cur_sample_gen_param_dict["SCALER_TYPE"]=SCALER_TYPE
                                                                                cur_sample_gen_param_dict["FEATURE_RANGE"]=FEATURE_RANGE
                                                                                cur_sample_gen_param_dict["RESCALE_TARGET"]=RESCALE_TARGET
                                                                                cur_sample_gen_param_dict["INPUT_TWITTER_LOGNORM"]=INPUT_TWITTER_LOGNORM
                                                                                cur_sample_gen_param_dict["OUTPUT_TWITTER_LOGNORM"]=OUTPUT_TWITTER_LOGNORM
                                                                                cur_sample_gen_param_dict["INPUT_OTHER_LOGNORM"]=INPUT_OTHER_LOGNORM
                                                                                cur_sample_gen_param_dict["OUTPUT_OTHER_LOGNORM"]=OUTPUT_OTHER_LOGNORM
                                                                                cur_sample_gen_param_dict["CORRELATION_FUNCS_TO_USE_LIST"]=CORRELATION_FUNCS_TO_USE_LIST
                                                                                cur_sample_gen_param_dict["CORRELATION_STR_TO_FUNC_DICT"]=CORRELATION_STR_TO_FUNC_DICT
                                                                                cur_sample_gen_param_dict["AGG_TIME_SERIES"]=AGG_TIME_SERIES
                                                                                cur_sample_gen_param_dict["DARPA_INPUT_FORMAT"] = DARPA_INPUT_FORMAT
                                                                                idx_to_sample_generator_param_dict[num_combos]=dict(cur_sample_gen_param_dict)
                                                                                num_combos+=1
    print("\nnum_combos: %d"%(num_combos+1))


    return idx_to_sample_generator_param_dict, num_combos

def resolve_count_conflicts(pred_df, new_user_count_target_col,old_user_count_target_col,num_actions_target_col,conflict_resolution_option):

    infoIDs = list(pred_df["infoID"].unique())
    target_fts = list(pred_df["target_ft"].unique())

    #========================== first split up pred df to a better form ==========================
    infoID_to_pred_df = {}
    for infoID in infoIDs:
        all_cur_target_dfs = []
        temp_pred_df = pred_df[pred_df["infoID"]==infoID]

        keep_cols = ["timestep", "infoID", "pred"]

        for target_ft in target_fts:

            temp2 = temp_pred_df[temp_pred_df["target_ft"]==target_ft]
            temp2 = temp2[keep_cols]
            temp2=temp2.rename(columns={"pred": target_ft})
            temp2[target_ft] = np.round(temp2[target_ft], 0)

            # temp2 = temp2.drop("target_ft", axis=1)
            all_cur_target_dfs.append(temp2.copy())

        cur_pred_df = merge_mult_dfs(all_cur_target_dfs, on=["timestep", "infoID"], how="inner")



        print("\ncur_pred_df")
        print(cur_pred_df)

        infoID_to_pred_df[infoID] = cur_pred_df

    #========================== insert count stats ==========================
    print("\nInserting count stats...")
    for infoID in infoIDs:
        cur_pred_df = infoID_to_pred_df[infoID]


        #rename cols for easier debugging
        cur_pred_df = cur_pred_df.rename(columns={new_user_count_target_col: "num_new_users"})
        cur_pred_df = cur_pred_df.rename(columns={old_user_count_target_col: "num_old_users"})
        cur_pred_df = cur_pred_df.rename(columns={num_actions_target_col: "num_actions"})

        print("\ncur_pred_df")
        print(cur_pred_df)

        cur_pred_df["num_total_users"] = cur_pred_df["num_new_users"] + cur_pred_df["num_old_users"]
        cur_pred_df["old_user_freq"] = cur_pred_df["num_old_users"]/(cur_pred_df["num_total_users"])
        cur_pred_df["new_user_freq"] = 1 - cur_pred_df["old_user_freq"]
        print(cur_pred_df)

        infoID_to_pred_df[infoID] = cur_pred_df

    #========================== resolve conflicts ==========================
    conflict_stats_dict = {}
    valid_and_invalid_preds_dict = {}
    print("\nResolving conflicts...")
    for infoID in infoIDs:
        cur_pred_df = infoID_to_pred_df[infoID]

        #fix zero action conflict
        zero_users_nonzero_actions_df  = cur_pred_df[(cur_pred_df["num_total_users"]==0) & (cur_pred_df["num_actions"]>0)].reset_index(drop=True)
        print("\nzero_users_nonzero_actions_df")
        print(zero_users_nonzero_actions_df)
        unique_timesteps = zero_users_nonzero_actions_df["timestep"].unique()
        opp_zero_users_nonzero_actions_df=cur_pred_df[~cur_pred_df["timestep"].isin(unique_timesteps)].reset_index(drop=True)
        print("\nopp_zero_users_nonzero_actions_df")
        print(opp_zero_users_nonzero_actions_df)
        zero_users_nonzero_actions_df["num_new_users"] = zero_users_nonzero_actions_df["num_actions"].copy()
        zero_users_nonzero_actions_df["old_user_freq"]=0
        zero_users_nonzero_actions_df["new_user_freq"]=1
        cur_pred_df = pd.concat([zero_users_nonzero_actions_df,opp_zero_users_nonzero_actions_df])
        cur_pred_df = cur_pred_df.sort_values("timestep").reset_index(drop=True)
        print("\ncur_pred_df")
        print(cur_pred_df)

        # if zero_users_nonzero_actions_df.shape[0] > 0:
        #     sys.exit(0)

        #valid records
        valid_pred_df = cur_pred_df[cur_pred_df["num_total_users"]<=cur_pred_df["num_actions"]].reset_index(drop=True)
        num_valid_records = valid_pred_df.shape[0]

        #invalid records
        invalid_pred_df = cur_pred_df[cur_pred_df["num_total_users"]>cur_pred_df["num_actions"]].reset_index(drop=True)
        num_invalid_records = invalid_pred_df.shape[0]



        valid_and_invalid_preds_dict[infoID] = {}
        valid_and_invalid_preds_dict[infoID]["valid_pred_df"] = valid_pred_df
        valid_and_invalid_preds_dict[infoID]["invalid_pred_df"] = invalid_pred_df

        conflict_stats_dict[infoID]={}
        conflict_stats_dict[infoID]["num_valid_records"]=num_valid_records
        conflict_stats_dict[infoID]["num_invalid_records"]=num_invalid_records

    #========================== conflict stats ==========================
    print("\nConflict record stats")
    for infoID in infoIDs:

        num_valid_records = conflict_stats_dict[infoID]["num_valid_records"]
        num_invalid_records = conflict_stats_dict[infoID]["num_invalid_records"]

        print("For %s: %d valid records, %d invalid records"%(infoID, num_valid_records, num_invalid_records))

    #========================== resolve conflicts ==========================
    print("\nResolving conflicts...")
    resolved_pred_df_dict = {}
    for infoID in infoIDs:

        #invalid records
        invalid_pred_df = valid_and_invalid_preds_dict[infoID]["invalid_pred_df"]

        print()
        print(infoID)


        invalid_pred_df["diff"] = invalid_pred_df["num_total_users"] - invalid_pred_df["num_actions"]

        print("\ninvalid_pred_df")
        print(invalid_pred_df)

        if conflict_resolution_option == "upsample_actions_to_users":
            invalid_pred_df["num_actions"] = invalid_pred_df["num_actions"] + invalid_pred_df["diff"]
            resolved_pred_df = invalid_pred_df.copy()
            resolved_pred_df["diff"] = resolved_pred_df["num_actions"] - resolved_pred_df["num_total_users"]
            diff_sum = resolved_pred_df["diff"].sum()
            if diff_sum == 0:
                print("\n%s: conflict resolved!"%infoID)
                print(resolved_pred_df)
            else:
                print("error! conflict not resolved")
                print("diff sum: %d"%diff_sum)
                error_df = resolved_pred_df[resolved_pred_df["diff"]>0]
                print(error_df)
                sys.exit(0)
        else:
            #downsample users to actions
            invalid_pred_df["num_total_users"] = invalid_pred_df["num_total_users"] - invalid_pred_df["diff"]
            invalid_pred_df["num_new_users"] = invalid_pred_df["num_total_users"] * invalid_pred_df["new_user_freq"]
            invalid_pred_df["num_new_users"] = np.round(invalid_pred_df["num_new_users"], 0)
            invalid_pred_df["num_old_users"] = invalid_pred_df["num_total_users"] - invalid_pred_df["num_new_users"]


            invalid_pred_df["num_old_users"] = np.round(invalid_pred_df["num_old_users"], 0)


            invalid_pred_df["diff"] =invalid_pred_df["num_new_users"] + invalid_pred_df["num_old_users"] - invalid_pred_df["num_actions"]
            print(invalid_pred_df)

            diff_sum = invalid_pred_df["diff"].sum()
            print(invalid_pred_df)

            resolved_pred_df=invalid_pred_df.copy()
            if diff_sum == 0:
                print("\n%s: conflict resolved!"%infoID)

            else:
                print("error! conflict not resolved")
                print("diff sum: %d"%diff_sum)
                error_df = resolved_pred_df[resolved_pred_df["diff"]>0]
                print(error_df)
                sys.exit(0)


        resolved_pred_df_dict[infoID]=resolved_pred_df

    #========================== combine with valid dfs ==========================
    cleaned_pred_df_dict = {}
    keep_cols = ["timestep" ,"num_new_users", "num_old_users", "num_actions"]
    for infoID in infoIDs:

        resolved_pred_df = resolved_pred_df_dict[infoID][keep_cols]
        valid_pred_df = valid_and_invalid_preds_dict[infoID]["valid_pred_df"][keep_cols]
        cleaned_pred_df = pd.concat([resolved_pred_df, valid_pred_df])
        cleaned_pred_df = cleaned_pred_df.sort_values("timestep").reset_index(drop=True)
        cleaned_pred_df_dict[infoID]=cleaned_pred_df

        print("\ncleaned_pred_df")
        print(cleaned_pred_df)

        #double check for any bugs
        diff_series = cleaned_pred_df["num_actions"] - cleaned_pred_df["num_new_users"]- cleaned_pred_df["num_old_users"]
        neg_diff_series = diff_series[diff_series<0]
        print(neg_diff_series)
        print("\nnum_neg_diff_records")
        num_neg_diff_records = neg_diff_series.shape[0]
        print(num_neg_diff_records)

        if num_neg_diff_records > 0:
            print("\nerror! num_neg_diff_records")
            sys.exit(0)

    print("\nDone cleaning preds!")

    return cleaned_pred_df_dict

def verify_pred_counts_so_far(sampled_new_user_records,sampled_old_user_records,expected_num_old_users, expected_num_new_users):

    print("\nCheckpoint so far...")
    actual_old_users = sampled_old_user_records["nodeUserID"].nunique()
    actual_new_users = sampled_new_user_records["nodeUserID"].nunique()

    print("\nexpected_num_old_users: %d"%expected_num_old_users)
    print("expected_num_new_users: %d"%expected_num_new_users)

    print("\nactual_old_users: %d"%actual_old_users)
    print("actual_new_users: %d"%actual_new_users)

    if expected_num_old_users != actual_old_users:
        print("\nError! expected_num_old_users != actual_old_users ")
        sys.exit(0)

    if expected_num_new_users != actual_new_users:
        print("\nError! expected_num_new_users != actual_new_users ")
        sys.exit(0)

    print("\nCounts are ok so far...")



    return

def verify_pred_counts_so_far_v2_debug_print(sampled_new_user_records,sampled_old_user_records,expected_num_old_users, expected_num_new_users, DEBUG_PRINT):

    debug_print("\nCheckpoint so far...")
    actual_old_users = sampled_old_user_records["nodeUserID"].nunique()
    actual_new_users = sampled_new_user_records["nodeUserID"].nunique()

    debug_print("\nexpected_num_old_users: %d"%expected_num_old_users)
    debug_print("expected_num_new_users: %d"%expected_num_new_users)

    debug_print("\nactual_old_users: %d"%actual_old_users)
    debug_print("actual_new_users: %d"%actual_new_users)

    if expected_num_old_users != actual_old_users:

        print("\nError! expected_num_old_users != actual_old_users ")
        print(expected_num_old_users)
        print(actual_old_users)
        sys.exit(0)

    if expected_num_new_users != actual_new_users:
        print("\nError! expected_num_new_users != actual_new_users ")
        print(expected_num_new_users)
        print(actual_new_users)
        sys.exit(0)

    debug_print("\nCounts are ok so far...")



    return

def verify_pred_counts(cur_infoID_df, cur_infoID_count_df_for_verif ):

    expected_num_new_users = cur_infoID_count_df_for_verif["num_new_users"].sum()
    expected_num_old_users = cur_infoID_count_df_for_verif["num_old_users"].sum()
    expected_num_actions = cur_infoID_count_df_for_verif["num_actions"].sum()

    acutal_num_new_users = cur_infoID_df[cur_infoID_df["nodeUserID_is_new"]==1].shape[0]
    acutal_num_old_users = cur_infoID_df[cur_infoID_df["nodeUserID_is_new"]==0].shape[0]
    actual_num_actions = cur_infoID_df.shape[0]

    print("\nexpected_num_new_users: %d"%expected_num_new_users)
    print("expected_num_old_users: %d"%expected_num_old_users)
    print("expected_num_actions: %d"%expected_num_actions)

    print("\nacutal_num_new_users: %d"%acutal_num_new_users)
    print("acutal_num_old_users: %d"%acutal_num_old_users)
    print("actual_num_actions: %d"%actual_num_actions)

    if expected_num_new_users != acutal_num_new_users:
        print("\nError! expected_num_new_users != acutal_num_new_users")
        sys.exit(0)

    if expected_num_old_users != acutal_num_old_users:
        print("\nError! expected_num_old_users != acutal_num_old_users")
        sys.exit(0)

    if expected_num_actions != acutal_num_actions:
        print("\nError! expected_num_actions != acutal_num_actions")
        sys.exit(0)

    print("\nCOUNTS ARE OK! cur_infoID_df AND cur_infoID_count_df_for_verif MATCH!")

    return


def load_full_model_weight_df(model_weight_dir,platform,infoIDs,hyp_dict,weight_start, weight_end):

    #load full model weight df
    print("\nLoading full model weight df for %s to %s..."%(weight_start, weight_end))
    cur_platform_model_weight_dir = model_weight_dir + platform + "/"
    all_model_weight_dfs = []
    for infoID in infoIDs:
        hyp_infoID = hyp_dict[infoID]
        print("\nLoading %s weight df..."%infoID)

        model_weight_fp = cur_platform_model_weight_dir + hyp_infoID + ".csv"
        model_weight_df = pd.read_csv(model_weight_fp)
        model_weight_df["nodeTime"] = pd.to_datetime(model_weight_df["nodeTime"], utc=True)
        model_weight_df = config_df_by_dates(model_weight_df, weight_start, weight_end, "nodeTime")
        # model_weight_df["infoID"]=infoID

        #we need user influence


        all_model_weight_dfs.append(model_weight_df)

    model_weight_df = pd.concat(all_model_weight_dfs).reset_index(drop=True)
    print("\nFull model_weight_df")
    print(model_weight_df)

    cols = list(model_weight_df)
    print("\nModel weight cols")
    for col in cols:
        print(col)

    return model_weight_df

def get_user_influence_from_parent_influence(model_weight_df,INFLUENCE_EPSILON):
    model_weight_df["infoID_user"] = model_weight_df["nodeUserID"] + "_" + model_weight_df["informationID"]
    model_weight_df["infoID_parent"] = model_weight_df["parentUserID"] + "_" + model_weight_df["informationID"]


    parent_to_influence_dict = convert_df_2_cols_to_dict(model_weight_df, "infoID_parent", "parentUserID_overall_influence_weight")
    model_weight_df["nodeUserID_overall_influence_weight"] = model_weight_df["infoID_user"].map(parent_to_influence_dict)
    model_weight_df["nodeUserID_overall_influence_weight"] = model_weight_df["nodeUserID_overall_influence_weight"].fillna(INFLUENCE_EPSILON)

    print(model_weight_df["nodeUserID_overall_influence_weight"].value_counts())

    model_weight_df = model_weight_df.drop(["infoID_parent", "infoID_user"], axis=1)

    return model_weight_df

def create_user_weight_table(history_record_df,infoID_user_to_latest_parent_dict,INFLUENCE_EPSILON):

    print("\nGetting overall action and influence weights...")
    groupby_add_cols = ["nodeUserID", "parentUserID", "informationID"]

    keep_cols = ["nodeUserID", "parentUserID", "informationID","nodeUserID_num_actions_this_timestep","parentUserID_influence_this_timestep"]
    user_weight_table = history_record_df[keep_cols]

    user_weight_table["parentUserID_overall_influence_weight"] = user_weight_table.groupby(["parentUserID", "informationID"])["parentUserID_influence_this_timestep"].transform("sum")
    user_weight_table = get_user_influence_from_parent_influence(user_weight_table,INFLUENCE_EPSILON)

    parent_influence_weight_table = user_weight_table[["parentUserID","informationID","parentUserID_overall_influence_weight"]].drop_duplicates().reset_index(drop=True)


    user_weight_table["nodeUserID_overall_action_weight"] = user_weight_table.groupby(["nodeUserID", "informationID"])["nodeUserID_num_actions_this_timestep"].transform("sum")
    user_weight_table = user_weight_table.drop_duplicates().reset_index(drop=True)

    user_weight_table = user_weight_table.drop(["nodeUserID_num_actions_this_timestep","parentUserID_influence_this_timestep", "parentUserID","parentUserID_overall_influence_weight"], axis=1)
    user_weight_table = user_weight_table.drop_duplicates().reset_index(drop=True)



    print("\nuser_weight_table")
    print(user_weight_table)
    user_weight_table["infoID_user"] = user_weight_table["nodeUserID"] + "_" + user_weight_table["informationID"]
    user_weight_table["latest_parent"] = user_weight_table["infoID_user"].map(infoID_user_to_latest_parent_dict)
    user_weight_table = user_weight_table.drop("infoID_user", axis=1)
    print("\nuser_weight_table with latest parents")
    print(user_weight_table)

    return user_weight_table,parent_influence_weight_table

def create_user_weight_table_v2_fix_epsilon(history_record_df,infoID_user_to_latest_parent_dict,INFLUENCE_EPSILON_DIV_VAL):

    print("\nGetting overall action and influence weights...")
    groupby_add_cols = ["nodeUserID", "parentUserID", "informationID"]

    keep_cols = ["nodeUserID", "parentUserID", "informationID","nodeUserID_num_actions_this_timestep","parentUserID_influence_this_timestep"]
    user_weight_table = history_record_df[keep_cols]

    user_weight_table["parentUserID_overall_influence_weight"] = user_weight_table.groupby(["parentUserID", "informationID"])["parentUserID_influence_this_timestep"].transform("sum")
    user_weight_table = get_user_influence_from_parent_influence_V2_no_fillna(user_weight_table)

    parent_influence_weight_table = user_weight_table[["parentUserID","informationID","parentUserID_overall_influence_weight"]].drop_duplicates().reset_index(drop=True)


    user_weight_table["nodeUserID_overall_action_weight"] = user_weight_table.groupby(["nodeUserID", "informationID"])["nodeUserID_num_actions_this_timestep"].transform("sum")
    user_weight_table = user_weight_table.drop_duplicates().reset_index(drop=True)

    user_weight_table = user_weight_table.drop(["nodeUserID_num_actions_this_timestep","parentUserID_influence_this_timestep", "parentUserID","parentUserID_overall_influence_weight"], axis=1)
    user_weight_table = user_weight_table.drop_duplicates().reset_index(drop=True)




    #ep1
    user_weight_table["nodeUserID_overall_action_weight"] = user_weight_table["nodeUserID_overall_action_weight"]/user_weight_table["nodeUserID_overall_action_weight"].sum()
    min_val = user_weight_table["nodeUserID_overall_action_weight"].min()
    EPSILON = min_val/INFLUENCE_EPSILON_DIV_VAL
    print("\nEPSILON")
    print(EPSILON)
    user_weight_table["nodeUserID_overall_action_weight"] = user_weight_table["nodeUserID_overall_action_weight"].fillna(EPSILON)

    #ep2
    user_weight_table["nodeUserID_overall_influence_weight"] = user_weight_table["nodeUserID_overall_influence_weight"]/user_weight_table["nodeUserID_overall_influence_weight"].sum()
    min_val = user_weight_table["nodeUserID_overall_influence_weight"].min()
    EPSILON = min_val/INFLUENCE_EPSILON_DIV_VAL
    print("\nEPSILON")
    print(EPSILON)
    user_weight_table["nodeUserID_overall_influence_weight"] = user_weight_table["nodeUserID_overall_influence_weight"].fillna(EPSILON)

    parent_influence_weight_table["parentUserID_overall_influence_weight"] = parent_influence_weight_table["parentUserID_overall_influence_weight"]/parent_influence_weight_table["parentUserID_overall_influence_weight"].sum()
    min_val = parent_influence_weight_table["parentUserID_overall_influence_weight"].min()
    EPSILON = min_val/INFLUENCE_EPSILON_DIV_VAL
    print("\nEPSILON")
    print(EPSILON)
    parent_influence_weight_table["parentUserID_overall_influence_weight"] = parent_influence_weight_table["parentUserID_overall_influence_weight"].fillna(EPSILON)


    print("\nuser_weight_table")
    print(user_weight_table)

    print("\nparent_influence_weight_table")
    print(parent_influence_weight_table)

    # sys.exit(0)


    print("\nuser_weight_table")
    print(user_weight_table)
    user_weight_table["infoID_user"] = user_weight_table["nodeUserID"] + "_" + user_weight_table["informationID"]
    user_weight_table["latest_parent"] = user_weight_table["infoID_user"].map(infoID_user_to_latest_parent_dict)
    user_weight_table = user_weight_table.drop("infoID_user", axis=1)
    print("\nuser_weight_table with latest parents")
    print(user_weight_table)

    return user_weight_table,parent_influence_weight_table

def get_latest_parent_dict(history_record_df):

    #get latest parent dict
    history_record_df["infoID_user"] = history_record_df["nodeUserID"] + "_" + history_record_df["informationID"]
    history_record_df["nodeTime"] = pd.to_datetime(history_record_df["nodeTime"], utc=True)
    temp_history_record_df = history_record_df[["nodeTime", "parentUserID","infoID_user"]].copy()
    temp_history_record_df = temp_history_record_df.sort_values("nodeTime", ascending=False)
    print("\ntemp_history_record_df so far")
    print(temp_history_record_df)
    temp_history_record_df = temp_history_record_df.drop("nodeTime", axis=1)
    temp_history_record_df = temp_history_record_df.drop_duplicates().reset_index(drop=True)
    print("\ntemp_history_record_df after drop duplicates")
    print(temp_history_record_df)
    infoID_user_to_latest_parent_dict = convert_df_2_cols_to_dict(temp_history_record_df, "infoID_user", "parentUserID")
    history_record_df = history_record_df.drop("infoID_user", axis=1)

    return infoID_user_to_latest_parent_dict



def get_user_influence_from_parent_influence_V2_no_fillna(model_weight_df):
    model_weight_df["infoID_user"] = model_weight_df["nodeUserID"] + "_" + model_weight_df["informationID"]
    model_weight_df["infoID_parent"] = model_weight_df["parentUserID"] + "_" + model_weight_df["informationID"]


    parent_to_influence_dict = convert_df_2_cols_to_dict(model_weight_df, "infoID_parent", "parentUserID_overall_influence_weight")
    model_weight_df["nodeUserID_overall_influence_weight"] = model_weight_df["infoID_user"].map(parent_to_influence_dict)
    # model_weight_df["nodeUserID_overall_influence_weight"] = model_weight_df["nodeUserID_overall_influence_weight"].fillna(INFLUENCE_EPSILON)

    print(model_weight_df["nodeUserID_overall_influence_weight"].value_counts())

    model_weight_df = model_weight_df.drop(["infoID_parent", "infoID_user"], axis=1)

    return model_weight_df

def create_user_weight_and_influence_tables_with_follower_net_info(history_record_df,infoID_user_to_latest_parent_dict,INFLUENCE_EPSILON):

    #come back to this
    # edge_cols = ["nodeUserID", "parentUserID", "informationID","edge_weight_this_timestep"]
    # edge_table= history_record_df[edge_cols]

    # #============= for new wusers ===================

    # new_child_user_history_record_df = history_record_df[history_record_df["nodeUserID_is_new"]==1]
    # print("\nnew_child_user_history_record_df")
    # print(new_child_user_history_record_df)

    # sys.exit(0)




    #============= for other stuff =============

    cols = list(history_record_df)
    for col in cols:
        print(col)
    # sys.exit(0)

    print("\nGetting overall action and influence weights...")
    groupby_add_cols = ["nodeUserID", "parentUserID", "informationID"]

    keep_cols = ["nodeUserID", "parentUserID", "informationID","nodeUserID_num_actions_this_timestep","parentUserID_influence_this_timestep"  ]
    user_weight_table = history_record_df[keep_cols]




    #user actions
    user_weight_table["nodeUserID_overall_action_weight"] = user_weight_table.groupby(["nodeUserID", "informationID"])["nodeUserID_num_actions_this_timestep"].transform("sum")

    #user influence
    user_weight_table["parentUserID_overall_influence_weight"] = user_weight_table.groupby(["parentUserID", "informationID"])["parentUserID_influence_this_timestep"].transform("sum")
    user_weight_table = get_user_influence_from_parent_influence_V2_no_fillna(user_weight_table)

    user_weight_table = user_weight_table.drop(["nodeUserID_num_actions_this_timestep","parentUserID_influence_this_timestep"], axis=1)

    print("\nuser_weight_table")
    print(user_weight_table)

    #include parents into this as well
    users = set(user_weight_table["nodeUserID"])
    parents = set(user_weight_table["parentUserID"])
    parent_set_diff = list(parents.difference(users))
    print("\nparent_set_diff")
    print(parent_set_diff)

    parent_df_to_append = pd.DataFrame(data={"nodeUserID":parent_set_diff, "parentUserID":parent_set_diff})


def create_user_weight_and_influence_tables_with_follower_net_info_v2_user_split(history_record_df,infoID_user_to_latest_parent_dict,EPSILON_DIV_NUM):

    # nodeTime
    # nodeUserID
    # parentUserID
    # platform
    # informationID
    # actionType
    # nodeUserID_num_actions_this_timestep
    # parentUserID_influence_this_timestep
    # edge_weight_this_timestep
    # user_birthdate
    # user_age
    # nodeUserID_is_new
    # parent_birthdate
    # parent_age
    # parentUserID_is_new
    # infoID_user

    temp_history_record_df = history_record_df.copy()
    print("\ntemp_history_record_df")
    print(temp_history_record_df)

    #get the proba that a user acts on a new user
    user_is_new_flags = list(temp_history_record_df["nodeUserID_is_new"])
    temp_history_record_df["nodeUserID_is_old"] = [1 if user_is_new_flag==0 else 0 for user_is_new_flag in user_is_new_flags]
    print(temp_history_record_df[["nodeUserID_is_old","nodeUserID_is_new"]])

    #set parent infl
    temp_history_record_df["parentUserID_new_user_influence"] = temp_history_record_df.groupby(["parentUserID", "informationID"])["nodeUserID_is_new"].transform("sum")
    temp_history_record_df["parentUserID_old_user_influence"] = temp_history_record_df.groupby(["parentUserID", "informationID"])["nodeUserID_is_old"].transform("sum")
    temp_history_record_df["parentUserID_overall_user_influence"] = temp_history_record_df["parentUserID_new_user_influence"] + temp_history_record_df["parentUserID_old_user_influence"]
    print("\ntemp_history_record_df with infl")
    print(temp_history_record_df)

    #set user action probas
    parent_is_new_flags = list(temp_history_record_df["parentUserID_is_new"])
    temp_history_record_df["parentUserID_is_old"] = [1 if parent_is_new_flag==0 else 0 for parent_is_new_flag in parent_is_new_flags]
    print(temp_history_record_df[["parentUserID_is_old","parentUserID_is_new"]])
    temp_history_record_df["nodeUserID_to_new_user_act_proba"] = temp_history_record_df.groupby(["nodeUserID", "informationID"])["parentUserID_is_new"].transform("sum")
    temp_history_record_df["nodeUserID_to_old_user_act_proba"] = temp_history_record_df.groupby(["nodeUserID", "informationID"])["parentUserID_is_old"].transform("sum")
    temp_history_record_df["nodeUserID_overall_act_proba"] =temp_history_record_df["nodeUserID_to_new_user_act_proba"] + temp_history_record_df["nodeUserID_to_old_user_act_proba"]
    print("\ntemp_history_record_df with actions")
    print(temp_history_record_df)

    #make dicts
    user_to_new_infl_dict = convert_df_2_cols_to_dict(temp_history_record_df, "parentUserID", "parentUserID_new_user_influence")
    user_to_old_infl_dict = convert_df_2_cols_to_dict(temp_history_record_df, "parentUserID", "parentUserID_old_user_influence")
    user_to_overall_infl_dict = convert_df_2_cols_to_dict(temp_history_record_df, "parentUserID", "parentUserID_overall_user_influence")

    user_to_new_action_dict = convert_df_2_cols_to_dict(temp_history_record_df, "nodeUserID", "nodeUserID_to_new_user_act_proba")
    user_to_old_action_dict = convert_df_2_cols_to_dict(temp_history_record_df, "nodeUserID", "nodeUserID_to_old_user_act_proba")
    user_to_overall_action_dict = convert_df_2_cols_to_dict(temp_history_record_df, "nodeUserID", "nodeUserID_overall_act_proba")

    #combine nodes
    all_users = list(temp_history_record_df["nodeUserID"].unique()) +  list(temp_history_record_df["parentUserID"].unique())
    all_users= list(set(all_users))

    user_weight_table = pd.DataFrame(data={"nodeUserID":all_users})
    user_weight_table["parentUserID_new_user_influence"] = user_weight_table["nodeUserID"].map(user_to_new_infl_dict)
    user_weight_table["parentUserID_old_user_influence"] = user_weight_table["nodeUserID"].map(user_to_old_infl_dict)
    user_weight_table["parentUserID_overall_user_influence"] = user_weight_table["nodeUserID"].map(user_to_overall_infl_dict)
    user_weight_table["nodeUserID_to_new_user_act_proba"] = user_weight_table["nodeUserID"].map(user_to_new_action_dict)
    user_weight_table["nodeUserID_to_old_user_act_proba"] = user_weight_table["nodeUserID"].map(user_to_old_action_dict)
    user_weight_table["nodeUserID_overall_act_proba"] = user_weight_table["nodeUserID"].map(user_to_overall_action_dict)

    cols = ["nodeUserID", "nodeUserID_to_new_user_act_proba","nodeUserID_to_old_user_act_proba","nodeUserID_overall_act_proba","parentUserID_new_user_influence","parentUserID_old_user_influence","parentUserID_overall_user_influence"]
    user_weight_table = user_weight_table[cols]

    for col in cols:
        if col != "nodeUserID":
            user_weight_table[col] = user_weight_table[col]/user_weight_table[col].sum()
            temp = user_weight_table[col].copy()
            temp=temp[temp>0]
            min_val = temp.min()
            user_weight_table[col]=user_weight_table[col].fillna(0)
            cur_epsilon = min_val/EPSILON_DIV_NUM
            user_weight_table[col] = user_weight_table[col] + cur_epsilon
            print(col)
            print(cur_epsilon)

    print("\nuser_weight_table")
    print(user_weight_table)


    #more stats
    #follower net size
    temp_history_record_df = history_record_df[["nodeUserID", "parentUserID", "informationID"]].drop_duplicates()
    temp_history_record_df["follower_net_size"] = temp_history_record_df.groupby()

    return user_weight_table




def create_user_weight_and_influence_tables_with_follower_net_info_v3_user_split_edge_weight(history_record_df,infoID_user_to_latest_parent_dict,EPSILON_DIV_NUM,infoIDs):

    # nodeTime
    # nodeUserID
    # parentUserID
    # platform
    # informationID
    # actionType
    # nodeUserID_num_actions_this_timestep
    # parentUserID_influence_this_timestep
    # edge_weight_this_timestep
    # user_birthdate
    # user_age
    # nodeUserID_is_new
    # parent_birthdate
    # parent_age
    # parentUserID_is_new
    # infoID_user

    all_new_user_record_dfs = []

    infoID_to_new_user_tag_dict = {}

    for infoID in infoIDs:
        print()
        print(infoID)
        temp_history_record_df = history_record_df[history_record_df["informationID"]==infoID]
        keep_cols = ["nodeTime","nodeUserID","parentUserID", "nodeUserID_is_new", "parentUserID_is_new", "informationID","edge_weight_this_timestep" ]
        temp_history_record_df = temp_history_record_df[keep_cols]
        print("\ntemp_history_record_df")
        print(temp_history_record_df)

        #make a new user record
        new_user_record_keep_cols = ["nodeTime", "nodeUserID_is_new", "parentUserID_is_new", "informationID","edge_weight_this_timestep"]
        new_user_record_df = temp_history_record_df[new_user_record_keep_cols]
        user_is_new_flags = list(new_user_record_df["nodeUserID_is_new"])
        new_user_record_df["nodeUserID_is_old"] = [1 if user_is_new_flag==0 else 0 for user_is_new_flag in user_is_new_flags]
        new_user_influence_for_new_user_records = (new_user_record_df["nodeUserID_is_new"] * new_user_record_df["edge_weight_this_timestep"]).sum()
        temp = new_user_record_df[new_user_record_df["parentUserID_is_new"]==1].reset_index(drop=True)
        old_user_influence_for_new_user_records = (temp["nodeUserID_is_old"] * temp["edge_weight_this_timestep"]).sum()
        print("\nnew_user_influence_for_new_user_records: %d, old user influence: %d"%(new_user_influence_for_new_user_records,old_user_influence_for_new_user_records))
        overall_influence_for_new_user_records = new_user_influence_for_new_user_records + old_user_influence_for_new_user_records
        print("\noverall_influence_for_new_user_records: %d"%overall_influence_for_new_user_records)

        #get user action probas
        temp = new_user_record_df[new_user_record_df["nodeUserID_is_new"]==1].reset_index(drop=True)
        new_user_action_count_for_new_user_records = (temp["parentUserID_is_new"]* temp["edge_weight_this_timestep"]).sum()
        parent_is_new_flags = list(temp["parentUserID_is_new"])
        temp["parentUserID_is_old"] = [1 if parent_is_new_flag==0 else 0 for parent_is_new_flag in parent_is_new_flags]
        old_user_action_count_for_new_user_records= (temp["parentUserID_is_old"]* temp["edge_weight_this_timestep"]).sum()
        overall_user_action_count_for_new_records = new_user_action_count_for_new_user_records + old_user_action_count_for_new_user_records
        print("\nnew_user_action_count_for_new_user_records: %d, old_user_action_count_for_new_user_records: %d"%(new_user_action_count_for_new_user_records, old_user_action_count_for_new_user_records))
        print("\noverall_user_action_count_for_new_records: %d"%overall_user_action_count_for_new_records)

        new_user_tag = "new_user_%s"%infoID
        new_user_simple_df = pd.DataFrame(data={"nodeUserID": [new_user_tag]})
        infoID_to_new_user_tag_dict[infoID]=new_user_tag
        new_user_simple_df["nodeUserID_new_user_influence"] = new_user_influence_for_new_user_records
        new_user_simple_df["nodeUserID_old_user_influence"] = old_user_influence_for_new_user_records
        new_user_simple_df["nodeUserID_overall_user_influence"] = overall_influence_for_new_user_records
        new_user_simple_df["nodeUserID_to_new_user_act_proba"] = new_user_action_count_for_new_user_records
        new_user_simple_df["nodeUserID_to_old_user_act_proba"] = old_user_action_count_for_new_user_records
        new_user_simple_df["nodeUserID_overall_act_proba"] = overall_user_action_count_for_new_records
        new_user_simple_df["informationID"] = infoID
        all_new_user_record_dfs.append(new_user_simple_df)


    new_user_simple_df = pd.concat(all_new_user_record_dfs).reset_index(drop=True)
    print("\nnew_user_simple_df")
    print(new_user_simple_df)





    #get the proba that a user acts on a new user
    user_is_new_flags = list(temp_history_record_df["nodeUserID_is_new"])
    temp_history_record_df["nodeUserID_is_old"] = [1 if user_is_new_flag==0 else 0 for user_is_new_flag in user_is_new_flags]
    print(temp_history_record_df[["nodeUserID_is_old","nodeUserID_is_new"]])

    #set parent infl
    temp_history_record_df["nodeUserID_is_new"] = temp_history_record_df["nodeUserID_is_new"] * temp_history_record_df["edge_weight_this_timestep"]
    temp_history_record_df["nodeUserID_is_old"] = temp_history_record_df["nodeUserID_is_old"] * temp_history_record_df["edge_weight_this_timestep"]
    temp_history_record_df["parentUserID_new_user_influence"] = temp_history_record_df.groupby(["parentUserID", "informationID"])["nodeUserID_is_new"].transform("sum")
    temp_history_record_df["parentUserID_old_user_influence"] = temp_history_record_df.groupby(["parentUserID", "informationID"])["nodeUserID_is_old"].transform("sum")
    temp_history_record_df["parentUserID_overall_user_influence"] = temp_history_record_df["parentUserID_new_user_influence"] + temp_history_record_df["parentUserID_old_user_influence"]
    print("\ntemp_history_record_df with infl")
    print(temp_history_record_df)

    #set user action probas
    parent_is_new_flags = list(temp_history_record_df["parentUserID_is_new"])
    temp_history_record_df["parentUserID_is_old"] = [1 if parent_is_new_flag==0 else 0 for parent_is_new_flag in parent_is_new_flags]
    print(temp_history_record_df[["parentUserID_is_old","parentUserID_is_new"]])

    temp_history_record_df["parentUserID_is_new"] = temp_history_record_df["parentUserID_is_new"] * temp_history_record_df["edge_weight_this_timestep"]
    temp_history_record_df["parentUserID_is_old"] = temp_history_record_df["parentUserID_is_old"] * temp_history_record_df["edge_weight_this_timestep"]
    temp_history_record_df["nodeUserID_to_new_user_act_proba"] = temp_history_record_df.groupby(["nodeUserID", "informationID"])["parentUserID_is_new"].transform("sum")
    temp_history_record_df["nodeUserID_to_old_user_act_proba"] = temp_history_record_df.groupby(["nodeUserID", "informationID"])["parentUserID_is_old"].transform("sum")
    temp_history_record_df["nodeUserID_overall_act_proba"] =temp_history_record_df["nodeUserID_to_new_user_act_proba"] + temp_history_record_df["nodeUserID_to_old_user_act_proba"]
    print("\ntemp_history_record_df with actions")
    print(temp_history_record_df)

    #little hack
    temp_history_record_df["original_nodeUserID"]=temp_history_record_df["nodeUserID"].copy()
    temp_history_record_df["original_parentUserID"]=temp_history_record_df["parentUserID"].copy()
    temp_history_record_df["nodeUserID"] = temp_history_record_df["nodeUserID"]  + "<with>"+temp_history_record_df["informationID"]
    temp_history_record_df["parentUserID"] = temp_history_record_df["parentUserID"]  + "<with>"+temp_history_record_df["informationID"]



    #make dicts
    user_to_new_infl_dict = convert_df_2_cols_to_dict(temp_history_record_df, "parentUserID", "parentUserID_new_user_influence")
    user_to_old_infl_dict = convert_df_2_cols_to_dict(temp_history_record_df, "parentUserID", "parentUserID_old_user_influence")
    user_to_overall_infl_dict = convert_df_2_cols_to_dict(temp_history_record_df, "parentUserID", "parentUserID_overall_user_influence")

    user_to_new_action_dict = convert_df_2_cols_to_dict(temp_history_record_df, "nodeUserID", "nodeUserID_to_new_user_act_proba")
    user_to_old_action_dict = convert_df_2_cols_to_dict(temp_history_record_df, "nodeUserID", "nodeUserID_to_old_user_act_proba")
    user_to_overall_action_dict = convert_df_2_cols_to_dict(temp_history_record_df, "nodeUserID", "nodeUserID_overall_act_proba")

    #combine nodes
    all_users = list(temp_history_record_df["nodeUserID"].unique()) +  list(temp_history_record_df["parentUserID"].unique())
    all_users= list(set(all_users))

    user_weight_table = pd.DataFrame(data={"nodeUserID":all_users})
    user_weight_table["nodeUserID_new_user_influence"] = user_weight_table["nodeUserID"].map(user_to_new_infl_dict)
    user_weight_table["nodeUserID_old_user_influence"] = user_weight_table["nodeUserID"].map(user_to_old_infl_dict)
    user_weight_table["nodeUserID_overall_user_influence"] = user_weight_table["nodeUserID"].map(user_to_overall_infl_dict)
    user_weight_table["nodeUserID_to_new_user_act_proba"] = user_weight_table["nodeUserID"].map(user_to_new_action_dict)
    user_weight_table["nodeUserID_to_old_user_act_proba"] = user_weight_table["nodeUserID"].map(user_to_old_action_dict)
    user_weight_table["nodeUserID_overall_act_proba"] = user_weight_table["nodeUserID"].map(user_to_overall_action_dict)

    print("\nuser_weight_table before hack")
    print(user_weight_table)

    #undo hack
    new = user_weight_table["nodeUserID"].str.split("<with>",  expand = True)
    user_weight_table["nodeUserID"] = new[0]
    user_weight_table["informationID"] = new[1]

    print("\nuser_weight_table after hack")
    print(user_weight_table)

    cols = ["nodeUserID", "nodeUserID_to_new_user_act_proba","nodeUserID_to_old_user_act_proba","nodeUserID_overall_act_proba","nodeUserID_new_user_influence","nodeUserID_old_user_influence","nodeUserID_overall_user_influence","informationID"]
    user_weight_table = user_weight_table[cols]

    user_weight_table = pd.concat([user_weight_table,new_user_simple_df]).reset_index(drop=True)

    for col in cols:
        if col not in  ["nodeUserID", "informationID"]:
            user_weight_table[col] = user_weight_table[col]/user_weight_table[col].sum()
            temp = user_weight_table[col].copy()
            temp=temp[temp>0]
            min_val = temp.min()
            user_weight_table[col]=user_weight_table[col].fillna(0)
            cur_epsilon = min_val/EPSILON_DIV_NUM
            user_weight_table[col] = user_weight_table[col] + cur_epsilon
            print(col)
            print(cur_epsilon)

    print("\nuser_weight_table")
    print(user_weight_table)


    #more stats
    #follower net size
    temp_history_record_df = history_record_df[["nodeUserID", "parentUserID", "informationID"]].drop_duplicates()
    temp_history_record_df["parentUserID_follower_net_size"] = temp_history_record_df.groupby(["parentUserID", "informationID"])["nodeUserID"].transform("count")
    user_to_follower_net_size_dict = convert_df_2_cols_to_dict(temp_history_record_df, "parentUserID", "parentUserID_follower_net_size")
    user_weight_table["nodeUserID_follower_net_size"] = user_weight_table["nodeUserID"].map(user_to_follower_net_size_dict)
    user_weight_table["nodeUserID_follower_net_size"] =user_weight_table["nodeUserID_follower_net_size"].fillna(0)

    print("\nuser_weight_table")
    print(user_weight_table)

    #======== follower net itself =================
    edge_df = history_record_df[[ "nodeTime","nodeUserID", "parentUserID", "informationID", "edge_weight_this_timestep"]].drop_duplicates()
    edge_df["overall_edge_weight"] = edge_df.groupby(["nodeUserID", "parentUserID", "informationID"])["edge_weight_this_timestep"].transform("sum")
    edge_df = edge_df[["nodeUserID", "parentUserID", "informationID","overall_edge_weight"]].drop_duplicates()
    edge_df["overall_edge_weight"]=edge_df["overall_edge_weight"]/edge_df["overall_edge_weight"].sum()
    print("\nedge_df")
    print(edge_df)

    edge_df["nodeUserID_weight_tuple"] = list(zip(edge_df["nodeUserID"],edge_df["overall_edge_weight"] ))

    follower_network = {}
    follower_net_groupby_object = edge_df.groupby(["parentUserID","informationID"])["nodeUserID_weight_tuple"]
    for i,f in enumerate(follower_net_groupby_object):
        print()
        # print(f)

        user_infoID_tuple = f[0]
        all_users = list(f[1])
        follower_network[user_infoID_tuple] = all_users
        # print(user_infoID_tuple)
        # print(all_users)



    print("\nuser_weight_table")
    print(user_weight_table)


    #make proba infl table
    #parent, infoID, influence proba, new or old flag
    new_user_infl_proba_table_cols = ["nodeUserID", "informationID", "nodeUserID_new_user_influence"]
    old_user_infl_proba_table_cols = ["nodeUserID", "informationID", "nodeUserID_old_user_influence"]
    new_user_infl_proba_table = user_weight_table[new_user_infl_proba_table_cols]
    old_user_infl_proba_table = user_weight_table[old_user_infl_proba_table_cols]
    new_user_infl_proba_table = new_user_infl_proba_table.rename(columns={"nodeUserID_new_user_influence":"nodeUserID_user_influence"})
    old_user_infl_proba_table = old_user_infl_proba_table.rename(columns={"nodeUserID_old_user_influence":"nodeUserID_user_influence"})
    new_user_infl_proba_table["child_is_new"] = 1
    old_user_infl_proba_table["child_is_new"] = 0
    user_infl_proba_table = pd.concat([new_user_infl_proba_table,old_user_infl_proba_table]).reset_index(drop=True)
    print(user_infl_proba_table["nodeUserID_user_influence"].value_counts())
    print("\nuser_infl_proba_table")
    print(user_infl_proba_table)

    # #make proba action table
    # #parent, infoID, influence proba, new or old flag
    # new_user_infl_proba_table_cols = ["nodeUserID", "informationID", "nodeUserID_new_user_influence"]
    # old_user_infl_proba_table_cols = ["nodeUserID", "informationID", "nodeUserID_old_user_influence"]
    # new_user_infl_proba_table = user_weight_table[new_user_infl_proba_table_cols]
    # old_user_infl_proba_table = user_weight_table[old_user_infl_proba_table_cols]
    # new_user_infl_proba_table = new_user_infl_proba_table.rename(columns={"nodeUserID_new_user_influence":"nodeUserID_user_influence"})
    # old_user_infl_proba_table = old_user_infl_proba_table.rename(columns={"nodeUserID_old_user_influence":"nodeUserID_user_influence"})
    # new_user_infl_proba_table["child_is_new"] = 1
    # old_user_infl_proba_table["child_is_new"] = 0
    # user_infl_proba_table = pd.concat([new_user_infl_proba_table,old_user_infl_proba_table]).reset_index(drop=True)
    # print(user_infl_proba_table["nodeUserID_user_influence"].value_counts())
    # print("\nuser_infl_proba_table")
    # print(user_infl_proba_table)


    return user_weight_table,follower_network, user_infl_proba_table,infoID_to_new_user_tag_dict,edge_df


def get_cur_timestep_infoID_user_assignment_df(infoID, timestep,cur_pred_df,ADD_OLD_USERS_TO_NEW_USERS_IF_TOO_MANY,cur_user_weight_table):

    #get cur pred df
    hyp_infoID = hyp_dict[infoID]
    print("\nGetting new user links for %s"%infoID)
    cur_pred_df = cleaned_pred_df_dict[infoID]
    cur_pred_df = cur_pred_df[cur_pred_df["timestep"]==timestep].copy().reset_index(drop=True)
    print("\ncur_pred_df")
    print(cur_pred_df)

    #get numbers
    num_new_users = int(cur_pred_df["num_new_users"].iloc[0])
    num_old_users = int(cur_pred_df["num_old_users"].iloc[0])
    num_actions = int(cur_pred_df["num_actions"].iloc[0])

    # #cur weights
    # cur_user_weight_table = user_weight_table[user_weight_table["informationID"]==infoID].reset_index(drop=True)

    #============= get old users first -> check if there's a conflict =============

    num_old_users_in_weight_df = cur_user_weight_table["nodeUserID"].nunique()
    old_user_diff = num_old_users_in_weight_df - num_old_users
    print("\nNum old users in weight df: %d; num predicted: %d; Difference: %d"%(num_old_users_in_weight_df, num_old_users, old_user_diff))

    if old_user_diff >= 0:
        #get OLD users
        sampled_old_user_records = cur_user_weight_table.sample(n=num_old_users ,weights=cur_user_weight_table["nodeUserID_overall_action_weight"], replace=False)
    else:
        #get what you can
        amount_of_old_users_possible_to_sample = num_old_users_in_weight_df
        print("\nWe can only get %d old users"%amount_of_old_users_possible_to_sample)
        sampled_old_user_records= cur_user_weight_table.sample(n=amount_of_old_users_possible_to_sample ,weights=cur_user_weight_table["nodeUserID_overall_action_weight"], replace=False)

        remaining_old_users = num_old_users - amount_of_old_users_possible_to_sample

        initial_cur_pred_df = cur_pred_df.copy()
        print("\nThere are %d remaining_old_users that need to be sampled"%remaining_old_users )
        cur_pred_df["num_old_users"] = amount_of_old_users_possible_to_sample
        num_old_users = amount_of_old_users_possible_to_sample

        if ADD_OLD_USERS_TO_NEW_USERS_IF_TOO_MANY == True:
            print("\nADD_OLD_USERS_TO_NEW_USERS_IF_TOO_MANY==True, so we have to increase # of new users")
            num_new_users = num_new_users + remaining_old_users

            cur_pred_df["num_new_users"] = num_new_users

        print("\nInitial cur pred df")
        print(initial_cur_pred_df)

        print("\nModified cur pred df")
        print(cur_pred_df)

    print("\nsampled_old_user_records")
    sampled_old_user_records = sampled_old_user_records.rename(columns={"latest_parent":"parentUserID"})
    sampled_old_user_records = sampled_old_user_records.reset_index(drop=True)
    print(sampled_old_user_records)

    #========================== get new users ==========================
    #get new users
    try:
        sampled_new_user_records = cur_user_weight_table.sample(n=num_new_users ,weights=cur_user_weight_table["nodeUserID_overall_action_weight"], replace=False)
    except ValueError:
        sampled_new_user_records = cur_user_weight_table.sample(n=num_new_users ,weights=cur_user_weight_table["nodeUserID_overall_action_weight"], replace=True)
    sampled_new_user_records = sampled_new_user_records.rename(columns={"latest_parent":"parentUserID"})

    #check for self loops
    users = list(sampled_new_user_records["nodeUserID"])
    parents = list(sampled_new_user_records["parentUserID"])
    sampled_new_user_records["is_self_loop"] = [1 if user==parent else 0 for user,parent in zip(users,parents)]
    sampled_new_user_records["nodeUserID"] = ["synthetic_user_%d"%(i+1) for i in range(sampled_new_user_records.shape[0])]

    users = list(sampled_new_user_records["nodeUserID"])
    parents = list(sampled_new_user_records["parentUserID"])
    is_self_loop_list = list(sampled_new_user_records["is_self_loop"])
    sampled_new_user_records["parentUserID"] = [user if is_self_loop==1 else parent for user,parent,is_self_loop in zip(users,parents,is_self_loop_list)]
    sampled_new_user_records = sampled_new_user_records.drop("is_self_loop", axis=1)

    print("\nSampled new users")
    print(sampled_new_user_records)

    print("\nSampled old users")
    print(sampled_old_user_records)

    #=================== add a checkpoint here ===============
    verify_pred_counts_so_far_v2_debug_print(sampled_new_user_records,sampled_old_user_records,num_old_users, num_new_users, DEBUG_PRINT)

    #combine these records into 1 df
    new_and_old_user_record_df = pd.concat([sampled_old_user_records,sampled_new_user_records]).reset_index(drop=True)

    print("\nnew_and_old_user_record_df")
    print(new_and_old_user_record_df)
    print(new_and_old_user_record_df["parentUserID"].value_counts())


    #======== get remaining action preds  ==============
    num_actions_so_far = new_and_old_user_record_df.shape[0]
    num_remaining_actions = num_actions - num_actions_so_far
    print("\nnum_actions_so_far: %d, num_remaining_actions:%d"%(num_actions_so_far , num_remaining_actions))
    print("\nWe need to get %d more actions..."%num_remaining_actions)

    repeat_user_records = new_and_old_user_record_df.sample(n=num_remaining_actions, weights=new_and_old_user_record_df["nodeUserID_overall_action_weight"], replace=True)
    repeat_user_records = repeat_user_records.reset_index(drop=True)

    print("\nrepeat_user_records")
    print(repeat_user_records)


    new_and_old_user_record_df = pd.concat([new_and_old_user_record_df , repeat_user_records]).reset_index(drop=True)

    print("\nnew_and_old_user_record_df")
    print(new_and_old_user_record_df)

    new_and_old_user_record_df["timestep"] = timestep

    users = list(new_and_old_user_record_df["nodeUserID"])
    new_and_old_user_record_df["nodeUserID_is_new"] = [1 if ("synthetic" in user) else 0 for user in users]
    parents = list(new_and_old_user_record_df["parentUserID"])
    new_and_old_user_record_df["parentUserID_is_new"] = [1 if ("synthetic" in parent) else 0 for parent in parents]
    final_keep_cols = ["timestep" ,"nodeUserID", "parentUserID", "informationID", "nodeUserID_is_new", "parentUserID_is_new"]
    final_pred_df = new_and_old_user_record_df[final_keep_cols]

    print("\nfinal_pred_df")
    print(final_pred_df)

    return final_pred_df

def generate_user_assign_tag_from_param_dict(param_dict):

    platform = param_dict["platform"]


    try:
        model_tag = param_dict["%s_model_tag"%platform]
    except KeyError:
        model_tag = param_dict["model_tag"]
    GRAN = param_dict["GRAN"]
    weight_start = param_dict["weight_start"]
    weight_end = param_dict["weight_end"]
    weight_start = pd.to_datetime(weight_start, utc=True)
    weight_end = pd.to_datetime(weight_end, utc=True)
    # weight_end=weight_end.dt.round(GRAN)

    time_delta = weight_end - weight_start
    time_delta=str(time_delta)
    num_days = time_delta.split(" ")[0]
    num_days =int(num_days)+1

    # print("\ntime_delta")
    # print(time_delta)
    # print("\nnum_days")
    # print(num_days)

    model_tag = model_tag + "-" + str(num_days) + "Dhist"

    user_action_conflict_resolution_option = param_dict["user_action_conflict_resolution_option"]
    if user_action_conflict_resolution_option == "downsample_users_to_actions":
        conflict_tag = "D_U2A"
    else:
        conflict_tag = "U_A2U"


    model_tag = model_tag + "-" + conflict_tag


    infl_eps = param_dict["INFLUENCE_EPSILON_DIV_VAL"]
    model_tag = model_tag + "-IE_%s"%infl_eps

    ADD_OLD_USERS_TO_NEW_USERS_IF_TOO_MANY = param_dict["ADD_OLD_USERS_TO_NEW_USERS_IF_TOO_MANY"]
    model_tag = model_tag + "-AO2N-" + str(ADD_OLD_USERS_TO_NEW_USERS_IF_TOO_MANY)



    return model_tag

def get_gt_baseline_and_model_dicts(cur_gt_sim_dir,cur_baseline_sim_dir,platform,infoIDs, num_timesteps,main_model_input_dir ,main_model_tag,hyp_dict):

    #get baseline dirs
    baseline_infoID_to_timestep_df_dict = {}
    for infoID in infoIDs:
        hyp_infoID = hyp_dict[infoID]
        cur_baseline_fp = cur_baseline_sim_dir + hyp_infoID + ".csv"
        cur_baseline_df = pd.read_csv(cur_baseline_fp)
        baseline_infoID_to_timestep_df_dict[infoID] = {}

        for timestep in range(1, num_timesteps+1):
            cur_timestep_df = cur_baseline_df[cur_baseline_df["timestep"]==timestep].reset_index(drop=True)
            baseline_infoID_to_timestep_df_dict[infoID][timestep] = cur_timestep_df
            print("\ncur_timestep_df")
            print(cur_timestep_df)

    #get baseline dirs
    gt_infoID_to_timestep_df_dict = {}
    for infoID in infoIDs:
        hyp_infoID = hyp_dict[infoID]
        cur_gt_fp = cur_gt_sim_dir + hyp_infoID + ".csv"
        cur_gt_df = pd.read_csv(cur_gt_fp)
        gt_infoID_to_timestep_df_dict[infoID] = {}

        for timestep in range(1, num_timesteps+1):
            cur_timestep_df = cur_gt_df[cur_gt_df["timestep"]==timestep].reset_index(drop=True)
            gt_infoID_to_timestep_df_dict[infoID][timestep] = cur_timestep_df
            print("\ncur_timestep_df")
            print(cur_timestep_df)

            # sys.exit(0)

    #get model dir
    model_to_timestep_df_dict = {}
    cur_model_dir = main_model_input_dir +"Simulations/"
    for infoID in infoIDs:
        hyp_infoID = hyp_dict[infoID]
        model_to_timestep_df_dict[infoID]={}
        cur_model_fp = cur_model_dir + hyp_infoID + ".csv"
        cur_model_df = pd.read_csv(cur_model_fp)
        print(cur_model_df)

        for timestep in range(1, num_timesteps+1):
            cur_timestep_df = cur_model_df[cur_model_df["timestep"]==timestep].reset_index(drop=True)
            model_to_timestep_df_dict[infoID][timestep] = cur_timestep_df
            print("\ncur_timestep_df")
            print(cur_timestep_df)

    return baseline_infoID_to_timestep_df_dict,gt_infoID_to_timestep_df_dict,model_to_timestep_df_dict

def get_node_sets_per_timestep(infoID_to_timestep_df_dict):

    for infoID, infoID_dict in infoID_to_timestep_df_dict.items():
        for timestep, cur_timestep_df in infoID_dict.items():

            child_user_df = cur_timestep_df[["nodeUserID", "nodeUserID_is_new"]]
            child_user_df=child_user_df[child_user_df["nodeUserID_is_new"]==0]
            old_child_users = set(child_user_df["nodeUserID"].unique())

            parent_user_df = cur_timestep_df[["parentUserID", "parentUserID_is_new"]]
            parent_user_df=parent_user_df[parent_user_df["parentUserID_is_new"]==0]
            old_parent_users = set(parent_user_df["parentUserID"].unique())

            old_users = old_child_users.union(old_parent_users)
            print(len(old_users))

            infoID_to_timestep_df_dict[infoID][timestep]=old_users


    return infoID_to_timestep_df_dict

def get_edge_info_from_edge_dicts(infoID_to_timestep_df_dict):

    for infoID, infoID_dict in infoID_to_timestep_df_dict.items():
        for timestep, cur_timestep_df in infoID_dict.items():

            cur_timestep_df["edge_weight"] = cur_timestep_df.groupby(["nodeUserID", "parentUserID"])["nodeUserID"].transform("count")
            cur_timestep_df["edge_is_new"] = cur_timestep_df["nodeUserID_is_new"] + cur_timestep_df["parentUserID_is_new"]
            cur_timestep_df=cur_timestep_df[["nodeUserID", "parentUserID", "edge_weight", "edge_is_new"]].drop_duplicates()
            cur_timestep_df=cur_timestep_df.sort_values("edge_weight", ascending=False).reset_index(drop=True)
            # print("\ncur_timestep_df")
            # print(cur_timestep_df)

            cur_timestep_df=cur_timestep_df[cur_timestep_df["edge_is_new"]==0].reset_index(drop=True)
            cur_timestep_df=cur_timestep_df.drop("edge_is_new", axis=1)
            print("\ncur_timestep_df old records only")
            print(cur_timestep_df)

            infoID_to_timestep_df_dict[infoID][timestep]=cur_timestep_df


    return infoID_to_timestep_df_dict

def get_edge_info_from_edge_dicts_with_new_users(infoID_to_timestep_df_dict):

    for infoID, infoID_dict in infoID_to_timestep_df_dict.items():
        for timestep, cur_timestep_df in infoID_dict.items():
            print(cur_timestep_df)
            # sys.exit(0)
            nodeUsers = list(cur_timestep_df["nodeUserID"])

            try:
                nodeUser_is_new_list = list(cur_timestep_df["nodeUserID_is_new"])
                cur_timestep_df["nodeUserID"] = [nodeUser if nodeUser_is_new==0 else "new_user" for nodeUser,nodeUser_is_new in zip(nodeUsers,nodeUser_is_new_list)]
            except KeyError:
                cur_timestep_df["nodeUserID"] = [nodeUser if "synthetic_user" not in nodeUser else "new_user" for nodeUser in nodeUsers]

            parentUsers = list(cur_timestep_df["parentUserID"])


            try:
                parentUser_is_new_list = list(cur_timestep_df["parentUserID_is_new"])
                cur_timestep_df["parentUserID"] = [parentUser if parentUser_is_new==0 else "new_user" for parentUser,parentUser_is_new in zip(parentUsers,parentUser_is_new_list)]
            except KeyError:
                cur_timestep_df["parentUserID"] = [parentUser if "synthetic_user" not in parentUser else "new_user" for parentUser in parentUsers]



            cur_timestep_df["edge_weight"] = cur_timestep_df.groupby(["nodeUserID", "parentUserID"])["nodeUserID"].transform("count")


            # cur_timestep_df["edge_is_new"] = cur_timestep_df["nodeUserID_is_new"] + cur_timestep_df["parentUserID_is_new"]
            cur_timestep_df=cur_timestep_df[["nodeUserID", "parentUserID", "edge_weight"]].drop_duplicates()
            cur_timestep_df=cur_timestep_df.sort_values("edge_weight", ascending=False).reset_index(drop=True)
            # print("\ncur_timestep_df")
            # print(cur_timestep_df)

            # cur_timestep_df=cur_timestep_df[cur_timestep_df["edge_is_new"]==0].reset_index(drop=True)
            # cur_timestep_df=cur_timestep_df.drop("edge_is_new", axis=1)
            print("\ncur_timestep_df old records only")
            print(cur_timestep_df)

            infoID_to_timestep_df_dict[infoID][timestep]=cur_timestep_df


    return infoID_to_timestep_df_dict


def rank_edges(infoID_to_timestep_df_dict, edge_weight_col):

    for infoID, infoID_dict in infoID_to_timestep_df_dict.items():
        for timestep, cur_timestep_df in infoID_dict.items():
            print(timestep)
            # sys.exit(0)
            print(cur_timestep_df)

            edge_weight_df = cur_timestep_df[[edge_weight_col]].drop_duplicates()
            edge_weight_df = edge_weight_df.sort_values(edge_weight_col, ascending=False).reset_index(drop=True)
            print("\nedge_weight_df")
            print(edge_weight_df)

            edge_weight_df["rank"] = [(i+1) for i in range(edge_weight_df.shape[0])]

            # weight_to_rank_dict = convert_df_2_cols_to_dict(edge_weight_df, edge_weight_col, "rank")
            cur_timestep_df= pd.merge(cur_timestep_df,edge_weight_df, on=edge_weight_col, how="inner").reset_index(drop=True)

            cur_timestep_df = cur_timestep_df.sort_values("rank", ascending=True)
            print("\ncur_timestep_df")
            print(cur_timestep_df)
            infoID_to_timestep_df_dict[infoID][timestep]=cur_timestep_df

            # sys.exit(0)


    return infoID_to_timestep_df_dict

def create_gt_and_model_relevance_dict(gt_infoID_to_timestep_df_dict, model_infoID_to_timestep_df_dict, num_timesteps, infoIDs):

    infoID_to_relevance_df_dict = {}
    for infoID in infoIDs:
        infoID_to_relevance_df_dict[infoID] = {}
        for timestep in range(1, num_timesteps+1):
            model_result_df = model_infoID_to_timestep_df_dict[infoID][timestep]
            gt_df = gt_infoID_to_timestep_df_dict[infoID][timestep]


            model_result_df = model_result_df.rename(columns={"edge_weight":"predicted_edge_weight"})
            gt_df = gt_df.rename(columns={"edge_weight":"gt_edge_weight"})


            relevance_df = pd.merge(model_result_df, gt_df, on=["nodeUserID", "parentUserID"], how="outer").reset_index(drop=True)

            relevance_df = relevance_df.fillna(0)

            print("\nrelevance_df")
            print(relevance_df)

            infoID_to_relevance_df_dict[infoID][timestep] = relevance_df

            # sys.exit(0)

    infoID_to_relevance_df_dict = rank_edges(infoID_to_relevance_df_dict, "gt_edge_weight")



    return infoID_to_relevance_df_dict

def get_model_infoID_to_ndcg_dict(infoID_to_relevance_df_dict, pred_col, gt_col, infoIDs, num_timesteps):

    infoID_to_result_score_df_dict = {}
    for infoID in infoIDs:

        cur_infoID_results = []
        for timestep in range(1, num_timesteps+1):

            cur_rel_df = infoID_to_relevance_df_dict[infoID][timestep]
            cur_rel_df=cur_rel_df.sort_values(gt_col, ascending=False).reset_index(drop=True)
            print("\ncur_rel_df")
            print(cur_rel_df)

            gt_vals = np.asarray([list(cur_rel_df[gt_col])])
            pred_vals = np.asarray([list(cur_rel_df[pred_col])])
            cur_score = ndcg_score(gt_vals,pred_vals)
            cur_infoID_results.append(cur_score)

        infoID_score_df = pd.DataFrame(data={"ndcg_score":cur_infoID_results})
        infoID_score_df["infoID"]=infoID
        infoID_score_df["timestep"] = [i for i in range(1, num_timesteps+1)]
        infoID_score_df=infoID_score_df[["infoID" ,"timestep","ndcg_score"]]
        print()
        print(infoID_score_df)
        infoID_to_result_score_df_dict[infoID] = infoID_score_df



    return infoID_to_result_score_df_dict

def get_avg_scores_of_model(score_dict, infoIDs, num_timesteps, score_col="ndcg_score"):
    avg_scores = []
    for infoID in infoIDs:
        score_df = score_dict[infoID]
        avg_score = score_df[score_col].mean()
        avg_scores.append(avg_score)

    avg_score_df = pd.DataFrame(data={"avg_" + score_col : avg_scores, "infoID":infoIDs})
    print()
    print(avg_score_df)

    return avg_score_df

def compare_2_model_dfs(model_tag, baseline_tag, model_score_df, baseline_score_df, score_col):

    model_score_df = model_score_df.rename(columns={score_col: model_tag})
    baseline_score_df = baseline_score_df.rename(columns={score_col: baseline_tag})

    comp_df = pd.merge(model_score_df,baseline_score_df, on="infoID", how="inner")
    print(comp_df)

    cols = ["infoID", model_tag, baseline_tag]
    comp_df=comp_df[cols]

    model_scores = comp_df[model_tag]
    baseline_scores = comp_df[baseline_tag]
    comp_df["%s_is_winner"%model_tag] = [1 if model_score>baseline_score else 0 for model_score, baseline_score in zip(model_scores, baseline_scores)]

    return comp_df

def rh_distance(ground_truth,simulation):
    """
    Metric: rh_distance
    Description: Relative Hausdorff distance between the ground
    truth and simulation data.
    Meant for degree sequence measurements
    Inputs:
    ground_truth - ground truth in degree distrib (iterable)
    simulation - simulation in degree distrib (iterable)
    Citation:
    Aksoy, S.G., K.E. Nowak, S.J. Young,
    "A linear time algorithm and analysis of graph Relative Hausdorff
    distance", SIAM Journal on Mathematics of Data Science 1,
    no. 4 (2019): 647-666.
    """

    #Rename for notional convience, convert to lists for speed
    G=list(ground_truth)
    F=list(simulation)

    #If no edges, RH computed on closest nonzero degree sequence of [1]
    try:
        if max(G)==0:
            G=[1]
    except ValueError:
        G=[1]

    try:
        if max(F)==0:
            F=[1]
    except ValueError:
        F=[1]

    #convert degree sequence to CCDH sequence
    G.sort()
    F.sort()
    G=[len(G) - bisect(G, z) for z in range(0,max(G))]
    F=[len(F) - bisect(F, z) for z in range(0,max(F))]

    #Convert CCDH sequence to dictionary of CCDH values
    #keyed by degrees 1 to max degree.
    mDegF = len(F)
    degF = {i : F[i-1] for i in range(1, mDegF+1)}
    mDegG = len(G)
    degG = {i : G[i-1] for i in range(1, mDegG+1)}

    # Main subroutine: Return max of epsilon, delta, where delta is
    # min size box around (d,Fd) containing smooth path through G.
    def epsilon_box(d,Fd,G,mDegG,epsilon):
        right = d*(1+epsilon)
        r = int(np.floor(right))
        r_frac = right - r
        rightF = Fd*(1+epsilon)

        left = d*(1-epsilon)
        k = int(np.ceil(left))
        k_frac = k - left
        leftF = Fd*(1-epsilon)

        if ((leftF <= mDegG)
            and (rightF >= G.get(r,0)*(1-r_frac) + G.get(r+1,0)*r_frac)
            and (leftF <= G.get(k,G[1])*(1-k_frac) + G.get(k-1,G[1])*k_frac)):
            return epsilon

        if ((mDegG < leftF) and (leftF <= mDegG+1)
            and (rightF >= G.get(r,0)*(1-r_frac) + G.get(r+1,0)*r_frac)
            and (leftF <= G[mDegG]*k_frac)):
            return epsilon

        # If here, G doesn't pass through current epsilon box,
        # must find necessary epsilon box size. If G pass below box
        if G.get(d,0) < Fd:
            while k > mDegG + 1 or G.get(k-1,G[1]) < Fd*float(k-1)/d:
                k -= 1
            if k == mDegG + 1:
                return 1 - float(G[mDegG]*(mDegG+1))/(Fd + G[mDegG]*d)
            else:
                return 1 - float((1-k)*G.get(k,G[1])
                        + k*G.get(k-1,G[1]))/(Fd + d*(G.get(k-1,G[1])
                        - G.get(k,G[1])))
        # otherwise, G pass above box
        else:
            while G.get(r+1,0) > Fd*float(r+1)/d:
                r += 1
            return float((1+r)*G.get(r,0)
                        - r*G.get(r+1,0))/(d*(G.get(r,0)
                        - G.get(r+1,0)) + Fd) - 1

    #Compute RH distance
    checkF = set([1,mDegF] + [j for j in range(2,mDegF)
                if not ((degF[j] == degF[j-1]) and (degF[j] == degF[j+1]))])
    checkG = set([1,mDegG] + [j for j in range(2,mDegG)
                if not ((degG[j] == degG[j-1]) and (degG[j] == degG[j+1]))])
    epsilon = 0.0
    for d in range(1,max(mDegG,mDegF)+1):
        if d in checkF:
            epsilon_prime = epsilon_box(d,degF[d],degG,mDegG,epsilon)
            if epsilon_prime > epsilon:
                epsilon = epsilon_prime
        if d in checkG:
            epsilon_prime = epsilon_box(d,degG[d],degF,mDegF,epsilon)
            if epsilon_prime > epsilon:
                epsilon = epsilon_prime
    return epsilon


def graph_pred_vs_gt_with_baseline_from_comp_df_metric_options(proper_comp_df, model_tag, baseline_tag, metric_tag,platform,graph_output_dir):
    create_output_dir(graph_output_dir)


    model_pred_col = model_tag + "_pred"
    baseline_pred_col = baseline_tag + "_pred"
    infoIDs = list(proper_comp_df["infoID"].unique())
    num_timesteps = proper_comp_df["timestep"].nunique()
    target_fts = list(proper_comp_df["target_ft"].unique())

    hyp_dict = hyphenate_infoID_dict(infoIDs)

    for infoID in infoIDs:

        hyp_infoID = hyp_dict[infoID]
        #============ plot stuff ==============

        coord_list = [0, 1, 2]
        fig, axs = plt.subplots(3,figsize=(9,9))

        target_ft_to_axis_coordinates_dict = {}

        # num_coords = len(coord_list)
        idx=0
        for target_ft,coord in zip(target_fts,coord_list ):
            target_ft_to_axis_coordinates_dict[target_ft] = coord
            print("%s: %s"%(target_ft, str(coord)))

        for coord,target_ft in enumerate(target_fts):
            print("\nPlotting %s %s %s..."%(platform, infoID, target_ft))

            target_ft_to_axis_coordinates_dict[target_ft] = coord



            temp_df = proper_comp_df[(proper_comp_df["infoID"]==infoID) & (proper_comp_df["platform"]==platform) & (proper_comp_df["target_ft"]==target_ft)]
            temp_df = temp_df.reset_index(drop=True)
            print("\ntemp_df")
            print(temp_df)

            if temp_df.shape[0] != num_timesteps:
                print("\nError! temp_df.shape[0] != num_timesteps")
                print(temp_df.shape[0])
                print(num_timesteps)
                sys.exit(0)
            else:
                print("\nCounts are good...Plotting...")

            #======================= get series =======================
            model_pred_series = temp_df[model_pred_col]
            baseline_pred_series = temp_df[baseline_pred_col]
            gt_series = temp_df["actual"]

            #======================= score stuff =======================
            model_mean_error = temp_df[model_tag + "_" + metric_tag].mean()
            model_wins = temp_df[temp_df["is_winner"]==1].shape[0]
            num_trials_no_ties = temp_df[temp_df["is_winner"]!="tie"].shape[0]
            try:
                model_win_score = model_wins/num_trials_no_ties
            except ZeroDivisionError:
                model_win_score = 0
            model_error_result_tag = "%s %s: %.4f"%(model_tag, metric_tag, model_mean_error)
            # print()
            # print(model_error_result_tag)

            baseline_mean_error = temp_df[baseline_tag + "_" + metric_tag].mean()
            baseline_error_result_tag = "%s %s: %.4f"%(baseline_tag , metric_tag, baseline_mean_error)
            # print(baseline_error_result_tag)

            model_score_tag = "%s win freq: %.4f"%(model_tag,  model_win_score)
            # print(model_score_tag)

            full_report_tag = model_error_result_tag + "\n" + baseline_error_result_tag + "\n" + model_score_tag
            print()
            print(full_report_tag)

            # target_no_underscore = target_ft.replace("_"," ")
            # title_tag = "%s %s"%(infoID, target_no_underscore)
            # title_tag = title_tag + "\n"+ full_report_tag
            # title = ("\n".join(wrap(title_tag, 50)))

            target_no_underscore = target_ft.replace("_"," ")
            infoID_target_tag = "%s %s"%(infoID, target_no_underscore)
            title_tag = infoID_target_tag + "\n"+ full_report_tag
            title = title_tag

            # print("\ncur_y_pred shape")
            # print(cur_y_pred.shape)
            # print("\ncur_y_ground_truth shape")
            # print(cur_y_ground_truth.shape)
            axs[coord].plot(model_pred_series,"-r" ,label=model_tag)
            axs[coord].plot(baseline_pred_series,":g" ,label=baseline_tag)
            axs[coord].plot(gt_series, "-k" ,label="Ground Truth")
            axs[coord].set_title(title)

        for ax in axs.flat:
            ax.set(xlabel='days', ylabel='Volume')
            ax.legend()

        # plt.legend()
        plt.tight_layout()
        output_fp = graph_output_dir + "%s.png"%( hyp_infoID)
        fig.savefig(output_fp)
        plt.close()
        print(output_fp)


def get_infoID_summary_df(pred_df,metric_func_strs,model_tag,baseline_tag,metric_str_to_function_dict,infoIDs, model_type):

    pred_col = model_type + "_pred"
    baseline_col = baseline_tag + "_pred"

    col_order = []
    model_metric_to_result_list_dict = {}
    for metric in metric_func_strs:
        model_error_col = model_tag + "_" + metric
        baseline_error_col = baseline_tag + "_" + metric
        model_metric_to_result_list_dict[model_error_col] = []
        model_metric_to_result_list_dict[baseline_error_col] = []
        col_order.append(model_error_col)
        col_order.append(baseline_error_col)

    for infoID in infoIDs:
        temp = pred_df[pred_df["infoID"]==infoID]
        temp = temp.sort_values("timestep").reset_index(drop=True)
        print(temp)

        model_preds = temp[pred_col]
        baseline_preds = temp[baseline_col]
        actuals = temp["actual"]


        for metric in metric_func_strs:
            model_error_col = model_tag + "_" + metric
            baseline_error_col = baseline_tag + "_" + metric
            metric_func = metric_str_to_function_dict[metric]

            if metric == "rmse":
                model_metric_error_result = metric_func( actuals,model_preds, squared=False)
                baseline_metric_error_result = metric_func( actuals,baseline_preds, squared=False)
            else:
                model_metric_error_result = metric_func( actuals,model_preds)
                baseline_metric_error_result = metric_func( actuals,baseline_preds)
            model_metric_to_result_list_dict[model_error_col].append(model_metric_error_result)
            model_metric_to_result_list_dict[baseline_error_col].append(baseline_metric_error_result)

    print(model_metric_to_result_list_dict[model_error_col])
    data = model_metric_to_result_list_dict
    data["infoID"] = infoIDs
    result_df = pd.DataFrame(data=data)
    col_order = ["infoID"] + list(col_order)
    result_df=result_df[col_order]

    print("\nresult_df")
    print(result_df)

    # cols = list(result_df)
    # for col in cols:
    #     result_df = result_df.rename(columns={col: col.replace("_", " ")})

    # output_fp = cur_output_dir + "y_test_infoID_results.csv"
    # result_df.to_csv(output_fp, index=False)
    # print(output_fp)

    return result_df

def create_user_weight_table_v3_fix_epsilon_simple(history_record_df,INFLUENCE_EPSILON_DIV_VAL):

    print("\nGetting overall action and influence weights...")
    groupby_add_cols = ["nodeUserID", "parentUserID", "informationID"]

    keep_cols = ["nodeUserID", "parentUserID", "informationID","nodeUserID_num_actions_this_timestep","parentUserID_influence_this_timestep"]
    user_weight_table = history_record_df[keep_cols]

    user_weight_table["parentUserID_overall_influence_weight"] = user_weight_table.groupby(["parentUserID", "informationID"])["parentUserID_influence_this_timestep"].transform("sum")
    user_weight_table = get_user_influence_from_parent_influence_V2_no_fillna(user_weight_table)

    parent_influence_weight_table = user_weight_table[["parentUserID","informationID","parentUserID_overall_influence_weight"]].drop_duplicates().reset_index(drop=True)


    user_weight_table["nodeUserID_overall_action_weight"] = user_weight_table.groupby(["nodeUserID", "informationID"])["nodeUserID_num_actions_this_timestep"].transform("sum")
    user_weight_table = user_weight_table.drop_duplicates().reset_index(drop=True)

    user_weight_table = user_weight_table.drop(["nodeUserID_num_actions_this_timestep","parentUserID_influence_this_timestep", "parentUserID","parentUserID_overall_influence_weight"], axis=1)
    user_weight_table = user_weight_table.drop_duplicates().reset_index(drop=True)

    #ep1
    user_weight_table["nodeUserID_overall_action_weight"] = user_weight_table["nodeUserID_overall_action_weight"]/user_weight_table["nodeUserID_overall_action_weight"].sum()
    min_val = user_weight_table["nodeUserID_overall_action_weight"].min()
    EPSILON = min_val/INFLUENCE_EPSILON_DIV_VAL
    print("\nEPSILON")
    print(EPSILON)
    user_weight_table["nodeUserID_overall_action_weight"] = user_weight_table["nodeUserID_overall_action_weight"].fillna(EPSILON)

    #ep2
    user_weight_table["nodeUserID_overall_influence_weight"] = user_weight_table["nodeUserID_overall_influence_weight"]/user_weight_table["nodeUserID_overall_influence_weight"].sum()
    min_val = user_weight_table["nodeUserID_overall_influence_weight"].min()
    EPSILON = min_val/INFLUENCE_EPSILON_DIV_VAL
    print("\nEPSILON")
    print(EPSILON)
    user_weight_table["nodeUserID_overall_influence_weight"] = user_weight_table["nodeUserID_overall_influence_weight"].fillna(EPSILON)

    print("\nuser_weight_table")
    print(user_weight_table)

    return user_weight_table


def create_infoID_to_all_users_weight_table_dict_v2_missing_infoID(history_record_df,INFLUENCE_EPSILON_DIV_VAL,infoIDs):

    print("\nGetting overall action and influence weights...")
    groupby_add_cols = ["nodeUserID", "parentUserID", "informationID"]

    keep_cols = ["nodeUserID", "parentUserID", "informationID","nodeUserID_num_actions_this_timestep","parentUserID_influence_this_timestep"]
    user_weight_table = history_record_df[keep_cols]

    user_weight_table["parentUserID_overall_influence_weight"] = user_weight_table.groupby(["parentUserID", "informationID"])["parentUserID_influence_this_timestep"].transform("sum")
    user_weight_table = get_user_influence_from_parent_influence_V2_no_fillna(user_weight_table)

    parent_influence_weight_table = user_weight_table[["parentUserID","informationID","parentUserID_overall_influence_weight"]].drop_duplicates().reset_index(drop=True)


    user_weight_table["nodeUserID_overall_action_weight"] = user_weight_table.groupby(["nodeUserID", "informationID"])["nodeUserID_num_actions_this_timestep"].transform("sum")
    user_weight_table = user_weight_table.drop_duplicates().reset_index(drop=True)

    user_weight_table = user_weight_table.drop(["nodeUserID_num_actions_this_timestep","parentUserID_influence_this_timestep", "parentUserID","parentUserID_overall_influence_weight"], axis=1)
    user_weight_table = user_weight_table.drop_duplicates().reset_index(drop=True)

    infoID_to_all_users_weight_table_dict = {}
    # infoIDs = history_record_df["informationID"].unique()
    for infoID in infoIDs:

        cur_user_weight_table = user_weight_table[user_weight_table["informationID"]==infoID].reset_index(drop=True)

        #ep1
        cur_user_weight_table["nodeUserID_overall_action_weight"] = cur_user_weight_table["nodeUserID_overall_action_weight"]/cur_user_weight_table["nodeUserID_overall_action_weight"].sum()
        min_val = cur_user_weight_table["nodeUserID_overall_action_weight"].min()
        EPSILON = min_val/INFLUENCE_EPSILON_DIV_VAL
        print("\nACTION EPSILON")
        print(EPSILON)
        cur_user_weight_table["nodeUserID_overall_action_weight"] = cur_user_weight_table["nodeUserID_overall_action_weight"].fillna(EPSILON)

        #ep2
        cur_user_weight_table["nodeUserID_overall_influence_weight"] = cur_user_weight_table["nodeUserID_overall_influence_weight"]/cur_user_weight_table["nodeUserID_overall_influence_weight"].sum()
        #min_val = cur_user_weight_table["nodeUserID_overall_influence_weight"].min()


        infl_temp = cur_user_weight_table[cur_user_weight_table["nodeUserID_overall_influence_weight"] != 0].reset_index(drop=True)["nodeUserID_overall_influence_weight"]
        min_val = infl_temp.min()
        EPSILON = min_val/INFLUENCE_EPSILON_DIV_VAL
        print("\nINFL EPSILON")
        print(EPSILON)
        cur_user_weight_table["nodeUserID_overall_influence_weight"] = cur_user_weight_table["nodeUserID_overall_influence_weight"].fillna(EPSILON)

        print("\ncur_user_weight_table")
        print(cur_user_weight_table)
        infoID_to_all_users_weight_table_dict[infoID]=cur_user_weight_table

    return infoID_to_all_users_weight_table_dict

def create_infoID_to_all_users_weight_table_dict(history_record_df,INFLUENCE_EPSILON_DIV_VAL):

    print("\nGetting overall action and influence weights...")
    groupby_add_cols = ["nodeUserID", "parentUserID", "informationID"]

    keep_cols = ["nodeUserID", "parentUserID", "informationID","nodeUserID_num_actions_this_timestep","parentUserID_influence_this_timestep"]
    user_weight_table = history_record_df[keep_cols]

    user_weight_table["parentUserID_overall_influence_weight"] = user_weight_table.groupby(["parentUserID", "informationID"])["parentUserID_influence_this_timestep"].transform("sum")
    user_weight_table = get_user_influence_from_parent_influence_V2_no_fillna(user_weight_table)

    parent_influence_weight_table = user_weight_table[["parentUserID","informationID","parentUserID_overall_influence_weight"]].drop_duplicates().reset_index(drop=True)


    user_weight_table["nodeUserID_overall_action_weight"] = user_weight_table.groupby(["nodeUserID", "informationID"])["nodeUserID_num_actions_this_timestep"].transform("sum")
    user_weight_table = user_weight_table.drop_duplicates().reset_index(drop=True)

    user_weight_table = user_weight_table.drop(["nodeUserID_num_actions_this_timestep","parentUserID_influence_this_timestep", "parentUserID","parentUserID_overall_influence_weight"], axis=1)
    user_weight_table = user_weight_table.drop_duplicates().reset_index(drop=True)

    infoID_to_all_users_weight_table_dict = {}
    infoIDs = history_record_df["informationID"].unique()
    for infoID in infoIDs:

        cur_user_weight_table = user_weight_table[user_weight_table["informationID"]==infoID].reset_index(drop=True)

        #ep1
        cur_user_weight_table["nodeUserID_overall_action_weight"] = cur_user_weight_table["nodeUserID_overall_action_weight"]/cur_user_weight_table["nodeUserID_overall_action_weight"].sum()
        min_val = cur_user_weight_table["nodeUserID_overall_action_weight"].min()
        EPSILON = min_val/INFLUENCE_EPSILON_DIV_VAL
        print("\nEPSILON")
        print(EPSILON)
        cur_user_weight_table["nodeUserID_overall_action_weight"] = cur_user_weight_table["nodeUserID_overall_action_weight"].fillna(EPSILON)

        #ep2
        cur_user_weight_table["nodeUserID_overall_influence_weight"] = cur_user_weight_table["nodeUserID_overall_influence_weight"]/cur_user_weight_table["nodeUserID_overall_influence_weight"].sum()
        min_val = cur_user_weight_table["nodeUserID_overall_influence_weight"].min()
        EPSILON = min_val/INFLUENCE_EPSILON_DIV_VAL
        print("\nEPSILON")
        print(EPSILON)
        cur_user_weight_table["nodeUserID_overall_influence_weight"] = cur_user_weight_table["nodeUserID_overall_influence_weight"].fillna(EPSILON)

        print("\ncur_user_weight_table")
        print(cur_user_weight_table)
        infoID_to_all_users_weight_table_dict[infoID]=cur_user_weight_table

    return infoID_to_all_users_weight_table_dict


def create_new_user_only_user_weight_table_v3_fix_epsilon_simple(history_record_df,INFLUENCE_EPSILON_DIV_VAL):

    print("\nGetting overall action and influence weights...")
    groupby_add_cols = ["nodeUserID", "parentUserID", "informationID"]

    keep_cols = ["nodeUserID", "parentUserID", "informationID","nodeUserID_num_actions_this_timestep","parentUserID_influence_this_timestep", "nodeUserID_is_new"]
    # user_weight_table = history_record_df[keep_cols]

    new_parent_only_history_record_df = history_record_df[history_record_df["parentUserID_is_new"]==1]
    new_parent_only_history_record_df = new_parent_only_history_record_df[keep_cols]

    print("\nnew_parent_only_history_record_df")
    print(new_parent_only_history_record_df)




    new_parent_only_history_record_df["parentUserID_overall_influence_weight"] = new_parent_only_history_record_df.groupby(["parentUserID", "informationID"])["parentUserID_influence_this_timestep"].transform("sum")
    new_parent_only_history_record_df = get_user_influence_from_parent_influence_V2_no_fillna(new_parent_only_history_record_df)

    print("\nnew_parent_only_history_record_df")
    print(new_parent_only_history_record_df)

    new_users_only_weight_table = new_parent_only_history_record_df[new_parent_only_history_record_df["nodeUserID_is_new"]==1].reset_index(drop=True)
    print("\nnew_users_only_weight_table")
    print(new_users_only_weight_table)



    new_users_only_weight_table["nodeUserID_overall_action_weight"] = new_users_only_weight_table.groupby(["nodeUserID", "informationID"])["nodeUserID_num_actions_this_timestep"].transform("sum")
    new_users_only_weight_table = new_users_only_weight_table.drop_duplicates().reset_index(drop=True)

    new_users_only_weight_table = new_users_only_weight_table.drop(["nodeUserID_num_actions_this_timestep","parentUserID_influence_this_timestep", "parentUserID","parentUserID_overall_influence_weight"], axis=1)
    new_users_only_weight_table = new_users_only_weight_table.drop_duplicates().reset_index(drop=True)

    #ep1
    new_users_only_weight_table["nodeUserID_overall_action_weight"] = new_users_only_weight_table["nodeUserID_overall_action_weight"]/new_users_only_weight_table["nodeUserID_overall_action_weight"].sum()
    min_val = new_users_only_weight_table["nodeUserID_overall_action_weight"].min()
    EPSILON = min_val/INFLUENCE_EPSILON_DIV_VAL
    print("\nEPSILON")
    print(EPSILON)
    new_users_only_weight_table["nodeUserID_overall_action_weight"] = new_users_only_weight_table["nodeUserID_overall_action_weight"].fillna(EPSILON)

    #ep2
    new_users_only_weight_table["nodeUserID_overall_influence_weight"] = new_users_only_weight_table["nodeUserID_overall_influence_weight"]/new_users_only_weight_table["nodeUserID_overall_influence_weight"].sum()
    min_val = new_users_only_weight_table["nodeUserID_overall_influence_weight"].min()
    EPSILON = min_val/INFLUENCE_EPSILON_DIV_VAL
    print("\nEPSILON")
    print(EPSILON)
    new_users_only_weight_table["nodeUserID_overall_influence_weight"] = new_users_only_weight_table["nodeUserID_overall_influence_weight"].fillna(EPSILON)

    print("\nnew_users_only_weight_table")
    print(new_users_only_weight_table)

    return new_users_only_weight_table

def create_infoID_to_new_users_only_weight_table_dict(history_record_df,INFLUENCE_EPSILON_DIV_VAL):

    print("\nGetting overall action and influence weights...")
    groupby_add_cols = ["nodeUserID", "parentUserID", "informationID"]

    keep_cols = ["nodeUserID", "parentUserID", "informationID","nodeUserID_num_actions_this_timestep","parentUserID_influence_this_timestep", "nodeUserID_is_new"]
    # user_weight_table = history_record_df[keep_cols]

    new_parent_only_history_record_df = history_record_df[history_record_df["parentUserID_is_new"]==1]
    new_parent_only_history_record_df = new_parent_only_history_record_df[keep_cols]

    print("\nnew_parent_only_history_record_df")
    print(new_parent_only_history_record_df)



    #first just get influence dict, fix names after
    new_parent_only_history_record_df["parentUserID_overall_influence_weight"] = new_parent_only_history_record_df.groupby(["parentUserID", "informationID"])["parentUserID_influence_this_timestep"].transform("sum")
    new_parent_only_history_record_df = get_user_influence_from_parent_influence_V2_no_fillna(new_parent_only_history_record_df)
    new_parent_only_history_record_df["infoID_user"] = new_parent_only_history_record_df["nodeUserID"] + "_" + new_parent_only_history_record_df["informationID"]
    new_parent_only_history_record_df["infoID_parent"] = new_parent_only_history_record_df["parentUserID"] + "_" + new_parent_only_history_record_df["informationID"]
    user_to_infl_dict = convert_df_2_cols_to_dict(new_parent_only_history_record_df, "infoID_user", "nodeUserID_overall_influence_weight")



    #get new user action weights
    new_users_only_weight_table = history_record_df[history_record_df["nodeUserID_is_new"]==1].reset_index(drop=True)
    print("\nnew_users_only_weight_table")
    print(new_users_only_weight_table)

    new_users_only_weight_table["nodeUserID_overall_action_weight"] = new_users_only_weight_table.groupby(["nodeUserID", "informationID"])["nodeUserID_num_actions_this_timestep"].transform("sum")
    new_users_only_weight_table = new_users_only_weight_table.drop_duplicates().reset_index(drop=True)

    new_users_only_weight_table = new_users_only_weight_table.drop(["nodeUserID_num_actions_this_timestep","parentUserID_influence_this_timestep", "parentUserID"], axis=1)
    new_users_only_weight_table = new_users_only_weight_table.drop_duplicates().reset_index(drop=True)

    #fix names
    new_users_only_weight_table["infoID_user"] = new_users_only_weight_table["nodeUserID"] + "_" + new_users_only_weight_table["informationID"]
    new_users_only_weight_table["nodeUserID_overall_influence_weight"] = new_users_only_weight_table["infoID_user"].map(user_to_infl_dict)

    infoID_to_new_users_only_weight_table_dict = {}
    infoIDs = list(history_record_df["informationID"].unique())
    for infoID in infoIDs:
        cur_new_users_only_weight_table = new_users_only_weight_table[new_users_only_weight_table["informationID"]==infoID]

        cur_new_users_only_weight_table = cur_new_users_only_weight_table[["nodeUserID",  "nodeUserID_overall_action_weight", "nodeUserID_overall_influence_weight", "informationID"]]

        #ep1
        cur_new_users_only_weight_table["nodeUserID_overall_action_weight"] = cur_new_users_only_weight_table["nodeUserID_overall_action_weight"]/cur_new_users_only_weight_table["nodeUserID_overall_action_weight"].sum()
        min_val = cur_new_users_only_weight_table["nodeUserID_overall_action_weight"].min()
        EPSILON = min_val/INFLUENCE_EPSILON_DIV_VAL
        print("\nACTION EPSILON")
        print(EPSILON)
        cur_new_users_only_weight_table["nodeUserID_overall_action_weight"] = cur_new_users_only_weight_table["nodeUserID_overall_action_weight"].fillna(EPSILON)

        #ep2
        cur_new_users_only_weight_table["nodeUserID_overall_influence_weight"] = cur_new_users_only_weight_table["nodeUserID_overall_influence_weight"]/cur_new_users_only_weight_table["nodeUserID_overall_influence_weight"].sum()

        infl_temp = cur_new_users_only_weight_table[cur_new_users_only_weight_table["nodeUserID_overall_influence_weight"] != 0].reset_index(drop=True)["nodeUserID_overall_influence_weight"]
        min_val = infl_temp.min()
        EPSILON = min_val/INFLUENCE_EPSILON_DIV_VAL
        print("\nINFLUENCE EPSILON")
        print(EPSILON)
        cur_new_users_only_weight_table["nodeUserID_overall_influence_weight"] = cur_new_users_only_weight_table["nodeUserID_overall_influence_weight"].fillna(EPSILON)

        cur_new_users_only_weight_table = cur_new_users_only_weight_table.drop_duplicates()

        print("\ncur_new_users_only_weight_table")
        print(cur_new_users_only_weight_table)

        # sys.exit(0)

        infoID_to_new_users_only_weight_table_dict[infoID] = cur_new_users_only_weight_table


    return infoID_to_new_users_only_weight_table_dict

def get_new_user_edge_weight_dist_table(edge_weight_df):
    #new user edge weight dist table
    new_user_edge_weight_dist_table  = edge_weight_df[edge_weight_df["nodeUserID_is_new"]==1].reset_index(drop=True)
    new_to_new_edge_weight_table = new_user_edge_weight_dist_table[new_user_edge_weight_dist_table["parentUserID_is_new"]==1]
    new_to_new_edge_weight_table_no_self_loops = new_to_new_edge_weight_table[new_to_new_edge_weight_table["nodeUserID"] !=new_to_new_edge_weight_table["parentUserID"] ]
    new_to_new_edge_weight_table_with_self_loops = new_to_new_edge_weight_table[new_to_new_edge_weight_table["nodeUserID"] ==new_to_new_edge_weight_table["parentUserID"] ]

    print("\nnew_to_new_edge_weight_table_no_self_loops")
    print(new_to_new_edge_weight_table_no_self_loops)
    new_to_new_edge_weight_table_no_self_loops["parentUserID"] = "new_user"

    print("\nnew_to_new_edge_weight_table_with_self_loops")
    print(new_to_new_edge_weight_table_with_self_loops)
    new_to_new_edge_weight_table_with_self_loops["parentUserID"] = "self"

    new_to_new_edge_weight_table = pd.concat([new_to_new_edge_weight_table_no_self_loops, new_to_new_edge_weight_table_with_self_loops]).reset_index(drop=True)
    print("\nnew_to_new_edge_weight_table")
    print(new_to_new_edge_weight_table)

    new_to_old_edge_weight_table  = new_user_edge_weight_dist_table[new_user_edge_weight_dist_table["parentUserID_is_new"]==0]

    new_user_edge_weight_dist_table = pd.concat([new_to_old_edge_weight_table, new_to_new_edge_weight_table]).reset_index(drop=True)
    print("\nnew_user_edge_weight_dist_table")
    print(new_user_edge_weight_dist_table)

    new_user_edge_weight_dist_table["edge_weight"] = new_user_edge_weight_dist_table.groupby(["nodeUserID", "informationID", "parentUserID"])["edge_weight_this_timestep"].transform("sum")
    new_user_edge_weight_dist_table = new_user_edge_weight_dist_table.drop("edge_weight_this_timestep", axis=1)
    new_user_edge_weight_dist_table["edge_weight"]=new_user_edge_weight_dist_table["edge_weight"]/new_user_edge_weight_dist_table["edge_weight"].sum()

    print("\nnew_user_edge_weight_dist_table")
    print(new_user_edge_weight_dist_table)

    return new_user_edge_weight_dist_table

def get_infoID_to_new_user_edge_weight_dist_table_dict(edge_weight_df):
    #new user edge weight dist table
    new_user_edge_weight_dist_table  = edge_weight_df[edge_weight_df["nodeUserID_is_new"]==1].reset_index(drop=True)
    new_to_new_edge_weight_table = new_user_edge_weight_dist_table[new_user_edge_weight_dist_table["parentUserID_is_new"]==1]
    new_to_new_edge_weight_table_no_self_loops = new_to_new_edge_weight_table[new_to_new_edge_weight_table["nodeUserID"] !=new_to_new_edge_weight_table["parentUserID"] ]
    new_to_new_edge_weight_table_with_self_loops = new_to_new_edge_weight_table[new_to_new_edge_weight_table["nodeUserID"] ==new_to_new_edge_weight_table["parentUserID"] ]

    print("\nnew_to_new_edge_weight_table_no_self_loops")
    print(new_to_new_edge_weight_table_no_self_loops)
    new_to_new_edge_weight_table_no_self_loops["parentUserID"] = "new_user"

    print("\nnew_to_new_edge_weight_table_with_self_loops")
    print(new_to_new_edge_weight_table_with_self_loops)
    new_to_new_edge_weight_table_with_self_loops["parentUserID"] = "self"

    new_to_new_edge_weight_table = pd.concat([new_to_new_edge_weight_table_no_self_loops, new_to_new_edge_weight_table_with_self_loops]).reset_index(drop=True)
    print("\nnew_to_new_edge_weight_table")
    print(new_to_new_edge_weight_table)

    new_to_old_edge_weight_table  = new_user_edge_weight_dist_table[new_user_edge_weight_dist_table["parentUserID_is_new"]==0]

    new_user_edge_weight_dist_table = pd.concat([new_to_old_edge_weight_table, new_to_new_edge_weight_table]).reset_index(drop=True)
    print("\nnew_user_edge_weight_dist_table")
    print(new_user_edge_weight_dist_table)

    new_user_edge_weight_dist_table["edge_weight"] = new_user_edge_weight_dist_table.groupby(["nodeUserID", "informationID", "parentUserID"])["edge_weight_this_timestep"].transform("sum")
    new_user_edge_weight_dist_table = new_user_edge_weight_dist_table.drop("edge_weight_this_timestep", axis=1)

    infoID_to_new_user_edge_weight_dist_table_dict = {}
    infoIDs = edge_weight_df["informationID"].unique()

    for infoID in infoIDs:
        cur_new_user_edge_weight_dist_table = new_user_edge_weight_dist_table[new_user_edge_weight_dist_table["informationID"]==infoID].reset_index(drop=True)
        cur_new_user_edge_weight_dist_table = cur_new_user_edge_weight_dist_table[["nodeUserID", "parentUserID", "edge_weight", "informationID"]].drop_duplicates().reset_index(drop=True)
        cur_new_user_edge_weight_dist_table["edge_weight"]=cur_new_user_edge_weight_dist_table["edge_weight"]/cur_new_user_edge_weight_dist_table["edge_weight"].sum()
        infoID_to_new_user_edge_weight_dist_table_dict[infoID] = cur_new_user_edge_weight_dist_table

        print("\ncur_new_user_edge_weight_dist_table")
        print(cur_new_user_edge_weight_dist_table)

    return infoID_to_new_user_edge_weight_dist_table_dict

def get_infoID_to_new_user_edge_weight_dist_table_dict_v2_MORE_EXPLAINABLE(edge_weight_df):
    #new user edge weight dist table
    # new_user_edge_weight_dist_table  = edge_weight_df[edge_weight_df["nodeUserID_is_new"]==1].reset_index(drop=True)
    # new_to_new_edge_weight_table = new_user_edge_weight_dist_table[new_user_edge_weight_dist_table["parentUserID_is_new"]==1]


    new_users = edge_weight_df[edge_weight_df["nodeUserID_is_new"]==1]["nodeUserID"].unique()
    new_users = set(new_users)
    new_user_edge_weight_dist_table  = edge_weight_df[edge_weight_df["nodeUserID"].isin(new_users)].reset_index(drop=True)



    new_to_new_edge_weight_table = new_user_edge_weight_dist_table[new_user_edge_weight_dist_table["parentUserID_is_new"]==1]


    new_to_new_edge_weight_table_no_self_loops = new_to_new_edge_weight_table[new_to_new_edge_weight_table["nodeUserID"] !=new_to_new_edge_weight_table["parentUserID"] ]
    new_to_new_edge_weight_table_with_self_loops = new_to_new_edge_weight_table[new_to_new_edge_weight_table["nodeUserID"] ==new_to_new_edge_weight_table["parentUserID"] ]

    print("\nnew_to_new_edge_weight_table_no_self_loops")
    print(new_to_new_edge_weight_table_no_self_loops)
    new_to_new_edge_weight_table_no_self_loops["parentUserID"] = "new_user"

    print("\nnew_to_new_edge_weight_table_with_self_loops")
    print(new_to_new_edge_weight_table_with_self_loops)
    new_to_new_edge_weight_table_with_self_loops["parentUserID"] = "self"

    new_to_new_edge_weight_table = pd.concat([new_to_new_edge_weight_table_no_self_loops, new_to_new_edge_weight_table_with_self_loops]).reset_index(drop=True)
    print("\nnew_to_new_edge_weight_table")
    print(new_to_new_edge_weight_table)

    new_to_old_edge_weight_table  = new_user_edge_weight_dist_table[new_user_edge_weight_dist_table["parentUserID_is_new"]==0]

    new_user_edge_weight_dist_table = pd.concat([new_to_old_edge_weight_table, new_to_new_edge_weight_table]).reset_index(drop=True)
    print("\nnew_user_edge_weight_dist_table")
    print(new_user_edge_weight_dist_table)

    new_user_edge_weight_dist_table["edge_weight"] = new_user_edge_weight_dist_table.groupby(["nodeUserID", "informationID", "parentUserID"])["edge_weight_this_timestep"].transform("sum")
    new_user_edge_weight_dist_table = new_user_edge_weight_dist_table.drop("edge_weight_this_timestep", axis=1)

    infoID_to_new_user_edge_weight_dist_table_dict = {}
    infoIDs = edge_weight_df["informationID"].unique()

    for infoID in infoIDs:
        cur_new_user_edge_weight_dist_table = new_user_edge_weight_dist_table[new_user_edge_weight_dist_table["informationID"]==infoID].reset_index(drop=True)
        cur_new_user_edge_weight_dist_table = cur_new_user_edge_weight_dist_table[["nodeUserID", "parentUserID", "edge_weight", "informationID"]].drop_duplicates().reset_index(drop=True)
        cur_new_user_edge_weight_dist_table["edge_weight"]=cur_new_user_edge_weight_dist_table["edge_weight"]/cur_new_user_edge_weight_dist_table["edge_weight"].sum()
        infoID_to_new_user_edge_weight_dist_table_dict[infoID] = cur_new_user_edge_weight_dist_table

        print("\ncur_new_user_edge_weight_dist_table")
        print(cur_new_user_edge_weight_dist_table)

    return infoID_to_new_user_edge_weight_dist_table_dict

def get_initial_old_links(num_old_users, NUM_SAMPLE_ATTEMPTS, sampled_old_user_records,cur_all_users_edge_weight_dist_table):

    #get initial old user links
    print("\nGetting initial old user links...")
    all_users_set = sampled_old_user_records["nodeUserID"].unique()
    temp_all_users_edge_weight_dist_table = cur_all_users_edge_weight_dist_table[cur_all_users_edge_weight_dist_table["nodeUserID"].isin(all_users_set)].reset_index(drop=True)
    print("\ncur_all_users_edge_weight_dist_table")
    print(cur_all_users_edge_weight_dist_table)

    print("\ntemp_all_users_edge_weight_dist_table")
    print(temp_all_users_edge_weight_dist_table)
    all_initial_link_records = []
    cur_num_old_users = num_old_users
    cur_retrieved_user_set = set()
    for ATTEMPT in range(NUM_SAMPLE_ATTEMPTS):
        print("\ntemp table with mult factor")
        print(temp_all_users_edge_weight_dist_table["edge_weight"])

        #==================== sample some users ============
        initial_old_links = temp_all_users_edge_weight_dist_table.sample(n=cur_num_old_users, weights=temp_all_users_edge_weight_dist_table["edge_weight"])
        initial_old_links = initial_old_links.reset_index(drop=True)
        initial_old_links = initial_old_links.drop_duplicates("nodeUserID").reset_index(drop=True)
        print("\ninitial_old_links")
        print(initial_old_links)

        #================= count #of links so far =================
        num_unique_user_links_so_far = int(initial_old_links.shape[0])
        num_remaining_links_to_get = cur_num_old_users - num_unique_user_links_so_far
        print("\nnum_unique_user_links_so_far")
        print(num_unique_user_links_so_far)
        print("\nnum_remaining_links_to_get")
        print(num_remaining_links_to_get)

        #=============== add users to set ===============
        cur_users = set(initial_old_links["nodeUserID"])
        cur_retrieved_user_set.add(cur_users)
        cur_retrieved_user_set_size = len(ur_retrieved_user_set)
        print("\ncur_retrieved_user_set_size: %d"%cur_retrieved_user_set_size)

        old_users_so_far = all_users_set.intersection(cur_retrieved_user_set)

        cur_num_old_users = num_remaining_links_to_get

        # old_users_left_to_get=list(old_users_left_to_get["nodeUserID"])

        temp_all_users_edge_weight_dist_table = cur_all_users_edge_weight_dist_table[cur_all_users_edge_weight_dist_table["nodeUserID"].isin(old_users_left_to_get)]
        temp_all_users_edge_weight_dist_table=temp_all_users_edge_weight_dist_table.reset_index(drop=True)
        initial_old_links = initial_old_links.reset_index(drop=True)
        all_initial_link_records.append(initial_old_links)
        # cur_combined_links = pd.concat(all_initial_link_records).reset_index(drop=True)

        if cur_num_old_users == 0:
            break

    initial_old_links = pd.concat(all_initial_link_records).reset_index(drop=True)
    print(initial_old_links.shape[0])
    print(num_old_users)
    if initial_old_links.shape[0] > num_old_users:
        print("\nError! initial_old_links.shape[0] > num_old_users")
        sys.exit(0)

    print("\nNeed to get %d more users"%cur_num_old_users)

    if cur_num_old_users != 0:
        remaining_initial_old_link_records = []
        print("\nGetting remaining old users...")
        for old_idx,old_user in enumerate(old_users_left_to_get):
            print("\nGetting old user %d of %d"%(old_idx, cur_num_old_users))

            temp2 = cur_all_users_edge_weight_dist_table[cur_all_users_edge_weight_dist_table["nodeUserID"].isin([old_user])].reset_index(drop=True)
            print(temp2)
            old_link_record = temp2.sample(n=1, weights=temp2["edge_weight"])
            remaining_initial_old_link_records.append(old_link_record)
            print(old_link_record)

            if len(remaining_initial_old_link_records) == old_idx+1:
                break

        remaining_initial_old_link_records = pd.concat(remaining_initial_old_link_records).reset_index(drop=True)
        initial_old_links = pd.concat([initial_old_links, remaining_initial_old_link_records]).reset_index(drop=True)



    print("\ninitial_old_links")
    print(initial_old_links)
    print(initial_old_links.shape[0])
    print(num_old_users)
    if initial_old_links.shape[0] != num_old_users:
        print("\nError! initial_old_links.shape[0] != num_old_users")
        sys.exit(0)
    num_initial_old_users = initial_old_links["nodeUserID"].nunique()
    # print(num_initial_old_users)
    # print(num_old_users)
    # if num_initial_old_users != num_old_users:
    #     print("\nError! num_initial_old_users != num_old_users")
    #     print("\nAre there nan vals:")
    #     print(initial_old_links["nodeUserID"].isnull().values.any())
    #     sys.exit(0)
    print("Counts are ok!")


    return initial_old_links

def get_initial_links(total_num_users, NUM_SAMPLE_ATTEMPTS, sampled_user_records,cur_all_users_edge_weight_dist_table):

    #get initial old user links
    print("\nGetting initial user links...")
    all_sampled_users_set = set(sampled_user_records["nodeUserID"])
    temp_all_users_edge_weight_dist_table = cur_all_users_edge_weight_dist_table[cur_all_users_edge_weight_dist_table["nodeUserID"].isin(all_sampled_users_set)].reset_index(drop=True)
    print("\ncur_all_users_edge_weight_dist_table")
    print(cur_all_users_edge_weight_dist_table)
    print("\ntemp_all_users_edge_weight_dist_table")
    print(temp_all_users_edge_weight_dist_table)
    all_initial_link_records = []


    num_remaining_users = total_num_users
    retrieved_user_set_so_far = set()
    #attempt
    for ATTEMPT in range(NUM_SAMPLE_ATTEMPTS):
        print("\ntemp table with mult factor")
        print(temp_all_users_edge_weight_dist_table["edge_weight"])

        #==================== sample some users ============
        initial_links = temp_all_users_edge_weight_dist_table.sample(n=num_remaining_users, weights=temp_all_users_edge_weight_dist_table["edge_weight"])
        initial_links = initial_links.reset_index(drop=True)
        initial_links = initial_links.drop_duplicates("nodeUserID").reset_index(drop=True)
        print("\ninitial_links")
        print(initial_links)

        #=============== add users to set ===============
        cur_user_set = set(initial_links["nodeUserID"])
        retrieved_user_set_so_far = retrieved_user_set_so_far.union(cur_user_set)
        num_retrieved_users_so_far = len(retrieved_user_set_so_far)
        # print("\nnum_retrieved_users_so_far: %d"%num_retrieved_users_so_far)

        users_yet_to_be_retrieved_set =all_sampled_users_set -  retrieved_user_set_so_far
        num_remaining_users = len(users_yet_to_be_retrieved_set)
        # print("\nnum_remaining_users: %d"%num_remaining_users)
        print("\nThere are %d total users. We have %d so far. We need %d more"%(total_num_users,num_retrieved_users_so_far, num_remaining_users))

        #update table
        temp_all_users_edge_weight_dist_table =temp_all_users_edge_weight_dist_table[temp_all_users_edge_weight_dist_table["nodeUserID"].isin(users_yet_to_be_retrieved_set)].reset_index(drop=True)
        all_initial_link_records.append(initial_links)

        if num_remaining_users == 0:
            break


    initial_links = pd.concat(all_initial_link_records).reset_index(drop=True)
    print(initial_links.shape[0])
    print(total_num_users)
    if initial_links.shape[0] > total_num_users:
        print("\nError! initial_links.shape[0] > total_num_users")
        sys.exit(0)

    print("\nThere are %d total users. We have %d so far. We need %d more"%(total_num_users,num_retrieved_users_so_far, num_remaining_users))

    if num_remaining_users != 0:
        remaining_initial_link_records = []
        users_yet_to_be_retrieved_set = list(users_yet_to_be_retrieved_set)
        print("\nGetting remaining old users...")
        for user_idx,user in enumerate(users_yet_to_be_retrieved_set):
            print("\nGetting user %d of %d"%(user_idx, num_remaining_users))

            temp2 = cur_all_users_edge_weight_dist_table[cur_all_users_edge_weight_dist_table["nodeUserID"].isin([user])].reset_index(drop=True)
            print(temp2)
            link_record = temp2.sample(n=1, weights=temp2["edge_weight"])
            remaining_initial_link_records.append(link_record)
            print(link_record)

            # if len(remaining_initial_link_records) == user_idx+1:
            #     break

        remaining_initial_link_records = pd.concat(remaining_initial_link_records).reset_index(drop=True)
        initial_links = pd.concat([initial_links, remaining_initial_link_records]).reset_index(drop=True)
    # sys.exit(0)



    print("\ninitial_links")
    print(initial_links)
    print(initial_links.shape[0])
    print(total_num_users)
    if initial_links.shape[0] != total_num_users:
        print("\nError! initial_links.shape[0] != total_num_users")
        sys.exit(0)

    print("Counts are ok!")


    return initial_links

def assign_parents_to_new_user_placeholders(initial_new_links, cur_new_users_only_weight_table):

    initial_new_links["parentUserID_is_new"] = [1 if "new_user" in parentUserID else 0 for parentUserID in list(initial_new_links["parentUserID"])]
    print("\ninitial_new_links")
    print(initial_new_links)


    new_new_links = initial_new_links[(initial_new_links["parentUserID_is_new"]==1)].reset_index(drop=True)
    print("\nnew_new_links")
    print(new_new_links)

    if new_new_links.shape[0] > 0:

        total_initial_links = initial_new_links.shape[0]

        new_old_links = initial_new_links[(initial_new_links["parentUserID_is_new"]==0)].reset_index(drop=True)
        print("\nnew_old_links")
        print(new_old_links)
        # sys.exit(0)

        #get likely new users
        num_initial_new_user_actions = new_new_links.shape[0]
        print("\nnum_initial_new_user_actions: %d"%num_initial_new_user_actions)

        new_user_parents = cur_new_users_only_weight_table.sample(n=num_initial_new_user_actions, weights=cur_new_users_only_weight_table["nodeUserID_overall_influence_weight"], replace=True)
        new_user_parents = new_user_parents.reset_index(drop=True)
        print("\nnew_user_parents")
        print(new_user_parents)

        new_new_links["parentUserID"] = new_user_parents["nodeUserID"].copy()
        print("\nnew_new_links")
        print(new_new_links)


        initial_new_links=pd.concat([new_new_links,new_old_links]).reset_index(drop=True)

        num_final_links = initial_new_links.shape[0]

        if total_initial_links != num_final_links:
            print(total_initial_links)
            print(num_final_links)
            print("\nError! total_initial_links != num_final_links. Exiting!")
            sys.exit(0)

    return initial_new_links








def get_user_sim_pred_records_for_cur_infoID_v2_add_more_info( infoID_materials_tuple,NUM_LINK_SAMPLE_ATTEMPTS,ADD_OLD_USERS_TO_NEW_USERS_IF_TOO_MANY, timestep, DEBUG_PRINT):

    all_pred_count_dfs_for_cur_infoID_for_verif = []
    all_timestep_dfs_for_cur_infoID = []

    #get infoID materials
    infoID ,hyp_infoID, cur_pred_df, cur_all_users_weight_table, cur_new_users_only_weight_table, cur_all_users_edge_weight_dist_table,cur_new_user_edge_weight_dist_table,num_infoIDs,infoID_idx,CUR_INFO_ID_NEW_USER_COUNTER = infoID_materials_tuple

    print("Simulating infoID %d of %d: %s"%((infoID_idx+1), num_infoIDs, infoID))

    #get numbers
    num_new_users = int(cur_pred_df["num_new_users"].iloc[0])
    num_old_users = int(cur_pred_df["num_old_users"].iloc[0])
    num_actions = int(cur_pred_df["num_actions"].iloc[0])

    def debug_print(print_val="\n"):
        if DEBUG_PRINT == True:
            print(str(print_val))




    #============= get old users first -> check if there's a conflict =============

    num_old_users_in_weight_df = cur_all_users_weight_table["nodeUserID"].nunique()
    old_user_diff = num_old_users_in_weight_df - num_old_users
    debug_print("\nNum old users in weight df: %d; num predicted: %d; Difference: %d"%(num_old_users_in_weight_df, num_old_users, old_user_diff))


    # if (old_user_diff >= 0) and (num_old_users > 0):
    if (old_user_diff >= 0):
        # print("\nnum old users: %d"%num_old_users)

        old_user_weights = cur_all_users_weight_table["nodeUserID_overall_action_weight"]
        print("\nnum old users: %d, old user weights: %s"%(num_old_users, str(old_user_weights )))
        #get OLD users

        try:
            sampled_old_user_records = cur_all_users_weight_table.sample(n=num_old_users ,weights=cur_all_users_weight_table["nodeUserID_overall_action_weight"], replace=False)
        except ValueError:
            sampled_old_user_records = cur_all_users_weight_table.sample(n=num_old_users ,replace=False)
    else:
        #get what you can
        amount_of_old_users_possible_to_sample = num_old_users_in_weight_df
        debug_print("\nWe can only get %d old users"%amount_of_old_users_possible_to_sample)
        sampled_old_user_records= cur_all_users_weight_table.sample(n=amount_of_old_users_possible_to_sample ,weights=cur_all_users_weight_table["nodeUserID_overall_action_weight"], replace=False)

        remaining_old_users = num_old_users - amount_of_old_users_possible_to_sample

        initial_cur_pred_df = cur_pred_df.copy()
        debug_print("\nThere are %d remaining_old_users that need to be sampled"%remaining_old_users )
        cur_pred_df["num_old_users"] = amount_of_old_users_possible_to_sample
        num_old_users = amount_of_old_users_possible_to_sample

        if ADD_OLD_USERS_TO_NEW_USERS_IF_TOO_MANY == True:
            debug_print("\nADD_OLD_USERS_TO_NEW_USERS_IF_TOO_MANY==True, so we have to increase # of new users")
            num_new_users = num_new_users + remaining_old_users

            cur_pred_df["num_new_users"] = num_new_users

        debug_print("\nInitial cur pred df")
        debug_print(initial_cur_pred_df)

        debug_print("\nModified cur pred df")
        debug_print(cur_pred_df)

    debug_print("\nsampled_old_user_records")
    sampled_old_user_records = sampled_old_user_records.reset_index(drop=True)
    # sampled_old_user_records = sampled_old_user_records.rename(columns={"latest_parent":"parentUserID"})
    # sampled_old_user_records = sampled_old_user_records.reset_index(drop=True)
    debug_print(sampled_old_user_records)

    print(sampled_old_user_records)
    # sys.exit(0)

    # num_old_users = sampled_old_user_records.shape[0]

    # sys.exit(0)

    #=================== this is for faster runtime ===================
    debug_print("\ncur_all_users_edge_weight_dist_table")
    debug_print(cur_all_users_edge_weight_dist_table)

    # cur_all_users_edge_weight_dist_table["edge_weight"] = cur_all_users_edge_weight_dist_table["edge_weight"]*WEIGHT_MULT_FACTOR
    # debug_print("\ncur_all_users_edge_weight_dist_table with mult factor")
    # debug_print(cur_all_users_edge_weight_dist_table)

    print("\nnum_old_users: %d"%num_old_users)

    if cur_all_users_edge_weight_dist_table["edge_weight"].sum() == 0:
        print("\nError! Weight table weights sum to 0... Num old users: %d; table: \n%s"%(num_old_users, str(cur_all_users_edge_weight_dist_table)))
    initial_old_links = get_initial_links_v2_debug_print(num_old_users, NUM_LINK_SAMPLE_ATTEMPTS, sampled_old_user_records,cur_all_users_edge_weight_dist_table,DEBUG_PRINT)

    print(initial_old_links)
    # sys.exit(0)


    #========================== get new users ==========================
    #get new users
    try:
        sampled_new_user_records = cur_new_users_only_weight_table.sample(n=num_new_users ,weights=cur_new_users_only_weight_table["nodeUserID_overall_action_weight"], replace=False)
    except ValueError:
        try:
            sampled_new_user_records = cur_new_users_only_weight_table.sample(n=num_new_users ,weights=cur_new_users_only_weight_table["nodeUserID_overall_action_weight"], replace=True)
        except ValueError:
            try:
                print("\nThere were not enough new users from prev timestep. Sampling from old user table without replacement.")
                sampled_new_user_records = cur_all_users_weight_table.sample(n=num_new_users ,weights=cur_all_users_weight_table["nodeUserID_overall_action_weight"], replace=False)
            except:
                print("\nThere were not enough new users from prev timestep. Sampling from old user table WITH replacement.")
                sampled_new_user_records = cur_all_users_weight_table.sample(n=num_new_users ,weights=cur_all_users_weight_table["nodeUserID_overall_action_weight"], replace=True)




    debug_print("\nSampled new users")
    sampled_new_user_records = sampled_new_user_records.reset_index(drop=True)
    debug_print(sampled_new_user_records)

    # cur_new_user_edge_weight_dist_table = infoID_to_new_user_edge_weight_dist_table_dict[infoID]

    debug_print("\nThese records say whose attributes the new users will copy.")
    # sampled_new_user_records["original_nodeUserID"] = sampled_new_user_records["nodeUserID"].copy()
    sampled_new_user_records["new_nodeUserID"] = ["new_synthetic_user_%d"%(CUR_INFO_ID_NEW_USER_COUNTER + i+1) for i in range(num_new_users)]
    CUR_INFO_ID_NEW_USER_COUNTER+=num_new_users
    debug_print(sampled_new_user_records)

    debug_print("\nNow we must get a custom distribution table for the new users we have just created")
    temp = sampled_new_user_records[["nodeUserID", "new_nodeUserID","informationID"]].copy()
    cur_custom_new_user_edge_weight_dist_table = pd.merge(temp,cur_new_user_edge_weight_dist_table,on=["nodeUserID","informationID"], how="inner")
    debug_print("\ncur_custom_new_user_edge_weight_dist_table")
    debug_print(cur_custom_new_user_edge_weight_dist_table)
    cur_custom_new_user_edge_weight_dist_table["original_nodeUserID"]=cur_custom_new_user_edge_weight_dist_table["nodeUserID"].copy()
    cur_custom_new_user_edge_weight_dist_table["nodeUserID"] =cur_custom_new_user_edge_weight_dist_table["new_nodeUserID"]
    # cur_custom_new_user_edge_weight_dist_table= cur_custom_new_user_edge_weight_dist_table.rename(columns={"new_nodeUserID":"nodeUserID"})

    debug_print("\ncur_custom_new_user_edge_weight_dist_table")
    debug_print(cur_custom_new_user_edge_weight_dist_table)

    nodeUserIDs = cur_custom_new_user_edge_weight_dist_table["nodeUserID"]
    parentUserIDs = cur_custom_new_user_edge_weight_dist_table["parentUserID"]
    cur_custom_new_user_edge_weight_dist_table["parentUserID"] = [nodeUserID if parentUserID=="self" else parentUserID for nodeUserID,parentUserID in zip(nodeUserIDs, parentUserIDs)]

    debug_print("\ncur_custom_new_user_edge_weight_dist_table with fixed self loops")
    debug_print(cur_custom_new_user_edge_weight_dist_table)

    print("\ncur_custom_new_user_edge_weight_dist_table")
    print(cur_custom_new_user_edge_weight_dist_table)
    # sys.exit(0)

    sampled_new_user_records["original_nodeUserID"]=sampled_new_user_records["nodeUserID"].copy()
    sampled_new_user_records["nodeUserID"] =sampled_new_user_records["new_nodeUserID"]
    debug_print(sampled_new_user_records)

    parent_val_counts = cur_custom_new_user_edge_weight_dist_table["parentUserID"].value_counts()
    debug_print("\nparent_val_counts")
    debug_print(parent_val_counts)

    orig_user_val_counts = cur_custom_new_user_edge_weight_dist_table["original_nodeUserID"].value_counts()
    debug_print("\norig_user_val_counts")
    debug_print(orig_user_val_counts)



    #=================== add a checkpoint here ===============
    verify_pred_counts_so_far_v2_debug_print(sampled_new_user_records,sampled_old_user_records,num_old_users, num_new_users, DEBUG_PRINT)

    #get initial new links
    debug_print("\nGetting initial new links...")
    initial_new_links = get_initial_links_v2_debug_print(num_new_users, NUM_LINK_SAMPLE_ATTEMPTS, sampled_new_user_records,cur_custom_new_user_edge_weight_dist_table,DEBUG_PRINT)
    # initial_new_links["parentUserID"]="new_user"

    try:
        initial_new_links = assign_parents_to_new_user_placeholders_v2_debug_print(initial_new_links, cur_new_users_only_weight_table, DEBUG_PRINT)
    except ValueError:
        initial_new_links = assign_parents_to_new_user_placeholders_v2_debug_print(initial_new_links, cur_all_users_weight_table, DEBUG_PRINT)


    debug_print("\ninitial_new_links")
    debug_print(initial_new_links)

    # sys.exit(0)

    #=================== combine old and new users ===============
    verify_pred_counts_so_far_v2_debug_print(initial_new_links,initial_old_links,num_old_users, num_new_users, DEBUG_PRINT)

    all_initial_action_records = pd.concat([initial_old_links, initial_new_links]).reset_index(drop=True)
    all_initial_action_records=all_initial_action_records[["nodeUserID", "parentUserID", "informationID", "edge_weight"]]
    parent_val_counts = all_initial_action_records["parentUserID"].value_counts()
    debug_print("\nparent_val_counts")
    debug_print(parent_val_counts)


    debug_print("\nnum_actions: %d"%num_actions)
    num_all_initial_action_records = all_initial_action_records.shape[0]
    debug_print("num_all_initial_action_records: %d"%num_all_initial_action_records)

    num_remaining_actions = num_actions - num_all_initial_action_records
    debug_print("num_remaining_actions: %d"%num_remaining_actions)

    #get likely links
    #first make a combined edge weight table
    new_and_old_link_table = pd.concat([cur_custom_new_user_edge_weight_dist_table ,cur_all_users_edge_weight_dist_table]).reset_index(drop=True)
    debug_print("\nnew_and_old_link_table")
    debug_print(new_and_old_link_table)

    #lets make sure it only has the users we want
    users_so_far = all_initial_action_records["nodeUserID"].unique()
    new_and_old_link_table = new_and_old_link_table[new_and_old_link_table["nodeUserID"].isin(users_so_far)].reset_index(drop=True)
    debug_print("\nnew_and_old_link_table")
    debug_print(new_and_old_link_table)

    new_and_old_link_table_val_counts = new_and_old_link_table["parentUserID"].value_counts()
    debug_print("\nnew_and_old_link_table_val_counts")
    debug_print(new_and_old_link_table_val_counts)

    records_to_concat = [all_initial_action_records]
    if num_remaining_actions > 0:

        #get more users
        remaining_sampled_links = new_and_old_link_table.sample(n=num_remaining_actions, weights=new_and_old_link_table["edge_weight"], replace=True)
        remaining_sampled_links=remaining_sampled_links.reset_index(drop=True)
        debug_print("\nremaining_sampled_links")
        debug_print(remaining_sampled_links)

        #remaining_sampled_links = assign_parents_to_new_user_placeholders_v2_debug_print(remaining_sampled_links, cur_new_users_only_weight_table, DEBUG_PRINT)

        try:
            remaining_sampled_links = assign_parents_to_new_user_placeholders_v2_debug_print(remaining_sampled_links, cur_new_users_only_weight_table, DEBUG_PRINT)
        except ValueError:
            remaining_sampled_links = assign_parents_to_new_user_placeholders_v2_debug_print(remaining_sampled_links, cur_all_users_weight_table, DEBUG_PRINT)

        debug_print("\nremaining_sampled_links with new users fixed")
        debug_print(remaining_sampled_links)

        records_to_concat.append(remaining_sampled_links)

    #     #complete records
    #     final_link_records = pd.concat([remaining_sampled_links, all_initial_action_records ])
    # else:
    final_link_records = pd.concat(records_to_concat)


    # final_link_records["edge_weight"] = final_link_records.groupby(["nodeUserID", "parentUserID"])["nodeUserID"].transform("count")
    # final_link_records = final_link_records.sort_values("edge_weight", ascending=False)
    final_link_records=final_link_records[["nodeUserID", "parentUserID", "informationID", "edge_weight"]]
    final_link_records=final_link_records.reset_index(drop=True)
    debug_print("\nfinal_link_records")
    debug_print(final_link_records)


    final_parent_pred_val_counts = final_link_records["parentUserID"].value_counts()
    debug_print("\nfinal_parent_pred_val_counts")
    debug_print(final_parent_pred_val_counts)

    total_actions_gotten = final_link_records.shape[0]

    debug_print(total_actions_gotten)
    debug_print(num_actions)
    if total_actions_gotten != num_actions:
        debug_print("\nError! total_actions_gotten != num_actions")
        sys.exit(0)

    final_link_records["timestep"] = timestep
    # infoID_to_df_list_dict[infoID].append(final_link_records)

    final_link_records["nodeUserID_is_new"] = [1 if ("new_synthetic" in user) else 0 for user in list(final_link_records["nodeUserID"])]
    final_link_records["parentUserID_is_new"] = [1 if ("new_synthetic" in user) else 0 for user in list(final_link_records["parentUserID"])]

    print("\nsampled_new_user_records")
    print(sampled_new_user_records)
    cols = list(sampled_new_user_records)
    for col in cols:
        print(sampled_new_user_records[col])
    # sys.exit(0)

    full_proba_table = pd.concat([sampled_old_user_records, sampled_new_user_records]).reset_index(drop=True)
    print("\nfull_proba_table")
    print(full_proba_table)
    full_proba_table = full_proba_table[["nodeUserID", "nodeUserID_overall_action_weight", "nodeUserID_overall_influence_weight"]]
    float_cols = ["nodeUserID_overall_action_weight", "nodeUserID_overall_influence_weight"]
    for col in float_cols:
        full_proba_table[col] = full_proba_table[col].astype("float64")
        # final_link_records[col] = final_link_records[col].astype("float64")

    final_link_records["nodeUserID"]=final_link_records["nodeUserID"].astype("str")
    full_proba_table["nodeUserID"]=full_proba_table["nodeUserID"].astype("str")
    size_before_merge = final_link_records.shape[0]
    final_link_records = pd.merge(final_link_records, full_proba_table, on="nodeUserID", how="inner")


    size_after_merge = final_link_records.shape[0]
    print(size_before_merge)
    print(size_after_merge)
    if size_before_merge != size_after_merge:
        print("\nError! size_before_merge != size_after_merge")
        sys.exit(0)

    debug_print("\nfinal_link_records")
    debug_print(final_link_records)

    print("Done simulating infoID %d of %d: %s"%((infoID_idx+1), num_infoIDs, infoID))

    return (final_link_records,infoID, CUR_INFO_ID_NEW_USER_COUNTER)







def get_user_sim_pred_records_for_cur_infoID( infoID_materials_tuple,NUM_LINK_SAMPLE_ATTEMPTS,ADD_OLD_USERS_TO_NEW_USERS_IF_TOO_MANY, timestep, DEBUG_PRINT):

    all_pred_count_dfs_for_cur_infoID_for_verif = []
    all_timestep_dfs_for_cur_infoID = []

    #get infoID materials
    infoID ,hyp_infoID, cur_pred_df, cur_all_users_weight_table, cur_new_users_only_weight_table, cur_all_users_edge_weight_dist_table,cur_new_user_edge_weight_dist_table,num_infoIDs,infoID_idx  = infoID_materials_tuple

    print("Simulating infoID %d of %d: %s"%((infoID_idx+1), num_infoIDs, infoID))

    #get numbers
    num_new_users = int(cur_pred_df["num_new_users"].iloc[0])
    num_old_users = int(cur_pred_df["num_old_users"].iloc[0])
    num_actions = int(cur_pred_df["num_actions"].iloc[0])

    def debug_print(print_val="\n"):
        if DEBUG_PRINT == True:
            print(str(print_val))




    #============= get old users first -> check if there's a conflict =============

    num_old_users_in_weight_df = cur_all_users_weight_table["nodeUserID"].nunique()
    old_user_diff = num_old_users_in_weight_df - num_old_users
    debug_print("\nNum old users in weight df: %d; num predicted: %d; Difference: %d"%(num_old_users_in_weight_df, num_old_users, old_user_diff))


    if old_user_diff >= 0:
        #get OLD users
        sampled_old_user_records = cur_all_users_weight_table.sample(n=num_old_users ,weights=cur_all_users_weight_table["nodeUserID_overall_action_weight"], replace=False)
    else:
        #get what you can
        amount_of_old_users_possible_to_sample = num_old_users_in_weight_df
        debug_print("\nWe can only get %d old users"%amount_of_old_users_possible_to_sample)
        sampled_old_user_records= cur_all_users_weight_table.sample(n=amount_of_old_users_possible_to_sample ,weights=cur_all_users_weight_table["nodeUserID_overall_action_weight"], replace=False)

        remaining_old_users = num_old_users - amount_of_old_users_possible_to_sample

        initial_cur_pred_df = cur_pred_df.copy()
        debug_print("\nThere are %d remaining_old_users that need to be sampled"%remaining_old_users )
        cur_pred_df["num_old_users"] = amount_of_old_users_possible_to_sample
        num_old_users = amount_of_old_users_possible_to_sample

        if ADD_OLD_USERS_TO_NEW_USERS_IF_TOO_MANY == True:
            debug_print("\nADD_OLD_USERS_TO_NEW_USERS_IF_TOO_MANY==True, so we have to increase # of new users")
            num_new_users = num_new_users + remaining_old_users

            cur_pred_df["num_new_users"] = num_new_users

        debug_print("\nInitial cur pred df")
        debug_print(initial_cur_pred_df)

        debug_print("\nModified cur pred df")
        debug_print(cur_pred_df)

    debug_print("\nsampled_old_user_records")
    sampled_old_user_records = sampled_old_user_records.reset_index(drop=True)
    # sampled_old_user_records = sampled_old_user_records.rename(columns={"latest_parent":"parentUserID"})
    # sampled_old_user_records = sampled_old_user_records.reset_index(drop=True)
    debug_print(sampled_old_user_records)

    print(sampled_old_user_records)
    # sys.exit(0)

    # num_old_users = sampled_old_user_records.shape[0]

    # sys.exit(0)

    #=================== this is for faster runtime ===================
    debug_print("\ncur_all_users_edge_weight_dist_table")
    debug_print(cur_all_users_edge_weight_dist_table)

    # cur_all_users_edge_weight_dist_table["edge_weight"] = cur_all_users_edge_weight_dist_table["edge_weight"]*WEIGHT_MULT_FACTOR
    # debug_print("\ncur_all_users_edge_weight_dist_table with mult factor")
    # debug_print(cur_all_users_edge_weight_dist_table)

    print("\nnum_old_users: %d"%num_old_users)

    if cur_all_users_edge_weight_dist_table["edge_weight"].sum() == 0:
        print("\nError! Weight table weights sum to 0... Num old users: %d; table: \n%s"%num_old_users, str(cur_all_users_edge_weight_dist_table))
    initial_old_links = get_initial_links_v2_debug_print(num_old_users, NUM_LINK_SAMPLE_ATTEMPTS, sampled_old_user_records,cur_all_users_edge_weight_dist_table,DEBUG_PRINT)

    print(initial_old_links)
    # sys.exit(0)


    #========================== get new users ==========================
    #get new users
    try:
        sampled_new_user_records = cur_new_users_only_weight_table.sample(n=num_new_users ,weights=cur_new_users_only_weight_table["nodeUserID_overall_action_weight"], replace=False)
    except ValueError:
        try:
            sampled_new_user_records = cur_new_users_only_weight_table.sample(n=num_new_users ,weights=cur_new_users_only_weight_table["nodeUserID_overall_action_weight"], replace=True)
        except ValueError:
            try:
                print("\nThere were not enough new users from prev timestep. Sampling from old user table without replacement.")
                sampled_new_user_records = cur_all_users_weight_table.sample(n=num_new_users ,weights=cur_all_users_weight_table["nodeUserID_overall_action_weight"], replace=False)
            except:
                print("\nThere were not enough new users from prev timestep. Sampling from old user table WITH replacement.")
                sampled_new_user_records = cur_all_users_weight_table.sample(n=num_new_users ,weights=cur_all_users_weight_table["nodeUserID_overall_action_weight"], replace=True)




    debug_print("\nSampled new users")
    sampled_new_user_records = sampled_new_user_records.reset_index(drop=True)
    debug_print(sampled_new_user_records)

    # cur_new_user_edge_weight_dist_table = infoID_to_new_user_edge_weight_dist_table_dict[infoID]

    debug_print("\nThese records say whose attributes the new users will copy.")
    # sampled_new_user_records["original_nodeUserID"] = sampled_new_user_records["nodeUserID"].copy()
    sampled_new_user_records["new_nodeUserID"] = ["synthetic_user_%d"%(i+1) for i in range(num_new_users)]
    debug_print(sampled_new_user_records)

    debug_print("\nNow we must get a custom distribution table for the new users we have just created")
    temp = sampled_new_user_records[["nodeUserID", "new_nodeUserID","informationID"]].copy()
    cur_custom_new_user_edge_weight_dist_table = pd.merge(temp,cur_new_user_edge_weight_dist_table,on=["nodeUserID","informationID"], how="inner")
    debug_print("\ncur_custom_new_user_edge_weight_dist_table")
    debug_print(cur_custom_new_user_edge_weight_dist_table)
    cur_custom_new_user_edge_weight_dist_table["original_nodeUserID"]=cur_custom_new_user_edge_weight_dist_table["nodeUserID"].copy()
    cur_custom_new_user_edge_weight_dist_table["nodeUserID"] =cur_custom_new_user_edge_weight_dist_table["new_nodeUserID"]
    # cur_custom_new_user_edge_weight_dist_table= cur_custom_new_user_edge_weight_dist_table.rename(columns={"new_nodeUserID":"nodeUserID"})

    debug_print("\ncur_custom_new_user_edge_weight_dist_table")
    debug_print(cur_custom_new_user_edge_weight_dist_table)

    nodeUserIDs = cur_custom_new_user_edge_weight_dist_table["nodeUserID"]
    parentUserIDs = cur_custom_new_user_edge_weight_dist_table["parentUserID"]
    cur_custom_new_user_edge_weight_dist_table["parentUserID"] = [nodeUserID if parentUserID=="self" else parentUserID for nodeUserID,parentUserID in zip(nodeUserIDs, parentUserIDs)]

    debug_print("\ncur_custom_new_user_edge_weight_dist_table with fixed self loops")
    debug_print(cur_custom_new_user_edge_weight_dist_table)

    sampled_new_user_records["original_nodeUserID"]=sampled_new_user_records["nodeUserID"].copy()
    sampled_new_user_records["nodeUserID"] =sampled_new_user_records["new_nodeUserID"]
    debug_print(sampled_new_user_records)

    parent_val_counts = cur_custom_new_user_edge_weight_dist_table["parentUserID"].value_counts()
    debug_print("\nparent_val_counts")
    debug_print(parent_val_counts)

    orig_user_val_counts = cur_custom_new_user_edge_weight_dist_table["original_nodeUserID"].value_counts()
    debug_print("\norig_user_val_counts")
    debug_print(orig_user_val_counts)



    #=================== add a checkpoint here ===============
    verify_pred_counts_so_far_v2_debug_print(sampled_new_user_records,sampled_old_user_records,num_old_users, num_new_users, DEBUG_PRINT)

    #get initial new links
    debug_print("\nGetting initial new links...")
    initial_new_links = get_initial_links_v2_debug_print(num_new_users, NUM_LINK_SAMPLE_ATTEMPTS, sampled_new_user_records,cur_custom_new_user_edge_weight_dist_table,DEBUG_PRINT)
    # initial_new_links["parentUserID"]="new_user"

    try:
        initial_new_links = assign_parents_to_new_user_placeholders_v2_debug_print(initial_new_links, cur_new_users_only_weight_table, DEBUG_PRINT)
    except ValueError:
        initial_new_links = assign_parents_to_new_user_placeholders_v2_debug_print(initial_new_links, cur_all_users_weight_table, DEBUG_PRINT)


    debug_print("\ninitial_new_links")
    debug_print(initial_new_links)

    # sys.exit(0)

    #=================== combine old and new users ===============
    verify_pred_counts_so_far_v2_debug_print(initial_new_links,initial_old_links,num_old_users, num_new_users, DEBUG_PRINT)

    all_initial_action_records = pd.concat([initial_old_links, initial_new_links]).reset_index(drop=True)
    all_initial_action_records=all_initial_action_records[["nodeUserID", "parentUserID", "informationID", "edge_weight"]]
    parent_val_counts = all_initial_action_records["parentUserID"].value_counts()
    debug_print("\nparent_val_counts")
    debug_print(parent_val_counts)


    debug_print("\nnum_actions: %d"%num_actions)
    num_all_initial_action_records = all_initial_action_records.shape[0]
    debug_print("num_all_initial_action_records: %d"%num_all_initial_action_records)

    num_remaining_actions = num_actions - num_all_initial_action_records
    debug_print("num_remaining_actions: %d"%num_remaining_actions)

    #get likely links
    #first make a combined edge weight table
    new_and_old_link_table = pd.concat([cur_custom_new_user_edge_weight_dist_table ,cur_all_users_edge_weight_dist_table]).reset_index(drop=True)
    debug_print("\nnew_and_old_link_table")
    debug_print(new_and_old_link_table)

    #lets make sure it only has the users we want
    users_so_far = all_initial_action_records["nodeUserID"].unique()
    new_and_old_link_table = new_and_old_link_table[new_and_old_link_table["nodeUserID"].isin(users_so_far)].reset_index(drop=True)
    debug_print("\nnew_and_old_link_table")
    debug_print(new_and_old_link_table)

    new_and_old_link_table_val_counts = new_and_old_link_table["parentUserID"].value_counts()
    debug_print("\nnew_and_old_link_table_val_counts")
    debug_print(new_and_old_link_table_val_counts)

    records_to_concat = [all_initial_action_records]
    if num_remaining_actions > 0:

        #get more users
        remaining_sampled_links = new_and_old_link_table.sample(n=num_remaining_actions, weights=new_and_old_link_table["edge_weight"], replace=True)
        remaining_sampled_links=remaining_sampled_links.reset_index(drop=True)
        debug_print("\nremaining_sampled_links")
        debug_print(remaining_sampled_links)

        #remaining_sampled_links = assign_parents_to_new_user_placeholders_v2_debug_print(remaining_sampled_links, cur_new_users_only_weight_table, DEBUG_PRINT)

        try:
            remaining_sampled_links = assign_parents_to_new_user_placeholders_v2_debug_print(remaining_sampled_links, cur_new_users_only_weight_table, DEBUG_PRINT)
        except ValueError:
            remaining_sampled_links = assign_parents_to_new_user_placeholders_v2_debug_print(remaining_sampled_links, cur_all_users_weight_table, DEBUG_PRINT)

        debug_print("\nremaining_sampled_links with new users fixed")
        debug_print(remaining_sampled_links)

        records_to_concat.append(remaining_sampled_links)

    #     #complete records
    #     final_link_records = pd.concat([remaining_sampled_links, all_initial_action_records ])
    # else:
    final_link_records = pd.concat(records_to_concat)


    # final_link_records["edge_weight"] = final_link_records.groupby(["nodeUserID", "parentUserID"])["nodeUserID"].transform("count")
    # final_link_records = final_link_records.sort_values("edge_weight", ascending=False)
    final_link_records=final_link_records[["nodeUserID", "parentUserID", "informationID", "edge_weight"]]
    final_link_records=final_link_records.reset_index(drop=True)
    debug_print("\nfinal_link_records")
    debug_print(final_link_records)


    final_parent_pred_val_counts = final_link_records["parentUserID"].value_counts()
    debug_print("\nfinal_parent_pred_val_counts")
    debug_print(final_parent_pred_val_counts)

    total_actions_gotten = final_link_records.shape[0]

    debug_print(total_actions_gotten)
    debug_print(num_actions)
    if total_actions_gotten != num_actions:
        debug_print("\nError! total_actions_gotten != num_actions")
        sys.exit(0)

    final_link_records["timestep"] = timestep
    # infoID_to_df_list_dict[infoID].append(final_link_records)

    debug_print("\nfinal_link_records")
    debug_print(final_link_records)

    print("Done simulating infoID %d of %d: %s"%((infoID_idx+1), num_infoIDs, infoID))

    return (final_link_records,infoID)

def get_user_sim_pred_records_for_cur_infoID_BACKUP( infoID_materials_tuple,NUM_LINK_SAMPLE_ATTEMPTS,ADD_OLD_USERS_TO_NEW_USERS_IF_TOO_MANY, timestep, DEBUG_PRINT):

    all_pred_count_dfs_for_cur_infoID_for_verif = []
    all_timestep_dfs_for_cur_infoID = []

    #get infoID materials
    infoID ,hyp_infoID, cur_pred_df, cur_all_users_weight_table, cur_new_users_only_weight_table, cur_all_users_edge_weight_dist_table,cur_new_user_edge_weight_dist_table,num_infoIDs,infoID_idx  = infoID_materials_tuple

    print("Simulating infoID %d of %d: %s"%((infoID_idx+1), num_infoIDs, infoID))

    #get numbers
    num_new_users = int(cur_pred_df["num_new_users"].iloc[0])
    num_old_users = int(cur_pred_df["num_old_users"].iloc[0])
    num_actions = int(cur_pred_df["num_actions"].iloc[0])

    def debug_print(print_val="\n"):
        if DEBUG_PRINT == True:
            print(str(print_val))




    #============= get old users first -> check if there's a conflict =============

    num_old_users_in_weight_df = cur_all_users_weight_table["nodeUserID"].nunique()
    old_user_diff = num_old_users_in_weight_df - num_old_users
    debug_print("\nNum old users in weight df: %d; num predicted: %d; Difference: %d"%(num_old_users_in_weight_df, num_old_users, old_user_diff))


    if old_user_diff >= 0:
        #get OLD users
        sampled_old_user_records = cur_all_users_weight_table.sample(n=num_old_users ,weights=cur_all_users_weight_table["nodeUserID_overall_action_weight"], replace=False)
    else:
        #get what you can
        amount_of_old_users_possible_to_sample = num_old_users_in_weight_df
        debug_print("\nWe can only get %d old users"%amount_of_old_users_possible_to_sample)
        sampled_old_user_records= cur_all_users_weight_table.sample(n=amount_of_old_users_possible_to_sample ,weights=cur_all_users_weight_table["nodeUserID_overall_action_weight"], replace=False)

        remaining_old_users = num_old_users - amount_of_old_users_possible_to_sample

        initial_cur_pred_df = cur_pred_df.copy()
        debug_print("\nThere are %d remaining_old_users that need to be sampled"%remaining_old_users )
        cur_pred_df["num_old_users"] = amount_of_old_users_possible_to_sample
        num_old_users = amount_of_old_users_possible_to_sample

        if ADD_OLD_USERS_TO_NEW_USERS_IF_TOO_MANY == True:
            debug_print("\nADD_OLD_USERS_TO_NEW_USERS_IF_TOO_MANY==True, so we have to increase # of new users")
            num_new_users = num_new_users + remaining_old_users

            cur_pred_df["num_new_users"] = num_new_users

        debug_print("\nInitial cur pred df")
        debug_print(initial_cur_pred_df)

        debug_print("\nModified cur pred df")
        debug_print(cur_pred_df)

    debug_print("\nsampled_old_user_records")
    sampled_old_user_records = sampled_old_user_records.reset_index(drop=True)
    # sampled_old_user_records = sampled_old_user_records.rename(columns={"latest_parent":"parentUserID"})
    # sampled_old_user_records = sampled_old_user_records.reset_index(drop=True)
    debug_print(sampled_old_user_records)

    # num_old_users = sampled_old_user_records.shape[0]

    # sys.exit(0)

    #=================== this is for faster runtime ===================
    debug_print("\ncur_all_users_edge_weight_dist_table")
    debug_print(cur_all_users_edge_weight_dist_table)

    # cur_all_users_edge_weight_dist_table["edge_weight"] = cur_all_users_edge_weight_dist_table["edge_weight"]*WEIGHT_MULT_FACTOR
    # debug_print("\ncur_all_users_edge_weight_dist_table with mult factor")
    # debug_print(cur_all_users_edge_weight_dist_table)
    initial_old_links = get_initial_links_v2_debug_print(num_old_users, NUM_LINK_SAMPLE_ATTEMPTS, sampled_old_user_records,cur_all_users_edge_weight_dist_table,DEBUG_PRINT)
    # sys.exit(0)


    #========================== get new users ==========================
    #get new users
    try:
        sampled_new_user_records = cur_new_users_only_weight_table.sample(n=num_new_users ,weights=cur_new_users_only_weight_table["nodeUserID_overall_action_weight"], replace=False)
    except ValueError:
        try:
            sampled_new_user_records = cur_new_users_only_weight_table.sample(n=num_new_users ,weights=cur_new_users_only_weight_table["nodeUserID_overall_action_weight"], replace=True)
        except ValueError:
            try:
                print("\nThere were not enough new users from prev timestep. Sampling from old user table without replacement.")
                sampled_new_user_records = cur_all_users_weight_table.sample(n=num_new_users ,weights=cur_all_users_weight_table["nodeUserID_overall_action_weight"], replace=False)
            except:
                print("\nThere were not enough new users from prev timestep. Sampling from old user table WITH replacement.")
                sampled_new_user_records = cur_all_users_weight_table.sample(n=num_new_users ,weights=cur_all_users_weight_table["nodeUserID_overall_action_weight"], replace=True)




    debug_print("\nSampled new users")
    sampled_new_user_records = sampled_new_user_records.reset_index(drop=True)
    debug_print(sampled_new_user_records)

    # cur_new_user_edge_weight_dist_table = infoID_to_new_user_edge_weight_dist_table_dict[infoID]

    debug_print("\nThese records say whose attributes the new users will copy.")
    # sampled_new_user_records["original_nodeUserID"] = sampled_new_user_records["nodeUserID"].copy()
    sampled_new_user_records["new_nodeUserID"] = ["synthetic_user_%d"%(i+1) for i in range(num_new_users)]
    debug_print(sampled_new_user_records)

    debug_print("\nNow we must get a custom distribution table for the new users we have just created")
    temp = sampled_new_user_records[["nodeUserID", "new_nodeUserID","informationID"]].copy()
    cur_custom_new_user_edge_weight_dist_table = pd.merge(temp,cur_new_user_edge_weight_dist_table,on=["nodeUserID","informationID"], how="inner")
    debug_print("\ncur_custom_new_user_edge_weight_dist_table")
    debug_print(cur_custom_new_user_edge_weight_dist_table)
    cur_custom_new_user_edge_weight_dist_table["original_nodeUserID"]=cur_custom_new_user_edge_weight_dist_table["nodeUserID"].copy()
    cur_custom_new_user_edge_weight_dist_table["nodeUserID"] =cur_custom_new_user_edge_weight_dist_table["new_nodeUserID"]
    # cur_custom_new_user_edge_weight_dist_table= cur_custom_new_user_edge_weight_dist_table.rename(columns={"new_nodeUserID":"nodeUserID"})

    debug_print("\ncur_custom_new_user_edge_weight_dist_table")
    debug_print(cur_custom_new_user_edge_weight_dist_table)

    nodeUserIDs = cur_custom_new_user_edge_weight_dist_table["nodeUserID"]
    parentUserIDs = cur_custom_new_user_edge_weight_dist_table["parentUserID"]
    cur_custom_new_user_edge_weight_dist_table["parentUserID"] = [nodeUserID if parentUserID=="self" else parentUserID for nodeUserID,parentUserID in zip(nodeUserIDs, parentUserIDs)]

    debug_print("\ncur_custom_new_user_edge_weight_dist_table with fixed self loops")
    debug_print(cur_custom_new_user_edge_weight_dist_table)

    sampled_new_user_records["original_nodeUserID"]=sampled_new_user_records["nodeUserID"].copy()
    sampled_new_user_records["nodeUserID"] =sampled_new_user_records["new_nodeUserID"]
    debug_print(sampled_new_user_records)

    parent_val_counts = cur_custom_new_user_edge_weight_dist_table["parentUserID"].value_counts()
    debug_print("\nparent_val_counts")
    debug_print(parent_val_counts)

    orig_user_val_counts = cur_custom_new_user_edge_weight_dist_table["original_nodeUserID"].value_counts()
    debug_print("\norig_user_val_counts")
    debug_print(orig_user_val_counts)



    #=================== add a checkpoint here ===============
    verify_pred_counts_so_far_v2_debug_print(sampled_new_user_records,sampled_old_user_records,num_old_users, num_new_users, DEBUG_PRINT)

    #get initial new links
    debug_print("\nGetting initial new links...")
    initial_new_links = get_initial_links_v2_debug_print(num_new_users, NUM_LINK_SAMPLE_ATTEMPTS, sampled_new_user_records,cur_custom_new_user_edge_weight_dist_table,DEBUG_PRINT)
    # initial_new_links["parentUserID"]="new_user"


    initial_new_links = assign_parents_to_new_user_placeholders_v2_debug_print(initial_new_links, cur_new_users_only_weight_table, DEBUG_PRINT)

    debug_print("\ninitial_new_links")
    debug_print(initial_new_links)

    # sys.exit(0)

    #=================== combine old and new users ===============
    verify_pred_counts_so_far_v2_debug_print(initial_new_links,initial_old_links,num_old_users, num_new_users, DEBUG_PRINT)

    all_initial_action_records = pd.concat([initial_old_links, initial_new_links]).reset_index(drop=True)
    all_initial_action_records=all_initial_action_records[["nodeUserID", "parentUserID", "informationID", "edge_weight"]]
    parent_val_counts = all_initial_action_records["parentUserID"].value_counts()
    debug_print("\nparent_val_counts")
    debug_print(parent_val_counts)


    debug_print("\nnum_actions: %d"%num_actions)
    num_all_initial_action_records = all_initial_action_records.shape[0]
    debug_print("num_all_initial_action_records: %d"%num_all_initial_action_records)

    num_remaining_actions = num_actions - num_all_initial_action_records
    debug_print("num_remaining_actions: %d"%num_remaining_actions)

    #get likely links
    #first make a combined edge weight table
    new_and_old_link_table = pd.concat([cur_custom_new_user_edge_weight_dist_table ,cur_all_users_edge_weight_dist_table]).reset_index(drop=True)
    debug_print("\nnew_and_old_link_table")
    debug_print(new_and_old_link_table)

    #lets make sure it only has the users we want
    users_so_far = all_initial_action_records["nodeUserID"].unique()
    new_and_old_link_table = new_and_old_link_table[new_and_old_link_table["nodeUserID"].isin(users_so_far)].reset_index(drop=True)
    debug_print("\nnew_and_old_link_table")
    debug_print(new_and_old_link_table)

    new_and_old_link_table_val_counts = new_and_old_link_table["parentUserID"].value_counts()
    debug_print("\nnew_and_old_link_table_val_counts")
    debug_print(new_and_old_link_table_val_counts)

    #get more users
    remaining_sampled_links = new_and_old_link_table.sample(n=num_remaining_actions, weights=new_and_old_link_table["edge_weight"], replace=True)
    remaining_sampled_links=remaining_sampled_links.reset_index(drop=True)
    debug_print("\nremaining_sampled_links")
    debug_print(remaining_sampled_links)

    remaining_sampled_links = assign_parents_to_new_user_placeholders_v2_debug_print(remaining_sampled_links, cur_new_users_only_weight_table, DEBUG_PRINT)

    debug_print("\nremaining_sampled_links with new users fixed")
    debug_print(remaining_sampled_links)

    #complete records
    final_link_records = pd.concat([remaining_sampled_links, all_initial_action_records ])
    # final_link_records["edge_weight"] = final_link_records.groupby(["nodeUserID", "parentUserID"])["nodeUserID"].transform("count")
    # final_link_records = final_link_records.sort_values("edge_weight", ascending=False)
    final_link_records=final_link_records[["nodeUserID", "parentUserID", "informationID", "edge_weight"]]
    final_link_records=final_link_records.reset_index(drop=True)
    debug_print("\nfinal_link_records")
    debug_print(final_link_records)


    final_parent_pred_val_counts = final_link_records["parentUserID"].value_counts()
    debug_print("\nfinal_parent_pred_val_counts")
    debug_print(final_parent_pred_val_counts)

    total_actions_gotten = final_link_records.shape[0]

    debug_print(total_actions_gotten)
    debug_print(num_actions)
    if total_actions_gotten != num_actions:
        debug_print("\nError! total_actions_gotten != num_actions")
        sys.exit(0)

    final_link_records["timestep"] = timestep
    # infoID_to_df_list_dict[infoID].append(final_link_records)

    debug_print("\nfinal_link_records")
    debug_print(final_link_records)

    print("Done simulating infoID %d of %d: %s"%((infoID_idx+1), num_infoIDs, infoID))

    return final_link_records

def get_initial_links_v2_debug_print(total_num_users, NUM_SAMPLE_ATTEMPTS, sampled_user_records,cur_all_users_edge_weight_dist_table, DEBUG_PRINT):

    def debug_print(print_val="\n"):
        if DEBUG_PRINT == True:
            print(str(print_val))

    #get initial old user links
    debug_print("\nGetting initial user links...Number total users to start: %d"%total_num_users)
    all_sampled_users_set = set(sampled_user_records["nodeUserID"])
    temp_all_users_edge_weight_dist_table = cur_all_users_edge_weight_dist_table[cur_all_users_edge_weight_dist_table["nodeUserID"].isin(all_sampled_users_set)].reset_index(drop=True)
    debug_print("\ncur_all_users_edge_weight_dist_table")
    debug_print(cur_all_users_edge_weight_dist_table)
    debug_print("\ntemp_all_users_edge_weight_dist_table")
    debug_print(temp_all_users_edge_weight_dist_table)
    all_initial_link_records = []

    if total_num_users == 0:
        cols = list(temp_all_users_edge_weight_dist_table)
        initial_links = pd.DataFrame()
        for col in cols:
            initial_links[col] = []
        debug_print("\ninitial_links empty df")
        print(initial_links)
        return initial_links


    num_remaining_users = total_num_users
    retrieved_user_set_so_far = set()
    #attempt
    # for ATTEMPT in range(NUM_SAMPLE_ATTEMPTS):
    while (True):
        debug_print("\ntemp table with mult factor")
        debug_print(temp_all_users_edge_weight_dist_table["edge_weight"])

        #==================== sample some users ============


        initial_links = temp_all_users_edge_weight_dist_table.sample(n=num_remaining_users, weights=temp_all_users_edge_weight_dist_table["edge_weight"], replace=False)

        initial_links = initial_links.reset_index(drop=True)
        initial_links = initial_links.drop_duplicates("nodeUserID").reset_index(drop=True)
        debug_print("\ninitial_links")
        debug_print(initial_links)

        #=============== add users to set ===============
        cur_user_set = set(initial_links["nodeUserID"])
        retrieved_user_set_so_far = retrieved_user_set_so_far.union(cur_user_set)
        num_retrieved_users_so_far = len(retrieved_user_set_so_far)
        # debug_print("\nnum_retrieved_users_so_far: %d"%num_retrieved_users_so_far)

        users_yet_to_be_retrieved_set =all_sampled_users_set -  retrieved_user_set_so_far
        num_remaining_users = len(users_yet_to_be_retrieved_set)
        # debug_print("\nnum_remaining_users: %d"%num_remaining_users)
        debug_print("\nThere are %d total users. We have %d so far. We need %d more"%(total_num_users,num_retrieved_users_so_far, num_remaining_users))

        #update table
        temp_all_users_edge_weight_dist_table =temp_all_users_edge_weight_dist_table[temp_all_users_edge_weight_dist_table["nodeUserID"].isin(users_yet_to_be_retrieved_set)].reset_index(drop=True)
        all_initial_link_records.append(initial_links)

        if num_remaining_users == 0:
            break


    initial_links = pd.concat(all_initial_link_records).reset_index(drop=True)
    debug_print(initial_links.shape[0])
    debug_print(total_num_users)
    if initial_links.shape[0] > total_num_users:
        print(initial_links.shape[0])
        print(total_num_users)
        print("\nError! initial_links.shape[0] > total_num_users")
        sys.exit(0)

    debug_print("\nThere are %d total users. We have %d so far. We need %d more"%(total_num_users,num_retrieved_users_so_far, num_remaining_users))

    if num_remaining_users != 0:
        remaining_initial_link_records = []
        users_yet_to_be_retrieved_set = list(users_yet_to_be_retrieved_set)
        debug_print("\nGetting remaining old users...")
        for user_idx,user in enumerate(users_yet_to_be_retrieved_set):
            debug_print("\nGetting user %d of %d"%(user_idx, num_remaining_users))

            temp2 = cur_all_users_edge_weight_dist_table[cur_all_users_edge_weight_dist_table["nodeUserID"].isin([user])].reset_index(drop=True)
            debug_print(temp2)
            link_record = temp2.sample(n=1, weights=temp2["edge_weight"])
            remaining_initial_link_records.append(link_record)
            debug_print(link_record)

            # if len(remaining_initial_link_records) == user_idx+1:
            #     break

        remaining_initial_link_records = pd.concat(remaining_initial_link_records).reset_index(drop=True)
        initial_links = pd.concat([initial_links, remaining_initial_link_records]).reset_index(drop=True)
    # sys.exit(0)



    debug_print("\ninitial_links")
    debug_print(initial_links)
    debug_print(initial_links.shape[0])
    debug_print(total_num_users)
    if initial_links.shape[0] != total_num_users:
        print(initial_links)
        print(initial_links.shape[0])
        print(total_num_users)
        print("\nError! initial_links.shape[0] != total_num_users")
        sys.exit(0)

    debug_print("Counts are ok!")


    return initial_links




def assign_parents_to_new_user_placeholders_v2_debug_print(initial_new_links, cur_new_users_only_weight_table, DEBUG_PRINT):

    def debug_print(print_val="\n"):
        if DEBUG_PRINT == True:
            print(str(print_val))

    print("\ninitial_new_links cols: %s"%str(list(initial_new_links)))

    initial_new_links["parentUserID_is_new"] = [1 if "new_user" in parentUserID else 0 for parentUserID in list(initial_new_links["parentUserID"])]
    debug_print("\ninitial_new_links")
    debug_print(initial_new_links)


    new_new_links = initial_new_links[(initial_new_links["parentUserID_is_new"]==1)].reset_index(drop=True)
    debug_print("\nnew_new_links")
    debug_print(new_new_links)

    if new_new_links.shape[0] > 0:

        total_initial_links = initial_new_links.shape[0]

        new_old_links = initial_new_links[(initial_new_links["parentUserID_is_new"]==0)].reset_index(drop=True)
        debug_print("\nnew_old_links")
        debug_print(new_old_links)
        # sys.exit(0)

        #get likely new users
        num_initial_new_user_actions = new_new_links.shape[0]
        debug_print("\nnum_initial_new_user_actions: %d"%num_initial_new_user_actions)

        new_user_parents = cur_new_users_only_weight_table.sample(n=num_initial_new_user_actions, weights=cur_new_users_only_weight_table["nodeUserID_overall_influence_weight"], replace=True)
        new_user_parents = new_user_parents.reset_index(drop=True)
        debug_print("\nnew_user_parents")
        debug_print(new_user_parents)

        new_new_links["parentUserID"] = new_user_parents["nodeUserID"].copy()
        debug_print("\nnew_new_links")
        debug_print(new_new_links)


        initial_new_links=pd.concat([new_new_links,new_old_links]).reset_index(drop=True)

        num_final_links = initial_new_links.shape[0]

        if total_initial_links != num_final_links:
            print(total_initial_links)
            print(num_final_links)
            print("\nError! total_initial_links != num_final_links. Exiting!")
            sys.exit(0)

    return initial_new_links
















