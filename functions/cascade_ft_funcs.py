import pandas as pd
import os,sys
import numpy as np
from sklearn.metrics import mean_squared_error
from nn_funcs_7_27 import *
import keras.regularizers as Reg
from scipy import stats
from sklearn.metrics import f1_score
from time import time
import random
import os
import shutil
import stat
import matplotlib.dates as mdates
from random import seed
from random import random
from random import randrange
from keras.utils.vis_utils import plot_model
from ft_categories import *
from joblib import Parallel, delayed
from textwrap  import wrap



# print("\nSaving input fts...")
# with open(output_dir + "input_fts.txt", "w") as f:
#     for line in input_fts:
#         f.write(line + "\n")
#         print(line)

def save_ft_list_as_text_file(output_dir, tag,ft_list):
    output_fp = output_dir + tag + ".txt"
    with open(output_fp, "w") as f:
        print("\n%s"%tag)
        for line in ft_list:
            f.write(line + "\n")
            print(line)


def remove_static_1hot_fts(x_array, dynamic_fts,static_fts,dynamic_and_static_fts,INPUT_TIMESTEPS ,RESHAPE=True):

    print("\nx_array shape: %s"%str(x_array.shape))

    num_static_fts = len(static_fts)
    print("num_static_fts: %d"%num_static_fts)

    num_dynamic_fts = len(dynamic_fts)
    print("num_dynamic_fts: %d"%num_dynamic_fts)


    num_dynamic_and_static_fts = len(dynamic_and_static_fts)
    print("num_dynamic_and_static_fts: %d"%num_dynamic_and_static_fts)

    x_array = x_array[:, :-num_static_fts]
    if RESHAPE == True:
        x_array = x_array.reshape((x_array.shape[0], INPUT_TIMESTEPS ,int(x_array.shape[1]/INPUT_TIMESTEPS)))
    print("\nx_array after removing static fts")
    print(x_array.shape)
    return x_array







def insert_static_1hot_fts_for_platform_samples(infoID_of_interest,x_array, dynamic_fts, static_fts,ft_to_idx_dict ,infoIDs,GET_1HOT_INFO_ID_FTS):

    #put 1 hot features now
    num_1hot_vecs_to_get = x_array.shape[0]
    x_array = x_array.reshape((x_array.shape[0], x_array.shape[1]*x_array.shape[2]))
    print("\nx_array: %s"%str(x_array.shape))

    num_dynamic_fts = len(dynamic_fts)

    new_ft_to_idx_dict = {}
    new_idx_to_ft_dict = {}
    flattened_fts = []
    print("\nflattened_fts")
    for idx in range(x_array.shape[1]):
        cur_dyn_idx = idx%num_dynamic_fts
        # print(cur_dyn_idx)
        cur_dyn_ft = dynamic_fts[cur_dyn_idx]
        new_idx_to_ft_dict[idx]= cur_dyn_ft + "_%d"%idx
        new_ft_to_idx_dict[cur_dyn_ft + "_%d"%idx] = idx
        flattened_fts.append(cur_dyn_ft + "_%d"%idx)
        # print(cur_dyn_ft + "_%d"%idx)
    # sys.exit(0)

    ft_to_idx_dict = dict(new_ft_to_idx_dict)
    idx_to_ft_dict = dict(new_idx_to_ft_dict)


    all_1hot_vec_fts = []
    infoID_1hot_fts = [0 for idx in range(len(infoIDs))]

    max_ft_dict_idx = len(ft_to_idx_dict) - 1
    print("\nmax_ft_dict_idx: %d"%max_ft_dict_idx)

    # all_1hot_action_fts = []
    all_1hot_infoID_fts = []
    if GET_1HOT_INFO_ID_FTS == True:
        for idx,cur_infoID in enumerate(infoIDs):
            flattened_fts.append(cur_infoID)
            max_ft_dict_idx+=1
            if cur_infoID == infoID_of_interest:
                infoID_1hot_fts[idx] = 1
            ft_to_idx_dict[cur_infoID] = max_ft_dict_idx
            idx_to_ft_dict[max_ft_dict_idx]=cur_infoID

        for idx in range(num_1hot_vecs_to_get):
            all_1hot_infoID_fts.append(np.asarray(infoID_1hot_fts))

        all_1hot_infoID_fts = np.asarray(all_1hot_infoID_fts)
        x_array = np.concatenate([x_array, all_1hot_infoID_fts], axis=1)
        print("\nx_array shape after all_1hot_infoID_fts concat: %s"%str(x_array.shape))



    num_flattened_fts = len(flattened_fts)
    max_idx = num_flattened_fts - 1
    max_idx_ft = idx_to_ft_dict[max_idx]
    # print()
    # print(max_idx)
    # print(max_idx_ft)

    return x_array,ft_to_idx_dict,idx_to_ft_dict,flattened_fts

def get_combo_to_df_dict_with_complement_features(main_input_dir ,hyp_dict,overall_dates, global_complement_cols_to_get,infoIDs,platforms,user_types,user_statuses, GET_COMPLEMENT_ACTIVITY_NARR_PAIRS_OF_INTEREST,GET_COMPLEMENT_USER_AGE_NARR_PAIRS_OF_INTEREST):

    NUM_DATES = len(overall_dates)

    #make blank df
    global_count_df = pd.DataFrame(data={"nodeTime":overall_dates})
    global_count_df["nunique_users"] = [0 for i in range(NUM_DATES)]
    global_count_df["num_actions"] = [0 for i in range(NUM_DATES)]

    #first make 1 big df
    print("\nGetting original dfs and making global df...")

    num_combos = len(infoIDs) * len(platforms) * len(user_statuses) * len(user_types)
    print("\nnum_combos: %d" %num_combos)

    combo_to_df_dict = {}

    i=1
    for infoID in infoIDs:
        combo_to_df_dict[infoID] = {}
        hyp_infoID = hyp_dict[infoID]
        for platform in platforms:
            combo_to_df_dict[infoID][platform] = {}
            for user_status in user_statuses:
                combo_to_df_dict[infoID][platform][user_status] = {}
                for user_type in user_types:

                    input_fp = main_input_dir + platform + "/" + user_status + "/" + hyp_infoID + "/%s-user-data.csv"%user_type
                    df = pd.read_csv(input_fp)

                    if df.shape[0] != global_count_df.shape[0]:
                        print("\nError! df.shape[0] != global_count_df.shape[0]")
                        print(df.shape[0])
                        print(global_count_df.shape[0])
                        sys.exit(0)

                    for global_complement_col in global_complement_cols_to_get:
                        global_count_df[global_complement_col] = global_count_df[global_complement_col] + df[global_complement_col]
                        # global_count_df["num_actions"] = global_count_df["num_actions"] + df["num_actions"]
                        # print("\noriginal df")
                        # print(df)
                        # print("\nCurrent global_count df")
                        # print(global_count_df)



                    combo_to_df_dict[infoID][platform][user_status][user_type] = df

                    # print(df)
                    print("Got original combo df %d of %d"%(i, num_combos))
                    i+=1

    #now get complement ts counts
    i=1
    print("\nGetting complement count features...")
    # global_complement_cols_to_get = ["nunique_users", "num_actions"]
    global_complement_fts = set()
    for infoID in infoIDs:
        hyp_infoID = hyp_dict[infoID]
        for platform in platforms:
            for user_status in user_statuses:
                for user_type in user_types:

                    input_fp = main_input_dir + platform + "/" + user_status + "/" + hyp_infoID + "/%s-user-data.csv"%user_type
                    df = combo_to_df_dict[infoID][platform][user_status][user_type]

                    for global_complement_col in global_complement_cols_to_get:

                        #get complement
                        global_complement_fts.add("global_complement_%s"%global_complement_col)
                        df["global_complement_%s"%global_complement_col] = global_count_df["%s"%global_complement_col] - df["%s"%global_complement_col]
                        # print(df)

                        if df["global_complement_%s"%global_complement_col].sum() + df["%s"%global_complement_col].sum() != global_count_df["%s"%global_complement_col].sum():
                            print('Error! df["global_complement_%s"].sum() + df["%s"].sum() != global_count_df["%s"].sum()'%(global_complement_col , global_complement_col , global_complement_col))
                            print(df["global_complement_%s"%global_complement_col].sum())
                            print(df["%s"%global_complement_col].sum())
                            print( global_count_df["%s"%global_complement_col].sum())

                    #update
                    combo_to_df_dict[infoID][platform][user_status][user_type] = df
                    print("Got complement feature set %d of %d"%(i, num_combos))
                    i+=1

                    # print(df)

    global_complement_fts = list(global_complement_fts)
    print("\nglobal_complement_fts")
    print(global_complement_fts)
    # sys.exit(0)
    return combo_to_df_dict,global_complement_fts

def convert_pred_dict_to_a_df_v2_cascade_func_version(pair_pred_dict,infoIDs,actions,test_start,test_end,GRAN):

    print("\nGetting pred and test df...")

    #first get array length
    for infoID in infoIDs:
        for action in actions:
            num_timesteps = pair_pred_dict[infoID][action].shape[0]
            print("\nnum_timesteps: %d"%num_timesteps)
            break
        break

    #get timesteps
    y_pred = np.asarray([0 for i in range(num_timesteps)])


    #add up time series
    for infoID in infoIDs:
        for action in actions:
            print(infoID)
            print(action)

            #========================== y pred stuff ==========================
            #get y pred
            cur_y_pred_to_add = pair_pred_dict[infoID][action].flatten()
            print("\ncur_y_pred_to_add: %s"%str(cur_y_pred_to_add))

            #add to y pred
            print("\nCur y_pred before adding: %s"%str(y_pred))
            y_pred = np.sum([y_pred,cur_y_pred_to_add],axis=0)
            print("\nCur y_pred after adding: %s"%str(y_pred))

    #round y_pred
    y_pred = np.round(y_pred, 0)

    dates = pd.date_range(test_start,test_end,freq=GRAN)


    #make df
    df = pd.DataFrame(data={"y_pred":y_pred,"nodeTime":dates})
    df["nodeTime"] = pd.to_datetime(df["nodeTime"], utc=True)
    df = df[["nodeTime", "y_pred"]]
    print("\n\ndf")
    print(df)


    return df

def get_infoID_quartet_train_and_test_subdict_v2_bigger_module1( infoID, infoID_idx,infoIDs,hyp_dict, platforms, user_statuses, user_types, basic_quartet_data_dict, aux_df, GRAN,sum_fts,
    avg_fts, test_start, test_end, val_start, val_end,IO_TUPLE,input_fts,target_fts):
    # num_quartets = len(infoIDs) * len(platforms) * len(user_statuses) * len(user_types)
    # print("\nnum_quartets: %d" %num_quartets)

    infoID_quartet_train_and_test_subdict = {}
    hyp_infoID = hyp_dict[infoID]
    # i = 0
    for platform in platforms:
        infoID_quartet_train_and_test_subdict[platform] = {}
        for user_status in user_statuses:
            infoID_quartet_train_and_test_subdict[platform][user_status] = {}
            # for infoID in infoIDs:
            #
            #   infoID_quartet_train_and_test_subdict[platform][user_status][infoID] = {}
            for user_type in user_types:
                # fp = input_dir + "%s/%s/%s/%s-user-data.csv"%(platform, user_status, hyp_infoID, user_type)
                infoID_quartet_train_and_test_subdict[platform][user_status][user_type] = {}

                df = basic_quartet_data_dict[infoID][platform][user_status][user_type]
                # df = pd.read_csv(fp)

                #merge with aux dfs
                # df = merge_mult_dfs([df] + aux_dfs, on=["nodeTime"], how="inner")

                df["nodeTime"] = pd.to_datetime(df["nodeTime"], utc=True)
                df = pd.merge(df, aux_df, on="nodeTime", how="inner")

                df = alter_df_GRAN_v3_fix_duplicate_issue(df, GRAN, input_fts, target_fts,sum_fts, avg_fts)

                #get dfs
                train_and_val_df,test_df = split_dfs_into_train_and_test_with_nodeID_hack(df, test_start, test_end,IO_TUPLE,DEBUG_PRINT=False)
                train_df,val_df = split_dfs_into_train_and_test_with_nodeID_hack(df, val_start, val_end,IO_TUPLE,DEBUG_PRINT=False)

                print("\ntrain_df")
                print(train_df)

                print("\nval_df")
                print(val_df)

                print("\ntest_df")
                print(test_df)

                infoID_quartet_train_and_test_subdict[platform][user_status][user_type]["train_df"] = train_df.copy()
                infoID_quartet_train_and_test_subdict[platform][user_status][user_type]["val_df"] = val_df.copy()
                infoID_quartet_train_and_test_subdict[platform][user_status][user_type]["test_df"] = test_df.copy()

                # i+=1
                # print("Got quartet df %d of %d" %(i, num_quartets))

    print("Got quartet subdict %d of %d" %(infoID_idx, len(infoIDs)))
    return infoID_quartet_train_and_test_subdict

def get_infoID_quartet_train_and_test_subdict( infoID, infoID_idx,infoIDs,hyp_dict, platforms, user_statuses, user_types, input_dir, aux_df, GRAN,sum_fts,
    avg_fts, test_start, test_end, val_start, val_end,IO_TUPLE,USE_PLATFORM_HISTORY,input_fts,target_fts):
    # num_quartets = len(infoIDs) * len(platforms) * len(user_statuses) * len(user_types)
    # print("\nnum_quartets: %d" %num_quartets)

    infoID_quartet_train_and_test_subdict = {}
    hyp_infoID = hyp_dict[infoID]
    # i = 0
    for platform in platforms:
        infoID_quartet_train_and_test_subdict[platform] = {}
        for user_status in user_statuses:
            infoID_quartet_train_and_test_subdict[platform][user_status] = {}
            # for infoID in infoIDs:
            #
            #   infoID_quartet_train_and_test_subdict[platform][user_status][infoID] = {}
            for user_type in user_types:
                fp = input_dir + "%s/%s/%s/%s-user-data.csv"%(platform, user_status, hyp_infoID, user_type)
                infoID_quartet_train_and_test_subdict[platform][user_status][user_type] = {}
                df = pd.read_csv(fp)

                #merge with aux dfs
                # df = merge_mult_dfs([df] + aux_dfs, on=["nodeTime"], how="inner")

                df["nodeTime"] = pd.to_datetime(df["nodeTime"], utc=True)
                df = pd.merge(df, aux_df, on="nodeTime", how="inner")

                # print(df)

                if USE_PLATFORM_HISTORY == True:
                    df = alter_df_GRAN_v2(df, GRAN, input_fts, [],sum_fts, avg_fts)
                else:
                    df = alter_df_GRAN_v2(df, GRAN, input_fts, target_fts,sum_fts, avg_fts)

                #get dfs
                train_and_val_df,test_df = split_dfs_into_train_and_test_with_nodeID_hack(df, test_start, test_end,IO_TUPLE,DEBUG_PRINT=False)
                train_df,val_df = split_dfs_into_train_and_test_with_nodeID_hack(df, val_start, val_end,IO_TUPLE,DEBUG_PRINT=False)

                # print("\ntrain_df")
                # print(train_df)

                # print("\nval_df")
                # print(val_df)

                # print("\ntest_df")
                # print(test_df)

                infoID_quartet_train_and_test_subdict[platform][user_status][user_type]["train_df"] = train_df.copy()
                infoID_quartet_train_and_test_subdict[platform][user_status][user_type]["val_df"] = val_df.copy()
                infoID_quartet_train_and_test_subdict[platform][user_status][user_type]["test_df"] = test_df.copy()

                # i+=1
                # print("Got quartet df %d of %d" %(i, num_quartets))

    print("Got quartet subdict %d of %d" %(infoID_idx, len(infoIDs)))
    return infoID_quartet_train_and_test_subdict

def check_datetime_of_record_from_df(nodeTime_tuple, cur_idx, num_records,MOD_NUM=100000):
    if cur_idx%MOD_NUM == 0:
        print("Processing record %d of %d"%(cur_idx, num_records))


    #try to fix date
    nodeID,nodeTime =nodeTime_tuple
    try:
        nodeTime = pd.to_datetime(nodeTime, utc=True)
        # print(df[["nodeTime", "nodeID"]])
        return None
    except:
        print("Bad nodeTime: %s %s"%(nodeID,nodeTime))

    print("Done processing nodeTime %d of %d"%(cur_idx, num_records))

    return nodeTime_tuple

def autocorr(x, t=1):
    return np.corrcoef(np.array([x[:-t], x[t:]]))

def make_platform_time_series(df, platform, start, end,GRAN,MAKE_SPECIFIC_PLATFORM_COL_NAME=False):

    print("\nMaking platform time series for %s..."%platform)
    temp = df[df["platform"]==platform].reset_index(drop=True)
    temp = config_df_by_dates(temp, start, end, "nodeTime")

    #set gran
    temp["nodeTime"] = temp["nodeTime"].dt.floor(GRAN)

    #date df
    dates = pd.date_range(start, end,freq=GRAN)
    blank_date_df = pd.DataFrame(data={"nodeTime":dates})
    blank_date_df["nodeTime"] = pd.to_datetime(blank_date_df["nodeTime"], utc=True)

    #mark platforms
    # temp["num_actions"] = temp.groupby(["nodeTime"])["platform"].transform("count")
    temp["num_actions"] = 1
    temp = temp[["nodeTime", "num_actions"]].reset_index(drop=True)
    temp = pd.merge(temp, blank_date_df, on="nodeTime", how="outer").reset_index(drop=True)
    temp = temp.sort_values("nodeTime").reset_index(drop=True)
    temp = temp.fillna(0)
    temp["num_actions"] =temp.groupby(["nodeTime"])["num_actions"].transform("sum")

    if MAKE_SPECIFIC_PLATFORM_COL_NAME==True:
        temp = temp.rename(columns={"num_actions":"num_%s_actions"%platform})
    temp = temp.drop_duplicates().reset_index(drop=True)

    return temp

def plot_platform_to_time_series_dict(xlabel,ylabel,title, output_dir,platform_to_time_series_dict,output_fp,dates,normalize=True):

    fig, ax = plt.subplots()
    for platform, temp_df in platform_to_time_series_dict.items():
        time_series = temp_df["num_actions"].values
        max_val = np.max(time_series)
        min_val = np.min(time_series)

        if normalize == True:
            time_series = (time_series - min_val)/(max_val - min_val)
        ax.plot(time_series, label=platform)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    leg = ax.legend()
    # xfmt = mdates.DateFormatter('%Y-%m-%d')
    # ax.xaxis.set_major_formatter(xfmt)
    # ax.set_xticks(dates)
    # plt.xticks(rotation=25)

    # ax.xaxis.set_major_locator(dates)
    # ax.xaxis.set_major_formatter(dates)


    fig.savefig(output_fp)
    print(output_fp)
    plt.close()

    return

def pair_train_and_test_df_subdict_for_infoID(infoID,infoIDs, infoID_to_action_dict,desired_actions,GRAN,input_fts,target_fts,sum_fts,avg_fts,
    test_start, test_end,IO_TUPLE, val_start, val_end, aux_df, USE_PLATFORM_HISTORY,infoID_idx):
    num_infoIDs = len(infoIDs)
    print("\nGetting infoID dfs %d of %d"%((infoID_idx+1), num_infoIDs))
    pair_train_and_test_df_subdict_for_infoID = {}
    # pair_train_and_test_df_subdict_for_infoID[infoID] = {}
    for action in desired_actions:
        # print("\nGetting train and test dfs for")
        # print(infoID)
        # print(action)

        #get fp
        fp = infoID_to_action_dict[infoID][action]
        df = pd.read_csv(fp)
        df["nodeTime"] = pd.to_datetime(df["nodeTime"], utc=True)
        df = pd.merge(df, aux_df, on="nodeTime", how="inner")
        # df = alter_df_GRAN(df, GRAN, input_fts, sum_fts, avg_fts)

        if USE_PLATFORM_HISTORY == True:
            df = alter_df_GRAN_v2(df, GRAN, input_fts, [],sum_fts, avg_fts)
        else:
            df = alter_df_GRAN_v2(df, GRAN, input_fts, target_fts,sum_fts, avg_fts)

        #get dfs
        train_and_val_df,test_df = split_dfs_into_train_and_test_with_nodeID_hack(df, test_start, test_end,IO_TUPLE,DEBUG_PRINT=False)
        train_df,val_df = split_dfs_into_train_and_test_with_nodeID_hack(df, val_start, val_end,IO_TUPLE,DEBUG_PRINT=False)

        # print("\ntrain_df")
        # print(train_df)

        # print("\nval_df")
        # print(val_df)

        # print("\ntest_df")
        # print(test_df)

        pair_train_and_test_df_subdict_for_infoID[action] = {}
        pair_train_and_test_df_subdict_for_infoID[action]["train_df"] = train_df.copy()
        pair_train_and_test_df_subdict_for_infoID[action]["val_df"] = val_df.copy()
        pair_train_and_test_df_subdict_for_infoID[action]["test_df"] = test_df.copy()

    print("Done with infoID %d of %d"%((infoID_idx+1), num_infoIDs))

    return pair_train_and_test_df_subdict_for_infoID

def make_pair_train_and_test_array_dict(pair_train_and_test_df_dict, infoID, infoIDs,desired_actions,num_pairs,infoID_idx):

    pair_train_and_test_array_dict = {}
    # i = 0
    num_infoIDs = len(infoIDs)
    pair_train_and_test_array_dict[infoID] = {}
    print("\nWorking on infoID items %d of %d"%((infoID_idx+1), num_infoIDs))
    for action in desired_actions:
        pair_train_and_test_array_dict[infoID][action] = {}

        #get dfs
        train_df = pair_train_and_test_df_dict[infoID][action]["train_df"]
        val_df = pair_train_and_test_df_dict[infoID][action]["val_df"]
        test_df = pair_train_and_test_df_dict[infoID][action]["test_df"]

        #x_train,y_train,ft_to_idx_dict =XGBOOST_convert_df_to_sliding_window_form_diff_x_and_y_fts_no_print_with_idx_dict(train_df,static_features, INPUT_SIZE, EXTRACT_OUTPUT_SIZE, input_fts, target_fts)
        #x_train,y_train,ft_to_idx_dict =convert_df_to_sliding_window_form_diff_x_and_y_fts_no_print_with_idx_dict(train_df, INPUT_SIZE, EXTRACT_OUTPUT_SIZE, input_fts, target_fts)
        x_train,y_train,ft_to_idx_dict =convert_df_to_sliding_window_form_diff_x_and_y_fts_no_print_with_idx_dict(train_df, INPUT_SIZE, EXTRACT_OUTPUT_SIZE, dynamic_fts, target_fts)
        x_train,flat_ft_to_idx_dict,flat_idx_to_ft_dict,flattened_fts = insert_static_1hot_fts( action,infoID,x_train, dynamic_fts,ft_to_idx_dict ,static_fts,infoIDs,desired_actions,GET_1HOT_INFO_ID_FTS, GET_1HOT_ACTION_FTS)
        # print("\nx and y train arrays")
        # print(x_train.shape)
        # print(y_train.shape)



        #agg y
        y_train = agg_y_array(y_train)

        #get val arrays
        x_val,y_val = convert_df_to_test_sliding_window_form_diff_x_and_y_fts_no_print(val_df, INPUT_SIZE, EXTRACT_OUTPUT_SIZE,dynamic_fts, target_fts,MOD_NUM=1000)
        x_val,_,_,_ = insert_static_1hot_fts(action,infoID,x_val, dynamic_fts,ft_to_idx_dict ,static_fts,infoIDs,desired_actions,GET_1HOT_INFO_ID_FTS, GET_1HOT_ACTION_FTS)
        y_val = agg_y_array(y_val)
        # print("\nx and y val arrays")
        # print(x_val.shape)
        # print(y_val.shape)



        #sliding window train
        x_val_sliding_window,y_val_sliding_window,ft_to_idx_dict =convert_df_to_sliding_window_form_diff_x_and_y_fts_no_print_with_idx_dict(val_df, INPUT_SIZE, EXTRACT_OUTPUT_SIZE, dynamic_fts, target_fts)
        x_val_sliding_window,_,_,_ = insert_static_1hot_fts(action,infoID,x_val_sliding_window, dynamic_fts,ft_to_idx_dict ,static_fts,infoIDs,desired_actions,GET_1HOT_INFO_ID_FTS, GET_1HOT_ACTION_FTS)
        y_val_sliding_window = agg_y_array(y_val_sliding_window)
        # print("\nx and y val sliding arrays")
        # print(x_val_sliding_window.shape)
        # print(y_val_sliding_window.shape)

        #get test arrays
        x_test,y_test = convert_df_to_test_sliding_window_form_diff_x_and_y_fts_no_print(test_df, INPUT_SIZE, EXTRACT_OUTPUT_SIZE,dynamic_fts, target_fts,MOD_NUM=1000)
        x_test,_,_,_ = insert_static_1hot_fts(action,infoID,x_test, dynamic_fts,ft_to_idx_dict ,static_fts,infoIDs,desired_actions,GET_1HOT_INFO_ID_FTS, GET_1HOT_ACTION_FTS)
        # y_test = agg_y_array(y_test)
        # print("\ny_test shape")
        # print(y_test.shape)

        #sliding window test
        x_test_sliding_window,y_test_sliding_window,ft_to_idx_dict =convert_df_to_sliding_window_form_diff_x_and_y_fts_no_print_with_idx_dict(test_df, INPUT_SIZE, EXTRACT_OUTPUT_SIZE, dynamic_fts, target_fts)
        x_test_sliding_window,_,_,_ = insert_static_1hot_fts(action,infoID,x_test_sliding_window, dynamic_fts,ft_to_idx_dict ,static_fts,infoIDs,desired_actions,GET_1HOT_INFO_ID_FTS, GET_1HOT_ACTION_FTS)
        y_test_sliding_window = agg_y_array(y_test_sliding_window)
        # print("\nx and y test sliding arrays")
        # print(x_test_sliding_window.shape)
        # print(y_test_sliding_window.shape)
        # sys.exit(0)



        #data info
        pair_train_and_test_array_dict[infoID][action]["x_train"] = x_train
        pair_train_and_test_array_dict[infoID][action]["y_train"] = y_train
        pair_train_and_test_array_dict[infoID][action]["x_val"] = x_val
        pair_train_and_test_array_dict[infoID][action]["y_val"] = y_val
        pair_train_and_test_array_dict[infoID][action]["x_test"] = x_test
        pair_train_and_test_array_dict[infoID][action]["y_test"] = y_test

        pair_train_and_test_array_dict[infoID][action]["x_val_sliding_window"] = x_val_sliding_window
        pair_train_and_test_array_dict[infoID][action]["y_val_sliding_window"] = y_val_sliding_window

        pair_train_and_test_array_dict[infoID][action]["x_test_sliding_window"] = x_test_sliding_window
        pair_train_and_test_array_dict[infoID][action]["y_test_sliding_window"] = y_test_sliding_window

        i+=1
        # print("Got pair array %d of %d" %(i, num_pairs))

    print("\nDone working on infoID items %d of %d"%((infoID_idx+1), num_infoIDs))
    return pair_train_and_test_array_dict

def make_extra_ft_order_binary_code(extra_twitter_ft_category_order_list,NEW_FTS_ONLY_ft_category_bool_int_dict):

    code_list = []
    for cat in extra_twitter_ft_category_order_list:
        code= NEW_FTS_ONLY_ft_category_bool_int_dict[cat]
        code_list.append(str(code))
    final_code = "".join(code_list)

    return final_code

def get_desired_new_twitter_fts(new_ft_df, NEW_FTS_ONLY_ft_category_bool_int_dict,ft_category_to_list_dict):
    desired_new_twitter_fts = []
    print("\nGetting desired fts...")

    for cat, flag in NEW_FTS_ONLY_ft_category_bool_int_dict.items():
        if flag==1:
            print("\n%s"%cat)
            cur_desired_ft_list = ft_category_to_list_dict[cat]
            print(cur_desired_ft_list)
            desired_new_twitter_fts+=list(cur_desired_ft_list)


    return desired_new_twitter_fts

def save_ft_info(output_dir, model,ignore_fts,flattened_fts,ft_to_ft_cat_dict):

    ft_importances = model.feature_importances_
    ft_rank_df = pd.DataFrame(data={"feature":flattened_fts, "importance":ft_importances})
    ft_rank_df  = ft_rank_df[["feature", "importance"]]
    ft_rank_df = ft_rank_df.sort_values("importance", ascending=False).reset_index(drop=True)
    print("\nft_rank_df")
    print(ft_rank_df)

    ft_dir = output_dir + "Feature-Importances/"
    create_output_dir(ft_dir)
    raw_ft_fp = ft_dir + "Raw-Feature-Ranks-with-Timestep-Info.csv"
    ft_rank_df.to_csv(raw_ft_fp, index=False)
    print(raw_ft_fp)

    #get overall ft ranks
    # temp = pd.DataFrame(ft_rank_df["feature"].str.split('',1).tolist(),
    #                              columns = ['flips','row'])

    # ignore_fts = list(twitter_1hot_fts + youtube_1hot_fts + infoID_1hot_fts)

    print("\nGet ft importances without timestep info...")
    # def create_ft_importances_without_timestep_info(ft_rank_df,ignore_fts):

    #     features = list(ft_rank_df["feature"])
    #     new_fts = []
    #     for ft in features:
    #         if ft in ignore_fts:
    #             new_fts.append(ft)
    #         else:
    #             ft_str_list = ft.split("_")
    #             ft_str_list = ft_str_list[:-1]
    #             new_ft = "_".join(ft_str_list)
    #             new_fts.append(new_ft)
    #             if new_ft == "":
    #                 print("Blank ft: %s"%ft)
    #                 # sys.exit(0)
    #     ft_rank_df["feature"] = new_fts
    #     ft_rank_df["importance"] = ft_rank_df.groupby(["feature"])["importance"].transform("sum")
    #     # ft_rank_df["importance"] = (ft_rank_df["importance"] - ft_rank_df["importance"].min())/(ft_rank_df["importance"].max() - ft_rank_df["importance"].min())
    #     # ft_rank_df["importance"] = ft_rank_df["importance"]
    #     ft_rank_df = ft_rank_df.drop_duplicates().reset_index(drop=True)
    #     ft_rank_df["importance"] =ft_rank_df["importance"]/ft_rank_df["importance"].sum()
    #     ft_rank_df = ft_rank_df.sort_values("importance", ascending=False).reset_index(drop=True)
    #     return ft_rank_df

    ft_rank_df = create_ft_importances_without_timestep_info(ft_rank_df,ignore_fts)
    print("\nft_rank_df without timestep info")
    print(ft_rank_df)

    ft_fp = ft_dir + "Feature-Ranks-Without-Timestep-Info.csv"
    ft_rank_df.to_csv(ft_fp, index=False)
    print(ft_fp)

    #get ft ranks by category
    ft_rank_df["feature_category"] = ft_rank_df["feature"].map(ft_to_ft_cat_dict)
    print("\ncategory ft_rank_df")
    print(ft_rank_df)
    ft_rank_df["importance"] = ft_rank_df.groupby("feature_category")["importance"].transform("sum")
    ft_rank_df = ft_rank_df[["feature_category", "importance"]]
    ft_rank_df = ft_rank_df.drop_duplicates().reset_index(drop=True)
    ft_rank_df["importance"] =ft_rank_df["importance"]/ft_rank_df["importance"].sum()

    ft_rank_df = ft_rank_df.sort_values("importance", ascending=False).reset_index(drop=True)
    print("\nfinal category ft_rank_df")
    print(ft_rank_df)

    ft_fp = ft_dir + "Feature-Category-Ranks.csv"
    ft_rank_df.to_csv(ft_fp, index=False)
    print(ft_fp)

def create_ft_importances_without_timestep_info(ft_rank_df,ignore_fts):

    features = list(ft_rank_df["feature"])
    new_fts = []
    for ft in features:
        if ft in ignore_fts:
            new_fts.append(ft)
        else:
            ft_str_list = ft.split("_")
            ft_str_list = ft_str_list[:-1]
            new_ft = "_".join(ft_str_list)
            new_fts.append(new_ft)
            if new_ft == "":
                print("Blank ft: %s"%ft)
                # sys.exit(0)
    ft_rank_df["feature"] = new_fts
    ft_rank_df["importance"] = ft_rank_df.groupby(["feature"])["importance"].transform("sum")
    # ft_rank_df["importance"] = (ft_rank_df["importance"] - ft_rank_df["importance"].min())/(ft_rank_df["importance"].max() - ft_rank_df["importance"].min())
    # ft_rank_df["importance"] = ft_rank_df["importance"]
    ft_rank_df = ft_rank_df.drop_duplicates().reset_index(drop=True)
    ft_rank_df["importance"] =ft_rank_df["importance"]/ft_rank_df["importance"].sum()
    ft_rank_df = ft_rank_df.sort_values("importance", ascending=False).reset_index(drop=True)
    return ft_rank_df

def save_ft_info_v2_simpler(output_dir, model,ignore_fts,flattened_fts,tag):

    ft_importances = model.feature_importances_
    num_imp = len(ft_importances)
    num_fts = len(flattened_fts)
    print("\nnum_imp: %d"%num_imp)
    print("\nnum_fts: %d"%num_fts)
    ft_rank_df = pd.DataFrame(data={"feature":flattened_fts, "importance":ft_importances})
    ft_rank_df  = ft_rank_df[["feature", "importance"]]
    ft_rank_df = ft_rank_df.sort_values("importance", ascending=False).reset_index(drop=True)
    print("\nft_rank_df")
    print(ft_rank_df)

    ft_dir = output_dir + "Feature-Importances/%s/"%tag
    create_output_dir(ft_dir)
    raw_ft_fp = ft_dir + "Raw-Feature-Ranks-with-Timestep-Info.csv"
    ft_rank_df.to_csv(raw_ft_fp, index=False)
    print(raw_ft_fp)

    print("\nGet ft importances without timestep info...")
    ft_rank_df = create_ft_importances_without_timestep_info(ft_rank_df,ignore_fts)
    print("\nft_rank_df without timestep info")
    print(ft_rank_df)

    ft_fp = ft_dir + "Feature-Ranks-Without-Timestep-Info.csv"
    ft_rank_df.to_csv(ft_fp, index=False)
    print(ft_fp)


def copy_best_models(best_summ_df,output_dir):

    output_dir = output_dir + "saved-models/"
    create_output_dir(output_dir)

    model_names = list(best_summ_df["model_name"])
    model_dirs = list(best_summ_df["model_dir"])

    for i,model_name in enumerate(model_names):

        model_dir = model_dirs[i]
        print("\nGetting %s model..."%model_name)

        #make dest
        dest = output_dir + model_name + "/"
        create_output_dir(dest)

        src = model_dir
        copytree(src, dest, symlinks = False, ignore = None)
        print("Succesfully copied to %s"%dest)

def get_combined_param_df(best_summ_df,model_dir_to_model_freq_dict):

    model_dirs = list(best_summ_df["model_dir"].unique())

    #get param fields
    first_param_fp = model_dirs[1] + "params.csv"
    dummy_param_df = pd.read_csv(first_param_fp)
    param_list = list(dummy_param_df["param"])

    mult_param_dict = {}
    for param in param_list:
        mult_param_dict[param] = []
    mult_param_dict["model_acc"] = []


    for model_dir in model_dirs:
        param_fp = model_dir + "params.csv"

        param_df = pd.read_csv(param_fp)
        print(param_df)
        param_dict = convert_df_2_cols_to_dict(param_df, "param", "value")

        for param,val in param_dict.items():
            mult_param_dict[param].append(val)
        model_acc = model_dir_to_model_freq_dict[model_dir]
        mult_param_dict["model_acc"].append(model_acc)

    #get results
    for param in param_list:

        try:
            cur_param_val_list = mult_param_dict[param]
            print("\n%s"%param)
        except KeyError:
            continue
        model_accs = mult_param_dict["model_acc"]
        for model_acc,cur_param_val in zip(model_accs, cur_param_val_list):
            print("Model %.4f: %s"%(model_acc, cur_param_val))

    return

def get_and_save_combined_param_df(best_summ_df,model_dir_to_model_freq_dict,main_output_dir):

    model_dirs = list(best_summ_df["model_dir"].unique())

    #get param fields
    first_param_fp = model_dirs[1] + "params.csv"
    dummy_param_df = pd.read_csv(first_param_fp)
    param_list = list(dummy_param_df["param"])

    mult_param_dict = {}
    for param in param_list:
        mult_param_dict[param] = []
    mult_param_dict["model_acc"] = []


    for model_dir in model_dirs:
        param_fp = model_dir + "params.csv"

        param_df = pd.read_csv(param_fp)
        print(param_df)
        param_dict = convert_df_2_cols_to_dict(param_df, "param", "value")

        for param,val in param_dict.items():
            mult_param_dict[param].append(val)
        model_acc = model_dir_to_model_freq_dict[model_dir]
        mult_param_dict["model_acc"].append(model_acc)

    output_fp = main_output_dir + "all-model-params.txt"

    #get results
    with open(output_fp, "w") as f:
        for param in param_list:

            try:
                cur_param_val_list = mult_param_dict[param]
                print("\n%s"%param)
                f.write("\n%s\n"%param)
            except KeyError:
                continue
            model_accs = mult_param_dict["model_acc"]
            for model_acc,cur_param_val in zip(model_accs, cur_param_val_list):
                print("Model %.4f: %s"%(model_acc, cur_param_val))
                f.write("Model %.4f: %s\n"%(model_acc, cur_param_val))

    return

def get_model_dir_to_model_win_df(val_pair_to_model_dict,test_comp_fp):

    model_wins_list = []
    pairs = []
    for pair,model_dir in val_pair_to_model_dict.items():
        print()
        print(pair)
        cur_fp = model_dir + test_comp_fp
        cur_pair_comp_df = pd.read_csv(cur_fp)
        cur_pair_comp_df["pair"] = cur_pair_comp_df["infoID"] + "-" + cur_pair_comp_df["action"]

        cur_pair_comp_df=cur_pair_comp_df[cur_pair_comp_df["pair"]==pair].reset_index(drop=True)
        print(cur_pair_comp_df)
        model_is_winner = int(cur_pair_comp_df["model_is_winner"].iloc[0])
        model_wins_list.append(model_is_winner)
        pairs.append(pair)

    test_result_df = pd.DataFrame(data={"pair":pairs, "model_is_winner":model_wins_list})
    test_result_df = test_result_df[["pair", "model_is_winner"]]
    print("\ntest_result_df")
    print(test_result_df)
    num_wins = test_result_df["model_is_winner"].sum()
    model_win_freq = num_wins/float(test_result_df.shape[0])
    result_str = "Model won %d out of %d, or %.4f times"%(num_wins, test_result_df.shape[0],model_win_freq)
    print(result_str)
    return test_result_df,result_str

# Create a random subsample from the dataset with replacement
def subsample_func(dataset, ratio=1.0):
    sample = list()
    n_sample = round(len(dataset) * ratio)
    while len(sample) < n_sample:
        index = randrange(len(dataset))
        sample.append(dataset[index])
    return sample

def array_subsample_for_bagging(x, y, ratio=1.0,MOD_NUM=10):
    print("\nSubsampling for bagging....")
    new_x = []
    new_y = []
    n_samples = round(x.shape[0] * ratio)
    i=0
    while len(new_x) < n_samples:
        idx = randrange(x.shape[0])
        sampled_x = x[idx]
        sampled_y = y[idx]

        if i%MOD_NUM==0:
            print("\nGot %d out of %d samples"%(i, n_samples))
            print("sampled_x shape: %s"%str(sampled_x.shape))
            print("sampled_y shape: %s"%str(sampled_y.shape))

        new_x.append(sampled_x)
        new_y.append(sampled_y)

        i+=1

    x = np.asarray(new_x)
    y = np.asarray(new_y)
    print("\nFinal shape of x: %s"%str(x.shape))
    print("Final shape of y: %s"%str(y.shape))

    return x,y

def get_static_fts(action,infoID,x_array, dynamic_fts, static_fts,ft_to_idx_dict ,infoIDs,desired_actions,GET_1HOT_INFO_ID_FTS, GET_1HOT_ACTION_FTS):

    num_1hot_vecs_to_get = x_array.shape[0]

    all_1hot_vec_fts = []
    infoID_1hot_fts = [0 for idx in range(len(infoIDs))]
    action_1hot_fts = [0 for idx in range(len(desired_actions))]

    all_1hot_action_fts = []
    all_1hot_infoID_fts = []
    if GET_1HOT_INFO_ID_FTS == True:
        for idx,cur_infoID in enumerate(infoIDs):
            # flattened_fts.append(cur_infoID)
            # max_ft_dict_idx+=1
            if cur_infoID == infoID:
                infoID_1hot_fts[idx] = 1
            # ft_to_idx_dict[cur_infoID] = max_ft_dict_idx
            # idx_to_ft_dict[max_ft_dict_idx]=cur_infoID

        for idx in range(num_1hot_vecs_to_get):
            all_1hot_infoID_fts.append(np.asarray(infoID_1hot_fts))

        all_1hot_infoID_fts = np.asarray(all_1hot_infoID_fts)
        # x_array = np.concatenate([x_array, all_1hot_infoID_fts], axis=1)
        # print("\nx_array shape after all_1hot_infoID_fts concat: %s"%str(x_array.shape))


    if GET_1HOT_ACTION_FTS == True:
        for idx,cur_action in enumerate(desired_actions):
            # flattened_fts.append(cur_action)
            # max_ft_dict_idx+=1
            if cur_action == action:
                action_1hot_fts[idx] = 1
            # ft_to_idx_dict[cur_action] = max_ft_dict_idx
            # idx_to_ft_dict[max_ft_dict_idx]=cur_action

        for idx in range(num_1hot_vecs_to_get):
            all_1hot_action_fts.append(np.asarray(action_1hot_fts))

        all_1hot_action_fts = np.asarray(all_1hot_action_fts)

        all_1hot_static_fts = np.concatenate([all_1hot_infoID_fts, all_1hot_action_fts],axis=1)
        print("\nall_1hot_static_fts shape: %s"%str(all_1hot_static_fts.shape))



    return all_1hot_static_fts

def insert_static_1hot_fts_with_static_return(action,infoID,x_array, dynamic_fts, static_fts,ft_to_idx_dict ,infoIDs,desired_actions,GET_1HOT_INFO_ID_FTS, GET_1HOT_ACTION_FTS):

    #put 1 hot features now
    num_1hot_vecs_to_get = x_array.shape[0]
    x_array = x_array.reshape((x_array.shape[0], x_array.shape[1]*x_array.shape[2]))
    print("\nx_array: %s"%str(x_array.shape))

    num_dynamic_fts = len(dynamic_fts)

    new_ft_to_idx_dict = {}
    new_idx_to_ft_dict = {}
    flattened_fts = []
    print("\nflattened_fts")
    for idx in range(x_array.shape[1]):
        cur_dyn_idx = idx%num_dynamic_fts
        # print(cur_dyn_idx)
        cur_dyn_ft = dynamic_fts[cur_dyn_idx]
        new_idx_to_ft_dict[idx]= cur_dyn_ft + "_%d"%idx
        new_ft_to_idx_dict[cur_dyn_ft + "_%d"%idx] = idx
        flattened_fts.append(cur_dyn_ft + "_%d"%idx)
        # print(cur_dyn_ft + "_%d"%idx)
    # sys.exit(0)

    ft_to_idx_dict = dict(new_ft_to_idx_dict)
    idx_to_ft_dict = dict(new_idx_to_ft_dict)


    all_1hot_vec_fts = []
    infoID_1hot_fts = [0 for idx in range(len(infoIDs))]
    action_1hot_fts = [0 for idx in range(len(desired_actions))]

    max_ft_dict_idx = len(ft_to_idx_dict) - 1
    print("\nmax_ft_dict_idx: %d"%max_ft_dict_idx)

    all_1hot_action_fts = []
    all_1hot_infoID_fts = []
    if GET_1HOT_INFO_ID_FTS == True:
        for idx,cur_infoID in enumerate(infoIDs):
            flattened_fts.append(cur_infoID)
            max_ft_dict_idx+=1
            if cur_infoID == infoID:
                infoID_1hot_fts[idx] = 1
            ft_to_idx_dict[cur_infoID] = max_ft_dict_idx
            idx_to_ft_dict[max_ft_dict_idx]=cur_infoID

        for idx in range(num_1hot_vecs_to_get):
            all_1hot_infoID_fts.append(np.asarray(infoID_1hot_fts))

        all_1hot_infoID_fts = np.asarray(all_1hot_infoID_fts)
        x_array = np.concatenate([x_array, all_1hot_infoID_fts], axis=1)
        print("\nx_array shape after all_1hot_infoID_fts concat: %s"%str(x_array.shape))


    if GET_1HOT_ACTION_FTS == True:
        for idx,cur_action in enumerate(desired_actions):
            flattened_fts.append(cur_action)
            max_ft_dict_idx+=1
            if cur_action == action:
                action_1hot_fts[idx] = 1
            ft_to_idx_dict[cur_action] = max_ft_dict_idx
            idx_to_ft_dict[max_ft_dict_idx]=cur_action

        for idx in range(num_1hot_vecs_to_get):
            all_1hot_action_fts.append(np.asarray(action_1hot_fts))

        all_1hot_action_fts = np.asarray(all_1hot_action_fts)
        x_array = np.concatenate([x_array, all_1hot_action_fts], axis=1)
        print("\nx_array shape after all_1hot_action_fts concat: %s"%str(x_array.shape))

    # for ft in flattened_fts:
    #   print(ft)

    # print()
    num_flattened_fts = len(flattened_fts)
    # print("\nnum_flattened_fts: %d"%num_flattened_fts)

    # print("\nfts")
    # for idx,ft in idx_to_ft_dict.items():
    #   print("%d: %s"%(idx,ft))

    max_idx = num_flattened_fts - 1
    max_idx_ft = idx_to_ft_dict[max_idx]
    # print()
    # print(max_idx)
    # print(max_idx_ft)

    return x_array,ft_to_idx_dict,idx_to_ft_dict,flattened_fts

def insert_static_1hot_fts_v2_user_model(platform,user_status,infoID,user_type ,x_array, dynamic_fts, static_fts,ft_to_idx_dict ,infoIDs,GET_1HOT_INFO_ID_FTS, USE_WITHIN_PLATFORM_FEATURES_ONLY,TRAIN_CHILDREN_ONLY):

    #put 1 hot features now
    num_1hot_vecs_to_get = x_array.shape[0]
    x_array = x_array.reshape((x_array.shape[0], x_array.shape[1]*x_array.shape[2]))
    print("\nx_array: %s"%str(x_array.shape))

    num_dynamic_fts = len(dynamic_fts)

    new_ft_to_idx_dict = {}
    new_idx_to_ft_dict = {}
    flattened_fts = []
    print("\nflattened_fts")
    for idx in range(x_array.shape[1]):
        cur_dyn_idx = idx%num_dynamic_fts
        # print(cur_dyn_idx)
        cur_dyn_ft = dynamic_fts[cur_dyn_idx]
        new_idx_to_ft_dict[idx]= cur_dyn_ft + "_%d"%idx
        new_ft_to_idx_dict[cur_dyn_ft + "_%d"%idx] = idx
        flattened_fts.append(cur_dyn_ft + "_%d"%idx)
        # print(cur_dyn_ft + "_%d"%idx)
    # sys.exit(0)

    ft_to_idx_dict = dict(new_ft_to_idx_dict)
    idx_to_ft_dict = dict(new_idx_to_ft_dict)


    all_1hot_vec_fts = []
    infoID_1hot_fts = [0 for idx in range(len(infoIDs))]
    platform_1hot_fts = [0]
    user_status_1hot_fts = [0]
    user_type_1hot_fts = [0]
    # action_1hot_fts = [0 for idx in range(len(desired_actions))]

    max_ft_dict_idx = len(ft_to_idx_dict) - 1
    print("\nmax_ft_dict_idx: %d"%max_ft_dict_idx)

    # all_1hot_action_fts = []
    all_1hot_infoID_fts = []
    all_1hot_platform_fts = []
    all_1hot_user_status_fts = []
    all_1hot_user_type_fts = []
    if GET_1HOT_INFO_ID_FTS == True:
        for idx,cur_infoID in enumerate(infoIDs):
            flattened_fts.append(cur_infoID)
            max_ft_dict_idx+=1
            if cur_infoID == infoID:
                infoID_1hot_fts[idx] = 1
            ft_to_idx_dict[cur_infoID] = max_ft_dict_idx
            idx_to_ft_dict[max_ft_dict_idx]=cur_infoID

        for idx in range(num_1hot_vecs_to_get):
            all_1hot_infoID_fts.append(np.asarray(infoID_1hot_fts))

        all_1hot_infoID_fts = np.asarray(all_1hot_infoID_fts)
        x_array = np.concatenate([x_array, all_1hot_infoID_fts], axis=1)
        print("\nx_array shape after all_1hot_infoID_fts concat: %s"%str(x_array.shape))

    user_fts = ["is_child", "is_new", "is_twitter"]

    # if USE_WITHIN_PLATFORM_FEATURES_ONLY == True:
    #     user_fts.remove("is_twitter")

    # if TRAIN_CHILDREN_ONLY == True:
    #     user_fts.remove("is_child")

    if USE_WITHIN_PLATFORM_FEATURES_ONLY == False:

        flattened_fts.append("is_twitter")
        max_ft_dict_idx+=1
        if platform == "twitter":
            platform_1hot_fts[0] = 1
        ft_to_idx_dict["is_twitter"] = max_ft_dict_idx
        idx_to_ft_dict[max_ft_dict_idx]="is_twitter"

        for idx in range(num_1hot_vecs_to_get):
            all_1hot_platform_fts.append(np.asarray(platform_1hot_fts))

        all_1hot_platform_fts = np.asarray(all_1hot_platform_fts)
        x_array = np.concatenate([x_array, all_1hot_platform_fts], axis=1)
        print("\nx_array shape after all_1hot_platform_fts concat: %s"%str(x_array.shape))

    if TRAIN_CHILDREN_ONLY==False:

        flattened_fts.append("is_child")
        max_ft_dict_idx+=1
        if user_type == "child":
            user_type_1hot_fts[0] = 1
        ft_to_idx_dict["is_child"] = max_ft_dict_idx
        idx_to_ft_dict[max_ft_dict_idx]="is_child"

        for idx in range(num_1hot_vecs_to_get):
            all_1hot_user_type_fts.append(np.asarray(user_type_1hot_fts))

        all_1hot_user_type_fts = np.asarray(all_1hot_user_type_fts)
        x_array = np.concatenate([x_array, all_1hot_user_type_fts], axis=1)
        print("\nx_array shape after all_1hot_user_type_fts concat: %s"%str(x_array.shape))

    #get "is new" feature
    flattened_fts.append("is_new")
    max_ft_dict_idx+=1
    if user_status == "new":
        user_status_1hot_fts[0] = 1
    ft_to_idx_dict["is_new"] = max_ft_dict_idx
    idx_to_ft_dict[max_ft_dict_idx]="is_new"

    for idx in range(num_1hot_vecs_to_get):
        all_1hot_user_status_fts.append(np.asarray(user_status_1hot_fts))

    all_1hot_user_status_fts = np.asarray(all_1hot_user_status_fts)
    x_array = np.concatenate([x_array, all_1hot_user_status_fts], axis=1)
    print("\nx_array shape after all_1hot_user_status_fts concat: %s"%str(x_array.shape))

    num_flattened_fts = len(flattened_fts)
    # print("\nnum_flattened_fts: %d"%num_flattened_fts)

    # print("\nfts")
    # for idx,ft in idx_to_ft_dict.items():
    #   print("%d: %s"%(idx,ft))

    max_idx = num_flattened_fts - 1
    max_idx_ft = idx_to_ft_dict[max_idx]
    # print()
    # print(max_idx)
    # print(max_idx_ft)

    return x_array,ft_to_idx_dict,idx_to_ft_dict,flattened_fts

def insert_static_1hot_fts(action,infoID,x_array, dynamic_fts, static_fts,ft_to_idx_dict ,infoIDs,desired_actions,GET_1HOT_INFO_ID_FTS, GET_1HOT_ACTION_FTS):

    #put 1 hot features now
    num_1hot_vecs_to_get = x_array.shape[0]
    x_array = x_array.reshape((x_array.shape[0], x_array.shape[1]*x_array.shape[2]))
    print("\nx_array: %s"%str(x_array.shape))

    num_dynamic_fts = len(dynamic_fts)

    new_ft_to_idx_dict = {}
    new_idx_to_ft_dict = {}
    flattened_fts = []
    print("\nflattened_fts")
    for idx in range(x_array.shape[1]):
        cur_dyn_idx = idx%num_dynamic_fts
        # print(cur_dyn_idx)
        cur_dyn_ft = dynamic_fts[cur_dyn_idx]
        new_idx_to_ft_dict[idx]= cur_dyn_ft + "_%d"%idx
        new_ft_to_idx_dict[cur_dyn_ft + "_%d"%idx] = idx
        flattened_fts.append(cur_dyn_ft + "_%d"%idx)
        # print(cur_dyn_ft + "_%d"%idx)
    # sys.exit(0)

    ft_to_idx_dict = dict(new_ft_to_idx_dict)
    idx_to_ft_dict = dict(new_idx_to_ft_dict)


    all_1hot_vec_fts = []
    infoID_1hot_fts = [0 for idx in range(len(infoIDs))]
    action_1hot_fts = [0 for idx in range(len(desired_actions))]

    max_ft_dict_idx = len(ft_to_idx_dict) - 1
    print("\nmax_ft_dict_idx: %d"%max_ft_dict_idx)

    all_1hot_action_fts = []
    all_1hot_infoID_fts = []
    if GET_1HOT_INFO_ID_FTS == True:
        for idx,cur_infoID in enumerate(infoIDs):
            flattened_fts.append(cur_infoID)
            max_ft_dict_idx+=1
            if cur_infoID == infoID:
                infoID_1hot_fts[idx] = 1
            ft_to_idx_dict[cur_infoID] = max_ft_dict_idx
            idx_to_ft_dict[max_ft_dict_idx]=cur_infoID

        for idx in range(num_1hot_vecs_to_get):
            all_1hot_infoID_fts.append(np.asarray(infoID_1hot_fts))

        all_1hot_infoID_fts = np.asarray(all_1hot_infoID_fts)
        x_array = np.concatenate([x_array, all_1hot_infoID_fts], axis=1)
        print("\nx_array shape after all_1hot_infoID_fts concat: %s"%str(x_array.shape))


    if GET_1HOT_ACTION_FTS == True:
        for idx,cur_action in enumerate(desired_actions):
            flattened_fts.append(cur_action)
            max_ft_dict_idx+=1
            if cur_action == action:
                action_1hot_fts[idx] = 1
            ft_to_idx_dict[cur_action] = max_ft_dict_idx
            idx_to_ft_dict[max_ft_dict_idx]=cur_action

        for idx in range(num_1hot_vecs_to_get):
            all_1hot_action_fts.append(np.asarray(action_1hot_fts))

        all_1hot_action_fts = np.asarray(all_1hot_action_fts)
        x_array = np.concatenate([x_array, all_1hot_action_fts], axis=1)
        print("\nx_array shape after all_1hot_action_fts concat: %s"%str(x_array.shape))

    # for ft in flattened_fts:
    #   print(ft)

    # print()
    num_flattened_fts = len(flattened_fts)
    # print("\nnum_flattened_fts: %d"%num_flattened_fts)

    # print("\nfts")
    # for idx,ft in idx_to_ft_dict.items():
    #   print("%d: %s"%(idx,ft))

    max_idx = num_flattened_fts - 1
    max_idx_ft = idx_to_ft_dict[max_idx]
    # print()
    # print(max_idx)
    # print(max_idx_ft)

    return x_array,ft_to_idx_dict,idx_to_ft_dict,flattened_fts

def get_parent_uids(df, parent_node_col="parentID", node_col="nodeID", root_node_col="rootID", user_col="nodeUserID"):
    """
    :return: adds parentUserID column with user id of the parent if it exits in df
    if it doesn't exist, uses the user id of the root instead
    if both doesn't exist: NaN
    """
    tweet_uids = pd.Series(df[user_col].values, index=df[node_col]).to_dict()

    df['parentUserID'] = df[parent_node_col].map(tweet_uids)

    df.loc[(df[root_node_col] != df[node_col]) & (df['parentUserID'].isnull()), 'parentUserID'] = \
        df[(df[root_node_col] != df[node_col]) & (df['parentUserID'].isnull())][root_node_col].map(tweet_uids)

    df = df[df['nodeUserID'] != df['parentUserID']]

    return df

def get_summ_df_v2_PARAM_REQS(summ_baseline_fp,DESIRED_EXTRACT_OUTPUT_SIZE,cands,PARAM_REQS,exception_list=[]):
    #get best dfs
    all_summ_dfs = []
    for cand_dir in cands:
        try:
            param_fp = cand_dir + "params.csv"
            param_df = pd.read_csv(param_fp)
            param_cols = list(param_df["param"])
            if ("EXTRACT_OUTPUT_SIZE" not in param_cols):
                if cand_dir not in exception_list:
                    continue

            #get param dict
            param_dict = convert_df_2_cols_to_dict(param_df, "param", "value")

            param_req_is_met = True
            for req_param,req_val in PARAM_REQS.items():
                try:
                    print(req_param)
                    param_val = param_dict[req_param]
                except KeyError:
                    print("key error, moving on")
                    param_req_is_met = False
                    break

                print("\nparam val and req val")
                print(req_val)
                print(param_val)

                if str(param_val)!=str(req_val):
                    print("vals don't match")
                    param_req_is_met = False
                    break

            if param_req_is_met == False:
                continue


            try:
                param_df = param_df[param_df["param"]=="EXTRACT_OUTPUT_SIZE"].reset_index(drop=True)
                EXTRACT_OUTPUT_SIZE = int(param_df["value"].iloc[0])
            except:
                EXTRACT_OUTPUT_SIZE = DESIRED_EXTRACT_OUTPUT_SIZE
            print("\nEXTRACT_OUTPUT_SIZE")
            print(EXTRACT_OUTPUT_SIZE)
            # sys.exit(0)
            if DESIRED_EXTRACT_OUTPUT_SIZE != EXTRACT_OUTPUT_SIZE:
                continue
            summ_fp =cand_dir + summ_baseline_fp
            summ_df = pd.read_csv(summ_fp)
        except FileNotFoundError:
            print("\nCannot find:")
            print(summ_fp)
            continue

        summ_df["model_dir"] = cand_dir
        all_summ_dfs.append(summ_df)

    summ_df = pd.concat(all_summ_dfs).reset_index(drop=True)
    print(summ_df)

    return summ_df

def get_summ_df(summ_baseline_fp,DESIRED_EXTRACT_OUTPUT_SIZE,cands,exception_list=[]):
    #get best dfs
    all_summ_dfs = []
    for cand_dir in cands:
        try:
            param_fp = cand_dir + "params.csv"
            param_df = pd.read_csv(param_fp)
            param_cols = list(param_df["param"])
            if ("EXTRACT_OUTPUT_SIZE" not in param_cols):
                if cand_dir not in exception_list:
                    continue
            try:
                param_df = param_df[param_df["param"]=="EXTRACT_OUTPUT_SIZE"].reset_index(drop=True)
                EXTRACT_OUTPUT_SIZE = int(param_df["value"].iloc[0])
            except:
                EXTRACT_OUTPUT_SIZE = DESIRED_EXTRACT_OUTPUT_SIZE
            print("\nEXTRACT_OUTPUT_SIZE")
            print(EXTRACT_OUTPUT_SIZE)
            # sys.exit(0)
            if DESIRED_EXTRACT_OUTPUT_SIZE != EXTRACT_OUTPUT_SIZE:
                continue
            summ_fp =cand_dir + summ_baseline_fp
            summ_df = pd.read_csv(summ_fp)
        except FileNotFoundError:
            continue

        summ_df["model_dir"] = cand_dir
        all_summ_dfs.append(summ_df)

    summ_df = pd.concat(all_summ_dfs).reset_index(drop=True)
    print(summ_df)

    return summ_df

def get_best_summ_df_and_model_dir_to_model_freq_dict(summ_df,TOP_RESULT_NUM):
    model_dir_to_model_freq_dict = {}
    top_summ_results = []
    for i in range(TOP_RESULT_NUM):
        print("\nGetting top df %d of %d"%((i+1),TOP_RESULT_NUM))
        temp = summ_df.copy()
        temp = temp[temp["info_piece"]=="model_wins_div_by_num_trials_without_ties"].reset_index(drop=True)
        max_val = temp["value"].max()
        temp = temp[temp["value"]==max_val].reset_index(drop=True)
        # print(temp)
        best_model_dir_list = temp["model_dir"].unique()
        # print(best_model_dir_list)
        best_model_dir = best_model_dir_list[0]
        model_dir_to_model_freq_dict[best_model_dir] = max_val
        # print(best_model_dir)
        best_summ_df = summ_df[summ_df["model_dir"]==best_model_dir]
        best_summ_df["rank"] = i+1
        print(best_summ_df)
        summ_df = summ_df[summ_df["model_dir"]!=best_model_dir].reset_index(drop=True)
        top_summ_results.append(best_summ_df)

    best_summ_df = pd.concat(top_summ_results)
    print(best_summ_df)
    return best_summ_df,model_dir_to_model_freq_dict

def get_best_summ_df_and_model_dir_to_model_freq_dict_v2_with_metric_of_interest(summ_df,TOP_RESULT_NUM,metric_of_interest,desired_test_dir,desired_fp):
    model_dir_to_model_freq_dict = {}
    top_summ_results = []
    for i in range(TOP_RESULT_NUM):
        print("\nGetting top df %d of %d"%((i+1),TOP_RESULT_NUM))
        temp = summ_df.copy()
        temp = temp[temp["info_piece"]=="model_wins_div_by_num_trials_without_ties"].reset_index(drop=True)
        max_val = temp["value"].max()
        temp = temp[temp["value"]==max_val].reset_index(drop=True)
        # print(temp)
        best_model_dir_list = temp["model_dir"].unique()
        # print(best_model_dir_list)
        best_model_dir = best_model_dir_list[0]
        # print("\nbest_model_dir")
        # print(best_model_dir)
        model_dir_to_model_freq_dict[best_model_dir] = max_val

        best_summ_df = summ_df[summ_df["model_dir"]==best_model_dir]
        best_summ_df["rank"] = i+1

        summ_df = summ_df[summ_df["model_dir"]!=best_model_dir].reset_index(drop=True)

        #get rmse
        metric_result_dir = best_model_dir + desired_test_dir
        metric_result_fp = metric_result_dir + desired_fp
        # print("\nmetric_result_fp ")
        # print(metric_result_fp )
        metric_result_df = pd.read_csv(metric_result_fp )
        total_model_error = metric_result_df["VAM_%s"%metric_of_interest].sum()
        total_baseline_error = metric_result_df["Shifted-Baseline_%s"%metric_of_interest].sum()
        best_summ_df["total_model_%s"%metric_of_interest]=total_model_error
        best_summ_df["total_baseline_%s"%metric_of_interest]=total_baseline_error

        print()
        print(best_model_dir)
        print(best_summ_df)
        # sys.exit(0)




        top_summ_results.append(best_summ_df)

    best_summ_df = pd.concat(top_summ_results)
    print(best_summ_df)
    return best_summ_df,model_dir_to_model_freq_dict

def get_best_summ_df_and_model_dir_to_model_freq_dict_v4_sort_with_rmse(summ_df,TOP_RESULT_NUM,metric_of_interest,desired_test_dir,desired_fp,action_to_platform_dict,platforms):
    model_dir_to_model_freq_dict = {}
    top_summ_results = []
    for i in range(TOP_RESULT_NUM):
        print("\nGetting top df %d of %d"%((i+1),TOP_RESULT_NUM))
        temp = summ_df.copy()
        temp = temp[temp["info_piece"]=="model_wins_div_by_num_trials_without_ties"].reset_index(drop=True)
        max_val = temp["value"].max()
        temp = temp[temp["value"]==max_val].reset_index(drop=True)
        # print(temp)
        best_model_dir_list = temp["model_dir"].unique()
        # print(best_model_dir_list)
        best_model_dir = best_model_dir_list[0]
        # print("\nbest_model_dir")
        # print(best_model_dir)
        model_dir_to_model_freq_dict[best_model_dir] = max_val

        best_summ_df = summ_df[summ_df["model_dir"]==best_model_dir]
        # best_summ_df["rank"] = i+1

        summ_df = summ_df[summ_df["model_dir"]!=best_model_dir].reset_index(drop=True)

        #get rmse
        metric_result_dir = best_model_dir + desired_test_dir
        metric_result_fp = metric_result_dir + desired_fp
        print("\nmetric_result_fp ")
        print(metric_result_fp )
        # sys.exit(0)
        metric_result_df = pd.read_csv(metric_result_fp )
        total_model_error = metric_result_df["VAM_%s"%metric_of_interest].sum()
        total_baseline_error = metric_result_df["Shifted-Baseline_%s"%metric_of_interest].sum()
        best_summ_df["total_model_%s"%metric_of_interest]=total_model_error
        best_summ_df["total_baseline_%s"%metric_of_interest]=total_baseline_error

        print()
        print(best_model_dir)
        print(best_summ_df)
        # sys.exit(0)

        #get platform wins
        metric_result_df["platform"] = metric_result_df["action"].map(action_to_platform_dict)
        print(metric_result_df)

        for platform in platforms:
            temp = metric_result_df[metric_result_df["platform"]==platform].reset_index(drop=True)
            best_summ_df["VAM_%s_wins"%platform] = temp["model_is_winner"].sum()
            best_summ_df["total_%s_model_%s"%(platform ,metric_of_interest)] = temp["VAM_%s"%metric_of_interest].sum()
            best_summ_df["total_%s_baseline_%s"%(platform ,metric_of_interest)] = temp["Shifted-Baseline_%s"%metric_of_interest].sum()




        top_summ_results.append(best_summ_df)

    best_summ_df = pd.concat(top_summ_results)
    print(best_summ_df)
    best_summ_df["neg_model_metric"]=-1 * best_summ_df["total_model_%s"%metric_of_interest]
    best_summ_df = best_summ_df.sort_values(["value", "neg_model_metric"], ascending=False).reset_index(drop=True)
    best_summ_df =best_summ_df.drop("neg_model_metric",axis=1)
    best_summ_df["rank"] = [i+1 for i in range(best_summ_df.shape[0])]
    print(best_summ_df)
    # sys.exit(0)

    return best_summ_df,model_dir_to_model_freq_dict

def get_best_summ_df_and_model_dir_to_model_freq_dict_v3_with_wins_by_platform(summ_df,TOP_RESULT_NUM,metric_of_interest,desired_test_dir,desired_fp,action_to_platform_dict,platforms):
    model_dir_to_model_freq_dict = {}
    top_summ_results = []
    for i in range(TOP_RESULT_NUM):
        print("\nGetting top df %d of %d"%((i+1),TOP_RESULT_NUM))
        temp = summ_df.copy()
        temp = temp[temp["info_piece"]=="model_wins_div_by_num_trials_without_ties"].reset_index(drop=True)
        max_val = temp["value"].max()
        temp = temp[temp["value"]==max_val].reset_index(drop=True)
        # print(temp)
        best_model_dir_list = temp["model_dir"].unique()
        # print(best_model_dir_list)
        best_model_dir = best_model_dir_list[0]
        # print("\nbest_model_dir")
        # print(best_model_dir)
        model_dir_to_model_freq_dict[best_model_dir] = max_val

        best_summ_df = summ_df[summ_df["model_dir"]==best_model_dir]
        best_summ_df["rank"] = i+1

        summ_df = summ_df[summ_df["model_dir"]!=best_model_dir].reset_index(drop=True)

        #get rmse
        metric_result_dir = best_model_dir + desired_test_dir
        metric_result_fp = metric_result_dir + desired_fp
        print("\nmetric_result_fp ")
        print(metric_result_fp )
        # sys.exit(0)
        metric_result_df = pd.read_csv(metric_result_fp )
        total_model_error = metric_result_df["VAM_%s"%metric_of_interest].sum()
        total_baseline_error = metric_result_df["Shifted-Baseline_%s"%metric_of_interest].sum()
        best_summ_df["total_model_%s"%metric_of_interest]=total_model_error
        best_summ_df["total_baseline_%s"%metric_of_interest]=total_baseline_error

        print()
        print(best_model_dir)
        print(best_summ_df)
        # sys.exit(0)

        #get platform wins
        metric_result_df["platform"] = metric_result_df["action"].map(action_to_platform_dict)
        print(metric_result_df)

        for platform in platforms:
            temp = metric_result_df[metric_result_df["platform"]==platform].reset_index(drop=True)
            best_summ_df["VAM_%s_wins"%platform] = temp["model_is_winner"].sum()
            best_summ_df["total_%s_model_%s"%(platform ,metric_of_interest)] = temp["VAM_%s"%metric_of_interest].sum()
            best_summ_df["total_%s_baseline_%s"%(platform ,metric_of_interest)] = temp["Shifted-Baseline_%s"%metric_of_interest].sum()




        top_summ_results.append(best_summ_df)

    best_summ_df = pd.concat(top_summ_results)
    # print(best_summ_df)
    # best_summ_df["neg_model_metric"]=-1 * best_summ_df["total_model_%s"%metric_of_interest]
    # best_summ_df = best_summ_df.sort_values(["total_model_%s"%metric_of_interest, "neg_model_metric"]).reset_index(drop=True)
    # best_summ_df =best_summ_df.drop("neg_model_metric",axis=1)
    # best_summ_df["rank"] = [i+1 for i in range(best_summ_df.shape[0])]
    print(best_summ_df)
    # sys.exit(0)

    return best_summ_df,model_dir_to_model_freq_dict

def get_best_summ_df_and_model_dir_to_model_freq_dict_v4_choose_lowest_error(summ_df,TOP_RESULT_NUM,metric_of_interest,desired_test_dir,desired_fp,action_to_platform_dict,platforms):
    model_dir_to_model_freq_dict = {}
    top_summ_results = []
    for i in range(TOP_RESULT_NUM):
        print("\nGetting top df %d of %d"%((i+1),TOP_RESULT_NUM))
        temp = summ_df.copy()
        temp = temp[temp["info_piece"]=="model_wins_div_by_num_trials_without_ties"].reset_index(drop=True)
        print(temp)
        sys.exit(0)


        min_val = temp["value"].min()
        temp = temp[temp["value"]==min_val].reset_index(drop=True)
        # print(temp)
        best_model_dir_list = temp["model_dir"].unique()
        # print(best_model_dir_list)
        best_model_dir = best_model_dir_list[0]
        # print("\nbest_model_dir")
        # print(best_model_dir)
        model_dir_to_model_freq_dict[best_model_dir] = min_val

        best_summ_df = summ_df[summ_df["model_dir"]==best_model_dir]
        best_summ_df["rank"] = i+1

        summ_df = summ_df[summ_df["model_dir"]!=best_model_dir].reset_index(drop=True)

        #get rmse
        metric_result_dir = best_model_dir + desired_test_dir
        metric_result_fp = metric_result_dir + desired_fp
        print("\nmetric_result_fp ")
        print(metric_result_fp )
        # sys.exit(0)
        metric_result_df = pd.read_csv(metric_result_fp )
        total_model_error = metric_result_df["VAM_%s"%metric_of_interest].sum()
        total_baseline_error = metric_result_df["Shifted-Baseline_%s"%metric_of_interest].sum()
        best_summ_df["total_model_%s"%metric_of_interest]=total_model_error
        best_summ_df["total_baseline_%s"%metric_of_interest]=total_baseline_error

        print()
        print(best_model_dir)
        print(best_summ_df)
        # sys.exit(0)

        #get platform wins
        metric_result_df["platform"] = metric_result_df["action"].map(action_to_platform_dict)
        print(metric_result_df)

        for platform in platforms:
            temp = metric_result_df[metric_result_df["platform"]==platform].reset_index(drop=True)
            best_summ_df["VAM_%s_wins"%platform] = temp["model_is_winner"].sum()
            best_summ_df["total_%s_model_%s"%(platform ,metric_of_interest)] = temp["VAM_%s"%metric_of_interest].sum()
            best_summ_df["total_%s_baseline_%s"%(platform ,metric_of_interest)] = temp["Shifted-Baseline_%s"%metric_of_interest].sum()




        top_summ_results.append(best_summ_df)

    best_summ_df = pd.concat(top_summ_results)
    # print(best_summ_df)
    # best_summ_df["neg_model_metric"]=-1 * best_summ_df["total_model_%s"%metric_of_interest]
    # best_summ_df = best_summ_df.sort_values(["total_model_%s"%metric_of_interest, "neg_model_metric"]).reset_index(drop=True)
    # best_summ_df =best_summ_df.drop("neg_model_metric",axis=1)
    # best_summ_df["rank"] = [i+1 for i in range(best_summ_df.shape[0])]
    print(best_summ_df)
    # sys.exit(0)

    return best_summ_df,model_dir_to_model_freq_dict

def get_comp_df(comp_fp,model_dirs):
    all_comp_dfs = []
    for model_dir in model_dirs:
        cur_full_comp_fp = model_dir + comp_fp
        comp_df = pd.read_csv(cur_full_comp_fp)
        comp_df["model_dir"] = model_dir
        print(comp_df)
        all_comp_dfs.append(comp_df)

        #get best 1 results
    comp_df = pd.concat(all_comp_dfs).reset_index(drop=True)
    print(comp_df)
    return comp_df

def get_model_results_df(comp_df):
    # m1_get_model_win_info(comp_df,winner_tag)
    all_1_results = comp_df[comp_df["model_is_winner"]==1]
    all_1_results = all_1_results.drop_duplicates(["infoID", "action"]).reset_index(drop=True)
    all_1_results["pair"] = all_1_results["infoID"] + "-" + all_1_results["action"]
    print(all_1_results)
    all_0_results = comp_df[comp_df["model_is_winner"]==0].drop_duplicates(["infoID", "action"]).reset_index(drop=True)
    all_0_results["pair"] = all_0_results["infoID"] + "-" + all_0_results["action"]
    drop_pairs = all_1_results["pair"].unique()
    all_0_results = all_0_results[~all_0_results["pair"].isin(drop_pairs)]
    all_results = pd.concat([all_1_results ,all_0_results]).reset_index(drop=True)
    # all_results = all_results[[""]]
    print(all_results)

    model_wins = all_1_results.shape[0]
    model_losses = all_0_results.shape[0]
    model_win_freq = model_wins/float(all_results.shape[0]) * 100
    result_str = "\nModel won %d out of %d times, or %.2f times"%(model_wins, all_results.shape[0], model_win_freq)
    print(result_str)

    return all_results,result_str

def agg_y_array_v2_mult_output_fts(y):
    # print("\ny reshape")
    # y = y.reshape((y.shape[0],y.shape[1]*y.shape[2] ))
    print(y.shape)

    print("\ny agg")
    y = y.sum(axis=1)
    # y = y.reshape((y.shape[0], 1, 1))
    print(y.shape)

    return y

def agg_y_array(y):
    print("\ny reshape")
    y = y.reshape((y.shape[0],y.shape[1]*y.shape[2] ))
    print(y.shape)

    print("\ny agg")
    y = y.sum(axis=1)
    y = y.reshape((y.shape[0], 1, 1))
    print(y.shape)

    return y

def count_urls(df,GRAN,platform,url_col="urls_linked",MOD_NUM=10000):

    #reset idx
    df = df[df["platform"]==platform].reset_index(drop=True)

    #FLOOR IT
    df["nodeTime"] = df["nodeTime"].dt.floor(GRAN)

    df = df[["nodeTime",url_col]]
    df = df.dropna()
    print("\ndf after drop")
    print(df)

    url_counts = []
    url_list_list = list(df[url_col])
    num_records = len(url_list_list)
    for i,url_list_str in enumerate(url_list_list):
        url_list = literal_eval(url_list_str)
        # print(url_list)
        num_urls = len(url_list)
        if i%MOD_NUM == 0:
            print("Got %d url records of %d"%(i, num_records))
        url_counts.append(num_urls)

    df["num_%s_urls"%platform] = url_counts
    df["num_%s_urls"%platform] = df.groupby(["nodeTime"])["num_%s_urls"%platform].transform("sum")
    df = df[["nodeTime","num_%s_urls"%platform]].drop_duplicates().reset_index(drop=True)
    print(df)
    return df

def count_new_and_old_users_per_platform_v2_verify(df,platform,GRAN):

    df = df[df["platform"]==platform].reset_index(drop=True)

    #FLOOR IT
    df["nodeTime"] = df["nodeTime"].dt.floor(GRAN)

    #mark bdays
    print("\nGetting child user bdays...")
    df["nodeUserID_birthdate"] = df.groupby(["nodeUserID"])["nodeTime"].transform("min")

    #if nodetime==birthday, user is new
    print("\nGetting is new col for child users...")
    df["is_new_user"] = [1 if cur_time==bdate else 0 for cur_time,bdate in zip(list(df["nodeTime"]), list(df["nodeUserID_birthdate"]))]
    df["is_old_user"] = [0 if cur_time==bdate else 1 for cur_time,bdate in zip(list(df["nodeTime"]), list(df["nodeUserID_birthdate"]))]

    gt_user_count_over_time = df[["nodeTime", "nodeUserID"]].groupby("nodeTime")["nodeUserID"].transform("nunique").drop_duplicates().sum()

    df = df[["nodeTime", "nodeUserID", "is_new_user", "is_old_user"]].drop_duplicates().reset_index(drop=True)
    df["num_%s_new_users"%platform] = df.groupby(["nodeTime"])["is_new_user"].transform("sum")
    df["num_%s_old_users"%platform] = df.groupby(["nodeTime"])["is_old_user"].transform("sum")
    df = df[["nodeTime", "num_%s_new_users"%platform,"num_%s_old_users"%platform]].reset_index(drop=True)
    df = df.drop_duplicates().reset_index(drop=True)

    my_user_count_over_time = df["num_%s_new_users"%platform].sum() + df["num_%s_old_users"%platform].sum()

    print("\ngt_user_count_over_time: %d"%gt_user_count_over_time)
    print("\nmy_user_count_over_time: %d"%my_user_count_over_time)
    # if my_user_count_over_time != gt_user_count_over_time:
    #     print("Error! Counts don't match")
    #     sys.exit(0)
    # else:
    #     print("Counts are ok!")

    return df

def count_new_and_old_users_per_platform(df,platform,GRAN):

    df = df[df["platform"]==platform].reset_index(drop=True)

    #FLOOR IT
    df["nodeTime"] = df["nodeTime"].dt.floor(GRAN)

    #mark bdays
    print("\nGetting child user bdays...")
    df["nodeUserID_birthdate"] = df.groupby(["nodeUserID"])["nodeTime"].transform("min")

    #if nodetime==birthday, user is new
    print("\nGetting is new col for child users...")
    df["is_new_user"] = [1 if cur_time==bdate else 0 for cur_time,bdate in zip(list(df["nodeTime"]), list(df["nodeUserID_birthdate"]))]
    df["is_old_user"] = [0 if cur_time==bdate else 1 for cur_time,bdate in zip(list(df["nodeTime"]), list(df["nodeUserID_birthdate"]))]

    df = df[["nodeTime", "nodeUserID", "is_new_user", "is_old_user"]].drop_duplicates().reset_index(drop=True)
    df["num_%s_new_users"%platform] = df.groupby(["nodeTime"])["is_new_user"].transform("sum")
    df["num_%s_old_users"%platform] = df.groupby(["nodeTime"])["is_old_user"].transform("sum")
    df = df[["nodeTime", "num_%s_new_users"%platform,"num_%s_old_users"%platform]].reset_index(drop=True)
    df = df.drop_duplicates().reset_index(drop=True)

    return df

def fix_twitter_stance_fts(df,start,end,GRAN,kickout_other_cols=True):

    # num_records = df.shape[0]

    #stance stuff
    stance_list = ["stance.?","stance.am","stance.pm"]

    # for col in stance_list + ["stance"]:
    #     df[col] = df[col].astype(str)

    #split dfs
    twitter_df = df[df["platform"]=="twitter"]
    df = df[df["platform"]=="youtube"]

    #create fts
    print("\nGetting stance fts...")

    twitter_df["twitter_stance.?"] = twitter_df["stance"].isin(["?"]).astype("int32")
    twitter_df["twitter_stance.am"] = twitter_df["stance"].isin(["am"]).astype("int32")
    twitter_df["twitter_stance.pm"] = twitter_df["stance"].isin(["pm"]).astype("int32")
    twitter_df = twitter_df[["nodeTime","twitter_stance.?","twitter_stance.am","twitter_stance.pm"]]
    twitter_df = twitter_df.fillna(0)
    for col in stance_list:
        df = df.rename(columns={col:"youtube_%s"%col})
    df = df[["nodeTime","youtube_stance.?","youtube_stance.am","youtube_stance.pm"]]
    df = df.fillna(0)
    print(df)

    date_df = create_blank_date_df(start,end,GRAN)
    df = pd.merge(date_df, df, on="nodeTime", how="outer")
    twitter_df = pd.merge(date_df, twitter_df, on="nodeTime", how="outer")

    df = df.sort_values("nodeTime").reset_index(drop=True)
    twitter_df = twitter_df.sort_values("nodeTime").reset_index(drop=True)

    #recombine
    # df = pd.concat([df, twitter_df])
    print("\nMerging data...")
    twitter_df = twitter_df[["twitter_stance.?","twitter_stance.am","twitter_stance.pm"]]
    df = pd.concat([df, twitter_df], axis=1)
    # df = pd.merge(df, twitter_df, on="nodeTime", how="outer")
    # df = df.fillna(0)


    final_cols = ["nodeTime","twitter_stance.?","twitter_stance.am","twitter_stance.pm","youtube_stance.?","youtube_stance.am","youtube_stance.pm"]
    if kickout_other_cols==True:
        df = df[final_cols]
    df = df.dropna()
    df = df.sort_values("nodeTime").reset_index(drop=True)

    for col in final_cols:

        if col != "nodeTime":
            df[col] = df.groupby("nodeTime")[col].transform("mean")
            print()
            print(col)
            print(df[col].value_counts())
    df =  df.drop_duplicates().reset_index(drop=True)


    print(df)

    return df

def m1_make_comp_df(vam_df, baseline_df,merge_cols,rename_cols,vam_tag,baseline_tag,metric_of_interest):

    for rename_col in rename_cols:
        vam_df = vam_df.rename(columns={rename_col:vam_tag +"_"+ rename_col})
        baseline_df = baseline_df.rename(columns={rename_col:baseline_tag +"_"+ rename_col})

    vam_metric_col = vam_tag + "_" + metric_of_interest
    baseline_metric_col = baseline_tag + "_" + metric_of_interest

    comp_df = pd.merge(vam_df, baseline_df, on=merge_cols, how="inner")
    vam_errors = comp_df[vam_metric_col]
    baseline_errors = comp_df[baseline_metric_col]
    winner_tag = "model_is_winner"
    comp_df[winner_tag] = [1 if (vam_error < baseline_error) else 0 if (vam_error > baseline_error) else "tie" for (vam_error,baseline_error) in zip(vam_errors,baseline_errors) ]
    print(comp_df)
    return comp_df,winner_tag

def m1_get_model_win_info(comp_df,winner_tag):

    num_model_wins = comp_df[comp_df[winner_tag]==1].shape[0]
    num_overall_trials = comp_df.shape[0]
    num_ties = comp_df[comp_df[winner_tag]=="tie"].shape[0]
    num_baseline_wins = comp_df[comp_df[winner_tag]==0].shape[0]
    num_trials_without_ties = num_overall_trials - num_ties
    model_wins_div_by_num_trials_without_ties = num_model_wins/float(num_trials_without_ties)

    # model_win_info_dict = {
    # "num_model_wins" : num_model_wins,
    # "num_overall_trials" : num_overall_trials,
    # "num_ties" : num_ties,
    # # "num_baseline_wins":num_baseline_wins,
    # "num_trials_without_ties":num_trials_without_ties
    # "model_wins_div_num_trials_without_ties" : model_wins_div_num_trials_without_ties
    # }

    win_info_list = [
    "num_overall_trials",
    "num_ties",
    "num_trials_without_ties",
    "num_model_wins",
    "model_wins_div_by_num_trials_without_ties"
    ]

    win_info_val_list = [
    num_overall_trials,
    num_ties,
    num_trials_without_ties,
    num_model_wins,
    model_wins_div_by_num_trials_without_ties
    ]

    df = pd.DataFrame(data={"info_piece":win_info_list, "value":win_info_val_list})
    df = df[["info_piece", "value"]]
    print(df)

    return df

def m1_compare_results_to_baseline_v2_save_option(nn_result_df, baseline_result_df, output_dir,base_tag,nn_tag,rename_cols,compare_col,merge_cols, desired_actions, infoIDs,kickout_cols=[]):
    print("\nComparing to baseline...")


    #make dir
    if output_dir != None:
        comparison_output_dir = output_dir + "Baseline-Comparison/"
        create_output_dir(comparison_output_dir)

    for col in kickout_cols:
        if col in list(nn_result_df):
            nn_result_df =nn_result_df.drop(col, axis=1)
        if col in list(baseline_result_df):
            baseline_result_df =baseline_result_df.drop(col, axis=1)

    for col in rename_cols:
        nn_result_df = nn_result_df.rename(columns={col: nn_tag + "_" + col})
        baseline_result_df = baseline_result_df.rename(columns={col: base_tag + "_" + col})

    #will use these results for later
    result_type_list = []
    num_wins_list = []
    num_losses_list = []
    num_trials_list = []
    win_freq_list = []
    num_ties_list = []

    #merge
    # merge_cols = ["platform", "user_status", "informationID", "user_type"]
    result_df = pd.merge(nn_result_df, baseline_result_df, on=merge_cols, how="inner")
    print(result_df)
    nn_compare_col = nn_tag +"_"+ compare_col
    baseline_compare_col = base_tag +"_"+ compare_col
    result_df = result_df.sort_values(nn_compare_col).reset_index(drop=True)
    print(result_df)
    result_df,result_dict = calculate_winner_v2(result_df, nn_compare_col,baseline_compare_col,metric=compare_col)
    print(result_df)

    baseline_score = result_df[baseline_compare_col].sum()
    nn_score = result_df[nn_compare_col].sum()

    baseline_str = "%s: %.2f"%(baseline_compare_col, baseline_score)
    nn_str = "%s: %.2f"%(nn_compare_col, nn_score)
    print(baseline_str)
    print(nn_str)
    if output_dir != None:
        score_fp =output_dir + "NN-vs-Baseline-Scores.txt"
        with open(score_fp, "w") as f:
            f.write(str(baseline_str))
            f.write(str(nn_str))

    #append
    result_type_list.append("all")
    num_wins_list.append(result_dict["num_wins"])
    num_losses_list.append(result_dict["num_losses"])
    num_ties_list.append(result_dict["num_ties"])
    num_trials_list.append(result_dict["total_trials"])
    win_freq_list.append(result_dict["freq"])

    #save it
    if output_dir != None:
        output_fp = comparison_output_dir + "All-NN-vs-Baseline-Results.csv"
        result_df.to_csv(output_fp, index=False)
        print(output_fp)

    sub_dfs = []

    for user_status in user_statuses:
        for user_type in user_types:
            temp= get_infoID_nunique_users_for_user_status_and_user_type(result_df, user_type, user_status, output_dir)
            print(temp)
            temp,result_dict = calculate_winner_v2(temp, nn_compare_col,baseline_compare_col,metric=compare_col)

            #append
            result_type_list.append("%s-%s"%(user_status, user_type))
            num_wins_list.append(result_dict["num_wins"])
            num_losses_list.append(result_dict["num_losses"])
            num_ties_list.append(result_dict["num_ties"])
            num_trials_list.append(result_dict["total_trials"])
            win_freq_list.append(result_dict["freq"])

            if output_dir != None:
                output_fp = comparison_output_dir + "%s-%s-NN-vs-Baseline-Results.csv"%(user_status, user_type)
                temp.to_csv(output_fp, index=False)
                print(output_fp)
            sub_dfs.append(temp)

    #make df
    data={"result_type":result_type_list, "num_nn_wins_for_%s"%compare_col : num_wins_list,
        "num_nn_losses_for_%s"%compare_col : num_losses_list, "num_ties":num_ties_list,
        "num_trials":num_trials_list, "num_wins_as_freq":win_freq_list}
    breakdown_df = pd.DataFrame(data=data)
    col_order = ["result_type", "num_nn_wins_for_%s"%compare_col,
        "num_nn_losses_for_%s"%compare_col, "num_ties","num_trials","num_wins_as_freq"]
    breakdown_df = breakdown_df[col_order]
    print(breakdown_df)

    if output_dir != None:
        output_fp = comparison_output_dir + "Summary-Comparison.csv"
        breakdown_df.to_csv(output_fp, index=False)
        print(output_fp)



    return result_df,breakdown_df,sub_dfs

def get_m1_shifted_baseline_pred_dict_v2_agg_output(infoIDs, desired_actions,pair_train_and_test_array_dict,EXTRACT_OUTPUT_SIZE,ft_to_idx_dict,target_ft ):

    #predict!
    #get num combos
    num_pairs = len(desired_actions) * len(infoIDs)
    print("\n%d pairs"%num_pairs)
    baseline_pred_dict = {}
    i = 0
    for infoID in infoIDs:
        baseline_pred_dict[infoID] = {}
        for action in desired_actions:
            x_test = pair_train_and_test_array_dict[infoID][action]["x_test"]

            target_idx= ft_to_idx_dict[target_ft]
            print("\ntarget_idx: %d"%target_idx)
            y_baseline_pred = x_test[:, -EXTRACT_OUTPUT_SIZE:, target_idx]
            print("y_baseline_pred shape: %s"%str(y_baseline_pred.shape))
            y_baseline_pred = y_baseline_pred.reshape((y_baseline_pred.shape[0], y_baseline_pred.shape[1], 1))
            y_baseline_pred =  agg_y_array(y_baseline_pred)

            print("aggregated y_baseline_pred shape: %s"%str(y_baseline_pred.shape))


            baseline_pred_dict[infoID][action] = y_baseline_pred
            i+=1
            print("Got pair df %d of %d" %(i, num_pairs))

    return baseline_pred_dict

def get_m1_shifted_baseline_pred_dict_v4_x_array_option(infoIDs,flat_ft_to_idx_dict,flat_idx_to_ft_dict,flattened_fts, desired_actions,pair_train_and_test_array_dict,EXTRACT_OUTPUT_SIZE,ft_to_idx_dict,target_ft,x_array_tag ):

    #predict!
    #get num combos
    num_pairs = len(desired_actions) * len(infoIDs)
    print("\n%d pairs"%num_pairs)
    baseline_pred_dict = {}
    i = 0
    for infoID in infoIDs:
        baseline_pred_dict[infoID] = {}
        for action in desired_actions:
            x_test = pair_train_and_test_array_dict[infoID][action][x_array_tag]

            #get the last EXTRACT_OUTPUT_SIZE number of target vals
            mult_target_fts_list = []
            reversed_flattened_fts = reversed(flattened_fts)
            num_targets_found = 0
            for rev_ft in reversed_flattened_fts:
                if target_ft in rev_ft:
                    # print(rev_ft)
                    mult_target_fts_list.append(rev_ft)
                    num_targets_found+=1
                if num_targets_found == EXTRACT_OUTPUT_SIZE:
                    break
            # sys.exit(0)
            y_baseline_pred = []
            for cur_x_test_array in x_test:
                cur_y_target_val = 0
                for cur_target_ft in mult_target_fts_list:
                    idx = flat_ft_to_idx_dict[cur_target_ft]
                    cur_y_target_val+=cur_x_test_array[idx]
                y_baseline_pred.append(cur_y_target_val)
            y_baseline_pred = np.asarray(y_baseline_pred)
            y_baseline_pred = y_baseline_pred.reshape((y_baseline_pred.shape[0], 1, 1))


            # target_idx= ft_to_idx_dict[target_ft]
            # print("\ntarget_idx: %d"%target_idx)
            # y_baseline_pred = x_test[:, -EXTRACT_OUTPUT_SIZE:, target_idx]
            # print("y_baseline_pred shape: %s"%str(y_baseline_pred.shape))
            # y_baseline_pred = y_baseline_pred.reshape((y_baseline_pred.shape[0], y_baseline_pred.shape[1], 1))
            # y_baseline_pred =  agg_y_array(y_baseline_pred)

            print("aggregated y_baseline_pred shape: %s"%str(y_baseline_pred.shape))


            baseline_pred_dict[infoID][action] = y_baseline_pred
            i+=1
            print("Got pair df %d of %d" %(i, num_pairs))

    return baseline_pred_dict

def get_m1_shifted_baseline_pred_dict_v3_x_array_option(infoIDs, desired_actions,pair_train_and_test_array_dict,EXTRACT_OUTPUT_SIZE,ft_to_idx_dict,target_ft,x_array_tag ):

    #predict!
    #get num combos
    num_pairs = len(desired_actions) * len(infoIDs)
    print("\n%d pairs"%num_pairs)
    baseline_pred_dict = {}
    i = 0
    for infoID in infoIDs:
        baseline_pred_dict[infoID] = {}
        for action in desired_actions:
            x_test = pair_train_and_test_array_dict[infoID][action][x_array_tag]

            target_idx= ft_to_idx_dict[target_ft]
            print("\ntarget_idx: %d"%target_idx)
            y_baseline_pred = x_test[:, -EXTRACT_OUTPUT_SIZE:, target_idx]
            print("y_baseline_pred shape: %s"%str(y_baseline_pred.shape))
            y_baseline_pred = y_baseline_pred.reshape((y_baseline_pred.shape[0], y_baseline_pred.shape[1], 1))
            y_baseline_pred =  agg_y_array(y_baseline_pred)

            print("aggregated y_baseline_pred shape: %s"%str(y_baseline_pred.shape))


            baseline_pred_dict[infoID][action] = y_baseline_pred
            i+=1
            print("Got pair df %d of %d" %(i, num_pairs))

    return baseline_pred_dict

def get_m1_shifted_baseline_pred_dict(infoIDs, desired_actions,pair_train_and_test_array_dict,OUTPUT_SIZE,ft_to_idx_dict,target_ft ):

    #predict!
    #get num combos
    num_pairs = len(desired_actions) * len(infoIDs)
    print("\n%d pairs"%num_pairs)
    baseline_pred_dict = {}
    i = 0
    for infoID in infoIDs:
        baseline_pred_dict[infoID] = {}
        for action in desired_actions:
            x_test = pair_train_and_test_array_dict[infoID][action]["x_test"]

            target_idx= ft_to_idx_dict[target_ft]
            print("\ntarget_idx: %d"%target_idx)
            y_baseline_pred = x_test[:, -OUTPUT_SIZE, target_idx]
            print("y_baseline_pred shape: %s"%str(y_baseline_pred.shape))


            baseline_pred_dict[infoID][action] = y_baseline_pred
            i+=1
            print("Got pair df %d of %d" %(i, num_pairs))

    return baseline_pred_dict

def score_pred_results_v2(pair_pred_dict,pair_train_and_test_array_dict,desired_actions, infoIDs,output_dir,pred_tag):

    infoID_result_list = []
    action_result_list = []
    rmse_list = []
    mape_list = []
    actual_volume_list = []
    pred_volume_list = []
    for infoID in infoIDs:
        for action in desired_actions:
            y_pred = pair_pred_dict[infoID][action]
            y_test = pair_train_and_test_array_dict[infoID][action]["y_test"]
            y_pred = y_pred.flatten()
            y_test = y_test.flatten()

            infoID_result_list.append(infoID)
            action_result_list.append(action)

            # for i in range(NUM_TARGET_LOG_NORMS):
            #   y_test = np.expm1(y_test)

            # if RESCALE_TARGET == True:
            #   old_shape = y_test.shape
            #   y_test = y_test.reshape((y_test.shape[0], 1, 1))
            #   y_test = y_test.astype("float32")
            #   y_test =denormalize_single_array(y_test, y_scaler)
            #   y_test = y_test.reshape(old_shape)

            # pred_volume = np.sum(y_pred)
            # actual_volume = np.sum(y_test)
            # actual_volume_list.append(actual_volume)
            # pred_volume_list.append(pred_volume)

            #rmse
            cur_rmse = mean_squared_error(y_test, y_pred, squared=False)
            rmse_list.append(cur_rmse)

            #mape
            cur_mape = mean_absolute_percentage_error(y_test, y_pred)
            mape_list.append(cur_mape)

    #results
    result_df = pd.DataFrame(data={"infoID":infoID_result_list, "action":action_result_list, "rmse":rmse_list, "mape":mape_list})
    result_df = result_df[["infoID", "action", "mape","rmse"]]
    result_df = result_df.sort_values("mape", ascending=True).reset_index(drop=True)
    print(result_df)
    output_fp = output_dir + "%s-results.csv"%pred_tag
    result_df.to_csv(output_fp, index=False)
    print(output_fp)

    # total_pred_volume = result_df["pred_volume"].sum()
    # total_actual_volume = result_df["actual_volume"].sum()
    # print("\ntotal_pred_volume: %.4f"%total_pred_volume)
    # print("total_actual_volume: %4.f" %total_actual_volume)
    # final_rmse = mean_squared_error([total_pred_volume], [total_actual_volume], squared=False)
    # print("\nfinal_rmse: %.4f"%final_rmse)
    # with open(output_dir + "%s-final-rmse.txt"%pred_tag,"w") as f:
    #     f.write(str(final_rmse))
    print("done")
    return result_df

def score_pred_results_v3_with_array_option(pair_pred_dict,pair_train_and_test_array_dict,desired_actions, infoIDs,output_dir,pred_tag,array_tag):

    infoID_result_list = []
    action_result_list = []
    rmse_list = []
    mape_list = []
    actual_volume_list = []
    pred_volume_list = []
    for infoID in infoIDs:
        for action in desired_actions:
            y_pred = pair_pred_dict[infoID][action]
            y_test = pair_train_and_test_array_dict[infoID][action][array_tag]
            y_pred = y_pred.flatten()
            y_test = y_test.flatten()

            infoID_result_list.append(infoID)
            action_result_list.append(action)

            # for i in range(NUM_TARGET_LOG_NORMS):
            #   y_test = np.expm1(y_test)

            # if RESCALE_TARGET == True:
            #   old_shape = y_test.shape
            #   y_test = y_test.reshape((y_test.shape[0], 1, 1))
            #   y_test = y_test.astype("float32")
            #   y_test =denormalize_single_array(y_test, y_scaler)
            #   y_test = y_test.reshape(old_shape)

            # pred_volume = np.sum(y_pred)
            # actual_volume = np.sum(y_test)
            # actual_volume_list.append(actual_volume)
            # pred_volume_list.append(pred_volume)

            #rmse
            try:
                cur_rmse = mean_squared_error(y_test, y_pred, squared=False)
            except ValueError:
                print("\nIn func: score_pred_results_v3_with_array_option")
                print("ValueError: Input contains NaN, infinity or a value too large for dtype('float32').")
                print("\ny_test")
                print(y_test)
                print("\ny_pred")
                print(y_pred)
                sys.exit(0)
            rmse_list.append(cur_rmse)

            #mape
            cur_mape = mean_absolute_percentage_error(y_test, y_pred)
            mape_list.append(cur_mape)

    #results
    result_df = pd.DataFrame(data={"infoID":infoID_result_list, "action":action_result_list, "rmse":rmse_list, "mape":mape_list})
    result_df = result_df[["infoID", "action", "mape","rmse"]]
    result_df = result_df.sort_values("mape", ascending=True).reset_index(drop=True)
    print(result_df)
    output_fp = output_dir + "%s-results.csv"%pred_tag
    result_df.to_csv(output_fp, index=False)
    print(output_fp)

    # total_pred_volume = result_df["pred_volume"].sum()
    # total_actual_volume = result_df["actual_volume"].sum()
    # print("\ntotal_pred_volume: %.4f"%total_pred_volume)
    # print("total_actual_volume: %4.f" %total_actual_volume)
    # final_rmse = mean_squared_error([total_pred_volume], [total_actual_volume], squared=False)
    # print("\nfinal_rmse: %.4f"%final_rmse)
    # with open(output_dir + "%s-final-rmse.txt"%pred_tag,"w") as f:
    #     f.write(str(final_rmse))
    print("done")
    return result_df

def score_pred_results_v3_with_array_option_v4_timestep_dim(pair_pred_dict,pair_train_and_test_array_dict,desired_actions, infoIDs,output_dir,pred_tag,array_tag):

    infoID_result_list = []
    action_result_list = []
    rmse_list = []
    mape_list = []
    actual_volume_list = []
    pred_volume_list = []
    timestep_list = []
    for infoID in infoIDs:
        for action in desired_actions:
            y_pred = pair_pred_dict[infoID][action]
            y_test = pair_train_and_test_array_dict[infoID][action][array_tag]
            y_pred = y_pred.flatten()
            y_test = y_test.flatten()
            num_timesteps = y_test.shape[0]

            for timestep in range(1, num_timesteps+1):
                timestep_list.append(timestep)

                cur_y_pred = y_pred[timestep-1]
                cur_y_test = y_test[timestep-1]

                # print(cur_y_pred)
                # print(cur_y_test)


                infoID_result_list.append(infoID)
                action_result_list.append(action)


                #rmse
                cur_rmse = mean_squared_error([cur_y_test], [cur_y_pred], squared=False)
                rmse_list.append(cur_rmse)

                #mape
                cur_mape = mean_absolute_percentage_error(cur_y_test, cur_y_pred)
                mape_list.append(cur_mape)

    #results
    result_df = pd.DataFrame(data={"infoID":infoID_result_list, "action":action_result_list, "rmse":rmse_list, "mape":mape_list,"timestep":timestep_list})
    result_df = result_df[["infoID", "action", "timestep","mape","rmse"]]
    result_df = result_df.sort_values("mape", ascending=True).reset_index(drop=True)
    print(result_df)
    output_fp = output_dir + "%s-results.csv"%pred_tag
    result_df.to_csv(output_fp, index=False)
    print(output_fp)

    # total_pred_volume = result_df["pred_volume"].sum()
    # total_actual_volume = result_df["actual_volume"].sum()
    # print("\ntotal_pred_volume: %.4f"%total_pred_volume)
    # print("total_actual_volume: %4.f" %total_actual_volume)
    # final_rmse = mean_squared_error([total_pred_volume], [total_actual_volume], squared=False)
    # print("\nfinal_rmse: %.4f"%final_rmse)
    # with open(output_dir + "%s-final-rmse.txt"%pred_tag,"w") as f:
    #     f.write(str(final_rmse))
    print("done")
    return result_df


def score_pred_results(pair_pred_dict,pair_train_and_test_array_dict,desired_actions, infoIDs,output_dir,pred_tag):

    infoID_result_list = []
    action_result_list = []
    rmse_list = []
    mape_list = []
    actual_volume_list = []
    pred_volume_list = []
    for infoID in infoIDs:
        for action in desired_actions:
            y_pred = pair_pred_dict[infoID][action]
            y_test = pair_train_and_test_array_dict[infoID][action]["y_test"]
            y_pred = y_pred.flatten()
            y_test = y_test.flatten()

            infoID_result_list.append(infoID)
            action_result_list.append(action)

            # for i in range(NUM_TARGET_LOG_NORMS):
            #   y_test = np.expm1(y_test)

            # if RESCALE_TARGET == True:
            #   old_shape = y_test.shape
            #   y_test = y_test.reshape((y_test.shape[0], 1, 1))
            #   y_test = y_test.astype("float32")
            #   y_test =denormalize_single_array(y_test, y_scaler)
            #   y_test = y_test.reshape(old_shape)

            pred_volume = np.sum(y_pred)
            actual_volume = np.sum(y_test)
            actual_volume_list.append(actual_volume)
            pred_volume_list.append(pred_volume)

            #rmse
            cur_rmse = mean_squared_error([actual_volume], [pred_volume], squared=False)
            rmse_list.append(cur_rmse)

            #mape
            cur_mape = mean_absolute_percentage_error(actual_volume, pred_volume)
            mape_list.append(cur_mape)

    #results
    result_df = pd.DataFrame(data={"infoID":infoID_result_list, "action":action_result_list, "rmse":rmse_list, "mape":mape_list, "actual_volume":actual_volume_list, "pred_volume":pred_volume_list})
    result_df = result_df[["infoID", "action", "pred_volume","actual_volume","mape","rmse"]]
    result_df = result_df.sort_values("mape", ascending=True).reset_index(drop=True)
    print(result_df)
    output_fp = output_dir + "%s-results.csv"%pred_tag
    result_df.to_csv(output_fp, index=False)
    print(output_fp)

    total_pred_volume = result_df["pred_volume"].sum()
    total_actual_volume = result_df["actual_volume"].sum()
    print("\ntotal_pred_volume: %.4f"%total_pred_volume)
    print("total_actual_volume: %4.f" %total_actual_volume)
    final_rmse = mean_squared_error([total_pred_volume], [total_actual_volume], squared=False)
    print("\nfinal_rmse: %.4f"%final_rmse)
    with open(output_dir + "%s-final-rmse.txt"%pred_tag,"w") as f:
        f.write(str(final_rmse))
    print("done")
    return result_df

def get_infoID_combined_result(result_df,pred_tag,output_dir):
    #get combined result
    infoID_result_df = result_df.copy()
    infoID_result_df["pred_volume"] = infoID_result_df.groupby(["infoID"])["pred_volume"].transform("sum")
    infoID_result_df["actual_volume"] = infoID_result_df.groupby(["infoID"])["actual_volume"].transform("sum")
    infoID_result_df = infoID_result_df.drop("action", axis=1)
    infoID_result_df = infoID_result_df[["infoID","pred_volume", "actual_volume"]].drop_duplicates().reset_index(drop=True)

    #get mapes
    mapes = []
    for pred,actual in zip(list(infoID_result_df["actual_volume"]), list(infoID_result_df["pred_volume"])):
        mapes.append(mean_absolute_percentage_error(pred,actual ))
    infoID_result_df["mape"]=mapes

    #get rmse
    rmse_list = []
    for pred,actual in zip(list(infoID_result_df["actual_volume"]), list(infoID_result_df["pred_volume"])):

        cur_rmse =  mean_squared_error([actual], [pred], squared=False)
        rmse_list.append(cur_rmse)
    infoID_result_df["rmse"]=rmse_list

    #sort
    infoID_result_df = infoID_result_df.sort_values("mape", ascending=True).reset_index(drop=True)
    infoID_result_df = infoID_result_df[["infoID", "pred_volume","actual_volume","mape","rmse"]]
    print(infoID_result_df)

    output_fp = output_dir + "%s-infoID_result_df.csv"%pred_tag
    infoID_result_df.to_csv(output_fp)
    print(output_fp)
    print("done")

    return infoID_result_df

def action_result_df(result_df, output_dir,pred_tag):
    #get combined result
    action_result_df = result_df.copy()
    action_result_df["pred_volume"] = action_result_df.groupby(["action"])["pred_volume"].transform("sum")
    action_result_df["actual_volume"] = action_result_df.groupby(["action"])["actual_volume"].transform("sum")
    action_result_df = action_result_df.drop("infoID", axis=1)
    action_result_df = action_result_df[["action","pred_volume", "actual_volume"]].drop_duplicates().reset_index(drop=True)

    #get mapes
    mapes = []
    for pred,actual in zip(list(action_result_df["actual_volume"]), list(action_result_df["pred_volume"])):
        mapes.append(mean_absolute_percentage_error(pred,actual ))
    action_result_df["mape"]=mapes

    #get rmse
    rmse_list = []
    for pred,actual in zip(list(action_result_df["actual_volume"]), list(action_result_df["pred_volume"])):

        cur_rmse =  mean_squared_error([actual], [pred], squared=False)
        rmse_list.append(cur_rmse)
    action_result_df["rmse"]=rmse_list

    #sort
    action_result_df = action_result_df.sort_values("mape", ascending=True).reset_index(drop=True)
    action_result_df = action_result_df[["action", "pred_volume","actual_volume","mape","rmse"]]
    print(action_result_df)

    output_fp = output_dir + "%s-action_result_df.csv"%pred_tag
    action_result_df.to_csv(output_fp, index=False)
    print(output_fp)

    return action_result_df

def convert_df_to_test_sliding_window_form_diff_x_and_y_fts_no_print(df, INPUT_SIZE, OUTPUT_SIZE,input_features, target_fts,MOD_NUM=10):

    SEQUENCE_SIZE = INPUT_SIZE + OUTPUT_SIZE
    TOTAL_EVENTS = df.shape[0] - INPUT_SIZE
    NUM_LOOPS = int(TOTAL_EVENTS/OUTPUT_SIZE)
    print("NUM_LOOPS: %d" %NUM_LOOPS)

    #get arrays
    #get arrays
    x_data_arrays = df[input_features].values
    y_data_arrays = df[target_fts].values
    print("\ndata_arrays shape:" )
    print(x_data_arrays.shape)
    print(y_data_arrays.shape)
    x_data = []
    y_data = []

    for i in range(NUM_LOOPS):
        start = i *OUTPUT_SIZE
        end = start + SEQUENCE_SIZE
        # print("\nIdx %d: start: %d, end: %d" %(i, start,end))

        x_cur_seq = x_data_arrays[start:end, :]
        y_cur_seq = y_data_arrays[start:end, :]

        x_seq = np.asarray(x_cur_seq[:INPUT_SIZE, :])
        y_seq = np.asarray(y_cur_seq[INPUT_SIZE:INPUT_SIZE+OUTPUT_SIZE, :])
        # if i%MOD_NUM == 0:
        #     # print("cur_seq shape: %s" %str(cur_seq.shape))
        #     print("\nCurrent loop: %d of %d" %((i+1), NUM_LOOPS))
        #     print("x shape: %s" %str(x_seq.shape))
        #     print("y shape: %s" %str(y_seq.shape))
        x_data.append(x_seq)
        y_data.append(y_seq)

    #final data
    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)

    return x_data,y_data

def get_best_module2_model_dir(m2_cands):

    all_model_dirs = []
    for main_dir in m2_cands:

        subdirs = os.listdir(main_dir)
        for s in subdirs:
            s = main_dir + s + "/"
            all_model_dirs.append(s)

    all_bdfs = []
    for model_dir in all_model_dirs:
        baseline_fp = model_dir + "Baseline-Comparison/Summary-Comparison.csv"

        try:
            base_df = pd.read_csv(baseline_fp)
        except FileNotFoundError:
            print("Couldn't find %s"%baseline_fp)
        base_df["model_dir"] = model_dir
        all_bdfs.append(base_df)

    baseline_df = pd.concat(all_bdfs).reset_index(drop=True)
    print("\nbaseline_df")
    print(baseline_df)

    #get best
    temp = baseline_df.copy()
    temp = temp[temp["result_type"]=="all"]
    temp["best_score"] = temp["num_wins_as_freq"].max()
    temp = temp[temp["best_score"]==temp["num_wins_as_freq"]]
    model_dir = temp["model_dir"].unique()[0]
    baseline_df = baseline_df[baseline_df["model_dir"]==model_dir]

    print("\nBest")
    print(baseline_df)





    return baseline_df

def get_best_action_result_df_parent_dir(parent_dirs,actions,metric_of_interest):
    print()
    print("\nparent dirs")
    print(parent_dirs)
    # sys.exit(0)

    # action_to_best_model_dict = {}

    all_model_dirs = []

    for cur_dir in parent_dirs:
        print(cur_dir)
        model_dirs = os.listdir(cur_dir)

        for model_dir in model_dirs:
            if ".txt" in model_dir:
                continue
            model_dir = cur_dir + model_dir + "/"
            print(model_dir)
            all_model_dirs.append(model_dir)

    #get results
    all_action_result_dfs = []
    for model_dir in all_model_dirs:
        action_result_fp = model_dir + "action_result_df.csv"
        print(action_result_fp)

        try:
            action_result_df = pd.read_csv(action_result_fp)
        except FileNotFoundError:
            continue
        # print(action_result_df)
        action_result_df["model_dir"] = model_dir
        all_action_result_dfs.append(action_result_df)

    #get best results
    action_result_df = pd.concat(all_action_result_dfs)

    #get min mapes
    action_result_df["min_%s"%metric_of_interest] = action_result_df.groupby(["action"])[metric_of_interest].transform("min")
    action_result_df = action_result_df[action_result_df[metric_of_interest]==action_result_df["min_%s"%metric_of_interest]].reset_index(drop=True)
    print(action_result_df)
    action_result_df = action_result_df.drop_duplicates(["action"]).reset_index(drop=True)
    action_result_df["pred_volume"] = np.round(action_result_df["pred_volume"], 0)


    for col in list(action_result_df):
        if "Unnamed" in col:
            action_result_df = action_result_df.drop(col, axis=1)

    action_result_df = action_result_df.drop("min_%s"%metric_of_interest, axis=1)

    print(action_result_df)



    # action_to_best_model_dict = convert_df_2_cols_to_dict(action_result_df, "action", "model_dir")
    return action_result_df

def make_module1_dir(action_result_df, model_type_tag,module_output_dir):

    #get model dirs
    model_dirs = action_result_df["model_dir"].unique()

    #get model dir to action dict
    model_dir_to_action_list_dict = {}
    action_to_model_dir_dict = {}
    for model_dir in model_dirs:
        temp = action_result_df[action_result_df["model_dir"]==model_dir]
        temp_actions = list(temp["action"])
        model_dir_to_action_list_dict[model_dir] = temp_actions
        print()
        print(model_dir)
        print(temp_actions)

        for action in temp_actions:
            action_to_model_dir_dict[action] = model_dir

    print("\naction_to_model_dir_dict")
    print(action_to_model_dir_dict)

    main_module1_subdir =module_output_dir +  "MODULE1-%s/"%model_type_tag
    create_output_dir(main_module1_subdir)

    action_to_model_actions_tag_dir_dict = {}
    print("\nGetting best models...")
    for model_dir,action_list in model_dir_to_action_list_dict.items():
        model_actions_tag = ""
        for a in action_list:
            model_actions_tag = model_actions_tag + a + "-"
        model_actions_tag=model_actions_tag[:-1] + "/"

        src = model_dir
        dest = main_module1_subdir + model_actions_tag + "/"

        # try:
        #   destination = shutil.copytree(src, dest)
        #   print("Destination path:", destination)
        # except FileExistsError:
        #   continue

        # destination = shutil.copytree(src, dest)
        copytree(src, dest, symlinks = False, ignore = None)
        print("Got %s"%dest)


        for action in action_list:
            action_to_model_actions_tag_dir_dict[action] = model_actions_tag

    print("\naction_to_model_actions_tag_dir_dict")
    print(action_to_model_actions_tag_dir_dict)

    #save action result
    action_res_combined_fp = main_module1_subdir + "COMBINED-action_result_df.csv"
    action_result_df.to_csv(action_res_combined_fp, index=False)

    action_to_model_json_fp = main_module1_subdir + "action-to-model-config.json"
    save_as_json(action_to_model_actions_tag_dir_dict,action_to_model_json_fp)

    return


def make_module2_dir(best_m2_baseline_df,module2_dir):

    create_output_dir(module2_dir)

    #get model dir
    model_dir = best_m2_baseline_df["model_dir"].unique()[0]

    src = model_dir
    dest = module2_dir

    # destination = shutil.copytree(src, dest)
    copytree(src, dest, symlinks = False, ignore = None)
    print("Got %s"%dest)

    orig_fp = module2_dir + "orig_model_dir_info.txt"
    with open(orig_fp, "w") as f:
        f.write(str(model_dir))

    best_fp = module2_dir + "best-baseline-breakdown.csv"
    best_m2_baseline_df.to_csv(best_fp, index=False)

    return

def best_module2_model_dir_v2_using_dict(model_dir_to_comp_df_dict):
    print("\nGetting best df...")

    all_bdfs = []
    for model_dir, breakdown_df in model_dir_to_comp_df_dict.items():
        breakdown_df["model_dir"] = model_dir
        all_bdfs.append(breakdown_df)

    baseline_df = pd.concat(all_bdfs).reset_index(drop=True)
    print("\nbaseline_df")
    print(baseline_df)

    #get best
    temp = baseline_df.copy()
    temp = temp[temp["result_type"]=="all"]
    temp["best_score"] = temp["num_wins_as_freq"].max()
    temp = temp[temp["best_score"]==temp["num_wins_as_freq"]]
    model_dir = temp["model_dir"].unique()[0]
    baseline_df = baseline_df[baseline_df["model_dir"]==model_dir]

    print("\nBest")
    print(baseline_df)

    return baseline_df

def get_cand_dir_to_comparison_df_dict(baseline_result_df,model_dir_to_result_df_dict,baseline_quartet_pred_dict,all_model_dirs,quartet_train_and_test_array_dict,platforms, user_statuses, infoIDs, user_types):

    base_tag = "baseline"
    nn_tag = "nn"
    rename_cols = ["nrmse_over_time", "rmse_over_time"]
    compare_col = "rmse_over_time"
    merge_cols = ["platform", "user_status", "informationID", "user_type"]

    print("\nGetting each comp df...")
    model_dir_to_comp_df_dict = {}
    for model_dir in all_model_dirs:
        nn_result_df = model_dir_to_result_df_dict[model_dir]
        print(nn_result_df)

        #get comp dfs
        comp_df,breakdown_df,sub_dfs = compare_results_to_baseline_v2_save_option(nn_result_df, baseline_result_df, None,base_tag,nn_tag,rename_cols,compare_col,merge_cols, user_statuses, user_types)
        model_dir_to_comp_df_dict[model_dir] = breakdown_df
        print("\nbreakdown_df")
        print(breakdown_df)

    return model_dir_to_comp_df_dict

def get_cand_dir_to_result_df_dict(all_model_dirs,quartet_train_and_test_array_dict,platforms, user_statuses, infoIDs, user_types):

    model_dir_to_result_df_dict = {}
    for model_dir in all_model_dirs:
        quartet_pred_df = load_data_from_pickle(model_dir + "quartet_pred_dict")
        cur_result_tuple = get_quartet_result_df_v2_save_option(quartet_pred_df,quartet_train_and_test_array_dict,platforms, user_statuses, infoIDs, user_types,None,DEBUG_PRINT=True)
        # sys.exit(0)
        model_result_df = cur_result_tuple[0]
        model_dir_to_result_df_dict[model_dir] = model_result_df.copy()

    return model_dir_to_result_df_dict

def get_quartet_train_and_test_array_dict_from_dir(m2_dir):

    subdirs = os.listdir(m2_dir)
    dict_dir =m2_dir + subdirs[0] + "/"
    quartet_train_and_test_array_dict = load_data_from_pickle(dict_dir + "quartet_train_and_test_array_dict")


    return quartet_train_and_test_array_dict

def get_m2_cand_subdirs(m2_cands):

    all_model_dirs = []
    for main_dir in m2_cands:

        subdirs = os.listdir(main_dir)
        for s in subdirs:
            s = main_dir + s + "/"
            all_model_dirs.append(s)

    return all_model_dirs

def get_baseline_dates(test_start, test_end, GRAN="d"):
    test_start = pd.to_datetime(test_start, utc=True)
    test_end = pd.to_datetime(test_end, utc=True)
    dates = pd.date_range(test_start, test_end, freq=GRAN)
    num_dates = len(dates)
    base_start =test_start - pd.to_timedelta(num_dates, unit='d')
    base_end = test_end - pd.to_timedelta(num_dates, unit='d')
    print("base_start, base_end")
    print(base_start, base_end)
    return base_start, base_end

def copytree(src, dst, symlinks = False, ignore = None):
  if not os.path.exists(dst):
    os.makedirs(dst)
    shutil.copystat(src, dst)
  lst = os.listdir(src)
  if ignore:
    excl = ignore(src, lst)
    lst = [x for x in lst if x not in excl]
  for item in lst:
    s = os.path.join(src, item)
    d = os.path.join(dst, item)
    if symlinks and os.path.islink(s):
      if os.path.lexists(d):
        os.remove(d)
      os.symlink(os.readlink(s), d)
      try:
        st = os.lstat(s)
        mode = stat.S_IMODE(st.st_mode)
        os.lchmod(d, mode)
      except:
        pass # lchmod not available
    elif os.path.isdir(s):
      copytree(s, d, symlinks, ignore)
    else:
      shutil.copy2(s, d)


def get_cp4_pnnl_cols():
    return ["nodeTime", "nodeUserID", "nodeID", "parentID", "rootID", "actionType", "platform", "informationID"]

def f4_get_module2_training_user_fts(main_output_dir, start,end,DEBUG,platforms,main_df, GRANS):

    main_output_dir = main_output_dir + "f4-M2-User-Fts/"
    create_output_dir(main_output_dir)

        #debug stuff
    if DEBUG == True:
        GRANS = ["D"]
        # fp = "/data/Fmubang/cp4-cascade-model-5-6/CP4-User-Stuff/debug-file-01-01-2019-to-01-03-2019 23:59:59.csv"
        start = "01-01-2019"
        end = "01-03-2019 23:59:59"

    # #load data
    # print("\nLoading data...")
    # main_df = pd.read_csv(fp)

    #adjust records
    main_df = main_df[main_df["platform"] != "reddit"]

    #get infoIDs
    infoIDs = get_46_cp4_infoIDs()
    print(infoIDs)

    #kick out 1 infoID
    main_df = main_df[main_df["informationID"].isin(infoIDs)].reset_index(drop=True)

    #get parents
    print("\nGetting parents...")
    main_df = get_parentUserID_col(main_df)

    #fix user names
    main_df["nodeUserID"] = main_df["nodeUserID"] +"_" + main_df["informationID"] + "_" + main_df["actionType"] + "_" + main_df["platform"]
    main_df["parentUserID"] = main_df["parentUserID"] +"_" + main_df["informationID"]+ "_" + main_df["actionType"]+ "_" + main_df["platform"]

    print(main_df[["nodeUserID", "parentUserID"]])
    # sys.exit(0)

    #config dates
    main_df = config_df_by_dates(main_df,start,end,"nodeTime")
    print(main_df)

    #info id dict
    hyp_dict = hyphenate_infoID_dict(infoIDs)
    print(hyp_dict)

    #get each feature config
    user_status_categories = ["new", "old"]

    #GET INFO ID 1HOT
    infoID_1hot_dict = get_1hot_vectors(infoIDs)

    for GRAN in GRANS:
        for platform in platforms:
            df = main_df.copy()
            df["nodeTime"] = df["nodeTime"].dt.floor(GRAN)
            df = df[df["platform"]==platform]

            #mark first user appearances
            df = mark_user_first_appearances(df, GRAN, mark_parents=True)

            #make sure counts all add up
            num_all_actions = df.shape[0]
            num_new_actions = df[df["is_new_child_user"]==1].shape[0]
            num_old_actions = df[df["is_new_child_user"]==0].shape[0]

            status_to_action_count_dict={"new":num_new_actions,"old" : num_old_actions,"all" : num_all_actions}
            print("\nstatus_to_action_count_dict")
            print(status_to_action_count_dict)

            #get total nunique users
            temp = df[["nodeTime", "nodeUserID"]].copy().drop_duplicates().reset_index(drop=True)
            temp["nunique_child_users"] = temp.groupby(["nodeTime"])["nodeUserID"].transform("count")
            temp["nunique_child_users"] = temp["nunique_child_users"].astype("float32")
            temp = temp[["nodeTime", "nunique_child_users"]].drop_duplicates().reset_index(drop=True)
            print(temp)
            print(temp["nunique_child_users"])
            total_nunique_child_users_over_time = temp["nunique_child_users"].sum()
            print(total_nunique_child_users_over_time)

            #get total nunique parent users
            temp = df[["nodeTime", "parentUserID"]].copy().drop_duplicates().reset_index(drop=True)
            temp["nunique_parent_users"] = temp.groupby(["nodeTime"])["parentUserID"].transform("count")
            temp = temp[["nodeTime","nunique_parent_users"]].drop_duplicates().reset_index(drop=True)
            temp["nunique_parent_users"] = temp["nunique_parent_users"].astype("float32")
            print(temp)
            total_nunique_parent_users_over_time = temp["nunique_parent_users"].sum()

            #counts for later
            gt_child_user_counts = int(total_nunique_child_users_over_time)
            gt_parent_user_counts = int(total_nunique_parent_users_over_time)
            print("\ngt_child_user_counts: %d" %gt_child_user_counts)
            print("gt_parent_user_counts: %d" %gt_parent_user_counts)

            #for later checking
            my_nunique_parent_counts = 0
            my_nunique_child_counts = 0

            for user_status in user_status_categories:
                my_full_counts = 0
                for infoID in infoIDs:
                    hyp_infoID = hyp_dict[infoID]
                    print("\nWorking on:")
                    print(platform)
                    print(user_status)
                    print(infoID)

                    #make dir
                    output_dir =main_output_dir + "GRAN-%s-Dates-%s-to-%s-DEBUG-%s/%s/%s/%s/"%(GRAN, start,end,DEBUG,platform,user_status,hyp_infoID)
                    create_output_dir(output_dir)

                    #filter by infoID
                    temp = df.copy()
                    # temp = temp[temp["platform"]==platform]
                    temp = temp[temp["informationID"]==infoID].reset_index(drop=True)

                    #get counts
                    print("\nGetting child,parent, action counts...")
                    temp =  count_children_parents_and_actions(temp,user_status ,start,end,GRAN,kickout_other_cols=True)
                    temp["informationID"] = infoID

                    #get 1hot df
                    #get the 1hot
                    infoID_1hot_vector = infoID_1hot_dict[infoID]
                    infoID_1hot_vector = np.asarray(infoID_1hot_vector).reshape((1, len(infoIDs)))
                    print(infoID)
                    print(infoID_1hot_vector)
                    hot_df = pd.DataFrame(data=infoID_1hot_vector, columns=infoIDs)
                    hot_df["informationID"] = infoID
                    print(hot_df)
                    temp = pd.merge(hot_df,temp, on="informationID", how="inner")
                    print(temp)


                    print(temp)
                    my_full_counts+=temp["num_actions"].sum()

                    #============================= SPLIT =============================
                    #children
                    child_df = temp[["nodeTime", "nunique_child_users", "num_actions"] + infoIDs].copy()
                    child_df = child_df.rename(columns={"nunique_child_users":"nunique_users"})
                    child_df["is_child"] = 1
                    if user_status == "new":
                        child_df["is_new"] = 1
                    else:
                        child_df["is_new"] = 0

                    #parents
                    parent_df = temp[["nodeTime", "nunique_parent_users"] + infoIDs].copy()
                    parent_df = parent_df.rename(columns={"nunique_parent_users":"nunique_users"})
                    parent_df["is_child"] = 0
                    if user_status == "new":
                        parent_df["is_new"] = 1
                    else:
                        parent_df["is_new"] = 0

                    #platforms
                    if platform=="twitter":
                        child_df["is_twitter"] = 1
                        parent_df["is_twitter"] = 1
                    else:
                        child_df["is_twitter"] = 0
                        parent_df["is_twitter"] = 0

                    print("parent and child dfs")
                    print(parent_df)
                    print(child_df)

                    # sys.exit(0)

                    #save
                    child_fp = output_dir + "child-user-data.csv"
                    child_df.to_csv(child_fp, index=False)
                    parent_fp = output_dir + "parent-user-data.csv"
                    parent_df.to_csv(parent_fp, index=False)
                    print(child_fp)
                    print(parent_fp)

                    print(temp)
                    if user_status != "all":
                        my_nunique_parent_counts=my_nunique_parent_counts + parent_df["nunique_users"].sum()
                        my_nunique_child_counts=my_nunique_child_counts + child_df["nunique_users"].sum()

                print("\nVerifying final counts...")
                target_full_counts = status_to_action_count_dict[user_status]
                print("\ntarget_full_counts: %d" %target_full_counts)
                print("my_full_counts: %d"%my_full_counts)
                if my_full_counts != target_full_counts:
                    print("Error: my_full_counts != target_full_counts")
                    sys.exit(0)
                else:
                    print("my_full_counts == target_full_counts")
                    print("Counts are ok!")


            print("\nVerifying nunique user counts...")
            print("\ngt_child_user_counts: %d" %gt_child_user_counts)
            print("my_nunique_child_counts: %d" %my_nunique_child_counts)
            if my_nunique_child_counts != gt_child_user_counts:
                print("Error! my_nunique_child_counts != gt_child_user_counts")
                sys.exit(0)
            else:
                print("my_nunique_child_counts == gt_child_user_counts")
                print("Counts are ok!")

            print("\ngt_parent_user_counts: %d" %gt_parent_user_counts)
            print("my_nunique_parent_counts: %d" %my_nunique_parent_counts)
            if my_nunique_parent_counts != gt_parent_user_counts:
                print("Error! my_nunique_parent_counts != gt_parent_user_counts")
                sys.exit(0)
            else:
                print("my_nunique_parent_counts == gt_parent_user_counts")
                print("Counts are ok!")

    return

def f4_get_module2_training_user_fts_v2_choose_infoIDs( infoIDs,main_output_dir, start,end,DEBUG,platforms,main_df, GRANS):

    main_output_dir = main_output_dir + "f4-M2-User-Fts/"
    create_output_dir(main_output_dir)

        #debug stuff
    # if DEBUG == True:
    #     GRANS = ["D"]
    #     # fp = "/data/Fmubang/cp4-cascade-model-5-6/CP4-User-Stuff/debug-file-01-01-2019-to-01-03-2019 23:59:59.csv"
    #     start = "01-01-2019"
    #     end = "01-03-2019 23:59:59"

    # #load data
    # print("\nLoading data...")
    # main_df = pd.read_csv(fp)

    #adjust records
    main_df = main_df[main_df["platform"] != "reddit"]

    #get infoIDs
    # infoIDs = get_46_cp4_infoIDs()
    print(infoIDs)

    #kick out 1 infoID
    main_df = main_df[main_df["informationID"].isin(infoIDs)].reset_index(drop=True)

    #get parents
    print("\nGetting parents...")
    main_df = get_parentUserID_col(main_df)

    #fix user names
    main_df["nodeUserID"] = main_df["nodeUserID"] +"_" + main_df["informationID"] + "_" + main_df["actionType"] + "_" + main_df["platform"]
    main_df["parentUserID"] = main_df["parentUserID"] +"_" + main_df["informationID"]+ "_" + main_df["actionType"]+ "_" + main_df["platform"]

    print(main_df[["nodeUserID", "parentUserID"]])
    # sys.exit(0)

    #config dates
    main_df = config_df_by_dates(main_df,start,end,"nodeTime")
    print(main_df)

    #info id dict
    hyp_dict = hyphenate_infoID_dict(infoIDs)
    print(hyp_dict)

    #get each feature config
    user_status_categories = ["new", "old"]

    #GET INFO ID 1HOT
    infoID_1hot_dict = get_1hot_vectors(infoIDs)

    for GRAN in GRANS:
        for platform in platforms:
            df = main_df.copy()
            df["nodeTime"] = df["nodeTime"].dt.floor(GRAN)
            df = df[df["platform"]==platform]

            #mark first user appearances
            df = mark_user_first_appearances(df, GRAN, mark_parents=True)

            #make sure counts all add up
            num_all_actions = df.shape[0]
            num_new_actions = df[df["is_new_child_user"]==1].shape[0]
            num_old_actions = df[df["is_new_child_user"]==0].shape[0]

            status_to_action_count_dict={"new":num_new_actions,"old" : num_old_actions,"all" : num_all_actions}
            print("\nstatus_to_action_count_dict")
            print(status_to_action_count_dict)

            #get total nunique users
            temp = df[["nodeTime", "nodeUserID"]].copy().drop_duplicates().reset_index(drop=True)
            temp["nunique_child_users"] = temp.groupby(["nodeTime"])["nodeUserID"].transform("count")
            temp["nunique_child_users"] = temp["nunique_child_users"].astype("float32")
            temp = temp[["nodeTime", "nunique_child_users"]].drop_duplicates().reset_index(drop=True)
            print(temp)
            print(temp["nunique_child_users"])
            total_nunique_child_users_over_time = temp["nunique_child_users"].sum()
            print(total_nunique_child_users_over_time)

            #get total nunique parent users
            temp = df[["nodeTime", "parentUserID"]].copy().drop_duplicates().reset_index(drop=True)
            temp["nunique_parent_users"] = temp.groupby(["nodeTime"])["parentUserID"].transform("count")
            temp = temp[["nodeTime","nunique_parent_users"]].drop_duplicates().reset_index(drop=True)
            temp["nunique_parent_users"] = temp["nunique_parent_users"].astype("float32")
            print(temp)
            total_nunique_parent_users_over_time = temp["nunique_parent_users"].sum()

            #counts for later
            gt_child_user_counts = int(total_nunique_child_users_over_time)
            gt_parent_user_counts = int(total_nunique_parent_users_over_time)
            print("\ngt_child_user_counts: %d" %gt_child_user_counts)
            print("gt_parent_user_counts: %d" %gt_parent_user_counts)

            #for later checking
            my_nunique_parent_counts = 0
            my_nunique_child_counts = 0

            for user_status in user_status_categories:
                my_full_counts = 0
                for infoID in infoIDs:
                    hyp_infoID = hyp_dict[infoID]
                    print("\nWorking on:")
                    print(platform)
                    print(user_status)
                    print(infoID)

                    #make dir
                    output_dir =main_output_dir + "GRAN-%s-Dates-%s-to-%s-DEBUG-%s/%s/%s/%s/"%(GRAN, start,end,DEBUG,platform,user_status,hyp_infoID)
                    create_output_dir(output_dir)

                    #filter by infoID
                    temp = df.copy()
                    # temp = temp[temp["platform"]==platform]
                    temp = temp[temp["informationID"]==infoID].reset_index(drop=True)

                    #get counts
                    print("\nGetting child,parent, action counts...")
                    temp =  count_children_parents_and_actions(temp,user_status ,start,end,GRAN,kickout_other_cols=True)
                    temp["informationID"] = infoID

                    #get 1hot df
                    #get the 1hot
                    infoID_1hot_vector = infoID_1hot_dict[infoID]
                    infoID_1hot_vector = np.asarray(infoID_1hot_vector).reshape((1, len(infoIDs)))
                    print(infoID)
                    print(infoID_1hot_vector)
                    hot_df = pd.DataFrame(data=infoID_1hot_vector, columns=infoIDs)
                    hot_df["informationID"] = infoID
                    print(hot_df)
                    temp = pd.merge(hot_df,temp, on="informationID", how="inner")
                    print(temp)


                    print(temp)
                    my_full_counts+=temp["num_actions"].sum()

                    #============================= SPLIT =============================
                    #children
                    child_df = temp[["nodeTime", "nunique_child_users", "num_actions"] + infoIDs].copy()
                    child_df = child_df.rename(columns={"nunique_child_users":"nunique_users"})
                    child_df["is_child"] = 1
                    if user_status == "new":
                        child_df["is_new"] = 1
                    else:
                        child_df["is_new"] = 0

                    #parents
                    parent_df = temp[["nodeTime", "nunique_parent_users"] + infoIDs].copy()
                    parent_df = parent_df.rename(columns={"nunique_parent_users":"nunique_users"})
                    parent_df["is_child"] = 0
                    if user_status == "new":
                        parent_df["is_new"] = 1
                    else:
                        parent_df["is_new"] = 0

                    #platforms
                    if platform=="twitter":
                        child_df["is_twitter"] = 1
                        parent_df["is_twitter"] = 1
                    else:
                        child_df["is_twitter"] = 0
                        parent_df["is_twitter"] = 0

                    print("parent and child dfs")
                    print(parent_df)
                    print(child_df)

                    # sys.exit(0)

                    #save
                    child_fp = output_dir + "child-user-data.csv"
                    child_df.to_csv(child_fp, index=False)
                    parent_fp = output_dir + "parent-user-data.csv"
                    parent_df.to_csv(parent_fp, index=False)
                    print(child_fp)
                    print(parent_fp)

                    print(temp)
                    if user_status != "all":
                        my_nunique_parent_counts=my_nunique_parent_counts + parent_df["nunique_users"].sum()
                        my_nunique_child_counts=my_nunique_child_counts + child_df["nunique_users"].sum()

                print("\nVerifying final counts...")
                target_full_counts = status_to_action_count_dict[user_status]
                print("\ntarget_full_counts: %d" %target_full_counts)
                print("my_full_counts: %d"%my_full_counts)
                if my_full_counts != target_full_counts:
                    print("Error: my_full_counts != target_full_counts")
                    sys.exit(0)
                else:
                    print("my_full_counts == target_full_counts")
                    print("Counts are ok!")


            print("\nVerifying nunique user counts...")
            print("\ngt_child_user_counts: %d" %gt_child_user_counts)
            print("my_nunique_child_counts: %d" %my_nunique_child_counts)
            if my_nunique_child_counts != gt_child_user_counts:
                print("Error! my_nunique_child_counts != gt_child_user_counts")
                sys.exit(0)
            else:
                print("my_nunique_child_counts == gt_child_user_counts")
                print("Counts are ok!")

            print("\ngt_parent_user_counts: %d" %gt_parent_user_counts)
            print("my_nunique_parent_counts: %d" %my_nunique_parent_counts)
            if my_nunique_parent_counts != gt_parent_user_counts:
                print("Error! my_nunique_parent_counts != gt_parent_user_counts")
                sys.exit(0)
            else:
                print("my_nunique_parent_counts == gt_parent_user_counts")
                print("Counts are ok!")

    return

def f3_get_full_nn_features_with_gdelt_and_reddit_v2_choose_infoIDs(action_order ,infoIDs,gdelt_fp_template, reddit_fp_template, GRANS,main_output_dir, start, end,gran_to_input_dir_dict):

    for GRAN in GRANS:

        main_input_dir = gran_to_input_dir_dict[GRAN]
        print(main_input_dir)

        gdelt_fp = gdelt_fp_template.replace("<GRAN>",GRAN)
        reddit_fp = reddit_fp_template.replace("<GRAN>",GRAN)
        print(gdelt_fp)
        print(reddit_fp)

        output_dir = main_output_dir + "f3-MODULE1-NN-FEATURES/"

        output_tag = "gran-%s-start-%s-end-%s"%(GRAN, start, end)
        output_dir = output_dir + output_tag + "/"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        print(output_dir)
        #================#==============================#==============================#=======================

        #get reddit
        reddit_df = pd.read_csv(reddit_fp)
        reddit_fts = get_reddit_actions()
        reddit_df = reddit_df[["nodeTime"] + reddit_fts]
        print(reddit_df)

        #GET INFO IDS
        # infoIDs = get_46_cp4_infoIDs()

        #GET INFO ID 1HOT
        infoID_1hot_dict = get_1hot_vectors(infoIDs)

        #get files
        infoID_fps = os.listdir(main_input_dir)
        print(infoID_fps)

        #get conv dict
        hyp_dict = hyphenate_infoID_dict(infoIDs)
        print(hyp_dict)

        reg_infoID_dict = get_reverse_dict(hyp_dict)
        print(hyp_dict)


        #get actiontype order
        # action_order = get_twitter_actions() + get_youtube_actions()
        print(action_order)
        action_to_1hot_dict = create_1hot_vector_dict(action_order)

        #get gdelt
        gdelt_df = pd.read_csv(gdelt_fp)
        # print("\ngdelt_df before flaot64")
        # print(gdelt_df)
        # gcols = list(gdelt_df)
        # # for col in gcols:
        # #     if col!= "nodeTime":
        # #         gdelt_df[col] = gdelt_df[col].astype("float64")
        # # print("\ngdelt_df after flaot64")
        # # print(gdelt_df)

        # def convert_infoID_df_to_model_format(infoID_df):
        #   return

        infoID_to_action_fp_dict = {}

        for infoID_fp in infoID_fps:

            print("\n")
            print(infoID_fp)
            fp = main_input_dir + infoID_fp
            infoID_df = pd.read_csv(fp)
            print(infoID_df)

            #get infoID
            infoID_tag = infoID_fp.replace(".csv", "")
            infoID = reg_infoID_dict[infoID_tag]
            print(infoID)
            infoID_to_action_fp_dict[infoID] = {}
            infoID_df["informationID"] = infoID

            # #get spec glove row
            # cur_glove_df = glove_df[glove_df["informationID"]==infoID]
            # print(cur_glove_df)

            # infoID_df = pd.merge(cur_glove_df,infoID_df, on="informationID", how="inner")
            # print(infoID_df)

            #get the 1hot
            infoID_1hot_vector = infoID_1hot_dict[infoID]
            infoID_1hot_vector = np.asarray(infoID_1hot_vector).reshape((1, len(infoIDs)))
            print(infoID)
            print(infoID_1hot_vector)
            hot_df = pd.DataFrame(data=infoID_1hot_vector, columns=infoIDs)
            hot_df["informationID"] = infoID
            print(hot_df)
            infoID_df = pd.merge(hot_df,infoID_df, on="informationID", how="inner")
            print(infoID_df)
            # sys.exit(0)


            #make infoID dir
            infoID_output_dir = output_dir + infoID_tag + "/"
            if not os.path.exists(infoID_output_dir):
                os.makedirs(infoID_output_dir)

            action_to_df_dict = {}
            for action in action_order:

                print(action)
                cur_drop_cols = [a for a in action_order if a != action]
                print(cur_drop_cols)
                temp = infoID_df.copy()
                # temp = temp.drop(cur_drop_cols, axis=1)
                for col in cur_drop_cols:
                    if col in list(temp):
                        temp = temp.drop(col, axis=1)
                temp = temp.rename(columns={action:"target_action_value"})

                for action2 in action_order:
                    if action2 == action:
                        temp[action2]= 1
                    else:
                        temp[action2] = 0
                print(temp)

                # temp = temp.merge(gdelt_df, on="nodeTime", how="inner")
                temp = temp.merge(gdelt_df, on="nodeTime", how="outer")


                temp["target_action"] = action
                # temp = temp.merge(reddit_df, on="nodeTime", how="inner")
                temp = temp.merge(reddit_df, on="nodeTime", how="outer")

                temp = temp.fillna(0.0)
                temp = config_df_by_dates(temp, start,end,"nodeTime")
                temp = temp.sort_values("nodeTime").reset_index(drop=True)

                print(temp)

                print(temp["nodeTime"])

                date_test_list = pd.date_range(start, end, freq=GRAN)
                num_test_dates = len(date_test_list)
                print("\nnum_test_dates start end end")
                print(date_test_list[0])
                print(date_test_list[-1])
                print("\nDf start and end")
                print(temp["nodeTime"].iloc[0])
                print(temp["nodeTime"].iloc[-1])
                if num_test_dates != temp.shape[0]:
                    print(num_test_dates)
                    print(date_test_list)
                    print("\nError num_test_dates != temp.shape[0]")
                    sys.exit(0)
                print("counts are ok!")

                output_fp = infoID_output_dir + action + ".csv"
                temp.to_csv(output_fp, index=False)
                print(output_fp)
                infoID_to_action_fp_dict[infoID][action] = output_fp

        #save fts
        # full_fts = list(temp)
        model_fts = list(temp)
        remove_list = ["nodeTime", "informationID","target_action"]
        for ft in remove_list:
            model_fts.remove(ft)

        with open(output_dir + "model-fts.txt", "w") as f:
            for ft in model_fts:
                print(ft)
                f.write(ft + "\n")

        with open(output_dir + "infoID_to_action_fp_dict", "wb") as handle:
            pickle.dump(infoID_to_action_fp_dict, handle)
        print(output_dir + "infoID_to_action_fp_dict")
        print("done")



    return

def f3_get_full_nn_features_with_gdelt_and_reddit(gdelt_fp_template, reddit_fp_template, GRANS,main_output_dir, start, end,gran_to_input_dir_dict):

    for GRAN in GRANS:

        main_input_dir = gran_to_input_dir_dict[GRAN]
        print(main_input_dir)

        gdelt_fp = gdelt_fp_template.replace("<GRAN>",GRAN)
        reddit_fp = reddit_fp_template.replace("<GRAN>",GRAN)
        print(gdelt_fp)
        print(reddit_fp)

        output_dir = main_output_dir + "f3-MODULE1-NN-FEATURES/"

        output_tag = "gran-%s-start-%s-end-%s"%(GRAN, start, end)
        output_dir = output_dir + output_tag + "/"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        print(output_dir)
        #================#==============================#==============================#=======================

        #get reddit
        reddit_df = pd.read_csv(reddit_fp)
        reddit_fts = get_reddit_actions()
        reddit_df = reddit_df[["nodeTime"] + reddit_fts]
        print(reddit_df)

        #GET INFO IDS
        infoIDs = get_46_cp4_infoIDs()

        #GET INFO ID 1HOT
        infoID_1hot_dict = get_1hot_vectors(infoIDs)

        #get files
        infoID_fps = os.listdir(main_input_dir)
        print(infoID_fps)

        #get conv dict
        hyp_dict = hyphenate_infoID_dict(infoIDs)
        print(hyp_dict)

        reg_infoID_dict = get_reverse_dict(hyp_dict)
        print(hyp_dict)


        #get actiontype order
        action_order = get_twitter_actions() + get_youtube_actions()
        print(action_order)
        action_to_1hot_dict = create_1hot_vector_dict(action_order)

        #get gdelt
        gdelt_df = pd.read_csv(gdelt_fp)
        # print("\ngdelt_df before flaot64")
        # print(gdelt_df)
        # gcols = list(gdelt_df)
        # # for col in gcols:
        # #     if col!= "nodeTime":
        # #         gdelt_df[col] = gdelt_df[col].astype("float64")
        # # print("\ngdelt_df after flaot64")
        # # print(gdelt_df)

        # def convert_infoID_df_to_model_format(infoID_df):
        #   return

        infoID_to_action_fp_dict = {}

        for infoID_fp in infoID_fps:

            print("\n")
            print(infoID_fp)
            fp = main_input_dir + infoID_fp
            infoID_df = pd.read_csv(fp)
            print(infoID_df)

            #get infoID
            infoID_tag = infoID_fp.replace(".csv", "")
            infoID = reg_infoID_dict[infoID_tag]
            print(infoID)
            infoID_to_action_fp_dict[infoID] = {}
            infoID_df["informationID"] = infoID

            # #get spec glove row
            # cur_glove_df = glove_df[glove_df["informationID"]==infoID]
            # print(cur_glove_df)

            # infoID_df = pd.merge(cur_glove_df,infoID_df, on="informationID", how="inner")
            # print(infoID_df)

            #get the 1hot
            infoID_1hot_vector = infoID_1hot_dict[infoID]
            infoID_1hot_vector = np.asarray(infoID_1hot_vector).reshape((1, len(infoIDs)))
            print(infoID)
            print(infoID_1hot_vector)
            hot_df = pd.DataFrame(data=infoID_1hot_vector, columns=infoIDs)
            hot_df["informationID"] = infoID
            print(hot_df)
            infoID_df = pd.merge(hot_df,infoID_df, on="informationID", how="inner")
            print(infoID_df)
            # sys.exit(0)


            #make infoID dir
            infoID_output_dir = output_dir + infoID_tag + "/"
            if not os.path.exists(infoID_output_dir):
                os.makedirs(infoID_output_dir)

            action_to_df_dict = {}
            for action in action_order:

                print(action)
                cur_drop_cols = [a for a in action_order if a != action]
                print(cur_drop_cols)
                temp = infoID_df.copy()
                # temp = temp.drop(cur_drop_cols, axis=1)
                for col in cur_drop_cols:
                    if col in list(temp):
                        temp = temp.drop(col, axis=1)
                temp = temp.rename(columns={action:"target_action_value"})

                for action2 in action_order:
                    if action2 == action:
                        temp[action2]= 1
                    else:
                        temp[action2] = 0
                print(temp)

                # temp = temp.merge(gdelt_df, on="nodeTime", how="inner")
                temp = temp.merge(gdelt_df, on="nodeTime", how="outer")


                temp["target_action"] = action
                # temp = temp.merge(reddit_df, on="nodeTime", how="inner")
                temp = temp.merge(reddit_df, on="nodeTime", how="outer")

                temp = temp.fillna(0.0)
                temp = config_df_by_dates(temp, start,end,"nodeTime")
                temp = temp.sort_values("nodeTime").reset_index(drop=True)

                print(temp)

                print(temp["nodeTime"])

                date_test_list = pd.date_range(start, end, freq=GRAN)
                num_test_dates = len(date_test_list)
                print("\nnum_test_dates start end end")
                print(date_test_list[0])
                print(date_test_list[-1])
                print("\nDf start and end")
                print(temp["nodeTime"].iloc[0])
                print(temp["nodeTime"].iloc[-1])
                if num_test_dates != temp.shape[0]:
                    print(num_test_dates)
                    print(date_test_list)
                    print("\nError num_test_dates != temp.shape[0]")
                    sys.exit(0)
                print("counts are ok!")

                output_fp = infoID_output_dir + action + ".csv"
                temp.to_csv(output_fp, index=False)
                print(output_fp)
                infoID_to_action_fp_dict[infoID][action] = output_fp

        #save fts
        # full_fts = list(temp)
        model_fts = list(temp)
        remove_list = ["nodeTime", "informationID","target_action"]
        for ft in remove_list:
            model_fts.remove(ft)

        with open(output_dir + "model-fts.txt", "w") as f:
            for ft in model_fts:
                print(ft)
                f.write(ft + "\n")

        with open(output_dir + "infoID_to_action_fp_dict", "wb") as handle:
            pickle.dump(infoID_to_action_fp_dict, handle)
        print(output_dir + "infoID_to_action_fp_dict")
        print("done")



    return

def f2_get_actions_by_infoID(DEBUG, main_df, infoIDs,main_output_dir,start, end, GRANS,actions):

    gran_to_output_dir_dict = {}

    for GRAN in GRANS:

        hyp_dict = hyphenate_infoID_dict(infoIDs)
        print(hyp_dict)

        #get dates
        date_df = create_blank_date_df(start,end,GRAN)

        df = main_df.copy()
        df["nodeTime"] = df["nodeTime"].dt.floor(GRAN)

        #make output dir
        # main_output_dir = "/data/Fmubang/cp4-lstm/data/CP4-Actions-by-infoID/"
        output_tag = "f2-actions-by-infoID/GRAN-%s-Dates-%s-to-%s-DEBUG-%s"%(GRAN, start,end,DEBUG)
        output_dir = main_output_dir + output_tag + "/"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        print(output_dir)
        gran_to_output_dir_dict[GRAN] = output_dir

        #all_dfs = []
        for infoID in infoIDs:
            temp = df[df["informationID"].isin([infoID])].reset_index(drop=True)
            temp = get_action_counts_by_gran(temp, GRAN,actions,kickout_other_cols=True)
            temp = temp.merge(date_df, on="nodeTime", how="outer")
            temp = temp.fillna(0)
            temp = temp.drop_duplicates().reset_index(drop=True)
            temp = temp.sort_values("nodeTime").reset_index(drop=True)

            infoID = hyp_dict[infoID]
            print(infoID)


            output_fp = output_dir + infoID + ".csv"
            temp.to_csv(output_fp, index=False)
            print(output_fp)

            print(temp)
    return gran_to_output_dir_dict

def get_training_data_from_jsons_v2(df, train_start, train_end,output_fp):
    df["actionType"] = df["platform"] +"_"+ df["actionType"]
    print("\nSorting...")
    df["nodeTime"] = pd.to_datetime(df["nodeTime"], utc=True)
    df = config_df_by_dates(df, train_start, train_end, "nodeTime")
    df = df.sort_values("nodeTime").reset_index(drop=True)
    print(df)
    print(df["nodeTime"])

    df.to_csv(output_fp, index=False)
    print(output_fp)
    print("done")

    return df

def get_training_data_from_jsons(json_fps, reddit_fp, main_output_dir, train_start, train_end,output_fp):

    dfs = []
    reddit_df = pd.read_csv(reddit_fp)
    dfs.append(reddit_df)

    for json_fp in json_fps:
        print()
        print(json_fp)
        df = convert_json_to_df(json_fp, False)
        df["actionType"] = df["platform"] +"_"+ df["actionType"]
        dfs.append(df)
        print(df)

    print("\nCombining...")
    df = pd.concat(dfs)
    print(df)

    print(df["actionType"])

    print("\nSorting...")
    df["nodeTime"] = pd.to_datetime(df["nodeTime"], utc=True)
    df = config_df_by_dates(df, train_start, train_end, "nodeTime")
    df = df.sort_values("nodeTime").reset_index(drop=True)
    print(df)
    print(df["nodeTime"])

    df.to_csv(output_fp, index=False)
    print(output_fp)
    print("done")

    return df

def get_module1_action_to_pred_dir_dict_from_json(module1_dir,json_fp = "action-to-model-config.json"):

    print("\nGetting model dict from:")
    json_fp = module1_dir + json_fp
    print(json_fp)

    with open(json_fp) as json_file:
        action_to_pred_dir_dict = json.load(json_file)
        print(action_to_pred_dir_dict)

    for action,action_dir in action_to_pred_dir_dict.items():
        action_to_pred_dir_dict[action] = module1_dir + action_dir

    return action_to_pred_dir_dict

def combine_test_arrays(pair_test_array_dict, GET_GROUND_TRUTH, infoIDs, cp4_actions):
  x_test_arrays = []

  if GET_GROUND_TRUTH == True:
      y_test_arrays = []

  for infoID in infoIDs:
      for action in cp4_actions:
          x_test = pair_test_array_dict[infoID][action]["x_test"]
          x_test_arrays.append(np.copy(x_test))

          if GET_GROUND_TRUTH == True:
              y_test = pair_test_array_dict[infoID][action]["y_test"]
              y_test_arrays.append(np.copy(y_test))

  x_test = np.concatenate(x_test_arrays, axis=0)
  print("\nx test shape")
  print(x_test.shape)

  if GET_GROUND_TRUTH == True:
      y_test = np.concatenate(y_test_arrays, axis=0)
      print("\ny test shape")
      print(y_test.shape)
  else:
      y_test = None

  return x_test, y_test

def get_action_to_platform_dict():

    action_to_platform_dict = {
    "twitter_tweet" : "twitter",
    "twitter_retweet" : "twitter",
    "twitter_reply" : "twitter",
    "twitter_quote" : "twitter",
    "youtube_video" : "youtube",
    "youtube_comment" : "youtube"

    }

    return action_to_platform_dict

def myconverter(o):
    if isinstance(o, datetime):
        return o.__str__()

def save_sim_df_as_json_with_header_v2(sim_df,identifier, json_output_fp,scenario="1",simulation_period="january25-january31"):
    MOD_NUM = 5000
    #save it
    #SAVE TO JSON
    FINAL_SIMULATION =sim_df.to_dict("records")
    num_records = len(FINAL_SIMULATION)
    f = open(json_output_fp, "w")
    print("Saving to %s" %json_output_fp)

    #first put the file heading for the docker!
    #header = {"identifier": identifier, "team": "usf", "scenario": scenario}
    header = {"team": "usf", "model_identifier": identifier, "simulation_period": simulation_period}
    f.write(str(json.dumps(header)) + "\n")

    #now write the simulation
    for i,record in enumerate(FINAL_SIMULATION):
        if i%MOD_NUM == 0:
            print("Writing %d of %d to json" %((i), num_records))
        f.write(str(json.dumps(record, default=myconverter)) + "\n")

    print("Saved")
    print(json_output_fp)

def save_sim_df_as_json_with_header(sim_df, json_output_fp,scenario="1",):
    MOD_NUM = 5000
    #save it
    #SAVE TO JSON
    FINAL_SIMULATION =sim_df.to_dict("records")
    num_records = len(FINAL_SIMULATION)
    f = open(json_output_fp, "w")
    print("Saving to %s" %json_output_fp)

    #first put the file heading for the docker!
    header = {"identifier": json_output_fp, "team": "usf", "scenario": scenario}
    f.write(str(json.dumps(header)) + "\n")

    #now write the simulation
    for i,record in enumerate(FINAL_SIMULATION):
        if i%MOD_NUM == 0:
            print("Writing %d of %d to json" %((i), num_records))
        f.write(str(json.dumps(record, default=myconverter)) + "\n")

    print("Saved")
    print(json_output_fp)

def save_sim_df_as_json(sim_df, json_output_fp):
    MOD_NUM = 5000
    #save it
    #SAVE TO JSON
    FINAL_SIMULATION =sim_df.to_dict("records")
    num_records = len(FINAL_SIMULATION)
    f = open(json_output_fp, "w")
    print("Saving to %s" %json_output_fp)

    #first put the file heading for the docker!
    header = {"identifier": json_output_fp, "team": "usf", "scenario": "2"}
    f.write(str(json.dumps(header)) + "\n")

    #now write the simulation
    for i,record in enumerate(FINAL_SIMULATION):
        if i%MOD_NUM == 0:
            print("Writing %d of %d to json" %((i), num_records))
        f.write(str(json.dumps(record, default=myconverter)) + "\n")

    print("Saved")
    print(json_output_fp)

def get_cp4_challenge_18_infoIDs():

    desired_infoIDs=[
    "arrests",
    "arrests/opposition",
    "guaido/legitimate",
    "international/aid",
    "international/aid_rejected",
    "international/respect_sovereignty",
    "maduro/cuba_support",
    "maduro/dictator",
    "maduro/legitimate",
    "maduro/narco",
    "military",
    "military/desertions",
    "other/anti_socialism",
    "other/censorship_outage",
    "other/chavez",
    "other/chavez/anti",
    "protests",
    "violence"
    ]

    return desired_infoIDs



def resolve_simple_child_action_conflicts(clean_pred_df,test_dates, GRAN,infoIDs,platform_to_action_dict,platforms,user_statuses,user_types,children_action_conflict_option,cp4_actions):

    print("\nResolving user-action conflicts...")
    children_action_conflict_options = ["upsample_actions_to_children","downsample_children_to_actions", "avg"]
    print(children_action_conflict_options)
    #get new dfs
    new_dfs = []

    #fix nodetime
    clean_pred_df["nodeTime"] = pd.to_datetime(clean_pred_df["nodeTime"], utc=True)

    #get val cols
    val_cols = list(clean_pred_df)
    new_cols = []
    for col in val_cols:
        if col not in ["platform","informationID","nodeTime"]:
            new_cols.append(col)
    val_cols = new_cols
    print("\nval_cols")
    print(val_cols)

    child_cols = ["old_child_users","new_child_users"]

    #conflict tracker
    conflict_tracking_dict = {}
    # conflict_types = ["(total_actions > 0) and (total_child_users ==0)", "total_child_users > total_actions"]
    # conflict_types = ["zero_action_conflict", "more_users_than_actions_conflict"]
    # for c in conflict_types:
    #   conflict_tracking_dict[c] = 0

    #use pandas to check conflicts

    freq_action_cols = []
    for action in cp4_actions:
        freq_col = "freq_%s"%action
        freq_action_cols.append(freq_col)
        clean_pred_df[freq_col] = clean_pred_df[action]/clean_pred_df["total_actions"].astype("float32")

    freq_user_status_cols = []
    for user_status in user_statuses:
        user_col = "%s_child_users"%user_status
        freq_col = "freq_" + user_col
        freq_user_status_cols.append(freq_col)
        clean_pred_df[freq_col] = clean_pred_df[user_col]/clean_pred_df["total_child_users"].astype("float32")

    freq_cols = list(freq_user_status_cols + freq_action_cols)

    #new df
    print(clean_pred_df)

    #put pseudo nodeID
    clean_pred_df["pseudo_nodeID"] = [(i+1) for i in range(clean_pred_df.shape[0])]

    #maybe delete? July 9 2020
    clean_pred_df = clean_pred_df.fillna(0)
    print(clean_pred_df)

    # def mark_conflicts(clean_pred_df):

    #   #mark the conflicts
    #   print("\nGetting zero action conflict...")
    #   total_child_users_list = list(clean_pred_df["total_child_users"])
    #   total_actions_list = list(clean_pred_df["total_actions"])
    #   clean_pred_df["zero_action_conflict"] = [1 if ((total_actions>0) and (total_child_users ==0)) else 0 for (total_actions, total_child_users) in zip(total_actions_list, total_child_users_list)]
    #   print("\nzero_action_conflict")
    #   print(clean_pred_df["zero_action_conflict"])

    #   #mark the conflicts
    #   print("\nGetting more users than actiosn conflict...")
    #   clean_pred_df["more_users_conflict"] = [1 if (total_child_users > total_actions) else 0 for (total_actions, total_child_users) in zip(total_actions_list, total_child_users_list)]
    #   print("\nmore_users_conflict")
    #   print(clean_pred_df["more_users_conflict"])

    #   conflict_cols = ["more_users_conflict", "zero_action_conflict"]

    #   return clean_pred_df,conflict_cols

    clean_pred_df,conflict_cols = mark_conflicts(clean_pred_df)

    #conflict dict
    conflict_tracking_dict["zero_action_conflict"] = clean_pred_df["zero_action_conflict"].sum()
    conflict_tracking_dict["more_users_conflict"] = clean_pred_df["more_users_conflict"].sum()
    total_records = clean_pred_df.shape[0]
    print("\ntotal_records: %d"%total_records)
    for c,count in conflict_tracking_dict.items():
        freq = count/float(total_records)
        print("%s count: %d of %d; freq: %.4f"%(c, count, total_records,freq))

    #resolve conflicts
    zero_out_cols = ["total_actions","total_child_users"] + list(cp4_actions + child_cols + freq_cols)
    for col in zero_out_cols:
        clean_pred_df.loc[clean_pred_df["zero_action_conflict"] == 1, col] = 0

    #verify
    temp = clean_pred_df[clean_pred_df["zero_action_conflict"]==1].reset_index(drop=True)
    print("\nzero_action_conflict df")
    temp = temp[zero_out_cols]
    print(temp)
    temp_counts = 0
    for col in zero_out_cols:
        temp_counts+=temp[col].sum()
    print("\ntemp:")
    print(temp)
    if temp_counts != 0:
        print("\nError! temp should be 0.")
        sys.exit(0)
    else:
        print("\nTemp count is ok! Continuing")

    #more users conflict
    children_action_conflict_options = ["upsample_actions_to_children","downsample_children_to_actions", "avg"]

    #first get the nodeIDs that should not be modified!
    unmodified_records = clean_pred_df[clean_pred_df["more_users_conflict"]==0].reset_index(drop=True)

    #get the records to modify
    records_to_modify = clean_pred_df[clean_pred_df["more_users_conflict"]==1].reset_index(drop=True)

    #resolve conflict
    if children_action_conflict_option == "upsample_actions_to_children":
        records_to_modify["total_actions"] = records_to_modify["total_child_users"].copy()
    if children_action_conflict_option == "downsample_children_to_actions":
        records_to_modify["total_child_users"] = records_to_modify["total_actions"].copy()
    if children_action_conflict_option == "avg":
        avg_series = (records_to_modify["total_child_users"] + records_to_modify["total_actions"])/2.0
        avg_series = np.round(avg_series, 0).astype("int32")
        records_to_modify["total_child_users"] = avg_series
        records_to_modify["total_actions"] = avg_series
        print("\navg_series")
        print(avg_series)

    #fix the counts so they match!
    for action in cp4_actions:
        freq_col = "freq_%s"%action
        records_to_modify[action] = records_to_modify[freq_col] * records_to_modify["total_actions"]
        records_to_modify[action] = np.round(records_to_modify[action], 0)

    freq_user_status_cols = []
    for user_status in user_statuses:
        user_col = "%s_child_users"%user_status
        freq_col = "freq_" + user_col
        records_to_modify[user_col] = records_to_modify[freq_col] * records_to_modify["total_child_users"]
        records_to_modify[user_col] = np.round(records_to_modify[user_col], 0)


    #check for rounding error
    total_action_inc_sum = records_to_modify[cp4_actions].sum(axis=1)

    total_child_user_inc_sum = records_to_modify[child_cols].sum(axis=1)
    records_to_modify["total_action_inc_sum"] = total_action_inc_sum
    records_to_modify["total_child_user_inc_sum"] = total_child_user_inc_sum

    #fix rounding error
    records_to_modify["rounding_error"] = total_action_inc_sum - total_child_user_inc_sum
    print(records_to_modify[["total_action_inc_sum","total_child_user_inc_sum", "rounding_error"]].reset_index(drop=True))

    records_to_modify_with_round_error = records_to_modify[records_to_modify["rounding_error"] != 0].reset_index(drop=True)
    records_to_modify_without_round_error = records_to_modify[records_to_modify["rounding_error"] == 0].reset_index(drop=True)
    print("\nrecords_to_modify_with_round_error")
    print(records_to_modify_with_round_error["rounding_error"])
    print("\nrecords_to_modify_without_round_error")
    print(records_to_modify_without_round_error["rounding_error"])

    #if error is 1, that means we have more actions than users, which is ok
    #if error is -1, we have more users than actions which is not ok
    pos_rounding_error_records = records_to_modify_with_round_error[records_to_modify_with_round_error["rounding_error"] > 0].reset_index(drop=True)
    neg_rounding_error_records = records_to_modify_with_round_error[records_to_modify_with_round_error["rounding_error"] < 0].reset_index(drop=True)
    print("\npos_rounding_error_records")
    print(pos_rounding_error_records["rounding_error"])
    print("\nneg_rounding_error_records")
    print(neg_rounding_error_records["rounding_error"])

    #split by youtube and twitter
    twitter_neg_rounding_error_records = neg_rounding_error_records[neg_rounding_error_records["platform"]=="twitter"].reset_index(drop=True)
    youtube_neg_rounding_error_records = neg_rounding_error_records[neg_rounding_error_records["platform"]=="youtube"].reset_index(drop=True)
    print("\ntwitter_neg_rounding_error_records")
    print(twitter_neg_rounding_error_records["rounding_error"])
    print("\nyoutube_neg_rounding_error_records")
    print(youtube_neg_rounding_error_records["rounding_error"])
    twitter_neg_rounding_error_records["twitter_retweet"] = twitter_neg_rounding_error_records["twitter_retweet"] + twitter_neg_rounding_error_records["rounding_error"].abs()
    youtube_neg_rounding_error_records["youtube_comment"] = youtube_neg_rounding_error_records["youtube_comment"] + youtube_neg_rounding_error_records["rounding_error"].abs()
    twitter_neg_rounding_error_records["total_action_inc_sum"] = twitter_neg_rounding_error_records["total_action_inc_sum"] + twitter_neg_rounding_error_records["rounding_error"].abs()
    youtube_neg_rounding_error_records["total_action_inc_sum"] = youtube_neg_rounding_error_records["total_action_inc_sum"] + youtube_neg_rounding_error_records["rounding_error"].abs()

    #set new totals
    twitter_neg_rounding_error_records["total_actions"] = twitter_neg_rounding_error_records["total_action_inc_sum"]
    youtube_neg_rounding_error_records["total_actions"] = youtube_neg_rounding_error_records["total_action_inc_sum"]
    final_records= pd.concat([unmodified_records, youtube_neg_rounding_error_records, twitter_neg_rounding_error_records,pos_rounding_error_records,records_to_modify_without_round_error])
    print("\nfinal_records")
    print(final_records)

    final_records = final_records.sort_values("nodeTime").reset_index(drop=True)
    final_records["total_child_users"] = final_records[child_cols].sum(axis=1)
    final_records["total_actions"] = final_records[cp4_actions].sum(axis=1)

    # #make sure columns add up
    # user_col_sum_series = final_records[child_cols].sum(axis=1)
    # action_col_sum_series = final_records[cp4_actions].sum(axis=1)
    # print("\nuser_col_sum_series")
    # print(user_col_sum_series)
    # print("\naction_col_sum_series")
    # print(action_col_sum_series)
    # print()
    # print("\ntotal_child_users series")
    # print(final_records["total_child_users"])
    # print("\ntotal_actions series")
    # print(final_records["total_actions"])

    # if user_col_sum_series.equals(final_records["total_child_users"]) == True:
    #   print("\nUser series are equal!")
    # else:
    #   print("\nError! User series not equal!")
    #   sys.exit(0)

    # if action_col_sum_series.equals(final_records["total_actions"]) == True:
    #   print("\nAction series are equal!")
    # else:
    #   print("\nError! Action series not equal!")
    #   sys.exit(0)

    #make sure there are no conflicts
    final_records,conflict_cols = mark_conflicts(final_records)
    print("\nBefore bug fix")
    temp = final_records[final_records["zero_action_conflict"]==1]
    print(temp)
    for col in zero_out_cols:
        final_records.loc[final_records["zero_action_conflict"] == 1, col] = 0
    final_records,conflict_cols = mark_conflicts(final_records)
    print("\nAfter bug fix")
    temp = final_records[final_records["zero_action_conflict"]==1]
    print(temp)

    for col in conflict_cols:
        print(col)
        conflict_sum = final_records[col].sum()
        print("conflict sum: %d" %conflict_sum)
        if conflict_sum > 0:

            print("Error! Conflict sum is more than 0!")
            sys.exit(0)
        else:
            print("No conflicts! Continuing...")


    return final_records

def resolve_simple_child_action_conflicts_v3_problem_df_hack(clean_pred_df,test_dates, GRAN,infoIDs,platform_to_action_dict,platforms,user_statuses,user_types,children_action_conflict_option,cp4_actions):

    print("\nResolving user-action conflicts...")
    children_action_conflict_options = ["upsample_actions_to_children","downsample_children_to_actions", "avg"]
    print(children_action_conflict_options)
    #get new dfs
    new_dfs = []

    #fix nodetime
    clean_pred_df["nodeTime"] = pd.to_datetime(clean_pred_df["nodeTime"], utc=True)

    #get val cols
    val_cols = list(clean_pred_df)
    new_cols = []
    for col in val_cols:
        if col not in ["platform","informationID","nodeTime"]:
            new_cols.append(col)
    val_cols = new_cols
    print("\nval_cols")
    print(val_cols)

    child_cols = ["old_child_users","new_child_users"]

    #conflict tracker
    conflict_tracking_dict = {}
    # conflict_types = ["(total_actions > 0) and (total_child_users ==0)", "total_child_users > total_actions"]
    # conflict_types = ["zero_action_conflict", "more_users_than_actions_conflict"]
    # for c in conflict_types:
    #   conflict_tracking_dict[c] = 0

    #use pandas to check conflicts

    freq_action_cols = []
    for action in cp4_actions:
        freq_col = "freq_%s"%action
        freq_action_cols.append(freq_col)
        clean_pred_df[freq_col] = clean_pred_df[action]/clean_pred_df["total_actions"].astype("float32")

    freq_user_status_cols = []
    for user_status in user_statuses:
        user_col = "%s_child_users"%user_status
        freq_col = "freq_" + user_col
        freq_user_status_cols.append(freq_col)
        clean_pred_df[freq_col] = clean_pred_df[user_col]/clean_pred_df["total_child_users"].astype("float32")

    freq_cols = list(freq_user_status_cols + freq_action_cols)

    #new df
    print(clean_pred_df)

    #put pseudo nodeID
    clean_pred_df["pseudo_nodeID"] = [(i+1) for i in range(clean_pred_df.shape[0])]
    clean_pred_df = clean_pred_df.fillna(0)
    print(clean_pred_df)

    # def mark_conflicts(clean_pred_df):

    #   #mark the conflicts
    #   print("\nGetting zero action conflict...")
    #   total_child_users_list = list(clean_pred_df["total_child_users"])
    #   total_actions_list = list(clean_pred_df["total_actions"])
    #   clean_pred_df["zero_action_conflict"] = [1 if ((total_actions>0) and (total_child_users ==0)) else 0 for (total_actions, total_child_users) in zip(total_actions_list, total_child_users_list)]
    #   print("\nzero_action_conflict")
    #   print(clean_pred_df["zero_action_conflict"])

    #   #mark the conflicts
    #   print("\nGetting more users than actiosn conflict...")
    #   clean_pred_df["more_users_conflict"] = [1 if (total_child_users > total_actions) else 0 for (total_actions, total_child_users) in zip(total_actions_list, total_child_users_list)]
    #   print("\nmore_users_conflict")
    #   print(clean_pred_df["more_users_conflict"])

    #   conflict_cols = ["more_users_conflict", "zero_action_conflict"]

    #   return clean_pred_df,conflict_cols

    clean_pred_df,conflict_cols = mark_conflicts(clean_pred_df)

    #conflict dict
    conflict_tracking_dict["zero_action_conflict"] = clean_pred_df["zero_action_conflict"].sum()
    conflict_tracking_dict["more_users_conflict"] = clean_pred_df["more_users_conflict"].sum()
    total_records = clean_pred_df.shape[0]
    print("\ntotal_records: %d"%total_records)
    for c,count in conflict_tracking_dict.items():
        freq = count/float(total_records)
        print("%s count: %d of %d; freq: %.4f"%(c, count, total_records,freq))

    #resolve conflicts
    zero_out_cols = ["total_actions","total_child_users"] + list(cp4_actions + child_cols + freq_cols)
    for col in zero_out_cols:
        clean_pred_df.loc[clean_pred_df["zero_action_conflict"] == 1, col] = 0

    #verify
    temp = clean_pred_df[clean_pred_df["zero_action_conflict"]==1].reset_index(drop=True)
    print("\nzero_action_conflict df")
    temp = temp[zero_out_cols]
    print(temp)
    temp_counts = 0
    for col in zero_out_cols:
        temp_counts+=temp[col].sum()
    print("\ntemp:")
    print(temp)
    if temp_counts != 0:
        print("\nError! temp should be 0.")
        sys.exit(0)
    else:
        print("\nTemp count is ok! Continuing")

    #more users conflict
    children_action_conflict_options = ["upsample_actions_to_children","downsample_children_to_actions", "avg"]

    #first get the nodeIDs that should not be modified!
    unmodified_records = clean_pred_df[clean_pred_df["more_users_conflict"]==0].reset_index(drop=True)

    #get the records to modify
    records_to_modify = clean_pred_df[clean_pred_df["more_users_conflict"]==1].reset_index(drop=True)

    #resolve conflict
    if children_action_conflict_option == "upsample_actions_to_children":
        records_to_modify["total_actions"] = records_to_modify["total_child_users"].copy()
    if children_action_conflict_option == "downsample_children_to_actions":
        records_to_modify["total_child_users"] = records_to_modify["total_actions"].copy()
    if children_action_conflict_option == "avg":
        avg_series = (records_to_modify["total_child_users"] + records_to_modify["total_actions"])/2.0
        avg_series = np.round(avg_series, 0).astype("int32")
        records_to_modify["total_child_users"] = avg_series
        records_to_modify["total_actions"] = avg_series
        print("\navg_series")
        print(avg_series)

    #fix the counts so they match!
    for action in cp4_actions:
        freq_col = "freq_%s"%action
        records_to_modify[action] = records_to_modify[freq_col] * records_to_modify["total_actions"]
        records_to_modify[action] = np.round(records_to_modify[action], 0)

    freq_user_status_cols = []
    for user_status in user_statuses:
        user_col = "%s_child_users"%user_status
        freq_col = "freq_" + user_col
        records_to_modify[user_col] = records_to_modify[freq_col] * records_to_modify["total_child_users"]
        records_to_modify[user_col] = np.round(records_to_modify[user_col], 0)


    #check for rounding error
    total_action_inc_sum = records_to_modify[cp4_actions].sum(axis=1)

    total_child_user_inc_sum = records_to_modify[child_cols].sum(axis=1)
    records_to_modify["total_action_inc_sum"] = total_action_inc_sum
    records_to_modify["total_child_user_inc_sum"] = total_child_user_inc_sum

    #fix rounding error
    records_to_modify["rounding_error"] = total_action_inc_sum - total_child_user_inc_sum
    print(records_to_modify[["total_action_inc_sum","total_child_user_inc_sum", "rounding_error"]].reset_index(drop=True))

    records_to_modify_with_round_error = records_to_modify[records_to_modify["rounding_error"] != 0].reset_index(drop=True)
    records_to_modify_without_round_error = records_to_modify[records_to_modify["rounding_error"] == 0].reset_index(drop=True)
    print("\nrecords_to_modify_with_round_error")
    print(records_to_modify_with_round_error["rounding_error"])
    print("\nrecords_to_modify_without_round_error")
    print(records_to_modify_without_round_error["rounding_error"])

    #if error is 1, that means we have more actions than users, which is ok
    #if error is -1, we have more users than actions which is not ok
    pos_rounding_error_records = records_to_modify_with_round_error[records_to_modify_with_round_error["rounding_error"] > 0].reset_index(drop=True)
    neg_rounding_error_records = records_to_modify_with_round_error[records_to_modify_with_round_error["rounding_error"] < 0].reset_index(drop=True)
    print("\npos_rounding_error_records")
    print(pos_rounding_error_records["rounding_error"])
    print("\nneg_rounding_error_records")
    print(neg_rounding_error_records["rounding_error"])

    #split by youtube and twitter
    twitter_neg_rounding_error_records = neg_rounding_error_records[neg_rounding_error_records["platform"]=="twitter"].reset_index(drop=True)
    youtube_neg_rounding_error_records = neg_rounding_error_records[neg_rounding_error_records["platform"]=="youtube"].reset_index(drop=True)
    print("\ntwitter_neg_rounding_error_records")
    print(twitter_neg_rounding_error_records["rounding_error"])
    print("\nyoutube_neg_rounding_error_records")
    print(youtube_neg_rounding_error_records["rounding_error"])
    twitter_neg_rounding_error_records["twitter_retweet"] = twitter_neg_rounding_error_records["twitter_retweet"] + twitter_neg_rounding_error_records["rounding_error"].abs()
    youtube_neg_rounding_error_records["youtube_comment"] = youtube_neg_rounding_error_records["youtube_comment"] + youtube_neg_rounding_error_records["rounding_error"].abs()
    twitter_neg_rounding_error_records["total_action_inc_sum"] = twitter_neg_rounding_error_records["total_action_inc_sum"] + twitter_neg_rounding_error_records["rounding_error"].abs()
    youtube_neg_rounding_error_records["total_action_inc_sum"] = youtube_neg_rounding_error_records["total_action_inc_sum"] + youtube_neg_rounding_error_records["rounding_error"].abs()

    #set new totals
    twitter_neg_rounding_error_records["total_actions"] = twitter_neg_rounding_error_records["total_action_inc_sum"]
    youtube_neg_rounding_error_records["total_actions"] = youtube_neg_rounding_error_records["total_action_inc_sum"]
    final_records= pd.concat([unmodified_records, youtube_neg_rounding_error_records, twitter_neg_rounding_error_records,pos_rounding_error_records,records_to_modify_without_round_error])
    print("\nfinal_records")
    print(final_records)

    final_records = final_records.sort_values("nodeTime").reset_index(drop=True)
    final_records["total_child_users"] = final_records[child_cols].sum(axis=1)
    final_records["total_actions"] = final_records[cp4_actions].sum(axis=1)

    # #make sure columns add up
    # user_col_sum_series = final_records[child_cols].sum(axis=1)
    # action_col_sum_series = final_records[cp4_actions].sum(axis=1)
    # print("\nuser_col_sum_series")
    # print(user_col_sum_series)
    # print("\naction_col_sum_series")
    # print(action_col_sum_series)
    # print()
    # print("\ntotal_child_users series")
    # print(final_records["total_child_users"])
    # print("\ntotal_actions series")
    # print(final_records["total_actions"])

    # if user_col_sum_series.equals(final_records["total_child_users"]) == True:
    #   print("\nUser series are equal!")
    # else:
    #   print("\nError! User series not equal!")
    #   sys.exit(0)

    # if action_col_sum_series.equals(final_records["total_actions"]) == True:
    #   print("\nAction series are equal!")
    # else:
    #   print("\nError! Action series not equal!")
    #   sys.exit(0)

    #make sure there are no conflicts
    final_records,conflict_cols = mark_conflicts(final_records)
    print("\nBefore bug fix")
    temp = final_records[final_records["zero_action_conflict"]==1]
    print(temp)
    for col in zero_out_cols:
        final_records.loc[final_records["zero_action_conflict"] == 1, col] = 0
    final_records,conflict_cols = mark_conflicts(final_records)
    print("\nAfter bug fix")
    temp = final_records[final_records["zero_action_conflict"]==1]
    print(temp)

    for col in conflict_cols:
        print(col)
        conflict_sum = final_records[col].sum()
        print("conflict sum: %d" %conflict_sum)
        if conflict_sum > 0:
            problem_df = final_records[final_records[col]!=0]

            print("\nproblem_df")
            print(problem_df)
            print("Error! Conflict sum is more than 0!")

            final_records = final_records[final_records[col]==0]
            # problem_df[]

            #get the platform
            twitter_problem_df = problem_df[problem_df["platform"]=="twitter"]
            youtube_problem_df = problem_df[problem_df["platform"]=="youtube"]

            twitter_num_cons = twitter_problem_df["more_users_conflict"].sum()
            youtube_num_cons = youtube_problem_df["more_users_conflict"].sum()
            print("\ntwitter_num_cons: %d" %twitter_num_cons)
            print("\nyoutube_num_cons: %d" %youtube_num_cons)

            for tc in range(twitter_num_cons):
                twitter_problem_df["twitter_retweet"] = twitter_problem_df["twitter_retweet"] + 1
                twitter_problem_df["total_actions"] = twitter_problem_df["total_actions"] + 1

            for yc in range(youtube_num_cons):
                youtube_problem_df["youtube_comment"] = youtube_problem_df["youtube_comment"] + 1
                youtube_problem_df["total_actions"] = youtube_problem_df["total_actions"] + 1

            final_records = pd.concat([final_records, twitter_problem_df, youtube_problem_df])
            final_records = final_records.sort_values("nodeTime").reset_index(drop=True)

            print("\nfinal_records after conflict fix")
            print(final_records)


            #keep increasing # of actions until conflict is solved
            # for c in range(conflict_sum):



            # sys.exit(0)
        else:
            print("No conflicts! Continuing...")


    return final_records

def resolve_simple_child_action_conflicts_v2_fillna_fix(clean_pred_df,test_dates, GRAN,infoIDs,platform_to_action_dict,platforms,user_statuses,user_types,children_action_conflict_option,cp4_actions):

    print("\nResolving user-action conflicts...")
    children_action_conflict_options = ["upsample_actions_to_children","downsample_children_to_actions", "avg"]
    print(children_action_conflict_options)
    #get new dfs
    new_dfs = []

    #fix nodetime
    clean_pred_df["nodeTime"] = pd.to_datetime(clean_pred_df["nodeTime"], utc=True)

    #get val cols
    val_cols = list(clean_pred_df)
    new_cols = []
    for col in val_cols:
        if col not in ["platform","informationID","nodeTime"]:
            new_cols.append(col)
    val_cols = new_cols
    print("\nval_cols")
    print(val_cols)

    child_cols = ["old_child_users","new_child_users"]

    #conflict tracker
    conflict_tracking_dict = {}
    # conflict_types = ["(total_actions > 0) and (total_child_users ==0)", "total_child_users > total_actions"]
    # conflict_types = ["zero_action_conflict", "more_users_than_actions_conflict"]
    # for c in conflict_types:
    #   conflict_tracking_dict[c] = 0

    #use pandas to check conflicts

    freq_action_cols = []
    for action in cp4_actions:
        freq_col = "freq_%s"%action
        freq_action_cols.append(freq_col)
        clean_pred_df[freq_col] = clean_pred_df[action]/clean_pred_df["total_actions"].astype("float32")

    freq_user_status_cols = []
    for user_status in user_statuses:
        user_col = "%s_child_users"%user_status
        freq_col = "freq_" + user_col
        freq_user_status_cols.append(freq_col)
        clean_pred_df[freq_col] = clean_pred_df[user_col]/clean_pred_df["total_child_users"].astype("float32")

    freq_cols = list(freq_user_status_cols + freq_action_cols)

    #new df
    print(clean_pred_df)

    #put pseudo nodeID
    clean_pred_df["pseudo_nodeID"] = [(i+1) for i in range(clean_pred_df.shape[0])]
    clean_pred_df = clean_pred_df.fillna(0)
    print(clean_pred_df)

    # def mark_conflicts(clean_pred_df):

    #   #mark the conflicts
    #   print("\nGetting zero action conflict...")
    #   total_child_users_list = list(clean_pred_df["total_child_users"])
    #   total_actions_list = list(clean_pred_df["total_actions"])
    #   clean_pred_df["zero_action_conflict"] = [1 if ((total_actions>0) and (total_child_users ==0)) else 0 for (total_actions, total_child_users) in zip(total_actions_list, total_child_users_list)]
    #   print("\nzero_action_conflict")
    #   print(clean_pred_df["zero_action_conflict"])

    #   #mark the conflicts
    #   print("\nGetting more users than actiosn conflict...")
    #   clean_pred_df["more_users_conflict"] = [1 if (total_child_users > total_actions) else 0 for (total_actions, total_child_users) in zip(total_actions_list, total_child_users_list)]
    #   print("\nmore_users_conflict")
    #   print(clean_pred_df["more_users_conflict"])

    #   conflict_cols = ["more_users_conflict", "zero_action_conflict"]

    #   return clean_pred_df,conflict_cols

    clean_pred_df,conflict_cols = mark_conflicts(clean_pred_df)

    #conflict dict
    conflict_tracking_dict["zero_action_conflict"] = clean_pred_df["zero_action_conflict"].sum()
    conflict_tracking_dict["more_users_conflict"] = clean_pred_df["more_users_conflict"].sum()
    total_records = clean_pred_df.shape[0]
    print("\ntotal_records: %d"%total_records)
    for c,count in conflict_tracking_dict.items():
        freq = count/float(total_records)
        print("%s count: %d of %d; freq: %.4f"%(c, count, total_records,freq))

    #resolve conflicts
    zero_out_cols = ["total_actions","total_child_users"] + list(cp4_actions + child_cols + freq_cols)
    for col in zero_out_cols:
        clean_pred_df.loc[clean_pred_df["zero_action_conflict"] == 1, col] = 0

    #verify
    temp = clean_pred_df[clean_pred_df["zero_action_conflict"]==1].reset_index(drop=True)
    print("\nzero_action_conflict df")
    temp = temp[zero_out_cols]
    print(temp)
    temp_counts = 0
    for col in zero_out_cols:
        temp_counts+=temp[col].sum()
    print("\ntemp:")
    print(temp)
    if temp_counts != 0:
        print("\nError! temp should be 0.")
        sys.exit(0)
    else:
        print("\nTemp count is ok! Continuing")

    #more users conflict
    children_action_conflict_options = ["upsample_actions_to_children","downsample_children_to_actions", "avg"]

    #first get the nodeIDs that should not be modified!
    unmodified_records = clean_pred_df[clean_pred_df["more_users_conflict"]==0].reset_index(drop=True)

    #get the records to modify
    records_to_modify = clean_pred_df[clean_pred_df["more_users_conflict"]==1].reset_index(drop=True)

    #resolve conflict
    if children_action_conflict_option == "upsample_actions_to_children":
        records_to_modify["total_actions"] = records_to_modify["total_child_users"].copy()
    if children_action_conflict_option == "downsample_children_to_actions":
        records_to_modify["total_child_users"] = records_to_modify["total_actions"].copy()
    if children_action_conflict_option == "avg":
        avg_series = (records_to_modify["total_child_users"] + records_to_modify["total_actions"])/2.0
        avg_series = np.round(avg_series, 0).astype("int32")
        records_to_modify["total_child_users"] = avg_series
        records_to_modify["total_actions"] = avg_series
        print("\navg_series")
        print(avg_series)

    #fix the counts so they match!
    for action in cp4_actions:
        freq_col = "freq_%s"%action
        records_to_modify[action] = records_to_modify[freq_col] * records_to_modify["total_actions"]
        records_to_modify[action] = np.round(records_to_modify[action], 0)

    freq_user_status_cols = []
    for user_status in user_statuses:
        user_col = "%s_child_users"%user_status
        freq_col = "freq_" + user_col
        records_to_modify[user_col] = records_to_modify[freq_col] * records_to_modify["total_child_users"]
        records_to_modify[user_col] = np.round(records_to_modify[user_col], 0)


    #check for rounding error
    total_action_inc_sum = records_to_modify[cp4_actions].sum(axis=1)

    total_child_user_inc_sum = records_to_modify[child_cols].sum(axis=1)
    records_to_modify["total_action_inc_sum"] = total_action_inc_sum
    records_to_modify["total_child_user_inc_sum"] = total_child_user_inc_sum

    #fix rounding error
    records_to_modify["rounding_error"] = total_action_inc_sum - total_child_user_inc_sum
    print(records_to_modify[["total_action_inc_sum","total_child_user_inc_sum", "rounding_error"]].reset_index(drop=True))

    records_to_modify_with_round_error = records_to_modify[records_to_modify["rounding_error"] != 0].reset_index(drop=True)
    records_to_modify_without_round_error = records_to_modify[records_to_modify["rounding_error"] == 0].reset_index(drop=True)
    print("\nrecords_to_modify_with_round_error")
    print(records_to_modify_with_round_error["rounding_error"])
    print("\nrecords_to_modify_without_round_error")
    print(records_to_modify_without_round_error["rounding_error"])

    #if error is 1, that means we have more actions than users, which is ok
    #if error is -1, we have more users than actions which is not ok
    pos_rounding_error_records = records_to_modify_with_round_error[records_to_modify_with_round_error["rounding_error"] > 0].reset_index(drop=True)
    neg_rounding_error_records = records_to_modify_with_round_error[records_to_modify_with_round_error["rounding_error"] < 0].reset_index(drop=True)
    print("\npos_rounding_error_records")
    print(pos_rounding_error_records["rounding_error"])
    print("\nneg_rounding_error_records")
    print(neg_rounding_error_records["rounding_error"])

    #split by youtube and twitter
    twitter_neg_rounding_error_records = neg_rounding_error_records[neg_rounding_error_records["platform"]=="twitter"].reset_index(drop=True)
    youtube_neg_rounding_error_records = neg_rounding_error_records[neg_rounding_error_records["platform"]=="youtube"].reset_index(drop=True)
    print("\ntwitter_neg_rounding_error_records")
    print(twitter_neg_rounding_error_records["rounding_error"])
    print("\nyoutube_neg_rounding_error_records")
    print(youtube_neg_rounding_error_records["rounding_error"])
    twitter_neg_rounding_error_records["twitter_retweet"] = twitter_neg_rounding_error_records["twitter_retweet"] + twitter_neg_rounding_error_records["rounding_error"].abs()
    youtube_neg_rounding_error_records["youtube_comment"] = youtube_neg_rounding_error_records["youtube_comment"] + youtube_neg_rounding_error_records["rounding_error"].abs()
    twitter_neg_rounding_error_records["total_action_inc_sum"] = twitter_neg_rounding_error_records["total_action_inc_sum"] + twitter_neg_rounding_error_records["rounding_error"].abs()
    youtube_neg_rounding_error_records["total_action_inc_sum"] = youtube_neg_rounding_error_records["total_action_inc_sum"] + youtube_neg_rounding_error_records["rounding_error"].abs()

    #set new totals
    twitter_neg_rounding_error_records["total_actions"] = twitter_neg_rounding_error_records["total_action_inc_sum"]
    youtube_neg_rounding_error_records["total_actions"] = youtube_neg_rounding_error_records["total_action_inc_sum"]
    final_records= pd.concat([unmodified_records, youtube_neg_rounding_error_records, twitter_neg_rounding_error_records,pos_rounding_error_records,records_to_modify_without_round_error])
    print("\nfinal_records")
    print(final_records)

    final_records = final_records.sort_values("nodeTime").reset_index(drop=True)
    final_records["total_child_users"] = final_records[child_cols].sum(axis=1)
    final_records["total_actions"] = final_records[cp4_actions].sum(axis=1)

    # #make sure columns add up
    # user_col_sum_series = final_records[child_cols].sum(axis=1)
    # action_col_sum_series = final_records[cp4_actions].sum(axis=1)
    # print("\nuser_col_sum_series")
    # print(user_col_sum_series)
    # print("\naction_col_sum_series")
    # print(action_col_sum_series)
    # print()
    # print("\ntotal_child_users series")
    # print(final_records["total_child_users"])
    # print("\ntotal_actions series")
    # print(final_records["total_actions"])

    # if user_col_sum_series.equals(final_records["total_child_users"]) == True:
    #   print("\nUser series are equal!")
    # else:
    #   print("\nError! User series not equal!")
    #   sys.exit(0)

    # if action_col_sum_series.equals(final_records["total_actions"]) == True:
    #   print("\nAction series are equal!")
    # else:
    #   print("\nError! Action series not equal!")
    #   sys.exit(0)

    #make sure there are no conflicts
    final_records,conflict_cols = mark_conflicts(final_records)
    print("\nBefore bug fix")
    temp = final_records[final_records["zero_action_conflict"]==1]
    print(temp)
    for col in zero_out_cols:
        final_records.loc[final_records["zero_action_conflict"] == 1, col] = 0
    final_records,conflict_cols = mark_conflicts(final_records)
    print("\nAfter bug fix")
    temp = final_records[final_records["zero_action_conflict"]==1]
    print(temp)

    for col in conflict_cols:
        print(col)
        conflict_sum = final_records[col].sum()
        print("conflict sum: %d" %conflict_sum)
        if conflict_sum > 0:
            problem_df = final_records[final_records[col]!=0]

            print("\nproblem_df")
            print(problem_df)
            print("Error! Conflict sum is more than 0!")
            sys.exit(0)
        else:
            print("No conflicts! Continuing...")


    return final_records

def mark_conflicts(clean_pred_df):

    #mark the conflicts
    print("\nGetting zero action conflict...")
    total_child_users_list = list(clean_pred_df["total_child_users"])
    total_actions_list = list(clean_pred_df["total_actions"])
    clean_pred_df["zero_action_conflict"] = [1 if ((total_actions>0) and (total_child_users ==0)) else 0 for (total_actions, total_child_users) in zip(total_actions_list, total_child_users_list)]
    print("\nzero_action_conflict")
    print(clean_pred_df["zero_action_conflict"])

    #mark the conflicts
    print("\nGetting more users than actions conflict...")
    clean_pred_df["more_users_conflict"] = [1 if (total_child_users > total_actions) else 0 for (total_actions, total_child_users) in zip(total_actions_list, total_child_users_list)]
    print("\nmore_users_conflict")
    print(clean_pred_df["more_users_conflict"])

    conflict_cols = ["more_users_conflict", "zero_action_conflict"]

    return clean_pred_df,conflict_cols

def kickout_parent_cols(clean_pred_df):

    clean_cols = list(clean_pred_df)
    new_cols = []
    print("\nClean cols without parent-related cols:")
    for col in clean_cols:
        if "parent" not in col:
            new_cols.append(col)
            print(col)
    clean_pred_df = clean_pred_df[new_cols]

    return clean_pred_df

def regorganize_preds_into_df(test_dates, GRAN,infoIDs,platform_to_infoID_action_count_df_dict,platforms,platform_to_action_dict,main_module_dict):
    platform_list = []
    infoID_list = []
    test_date_list = []
    actionType_list = []
    total_action_count_list = []
    new_children_count_list = []
    old_children_count_list = []
    new_parent_count_list = []
    old_parent_count_list = []
    total_children_count_list = []
    total_parent_count_list = []

    cp4_actions = get_cp4_actions()
    action_to_count_dict = {}
    for action in cp4_actions:
        action_to_count_dict[action] = []

    for platform in platforms:
        actions =  platform_to_action_dict[platform]
        for infoID in infoIDs:
            cur_platform_df = platform_to_infoID_action_count_df_dict[platform]
            cur_infoID_df = cur_platform_df[cur_platform_df["informationID"]==infoID].copy().reset_index(drop=True)
            cur_infoID_df["nodeTime"] = pd.to_datetime(cur_infoID_df["nodeTime"], utc=True)
            for i,test_date in enumerate(test_dates):
                print("date %d: %s %s %s"%((i+1), str(test_date), platform, infoID))
                test_date = pd.to_datetime(test_date, utc=True)

                #first count up actions

                cur_infoID_date_df = cur_infoID_df[cur_infoID_df["nodeTime"]==test_date].reset_index(drop=True)
                print(cur_infoID_date_df)
                total_actions = cur_infoID_date_df[actions].sum(axis=1).reset_index(drop=True).iloc[0]
                for action in cp4_actions:
                    if action in actions:
                        print()
                        print(action)

                        action_count = int(np.round(cur_infoID_date_df[action].iloc[0], 0))
                        print(action_count)
                        action_to_count_dict[action].append(action_count)
                    else:
                        action_to_count_dict[action].append(0)

                #num children
                # total_children = 0
                # total_parents = 0
                num_old_child_users = main_module_dict["module2_pred_dict"][platform]["old"][infoID]["child"][i][0]
                num_new_child_users = main_module_dict["module2_pred_dict"][platform]["new"][infoID]["child"][i][0]
                num_old_parent_users = main_module_dict["module2_pred_dict"][platform]["old"][infoID]["parent"][i][0]
                num_new_parent_users = main_module_dict["module2_pred_dict"][platform]["new"][infoID]["parent"][i][0]
                total_children = num_old_child_users + num_new_child_users
                total_parents = num_old_parent_users + num_new_parent_users

                num_old_child_users = int(np.round(num_old_child_users, 0))
                num_new_child_users = int(np.round(num_new_child_users, 0))
                num_old_parent_users = int(np.round(num_old_parent_users, 0))
                num_new_parent_users = int(np.round(num_new_parent_users, 0))
                total_children = int(np.round(total_children, 0))
                total_parents = int(np.round(total_parents, 0))

                platform_list.append(platform)
                infoID_list.append(infoID)
                test_date_list.append(test_date)
                total_action_count_list.append(total_actions)
                new_children_count_list.append(num_new_child_users)
                old_children_count_list.append(num_old_child_users)
                new_parent_count_list.append(num_new_parent_users)
                old_parent_count_list.append(num_old_parent_users)
                total_children_count_list.append(total_children)
                total_parent_count_list.append(total_parents)

    #make df
    data={
    "platform":platform_list,
    "informationID": infoID_list,
    "nodeTime":test_date_list,
    "total_actions":total_action_count_list,
    "new_child_users":new_children_count_list,
    "old_child_users":old_children_count_list,
    "new_parent_users":new_parent_count_list,
    "old_parent_users":old_parent_count_list,
    "total_child_users": total_children_count_list,
    "total_parent_users" : total_parent_count_list
    }
    clean_up_df = pd.DataFrame(data=data)
    cols = ["platform","informationID","nodeTime","total_actions","new_child_users","old_child_users","total_child_users","new_parent_users","old_parent_users","total_parent_users"]

    for action in cp4_actions:
        cols.append(action)
        clean_up_df[action] = action_to_count_dict[action]
    clean_up_df = clean_up_df[cols]
    print("\nclean_up_df")
    print(clean_up_df)

    for col in cols:
        print()
        print(col)
        print(clean_up_df[col])

    return clean_up_df

def get_platform_actionType_to_actionType_dict():

    adict = {
    "twitter_tweet" : "tweet",
    "twitter_retweet" : "retweet",
    "twitter_quote" : "quote",
    "twitter_reply" : "reply",
    "youtube_video": "video",
    "youtube_comment" : "comment"
    }

    return adict

def convert_actionType_series_to_original_form(aseries):
    adict = get_platform_actionType_to_actionType_dict()
    print("\nConverting actionTypes to original form...")
    return aseries.map(adict)

def get_nearest_date(desired_nodeTime, temp_nodeTime_table):
    print("\nGETTING NEAREST DATE TO %s"%str(desired_nodeTime))
    temp_df = pd.DataFrame(data={"nodeTime":temp_nodeTime_table})
    temp_df["date_distance"] =  temp_df["nodeTime"] - desired_nodeTime
    temp_df["date_distance"] = temp_df["date_distance"].abs()
    min_distance = temp_df["date_distance"].min()
    temp_df = temp_df[temp_df["date_distance"]==min_distance].reset_index(drop=True)
    nearest_date = temp_df["nodeTime"].iloc[0]
    print("\nNearest date to %s is %s" %(str(desired_nodeTime),str(nearest_date)))
    nearest_date = pd.to_datetime(nearest_date, utc=True)
    return nearest_date

def infoID_platform_results_multiproc_v3_old_users(arg_tuple):

    #root_actions

    #get args
    infoID, test_dates, \
    main_module_dict,platform,infoID_weight_df_fp,infoID_history_table_fp,actions,cur_infoID_df_fp, \
    infoID_prop_delay_table_fp,main_output_dir,hyp_dict,output_dir,platform_weight_df,root_actions = arg_tuple

    #set timer
    start_time = time()

    print("\nOpen dfs...")
    infoID_weight_df = pd.read_csv(infoID_weight_df_fp)
    infoID_history_table = pd.read_csv(infoID_history_table_fp)
    infoID_history_table["nodeTime"] = pd.to_datetime(infoID_history_table["nodeTime"], utc=True)
    cur_infoID_df = pd.read_csv(cur_infoID_df_fp)
    cur_infoID_df["nodeTime"] = pd.to_datetime(cur_infoID_df["nodeTime"], utc=True)
    print("\ncur_infoID_df")
    print(cur_infoID_df)
    # sys.exit(0)
    infoID_prop_delay_table = pd.read_csv(infoID_prop_delay_table_fp)

    print("Got dfs!")

    # #make output dir for infoID
    # hyp_infoID = hyp_dict[infoID]
    # output_dir = main_output_dir + hyp_infoID + "/"
    # create_output_dir(output_dir)



    #new user tracker
    cur_new_user_count = 1

    #nodeID counter
    nodeID_counter = 1

    infoID_history_table_size = infoID_history_table.shape[0]

    # #make cur new user dict
    # new_user_df_list = []

    all_dfs_for_cur_infoID_platform_pair = []

    print()
    print(platform)
    print(infoID)
    for i,test_date in enumerate(test_dates):
        test_date = pd.to_datetime(test_date, utc=True)
        print("\nDate %d: %s" %((i+1), str(test_date)))

        #cur date dfs
        cur_date_dfs = []
        #=================================== OLD USERS ===================================
        #first get old child users
        num_old_child_users = main_module_dict["module2_pred_dict"][platform]["old"][infoID]["child"][i]
        num_old_child_users = int(num_old_child_users)
        print("num_old_child_users: %d" %num_old_child_users)

        #check if you even have any users to sample to begin with
        num_old_users_in_weight_df = infoID_weight_df.shape[0]

        #if your model predicts more old users than what actually exists... we need to convert them to new users...
        if num_old_child_users > num_old_users_in_weight_df:
            print("\nnum_old_child_users > num_old_users_in_weight_df")
            print("Converting some to new users...")
            diff =  num_old_child_users - num_old_users_in_weight_df
            prev_num_new_users = main_module_dict["module2_pred_dict"][platform]["new"][infoID]["child"][i]
            main_module_dict["module2_pred_dict"][platform]["new"][infoID]["child"][i]+=diff
            new_num_new_users = main_module_dict["module2_pred_dict"][platform]["new"][infoID]["child"][i]
            print("\nprev_num_new_users: %d,  new_num_new_users: %d"%(prev_num_new_users, new_num_new_users))
            num_old_child_users = num_old_users_in_weight_df

        if num_old_users_in_weight_df > 0:
            #sample user attribute rows
            # num_old_child_users = 10
            weight_sum = infoID_weight_df["user_action_proba"].sum()
            print("\nweight_sum")
            print(weight_sum)

            try:
                if weight_sum > 0:
                    sampled_weight_rows = infoID_weight_df.sample(n=num_old_child_users,weights="user_action_proba" ,replace=False).reset_index(drop=True)
                else:
                    sampled_weight_rows = infoID_weight_df.sample(n=num_old_child_users,replace=True).reset_index(drop=True)
            except:
                if weight_sum > 0:
                    sampled_weight_rows = infoID_weight_df.sample(n=num_old_child_users,weights="user_action_proba",replace=True).reset_index(drop=True)
                else:
                    sampled_weight_rows = infoID_weight_df.sample(n=num_old_child_users,replace=True).reset_index(drop=True)
            print(sampled_weight_rows)

            #save for later
            old_user_rows = sampled_weight_rows.copy()
        else:
            print("\nThere are no old users in the weight df. Just skipping this infoID-platform pair...")
            continue


        #=================================== NEW USERS ===================================
        #first get new child users
        num_new_child_users = main_module_dict["module2_pred_dict"][platform]["new"][infoID]["child"][i]
        num_new_child_users = int(num_new_child_users)
        print("num_new_child_users: %d" %num_new_child_users)

        #sample user attribute rows
        try:
            sampled_weight_rows = infoID_weight_df.sample(n=num_new_child_users,replace=True).reset_index(drop=True)
        except:
            sampled_weight_rows = platform_weight_df.sample(n=num_new_child_users,replace=True).reset_index(drop=True)
            sampled_weight_rows["informationID"] = infoID
        cur_new_user_count+=num_new_child_users
        print(sampled_weight_rows)

        #make new users
        new_users = ["%s_%s_syn_user_%d"%( platform,infoID,j+cur_new_user_count) for j in range(num_new_child_users)]
        sampled_weight_rows["nodeUserID"] = new_users
        print(sampled_weight_rows)

        #save for later
        new_user_rows = sampled_weight_rows.copy()
        #=================================== MATCH USERS TO ACTIONS ===================================

        #users so far
        if num_old_child_users > 0:
            cur_user_df = pd.concat([new_user_rows, old_user_rows]).reset_index(drop=True)
        else:
            cur_user_df = new_user_rows.copy().reset_index(drop=True)
        total_cur_users = cur_user_df.shape[0]
        print("\ntotal_cur_users: %d" %total_cur_users)

        # #get df for the infoID,platform, and date
        # cur_platform_df = platform_to_infoID_action_count_df_dict[platform]
        # cur_infoID_df = cur_platform_df[cur_platform_df["informationID"]==infoID]
        cur_infoID_date_df = cur_infoID_df[cur_infoID_df["nodeTime"]==test_date].reset_index(drop=True)
        print("\ncur_infoID_date_df")
        print(cur_infoID_date_df)
        print(cur_user_df)


        #count total actions
        total_actions = cur_infoID_date_df[actions].sum(axis=1)
        print("\ntotal_actions")
        print(total_actions)

        # sys.exit(0)
        try:
            total_actions = int(total_actions)
        except TypeError:
            total_actions = 0
        print("\ntotal_actions: %d" %total_actions)
        print("total_cur_users: %d" %total_cur_users)

        if total_actions == 0:
            print("\nZero actions... Continuing")
            continue

        if total_cur_users > total_actions:
            print("\nThere are more users than actions...Downsampling users...")
            total_cur_users = total_actions
            print("total_actions: %d" %total_actions)
            print("total_cur_users: %d" %total_cur_users)
            cur_user_df = cur_user_df.sample(n=total_cur_users,replace=False).reset_index(drop=True)
            print("\nUser df after down sampling:")
            print(cur_user_df)

        if total_cur_users < total_actions:
            print("\nThere are more actions than users...Adding repeat users...")
            diff = total_actions - total_cur_users
            print("diff: %d" %diff)
            try:
                repeat_user_rows = cur_user_df.sample(n=diff, weights="user_action_proba",replace=True).reset_index(drop=True)
            except ValueError:
                try:
                    repeat_user_rows = infoID_weight_df.sample(n=diff,replace=True).reset_index(drop=True)
                except:
                    repeat_user_rows = platform_weight_df.sample(n=diff,replace=True).reset_index(drop=True)

            cur_user_df = pd.concat([cur_user_df, repeat_user_rows]).reset_index(drop=True)
            print("\ncur_user_df after user upsample")
            print(cur_user_df)
            #verify size
            verify_df_size(cur_user_df, total_actions,"cur_user_df")

        actionTypes = []
        for action in actions:
            action_count = cur_infoID_date_df[action].iloc[0]
            print("%s count: %d" %(action, action_count))
            actionTypes+=[action for a in range(action_count)]

        #df of users so far
        user_sim_action_df = cur_user_df[["nodeUserID"]]
        user_sim_action_df["nodeTime"] = test_date
        user_sim_action_df = user_sim_action_df[["nodeTime", "nodeUserID"]]
        user_sim_action_df["platform"] = platform
        user_sim_action_df["informationID"] = infoID
        user_sim_action_df["actionType"] = actionTypes
        print("\nuser_sim_action_df")
        print(user_sim_action_df)
        # sys.exit(0)
            #sample users

        #size to enforce
        USER_SIM_ACTION_DF_ENFORCED_SIZE = user_sim_action_df.shape[0]
        print("\nUSER_SIM_ACTION_DF_ENFORCED_SIZE: %d" %USER_SIM_ACTION_DF_ENFORCED_SIZE)

        #verify size
        verify_df_size(user_sim_action_df, USER_SIM_ACTION_DF_ENFORCED_SIZE,"user_sim_action")


        #=================================== GET USER INTERSECTION ===================================
        kept_users = list(cur_user_df["nodeUserID"])
        final_new_user_rows = new_user_rows[new_user_rows["nodeUserID"].isin(kept_users)]
        print("\nfinal_new_user_rows")
        print(final_new_user_rows)

        #=================================== UPDATE WEIGHT TABLE! ===================================
        #get sizes for checking
        weight_table_size = infoID_weight_df.shape[0]
        num_final_new_user_rows = final_new_user_rows.shape[0]
        total_size = weight_table_size + num_final_new_user_rows

        #grow weight df
        #append to weight df
        infoID_weight_df = pd.concat([infoID_weight_df, final_new_user_rows]).reset_index(drop=True)
        print("\nUpdated weight df")
        print(infoID_weight_df)
        my_new_weight_size = infoID_weight_df.shape[0]
        print("\nmy_new_weight_size and theoretical total_size")
        print(my_new_weight_size)
        print(total_size)
        if my_new_weight_size != total_size:
            print("\nError! my_new_weight_size != total_size")
            sys.exit(0)
        else:
            print("\nmy_new_weight_size == total_size. Continuing!")

        #=================================== SELECT PARENT USERS! ===================================
        #relevan

        #get prop delays
        num_prop_delays = user_sim_action_df.shape[0]
        sampled_prop_delays = infoID_prop_delay_table["prop_delay_from_parent"].sample(n=num_prop_delays, replace=True)
        print(sampled_prop_delays)
        user_sim_action_df["prop_delay_from_parent"] = list(sampled_prop_delays)
        user_sim_action_df["nodeID"] = ["%s_%s_nodeID_%d"%(platform,infoID, n_idx + nodeID_counter) for n_idx in range(user_sim_action_df.shape[0])]
        print(user_sim_action_df)

        #now sample parents using the prop delays
        #using each influence proba, we select the nodeTime
        print("\ninfoID_history_table")
        print(infoID_history_table)

        #add to counter
        nodeID_counter+=user_sim_action_df.shape[0]

        #get probas
        cur_user_df = cur_user_df[["nodeUserID","user_infl_proba"]].drop_duplicates("nodeUserID").reset_index(drop=True)
        print("\ncur_user_df")
        print(cur_user_df)

        #merge
        user_sim_action_df = user_sim_action_df.merge(cur_user_df, on="nodeUserID", how="inner").reset_index(drop=True)
        print("\nuser_sim_action_df with weights")
        print(user_sim_action_df)

        #verify size
        verify_df_size(user_sim_action_df, USER_SIM_ACTION_DF_ENFORCED_SIZE,"user_sim_action")

        # print("done")
        # sys.exit(0)

        #first split df by actions
        root_action_df = user_sim_action_df[user_sim_action_df["actionType"].isin(root_actions)].reset_index(drop=True)
        root_action_df["parentUserID"] = root_action_df["nodeUserID"]
        root_action_df["parentID"] = root_action_df["nodeID"]
        root_action_df["rootID"] = root_action_df["nodeID"]

        #response df
        #shuffle
        response_action_df = user_sim_action_df[~user_sim_action_df["actionType"].isin(root_actions)].reset_index(drop=True)
        response_action_df = response_action_df.sample(frac=1).reset_index(drop=True)
        print("\nroot_action_df")
        print(root_action_df)
        print("\nresponse_action_df")
        print(response_action_df)

        #make cur df list
        root_action_df = root_action_df.drop("prop_delay_from_parent", axis=1)

        #might have to delete
        #make sure history table grows by the proper amount
        infoID_history_table = pd.concat([infoID_history_table, root_action_df])
        infoID_history_table_size = infoID_history_table_size+root_action_df.shape[0]
        verify_df_size(infoID_history_table, infoID_history_table_size,"infoID_history_table")

        # print("done")
        # sys.exit(0)

        #get unique prop delays
        response_action_df["prop_delay_from_parent"] = pd.to_timedelta(response_action_df["prop_delay_from_parent"])
        unique_prop_delays = list(response_action_df["prop_delay_from_parent"].unique())
        print("\nunique_prop_delays")
        print(unique_prop_delays)
        # sys.exit(0)

        #GET PARENT IDs/parent users
        parent_users = []
        parentIDs = []
        rootIDs = []

        #save here
        # new_dfs = [root_action_df]
        cur_date_dfs.append(root_action_df)
        for prop_delay in unique_prop_delays:

            #get relevant history
            if prop_delay == np.timedelta64(0,'ns'):
                # continue
                print("\nTIME DELAY IS ZERO, USING SAME PERIOD DATA.")
                relevant_history = root_action_df.copy()
                print("\nrelevant_history")
                print(relevant_history)
            else:
                desired_nodeTime = test_date - prop_delay
                desired_nodeTime = pd.to_datetime(desired_nodeTime, utc=True)
                print("\ndesired_nodeTime")
                print(desired_nodeTime)
                temp_nodeTime_table = infoID_history_table["nodeTime"].copy().drop_duplicates().reset_index(drop=True)
                desired_nodeTime = get_nearest_date(desired_nodeTime, temp_nodeTime_table)




                relevant_history = infoID_history_table[infoID_history_table["nodeTime"]==desired_nodeTime]
                print("\nrelevant_history")
                print(relevant_history)
                # sys.exit(0)
            response_cols = list(response_action_df)
            relevant_history_cols = list(relevant_history)
            print("\nresponse_cols")
            print(response_cols)
            print("\nrelevant_history_cols")
            print(relevant_history_cols)

            cur_response_df = response_action_df[response_action_df["prop_delay_from_parent"]==prop_delay].reset_index(drop=True)
            print("\ncur_response_df")
            print(cur_response_df)
            cur_response_df = cur_response_df.drop("prop_delay_from_parent", axis=1)
            col_order = list(relevant_history)
            num_rows = cur_response_df.shape[0]
            # sys.exit(0)

            for row_idx,row in cur_response_df.iterrows():
                print("Cur idx: %d of %d"%(row_idx, num_rows))
                # print(row)

                row_df = row.to_frame().T.reset_index(drop=True)
                # print("\nrow_df")
                # print(row_df)

                #sample parent
                try:
                    sampled_parent_row = relevant_history.sample(n=1,weights="user_infl_proba").reset_index(drop=True)
                except ValueError:
                    sampled_parent_row = relevant_history.sample(n=1).reset_index(drop=True)
                # print("\nsampled_parent_row")
                # print(sampled_parent_row)

                #add parent records
                row_df["parentID"] = sampled_parent_row["nodeID"]
                row_df["parentUserID"] = sampled_parent_row["nodeUserID"]
                row_df["rootID"] = sampled_parent_row["rootID"]
                row_df = row_df[col_order]
                # new_dfs.append(row_df)
                cur_date_dfs.append(row_df)

                #add this to history
                infoID_history_table = pd.concat([infoID_history_table, row_df]).reset_index(drop=True)
                # print("\nUpdated history table")
                # print(history_table)
            # sys.exit(0)

            # infoID_history_table = pd.concat([infoID_history_table, root_action_df])
            infoID_history_table_size = infoID_history_table_size+cur_response_df.shape[0]
            verify_df_size(infoID_history_table, infoID_history_table_size,"infoID_history_table")

        cur_date_df = pd.concat(cur_date_dfs)
        cur_date_df = cur_date_df.reset_index(drop=True)
        print("\ncur_date_df")
        print(cur_date_df)

        #verify size
        verify_df_size(cur_date_df, USER_SIM_ACTION_DF_ENFORCED_SIZE,"cur_date_df")

        all_dfs_for_cur_infoID_platform_pair.append(cur_date_df.copy())

    if len(all_dfs_for_cur_infoID_platform_pair) > 0:
        cur_infoID_platform_df = pd.concat(all_dfs_for_cur_infoID_platform_pair).reset_index(drop=True)
        print("\ncur_infoID_platform_df")
        print(cur_infoID_platform_df)

        data_fp = output_dir + "%s_%s_simulation.csv"%(platform, infoID)
        cur_infoID_platform_df.to_csv(data_fp, index=False)
        print(data_fp)
    else:
        data_fp = None

    end_time = time()
    total_time_in_minutes = (end_time - start_time) / 60.0
    total_time_in_minutes = np.round(total_time_in_minutes, 2)

    time_str = str("%s %s total_time_in_minutes: %.2f"%(platform, infoID,total_time_in_minutes))
    print(time_str)
    time_fp = output_dir + "time.txt"
    with open(time_fp, "w") as f:
        f.write(time_str)
    print(time_fp)




    return data_fp

def verify_or_change_old_user_counts(cur_date_pred_df,infoID_weight_df):

    #get num old users
    num_old_child_users = cur_date_pred_df["old_child_users"].iloc[0]
    print("num_old_child_users: %d" %num_old_child_users)

    #get num new users
    num_new_child_users = cur_date_pred_df["new_child_users"].iloc[0]
    print("num_new_child_users: %d" %num_new_child_users)

    #get num users in the weight df
    num_old_users_in_weight_df = infoID_weight_df.shape[0]
    print("num_old_users_in_weight_df: %d"%num_old_users_in_weight_df)

    if num_old_child_users > num_old_users_in_weight_df:
        print("\nnum_old_child_users > num_old_users_in_weight_df")
        print("Converting some to new users...")
        diff =  num_old_child_users - num_old_users_in_weight_df
        updated_num_new_child_users = num_new_child_users + diff
        updated_num_old_child_users = num_old_users_in_weight_df
        print("\nOriginal num new users: %d, added: %d, updated num new users: %d"%(num_new_child_users,diff ,updated_num_new_child_users))
        print("\nOriginal num old users: %d, subtracted: %d, updated num old users: %d"%(num_old_child_users,diff ,updated_num_old_child_users))

        cur_date_pred_df["old_child_users"] = updated_num_old_child_users
        cur_date_pred_df["new_child_users"] = updated_num_new_child_users
    else:
        print("\nOld user counts are ok! Continuing!")

    return cur_date_pred_df

def verify_or_change_old_user_counts_v2(cur_date_pred_df,infoID_weight_df,ADD_OLD_USER_COUNT_TO_NEW_USER_COUNT_IF_CONFLICT = True):

    #get num old users
    num_old_child_users = cur_date_pred_df["old_child_users"].iloc[0]
    print("num_old_child_users: %d" %num_old_child_users)

    #get num new users
    num_new_child_users = cur_date_pred_df["new_child_users"].iloc[0]
    print("num_new_child_users: %d" %num_new_child_users)

    #get num users in the weight df
    num_old_users_in_weight_df = infoID_weight_df.shape[0]
    print("num_old_users_in_weight_df: %d"%num_old_users_in_weight_df)

    if num_old_child_users > num_old_users_in_weight_df:
        print("\nnum_old_child_users > num_old_users_in_weight_df")

        diff =  num_old_child_users - num_old_users_in_weight_df
        updated_num_old_child_users = num_old_users_in_weight_df
        if ADD_OLD_USER_COUNT_TO_NEW_USER_COUNT_IF_CONFLICT == True:
            print("ADD_OLD_USER_COUNT_TO_NEW_USER_COUNT_IF_CONFLICT == True")
            print("\nThe number of old users will simply be the number of old users in the weight df. Adding diff to new user count.")
            #update new children
            updated_num_new_child_users = num_new_child_users + diff
            cur_date_pred_df["new_child_users"] = updated_num_new_child_users
        else:
            print("ADD_OLD_USER_COUNT_TO_NEW_USER_COUNT_IF_CONFLICT == False")
            print("\nThe number of old users will simply be the number of old users in the weight df. Not adding diff to new user count.")
            updated_num_new_child_users = num_new_child_users

        #update old children
        cur_date_pred_df["old_child_users"] = updated_num_old_child_users

        print("\nOriginal num new users: %d, added: %d, updated num new users: %d"%(num_new_child_users,diff ,updated_num_new_child_users))
        print("Original num old users: %d, subtracted: %d, updated num old users: %d"%(num_old_child_users,diff ,updated_num_old_child_users))

        updated_child_total = updated_num_old_child_users + updated_num_new_child_users
        cur_date_pred_df["total_child_users"] = updated_child_total
        print("updated_child_total: %d"%updated_child_total)
    else:
        print("\nOld user counts are ok! Continuing!")

    return cur_date_pred_df

#by this point you should have proper num of users
def sample_old_users(infoID_weight_df,num_old_child_users):
    #sample user attribute rows
    # num_old_child_users = 10
    weight_sum = infoID_weight_df["user_action_proba"].sum()
    print("\nweight_sum")
    print(weight_sum)

    num_old_child_users = int(num_old_child_users)
    print("\nnum_old_child_users: %s"%str(num_old_child_users))

    if weight_sum > 0:
        sampled_weight_rows = infoID_weight_df.sample(n=int(num_old_child_users),weights="user_action_proba" ,replace=False).reset_index(drop=True)
    else:
        sampled_weight_rows = infoID_weight_df.sample(n=int(num_old_child_users),replace=True).reset_index(drop=True)

    print("\nsampled old users")
    print(sampled_weight_rows)

    return sampled_weight_rows

# def match_children_to_parents(cur_sim_df, platform,platform_weight_df,infoID_weight_df):

#     #for now, we sample from within the same period
#     num_parents = cur_sim_df.shape[0]
#     parent_users = cur_user_df["nodeUserID"].sample(n=num_parents,weights=cur_user_df["user_infl_proba"] ,replace=False).reset_index(drop=True)
#     print("\nparent_users")
#     print(parent_users)
#     cur_sim_df["parentUserID"] = parent_users

#     return cur_sim_df

#first of all, each user gets 1 action
def match_users_to_actions(cur_date_pred_df, platform, actions, root_action,cur_user_df,test_date, infoID):

    #get actions
    actionTypes = []
    for action in actions:
        action_count = int(cur_date_pred_df[action].iloc[0])
        print("%s count: %d" %(action, action_count))
        actionTypes+=[action for a in range(action_count)]

    #shuffle
    random.shuffle(actionTypes)
    # print("\nShuffled action list")
    # print(actionTypes)

    #get root count
    root_action_count = cur_date_pred_df[root_action].iloc[0]
    print("%s count: %d"%(root_action, root_action_count))

    total_actions = cur_date_pred_df["total_actions"].iloc[0]
    total_child_users = cur_date_pred_df["total_child_users"].iloc[0]
    print("\ntotal_actions: %d"%total_actions)
    print("total_child_users: %d"%total_child_users)

    #sampled users
    sampled_users_list = []
    remaining_users = []

    #set remaining users
    num_remaining_users = int(total_actions)

    if total_actions >= total_child_users:
        sampled_users_list = list(cur_user_df["nodeUserID"])
        num_remaining_users = total_actions - len(sampled_users_list)
        num_remaining_users = int(num_remaining_users)
        print("\nnum_remaining_users: %d"%num_remaining_users)

    #randomly sample the rest of the users
    if num_remaining_users > 0:
        remaining_users = cur_user_df["nodeUserID"].sample(n=num_remaining_users,weights=cur_user_df["user_action_proba"] ,replace=True).reset_index(drop=True)
        print("\nremaining_users")
        print(remaining_users)

    #combine
    all_acting_users = list(sampled_users_list) + list(remaining_users)

    #fill in actions
    cur_sim_df = pd.DataFrame(data={"nodeUserID":all_acting_users, "actionType":actionTypes})

    #fill in extra data
    cur_sim_df["nodeTime"] = test_date
    cur_sim_df["platform"] = platform
    cur_sim_df["informationID"] = infoID

    #col order
    col_order = ["nodeTime", "nodeUserID", "actionType", "platform", "informationID"]
    cur_sim_df = cur_sim_df[col_order]

    return cur_sim_df

def match_children_to_parents(cur_sim_df, platform,cur_user_df, root_action):

    #get min proba
    temp_parent_proba_series = cur_user_df[cur_user_df["user_infl_proba"] > 0]["user_infl_proba"]
    print("\ntemp_parent_proba_series ")
    print(temp_parent_proba_series )
    if len(temp_parent_proba_series ) > 0:
        min_parent_proba = temp_parent_proba_series.min()
        print("\nmin_parent_proba: %s"%str(min_parent_proba))
        min_parent_proba = min_parent_proba/10.0
        print("\nmin_parent_proba with adj: %s"%str(min_parent_proba))
    else:
        min_parent_proba = 1

    #for now, we sample from within the same period
    temp_parent_weight_series = cur_user_df["user_infl_proba"].copy()
    temp_parent_weight_series = temp_parent_weight_series+ min_parent_proba
    print("\ntemp_parent_weight_series")
    print(temp_parent_weight_series)
    num_parents = cur_sim_df.shape[0]
    parent_users = cur_user_df["nodeUserID"].sample(n=num_parents,weights=temp_parent_weight_series ,replace=True).reset_index(drop=True)
    print("\nparent_users")
    print(parent_users)
    cur_sim_df["parentUserID"] = parent_users

    #fix root actions
    print("\nFixing root parents...")
    root_cur_sim_df = cur_sim_df[cur_sim_df["actionType"]==root_action].reset_index(drop=True)
    root_cur_sim_df["parentUserID"] = root_cur_sim_df["nodeUserID"]
    print("\nroot_cur_sim_df")
    print(root_cur_sim_df)

    #non roots
    non_root_cur_sim_df = cur_sim_df[cur_sim_df["actionType"]!=root_action].reset_index(drop=True)
    print("\nnon_root_cur_sim_df")
    print(non_root_cur_sim_df)

    #combine
    cur_sim_df = pd.concat([root_cur_sim_df,non_root_cur_sim_df]).reset_index(drop=True)
    print("\ncur_sim_df after fixing parents...")
    print(cur_sim_df)


    return cur_sim_df

def infoID_platform_results_multiproc_v5_BACKUP_JULY_13(arg_tuple):

    #get args
    infoID, test_dates, \
    main_module_dict,platform,infoID_weight_df,infoID_history_table,actions,infoID_pred_df, \
    infoID_prop_delay_table,main_output_dir,hyp_dict,output_dir,platform_weight_df,root_actions,tracker_fp,ADD_OLD_USER_COUNT_TO_NEW_USER_COUNT_IF_CONFLICT = arg_tuple

    if platform == "youtube":
        root_action = "youtube_video"
    if platform == "twitter":
        root_action = "twitter_tweet"

    #make sure indices are ok
    infoID_weight_df = infoID_weight_df.reset_index(drop=True)

    hyp_infoID = hyp_dict[infoID]

    #time it
    start_time = time()
    #=================================================== start function ===================================================


    #not sure if still needed
    infoID_history_table["nodeTime"] = pd.to_datetime(infoID_history_table["nodeTime"], utc=True)
    infoID_pred_df["nodeTime"] = pd.to_datetime(infoID_pred_df["nodeTime"], utc=True)
    print("\ninfoID_pred_df")
    print(infoID_pred_df)

    #track new users
    cur_new_user_count = 1

    #nodeID counter
    nodeID_counter = 1

    #get cur info
    print("\nPlatform: %s, infoID: %s" %(platform, infoID))

    #get cols
    cols = list(infoID_pred_df)
    print("\ncols: %s"%str(cols))

    #save sim dfs
    all_sim_dfs = []

    for i,test_date in enumerate(test_dates):
        test_date = pd.to_datetime(test_date, utc=True)
        print("\nDate %d: %s" %((i+1), str(test_date)))

        cur_date_pred_df = infoID_pred_df[infoID_pred_df["nodeTime"]==test_date].reset_index(drop=True)
        #=================================================== get old users ===================================================
        #change counts if need be
        # cur_date_pred_df = verify_or_change_old_user_counts(cur_date_pred_df,infoID_weight_df)
        cur_date_pred_df = verify_or_change_old_user_counts_v2(cur_date_pred_df,infoID_weight_df,ADD_OLD_USER_COUNT_TO_NEW_USER_COUNT_IF_CONFLICT)

        #check for 0 count
        total_actions = int(cur_date_pred_df["total_actions"].iloc[0])
        if total_actions == 0:
            print("total_actions == 0, moving on.")
            continue


        num_old_child_users = int(cur_date_pred_df["old_child_users"].iloc[0])
        print("num_old_child_users: %d" %num_old_child_users)

        #sample users if possible
        if num_old_child_users > 0:
            sampled_weight_rows = sample_old_users(infoID_weight_df,num_old_child_users)

            #save for later
            old_user_rows = sampled_weight_rows.copy()
        #=================================================== get new users ===================================================
        #first get new child users
        #get num new users
        num_new_child_users = int(cur_date_pred_df["new_child_users"].iloc[0])
        print("num_new_child_users: %d" %num_new_child_users)

        #sample user attribute rows
        try:
            sampled_weight_rows = infoID_weight_df.sample(n=num_new_child_users,replace=True).reset_index(drop=True)
        except:
            #this is what causes the headache
            #the platform weight df is big AF
            sampled_weight_rows = platform_weight_df.sample(n=num_new_child_users,replace=True).reset_index(drop=True)
            sampled_weight_rows["informationID"] = infoID
        print(sampled_weight_rows)

        #track new users
        cur_new_user_count+=num_new_child_users

        #make new users
        #new_users = ["%s_%s_syn_user_%d"%( platform,infoID,j+cur_new_user_count) for j in range(num_new_child_users)]
        new_users = ["%s_syn_user_%d"%( platform,j+cur_new_user_count) for j in range(num_new_child_users)]
        sampled_weight_rows["nodeUserID"] = new_users
        print(sampled_weight_rows)

        #save for later
        new_user_rows = sampled_weight_rows.copy()
        #=================================================== combine old and new users ===================================================
        #users so far
        if num_old_child_users > 0:
            cur_user_df = pd.concat([new_user_rows, old_user_rows]).reset_index(drop=True)
        else:
            cur_user_df = new_user_rows.copy().reset_index(drop=True)

        print("\ncur_user_df")
        print(cur_user_df)

        total_cur_users = cur_user_df.shape[0]
        print("\ntotal_cur_users: %d" %total_cur_users)
        #=================================================== match child users and actions ===================================================

        #match users and actions
        print("\nMatching users to actions...")
        cur_sim_df = match_users_to_actions(cur_date_pred_df, platform, actions, root_action,cur_user_df,test_date, infoID)
        print("\ncur_sim_df")
        print(cur_sim_df)

        #=================================================== match child users and parent users ===================================================
        #get cur_sim_df
        cur_sim_df = match_children_to_parents(cur_sim_df, platform,cur_user_df,root_action)
        print("\ncur_sim_df with parents")
        print(cur_sim_df)
        #=================================================== nodeID hack ===================================================
        cur_sim_df["nodeID"] = cur_sim_df["nodeUserID"].copy()
        cur_sim_df["parentID"] = cur_sim_df["parentUserID"].copy()
        cur_sim_df["rootID"] = cur_sim_df["parentUserID"].copy()

        #kick out parents
        cur_sim_df = cur_sim_df.drop("parentUserID", axis=1)

        #final df!
        print("\ncur_sim_df")
        print(cur_sim_df)

        col_order = ["nodeTime","nodeID" ,"nodeUserID", "parentID","rootID","actionType", "platform", "informationID"]
        cur_sim_df = cur_sim_df[col_order]

        #=================================== add new sim df ===================================

        all_sim_dfs.append(cur_sim_df)

        #=================================== UPDATE WEIGHT TABLE! ===================================
        #get sizes for checking
        weight_table_size = infoID_weight_df.shape[0]
        num_new_user_rows = new_user_rows.shape[0]
        total_size = weight_table_size + num_new_user_rows

        #grow weight df
        #append to weight df
        infoID_weight_df = pd.concat([infoID_weight_df, new_user_rows]).reset_index(drop=True)
        print("\nUpdated weight df")
        print(infoID_weight_df)
        my_new_weight_size = infoID_weight_df.shape[0]
        print("\nmy_new_weight_size and theoretical total_size")
        print(my_new_weight_size)
        print(total_size)
        if my_new_weight_size != total_size:
            print("\nError! my_new_weight_size != total_size")
            sys.exit(0)
        else:
            print("\nmy_new_weight_size == total_size. Continuing!")

    #=================================== combine sim dfs ===================================
    if total_actions > 0:
        sim_df = pd.concat(all_sim_dfs).reset_index(drop=True)
        print("\nsim_df")
        print(sim_df)

        #fix actions
        sim_df["actionType"] = convert_actionType_series_to_original_form(sim_df["actionType"])

        print("\nsim_df after fixing actionTypes")
        print(sim_df)

        #save it
        sim_fp = output_dir + "%s-%s-simulation.csv"%(platform, hyp_infoID)
        sim_df.to_csv(sim_fp, index=False)
    else:
        sim_fp = None


    #=================================================== end function ===================================================
    end_time = time()
    total_time_in_minutes = (end_time - start_time) / 60.0
    total_time_in_minutes = np.round(total_time_in_minutes, 2)

    time_str = str("%s %s total_time_in_minutes: %.2f"%(platform, infoID,total_time_in_minutes))
    print(time_str)
    time_fp = output_dir + "%s_%s_time.txt"%(platform, hyp_infoID)
    print(hyp_infoID)
    print(output_dir)

    print(time_fp)
    with open(time_fp, "w") as f:
        f.write(time_str)


    print(tracker_fp)
    with open(tracker_fp, "w") as f:
        f.write("")

    return (sim_fp, time_str)


def infoID_platform_results_multiproc_v5(arg_tuple):

    #get args
    infoID, test_dates, \
    main_module_dict,platform,infoID_weight_df,infoID_history_table,actions,infoID_pred_df, \
    infoID_prop_delay_table,main_output_dir,hyp_dict,output_dir,platform_weight_df,root_actions,tracker_fp,ADD_OLD_USER_COUNT_TO_NEW_USER_COUNT_IF_CONFLICT = arg_tuple

    if platform == "youtube":
        root_action = "youtube_video"
    if platform == "twitter":
        root_action = "twitter_tweet"

    #make sure indices are ok
    infoID_weight_df = infoID_weight_df.reset_index(drop=True)

    hyp_infoID = hyp_dict[infoID]

    #time it
    start_time = time()
    #=================================================== start function ===================================================


    #not sure if still needed
    # infoID_history_table["nodeTime"] = pd.to_datetime(infoID_history_table["nodeTime"], utc=True)
    infoID_pred_df["nodeTime"] = pd.to_datetime(infoID_pred_df["nodeTime"], utc=True)
    print("\ninfoID_pred_df")
    print(infoID_pred_df)

    #track new users
    cur_new_user_count = 1

    #nodeID counter
    nodeID_counter = 1

    #get cur info
    print("\nPlatform: %s, infoID: %s" %(platform, infoID))

    #get cols
    cols = list(infoID_pred_df)
    print("\ncols: %s"%str(cols))

    #save sim dfs
    all_sim_dfs = []

    for i,test_date in enumerate(test_dates):
        test_date = pd.to_datetime(test_date, utc=True)
        print("\nDate %d: %s" %((i+1), str(test_date)))

        cur_date_pred_df = infoID_pred_df[infoID_pred_df["nodeTime"]==test_date].reset_index(drop=True)
        #=================================================== get old users ===================================================
        #change counts if need be
        # cur_date_pred_df = verify_or_change_old_user_counts(cur_date_pred_df,infoID_weight_df)
        cur_date_pred_df = verify_or_change_old_user_counts_v2(cur_date_pred_df,infoID_weight_df,ADD_OLD_USER_COUNT_TO_NEW_USER_COUNT_IF_CONFLICT)

        #check for 0 count
        total_actions = int(cur_date_pred_df["total_actions"].iloc[0])
        if total_actions == 0:
            print("total_actions == 0, moving on.")
            continue


        num_old_child_users = int(cur_date_pred_df["old_child_users"].iloc[0])
        print("num_old_child_users: %d" %num_old_child_users)

        #sample users if possible
        if num_old_child_users > 0:
            sampled_weight_rows = sample_old_users(infoID_weight_df,num_old_child_users)

            #save for later
            old_user_rows = sampled_weight_rows.copy()
        #=================================================== get new users ===================================================
        #first get new child users
        #get num new users
        num_new_child_users = int(cur_date_pred_df["new_child_users"].iloc[0])
        print("num_new_child_users: %d" %num_new_child_users)

        #sample user attribute rows
        try:
            sampled_weight_rows = infoID_weight_df.sample(n=num_new_child_users,replace=True).reset_index(drop=True)
        except:
            #this is what causes the headache
            #the platform weight df is big AF
            sampled_weight_rows = platform_weight_df.sample(n=num_new_child_users,replace=True).reset_index(drop=True)
            sampled_weight_rows["informationID"] = infoID
        print(sampled_weight_rows)

        #track new users
        cur_new_user_count+=num_new_child_users

        #make new users
        #new_users = ["%s_%s_syn_user_%d"%( platform,infoID,j+cur_new_user_count) for j in range(num_new_child_users)]
        new_users = ["%s_syn_user_%d"%( platform,j+cur_new_user_count) for j in range(num_new_child_users)]
        sampled_weight_rows["nodeUserID"] = new_users
        print(sampled_weight_rows)

        #save for later
        new_user_rows = sampled_weight_rows.copy()
        #=================================================== combine old and new users ===================================================
        #users so far
        if num_old_child_users > 0:
            cur_user_df = pd.concat([new_user_rows, old_user_rows]).reset_index(drop=True)
        else:
            cur_user_df = new_user_rows.copy().reset_index(drop=True)

        print("\ncur_user_df")
        print(cur_user_df)

        total_cur_users = cur_user_df.shape[0]
        print("\ntotal_cur_users: %d" %total_cur_users)
        #=================================================== match child users and actions ===================================================

        #match users and actions
        print("\nMatching users to actions...")
        cur_sim_df = match_users_to_actions(cur_date_pred_df, platform, actions, root_action,cur_user_df,test_date, infoID)
        print("\ncur_sim_df")
        print(cur_sim_df)

        #=================================================== match child users and parent users ===================================================
        #get cur_sim_df
        cur_sim_df = match_children_to_parents(cur_sim_df, platform,cur_user_df,root_action)
        print("\ncur_sim_df with parents")
        print(cur_sim_df)
        #=================================================== nodeID hack ===================================================
        cur_sim_df["nodeID"] = cur_sim_df["nodeUserID"].copy()
        cur_sim_df["parentID"] = cur_sim_df["parentUserID"].copy()
        cur_sim_df["rootID"] = cur_sim_df["parentUserID"].copy()

        #kick out parents
        cur_sim_df = cur_sim_df.drop("parentUserID", axis=1)

        #final df!
        print("\ncur_sim_df")
        print(cur_sim_df)

        col_order = ["nodeTime","nodeID" ,"nodeUserID", "parentID","rootID","actionType", "platform", "informationID"]
        cur_sim_df = cur_sim_df[col_order]

        #=================================== add new sim df ===================================

        all_sim_dfs.append(cur_sim_df)

        #=================================== UPDATE WEIGHT TABLE! ===================================
        #get sizes for checking
        weight_table_size = infoID_weight_df.shape[0]
        num_new_user_rows = new_user_rows.shape[0]
        total_size = weight_table_size + num_new_user_rows

        #grow weight df
        #append to weight df
        infoID_weight_df = pd.concat([infoID_weight_df, new_user_rows]).reset_index(drop=True)
        print("\nUpdated weight df")
        print(infoID_weight_df)
        my_new_weight_size = infoID_weight_df.shape[0]
        print("\nmy_new_weight_size and theoretical total_size")
        print(my_new_weight_size)
        print(total_size)
        if my_new_weight_size != total_size:
            print("\nError! my_new_weight_size != total_size")
            sys.exit(0)
        else:
            print("\nmy_new_weight_size == total_size. Continuing!")

    #=================================== combine sim dfs ===================================
    if total_actions > 0:
        sim_df = pd.concat(all_sim_dfs).reset_index(drop=True)
        print("\nsim_df")
        print(sim_df)

        #fix actions
        sim_df["actionType"] = convert_actionType_series_to_original_form(sim_df["actionType"])

        print("\nsim_df after fixing actionTypes")
        print(sim_df)

        #save it
        sim_fp = output_dir + "%s-%s-simulation.csv"%(platform, hyp_infoID)
        sim_df.to_csv(sim_fp, index=False)
    else:
        sim_fp = None


    #=================================================== end function ===================================================
    end_time = time()
    total_time_in_minutes = (end_time - start_time) / 60.0
    total_time_in_minutes = np.round(total_time_in_minutes, 2)

    time_str = str("%s %s total_time_in_minutes: %.2f"%(platform, infoID,total_time_in_minutes))
    print(time_str)
    time_fp = output_dir + "%s_%s_time.txt"%(platform, hyp_infoID)
    print(hyp_infoID)
    print(output_dir)

    print(time_fp)
    with open(time_fp, "w") as f:
        f.write(time_str)


    print(tracker_fp)
    with open(tracker_fp, "w") as f:
        f.write("")

    return (sim_fp, time_str)

def infoID_platform_results_multiproc_v4_old_users(arg_tuple):

    #root_actions

    #get args
    infoID, test_dates, \
    main_module_dict,platform,infoID_weight_df,infoID_history_table,actions,cur_infoID_df, \
    infoID_prop_delay_table,main_output_dir,hyp_dict,output_dir,platform_weight_df,root_actions,tracker_fp = arg_tuple

    #set timer
    start_time = time()

    # print("\nOpen dfs...")
    # infoID_weight_df = pd.read_csv(infoID_weight_df_fp)
    # infoID_history_table = pd.read_csv(infoID_history_table_fp)
    infoID_history_table["nodeTime"] = pd.to_datetime(infoID_history_table["nodeTime"], utc=True)
    # cur_infoID_df = pd.read_csv(cur_infoID_df_fp)
    cur_infoID_df["nodeTime"] = pd.to_datetime(cur_infoID_df["nodeTime"], utc=True)
    print("\ncur_infoID_df")
    print(cur_infoID_df)
    # sys.exit(0)
    # infoID_prop_delay_table = pd.read_csv(infoID_prop_delay_table_fp)

    print("Got dfs!")

    # #make output dir for infoID
    # hyp_infoID = hyp_dict[infoID]
    # output_dir = main_output_dir + hyp_infoID + "/"
    # create_output_dir(output_dir)



    #new user tracker
    cur_new_user_count = 1

    #nodeID counter
    nodeID_counter = 1

    infoID_history_table_size = infoID_history_table.shape[0]

    # #make cur new user dict
    # new_user_df_list = []

    all_dfs_for_cur_infoID_platform_pair = []

    print()
    print(platform)
    print(infoID)
    for i,test_date in enumerate(test_dates):
        test_date = pd.to_datetime(test_date, utc=True)
        print("\nDate %d: %s" %((i+1), str(test_date)))

        #cur date dfs
        cur_date_dfs = []
        #=================================== OLD USERS ===================================
        #first get old child users
        num_old_child_users = main_module_dict["module2_pred_dict"][platform]["old"][infoID]["child"][i]
        num_old_child_users = int(num_old_child_users)
        print("num_old_child_users: %d" %num_old_child_users)

        #check if you even have any users to sample to begin with
        num_old_users_in_weight_df = infoID_weight_df.shape[0]

        #if your model predicts more old users than what actually exists... we need to convert them to new users...
        if num_old_child_users > num_old_users_in_weight_df:
            print("\nnum_old_child_users > num_old_users_in_weight_df")
            print("Converting some to new users...")
            diff =  num_old_child_users - num_old_users_in_weight_df
            prev_num_new_users = main_module_dict["module2_pred_dict"][platform]["new"][infoID]["child"][i]
            main_module_dict["module2_pred_dict"][platform]["new"][infoID]["child"][i]+=diff
            new_num_new_users = main_module_dict["module2_pred_dict"][platform]["new"][infoID]["child"][i]
            print("\nprev_num_new_users: %d,  new_num_new_users: %d"%(prev_num_new_users, new_num_new_users))
            num_old_child_users = num_old_users_in_weight_df

        if num_old_users_in_weight_df > 0:
            #sample user attribute rows
            # num_old_child_users = 10
            weight_sum = infoID_weight_df["user_action_proba"].sum()
            print("\nweight_sum")
            print(weight_sum)

            try:
                if weight_sum > 0:
                    sampled_weight_rows = infoID_weight_df.sample(n=num_old_child_users,weights="user_action_proba" ,replace=False).reset_index(drop=True)
                else:
                    sampled_weight_rows = infoID_weight_df.sample(n=num_old_child_users,replace=True).reset_index(drop=True)
            except:
                if weight_sum > 0:
                    sampled_weight_rows = infoID_weight_df.sample(n=num_old_child_users,weights="user_action_proba",replace=True).reset_index(drop=True)
                else:
                    sampled_weight_rows = infoID_weight_df.sample(n=num_old_child_users,replace=True).reset_index(drop=True)
            print(sampled_weight_rows)

            #save for later
            old_user_rows = sampled_weight_rows.copy()
        else:
            print("\nThere are no old users in the weight df. Just skipping this infoID-platform pair...")
            continue


        #=================================== NEW USERS ===================================
        #first get new child users
        num_new_child_users = main_module_dict["module2_pred_dict"][platform]["new"][infoID]["child"][i]
        num_new_child_users = int(num_new_child_users)
        print("num_new_child_users: %d" %num_new_child_users)

        #sample user attribute rows
        try:
            sampled_weight_rows = infoID_weight_df.sample(n=num_new_child_users,replace=True).reset_index(drop=True)
        except:
            sampled_weight_rows = platform_weight_df.sample(n=num_new_child_users,replace=True).reset_index(drop=True)
            sampled_weight_rows["informationID"] = infoID
        cur_new_user_count+=num_new_child_users
        print(sampled_weight_rows)

        #make new users
        new_users = ["%s_%s_syn_user_%d"%( platform,infoID,j+cur_new_user_count) for j in range(num_new_child_users)]
        sampled_weight_rows["nodeUserID"] = new_users
        print(sampled_weight_rows)

        #save for later
        new_user_rows = sampled_weight_rows.copy()
        #=================================== MATCH USERS TO ACTIONS ===================================

        #users so far
        if num_old_child_users > 0:
            cur_user_df = pd.concat([new_user_rows, old_user_rows]).reset_index(drop=True)
        else:
            cur_user_df = new_user_rows.copy().reset_index(drop=True)
        total_cur_users = cur_user_df.shape[0]
        print("\ntotal_cur_users: %d" %total_cur_users)

        # #get df for the infoID,platform, and date
        # cur_platform_df = platform_to_infoID_action_count_df_dict[platform]
        # cur_infoID_df = cur_platform_df[cur_platform_df["informationID"]==infoID]
        cur_infoID_date_df = cur_infoID_df[cur_infoID_df["nodeTime"]==test_date].reset_index(drop=True)
        print("\ncur_infoID_date_df")
        print(cur_infoID_date_df)
        print(cur_user_df)


        #count total actions
        total_actions = cur_infoID_date_df[actions].sum(axis=1)
        print("\ntotal_actions")
        print(total_actions)

        # sys.exit(0)
        try:
            total_actions = int(total_actions)
        except TypeError:
            total_actions = 0
        print("\ntotal_actions: %d" %total_actions)
        print("total_cur_users: %d" %total_cur_users)

        if total_actions == 0:
            print("\nZero actions... Continuing")
            continue

        if total_cur_users > total_actions:
            print("\nThere are more users than actions...Downsampling users...")
            total_cur_users = total_actions
            print("total_actions: %d" %total_actions)
            print("total_cur_users: %d" %total_cur_users)
            cur_user_df = cur_user_df.sample(n=total_cur_users,replace=False).reset_index(drop=True)
            print("\nUser df after down sampling:")
            print(cur_user_df)

        if total_cur_users < total_actions:
            print("\nThere are more actions than users...Adding repeat users...")
            diff = total_actions - total_cur_users
            print("diff: %d" %diff)
            try:
                repeat_user_rows = cur_user_df.sample(n=diff, weights="user_action_proba",replace=True).reset_index(drop=True)
            except ValueError:
                try:
                    repeat_user_rows = infoID_weight_df.sample(n=diff,replace=True).reset_index(drop=True)
                except:
                    repeat_user_rows = platform_weight_df.sample(n=diff,replace=True).reset_index(drop=True)

            cur_user_df = pd.concat([cur_user_df, repeat_user_rows]).reset_index(drop=True)
            print("\ncur_user_df after user upsample")
            print(cur_user_df)
            #verify size
            verify_df_size(cur_user_df, total_actions,"cur_user_df")

        actionTypes = []
        for action in actions:
            action_count = cur_infoID_date_df[action].iloc[0]
            print("%s count: %d" %(action, action_count))
            actionTypes+=[action for a in range(action_count)]

        #df of users so far
        user_sim_action_df = cur_user_df[["nodeUserID"]]
        user_sim_action_df["nodeTime"] = test_date
        user_sim_action_df = user_sim_action_df[["nodeTime", "nodeUserID"]]
        user_sim_action_df["platform"] = platform
        user_sim_action_df["informationID"] = infoID
        user_sim_action_df["actionType"] = actionTypes
        print("\nuser_sim_action_df")
        print(user_sim_action_df)
        # sys.exit(0)
            #sample users

        #size to enforce
        USER_SIM_ACTION_DF_ENFORCED_SIZE = user_sim_action_df.shape[0]
        print("\nUSER_SIM_ACTION_DF_ENFORCED_SIZE: %d" %USER_SIM_ACTION_DF_ENFORCED_SIZE)

        #verify size
        verify_df_size(user_sim_action_df, USER_SIM_ACTION_DF_ENFORCED_SIZE,"user_sim_action")


        #=================================== GET USER INTERSECTION ===================================
        kept_users = list(cur_user_df["nodeUserID"])
        final_new_user_rows = new_user_rows[new_user_rows["nodeUserID"].isin(kept_users)]
        print("\nfinal_new_user_rows")
        print(final_new_user_rows)

        #=================================== UPDATE WEIGHT TABLE! ===================================
        #get sizes for checking
        weight_table_size = infoID_weight_df.shape[0]
        num_final_new_user_rows = final_new_user_rows.shape[0]
        total_size = weight_table_size + num_final_new_user_rows

        #grow weight df
        #append to weight df
        infoID_weight_df = pd.concat([infoID_weight_df, final_new_user_rows]).reset_index(drop=True)
        print("\nUpdated weight df")
        print(infoID_weight_df)
        my_new_weight_size = infoID_weight_df.shape[0]
        print("\nmy_new_weight_size and theoretical total_size")
        print(my_new_weight_size)
        print(total_size)
        if my_new_weight_size != total_size:
            print("\nError! my_new_weight_size != total_size")
            sys.exit(0)
        else:
            print("\nmy_new_weight_size == total_size. Continuing!")

        #=================================== SELECT PARENT USERS! ===================================
        #relevan

        #get prop delays
        num_prop_delays = user_sim_action_df.shape[0]
        sampled_prop_delays = infoID_prop_delay_table["prop_delay_from_parent"].sample(n=num_prop_delays, replace=True)
        print(sampled_prop_delays)
        user_sim_action_df["prop_delay_from_parent"] = list(sampled_prop_delays)
        user_sim_action_df["nodeID"] = ["%s_%s_nodeID_%d"%(platform,infoID, n_idx + nodeID_counter) for n_idx in range(user_sim_action_df.shape[0])]
        print(user_sim_action_df)

        #now sample parents using the prop delays
        #using each influence proba, we select the nodeTime
        print("\ninfoID_history_table")
        print(infoID_history_table)

        #add to counter
        nodeID_counter+=user_sim_action_df.shape[0]

        #get probas
        cur_user_df = cur_user_df[["nodeUserID","user_infl_proba"]].drop_duplicates("nodeUserID").reset_index(drop=True)
        print("\ncur_user_df")
        print(cur_user_df)

        #merge
        user_sim_action_df = user_sim_action_df.merge(cur_user_df, on="nodeUserID", how="inner").reset_index(drop=True)
        print("\nuser_sim_action_df with weights")
        print(user_sim_action_df)

        #verify size
        verify_df_size(user_sim_action_df, USER_SIM_ACTION_DF_ENFORCED_SIZE,"user_sim_action")

        # print("done")
        # sys.exit(0)

        #first split df by actions
        root_action_df = user_sim_action_df[user_sim_action_df["actionType"].isin(root_actions)].reset_index(drop=True)
        root_action_df["parentUserID"] = root_action_df["nodeUserID"]
        root_action_df["parentID"] = root_action_df["nodeID"]
        root_action_df["rootID"] = root_action_df["nodeID"]

        #response df
        #shuffle
        response_action_df = user_sim_action_df[~user_sim_action_df["actionType"].isin(root_actions)].reset_index(drop=True)
        response_action_df = response_action_df.sample(frac=1).reset_index(drop=True)
        print("\nroot_action_df")
        print(root_action_df)
        print("\nresponse_action_df")
        print(response_action_df)

        #make cur df list
        root_action_df = root_action_df.drop("prop_delay_from_parent", axis=1)

        #might have to delete
        #make sure history table grows by the proper amount
        infoID_history_table = pd.concat([infoID_history_table, root_action_df])
        infoID_history_table_size = infoID_history_table_size+root_action_df.shape[0]
        verify_df_size(infoID_history_table, infoID_history_table_size,"infoID_history_table")

        # print("done")
        # sys.exit(0)

        #get unique prop delays
        response_action_df["prop_delay_from_parent"] = pd.to_timedelta(response_action_df["prop_delay_from_parent"])
        unique_prop_delays = list(response_action_df["prop_delay_from_parent"].unique())
        print("\nunique_prop_delays")
        print(unique_prop_delays)
        # sys.exit(0)

        #GET PARENT IDs/parent users
        parent_users = []
        parentIDs = []
        rootIDs = []

        #save here
        # new_dfs = [root_action_df]
        cur_date_dfs.append(root_action_df)
        for prop_delay in unique_prop_delays:

            #get relevant history
            if prop_delay == np.timedelta64(0,'ns'):
                # continue
                print("\nTIME DELAY IS ZERO, USING SAME PERIOD DATA.")
                relevant_history = root_action_df.copy()
                print("\nrelevant_history")
                print(relevant_history)
            else:
                desired_nodeTime = test_date - prop_delay
                desired_nodeTime = pd.to_datetime(desired_nodeTime, utc=True)
                print("\ndesired_nodeTime")
                print(desired_nodeTime)
                temp_nodeTime_table = infoID_history_table["nodeTime"].copy().drop_duplicates().reset_index(drop=True)
                desired_nodeTime = get_nearest_date(desired_nodeTime, temp_nodeTime_table)




                relevant_history = infoID_history_table[infoID_history_table["nodeTime"]==desired_nodeTime]
                print("\nrelevant_history")
                print(relevant_history)
                # sys.exit(0)
            response_cols = list(response_action_df)
            relevant_history_cols = list(relevant_history)
            print("\nresponse_cols")
            print(response_cols)
            print("\nrelevant_history_cols")
            print(relevant_history_cols)

            cur_response_df = response_action_df[response_action_df["prop_delay_from_parent"]==prop_delay].reset_index(drop=True)
            print("\ncur_response_df")
            print(cur_response_df)
            cur_response_df = cur_response_df.drop("prop_delay_from_parent", axis=1)
            col_order = list(relevant_history)
            num_rows = cur_response_df.shape[0]
            # sys.exit(0)

            for row_idx,row in cur_response_df.iterrows():
                print("Cur idx: %d of %d"%(row_idx, num_rows))
                # print(row)

                row_df = row.to_frame().T.reset_index(drop=True)
                # print("\nrow_df")
                # print(row_df)

                #sample parent
                try:
                    sampled_parent_row = relevant_history.sample(n=1,weights="user_infl_proba").reset_index(drop=True)
                except ValueError:
                    sampled_parent_row = relevant_history.sample(n=1).reset_index(drop=True)
                # print("\nsampled_parent_row")
                # print(sampled_parent_row)

                #add parent records
                row_df["parentID"] = sampled_parent_row["nodeID"]
                row_df["parentUserID"] = sampled_parent_row["nodeUserID"]
                row_df["rootID"] = sampled_parent_row["rootID"]
                row_df = row_df[col_order]
                # new_dfs.append(row_df)
                cur_date_dfs.append(row_df)

                #add this to history
                infoID_history_table = pd.concat([infoID_history_table, row_df]).reset_index(drop=True)
                # print("\nUpdated history table")
                # print(history_table)
            # sys.exit(0)

            # infoID_history_table = pd.concat([infoID_history_table, root_action_df])
            infoID_history_table_size = infoID_history_table_size+cur_response_df.shape[0]
            verify_df_size(infoID_history_table, infoID_history_table_size,"infoID_history_table")

        cur_date_df = pd.concat(cur_date_dfs)
        cur_date_df = cur_date_df.reset_index(drop=True)
        print("\ncur_date_df")
        print(cur_date_df)

        #verify size
        verify_df_size(cur_date_df, USER_SIM_ACTION_DF_ENFORCED_SIZE,"cur_date_df")

        all_dfs_for_cur_infoID_platform_pair.append(cur_date_df.copy())

    if len(all_dfs_for_cur_infoID_platform_pair) > 0:
        cur_infoID_platform_df = pd.concat(all_dfs_for_cur_infoID_platform_pair).reset_index(drop=True)
        print("\ncur_infoID_platform_df")
        print(cur_infoID_platform_df)

        data_fp = output_dir + "%s_%s_simulation.csv"%(platform, infoID)
        cur_infoID_platform_df.to_csv(data_fp, index=False)
        print(data_fp)
    else:
        data_fp = None

    end_time = time()
    total_time_in_minutes = (end_time - start_time) / 60.0
    total_time_in_minutes = np.round(total_time_in_minutes, 2)

    time_str = str("%s %s total_time_in_minutes: %.2f"%(platform, infoID,total_time_in_minutes))
    print(time_str)
    time_fp = output_dir + "%s_%s_time.txt"%(platform, infoID)
    with open(time_fp, "w") as f:
        f.write(time_str)
    print(time_fp)

    with open(tracker_fp, "w") as f:
        f.write("")




    return (data_fp,time_str)

def infoID_platform_results_multiproc_v2(arg_tuple):

    #get args
    infoID, test_dates, \
    main_module_dict,platform,infoID_weight_df_fp,infoID_history_table_fp,actions,cur_infoID_df_fp, \
    infoID_prop_delay_table_fp,main_output_dir,hyp_dict,output_dir = arg_tuple

    #set timer
    start_time = time()

    print("\nOpen dfs...")
    infoID_weight_df = pd.read_csv(infoID_weight_df_fp)
    infoID_history_table = pd.read_csv(infoID_history_table_fp)
    cur_infoID_df = pd.read_csv(cur_infoID_df_fp)
    infoID_prop_delay_table = pd.read_csv(infoID_prop_delay_table_fp)
    print("Got dfs!")

    # #make output dir for infoID
    # hyp_infoID = hyp_dict[infoID]
    # output_dir = main_output_dir + hyp_infoID + "/"
    # create_output_dir(output_dir)



    #new user tracker
    cur_new_user_count = 1

    #nodeID counter
    nodeID_counter = 1

    infoID_history_table_size = infoID_history_table.shape[0]

    # #make cur new user dict
    # new_user_df_list = []

    all_dfs_for_cur_infoID_platform_pair = []

    print()
    print(platform)
    print(infoID)
    for i,test_date in enumerate(test_dates):
        test_date = pd.to_datetime(test_date, utc=True)
        print("\nDate %d: %s" %((i+1), str(test_date)))

        #cur date dfs
        cur_date_dfs = []
        #=================================== OLD USERS ===================================
        #first get old child users
        num_old_child_users = main_module_dict["module2_pred_dict"][platform]["old"][infoID]["child"][i]
        num_old_child_users = int(num_old_child_users)
        print("num_old_child_users: %d" %num_old_child_users)

        #sample user attribute rows
        # num_old_child_users = 10
        weight_sum = infoID_weight_df["user_action_proba"].sum()
        print("\nweight_sum")
        print(weight_sum)

        try:
            if weight_sum > 0:
                sampled_weight_rows = infoID_weight_df.sample(n=num_old_child_users,weights="user_action_proba" ,replace=False).reset_index(drop=True)
            else:
                sampled_weight_rows = infoID_weight_df.sample(n=num_old_child_users,replace=True).reset_index(drop=True)
        except:
            if weight_sum > 0:
                sampled_weight_rows = infoID_weight_df.sample(n=num_old_child_users,weights="user_action_proba",replace=True).reset_index(drop=True)
            else:
                sampled_weight_rows = infoID_weight_df.sample(n=num_old_child_users,replace=True).reset_index(drop=True)
        print(sampled_weight_rows)

        #save for later
        old_user_rows = sampled_weight_rows.copy()

        #=================================== NEW USERS ===================================
        #first get new child users
        num_new_child_users = main_module_dict["module2_pred_dict"][platform]["new"][infoID]["child"][i]
        num_new_child_users = int(num_new_child_users)
        print("num_new_child_users: %d" %num_new_child_users)

        #sample user attribute rows
        sampled_weight_rows = infoID_weight_df.sample(n=num_new_child_users,replace=True).reset_index(drop=True)
        cur_new_user_count+=num_new_child_users
        print(sampled_weight_rows)

        #make new users
        new_users = ["%s_%s_syn_user_%d"%( platform,infoID,j+cur_new_user_count) for j in range(num_new_child_users)]
        sampled_weight_rows["nodeUserID"] = new_users
        print(sampled_weight_rows)

        #save for later
        new_user_rows = sampled_weight_rows.copy()
        #=================================== MATCH USERS TO ACTIONS ===================================

        #users so far
        cur_user_df = pd.concat([new_user_rows, old_user_rows]).reset_index(drop=True)
        total_cur_users = cur_user_df.shape[0]
        print("\ntotal_cur_users: %d" %total_cur_users)

        # #get df for the infoID,platform, and date
        # cur_platform_df = platform_to_infoID_action_count_df_dict[platform]
        # cur_infoID_df = cur_platform_df[cur_platform_df["informationID"]==infoID]
        cur_infoID_date_df = cur_infoID_df[cur_infoID_df["nodeTime"]==test_date].reset_index(drop=True)
        print(cur_infoID_date_df)
        print(cur_user_df)

        #count total actions
        total_actions = cur_infoID_date_df[actions].sum(axis=1)
        print("\ntotal_actions")
        print(total_actions)
        try:
            total_actions = int(total_actions)
        except TypeError:
            total_actions = 0
        print("\ntotal_actions: %d" %total_actions)
        print("total_cur_users: %d" %total_cur_users)

        if total_actions == 0:
            print("\nZero actions... Continuing")
            continue

        if total_cur_users > total_actions:
            print("\nThere are more users than actions...Downsampling users...")
            total_cur_users = total_actions
            print("total_actions: %d" %total_actions)
            print("total_cur_users: %d" %total_cur_users)
            cur_user_df = cur_user_df.sample(n=total_cur_users,replace=False).reset_index(drop=True)
            print("\nUser df after down sampling:")
            print(cur_user_df)

        if total_cur_users < total_actions:
            print("\nThere are more actions than users...Adding repeat users...")
            diff = total_actions - total_cur_users
            print("diff: %d" %diff)
            repeat_user_rows = cur_user_df.sample(n=diff, weights="user_action_proba",replace=True).reset_index(drop=True)
            cur_user_df = pd.concat([cur_user_df, repeat_user_rows]).reset_index(drop=True)
            print("\ncur_user_df after user upsample")
            print(cur_user_df)
            #verify size
            verify_df_size(cur_user_df, total_actions,"cur_user_df")

        actionTypes = []
        for action in actions:
            action_count = cur_infoID_date_df[action].iloc[0]
            print("%s count: %d" %(action, action_count))
            actionTypes+=[action for a in range(action_count)]

        #df of users so far
        user_sim_action_df = cur_user_df[["nodeUserID"]]
        user_sim_action_df["nodeTime"] = test_date
        user_sim_action_df = user_sim_action_df[["nodeTime", "nodeUserID"]]
        user_sim_action_df["platform"] = platform
        user_sim_action_df["informationID"] = infoID
        user_sim_action_df["actionType"] = actionTypes
        print("\nuser_sim_action_df")
        print(user_sim_action_df)
        # sys.exit(0)
            #sample users

        #size to enforce
        USER_SIM_ACTION_DF_ENFORCED_SIZE = user_sim_action_df.shape[0]
        print("\nUSER_SIM_ACTION_DF_ENFORCED_SIZE: %d" %USER_SIM_ACTION_DF_ENFORCED_SIZE)

        #verify size
        verify_df_size(user_sim_action_df, USER_SIM_ACTION_DF_ENFORCED_SIZE,"user_sim_action")


        #=================================== GET USER INTERSECTION ===================================
        kept_users = list(cur_user_df["nodeUserID"])
        final_new_user_rows = new_user_rows[new_user_rows["nodeUserID"].isin(kept_users)]
        print("\nfinal_new_user_rows")
        print(final_new_user_rows)

        #=================================== UPDATE WEIGHT TABLE! ===================================
        #get sizes for checking
        weight_table_size = infoID_weight_df.shape[0]
        num_final_new_user_rows = final_new_user_rows.shape[0]
        total_size = weight_table_size + num_final_new_user_rows

        #grow weight df
        #append to weight df
        infoID_weight_df = pd.concat([infoID_weight_df, final_new_user_rows]).reset_index(drop=True)
        print("\nUpdated weight df")
        print(infoID_weight_df)
        my_new_weight_size = infoID_weight_df.shape[0]
        print("\nmy_new_weight_size and theoretical total_size")
        print(my_new_weight_size)
        print(total_size)
        if my_new_weight_size != total_size:
            print("\nError! my_new_weight_size != total_size")
            sys.exit(0)
        else:
            print("\nmy_new_weight_size == total_size. Continuing!")

        #=================================== SELECT PARENT USERS! ===================================


        #get prop delays
        num_prop_delays = user_sim_action_df.shape[0]
        sampled_prop_delays = infoID_prop_delay_table["prop_delay_from_parent"].sample(n=num_prop_delays, replace=True)
        print(sampled_prop_delays)
        user_sim_action_df["prop_delay_from_parent"] = list(sampled_prop_delays)
        user_sim_action_df["nodeID"] = ["%s_%s_nodeID_%d"%(platform,infoID, n_idx + nodeID_counter) for n_idx in range(user_sim_action_df.shape[0])]
        print(user_sim_action_df)

        #now sample parents using the prop delays
        #using each influence proba, we select the nodeTime
        print("\ninfoID_history_table")
        print(infoID_history_table)

        #add to counter
        nodeID_counter+=user_sim_action_df.shape[0]

        #get probas
        cur_user_df = cur_user_df[["nodeUserID","user_infl_proba"]].drop_duplicates("nodeUserID").reset_index(drop=True)
        print("\ncur_user_df")
        print(cur_user_df)

        #merge
        user_sim_action_df = user_sim_action_df.merge(cur_user_df, on="nodeUserID", how="inner").reset_index(drop=True)
        print("\nuser_sim_action_df with weights")
        print(user_sim_action_df)

        #verify size
        verify_df_size(user_sim_action_df, USER_SIM_ACTION_DF_ENFORCED_SIZE,"user_sim_action")

        # print("done")
        # sys.exit(0)

        #first split df by actions
        root_action_df = user_sim_action_df[user_sim_action_df["actionType"].isin(root_actions)].reset_index(drop=True)
        root_action_df["parentUserID"] = root_action_df["nodeUserID"]
        root_action_df["parentID"] = root_action_df["nodeID"]
        root_action_df["rootID"] = root_action_df["nodeID"]

        #response df
        #shuffle
        response_action_df = user_sim_action_df[~user_sim_action_df["actionType"].isin(root_actions)].reset_index(drop=True)
        response_action_df = response_action_df.sample(frac=1).reset_index(drop=True)
        print("\nroot_action_df")
        print(root_action_df)
        print("\nresponse_action_df")
        print(response_action_df)

        #make cur df list
        root_action_df = root_action_df.drop("prop_delay_from_parent", axis=1)

        #might have to delete
        #make sure history table grows by the proper amount
        infoID_history_table = pd.concat([infoID_history_table, root_action_df])
        infoID_history_table_size = infoID_history_table_size+root_action_df.shape[0]
        verify_df_size(infoID_history_table, infoID_history_table_size,"infoID_history_table")

        # print("done")
        # sys.exit(0)

        #get unique prop delays
        response_action_df["prop_delay_from_parent"] = pd.to_timedelta(response_action_df["prop_delay_from_parent"])
        unique_prop_delays = list(response_action_df["prop_delay_from_parent"].unique())
        print("\nunique_prop_delays")
        print(unique_prop_delays)
        # sys.exit(0)

        #GET PARENT IDs/parent users
        parent_users = []
        parentIDs = []
        rootIDs = []

        #save here
        # new_dfs = [root_action_df]
        cur_date_dfs.append(root_action_df)
        for prop_delay in unique_prop_delays:

            #get relevant history
            if prop_delay == np.timedelta64(0,'ns'):
                # continue
                print("\nTIME DELAY IS ZERO, USING SAME PERIOD DATA.")
                relevant_history = root_action_df.copy()
                print("\nrelevant_history")
                print(relevant_history)
            else:
                desired_nodeTime = test_date - prop_delay
                print("\ndesired_nodeTime")
                print(desired_nodeTime)
                relevant_history = infoID_history_table[infoID_history_table["nodeTime"]==desired_nodeTime]
                print("\nrelevant_history")
                print(relevant_history)
                # sys.exit(0)
            response_cols = list(response_action_df)
            relevant_history_cols = list(relevant_history)
            print("\nresponse_cols")
            print(response_cols)
            print("\nrelevant_history_cols")
            print(relevant_history_cols)

            cur_response_df = response_action_df[response_action_df["prop_delay_from_parent"]==prop_delay].reset_index(drop=True)
            print("\ncur_response_df")
            print(cur_response_df)
            cur_response_df = cur_response_df.drop("prop_delay_from_parent", axis=1)
            col_order = list(relevant_history)
            num_rows = cur_response_df.shape[0]
            # sys.exit(0)

            for row_idx,row in cur_response_df.iterrows():
                print("Cur idx: %d of %d"%(row_idx, num_rows))
                # print(row)

                row_df = row.to_frame().T.reset_index(drop=True)
                # print("\nrow_df")
                # print(row_df)

                #sample parent
                try:
                    sampled_parent_row = relevant_history.sample(n=1,weights="user_infl_proba").reset_index(drop=True)
                except ValueError:
                    sampled_parent_row = relevant_history.sample(n=1).reset_index(drop=True)
                # print("\nsampled_parent_row")
                # print(sampled_parent_row)

                #add parent records
                row_df["parentID"] = sampled_parent_row["nodeID"]
                row_df["parentUserID"] = sampled_parent_row["nodeUserID"]
                row_df["rootID"] = sampled_parent_row["rootID"]
                row_df = row_df[col_order]
                # new_dfs.append(row_df)
                cur_date_dfs.append(row_df)

                #add this to history
                infoID_history_table = pd.concat([infoID_history_table, row_df]).reset_index(drop=True)
                # print("\nUpdated history table")
                # print(history_table)
            # sys.exit(0)

            # infoID_history_table = pd.concat([infoID_history_table, root_action_df])
            infoID_history_table_size = infoID_history_table_size+cur_response_df.shape[0]
            verify_df_size(infoID_history_table, infoID_history_table_size,"infoID_history_table")

        cur_date_df = pd.concat(cur_date_dfs)
        cur_date_df = cur_date_df.reset_index(drop=True)
        print("\ncur_date_df")
        print(cur_date_df)

        #verify size
        verify_df_size(cur_date_df, USER_SIM_ACTION_DF_ENFORCED_SIZE,"cur_date_df")

        all_dfs_for_cur_infoID_platform_pair.append(cur_date_df)

    if len(all_dfs_for_cur_infoID_platform_pair) != 0:
        cur_infoID_platform_df = pd.concat(all_dfs_for_cur_infoID_platform_pair).reset_index(drop=True)
        print("\ncur_infoID_platform_df")
        print(cur_infoID_platform_df)

        data_fp = output_dir + "cur_infoID_platform_df.csv"
        cur_infoID_platform_df.to_csv(data_fp, index=False)
        print(data_fp)
    else:
        data_fp = None

    end_time = time()
    total_time_in_minutes = (end_time - start_time) / 60.0
    total_time_in_minutes = np.round(total_time_in_minutes, 2)

    time_str = str("%s %s total_time_in_minutes: %.2f"%(platform, infoID,total_time_in_minutes))
    print(time_str)
    time_fp = output_dir + "time.txt"
    with open(time_fp, "w") as f:
        f.write(time_str)
    print(time_fp)




    return data_fp

def infoID_platform_results_multiproc(arg_tuple):

    #get args
    platform_weight_df, infoID, platform_history_table,test_dates, \
    main_module_dict,platform,infoID_weight_df,infoID_history_table,actions,cur_infoID_df, \
    infoID_prop_delay_table,main_output_dir,hyp_dict = arg_tuple

    #make output dir for infoID
    hyp_infoID = hyp_dict[infoID]
    output_dir = main_output_dir + hyp_infoID + "/"
    create_output_dir(output_dir)

    #set timer
    start_time = time()

    #new user tracker
    cur_new_user_count = 1

    #nodeID counter
    nodeID_counter = 1

    infoID_history_table_size = infoID_history_table.shape[0]

    # #make cur new user dict
    # new_user_df_list = []

    all_dfs_for_cur_infoID_platform_pair = []

    print()
    print(platform)
    print(infoID)
    for i,test_date in enumerate(test_dates):
        test_date = pd.to_datetime(test_date, utc=True)
        print("\nDate %d: %s" %((i+1), str(test_date)))

        #cur date dfs
        cur_date_dfs = []
        #=================================== OLD USERS ===================================
        #first get old child users
        num_old_child_users = main_module_dict["module2_pred_dict"][platform]["old"][infoID]["child"][i]
        num_old_child_users = int(num_old_child_users)
        print("num_old_child_users: %d" %num_old_child_users)

        #sample user attribute rows
        # num_old_child_users = 10
        try:
            sampled_weight_rows = infoID_weight_df.sample(n=num_old_child_users,weights="user_action_proba" ,replace=False).reset_index(drop=True)
        except:
            sampled_weight_rows = infoID_weight_df.sample(n=num_old_child_users,weights="user_action_proba",replace=True).reset_index(drop=True)
        print(sampled_weight_rows)

        #save for later
        old_user_rows = sampled_weight_rows.copy()

        #=================================== NEW USERS ===================================
        #first get new child users
        num_new_child_users = main_module_dict["module2_pred_dict"][platform]["new"][infoID]["child"][i]
        num_new_child_users = int(num_new_child_users)
        print("num_new_child_users: %d" %num_new_child_users)

        #sample user attribute rows
        sampled_weight_rows = infoID_weight_df.sample(n=num_new_child_users,replace=True).reset_index(drop=True)
        cur_new_user_count+=num_new_child_users
        print(sampled_weight_rows)

        #make new users
        new_users = ["%s_%s_syn_user_%d"%( platform,infoID,j+cur_new_user_count) for j in range(num_new_child_users)]
        sampled_weight_rows["nodeUserID"] = new_users
        print(sampled_weight_rows)

        #save for later
        new_user_rows = sampled_weight_rows.copy()
        #=================================== MATCH USERS TO ACTIONS ===================================

        #users so far
        cur_user_df = pd.concat([new_user_rows, old_user_rows]).reset_index(drop=True)
        total_cur_users = cur_user_df.shape[0]
        print("\ntotal_cur_users: %d" %total_cur_users)

        # #get df for the infoID,platform, and date
        # cur_platform_df = platform_to_infoID_action_count_df_dict[platform]
        # cur_infoID_df = cur_platform_df[cur_platform_df["informationID"]==infoID]
        cur_infoID_date_df = cur_infoID_df[cur_infoID_df["nodeTime"]==test_date].reset_index(drop=True)
        print(cur_infoID_date_df)
        print(cur_user_df)

        #count total actions
        total_actions = cur_infoID_date_df[actions].sum(axis=1)
        total_actions = int(total_actions)
        print("\ntotal_actions: %d" %total_actions)
        print("total_cur_users: %d" %total_cur_users)

        if total_cur_users > total_actions:
            print("\nThere are more users than actions...Downsampling users...")
            total_cur_users = total_actions
            print("total_actions: %d" %total_actions)
            print("total_cur_users: %d" %total_cur_users)
            cur_user_df = cur_user_df.sample(n=total_cur_users,replace=False).reset_index(drop=True)
            print("\nUser df after down sampling:")
            print(cur_user_df)

        if total_cur_users < total_actions:
            print("\nThere are more actions than users...Adding repeat users...")
            diff = total_actions - total_cur_users
            print("diff: %d" %diff)
            repeat_user_rows = cur_user_df.sample(n=diff, weights="user_action_proba",replace=True).reset_index(drop=True)
            cur_user_df = pd.concat([cur_user_df, repeat_user_rows]).reset_index(drop=True)
            print("\ncur_user_df after user upsample")
            print(cur_user_df)
            #verify size
            verify_df_size(cur_user_df, total_actions,"cur_user_df")

        actionTypes = []
        cur_infoID_date_df = cur_infoID_date_df.reset_index(drop=True)
        for action in actions:
            action_count = cur_infoID_date_df[action].iloc[0]
            print("%s count: %d" %(action, action_count))
            actionTypes+=[action for a in range(action_count)]

        #df of users so far
        user_sim_action_df = cur_user_df[["nodeUserID"]]
        user_sim_action_df["nodeTime"] = test_date
        user_sim_action_df = user_sim_action_df[["nodeTime", "nodeUserID"]]
        user_sim_action_df["platform"] = platform
        user_sim_action_df["informationID"] = infoID
        user_sim_action_df["actionType"] = actionTypes
        print("\nuser_sim_action_df")
        print(user_sim_action_df)
        # sys.exit(0)
            #sample users

        #size to enforce
        USER_SIM_ACTION_DF_ENFORCED_SIZE = user_sim_action_df.shape[0]
        print("\nUSER_SIM_ACTION_DF_ENFORCED_SIZE: %d" %USER_SIM_ACTION_DF_ENFORCED_SIZE)

        #verify size
        verify_df_size(user_sim_action_df, USER_SIM_ACTION_DF_ENFORCED_SIZE,"user_sim_action")


        #=================================== GET USER INTERSECTION ===================================
        kept_users = list(cur_user_df["nodeUserID"])
        final_new_user_rows = new_user_rows[new_user_rows["nodeUserID"].isin(kept_users)]
        print("\nfinal_new_user_rows")
        print(final_new_user_rows)

        #=================================== UPDATE WEIGHT TABLE! ===================================
        #get sizes for checking
        weight_table_size = infoID_weight_df.shape[0]
        num_final_new_user_rows = final_new_user_rows.shape[0]
        total_size = weight_table_size + num_final_new_user_rows

        #grow weight df
        #append to weight df
        infoID_weight_df = pd.concat([infoID_weight_df, final_new_user_rows]).reset_index(drop=True)
        print("\nUpdated weight df")
        print(infoID_weight_df)
        my_new_weight_size = infoID_weight_df.shape[0]
        print("\nmy_new_weight_size and theoretical total_size")
        print(my_new_weight_size)
        print(total_size)
        if my_new_weight_size != total_size:
            print("\nError! my_new_weight_size != total_size")
            sys.exit(0)
        else:
            print("\nmy_new_weight_size == total_size. Continuing!")

        #=================================== SELECT PARENT USERS! ===================================


        #get prop delays
        num_prop_delays = user_sim_action_df.shape[0]
        sampled_prop_delays = infoID_prop_delay_table["prop_delay_from_parent"].sample(n=num_prop_delays, replace=True)
        print(sampled_prop_delays)
        user_sim_action_df["prop_delay_from_parent"] = list(sampled_prop_delays)
        user_sim_action_df["nodeID"] = ["%s_%s_nodeID_%d"%(platform,infoID, n_idx + nodeID_counter) for n_idx in range(user_sim_action_df.shape[0])]
        print(user_sim_action_df)

        #now sample parents using the prop delays
        #using each influence proba, we select the nodeTime
        print("\ninfoID_history_table")
        print(infoID_history_table)

        #add to counter
        nodeID_counter+=user_sim_action_df.shape[0]

        #get probas
        cur_user_df = cur_user_df[["nodeUserID","user_infl_proba"]].drop_duplicates("nodeUserID").reset_index(drop=True)
        print("\ncur_user_df")
        print(cur_user_df)

        #merge
        user_sim_action_df = user_sim_action_df.merge(cur_user_df, on="nodeUserID", how="inner").reset_index(drop=True)
        print("\nuser_sim_action_df with weights")
        print(user_sim_action_df)

        #verify size
        verify_df_size(user_sim_action_df, USER_SIM_ACTION_DF_ENFORCED_SIZE,"user_sim_action")

        # print("done")
        # sys.exit(0)

        #first split df by actions
        root_action_df = user_sim_action_df[user_sim_action_df["actionType"].isin(root_actions)].reset_index(drop=True)
        root_action_df["parentUserID"] = root_action_df["nodeUserID"]
        root_action_df["parentID"] = root_action_df["nodeID"]
        root_action_df["rootID"] = root_action_df["nodeID"]

        #response df
        #shuffle
        response_action_df = user_sim_action_df[~user_sim_action_df["actionType"].isin(root_actions)].reset_index(drop=True)
        response_action_df = response_action_df.sample(frac=1).reset_index(drop=True)
        print("\nroot_action_df")
        print(root_action_df)
        print("\nresponse_action_df")
        print(response_action_df)

        #make cur df list
        root_action_df = root_action_df.drop("prop_delay_from_parent", axis=1)

        #might have to delete
        #make sure history table grows by the proper amount
        infoID_history_table = pd.concat([infoID_history_table, root_action_df])
        infoID_history_table_size = infoID_history_table_size+root_action_df.shape[0]
        verify_df_size(infoID_history_table, infoID_history_table_size,"infoID_history_table")

        # print("done")
        # sys.exit(0)

        #get unique prop delays
        response_action_df["prop_delay_from_parent"] = pd.to_timedelta(response_action_df["prop_delay_from_parent"])
        unique_prop_delays = list(response_action_df["prop_delay_from_parent"].unique())
        print("\nunique_prop_delays")
        print(unique_prop_delays)
        # sys.exit(0)

        #GET PARENT IDs/parent users
        parent_users = []
        parentIDs = []
        rootIDs = []

        #save here
        # new_dfs = [root_action_df]
        cur_date_dfs.append(root_action_df)
        for prop_delay in unique_prop_delays:

            #get relevant history
            if prop_delay == np.timedelta64(0,'ns'):
                # continue
                print("\nTIME DELAY IS ZERO, USING SAME PERIOD DATA.")
                relevant_history = root_action_df.copy()
                print("\nrelevant_history")
                print(relevant_history)
            else:
                desired_nodeTime = test_date - prop_delay
                print("\ndesired_nodeTime")
                print(desired_nodeTime)
                relevant_history = infoID_history_table[infoID_history_table["nodeTime"]==desired_nodeTime]
                print("\nrelevant_history")
                print(relevant_history)
                # sys.exit(0)
            response_cols = list(response_action_df)
            relevant_history_cols = list(relevant_history)
            print("\nresponse_cols")
            print(response_cols)
            print("\nrelevant_history_cols")
            print(relevant_history_cols)

            cur_response_df = response_action_df[response_action_df["prop_delay_from_parent"]==prop_delay].reset_index(drop=True)
            print("\ncur_response_df")
            print(cur_response_df)
            cur_response_df = cur_response_df.drop("prop_delay_from_parent", axis=1)
            col_order = list(relevant_history)
            num_rows = cur_response_df.shape[0]
            # sys.exit(0)

            for row_idx,row in cur_response_df.iterrows():
                print("Cur idx: %d of %d"%(row_idx, num_rows))
                # print(row)

                row_df = row.to_frame().T.reset_index(drop=True)
                # print("\nrow_df")
                # print(row_df)

                #sample parent
                try:
                    sampled_parent_row = relevant_history.sample(n=1,weights="user_infl_proba").reset_index(drop=True)
                except ValueError:
                    sampled_parent_row = relevant_history.sample(n=1).reset_index(drop=True)
                # print("\nsampled_parent_row")
                # print(sampled_parent_row)

                #add parent records
                row_df["parentID"] = sampled_parent_row["nodeID"]
                row_df["parentUserID"] = sampled_parent_row["nodeUserID"]
                row_df["rootID"] = sampled_parent_row["rootID"]
                row_df = row_df[col_order]
                # new_dfs.append(row_df)
                cur_date_dfs.append(row_df)

                #add this to history
                infoID_history_table = pd.concat([infoID_history_table, row_df]).reset_index(drop=True)
                # print("\nUpdated history table")
                # print(history_table)
            # sys.exit(0)

            # infoID_history_table = pd.concat([infoID_history_table, root_action_df])
            infoID_history_table_size = infoID_history_table_size+cur_response_df.shape[0]
            verify_df_size(infoID_history_table, infoID_history_table_size,"infoID_history_table")

        cur_date_df = pd.concat(cur_date_dfs)
        cur_date_df = cur_date_df.reset_index(drop=True)
        print("\ncur_date_df")
        print(cur_date_df)

        #verify size
        verify_df_size(cur_date_df, USER_SIM_ACTION_DF_ENFORCED_SIZE,"cur_date_df")

        all_dfs_for_cur_infoID_platform_pair.append(cur_date_df)
    cur_infoID_platform_df = pd.concat(all_dfs_for_cur_infoID_platform_pair).reset_index(drop=True)
    print("\ncur_infoID_platform_df")
    print(cur_infoID_platform_df)

    end_time = time()
    total_time_in_minutes = (end_time - start_time) / 60.0
    total_time_in_minutes = np.round(total_time_in_minutes, 2)

    time_str = str("%s total_time_in_minutes: %.2f"%total_time_in_minutes)
    print(time_str)
    time_fp = output_dir + "time.txt"
    with open(time_fp, "w") as f:
        f.write(time_str)
    print(time_fp)

    data_fp = output_dir + "cur_infoID_platform_df.csv"
    cur_infoID_platform_df.to_csv(data_fp, index=False)
    print(data_fp)


    return data_fp

def verify_df_size(df, size_to_enforce,tag):
    print("\nVerifying %s df size"%tag)
    print(df.shape[0])
    print(size_to_enforce)
    if df.shape[0] != size_to_enforce:
        print("\nError! %s df should be of size %d but it is %d!"%(tag ,size_to_enforce, df.shape[0]))
        sys.exit(0)
    else:
        print("Df size is ok! Continuing...")

def get_platform_to_infoID_action_count_df_dict_v2_split_action_module1(output_gran,infoIDs, platforms,test_start, test_end, test_dates, user_statuses,user_types,platform_to_action_dict,main_module_dict,action_to_gran_dict):

    #get all actions for later
    all_actions = []
    for platform in platforms:
        all_actions+=platform_to_action_dict[platform]

    platform_to_action_count_df_dict = {}

    #make action df
    for platform in platforms:
        all_dfs = []
        infoID_action_pred_dict = main_module_dict["module1_%s_pred_dict"%platform]
        actions = platform_to_action_dict[platform]
        for infoID in infoIDs:
            cur_action_count_dict = {}
            for action in actions:
                action_vector = infoID_action_pred_dict[infoID][action].flatten()
                # if action == "youtube_video":
                #     print("\nyoutube_video")
                #     print(action_vector)
                #     sys.exit(0)
                cur_action_count_dict[action] = action_vector

            #print(cur_action_count_dict["twitter_tweet"].shape)
            cur_action_count_df = pd.DataFrame(data=cur_action_count_dict)
            print(cur_action_count_df)

            input_gran = action_to_gran_dict[action]

            # if platform == "youtube":
            #     input_gran = module1_youtube_gran
            # else:
            #     input_gran = module1_twitter_gran

            #make infoID df
            cur_action_count_df = convert_action_df_to_daily_gran(cur_action_count_df, input_gran,output_gran,test_dates, test_start, test_end)
            # cur_action_count_df["platform"] = platform
            cur_action_count_df["informationID"] = infoID
            # cur_action_count_df = cur_action_count_df[["nodeTime", "platform", "informationID"] + all_actions]
            print("\nprint(cur_action_count_df)")
            print(cur_action_count_df)
            all_dfs.append(cur_action_count_df)

        #make final df
        infoID_action_df = pd.concat(all_dfs).sort_values("nodeTime").reset_index(drop=True)
        infoID_action_df = infoID_action_df[["nodeTime", "informationID"] + actions].fillna(0)

        #fix counts
        for action in actions:
           infoID_action_df[action] = np.round(infoID_action_df[action], 0).astype("int32")
           print(infoID_action_df[action])

        platform_to_action_count_df_dict[platform] = infoID_action_df
        # infoID_action_df = infoID_action_df[["nodeTime", "platform", "informationID"] + actions]
        print(platform_to_action_count_df_dict)

    print("\nplatform_to_action_count_df_dict")
    print(platform_to_action_count_df_dict)
    return platform_to_action_count_df_dict

def get_platform_to_infoID_action_count_df_dict(output_gran,infoIDs, platforms,test_start, test_end, test_dates, module1_twitter_gran,module1_youtube_gran,user_statuses,user_types,platform_to_action_dict,main_module_dict):

    #get all actions for later
    all_actions = []
    for platform in platforms:
        all_actions+=platform_to_action_dict[platform]

    platform_to_action_count_df_dict = {}

    #make action df
    for platform in platforms:
        all_dfs = []
        infoID_action_pred_dict = main_module_dict["module1_%s_pred_dict"%platform]
        actions = platform_to_action_dict[platform]
        for infoID in infoIDs:
            cur_action_count_dict = {}
            for action in actions:
                action_vector = infoID_action_pred_dict[infoID][action].flatten()
                # if action == "youtube_video":
                #     print("\nyoutube_video")
                #     print(action_vector)
                #     sys.exit(0)
                cur_action_count_dict[action] = action_vector

            #print(cur_action_count_dict["twitter_tweet"].shape)
            cur_action_count_df = pd.DataFrame(data=cur_action_count_dict)
            print(cur_action_count_df)

            if platform == "youtube":
                input_gran = module1_youtube_gran
            else:
                input_gran = module1_twitter_gran

            #make infoID df
            cur_action_count_df = convert_action_df_to_daily_gran(cur_action_count_df, input_gran,output_gran,test_dates, test_start, test_end)
            # cur_action_count_df["platform"] = platform
            cur_action_count_df["informationID"] = infoID
            # cur_action_count_df = cur_action_count_df[["nodeTime", "platform", "informationID"] + all_actions]
            print("\nprint(cur_action_count_df)")
            print(cur_action_count_df)
            all_dfs.append(cur_action_count_df)

        #make final df
        infoID_action_df = pd.concat(all_dfs).sort_values("nodeTime").reset_index(drop=True)
        infoID_action_df = infoID_action_df[["nodeTime", "informationID"] + actions].fillna(0)

        #fix counts
        for action in actions:
           infoID_action_df[action] = np.round(infoID_action_df[action], 0).astype("int32")
           print(infoID_action_df[action])

        platform_to_action_count_df_dict[platform] = infoID_action_df
        # infoID_action_df = infoID_action_df[["nodeTime", "platform", "informationID"] + actions]
        print(platform_to_action_count_df_dict)

    print("\nplatform_to_action_count_df_dict")
    print(platform_to_action_count_df_dict)
    return platform_to_action_count_df_dict

def convert_action_df_to_daily_gran(cur_action_count_df, input_gran,output_gran,output_dates, test_start, test_end):

    #get agg cols
    agg_cols = list(cur_action_count_df)

    #put in nodeTime
    input_dates = pd.date_range(test_start, test_end, freq=input_gran)
    cur_action_count_df["nodeTime"] = input_dates
    cur_action_count_df["nodeTime"] = pd.to_datetime(cur_action_count_df["nodeTime"], utc=True)
    # if input_gran == "D":
    #   return cur_action_count_df

    #floor it
    cur_action_count_df["nodeTime"] = cur_action_count_df["nodeTime"].dt.floor(output_gran)
    for col in agg_cols:
        cur_action_count_df[col] = cur_action_count_df.groupby(["nodeTime"])[col].transform("sum")
    cur_action_count_df = cur_action_count_df.drop_duplicates().reset_index(drop=True)
    # print(cur_action_count_df)

    return cur_action_count_df

def get_param_dict_from_model_dir(model_result_dir):
    param_fp = model_result_dir + "params.csv"
    param_df = pd.read_csv(param_fp)
    param_dict = convert_df_2_cols_to_dict(param_df, "param", "value")
    return param_dict

def get_gran_from_model_dir(model_result_dir):
    param_dict = get_param_dict_from_model_dir(model_result_dir)
    GRAN = param_dict["GRAN"]
    return GRAN

def merge_mult_dfs(merge_list, on, how):
    print("\nMerging multiple dfs...")
    return reduce(lambda  left,right: pd.merge(left,right,on=on, how=how), merge_list)

def get_user_category_nunique_user_counts(df, GRAN,start,end):

    #get parents
    temp = df.copy()
    total_records = temp.shape[0]
    print("\nGetting parents...")
    temp = get_parentUserID_col_with_platform(temp)

    #fix user names
    temp["original_nodeUserID"] = temp["nodeUserID"].copy()
    temp["original_parentUserID"] = temp["parentUserID"].copy()
    # temp["nodeUserID"] = temp["nodeUserID"] + "<with>" + temp["platform"]

    #mark first user appearances
    temp = mark_user_first_appearances(temp, GRAN, mark_parents=True)
    print(temp)

    #mark if absolutely new
    temp = temp.rename(columns={"is_new_child_user":"is_abs_new_child_user","is_new_parent_user":"is_abs_new_parent_user"})

    #rename nodeUserIDs
    temp["nodeUserID"] = temp["nodeUserID"] + "<with>" + temp["informationID"]
    temp["parentUserID"] = temp["parentUserID"] + "<with>" + temp["informationID"]

    #mark first user appearances for user-infoID pairs
    temp = mark_user_first_appearances(temp, GRAN, mark_parents=True)

    #rename columns
    temp = temp.rename(columns={"is_new_child_user":"is_new_child_user_infoID_pair","is_new_parent_user":"is_new_parent_user_infoID_pair"})
    temp = temp.rename(columns={"is_abs_new_child_user":"is_new_child_user","is_abs_new_parent_user":"is_new_parent_user"})
    user_cols = ["is_new_child_user","is_new_parent_user","is_new_child_user_infoID_pair","is_new_parent_user_infoID_pair"]
    print(temp[user_cols])

    #every child and parent should be accounted for
    child_user_set = set(temp["nodeUserID"])
    parent_user_set = set(temp["parentUserID"])
    nunique_child_users = len(child_user_set)
    nunique_parent_users = len(parent_user_set)
    print("\nnunique_child_users: %d" %nunique_child_users)
    print("nunique_parent_users: %d" %nunique_parent_users)


    #drop duplicates
    # drop_dupe_cols = ["nodeTime", "nodeID"] + user_cols
    # temp = temp[drop_dupe_cols].drop_duplicates()
    # print(temp)

    #for checking later
    temp["nunique_child_users_over_time"] = temp.groupby(["nodeTime"])["nodeUserID"].transform("nunique")
    temp["nunique_parent_users_over_time"] = temp.groupby(["nodeTime"])["parentUserID"].transform("nunique")
    child_temp = temp[["nodeTime", "nunique_child_users_over_time"]].drop_duplicates()
    nunique_child_users_over_time = child_temp["nunique_child_users_over_time"].sum()
    parent_temp = temp[["nodeTime", "nunique_parent_users_over_time"]].drop_duplicates()
    nunique_parent_users_over_time = parent_temp["nunique_parent_users_over_time"].sum()
    print("\nnunique_child_users_over_time")
    print(nunique_child_users_over_time)
    print("\nnunique_parent_users_over_time")
    print(nunique_parent_users_over_time)


    #rename category 0
    temp["is_old_child_user"] = (~(temp["is_new_child_user"].astype(bool).copy())).astype("int32")
    print(temp[["is_old_child_user","is_new_child_user"]])
    temp["is_old_parent_user"] = (~(temp["is_new_parent_user"].astype(bool).copy())).astype("int32")
    print(temp[["is_old_parent_user","is_new_parent_user"]])

    #rename category 1
    temp["old_child_new_infoID"] = temp["is_old_child_user"] * temp["is_new_child_user_infoID_pair"]
    print(temp[["old_child_new_infoID"]])
    temp["old_parent_new_infoID"] = temp["is_old_parent_user"] * temp["is_new_parent_user_infoID_pair"]
    print(temp[["old_parent_new_infoID"]])

    #rename category 2
    temp["old_child_old_infoID"] = (~(temp["is_new_child_user_infoID_pair"].astype(bool).copy())).astype("int32")
    print(temp[["old_child_old_infoID"]])
    temp["old_parent_old_infoID"] = (~(temp["is_new_parent_user_infoID_pair"].astype(bool).copy())).astype("int32")
    print(temp[["old_parent_old_infoID"]])

    #rename category 3
    temp["new_child_new_infoID"] = temp["is_new_child_user"] * temp["is_new_child_user_infoID_pair"]
    print(temp["new_child_new_infoID"])
    temp["new_parent_new_infoID"] = temp["is_new_parent_user"] * temp["is_new_parent_user_infoID_pair"]
    print(temp["new_parent_new_infoID"])

    #get cols
    user_cols = ['old_child_old_infoID','old_parent_old_infoID',
        "old_child_new_infoID","old_parent_new_infoID","new_child_new_infoID","new_parent_new_infoID"]



    ##################################### get nunique users #####################################

    #get date df
    blank_date_df = create_blank_date_df(start,end,GRAN)

    #save dfs here
    all_nunique_dfs = [blank_date_df]


    #make nunique cols
    nunique_cols = []

    #track counts
    my_nunique_child_users_over_time = 0
    my_nunique_parent_users_over_time = 0

    for col in user_cols:
        nunique_col = "nunique_" + col
        print(nunique_col)
        temp_user_df = temp[["nodeTime", "nodeUserID", "parentUserID", col]].copy()
        temp_user_df = temp_user_df[temp_user_df[col]==1]
        nunique_cols.append(nunique_col)
        if "child" in col:
            groupby_col = "nodeUserID"
        else:
            groupby_col = "parentUserID"
        temp_user_df[nunique_col] = temp_user_df.groupby(["nodeTime"])[groupby_col].transform("nunique")
        temp_user_df = temp_user_df[["nodeTime", nunique_col]].drop_duplicates().reset_index(drop=True)
        print(temp_user_df)
        all_nunique_dfs.append(temp_user_df)

        # if "child" in col:
        #     groupby_col = "nodeUserID"
        # else:
        #     groupby_col = "parentUserID"


    #merge dfs
    temp = merge_mult_dfs(all_nunique_dfs, on="nodeTime", how="outer")
    temp = temp.fillna(0)
    temp = temp.drop_duplicates().reset_index(drop=True)
    print("\nFull nunique df")
    print(temp)

    #check counts
    for col in nunique_cols:
        if "child" in col:
            my_nunique_child_users_over_time+=temp[col].sum()
        else:
            my_nunique_parent_users_over_time+=temp[col].sum()

    print("\nChecking child counts...")
    print(my_nunique_child_users_over_time)
    print(nunique_child_users_over_time)
    if my_nunique_child_users_over_time != nunique_child_users_over_time:
        print("\nError!my_nunique_child_users_over_time != nunique_child_users_over_time")
        sys.exit(0)
    else:
        print("\nmy_nunique_child_users_over_time == nunique_child_users_over_time")
        print("Counts are ok!")

    print("\nChecking parent counts...")
    print(my_nunique_parent_users_over_time)
    print(nunique_parent_users_over_time)
    if my_nunique_parent_users_over_time != nunique_parent_users_over_time:
        print("\nError!my_nunique_parent_users_over_time != nunique_parent_users_over_time")
        sys.exit(0)
    else:
        print("\nmy_nunique_parent_users_over_time == nunique_parent_users_over_time")
        print("Counts are ok!")

    # sys.exit(0)

    # #make df
    # for col in nunique_cols:
    #     temp[col] = temp.groupby(["nodeTime"])[col].transform("sum")
    # temp = temp[["nodeTime"] + user_cols]
    # temp = temp.drop_duplicates().reset_index(drop=True)
    # print(temp)

    #add up all child users
    # my_child_user_set = set(list(temp["nodeUserID"]))
    # my_parent_user_set = set(list(temp["parentUserID"]))
    # my_nunique_child_users = len(my_child_user_set)
    # my_nunique_parent_users = len(my_parent_user_set)

    # print()
    # print(my_nunique_child_users)
    # print(nunique_child_users)
    # if my_nunique_child_users != nunique_child_users:
    #    print("\nError! my_nunique_child_users != nunique_child_users")
    #    sys.exit(0)
    # else:
    #     print("\nmy_nunique_child_users == nunique_child_users")
    #     print("Counts are ok!")

    # print()
    # print(my_nunique_parent_users)
    # print(nunique_parent_users)
    # if my_nunique_parent_users != nunique_parent_users:
    #    print("\nError! my_nunique_parent_users != nunique_parent_users")
    #    sys.exit(0)
    # else:
    #     print("\nmy_nunique_parent_users == nunique_parent_users")
    #     print("Counts are ok!")

    # new_user_cols = []
    # for col in user_cols:
    #     new_col = "nunique_" + col
    #     new_user_cols.append(new_col)
    #     temp = temp.rename(columns={col: new_col})
    # user_cols = new_user_cols




    return temp, ["nodeTime"] + user_cols

def get_user_category_action_counts(df, GRAN):

    #get parents
    temp = df.copy()
    total_records = temp.shape[0]
    print("\nGetting parents...")
    temp = get_parentUserID_col_with_platform(temp)

    #fix user names
    temp["original_nodeUserID"] = temp["nodeUserID"].copy()
    temp["original_parentUserID"] = temp["parentUserID"].copy()
    # temp["nodeUserID"] = temp["nodeUserID"] + "<with>" + temp["platform"]

    #mark first user appearances
    temp = mark_user_first_appearances(temp, GRAN, mark_parents=True)
    print(temp)

    #mark if absolutely new
    temp = temp.rename(columns={"is_new_child_user":"is_abs_new_child_user","is_new_parent_user":"is_abs_new_parent_user"})

    #rename nodeUserIDs
    temp["nodeUserID"] = temp["nodeUserID"] + "<with>" + temp["informationID"]
    temp["parentUserID"] = temp["parentUserID"] + "<with>" + temp["informationID"]

    #mark first user appearances for user-infoID pairs
    temp = mark_user_first_appearances(temp, GRAN, mark_parents=True)

    #rename columns
    temp = temp.rename(columns={"is_new_child_user":"is_new_child_user_infoID_pair","is_new_parent_user":"is_new_parent_user_infoID_pair"})
    temp = temp.rename(columns={"is_abs_new_child_user":"is_new_child_user","is_abs_new_parent_user":"is_new_parent_user"})
    user_cols = ["is_new_child_user","is_new_parent_user","is_new_child_user_infoID_pair","is_new_parent_user_infoID_pair"]
    print(temp[user_cols])

    #get counts of each user type
    num_new_children = temp["is_new_child_user"].sum()
    num_old_children = len(temp) - num_new_children
    num_new_parents = temp["is_new_parent_user"].sum()
    num_old_parents = len(temp) - num_new_parents


    #rename category 0
    temp["is_old_child_user"] = (~(temp["is_new_child_user"].astype(bool).copy())).astype("int32")
    print(temp[["is_old_child_user","is_new_child_user"]])
    temp["is_old_parent_user"] = (~(temp["is_new_parent_user"].astype(bool).copy())).astype("int32")
    print(temp[["is_old_parent_user","is_new_parent_user"]])

    #rename category 1
    temp["old_child_new_infoID"] = temp["is_old_child_user"] * temp["is_new_child_user_infoID_pair"]
    print(temp[["old_child_new_infoID"]])
    temp["old_parent_new_infoID"] = temp["is_old_parent_user"] * temp["is_new_parent_user_infoID_pair"]
    print(temp[["old_parent_new_infoID"]])

    #rename category 2
    temp["old_child_old_infoID"] = (~(temp["is_new_child_user_infoID_pair"].astype(bool).copy())).astype("int32")
    print(temp[["old_child_old_infoID"]])
    temp["old_parent_old_infoID"] = (~(temp["is_new_parent_user_infoID_pair"].astype(bool).copy())).astype("int32")
    print(temp[["old_parent_old_infoID"]])

    #rename category 3
    temp["new_child_new_infoID"] = temp["is_new_child_user"] * temp["is_new_child_user_infoID_pair"]
    print(temp["new_child_new_infoID"])
    temp["new_parent_new_infoID"] = temp["is_new_parent_user"] * temp["is_new_parent_user_infoID_pair"]
    print(temp["new_parent_new_infoID"])

    #get cols
    user_cols = ['old_child_old_infoID','old_parent_old_infoID',
        "old_child_new_infoID","old_parent_new_infoID","new_child_new_infoID","new_parent_new_infoID"]

    #make df
    for col in user_cols:
        temp[col] = temp.groupby(["nodeTime"])[col].transform("sum")
    temp = temp[["nodeTime"] + user_cols]
    temp = temp.drop_duplicates().reset_index(drop=True)
    print(temp)

    print()
    # total_records = temp.shape[0]
    old_child_new_infoID_sum = temp["old_child_new_infoID"].sum()
    old_child_old_infoID_sum = temp["old_child_old_infoID"].sum()
    new_child_new_infoID_sum = temp["new_child_new_infoID"].sum()
    my_total_records = old_child_new_infoID_sum + old_child_old_infoID_sum + new_child_new_infoID_sum
    print(my_total_records)
    print(total_records)
    if my_total_records != total_records:
        print("my_total_records != total_records")
        sys.exit(0)

    new_user_cols = []
    for col in user_cols:
        new_col = "num_actions_" + col
        new_user_cols.append(new_col)
        temp = temp.rename(columns={col: new_col})
    user_cols = new_user_cols


    return temp, ["nodeTime"] + user_cols

def get_user_category_action_counts_backup(df, GRAN):

    #get parents
    temp = df.copy()
    total_records = temp.shape[0]
    print("\nGetting parents...")
    temp = get_parentUserID_col_with_platform(temp)

    #fix user names
    temp["original_nodeUserID"] = temp["nodeUserID"].copy()
    temp["original_parentUserID"] = temp["parentUserID"].copy()
    # temp["nodeUserID"] = temp["nodeUserID"] + "<with>" + temp["platform"]

    #mark first user appearances
    temp = mark_user_first_appearances(temp, GRAN, mark_parents=True)
    print(temp)

    #mark if absolutely new
    temp = temp.rename(columns={"is_new_child_user":"is_abs_new_child_user","is_new_parent_user":"is_abs_new_parent_user"})

    #rename nodeUserIDs
    temp["nodeUserID"] = temp["nodeUserID"] + "<with>" + temp["informationID"]
    temp["parentUserID"] = temp["parentUserID"] + "<with>" + temp["informationID"]

    #mark first user appearances for user-infoID pairs
    temp = mark_user_first_appearances(temp, GRAN, mark_parents=True)

    #rename columns
    temp = temp.rename(columns={"is_new_child_user":"is_new_child_user_infoID_pair","is_new_parent_user":"is_new_parent_user_infoID_pair"})
    temp = temp.rename(columns={"is_abs_new_child_user":"is_new_child_user","is_abs_new_parent_user":"is_new_parent_user"})
    user_cols = ["is_new_child_user","is_new_parent_user","is_new_child_user_infoID_pair","is_new_parent_user_infoID_pair"]
    print(temp[user_cols])

    #get counts of each user type
    num_new_children = temp["is_new_child_user"].sum()
    num_old_children = len(temp) - num_new_children
    num_new_parents = temp["is_new_parent_user"].sum()
    num_old_parents = len(temp) - num_new_parents


    #rename category 0
    temp["is_old_child_user"] = (~(temp["is_new_child_user"].astype(bool).copy())).astype("int32")
    print(temp[["is_old_child_user","is_new_child_user"]])
    temp["is_old_parent_user"] = (~(temp["is_new_parent_user"].astype(bool).copy())).astype("int32")
    print(temp[["is_old_parent_user","is_new_parent_user"]])

    #rename category 1
    temp["old_child_new_infoID"] = temp["is_old_child_user"] * temp["is_new_child_user_infoID_pair"]
    print(temp[["old_child_new_infoID"]])
    temp["old_parent_new_infoID"] = temp["is_old_parent_user"] * temp["is_new_parent_user_infoID_pair"]
    print(temp[["old_parent_new_infoID"]])

    #rename category 2
    temp["old_child_old_infoID"] = (~(temp["is_new_child_user_infoID_pair"].astype(bool).copy())).astype("int32")
    print(temp[["old_child_old_infoID"]])
    temp["old_parent_old_infoID"] = (~(temp["is_new_parent_user_infoID_pair"].astype(bool).copy())).astype("int32")
    print(temp[["old_parent_old_infoID"]])

    #rename category 3
    temp["new_child_new_infoID"] = temp["is_new_child_user"] * temp["is_new_child_user_infoID_pair"]
    print(temp["new_child_new_infoID"])
    temp["new_parent_new_infoID"] = temp["is_new_parent_user"] * temp["is_new_parent_user_infoID_pair"]
    print(temp["new_parent_new_infoID"])

    #get cols
    user_cols = ['old_child_old_infoID','old_parent_old_infoID',
        "old_child_new_infoID","old_parent_new_infoID","new_child_new_infoID","new_parent_new_infoID"]

    #make df
    for col in user_cols:
        temp[col] = temp.groupby(["nodeTime"])[col].transform("sum")
    temp = temp[["nodeTime"] + user_cols]
    temp = temp.drop_duplicates().reset_index(drop=True)
    print(temp)

    # #check counts
    # print()
    # my_old_children_sum = temp["is_old_child_user"].sum()
    # print(my_old_children_sum)
    # print(num_old_children)
    # if my_old_children_sum != num_old_children:
    #   print("Error! my_old_children_sum != num_old_children")
    #   sys.exit(0)
    # else:
    #   print("my_old_children_sum == num_old_children")
    #   print("Count is ok!")

    # #check counts
    # print()
    # my_new_children_sum = temp["is_new_child_user"].sum()
    # print(my_new_children_sum)
    # print(num_new_children)
    # if my_new_children_sum != num_new_children:
    #   print("Error! my_new_children_sum != num_new_children")
    #   sys.exit(0)
    # else:
    #   print("my_old_children_sum == num_old_children")
    #   print("Count is ok!")

    # #check counts
    # print()
    # my_old_parent_sum = temp["is_old_parent_user"].sum()
    # print(my_old_parent_sum)
    # print(num_old_parents)
    # if my_old_parent_sum != num_old_parents:
    #   print("Error! my_old_parent_sum != num_old_parents")
    #   sys.exit(0)
    # else:
    #   print("my_old_parent_sum == num_old_parents")
    #   print("Count is ok!")

    # #check counts
    # print()
    # my_new_parent_sum = temp["is_new_parent_user"].sum()
    # print(my_new_parent_sum)
    # print(num_new_parents)
    # if my_new_parent_sum != num_new_parents:
    #   print("Error! my_new_parent_sum != num_new_parents")
    #   sys.exit(0)
    # else:
    #   print("my_new_parent_sum == num_new_parents")
    #   print("Count is ok!")

    print()
    # total_records = temp.shape[0]
    old_child_new_infoID_sum = temp["old_child_new_infoID"].sum()
    old_child_old_infoID_sum = temp["old_child_old_infoID"].sum()
    new_child_new_infoID_sum = temp["new_child_new_infoID"].sum()
    my_total_records = old_child_new_infoID_sum + old_child_old_infoID_sum + new_child_new_infoID_sum
    print(my_total_records)
    print(total_records)
    if my_total_records != total_records:
        print("my_total_records != total_records")
        sys.exit(0)

    new_user_cols = []
    for col in user_cols:
        new_col = "num_actions_" + col
        new_user_cols.append(new_col)
        temp = temp.rename(columns={col: new_col})
    user_cols = new_user_cols


    return temp, ["nodeTime"] + user_cols

#get root function
def get_parentUserID_col_with_platform(df):

    print("Making nodeID to user dict...")
    nodeID_to_nodeUserID_dict = pd.Series(df.nodeUserID.values,index=df.nodeID).to_dict()
    print("Made nodeID dict. Num elements: %d" %len(nodeID_to_nodeUserID_dict.keys()))

    #I want to have a dict where I can put in the parentID  and get the parentIDUser
    parentIDs = list(df["parentID"])
    platforms = list(df["platform"])
    # infoIDs = list(df["informationID"])
    parentUserIDs = []
    print("Getting parentIDUsers...")
    for i,parentID in enumerate(parentIDs):
        platform = platforms[i]
        try:
            parentUserID = nodeID_to_nodeUserID_dict[parentID]
        except:
            parentUserID = "missing_parentUserID<with>" + platform
            # parentUserID = parentUserID + concat_tag + infoID

        parentUserIDs.append(parentUserID)
    df["parentUserID"] = parentUserIDs
    print(df["parentUserID"])

    return df

def f1_score_get_quartet_result_df(quartet_pred_dict,quartet_train_and_test_array_dict,platforms, user_statuses, infoIDs, user_types,output_dir):
    platform_result_list = []
    infoID_result_list = []
    user_status_list = []
    user_type_list = []

    # rmse_list = []
    # nrmse_list = []
    # actual_list = []
    # pred_list = []
    f1_scores = []
    y_test_zero_count_list = []
    y_test_nonzero_count_list = []
    y_pred_zero_count_list = []
    y_pred_nonzero_count_list = []

    num_quartets = len(infoIDs) * len(platforms) * len(user_statuses) * len(user_statuses)
    print("\nnum_quartets: %d" %num_quartets)

    i = 1
    for platform in platforms:
        for user_status in user_statuses:
            for infoID in infoIDs:
                for user_type in user_types:
                    platform_result_list.append(platform)
                    user_status_list.append(user_status)
                    user_type_list.append(user_type)
                    infoID_result_list.append(infoID)

                    #get pred and gt
                    y_test =quartet_train_and_test_array_dict[platform][user_status][infoID][user_type]["y_test"]
                    y_pred = quartet_pred_dict[platform][user_status][infoID][user_type]
                    y_test = y_test.flatten()
                    y_pred = y_pred.flatten()
                    y_test[y_test<0.5] = 0
                    y_test[y_test>=0.5] = 1
                    y_pred[y_pred<0.5] = 0
                    y_pred[y_pred>=0.5] = 1

                    #num elements
                    num_elements = y_test.shape[0]

                    #y_test counts
                    y_test_num_nonzero_elements = np.count_nonzero(y_test)
                    y_test_num_zero_elements = num_elements - y_test_num_nonzero_elements
                    y_test_nonzero_count_list.append(y_test_num_nonzero_elements)
                    y_test_zero_count_list.append(y_test_num_zero_elements)

                    #y_test counts
                    y_pred_num_nonzero_elements = np.count_nonzero(y_pred)
                    y_pred_num_zero_elements = num_elements - y_pred_num_nonzero_elements
                    y_pred_nonzero_count_list.append(y_pred_num_nonzero_elements)
                    y_pred_zero_count_list.append(y_pred_num_zero_elements)

                    cur_f1_score = f1_score(y_test, y_pred, average='binary')
                    f1_scores.append(cur_f1_score)


                    # #get volumes
                    # pred_volume = np.sum(y_pred)
                    # actual_volume = np.sum(y_test)
                    # actual_list.append(actual_volume)
                    # pred_list.append(pred_volume)

                    #rmse
                    # cur_rmse = mean_squared_error(y_test, y_pred, squared=False)
                    # rmse_list.append(cur_rmse)

                    # cur_nrmse = nunique_users_over_time_nrmse(y_test, y_pred)
                    # nrmse_list.append(cur_nrmse)


                    #mape
                    # cur_mape = mean_absolute_percentage_error(y_test, y_pred)
                    # mape_list.append(cur_mape)
                    i+=1
                    print("Got quartet score record %d of %d" %(i, num_quartets))

    #make df
    data = {"platform":platform_result_list, "user_status":user_status_list,
        "informationID":infoID_result_list, "user_type" : user_type_list, "f1":f1_scores,
        "y_test_num_nz":y_test_nonzero_count_list, "y_test_num_zeros":y_test_zero_count_list,
        "y_pred_num_nz":y_pred_nonzero_count_list, "y_pred_num_zeros":y_pred_zero_count_list,
        }
    result_df = pd.DataFrame(data=data)

    # result_df["nrmse_over_time"] = result_df["nrmse_over_time"].fillna(0)

    col_order = ["platform","user_status","informationID","user_type","f1","y_test_num_nz","y_test_num_zeros","y_pred_num_nz" ,"y_pred_num_zeros"]
    result_df = result_df[col_order]
    result_df = result_df.sort_values("f1", ascending=False).reset_index(drop=True)
    print(result_df)
    output_fp = output_dir + "f1-results.csv"
    result_df.to_csv(output_fp, index=False)
    print(output_fp)

    # total_nrmse_over_time = np.sum(result_df["nrmse_over_time"])
    # print("\ntotal_nrmse_over_time: %.4f"%total_nrmse_over_time)
    # with open(output_dir + "total_nrmse_over_time.txt","w") as f:
    #     f.write(str(total_nrmse_over_time))

    # total_rmse_over_time = np.sum(rmse_list)
    # print("\ntotal_rmse_over_time: %.4f"%total_rmse_over_time)
    # with open(output_dir + "total_rmse_over_time.txt","w") as f:
    #     f.write(str(total_rmse_over_time))


    return result_df

def get_infoID_nunique_users_for_user_status_and_user_type(quartet_result_df, user_type, user_status, output_dir):

    result_df = quartet_result_df.copy()
    result_df = result_df[result_df["user_status"]==user_status]
    result_df = result_df[result_df["user_type"]==user_type].reset_index(drop=True)

    if output_dir != None:
        output_fp = output_dir + "%s-%s-filtered-results.csv"%(user_status, user_type)
        result_df.to_csv(output_fp, index=False)
        print("\nSaved %s"%output_fp)
    return result_df

def compare_results_to_baseline(nn_quartet_result_df, baseline_quartet_result_df, output_dir,base_tag,nn_tag,rename_cols,compare_col,merge_cols, user_statuses, user_types,kickout_cols=[]):
    print("\nComparing to baseline...")


    #make dir
    comparison_output_dir = output_dir + "Baseline-Comparison/"
    create_output_dir(comparison_output_dir)

    for col in kickout_cols:
        if col in list(nn_quartet_result_df):
            nn_quartet_result_df =nn_quartet_result_df.drop(col, axis=1)
        if col in list(baseline_quartet_result_df):
            baseline_quartet_result_df =baseline_quartet_result_df.drop(col, axis=1)

    for col in rename_cols:
        nn_quartet_result_df = nn_quartet_result_df.rename(columns={col: nn_tag + "_" + col})
        baseline_quartet_result_df = baseline_quartet_result_df.rename(columns={col: base_tag + "_" + col})

    #will use these results for later
    result_type_list = []
    num_wins_list = []
    num_losses_list = []
    num_trials_list = []
    win_freq_list = []
    num_ties_list = []

    #merge
    # merge_cols = ["platform", "user_status", "informationID", "user_type"]
    result_df = pd.merge(nn_quartet_result_df, baseline_quartet_result_df, on=merge_cols, how="inner")
    print(result_df)
    nn_compare_col = nn_tag +"_"+ compare_col
    baseline_compare_col = base_tag +"_"+ compare_col
    result_df = result_df.sort_values(nn_compare_col).reset_index(drop=True)
    print(result_df)
    result_df,result_dict = calculate_winner_v2(result_df, nn_compare_col,baseline_compare_col,metric=compare_col)
    print(result_df)

    baseline_score = result_df[baseline_compare_col].sum()
    nn_score = result_df[nn_compare_col].sum()
    score_fp =output_dir + "NN-vs-Baseline-Scores.txt"
    baseline_str = "%s: %.2f"%(baseline_compare_col, baseline_score)
    nn_str = "%s: %.2f"%(nn_compare_col, nn_score)
    print(baseline_str)
    print(nn_str)
    with open(score_fp, "w") as f:
        f.write(str(baseline_str))
        f.write(str(nn_str))

    #append
    result_type_list.append("all")
    num_wins_list.append(result_dict["num_wins"])
    num_losses_list.append(result_dict["num_losses"])
    num_ties_list.append(result_dict["num_ties"])
    num_trials_list.append(result_dict["total_trials"])
    win_freq_list.append(result_dict["freq"])

    #save it
    output_fp = comparison_output_dir + "All-NN-vs-Baseline-Results.csv"
    result_df.to_csv(output_fp, index=False)
    print(output_fp)

    sub_dfs = []

    for user_status in user_statuses:
        for user_type in user_types:
            temp= get_infoID_nunique_users_for_user_status_and_user_type(result_df, user_type, user_status, output_dir)
            print(temp)
            temp,result_dict = calculate_winner_v2(temp, nn_compare_col,baseline_compare_col,metric=compare_col)

            #append
            result_type_list.append("%s-%s"%(user_status, user_type))
            num_wins_list.append(result_dict["num_wins"])
            num_losses_list.append(result_dict["num_losses"])
            num_ties_list.append(result_dict["num_ties"])
            num_trials_list.append(result_dict["total_trials"])
            win_freq_list.append(result_dict["freq"])


            output_fp = comparison_output_dir + "%s-%s-NN-vs-Baseline-Results.csv"%(user_status, user_type)
            temp.to_csv(output_fp, index=False)
            print(output_fp)
            sub_dfs.append(temp)

    #make df
    data={"result_type":result_type_list, "num_nn_wins_for_%s"%compare_col : num_wins_list,
        "num_nn_losses_for_%s"%compare_col : num_losses_list, "num_ties":num_ties_list,
        "num_trials":num_trials_list, "num_wins_as_freq":win_freq_list}
    breakdown_df = pd.DataFrame(data=data)
    col_order = ["result_type", "num_nn_wins_for_%s"%compare_col,
        "num_nn_losses_for_%s"%compare_col, "num_ties","num_trials","num_wins_as_freq"]
    breakdown_df = breakdown_df[col_order]
    print(breakdown_df)

    output_fp = comparison_output_dir + "Summary-Comparison.csv"
    breakdown_df.to_csv(output_fp, index=False)
    print(output_fp)



    return result_df,breakdown_df,sub_dfs

def compare_results_to_baseline_v2_save_option(nn_quartet_result_df, baseline_quartet_result_df, output_dir,base_tag,nn_tag,rename_cols,compare_col,merge_cols, user_statuses, user_types,kickout_cols=[]):
    print("\nComparing to baseline...")


    #make dir
    if output_dir != None:
        comparison_output_dir = output_dir + "Baseline-Comparison/"
        create_output_dir(comparison_output_dir)

    for col in kickout_cols:
        if col in list(nn_quartet_result_df):
            nn_quartet_result_df =nn_quartet_result_df.drop(col, axis=1)
        if col in list(baseline_quartet_result_df):
            baseline_quartet_result_df =baseline_quartet_result_df.drop(col, axis=1)

    for col in rename_cols:
        nn_quartet_result_df = nn_quartet_result_df.rename(columns={col: nn_tag + "_" + col})
        baseline_quartet_result_df = baseline_quartet_result_df.rename(columns={col: base_tag + "_" + col})

    #will use these results for later
    result_type_list = []
    num_wins_list = []
    num_losses_list = []
    num_trials_list = []
    win_freq_list = []
    num_ties_list = []

    #merge
    # merge_cols = ["platform", "user_status", "informationID", "user_type"]
    result_df = pd.merge(nn_quartet_result_df, baseline_quartet_result_df, on=merge_cols, how="inner")
    print(result_df)
    nn_compare_col = nn_tag +"_"+ compare_col
    baseline_compare_col = base_tag +"_"+ compare_col
    result_df = result_df.sort_values(nn_compare_col).reset_index(drop=True)
    print(result_df)
    result_df,result_dict = calculate_winner_v2(result_df, nn_compare_col,baseline_compare_col,metric=compare_col)
    print(result_df)

    baseline_score = result_df[baseline_compare_col].sum()
    nn_score = result_df[nn_compare_col].sum()

    baseline_str = "%s: %.2f"%(baseline_compare_col, baseline_score)
    nn_str = "%s: %.2f"%(nn_compare_col, nn_score)
    print(baseline_str)
    print(nn_str)
    if output_dir != None:
        score_fp =output_dir + "NN-vs-Baseline-Scores.txt"
        with open(score_fp, "w") as f:
            f.write(str(baseline_str))
            f.write(str(nn_str))

    #append
    result_type_list.append("all")
    num_wins_list.append(result_dict["num_wins"])
    num_losses_list.append(result_dict["num_losses"])
    num_ties_list.append(result_dict["num_ties"])
    num_trials_list.append(result_dict["total_trials"])
    win_freq_list.append(result_dict["freq"])

    #save it
    if output_dir != None:
        output_fp = comparison_output_dir + "All-NN-vs-Baseline-Results.csv"
        result_df.to_csv(output_fp, index=False)
        print(output_fp)

    sub_dfs = []

    for user_status in user_statuses:
        for user_type in user_types:
            temp= get_infoID_nunique_users_for_user_status_and_user_type(result_df, user_type, user_status, output_dir)
            print(temp)
            temp,result_dict = calculate_winner_v2(temp, nn_compare_col,baseline_compare_col,metric=compare_col)

            #append
            result_type_list.append("%s-%s"%(user_status, user_type))
            num_wins_list.append(result_dict["num_wins"])
            num_losses_list.append(result_dict["num_losses"])
            num_ties_list.append(result_dict["num_ties"])
            num_trials_list.append(result_dict["total_trials"])
            win_freq_list.append(result_dict["freq"])

            if output_dir != None:
                output_fp = comparison_output_dir + "%s-%s-NN-vs-Baseline-Results.csv"%(user_status, user_type)
                temp.to_csv(output_fp, index=False)
                print(output_fp)
            sub_dfs.append(temp)

    #make df
    data={"result_type":result_type_list, "num_nn_wins_for_%s"%compare_col : num_wins_list,
        "num_nn_losses_for_%s"%compare_col : num_losses_list, "num_ties":num_ties_list,
        "num_trials":num_trials_list, "num_wins_as_freq":win_freq_list}
    breakdown_df = pd.DataFrame(data=data)
    col_order = ["result_type", "num_nn_wins_for_%s"%compare_col,
        "num_nn_losses_for_%s"%compare_col, "num_ties","num_trials","num_wins_as_freq"]
    breakdown_df = breakdown_df[col_order]
    print(breakdown_df)

    if output_dir != None:
        output_fp = comparison_output_dir + "Summary-Comparison.csv"
        breakdown_df.to_csv(output_fp, index=False)
        print(output_fp)



    return result_df,breakdown_df,sub_dfs

def get_quartet_result_df_1dict_version(quartet_pred_dict,platforms, user_statuses, infoIDs, user_types,output_dir):
    create_output_dir(output_dir)

    platform_result_list = []
    infoID_result_list = []
    user_status_list = []
    user_type_list = []

    rmse_list = []
    nrmse_list = []
    actual_list = []
    pred_list = []

    num_quartets = len(infoIDs) * len(platforms) * len(user_statuses) * len(user_statuses)
    print("\nnum_quartets: %d" %num_quartets)

    i = 1
    for platform in platforms:
        for user_status in user_statuses:
            for infoID in infoIDs:
                for user_type in user_types:
                    platform_result_list.append(platform)
                    user_status_list.append(user_status)
                    user_type_list.append(user_type)
                    infoID_result_list.append(infoID)

                    #get pred and gt
                    y_test =quartet_pred_dict[platform][user_status][infoID][user_type]["y_test"]
                    y_pred = quartet_pred_dict[platform][user_status][infoID][user_type]["y_pred"]
                    y_test = y_test.flatten()
                    y_pred = y_pred.flatten()

                    # if DEBUG_PRINT == True:
                    #     print(y_test)
                    #     print(y_pred)

                    # #get volumes
                    # pred_volume = np.sum(y_pred)
                    # actual_volume = np.sum(y_test)
                    # actual_list.append(actual_volume)
                    # pred_list.append(pred_volume)

                    #rmse
                    cur_rmse = mean_squared_error(y_test, y_pred, squared=False)
                    rmse_list.append(cur_rmse)

                    cur_nrmse = nunique_users_over_time_nrmse(y_test, y_pred)
                    nrmse_list.append(cur_nrmse)


                    #mape
                    # cur_mape = mean_absolute_percentage_error(y_test, y_pred)
                    # mape_list.append(cur_mape)
                    i+=1
                    print("Got quartet score record %d of %d" %(i, num_quartets))

    #make df
    data = {"platform":platform_result_list, "user_status":user_status_list,
        "informationID":infoID_result_list, "user_type" : user_type_list, "nrmse_over_time":nrmse_list,
        "rmse_over_time":rmse_list}
    result_df = pd.DataFrame(data=data)

    result_df["nrmse_over_time"] = result_df["nrmse_over_time"].fillna(0)

    result_df = result_df.sort_values("rmse_over_time", ascending=True).reset_index(drop=True)
    print(result_df)
    output_fp = output_dir + "results.csv"
    result_df.to_csv(output_fp, index=False)
    print(output_fp)

    total_nrmse_over_time = np.sum(result_df["nrmse_over_time"])
    print("\ntotal_nrmse_over_time: %.4f"%total_nrmse_over_time)
    with open(output_dir + "total_nrmse_over_time.txt","w") as f:
        f.write(str(total_nrmse_over_time))

    total_rmse_over_time = np.sum(rmse_list)
    print("\ntotal_rmse_over_time: %.4f"%total_rmse_over_time)
    with open(output_dir + "total_rmse_over_time.txt","w") as f:
        f.write(str(total_rmse_over_time))


    return result_df

def nunique_users_over_time_nrmse(y_test, y_pred):

    df = pd.DataFrame(data={"value_sim": y_pred, "value_gt":y_test})
    df = df[["value_gt", "value_sim"]]
    # print(df)

    #cumul
    df['value_sim'] = df['value_sim'].cumsum()
    df['value_gt'] = df['value_gt'].cumsum()

    #norm it
    epsilon = 0.001*df[df['value_gt'] != 0.0]['value_gt'].min()
    df['value_sim'] = (df['value_sim'] + epsilon)/(df['value_sim'].max() + epsilon)
    df['value_gt'] = (df['value_gt'] + epsilon)/(df['value_gt'].max() + epsilon)

    return np.sqrt(((df["value_sim"]-df["value_gt"])**2).mean())


def rmse(self, ground_truth, simulation, join='inner', fill_value=0,
        relative=False, cumulative=False, normed=False):
        """
        Metric: rmse
        Description: Root mean squared error
        Inputs:
        ground_truth - ground truth measurement (data frame) with measurement in
            the "value" column
        simulation - simulation measurement (data frame) with measurement in the
            "value" column
        join - type of join to perform between ground truth and simulation
        fill_value - fill value for non-overlapping joins
        """

        if type(ground_truth) is np.ndarray:
            result = ground_truth - simulation
            result = (result ** 2).mean()
            result = np.sqrt(result)
            return result

        if type(ground_truth) is list:

            ground_truth = np.nan_to_num(ground_truth)
            simulation   = np.nan_to_num(simulation)

            result = np.asarray(ground_truth) - np.asarray(simulation)
            result = (result ** 2).mean()
            result = np.sqrt(result)

            return result

        df = self.join_dfs(ground_truth, simulation, join=join,
            fill_value=fill_value)


        if len(df.index) > 0:

            if cumulative:
                df['value_sim'] = df['value_sim'].cumsum()
                df['value_gt'] = df['value_gt'].cumsum()

            if normed:
                epsilon = 0.001*df[df['value_gt'] != 0.0]['value_gt'].min()
                df['value_sim'] = (df['value_sim'] + epsilon)/(df['value_sim'].max() + epsilon)
                df['value_gt'] = (df['value_gt'] + epsilon)/(df['value_gt'].max() + epsilon)

            if not relative:
                return np.sqrt(((df["value_sim"]-df["value_gt"])**2).mean())
            else:
                iq_range = float(iqr(df['value_gt'].values))

                result = df["value_sim"]-df["value_gt"]
                result = (result ** 2).mean()
                result = np.sqrt(result)

                if iq_range > 0:
                    result = result / iq_range
                else:
                    mean_value = df['value_gt'].mean()
                    if mean_value > 0:
                        result = result / mean_value
                    else:
                        return None

                return result
        else:
            return None

def get_quartet_result_df_backup2(quartet_pred_dict,quartet_train_and_test_array_dict,platforms, user_statuses, infoIDs, user_types,output_dir):
    platform_result_list = []
    infoID_result_list = []
    user_status_list = []
    user_type_list = []

    rmse_list = []
    nrmse_list = []
    actual_list = []
    pred_list = []


    for platform in platforms:
        for user_status in user_statuses:
            for infoID in infoIDs:
                for user_type in user_types:
                    platform_result_list.append(platform)
                    user_status_list.append(user_status)
                    user_type_list.append(user_type)
                    infoID_result_list.append(infoID)

                    #get pred and gt
                    y_test =quartet_train_and_test_array_dict[platform][user_status][infoID][user_type]["y_test"]
                    y_pred = quartet_pred_dict[platform][user_status][infoID][user_type]
                    y_test = y_test.flatten()
                    y_pred = y_pred.flatten()

                    # #get volumes
                    # pred_volume = np.sum(y_pred)
                    # actual_volume = np.sum(y_test)
                    # actual_list.append(actual_volume)
                    # pred_list.append(pred_volume)

                    #rmse
                    cur_rmse = mean_squared_error(y_test, y_pred, squared=False)
                    rmse_list.append(cur_rmse)

                    cur_nrmse = nunique_users_over_time_nrmse(y_test, y_pred)
                    nrmse_list.append(cur_nrmse)


                    #mape
                    # cur_mape = mean_absolute_percentage_error(y_test, y_pred)
                    # mape_list.append(cur_mape)

    #make df
    data = {"platform":platform_result_list, "user_status":user_status_list,
        "informationID":infoID_result_list, "user_type" : user_type_list, "nrmse_over_time":nrmse_list,
        "rmse_over_time":rmse_list}
    result_df = pd.DataFrame(data=data)

    result_df = result_df.sort_values("nrmse_over_time", ascending=True).reset_index(drop=True)
    print(result_df)
    output_fp = output_dir + "results.csv"
    result_df.to_csv(output_fp, index=False)
    print(output_fp)

    avg_nrmse_over_time = np.mean(nrmse_list)
    print("\navg_nrmse_over_time: %.4f"%avg_nrmse_over_time)
    with open(output_dir + "avg_nrmse_over_time.txt","w") as f:
        f.write(str(avg_nrmse_over_time))

    avg_rmse_over_time = np.mean(rmse_list)
    print("\navg_rmse_over_time: %.4f"%avg_rmse_over_time)
    with open(output_dir + "avg_rmse_over_time.txt","w") as f:
        f.write(str(avg_rmse_over_time))


    return result_df

def get_quartet_result_df(quartet_pred_dict,quartet_train_and_test_array_dict,platforms, user_statuses, infoIDs, user_types,output_dir,DEBUG_PRINT=False):

    create_output_dir(output_dir)
    platform_result_list = []
    infoID_result_list = []
    user_status_list = []
    user_type_list = []

    rmse_list = []
    nrmse_list = []
    actual_list = []
    pred_list = []

    num_quartets = len(infoIDs) * len(platforms) * len(user_statuses) * len(user_statuses)
    print("\nnum_quartets: %d" %num_quartets)

    i = 1
    for platform in platforms:
        for user_status in user_statuses:
            for infoID in infoIDs:
                for user_type in user_types:
                    platform_result_list.append(platform)
                    user_status_list.append(user_status)
                    user_type_list.append(user_type)
                    infoID_result_list.append(infoID)

                    #get pred and gt
                    y_test =quartet_train_and_test_array_dict[platform][user_status][infoID][user_type]["y_test"]
                    y_pred = quartet_pred_dict[platform][user_status][infoID][user_type]
                    y_test = y_test.flatten()
                    y_pred = y_pred.flatten()

                    if DEBUG_PRINT == True:
                        print(y_test)
                        print(y_pred)

                    # #get volumes
                    # pred_volume = np.sum(y_pred)
                    # actual_volume = np.sum(y_test)
                    # actual_list.append(actual_volume)
                    # pred_list.append(pred_volume)

                    #rmse
                    cur_rmse = mean_squared_error(y_test, y_pred, squared=False)
                    rmse_list.append(cur_rmse)

                    cur_nrmse = nunique_users_over_time_nrmse(y_test, y_pred)
                    nrmse_list.append(cur_nrmse)


                    #mape
                    # cur_mape = mean_absolute_percentage_error(y_test, y_pred)
                    # mape_list.append(cur_mape)
                    i+=1
                    print("Got quartet score record %d of %d" %(i, num_quartets))

    #make df
    data = {"platform":platform_result_list, "user_status":user_status_list,
        "informationID":infoID_result_list, "user_type" : user_type_list, "nrmse_over_time":nrmse_list,
        "rmse_over_time":rmse_list}
    result_df = pd.DataFrame(data=data)

    result_df["nrmse_over_time"] = result_df["nrmse_over_time"].fillna(0)

    result_df = result_df.sort_values("rmse_over_time", ascending=True).reset_index(drop=True)
    print(result_df)
    output_fp = output_dir + "results.csv"
    result_df.to_csv(output_fp, index=False)
    print(output_fp)

    total_nrmse_over_time = np.sum(result_df["nrmse_over_time"])
    print("\ntotal_nrmse_over_time: %.4f"%total_nrmse_over_time)
    with open(output_dir + "total_nrmse_over_time.txt","w") as f:
        f.write(str(total_nrmse_over_time))

    total_rmse_over_time = np.sum(rmse_list)
    print("\ntotal_rmse_over_time: %.4f"%total_rmse_over_time)
    with open(output_dir + "total_rmse_over_time.txt","w") as f:
        f.write(str(total_rmse_over_time))


    return result_df

def get_quartet_result_df_v2_save_option(quartet_pred_dict,quartet_train_and_test_array_dict,platforms, user_statuses, infoIDs, user_types,output_dir=None,DEBUG_PRINT=False):

    platform_result_list = []
    infoID_result_list = []
    user_status_list = []
    user_type_list = []

    rmse_list = []
    nrmse_list = []
    actual_list = []
    pred_list = []

    num_quartets = len(infoIDs) * len(platforms) * len(user_statuses) * len(user_statuses)
    print("\nnum_quartets: %d" %num_quartets)

    i = 1
    for platform in platforms:
        for user_status in user_statuses:
            for infoID in infoIDs:
                for user_type in user_types:
                    platform_result_list.append(platform)
                    user_status_list.append(user_status)
                    user_type_list.append(user_type)
                    infoID_result_list.append(infoID)

                    #get pred and gt
                    y_test =quartet_train_and_test_array_dict[platform][user_status][infoID][user_type]["y_test"]
                    y_pred = quartet_pred_dict[platform][user_status][infoID][user_type]
                    y_test = y_test.flatten()
                    y_pred = y_pred.flatten()

                    if DEBUG_PRINT == True:
                        print("\nytest and ypred")
                        print(y_test)
                        print(y_pred)

                    # #get volumes
                    # pred_volume = np.sum(y_pred)
                    # actual_volume = np.sum(y_test)
                    # actual_list.append(actual_volume)
                    # pred_list.append(pred_volume)

                    #rmse
                    cur_rmse = mean_squared_error(y_test, y_pred, squared=False)
                    rmse_list.append(cur_rmse)

                    cur_nrmse = nunique_users_over_time_nrmse(y_test, y_pred)
                    nrmse_list.append(cur_nrmse)


                    #mape
                    # cur_mape = mean_absolute_percentage_error(y_test, y_pred)
                    # mape_list.append(cur_mape)
                    i+=1
                    print("Got quartet score record %d of %d" %(i, num_quartets))

    #make df
    data = {"platform":platform_result_list, "user_status":user_status_list,
        "informationID":infoID_result_list, "user_type" : user_type_list, "nrmse_over_time":nrmse_list,
        "rmse_over_time":rmse_list}
    result_df = pd.DataFrame(data=data)

    result_df["nrmse_over_time"] = result_df["nrmse_over_time"].fillna(0)

    result_df = result_df.sort_values("rmse_over_time", ascending=True).reset_index(drop=True)
    print(result_df)


    total_nrmse_over_time = np.sum(result_df["nrmse_over_time"])
    print("\ntotal_nrmse_over_time: %.4f"%total_nrmse_over_time)


    total_rmse_over_time = np.sum(rmse_list)
    print("\ntotal_rmse_over_time: %.4f"%total_rmse_over_time)


    if output_dir != None:
        create_output_dir(output_dir)
        output_fp = output_dir + "results.csv"
        result_df.to_csv(output_fp, index=False)
        print(output_fp)

        with open(output_dir + "total_nrmse_over_time.txt","w") as f:
            f.write(str(total_nrmse_over_time))

        with open(output_dir + "total_rmse_over_time.txt","w") as f:
            f.write(str(total_rmse_over_time))


    return (result_df,total_nrmse_over_time,total_rmse_over_time)

# def get_quartet_result_df_v2(GET_GROUND_TRUTH,quartet_pred_dict,quartet_train_and_test_array_dict,platforms, user_statuses, infoIDs, user_types,output_dir,DEBUG_PRINT=False):

#     create_output_dir(output_dir)
#     platform_result_list = []
#     infoID_result_list = []
#     user_status_list = []
#     user_type_list = []

#     rmse_list = []
#     nrmse_list = []
#     actual_list = []
#     pred_list = []

#     num_quartets = len(infoIDs) * len(platforms) * len(user_statuses) * len(user_statuses)
#     print("\nnum_quartets: %d" %num_quartets)

#     i = 1
#     for platform in platforms:
#         for user_status in user_statuses:
#             for infoID in infoIDs:
#                 for user_type in user_types:
#                     platform_result_list.append(platform)
#                     user_status_list.append(user_status)
#                     user_type_list.append(user_type)
#                     infoID_result_list.append(infoID)

#                     #get pred and gt
#                     if GET_GROUND_TRUTH == True:
#                         y_test =quartet_train_and_test_array_dict[platform][user_status][infoID][user_type]["y_test"]
#                         y_test = y_test.flatten()

#                     y_pred = quartet_pred_dict[platform][user_status][infoID][user_type]

#                     y_pred = y_pred.flatten()

#                     if DEBUG_PRINT == True:
#                         if GET_GROUND_TRUTH == True:
#                             print(y_test)
#                         print(y_pred)

#                     # #get volumes
#                     # pred_volume = np.sum(y_pred)
#                     # actual_volume = np.sum(y_test)
#                     # actual_list.append(actual_volume)
#                     # pred_list.append(pred_volume)

#                     #rmse
#                     cur_rmse = mean_squared_error(y_test, y_pred, squared=False)
#                     rmse_list.append(cur_rmse)

#                     cur_nrmse = nunique_users_over_time_nrmse(y_test, y_pred)
#                     nrmse_list.append(cur_nrmse)


#                     #mape
#                     # cur_mape = mean_absolute_percentage_error(y_test, y_pred)
#                     # mape_list.append(cur_mape)
#                     i+=1
#                     print("Got quartet score record %d of %d" %(i, num_quartets))

#     #make df
#     data = {"platform":platform_result_list, "user_status":user_status_list,
#         "informationID":infoID_result_list, "user_type" : user_type_list, "nrmse_over_time":nrmse_list,
#         "rmse_over_time":rmse_list}
#     result_df = pd.DataFrame(data=data)

#     result_df["nrmse_over_time"] = result_df["nrmse_over_time"].fillna(0)

#     result_df = result_df.sort_values("rmse_over_time", ascending=True).reset_index(drop=True)
#     print(result_df)
#     output_fp = output_dir + "results.csv"
#     result_df.to_csv(output_fp, index=False)
#     print(output_fp)

#     total_nrmse_over_time = np.sum(result_df["nrmse_over_time"])
#     print("\ntotal_nrmse_over_time: %.4f"%total_nrmse_over_time)
#     with open(output_dir + "total_nrmse_over_time.txt","w") as f:
#         f.write(str(total_nrmse_over_time))

#     total_rmse_over_time = np.sum(rmse_list)
#     print("\ntotal_rmse_over_time: %.4f"%total_rmse_over_time)
#     with open(output_dir + "total_rmse_over_time.txt","w") as f:
#         f.write(str(total_rmse_over_time))


#     return result_df

def get_quartet_result_df_backup(quartet_pred_dict,quartet_train_and_test_array_dict,platforms, user_statuses, infoIDs, user_types,output_dir):
    platform_result_list = []
    infoID_result_list = []
    user_status_list = []
    user_type_list = []

    rmse_list = []
    mape_list = []
    actual_volume_list = []
    pred_volume_list = []


    for platform in platforms:
        for user_status in user_statuses:
            for infoID in infoIDs:
                for user_type in user_types:
                    platform_result_list.append(platform)
                    user_status_list.append(user_status)
                    user_type_list.append(user_type)
                    infoID_result_list.append(infoID)

                    #get pred and gt
                    y_test =quartet_train_and_test_array_dict[platform][user_status][infoID][user_type]["y_test"]
                    y_pred = quartet_pred_dict[platform][user_status][infoID][user_type]
                    y_test = y_test.flatten()
                    y_pred = y_pred.flatten()

                    #get volumes
                    pred_volume = np.sum(y_pred)
                    actual_volume = np.sum(y_test)
                    actual_volume_list.append(actual_volume)
                    pred_volume_list.append(pred_volume)

                    #rmse
                    cur_rmse = mean_squared_error([actual_volume], [pred_volume], squared=False)
                    rmse_list.append(cur_rmse)

                    #mape
                    cur_mape = mean_absolute_percentage_error(actual_volume, pred_volume)
                    mape_list.append(cur_mape)

    #make df
    data = {"platform":platform_result_list, "user_status":user_status_list,
        "informationID":infoID_result_list, "user_type" : user_type_list, "actual_volume":actual_volume_list,
        "pred_volume":pred_volume_list, "mape":mape_list, "rmse":rmse_list}
    result_df = pd.DataFrame(data=data)

    result_df = result_df.sort_values("mape", ascending=True).reset_index(drop=True)
    print(result_df)
    output_fp = output_dir + "results.csv"
    result_df.to_csv(output_fp, index=False)
    print(output_fp)

    total_pred_volume = result_df["pred_volume"].sum()
    total_actual_volume = result_df["actual_volume"].sum()
    print("\ntotal_pred_volume: %.4f"%total_pred_volume)
    print("total_actual_volume: %4.f" %total_actual_volume)
    final_rmse = mean_squared_error([total_pred_volume], [total_actual_volume], squared=False)
    print("\nfinal_rmse: %.4f"%final_rmse)
    with open(output_dir + "final-rmse.txt","w") as f:
        f.write(str(final_rmse))

    return result_df

def get_main_user_nn_fts():
    infoIDs = get_46_cp4_infoIDs()
    user_fts = ["is_child", "is_new", "is_twitter"]
    return infoIDs + user_fts

def get_corr_df_with_options(df, cols1, cols2,corr_func):
    print("\nGetting corrs...")

    corrs = []
    f1_list = []
    f2_list = []
    dense1_list = []
    dense2_list = []
    num_elements = df.shape[0]
    num_corrs = len(cols1) * len(cols2)

    i = 0
    for f1 in cols1:
        for f2 in cols2:
            if f1 != f2:
                corr = corr_func(df[f1].values, df[f2].values)[0]
                corrs.append(corr)

                #get densities
                d1 = np.count_nonzero(df[f1].values)/num_elements
                d2 = np.count_nonzero(df[f2].values)/num_elements
                dense1_list.append(d1)
                dense2_list.append(d2)
                f1_list.append(f1)
                f2_list.append(f2)

                i+=1
                print("Got corr %d of %d"%(i, num_corrs))

    #make df
    corr_df = pd.DataFrame(data={"f1":f1_list, "f2":f2_list,"corr":corrs, "density1":dense1_list,"density2":dense2_list})
    corr_df = corr_df[["f1", "f2", "corr", "density1", "density2"]]
    corr_df["abs_val_corr"] = np.absolute(corr_df["corr"])
    corr_df = corr_df.sort_values("abs_val_corr", ascending=False).reset_index(drop=True)

    return corr_df

def get_corr_df_with_options_v2_fixed(df, cols1, cols2,corr_func):
    print("\nGetting corrs...")

    corrs = []
    f1_list = []
    f2_list = []
    dense1_list = []
    dense2_list = []
    num_elements = df.shape[0]
    num_corrs = len(cols1) * len(cols2)

    i = 0
    for f1 in cols1:
        for f2 in cols2:
            if f1 != f2:
                corr = corr_func(df[f1].values, df[f2].values)[0]
                corrs.append(corr)

                #get densities
                d1 = np.count_nonzero(df[f1].values)/num_elements
                d2 = np.count_nonzero(df[f2].values)/num_elements
                dense1_list.append(d1)
                dense2_list.append(d2)
                f1_list.append(f1)
                f2_list.append(f2)

                i+=1
                print("Got corr %d of %d"%(i, num_corrs))

    #make df
    corr_df = pd.DataFrame(data={"f1":f1_list, "f2":f2_list,"corr":corrs, "density1":dense1_list,"density2":dense2_list})
    corr_df = corr_df[["f1", "f2", "corr", "density1", "density2"]]
    corr_df["abs_val_corr"] = np.absolute(corr_df["corr"])
    corr_df = corr_df.sort_values("abs_val_corr", ascending=False).reset_index(drop=True)

    return corr_df


def get_corr_df(df, cols1, cols2):
    print("\nGetting corrs...")

    corrs = []
    f1_list = []
    f2_list = []
    dense1_list = []
    dense2_list = []
    num_elements = df.shape[0]
    num_corrs = len(cols1) * len(cols2)

    i = 0
    for f1 in cols1:
        for f2 in cols2:
            if f1 != f2:
                corr = stats.pearsonr(df[f1].values, df[f2].values)[0]
                corrs.append(corr)

                #get densities
                d1 = np.count_nonzero(df[f1].values)/num_elements
                d2 = np.count_nonzero(df[f2].values)/num_elements
                dense1_list.append(d1)
                dense2_list.append(d2)
                f1_list.append(f1)
                f2_list.append(f2)

                i+=1
                print("Got corr %d of %d"%(i, num_corrs))

    #make df
    corr_df = pd.DataFrame(data={"f1":f1_list, "f2":f2_list,"corr":corrs, "density1":dense1_list,"density2":dense2_list})
    corr_df = corr_df[["f1", "f2", "corr", "density1", "density2"]]
    corr_df["abs_val_corr"] = np.absolute(corr_df["corr"])
    corr_df = corr_df.sort_values("abs_val_corr", ascending=False).reset_index(drop=True)

    return corr_df

def get_gdelt_rootcodes():
    return ['DISAPPROVE', 'EXPRESS INTENT TO COOPERATE', 'MAKE PUBLIC STATEMENT',
    'ENGAGE IN DIPLOMATIC COOPERATION', 'REJECT', 'CONSULT', 'DEMAND', 'COERCE',
    'APPEAL', 'FIGHT', 'ENGAGE IN MATERIAL COOPERATION', 'YIELD', 'REDUCE RELATIONS',
    'INVESTIGATE', 'ASSAULT', 'PROVIDE AID', 'THREATEN', 'PROTEST', 'EXHIBIT FORCE POSTURE',
    'USE UNCONVENTIONAL MASS VIOLENCE']

def read(filepath, SKIP_FIRST_LINE=False):
    MOD_NUM = 10000

    i=0
    json_list = []
    f = open(filepath)
    for line in f:
        if SKIP_FIRST_LINE == True:
            if i==0:
                i+=1
                continue
        if i%MOD_NUM==0:
            print("Getting line %d" %i)
        json_dict = json.loads(line)
        json_list.append(json_dict)
        # if DEBUG==True and i==LIMIT:
        #     break
        i+=1
    f.close()
    return json_list

def convert_json_to_df(json_fp,SKIP_FIRST_LINE=False):
    df=pd.io.json.json_normalize(read(json_fp, SKIP_FIRST_LINE))
    return df

def mark_user_first_appearances(df, GRAN, mark_parents=True,MOD_NUM=1000):

    #FLOOR IT
    df["nodeTime"] = df["nodeTime"].dt.floor(GRAN)

    #mark bdays
    print("\nGetting child user bdays...")
    df["nodeUserID_birthdate"] = df.groupby(["nodeUserID"])["nodeTime"].transform("min")

    #if nodetime==birthday, user is new
    print("\nGetting is new col for child users...")
    df["is_new_child_user"] = [1 if cur_time==bdate else 0 for cur_time,bdate in zip(list(df["nodeTime"]), list(df["nodeUserID_birthdate"]))]

    print(df[["nodeTime", "nodeUserID", "is_new_child_user"]].copy().drop_duplicates().reset_index(drop=True))

    if mark_parents==True:
        print("\nGetting user to bday dict...")
        user_to_bday_dict = convert_df_2_cols_to_dict(df, "nodeUserID", "nodeUserID_birthdate")

        #get parent bdays to account for missing parents
        print("\nGetting parent initial bdays...")
        df["parentUserID_birthdate"] = df.groupby(["parentUserID"])["nodeTime"].transform("min")

        parent_to_bday_dict = convert_df_2_cols_to_dict(df, "parentUserID", "parentUserID_birthdate")

        parents = list(df["parentUserID"].unique())
        num_parents = len(parents)

        for i,parent in enumerate(parents):
            if parent not in user_to_bday_dict:
                user_to_bday_dict[parent] = parent_to_bday_dict[parent]
            if i%MOD_NUM == 0:
                print("Processed %d of %d parent bdate records"%(i, num_parents))



        #fix it
        print("\nMapping parents to correct bdays...")
        df["parentUserID_birthdate"] = df["parentUserID"].map(user_to_bday_dict)

        #if nodetime==birthday, user is new
        print("\nGetting is new col for parent users...")
        df["parentUserID_birthdate"] = pd.to_datetime(df["parentUserID_birthdate"], utc=True)
        df["is_new_parent_user"] = [1 if cur_time==bdate else 0 for cur_time,bdate in zip(list(df["nodeTime"]), list(df["parentUserID_birthdate"]))]

        df = df.drop(["parentUserID_birthdate"], axis=1)

        print(df[["nodeTime", "parentUserID", "is_new_parent_user"]].copy().drop_duplicates().reset_index(drop=True))

    df = df.drop(["nodeUserID_birthdate"], axis=1)

    return df

def count_children_parents_and_actions(df,user_status ,start,end,GRAN,kickout_other_cols=True):

    df["nodeTime"] = df["nodeTime"].dt.floor(GRAN)

    #for actions and children
    if user_status == "new":
        temp_action_count_df = df.copy()
        temp_action_count_df = temp_action_count_df[temp_action_count_df["is_new_child_user"]==1].reset_index(drop=True)
    elif user_status == "old":
        temp_action_count_df = df.copy()
        temp_action_count_df = temp_action_count_df[temp_action_count_df["is_new_child_user"]==0].reset_index(drop=True)
    elif user_status == "all":
        print("\nUser status is all. Not changing df...")
        temp_action_count_df = df.copy()
    else:
        print("Error! Choose from these options for user status: new, old, all")
        print("You chose %s" %user_status)
        sys.exit(0)

    #for parents
    if user_status == "new":
        temp_parent_df = df.copy()
        temp_parent_df = temp_parent_df[temp_parent_df["is_new_parent_user"]==1].reset_index(drop=True)
    elif user_status == "old":
        temp_parent_df = df.copy()
        temp_parent_df = temp_parent_df[temp_parent_df["is_new_parent_user"]==0].reset_index(drop=True)
    elif user_status == "all":
        temp_parent_df = df.copy()
        print("\nUser status is all. Not changing df...")
    else:
        print("Error! Choose from these options for user status: new, old, all")
        print("You chose %s" %user_status)
        sys.exit(0)

    #get actual count
    num_actual_actions = temp_action_count_df.shape[0]


    #count actions
    temp_action_count_df["num_actions"] = temp_action_count_df.groupby(["nodeTime"])["actionType"].transform("count")

    #count children
    temp_action_count_df["nunique_child_users"] = temp_action_count_df.groupby(["nodeTime"])["nodeUserID"].transform("nunique")
    temp_action_count_df["nunique_child_users"] = temp_action_count_df["nunique_child_users"].astype("float32")
    temp_action_count_df = temp_action_count_df[["nodeTime", "num_actions","nunique_child_users"]].drop_duplicates().reset_index(drop=True)

    #self check
    post_process_num_actions = temp_action_count_df["num_actions"].sum()
    print("\npost_process_num_actions: %d" %post_process_num_actions)
    print("num_actual_actions: %d" %num_actual_actions)
    if num_actual_actions != post_process_num_actions:
        print("Error: num_actual_actions != post_process_num_actions")
        sys.exit(0)
    else:
        print("num_actual_actions == post_process_num_actions")
        print("Continuing!")

    #now count nunique parents
    temp_parent_df = temp_parent_df[["nodeTime", "parentUserID"]]
    temp_parent_df["nunique_parent_users"] = temp_parent_df.groupby(["nodeTime"])["parentUserID"].transform("nunique")
    temp_parent_df["nunique_parent_users"] = temp_parent_df["nunique_parent_users"].astype("float32")
    temp_parent_df = temp_parent_df[["nodeTime","nunique_parent_users"]].drop_duplicates().reset_index(drop=True)

    print(temp_parent_df)
    print(temp_action_count_df)

    final_df = pd.merge(temp_parent_df, temp_action_count_df, on="nodeTime", how="outer")
    print(final_df)
    # sys.exit(0)

    #get dates
    date_df = create_blank_date_df(start,end,GRAN)
    # final_df = temp_parent_df.merge(temp_action_count_df, on="nodeTime", how="outer")
    final_df = final_df.merge(date_df, on="nodeTime", how="outer")
    final_df = final_df.fillna(0)
    final_df = final_df.drop_duplicates().reset_index(drop=True)
    final_df = final_df.sort_values("nodeTime").reset_index(drop=True)



    return final_df

def make_tuple_list_from_2_cols(df,col1,col2):
    return list(zip(df[col1], df[col2]))

#actions by new and old users
def get_num_actions_of_old_and_new_users_by_infoID(df):

    temp = df.copy()

    temp["nodeTime"] = pd.to_datetime(temp["nodeTime"], utc=True)
    temp = temp.sort_values("nodeTime").reset_index(drop=True)

    #new
    temp["is_new_user"] = temp["user_status"].isin(["new"]).astype("int32")
    temp["num_new_user_actions"] = temp["num_actions"] * temp["is_new_user"]

    #old
    temp["is_old_user"] = temp["user_status"].isin(["old"]).astype("int32")
    temp["num_old_user_actions"] = temp["num_actions"] * temp["is_old_user"]

    temp["num_new_user_actions"] = temp.groupby(["nodeTime"])["num_new_user_actions"].transform("sum")
    temp["num_old_user_actions"] = temp.groupby(["nodeTime"])["num_old_user_actions"].transform("sum")
    temp = temp[["nodeTime","num_new_user_actions","num_old_user_actions"]]
    temp = temp.drop_duplicates().reset_index(drop=True)

    return temp

#first get dataframes
def make_user_status_pair_df(df):
    # print("\nMaking user_status_pair_df...")
    temp = df[["nodeTime","user_status","parent_status"]].copy()
    temp["status_pair"] = df["user_status"] + "_" + df["parent_status"]
    temp["status_pair_count"] = temp.groupby(["nodeTime","status_pair"])["status_pair"].transform("count")
    temp = temp.drop_duplicates().reset_index(drop=True)
    # print(temp)
    return temp

def get_cumulative_user_count(df,user_type_tag,user_status_tag):
    tag = "%s_%s"%(user_status_tag,user_type_tag)
    print(tag)
    nunique_col ="%s_daily_nunique"%tag
    cumsum_col = "%s_daily_nunique_cumcount"%tag

    df["nodeTime"] = pd.to_datetime(df["nodeTime"], utc=True)
    df = df.sort_values("nodeTime").reset_index(drop=True)

    df[nunique_col] = df.groupby(["nodeTime"])[user_type_tag].transform("nunique")


    #make groupby list
    df_groupby_list = make_df_groupby_tuple_list(df,"nodeTime")
    user_nunique_cumsum_list = []
    nodeTimes = list(df["nodeTime"].unique())
    user_set = set()
    for i,df_tuple in enumerate(df_groupby_list):
        nodeTime = df_tuple[0]
        cur_df = df_tuple[1]
        users_to_add = set(list(cur_df[user_type_tag]))
        if len(users_to_add) == np.nan:
            print("Error: len(users_to_add) ==np.nan")
            sys.exit(0)

        # print("\nusers_to_add")
        # print(users_to_add)
        prev_num_users = len(user_set)
        print("\nprev_num_users: %d" %prev_num_users)
        user_set.update(users_to_add)
        cur_num_users = len(user_set)

        user_nunique_cumsum_list.append(cur_num_users)
        print("Idx: %d, date: %s, cur_num_users: %d"%((i+1), str(nodeTime), cur_num_users))
        if cur_num_users < prev_num_users:
            print("Error: cur_num_users < prev_num_users")
            sys.exit(0)


    cumcount_df = pd.DataFrame(data={"nodeTime":nodeTimes, cumsum_col:user_nunique_cumsum_list})
    # print(cumcount_df)
    df["nodeTime"] = pd.to_datetime(df["nodeTime"],utc=True)
    cumcount_df["nodeTime"] = pd.to_datetime(cumcount_df["nodeTime"],utc=True)
    df = pd.merge(df,cumcount_df, on="nodeTime", how="inner")
    df = df[["nodeTime", nunique_col,cumsum_col]].drop_duplicates().reset_index(drop=True)



    return df,nunique_col,cumsum_col

# def merge_mult_dfs(merge_list, on, how):
#     print("\nMerging multiple dfs...")
#     return reduce(lambda  left,right: pd.merge(left,right,on=on, how=how), merge_list)

def make_df_groupby_tuple_list(df,groupby_col):
    print("\nGetting groupby list...")
    dfs = [(x,y) for x,y in df.groupby(groupby_col)]
    print("Got it")
    # for i,cur_tuple in enumerate(dfs):
    #     print(cur_tuple)
    #     if i==10:
    #         break
    return dfs


#cumul. user count df
def get_cumulative_user_count_df(df,start,end,GRAN="D"):

    full_dates = pd.date_range(start,end,freq=GRAN)
    date_df = pd.DataFrame(data={"nodeTime":full_dates})
    date_df["nodeTime"] = pd.to_datetime(date_df["nodeTime"], utc=True)

    temp = df[["nodeTime","user_status","parent_status","nodeUserID","parentUserID"]].copy()

    #we need 4 different counts
    new_node_user_temp = temp[temp["user_status"]=="new"].copy()
    old_node_user_temp = temp[temp["user_status"]=="old"].copy()
    new_parent_user_temp = temp[temp["parent_status"]=="new"].copy()
    old_parent_user_temp = temp[temp["parent_status"]=="old"].copy()

    temp1,nunique_col1, cumsum_col1 = get_cumulative_user_count(new_node_user_temp,"nodeUserID","new")
    temp2,nunique_col2, cumsum_col2  = get_cumulative_user_count(old_node_user_temp,"nodeUserID","old")
    temp3,nunique_col3, cumsum_col3  = get_cumulative_user_count(new_parent_user_temp,"parentUserID","new")
    temp4,nunique_col4, cumsum_col4  = get_cumulative_user_count(old_parent_user_temp,"parentUserID","old")
    merge_list = [temp1, temp2, temp3, temp4]

    # temp = reduce(lambda  left,right: pd.merge(left,right,on=['nodeTime'], how='outer'), merge_list)
    temp = merge_mult_dfs(merge_list, ["nodeTime"], "outer")
    FILLNA_VAL = -1
    temp = temp.fillna(FILLNA_VAL)

    #fix cumsum cols
    cumsum_cols = [cumsum_col1,cumsum_col2,cumsum_col3,cumsum_col4]
    for col in cumsum_cols:
        cur_cumsum_vals = temp[col]
        # print("\n%s cur_cumsum_vals"%col)
        # print(cur_cumsum_vals)
        cur_cumsum_vals = list(cur_cumsum_vals)
        for i,val in enumerate(cur_cumsum_vals):
            if (i==0) and (val==FILLNA_VAL):
                cur_cumsum_vals[i] = 0
            if (i >0) and (val==FILLNA_VAL):
                cur_cumsum_vals[i] = cur_cumsum_vals[i-1]
        temp[col] = cur_cumsum_vals
        # print("\nFixed")
        # print(temp[col])

    #fix the others
    nunique_cols = [nunique_col1, nunique_col2,nunique_col3,nunique_col4]
    for col in nunique_cols:
        cur_nu_vals = temp[col]
        # print("\n%s cur_nu_vals"%col)
        # print(cur_nu_vals)
        cur_nu_vals = list(cur_nu_vals)
        for i,val in enumerate(cur_nu_vals):
            if val == FILLNA_VAL:
                cur_nu_vals[i] = 0
        temp[col] = cur_nu_vals
        # print("\nFixed")
        # print(temp[col])

    # sys.exit(0)



    return temp

#get zero and nonzero counts
def get_zero_and_nonzero_counts(y_array, tag):

    num_elements = y_array.flatten().shape[0]
    nonzero_count = np.count_nonzero(y_array.flatten())
    zero_count = num_elements - nonzero_count
    nz_freq = nonzero_count/(1.0 * num_elements)
    zero_freq = 1 - nz_freq
    print("\nFor %s; nonzero count: %.4f, zero count: %.4f"%(tag, nonzero_count, zero_count))
    print("For %s; nonzero freq: %.4f, zero freq: %.4f"%(tag, nz_freq, zero_freq))

def down_sample_zero_target_samples(x_array,y_array,ZERO_TARGET_KEEP_FRACTION):

    zero_target_indices = []
    nzt_indices = []

    temp_y = y_array.flatten()
    num_elements = y_array.shape[0]
    for i,val in enumerate(temp_y):
        if val == 0:
            zero_target_indices.append(i)
        else:
            nzt_indices.append(i)
    num_zero_targets = len(zero_target_indices)
    print("\nnum_zero_targets")
    print(num_zero_targets)

    #num to kick out
    keep_num =int(num_zero_targets * ZERO_TARGET_KEEP_FRACTION)
    print("\nKeeping %d of these"%keep_num)

    kept_indices=np.random.choice(zero_target_indices, keep_num, replace=False)
    print("\nNum kept_indices: %d" %kept_indices.shape[0])

    print("\nNew x and y shapes before sampling")
    print(x_array.shape)
    print(y_array.shape)

    new_x_arrays = []
    new_y_arrays = []

    # x_array = x_array.reshape((x_array.shape[0], x_array.shape[1] * x_array.shape[2]))
    # y_array = y_array.reshape((y_array.shape[0], y_array.shape[1] * y_array.shape[2]))
    for i in range(num_elements):
        if (i in kept_indices) or (i in nzt_indices):
            new_x_arrays.append(x_array[i])
            new_y_arrays.append(y_array[i])

    # x_array = np.concatenate(new_x_arrays, axis=0)
    # y_array = np.concatenate(new_y_arrays, axis=0)
    x_array = np.asarray(new_x_arrays)
    y_array = np.asarray(new_y_arrays)
    print("\nNew x and y shapes after sampling")
    print(x_array.shape)
    print(y_array.shape)

    return x_array,y_array

def down_sample_nonzero_target_samples(x_array,y_array,NONZERO_TARGET_KEEP_FRACTION):

    zero_target_indices = []
    nzt_indices = []

    temp_y = y_array.flatten()
    num_elements = y_array.shape[0]
    for i,val in enumerate(temp_y):
        if val > 0:
            nzt_indices.append(i)
        else:
            zero_target_indices.append(i)
    num_nz_targets = len(nzt_indices)
    print("\nnum_nz_targets")
    print(num_nz_targets)

    #num to kick out
    keep_num =int(num_nz_targets * NONZERO_TARGET_KEEP_FRACTION)
    print("\nKeeping %d of these"%keep_num)

    kept_indices=np.random.choice(nzt_indices, keep_num, replace=False)
    print("\nNum kept_indices: %d" %kept_indices.shape[0])

    print("\nNew x and y shapes before sampling")
    print(x_array.shape)
    print(y_array.shape)

    new_x_arrays = []
    new_y_arrays = []

    # x_array = x_array.reshape((x_array.shape[0], x_array.shape[1] * x_array.shape[2]))
    # y_array = y_array.reshape((y_array.shape[0], y_array.shape[1] * y_array.shape[2]))
    for i in range(num_elements):
        if (i in kept_indices) or (i in zero_target_indices):
            new_x_arrays.append(x_array[i])
            new_y_arrays.append(y_array[i])

    # x_array = np.concatenate(new_x_arrays, axis=0)
    # y_array = np.concatenate(new_y_arrays, axis=0)
    x_array = np.asarray(new_x_arrays)
    y_array = np.asarray(new_y_arrays)
    print("\nNew x and y shapes after sampling")
    print(x_array.shape)
    print(y_array.shape)

    return x_array,y_array

def add_proba_fts_to_train_and_test_dfs(train_df,test_df, PROBA_TOKENS, token_to_df_dict,infoID,action):

    #merge with
    for PT in PROBA_TOKENS:
        # print(PT)
        if PT == "infoID_action_pair":
            pair = infoID + "_" + action
            train_df["infoID_action_pair"] = pair
        proba_df =  token_to_df_dict[PT].copy()

        #get chosen cols
        proba_cols = list(proba_df)
        proba_cols.remove(PT)

        #filter pdf
        if PT == "infoID_action_pair":
            proba_df = proba_df[proba_df[PT]==pair]

        if PT == "actionType":
            proba_df = proba_df[proba_df[PT]==action]

        if PT == "informationID":
            proba_df = proba_df[proba_df[PT]==infoID]

        # print("\nfiltered pdf")
        # print(proba_df)

        for proba_col in proba_cols:
            try:
                proba_val = proba_df[proba_col].unique()[0]
            except IndexError:
                proba_val = 0
            train_df[proba_col] = proba_val
            test_df[proba_col] = proba_val

        # print("\nNew train and test dfs")
        # print(train_df)
        # print(test_df)

    return train_df,test_df

#convert count vectors to glove vectors
def convert_infoID_vecs_to_cumavg_glove_vecs_per_user_per_nodeTime(df, glove_df,infoIDs, TRANSPOSE_GLOVE_DF=True,glove_d = 25,count_infoIDs=True, MOD_NUM=5000):
    print("\nConverting infoID vectors to glove embs...")

    df["nodeTime"] = df["nodeTime"].astype(str)

    if TRANSPOSE_GLOVE_DF == True:
        #transpose
        glove_df = glove_df.set_index("informationID")
        glove_df = glove_df.T
        print("\nTransposed glove df")
        print(glove_df)

    #get new cols
    desired_cols = ["nodeUserID","nodeTime"] + infoIDs

    #count up nodeID infoIDs
    if count_infoIDs == True:
        df["cumsum_num_infoIDs"] = df[infoIDs].sum(axis=1)
        infoID_count_df = df[["nodeUserID","nodeTime" ,"cumsum_num_infoIDs"]].copy().drop_duplicates().reset_index(drop=True)
        df = df.drop("cumsum_num_infoIDs", axis=1)
        # desired_cols = desired_cols + ["cumsum_num_infoIDs"]

    #get nodeIDs
    temp = df[desired_cols].drop_duplicates().reset_index(drop=True)
    print(temp)

    #save glove rows here
    #will use for df later
    glove_rows = []

    #get infoID vectors
    infoID_vectors = temp[infoIDs].values
    print("\nShape of infoIDs vecs: %s" %str(infoID_vectors.shape))

    #iter
    num_records = infoID_vectors.shape[0]
    print("\nnum_records: %d" %num_records)

    #nodeIDs
    nodeUserIDs = list(temp["nodeUserID"])
    nodeTimes = list(temp["nodeTime"])

    for i,infoID_count_vec in enumerate(infoID_vectors):
        # print("\nOn infoID count vec: %d" %i)
        nodeTime = nodeTimes[i]
        nodeUserID = nodeUserIDs[i]

        cur_infoID_emb_list = []

        for j, infoID_count in enumerate(infoID_count_vec):

            #get which infoID it is
            if infoID_count > 0:
                infoID = infoIDs[j]
                # print("\nInfo ID vec %d, idx %d has %d %s" %(i, j, infoID_count, infoID))

                #get them from glove
                cur_infoID_emb_list.append(infoID_count * glove_df[infoID].values)
                # for c in range(infoID_count):
                #     cur_infoID_emb_list.append(glove_df[infoID].values)

        #now avg
        cur_infoID_record_vec = np.mean(cur_infoID_emb_list, axis=0)
        # print("\nCurrent shape of cur_infoID_record_vec: %s" %str(cur_infoID_record_vec.shape))
        glove_rows.append(np.asarray([nodeTime, nodeUserID] + list(cur_infoID_record_vec)))

        if i%MOD_NUM == 0:
            print("Processed %d out of %d records"%(i, num_records))

    glove_rows = np.asarray(glove_rows)
    print("\nShape of glove rows: %s" %str(glove_rows.shape))

    #make cols
    cols = ["nodeTime", "nodeUserID"] + ["g%d"%i for i in range(glove_d)]
    print("\nGlove final cols")
    print(cols)


    glove_nodeID_df = pd.DataFrame(data=glove_rows, columns=cols)

    #count up nodeID infoIDs
    if count_infoIDs == True:
        glove_nodeID_df = glove_nodeID_df.merge(infoID_count_df, on=["nodeUserID","nodeTime"], how="inner").drop_duplicates().reset_index(drop=True)

    print("\nFinal shape of glove_nodeID_df: %s" %str(glove_nodeID_df.shape))
    glove_nodeID_df["nodeTime"] = pd.to_datetime(glove_nodeID_df["nodeTime"],utc=True)
    return glove_nodeID_df

#get infoID counts
def get_cumsum_infoID_count_vectors_per_user_per_timestep(df, infoIDs,add_cumsum_tag=False):
    #assumes you've already set the granularity you want

    # temp = df[["nodeUserID", "nodeTime", "nodeID"]]

    #how many times each user used infoID per day
    for infoID in infoIDs:
        df[infoID] = df["informationID"].isin([infoID]).astype("int32")
        df[infoID] = df.groupby(["nodeTime", "nodeUserID"])[infoID].transform("sum")

    print("\nCurrent infoID count vecs")
    temp = df[["nodeUserID", "nodeTime"] + infoIDs].drop_duplicates()
    print(temp)

    #get cumsum
    cumsum_infoID_cols = []
    for infoID in infoIDs:

        if add_cumsum_tag == True:
            new_col = "cumsum_" + infoID
        else:
            new_col = infoID
        temp[new_col] = temp.groupby(["nodeUserID"])[infoID].transform("cumsum")
        cumsum_infoID_cols.append(new_col)
        print(temp[new_col].value_counts())

    if add_cumsum_tag == False:
        temp = temp.drop(infoIDs, axis=1)

    # if add_cumsum_tag == True:
    #     merge_cols = ["nodeUserID", "nodeTime"]
    # else:
    #     merge_cols = ["nodeUserID", "nodeTime"]+ cumsum_infoID_cols
    # df = df.drop(infoIDs, axis=1)

    df = df.merge(temp, on=["nodeUserID", "nodeTime"], how="inner")

    return df,cumsum_infoID_cols

def get_cumsum_total_actions(df):
    num_records = df.shape[0]

    #get total num activities
    df["nodeTime_str"] = df["nodeTime"].astype(str)
    df["total_actions"] = df.groupby(["nodeUserID"])["nodeTime_str"].transform("count")

    #get cumsum
    temp = df[["nodeUserID", "nodeTime", "total_actions"]].drop_duplicates()
    df["cumsum_total_user_actions"] = df.groupby(["nodeUserID", "nodeTime"])["total_actions"].transform("cumsum")
    temp = temp.drop("total_actions", axis=1)
    df = df.merge(temp, on=["nodeUserID", "nodeTime"], how="inner")
    df = df.drop("nodeTime_str", axis=1)
    print(df)

    if num_records != df.shape[0]:
        print("\nError! num_records != df.shape[0]")
        print(num_records)
        print(df.shape[0])
        sys.exit(0)

    return df

def get_cumavg_stance(df):
    num_records = df.shape[0]

    #stance stuff
    stance_list = ["stance.?","stance.am","stance.pm"]

    #split dfs
    twitter_df = df[df["platform"]=="twitter"]
    df = df[df["platform"]=="youtube"]

    #create fts
    print("\nGetting stance fts...")
    twitter_df["stance.?"] = twitter_df["stance"].isin(["?"]).astype("int32")
    twitter_df["stance.am"] = twitter_df["stance"].isin(["am"]).astype("int32")
    twitter_df["stance.pm"] = twitter_df["stance"].isin(["pm"]).astype("int32")

    #recombine
    df = pd.concat([df, twitter_df])
    df = df.sort_values("nodeTime").reset_index(drop=True)
    print(df)

    temp = df[["nodeUserID", "nodeID","nodeTime","cumsum_num_active_timesteps"] + stance_list].drop_duplicates().reset_index(drop=True)

    # temp["cumsum_num_user_nodeIDs"] = temp.groupby(["nodeUserID"])["nodeID"].transform("cumcount")

    #get avg stance ft daily for a user
    new_stance_cols = []
    for stance_col in stance_list:
        new_col = "daily_user_avg_%s"%stance_col
        temp[new_col] = temp.groupby(["nodeUserID", "nodeTime"])[stance_col].transform("mean")
        new_stance_cols.append(new_col)

    temp = temp[["nodeUserID", "nodeTime","cumsum_num_active_timesteps"] + new_stance_cols].drop_duplicates().reset_index(drop=True)

    #get number of user appearances
    cumavg_stance_cols = []
    for stance_col in stance_list:
        new_col = "cumavg_daily_user_%s"%stance_col
        temp[new_col] = temp["daily_user_avg_%s"%stance_col]/temp["cumsum_num_active_timesteps"]
        cumavg_stance_cols.append(new_col)

    temp = temp[["nodeUserID", "nodeTime"] + cumavg_stance_cols]


    #merge
    df = df.merge(temp, on=["nodeUserID", "nodeTime"], how="inner")



    if num_records != df.shape[0]:
        print("\nError! num_records != df.shape[0]")
        print(num_records)
        print(df.shape[0])
        sys.exit(0)

    return df,cumavg_stance_cols

def get_cumsum_user_influence(df, MOD_NUM=1000):
    num_records = df.shape[0]

    df = get_parentUserID_col(df)
    df = get_rootUserID_col(df)

    date_to_user_to_inf_dict = {}
    unique_dates = list(df["nodeTime"].unique())

    node_users = list(df["nodeUserID"])
    root_users = list(df["rootUserID"])
    parent_users = list(df["parentUserID"])
    all_dates = list(df["nodeTime"])

    #num recs
    num_records = df.shape[0]

    for i in range(num_records):

        node_user = node_users[i]
        parent_user = parent_users[i]
        root_user = root_users[i]
        date = all_dates[i]

        if date not in date_to_user_to_inf_dict:
            date_to_user_to_inf_dict[date] = {}

        if node_user not in date_to_user_to_inf_dict[date]:
            date_to_user_to_inf_dict[date][node_user] = 0

        if parent_user != node_user:
            if parent_user not in date_to_user_to_inf_dict[date]:
                date_to_user_to_inf_dict[date][parent_user] = 1
            else:
                date_to_user_to_inf_dict[date][parent_user] +=1
        if root_user != node_user:
            if root_user not in date_to_user_to_inf_dict[date]:
                date_to_user_to_inf_dict[date][root_user] = 1
            else:
                date_to_user_to_inf_dict[date][root_user] +=1

        if i%MOD_NUM == 0:
            print("Processed user %d of %d infl. info" %(i, num_records))

    #put the influences in df
    temp = df[["nodeUserID", "nodeTime"]].drop_duplicates().reset_index(drop=True)
    node_users = temp["nodeUserID"]
    dates = temp["nodeTime"]

    num_temp_records = temp.shape[0]
    print("\nnum_temp_records: %d" %num_temp_records)

    infl_list = []

    print("\nAdding influence recordings back to df...")
    for i in range(num_temp_records):
        date = dates[i]
        node_user = node_users[i]
        infl_list.append(date_to_user_to_inf_dict[date][node_user])

        if i%MOD_NUM == 0:
            print("Processed user %d of %d infl. info" %(i, num_records))

    temp["cumsum_user_influence"] = infl_list

    print("\nmerging user infl with orig df...")
    df = df.merge(temp, on=["nodeUserID", "nodeTime"], how="inner")
    df = df.reset_index(drop=True)

    if num_records != df.shape[0]:
        print("\nError! num_records != df.shape[0]")
        print(num_records)
        print(df.shape[0])
        sys.exit(0)

    return df


def get_cumsum_num_active_timesteps(df):
    num_records = df.shape[0]
    #get cumsum of user active timesteps
    #unique timesteps
    temp = df[["nodeUserID", "nodeTime"]].drop_duplicates().reset_index(drop=True)
    # temp["num_active_timesteps"] = 1
    temp["cumsum_num_active_timesteps"] = temp.groupby(["nodeUserID"])["nodeUserID"].transform("cumcount") + 1
    df = df.merge(temp, on=["nodeUserID", "nodeTime"],how="inner").reset_index(drop=True)
    print("cumsum_num_active_timesteps")
    print(df["cumsum_num_active_timesteps"])
    # df = df.drop(["num_active_timesteps"], axis=1)
    print(df)

    if num_records != df.shape[0]:
        print("\nError! num_records != df.shape[0]")
        print(num_records)
        print(df.shape[0])
        sys.exit(0)

    return df

def convert_infoID_vectors_to_glove_embs(df, glove_df,infoIDs,glove_d = 25,count_infoIDs=True, MOD_NUM=5000):
    print("\nConverting infoID vectors to glove embs...")

    #get new cols
    desired_cols = ["nodeID"] + infoIDs

    #count up nodeID infoIDs
    if count_infoIDs == True:
        df["num_infoIDs"] = df[infoIDs].sum(axis=1)
        infoID_count_df = df[["nodeID", "num_infoIDs"]].drop_duplicates().reset_index(drop=True)

    #get nodeIDs
    temp = df[desired_cols].drop_duplicates().reset_index(drop=True)
    print(temp)

    #save glove rows here
    #will use for df later
    glove_rows = []

    #get infoID vectors
    infoID_vectors = temp[infoIDs].values
    print("\nShape of infoIDs vecs: %s" %str(infoID_vectors.shape))

    #iter
    num_records = infoID_vectors.shape[0]
    print("\nnum_records: %d" %num_records)

    #nodeIDs
    nodeIDs = list(temp["nodeID"])

    for i,infoID_count_vec in enumerate(infoID_vectors):
        # print("\nOn infoID count vec: %d" %i)
        nodeID = nodeIDs[i]

        cur_infoID_emb_list = []

        for j, infoID_count in enumerate(infoID_count_vec):

            #get which infoID it is
            if infoID_count > 0:
                infoID = infoIDs[j]
                # print("\nInfo ID vec %d, idx %d has %d %s" %(i, j, infoID_count, infoID))

                #get them from glove
                for c in range(infoID_count):
                    cur_infoID_emb_list.append(glove_df[infoID].values)

        #now avg
        cur_infoID_record_vec = np.mean(cur_infoID_emb_list, axis=0)
        # print("\nCurrent shape of cur_infoID_record_vec: %s" %str(cur_infoID_record_vec.shape))
        glove_rows.append(np.asarray([nodeID] + list(cur_infoID_record_vec)))

        if i%MOD_NUM == 0:
            print("Processed %d out of %d records"%(i, num_records))

    glove_rows = np.asarray(glove_rows)
    print("\nShape of glove rows: %s" %str(glove_rows.shape))

    #make cols
    cols = ["nodeID"] + ["g%d"%i for i in range(glove_d)]
    print("\nGlove final cols")
    print(cols)


    glove_nodeID_df = pd.DataFrame(data=glove_rows, columns=cols)

    #count up nodeID infoIDs
    if count_infoIDs == True:
        glove_nodeID_df = glove_nodeID_df.merge(infoID_count_df, on="nodeID", how="inner").drop_duplicates().reset_index(drop=True)

    print("\nFinal shape of glove_nodeID_df: %s" %str(glove_nodeID_df.shape))
    return glove_nodeID_df

def get_infoID_scores(df):
    # infoID_result_list = []
    actual_volume_list = []
    rmse_list = []
    mape_list = []
    pred_volume_list = []

    print("\nGetting dicts...")
    actual_dict = convert_df_2_cols_to_dict(df, "infoID", "actual_volume")
    pred_dict = convert_df_2_cols_to_dict(df, "infoID", "pred_volume")
    infoIDs = list(df["infoID"])


    for infoID in infoIDs:
        pred_volume = pred_dict[infoID]
        actual_volume = actual_dict[infoID]

        try:
            pred_volume =pred_dict[infoID]
        except KeyError:
            pred_dict[infoID] = 0
            pred_volume = 0

        try:
            actual_volume = actual_dict[infoID]
        except KeyError:
            actual_dict[infoID] = 0
            actual_volume = 0

        actual_volume_list.append(actual_volume)
        pred_volume_list.append(pred_volume)

        #rmse
        cur_rmse = mean_squared_error([actual_volume], [pred_volume], squared=False)
        rmse_list.append(cur_rmse)

        #mape
        cur_mape = mean_absolute_percentage_error(actual_volume, pred_volume)
        mape_list.append(cur_mape)

    #add results
    df["rmse"] = rmse_list
    df["mape"] = mape_list
    df = df[["infoID",  "pred_volume","actual_volume","mape","rmse"]]
    return df.sort_values("mape").reset_index(drop=True)

def calculate_winner(df, nn_col,baseline_col,metric="mape"):

    nn_vals = df[nn_col]
    base_vals = df[baseline_col]
    nn_wins_list = []
    for nn,b in zip(nn_vals, base_vals):
        if nn < b:
            nn_wins_list.append(1)
        elif nn== b:
            nn_wins_list.append("tie")
        else:
            nn_wins_list.append(0)



    tag = "nn_" + metric + "_wins"
    df[tag] = nn_wins_list
    temp = df[df[tag] != "tie"]
    num_wins = temp[tag].sum()
    num_losses = temp[tag].shape[0] - num_wins
    freq = num_wins/(1.0 * temp[tag].shape[0])
    print("%d total records" %(df.shape[0]))
    print("\n%s won %d out of %d times, or %.4f of the time"%(nn_col, num_wins, temp[tag].shape[0], freq))
    num_ties = df[df[tag]=="tie"].shape[0]
    print("%d ties"%num_ties)
    return df

def calculate_winner_v2(df, nn_col,baseline_col,metric="mape"):

    nn_vals = df[nn_col]
    base_vals = df[baseline_col]
    nn_wins_list = []
    for nn,b in zip(nn_vals, base_vals):
        if nn < b:
            nn_wins_list.append(1)
        elif nn== b:
            nn_wins_list.append("tie")
        else:
            nn_wins_list.append(0)



    tag = "nn_" + metric + "_wins"
    df[tag] = nn_wins_list
    temp = df[df[tag] != "tie"]
    num_wins = temp[tag].sum()
    num_losses = temp[tag].shape[0] - num_wins
    freq = num_wins/(1.0 * temp[tag].shape[0])
    print("%d total records" %(df.shape[0]))
    print("\n%s won %d out of %d times, or %.4f of the time"%(nn_col, num_wins, temp[tag].shape[0], freq))
    num_ties = df[df[tag]=="tie"].shape[0]
    print("%d ties"%num_ties)

    result_dict = {}
    result_dict["num_ties"] = num_ties
    result_dict["freq"] = freq
    result_dict["num_wins"] = num_wins
    result_dict["num_losses"] = num_losses
    result_dict["total_trials"] = df.shape[0]
    return df,result_dict

def remove_unnamed(df):
    if "Unnamed: 0" in list(df):
        df = df.drop("Unnamed: 0", axis=1)
    return df

def save_pickle(data, output_fp):
    print("\nSaving to %s..."%output_fp)
    with open(output_fp, 'wb') as handle:
        pickle.dump(data, handle)
    print("Saved pickle!")

def load_data_from_pickle(data_fp):
    print("\nGetting data from %s..."%data_fp)
    file = open(data_fp,'rb')
    data = pickle.load(file)
    print("Got data!")
    return data

def insert_time_features_into_df_v2(df,  TIME_FEATURES_TO_GET, timecol="nodeTime"):

    ALL_POSSIBLE_TIME_FTS=['dayofweek','quarter','month','year',
               'dayofyear','dayofmonth','weekofyear', 'hour']

    df[timecol] = pd.to_datetime(df[timecol], utc=True)

    df["dayofweek"] = df[timecol].dt.dayofweek
    df["quarter"] = df[timecol].dt.quarter
    df["month"] = df[timecol].dt.month
    df["year"] = df[timecol].dt.year
    df["dayofyear"] = df[timecol].dt.dayofyear
    df["dayofmonth"] = df[timecol].dt.day
    df["weekofyear"]= df[timecol].dt.weekofyear
    df["hour"]= df[timecol].dt.hour

    for ft in ALL_POSSIBLE_TIME_FTS:
        if ft not in TIME_FEATURES_TO_GET:
            if ft in list(df):
                df = df.drop(ft, axis=1)

    return df

def convert_df_2_cols_to_dict(df, key_col, val_col):
    return pd.Series(df[val_col].values,index=df[key_col]).to_dict()

def get_fts_from_fp(input_fp):
    model_fts = []
    with open(input_fp) as f:
        for line in f:
            line = line.replace("\n","")
            model_fts.append(line)
            print(line)
    return model_fts

def create_1hot_vector_dict(cols):
    hot_dict = {}
    for i,col in enumerate(cols):
        hot_vector = [0 for i in range(len(cols))]
        hot_vector[i]=1
        hot_dict[col] = hot_vector
        print("%s: %s"%(str(col), str(hot_vector)))
    return hot_dict

def get_glove_fts(glove_dim):
    return ["g%d"%i for i in range(glove_dim)]

def get_gdelt_basic_fts():
    return ["AvgTone", "GoldsteinScale", "NumMentions"]

def get_twitter_actions():
    return ["twitter_tweet", "twitter_retweet", "twitter_reply", "twitter_quote"]

def get_youtube_actions():
    return ["youtube_video", "youtube_comment"]

def get_cp4_actions():
    return list(get_twitter_actions()) + list(get_youtube_actions())

def make_platform_to_action_dict():
    action_dict = {"twitter": get_twitter_actions(), "youtube":get_youtube_actions()}
    return action_dict

def get_reddit_actions():
    return ["reddit_post", "reddit_comment"]

def convert_pred_array_to_df_for_2d_non_nn(y_pred, fts,OUTPUT_SIZE):
    #get shape
    # shape = y_pred.shape

    # if len(shape) == 3:
    #     new_shape = (shape[0] * shape[1], shape[2])
    #     y_pred = y_pred.reshape(new_shape)

    y_pred = y_pred.reshape((y_pred.shape[0] * OUTPUT_SIZE, len(fts)))

    pred_dict = {}
    for i,ft in enumerate(fts):
        cur_array = y_pred[:, i].flatten()
        pred_dict[ft] = cur_array

    df = pd.DataFrame(data=pred_dict)
    return df

#train_df,test_df = get_train_and_test_dfs(df,test_start, test_end)

def split_dfs_into_train_and_test_with_nodeID_hack(df, test_start, test_end,IO_TUPLE,DEBUG_PRINT=False):

    if "nodeID" not in list(df):
        df["nodeID"] = [i for i in range(df.shape[0])]
    # if DEBUG_PRINT == False:
    train_df,test_df=split_dfs_into_train_and_test_v2_5_26_no_print(df, test_start, test_end,IO_TUPLE)
    # else:
    #     train_df,test_df=split_dfs_into_train_and_test_v2_5_26(df, test_start, test_end,IO_TUPLE)

    #train_df,test_df=split_dfs_into_train_and_test_v2_5_26(df, test_start, test_end,IO_TUPLE)

    train_df = train_df.drop(["nodeID"], axis=1)
    test_df = test_df.drop(["nodeID"], axis=1)
    # print(train_df)
    # print(test_df)

    return train_df,test_df

def get_train_and_test_dfs(df,test_start, test_end):

    test_df = config_df_by_dates(df,test_start, test_end)
    test_size = test_df.shape[0]
    train_size = df.shape[0] - test_size
    train_df = df.head(train_size)

    print(train_df)
    print(test_df)
    return train_df,test_df

def fix_dates(df,date_col="nodeTime"):

    new_dates = []
    dates = list(df[date_col])
    print("\nGetting dates...")
    for date in dates:
        date = str(date)
        # print(date)
        year = date[:4]
        month = date[4:6]
        day = date[6:8]
        hour = date[8:10]
        minute = date[10:12]
        sec = date[12:]

        timestamp = "%s-%s-%s %s:%s:%s"%(year,month,day,hour,minute,sec)
        # print(timestamp)
        new_dates.append(timestamp)
    df[date_col] = new_dates
    df[date_col] = pd.to_datetime(df[date_col], utc=True)
    df = df.sort_values(date_col).reset_index(drop=True)

    return df

#count em up
def get_infoID_count_by_gran(df, GRAN, infoIDs,kickout_other_cols=True):

    for infoID in infoIDs:
        df[infoID] = df["informationID"].isin([infoID]).astype("int32")
        df[infoID] = df.groupby(["nodeTime"])[infoID].transform("sum")

    if kickout_other_cols == True:
        df = df[["nodeTime"] + infoIDs]
        df = df.drop_duplicates().reset_index(drop=True)

    return df

def get_infoID_count_by_gran_v2_verify(df, GRAN, infoIDs,kickout_other_cols=True):

    infoID_count_dict = {}
    for infoID in infoIDs:
        infoID_gt_count = df[df["informationID"]==infoID].shape[0]
        infoID_count_dict[infoID] = infoID_gt_count

    for infoID in infoIDs:
        df[infoID] = df["informationID"].isin([infoID]).astype("int32")
        df[infoID] = df.groupby(["nodeTime"])[infoID].transform("sum")

    if kickout_other_cols == True:
        df = df[["nodeTime"] + infoIDs]
        df = df.drop_duplicates().reset_index(drop=True)

    #check counts
    for infoID in infoIDs:
        infoID_gt_count = infoID_count_dict[infoID]
        my_infoID_count = df[infoID].sum()
        print("infoID_gt_count: %d "%infoID_gt_count)
        print("my_infoID_count: %d "%my_infoID_count)
        if infoID_gt_count != my_infoID_count:
            print("Error! Counts are different! Exiting")
            sys.exit(0)
    print("infoID counts are ok! Continuing")

    return df

def get_action_counts_by_gran_v2_verify(df, GRAN,actionTypes,kickout_other_cols=True):

    #to verify
    action_count_dict = {}
    for action in actionTypes:
        action_gt_count = df[df["actionType"]==action].shape[0]
        action_count_dict[action] = action_gt_count

    for actionType in actionTypes:
        df[actionType] = df["actionType"].isin([actionType]).astype("int32")
        df[actionType] = df.groupby(["nodeTime"])[actionType].transform("sum")

    if kickout_other_cols == True:
        df = df[["nodeTime"] + actionTypes]
        df = df.drop_duplicates().reset_index(drop=True)

    #check counts
    for action in actionTypes:
        action_gt_count = action_count_dict[action]
        my_action_count = df[action].sum()
        print("action_gt_count: %d "%action_gt_count)
        print("my_action_count: %d "%my_action_count)
        if action_gt_count != my_action_count:
            print("Error! Counts are different! Exiting")
            sys.exit(0)
    print("Action counts are ok! Continuing")

    return df


def get_action_counts_by_gran(df, GRAN,actionTypes,kickout_other_cols=True):

    for actionType in actionTypes:
        df[actionType] = df["actionType"].isin([actionType]).astype("int32")
        df[actionType] = df.groupby(["nodeTime"])[actionType].transform("sum")

    if kickout_other_cols == True:
        df = df[["nodeTime"] + actionTypes]
        df = df.drop_duplicates().reset_index(drop=True)

    return df

def create_blank_date_df(start,end,GRAN):

    dates = pd.date_range(start,end,freq=GRAN)
    blank_date_df = pd.DataFrame(data={"nodeTime":dates})
    blank_date_df["nodeTime"] = pd.to_datetime(blank_date_df["nodeTime"], utc=True)
    return blank_date_df

def convert_str_to_bool(bool_str):
    if (bool_str == "True") or (bool_str ==1):
        return True
    return False

# def custom_mse(true,pred):

#     diff = true - pred

#     greater = K.greater(diff,0)
#     greater = K.cast(greater, K.floatx()) #0 for lower, 1 for greater
#     greater = greater * (COST-1)
#     greater = greater + 1                 #1 for lower, 2 for greater

#     #use some kind of loss here, such as mse or mae, or pick one from keras
#     #using mse:
#     return K.mean(greater*K.square(diff))



# def custom_mae(true,pred):

#     diff = true - pred

#     greater = K.greater(diff,0)
#     greater = K.cast(greater, K.floatx()) #0 for lower, 1 for greater
#     greater = greater * (COST-1)
#     greater = greater + 1                 #1 for lower, 2 for greater

#     #use some kind of loss here, such as mse or mae, or pick one from keras
#     #using mse:
#     return K.mean(greater*K.abs(diff))


# def custom_mse(COST):
#     def actual_custom_mse(true,pred):

#         diff = true - pred

#         greater = K.greater(diff,0)
#         greater = K.cast(greater, K.floatx()) #0 for lower, 1 for greater
#         greater = greater * (COST-1)
#         greater = greater + 1                 #1 for lower, 2 for greater

#         #use some kind of loss here, such as mse or mae, or pick one from keras
#         #using mse:
#         return K.mean(greater*K.square(diff))
#     return actual_custom_mse

# def custom_mae(COST):
#     def actual_custom_mae(true,pred):

#         diff = true - pred

#         greater = K.greater(diff,0)
#         greater = K.cast(greater, K.floatx()) #0 for lower, 1 for greater
#         greater = greater * (COST-1)
#         greater = greater + 1                 #1 for lower, 2 for greater

#         #use some kind of loss here, such as mse or mae, or pick one from keras
#         #using mse:
#         return K.mean(greater*K.abs(diff))
#     return actual_custom_mae

def build_simple_dense_model_with_cost_loss(ARCH_LIST,num_inputs,num_outputs,l1,l2,LOSS,OPTIMIZER,REG_OUTPUT,COST):

    #make model
    INPUT_SHAPE=(num_inputs,)

    model_input = Input(shape=INPUT_SHAPE)
    num_layers_with_input = len(ARCH_LIST) + 1
    arch_list = [None] * (num_layers_with_input + 1)

    arch_list[0] = model_input
    for i in range(num_layers_with_input - 1):
        layer_tuple = ARCH_LIST[i]
        layer_type = layer_tuple[0]
        if layer_type == "f":
            num_units = layer_tuple[1]
            arch_list[i+1] = Dense(num_units,activation="tanh", activity_regularizer=Reg.l1_l2(l1=l1, l2=l2))(arch_list[i])

    if REG_OUTPUT == False:
        l1 = 0
        l2 = 0
    arch_list[num_layers_with_input] = Dense(num_outputs, activity_regularizer=Reg.l1_l2(l1=l1, l2=l2))(arch_list[num_layers_with_input - 1])
    model = Model(input=model_input, output=arch_list[num_layers_with_input])
    print(model.summary())
    model.compile(loss=LOSS, optimizer=OPTIMIZER)

    return model

def listify_df(df, groupby_cols, col_to_listify):
    print("\nListifying %s..."%col_to_listify)
    df = df.groupby(groupby_cols)[col_to_listify].apply(list).reset_index(name=col_to_listify)
    print("Done with listify!")
    print(df)
    return df

def split_infoIDs_into_keywords(infoIDs):

    print("\nGetting split infoIDs...")
    key_words = []
    for infoID in infoIDs:
        if "/" in infoID:
            infoID_list = infoID.split("/")
            for split_id in infoID_list:
                if "_" in split_id:
                    split_id_list = split_id.split("_")
                    for s in split_id_list:
                        key_words.append(s)
                        print(s)
                else:
                    key_words.append(split_id)
                    print(split_id)
        else:
            key_words.append(infoID)
            print(infoID)
    return key_words


def check_if_infoIDs_in_model(infoIDs, model):

    keywords = split_infoIDs_into_keywords(infoIDs)
    keywords = set(keywords)
    num_keywords = len(keywords)
    print("\nGot %d keywords"%num_keywords)

    not_in = set()

    for word in keywords:
        try:
            emb = model[word]
        except:
            print("\n%s not in model!"%word)
            not_in.add(word)

    num_missing = len(not_in)
    print("%d words not in set"%num_missing)
    print(not_in)

def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.",len(model)," words loaded!")
    return model

def make_cp4_convert_dict(mini_convert_dict, infoIDs):
    convert_dict = dict(mini_convert_dict)
    for infoID in infoIDs:
        if infoID not in convert_dict:
            convert_dict[infoID] = infoID
    return convert_dict

def get_reverse_dict(main_dict):

    new_dict = {}
    for key,val in main_dict.items():
        new_dict[val] = key
    return new_dict

def hyphenate_infoID_dict(infoIDs):

    h_dict = {}
    for infoID in infoIDs:
        if "/" in infoID:
            h_dict[infoID] = infoID.replace("/", "_")
        else:
            h_dict[infoID] =infoID

    return h_dict


def convert_infoID_to_underscore_form(infoID):
    if "/" in infoID:
        return infoID.replace("/", "_")
    return infoID

def get_46_cp4_infoIDs_and_global():
    return get_46_cp4_infoIDs() + ["global"]

def get_46_cp4_infoIDs():

    #took out 1 that only appeared once in 2017

    infoIDs=[
    'other/chavez/pro',
    'military',
    'other/chavez',
    'guaido/legitimate',
    'other/chavez/anti',
    'international/respect_sovereignty',
    'other/anti_socialism',
    'maduro/dictator',
    'violence',
    'other/restore_democracy',
    'international/aid',
    'other/planned_coup',
    'crisis',
    'violence/against_opposition',
    'crisis/lack_essentials',
    'arrests/opposition',
    'arrests',
    'maduro/narco',
    'violence/against_opposition/protesters',
    'protests',
    'maduro/illegitimate',
    'other/media_bias',
    'guaido/us_support',
    'maduro/events',
    'guaido/legitimate/international',
    'maduro/cuba_support',
    'international/aid_rejected',
    'other/censorship_outage',
    'assembly/legitimate',
    'arrests/opposition/protesters',
    'international/military',
    'maduro/legitimate',
    'international/us_sanctions',
    'military/desertions',
    # 'other/anti_capitalism',
    'international/emigration',
    'maduro/russia_support',
    'maduro/illegitimate/international',
    'international/break_us_relations',
    'arrests/opposition/media',
    'crisis/looting',
    'maduro/events/pro',
    'maduro/events/anti',
    'maduro/legitimate/international',
    'other/request_observers',
    'violence/against_maduro',
    'guaido/illegitimate']

    return infoIDs

def get_1hot_vectors(fts):

    hot_dict = {}
    for i,ft in enumerate(fts):
        hot_dict[ft] = [0 for j in range(len(fts))]
        hot_dict[ft][i] = 1
        print("%s: %s" %(ft, str(hot_dict[ft][i])))

    return hot_dict


def get_cp4_infoIDs():

    infoIDs=[
    'other/chavez/pro',
    'military',
    'other/chavez',
    'guaido/legitimate',
    'other/chavez/anti',
    'international/respect_sovereignty',
    'other/anti_socialism',
    'maduro/dictator',
    'violence',
    'other/restore_democracy',
    'international/aid',
    'other/planned_coup',
    'crisis',
    'violence/against_opposition',
    'crisis/lack_essentials',
    'arrests/opposition',
    'arrests',
    'maduro/narco',
    'violence/against_opposition/protesters',
    'protests',
    'maduro/illegitimate',
    'other/media_bias',
    'guaido/us_support',
    'maduro/events',
    'guaido/legitimate/international',
    'maduro/cuba_support',
    'international/aid_rejected',
    'other/censorship_outage',
    'assembly/legitimate',
    'arrests/opposition/protesters',
    'international/military',
    'maduro/legitimate',
    'international/us_sanctions',
    'military/desertions',
    'other/anti_capitalism',
    'international/emigration',
    'maduro/russia_support',
    'maduro/illegitimate/international',
    'international/break_us_relations',
    'arrests/opposition/media',
    'crisis/looting',
    'maduro/events/pro',
    'maduro/events/anti',
    'maduro/legitimate/international',
    'other/request_observers',
    'violence/against_maduro',
    'guaido/illegitimate']

    return infoIDs

def get_latest_time_delta(x_test,time_ft_idx,scaler, MINMAX, NUM_LOG_NORMS):

    temp_x_test = x_test.copy()

    print("\nDe-Log normalizing data...")
    for LN in range(NUM_LOG_NORMS):
        temp_x_test = np.expm1(temp_x_test)
    print("Done with de-log norm.")

    if MINMAX==True:
        temp_x_test = denormalize_single_array(temp_x_test, scaler)

    #get latest time delta
    latest_time_delta = temp_x_test[-1][-1][time_ft_idx]



    return latest_time_delta

def get_latest_nodeTime_from_df(test_df):

    test_df["nodeTime"] = pd.to_datetime(test_df["nodeTime"], utc=True)
    test_df = test_df.sort_values("nodeTime").reset_index(drop=True)

    return test_df["nodeTime"].iloc[-1]

def get_latest_nodeTime(current_nodeTime, x_test,time_ft_idx,scaler, MINMAX, NUM_LOG_NORMS,FORCE_TIME_DELTA_ABS_VAL):

    temp_x_test = x_test.copy()

    print("\nDe-Log normalizing data...")
    for LN in range(NUM_LOG_NORMS):
        temp_x_test = np.expm1(temp_x_test)
    print("Done with de-log norm.")

    if MINMAX==True:
        temp_x_test = denormalize_single_array(temp_x_test, scaler)

    #get latest time delta
    time_delta_array = temp_x_test[-1,:,time_ft_idx]
    print("\ntime_delta_array: %s"%str(time_delta_array))

    if FORCE_TIME_DELTA_ABS_VAL == True:
    	time_delta_array = np.absolute(time_delta_array)
    	print("\ntime_delta_array abs val: %s"%str(time_delta_array))
    else:
    	time_delta_array[time_delta_array<0] = 0
    	print("\ntime_delta_array after zero change: %s"%str(time_delta_array))
    time_delta_sum = np.sum(time_delta_array)
    print("\ntime_delta_sum: %.4f"%time_delta_sum)

    current_nodeTime = pd.to_datetime(current_nodeTime, utc=True)
    time_delta_sum = pd.to_timedelta("%s D"%str(time_delta_sum))
    current_nodeTime = current_nodeTime + time_delta_sum
    print("\ncurrent_nodeTime: %s" %current_nodeTime)
    return current_nodeTime

def simulate_fixed_nodeTime_deltas(model,latest_init_condit_nodeTime, x_init_condit, MINMAX,scaler, MINMAX_TARGET,fts,NUM_LOG_NORMS, initial_start_time,test_end, INPUT_SIZE,OUTPUT_SIZE,FORCE_TIME_DELTA_ABS_VAL,FORCE_PRED_ABS_VALUE_DURING_PRED,STOP_SIMULATION_ITERATION,time_delta_ft="nodeTime_delta"):
    print("\n########################### STARTING SIMULATION ###########################\n")

    current_nodeTime = latest_init_condit_nodeTime
    print("\ncurrent_nodeTime: %s" %str(current_nodeTime))

    #get test end
    test_end = pd.to_datetime(test_end , utc=True)
    print("test_end: %s" %str(test_end) )

    #just to be sure!
    x_init_condit = x_init_condit[:1, :, :]
    print("\nx_init_condit shape: %s" %str(x_init_condit.shape))

    for i,ft in enumerate(fts):
        if ft==time_delta_ft:
            time_ft_idx = i
            print("\nTime ft is %s at idx %d"%(time_delta_ft, time_ft_idx))
            break

        print("\nError! Could not find %s idx. Terminating" %time_delta_ft)
        sys.exit(0)



    # #first we need the latest time delta
    # latest_time_delta = get_latest_time_delta(x_init_condit,time_ft_idx,scaler, MINMAX, NUM_LOG_NORMS)
    # print("\nBeginning time delta: %.4f" %latest_time_delta)

    # #get our ending time delta
    # ending_time_delta = get_difference_between_two_dates(initial_start_time, test_end)
    # print("\nending_time_delta: %.4f"%ending_time_delta)

    #save sims here
    y_pred = []

    #loop and simulate
    cur_x_test = x_init_condit
    num_loops = 0
    while (1):
        print("\nSim loop %d; current_nodeTime: %s" %(num_loops,str(current_nodeTime)))
        cur_y_pred = model.predict(cur_x_test)

        if FORCE_PRED_ABS_VALUE_DURING_PRED == True:
            cur_y_pred = np.absolute(cur_y_pred)

        y_pred.append(np.absolute(np.copy(cur_y_pred)))

        if (MINMAX_TARGET == False) and (MINMAX==True):
            #if y pred is not normalized, we must normalize it
            cur_y_pred = normalize_single_array_with_scaler(cur_y_pred, scaler)

        cur_x_test = [cur_x_test[:,-(INPUT_SIZE - OUTPUT_SIZE):,:], cur_y_pred ]
        cur_x_test = np.concatenate(cur_x_test, axis=1)
        print("Shape of new x_test: %s" %str(cur_x_test.shape))

        current_nodeTime = get_latest_nodeTime(current_nodeTime, cur_x_test,time_ft_idx,scaler, MINMAX, NUM_LOG_NORMS,FORCE_TIME_DELTA_ABS_VAL)

        # sys.exit(0)

        # #get latest time stamp
        # latest_time_delta = get_latest_time_delta(cur_x_test,time_ft_idx,scaler, MINMAX, NUM_LOG_NORMS)
        # if FORCE_TIME_DELTA_ABS_VAL==True:
        #     latest_time_delta = np.absolute(latest_time_delta)

        # if current_nodeTime >= test_end:
        #     print("\n%s >= %s" %(current_nodeTime, test_end))
        #     print("Ending simulation")
        #     break

        if num_loops>=STOP_SIMULATION_ITERATION:
            print("\nReached sim limit of %d. Breaking!"%STOP_SIMULATION_ITERATION)
            break

        num_loops+=OUTPUT_SIZE

    y_pred = np.asarray(y_pred)
    y_pred = y_pred.reshape((y_pred.shape[0], y_pred.shape[2],y_pred.shape[3]))
    print("\nFinal shape of y_pred: %s" %str(y_pred.shape))
    return y_pred

def get_nodeTime_delta_col(df,  time_gran,timecol="nodeTime"):
    df[timecol] = pd.to_datetime(df[timecol], utc=True).reset_index(drop=True)
    df = df.sort_values(timecol).reset_index(drop=True)

    INITIAL_DATE_FOR_TIME_DELTA_MANIPULATION = df[timecol].iloc[0]

    print("Original dates: ")
    ORIGINAL_DATETIME_SERIES = df[timecol].copy()
    print(ORIGINAL_DATETIME_SERIES)
    print("\n\n INITIAL_DATE_FOR_TIME_DELTA_MANIPULATION: %s" %INITIAL_DATE_FOR_TIME_DELTA_MANIPULATION)

    df[timecol] = df[timecol].diff()
    print(df[timecol])
    td_str = "0 %s"%(time_gran)
    # df[timecol][0] =  pd.to_timedelta(td_str)
    df.at[0, timecol] = pd.to_timedelta(td_str)
    # sys.exit(0)
    print("time deltas")
    print(df[timecol] )
    # sys.exit(0)

    #get divide dict
    #get divide dict
    time_delta_div_dict = {
    "seconds" : 1.0,
    "second" : 1.0,
    "s":1.0,
    "minutes" : 60.0,
    "minute":60.0,
    "min":60.0,
    "m":60.0,
    "hours" : 3600.0,
    "hour" : 3600.0,
    "h" : 3600.0,
    "D" : 86400.0,
    "days" : 86400.0,
    "day" : 86400.0,
    "weeks" : 604800.0,
    "week" : 604800.0
    }

    #get div factor
    td_divisor = time_delta_div_dict[time_gran]

    #convert to seconds
    df["nodeTime"] = df["nodeTime"].dt.total_seconds()/td_divisor
    df["nodeTime"] = df["nodeTime"].astype("float64")
    print(df["nodeTime"])
    df["nodeTime"] = pd.to_numeric(df["nodeTime"])

    print(df["nodeTime"])
    df["nodeTime_delta"] = df["nodeTime"].copy()
    df["nodeTime"] = ORIGINAL_DATETIME_SERIES
    # sys.exit(0)
    return df,INITIAL_DATE_FOR_TIME_DELTA_MANIPULATION,ORIGINAL_DATETIME_SERIES

def convert_datetimes_to_timedeltas_v1(df,  time_gran,timecol="nodeTime"):
    df[timecol] = pd.to_datetime(df[timecol], utc=True).reset_index(drop=True)

    INITIAL_DATE_FOR_TIME_DELTA_MANIPULATION = df[timecol].iloc[0]

    print("Original dates: ")
    ORIGINAL_DATETIME_SERIES = df[timecol].copy()
    print(ORIGINAL_DATETIME_SERIES)
    print("\n\n INITIAL_DATE_FOR_TIME_DELTA_MANIPULATION: %s" %INITIAL_DATE_FOR_TIME_DELTA_MANIPULATION)

    df[timecol] = df[timecol].diff()
    print(df[timecol])
    td_str = "0 %s"%(time_gran)
    # df[timecol][0] =  pd.to_timedelta(td_str)
    df.at[0, timecol] = pd.to_timedelta(td_str)
    # sys.exit(0)
    print("time deltas")
    print(df[timecol] )
    # sys.exit(0)

    #get divide dict
    #get divide dict
    time_delta_div_dict = {
    "seconds" : 1.0,
    "second" : 1.0,
    "s":1.0,
    "minutes" : 60.0,
    "minute":60.0,
    "min":60.0,
    "m":60.0,
    "hours" : 3600.0,
    "hour" : 3600.0,
    "h" : 3600.0,
    "H" : 3600.0,
    "D" : 86400.0,
    "days" : 86400.0,
    "day" : 86400.0,
    "weeks" : 604800.0,
    "week" : 604800.0
    }

    #get div factor
    td_divisor = time_delta_div_dict[time_gran]

    #convert to seconds
    df["nodeTime"] = df["nodeTime"].dt.total_seconds()/td_divisor
    df["nodeTime"] = df["nodeTime"].astype("float64")
    print(df["nodeTime"])
    df["nodeTime"] = pd.to_numeric(df["nodeTime"])

    print(df["nodeTime"])
    # sys.exit(0)
    return df,INITIAL_DATE_FOR_TIME_DELTA_MANIPULATION,ORIGINAL_DATETIME_SERIES

#convert to time deltas
def convert_datetimes_to_timedeltas_v2(df,  time_gran):
    df["nodeTime"] = pd.to_datetime(df["nodeTime"], utc=True).reset_index(drop=True)

    INITIAL_DATE_FOR_TIME_DELTA_MANIPULATION = df["nodeTime"].iloc[0]

    print("Original dates: ")
    ORIGINAL_DATETIME_SERIES = df["nodeTime"].copy()
    print(ORIGINAL_DATETIME_SERIES)
    print("\n\n INITIAL_DATE_FOR_TIME_DELTA_MANIPULATION: %s" %INITIAL_DATE_FOR_TIME_DELTA_MANIPULATION)

    # df["nodeTime_delta"] = df[df["nodeTime"]].diff()
    df["nodeTime_delta"] = df["nodeTime"].diff()
    td_str = "0 %s"%(time_gran)
    df["nodeTime_delta"][0] =  pd.to_timedelta(td_str)
    print("time deltas")
    print(df["nodeTime_delta"] )
    # sys.exit(0)

    #get divide dict
    #get divide dict
    time_delta_div_dict = {
    "seconds" : 1.0,
    "second" : 1.0,
    "s":1.0,
    "minutes" : 60.0,
    "minute":60.0,
    "min":60.0,
    "m":60.0,
    "hours" : 3600.0,
    "hour" : 3600.0,
    "h" : 3600.0,
    "D" : 86400.0,
    "days" : 86400.0,
    "day" : 86400.0,
    "weeks" : 604800.0,
    "week" : 604800.0
    }

    #get div factor
    td_divisor = time_delta_div_dict[time_gran]

    #convert to seconds
    df["nodeTime_delta"] = df["nodeTime"].dt.total_seconds()/td_divisor
    df["nodeTime_delta"] = df["nodeTime_delta"].astype("float64")
    print(df["nodeTime_delta"])
    df["nodeTime_delta"] = pd.to_numeric(df["nodeTime_delta"])

    print(df["nodeTime_delta"])
    # sys.exit(0)
    return df,INITIAL_DATE_FOR_TIME_DELTA_MANIPULATION,ORIGINAL_DATETIME_SERIES

def kickout_records(df, kickout_fts, fts):

    for ft in kickout_fts:
        if ft in list(df):
            df = df[df[ft]==0]
    return df

def kickout_ignore_cols(df, ignore_fts,fts):

    for ft in ignore_fts:
        if ft in list(df):
            df = df.drop(ft, axis=1)
    return df

def convert_pred_array_to_df(y_pred, fts):
    #get shape
    shape = y_pred.shape

    if len(shape) == 3:
        new_shape = (shape[0] * shape[1], shape[2])
        y_pred = y_pred.reshape(new_shape)

    pred_dict = {}
    for i,ft in enumerate(fts):
        cur_array = y_pred[:, i].flatten()
        pred_dict[ft] = cur_array

    df = pd.DataFrame(data=pred_dict)
    return df

def convert_pred_array_to_df_and_convert_time_col(y_pred, fts,time_delta_ft,latest_init_condit_nodeTime):

    df = convert_pred_array_to_df(y_pred, fts)
    df[time_delta_ft] = np.absolute(df[time_delta_ft])

    time_deltas = df[time_delta_ft].cumsum()
    print(time_deltas)

    time_deltas = pd.to_timedelta(time_deltas, unit="d")
    latest_init_condit_nodeTime = pd.to_datetime(latest_init_condit_nodeTime, utc=True)
    df["nodeTime"] = time_deltas + latest_init_condit_nodeTime

    return df

def round_y_pred(y_pred, fts,round_ignore_fts):

    shape = y_pred.shape

    if len(shape) == 3:
        new_shape = (shape[0] * shape[1], shape[2])
        y_pred = y_pred.reshape(new_shape)

    for i,ft in enumerate(fts):
        if ft not in round_ignore_fts:
            y_pred[:, i] = np.round(y_pred[:, i], 0)

    y_pred = y_pred.reshape(shape)

    return y_pred



def get_difference_between_two_dates(date1, date2,unit="D"):
    import pandas as pd

    date1 = pd.to_datetime(date1)
    date2 = pd.to_datetime(date2)
    diff = (date2 - date1)/pd.to_timedelta(1, unit=unit)


    return diff

def simulate(model, x_init_condit, MINMAX,scaler, MINMAX_TARGET,fts,NUM_LOG_NORMS, initial_start_time,test_end, INPUT_SIZE,OUTPUT_SIZE,FORCE_TIME_DELTA_ABS_VAL,FORCE_PRED_ABS_VALUE_DURING_PRED,STOP_SIMULATION_ITERATION,time_delta_ft="cascade_start_nodeTime_delta"):
    print("\n########################### STARTING SIMULATION ###########################\n")

    #just to be sure!
    x_init_condit = x_init_condit[:1, :, :]
    print("\nx_init_condit shape: %s" %str(x_init_condit))

    for i,ft in enumerate(fts):
        if ft==time_delta_ft:
            time_ft_idx = i
            print("\nTime ft is %s at idx %d"%(time_delta_ft, time_ft_idx))

    #first we need the latest time delta
    latest_time_delta = get_latest_time_delta(x_init_condit,time_ft_idx,scaler, MINMAX, NUM_LOG_NORMS)
    print("\nBeginning time delta: %.4f" %latest_time_delta)

    #get our ending time delta
    ending_time_delta = get_difference_between_two_dates(initial_start_time, test_end)
    print("\nending_time_delta: %.4f"%ending_time_delta)

    #save sims here
    y_pred = []

    #loop and simulate
    cur_x_test = x_init_condit
    num_loops = 0
    while (1):
        print("\nSim loop %d; Current time delta: %.4f" %(num_loops,latest_time_delta))
        cur_y_pred = model.predict(cur_x_test)

        if FORCE_PRED_ABS_VALUE_DURING_PRED == True:
            cur_y_pred = np.absolute(cur_y_pred)

        y_pred.append(np.absolute(np.copy(cur_y_pred)))

        if (MINMAX_TARGET == False) and (MINMAX==True):
            #if y pred is not normalized, we must normalize it
            cur_y_pred = normalize_single_array_with_scaler(cur_y_pred, scaler)

        cur_x_test = [cur_x_test[:,-OUTPUT_SIZE:,:], cur_y_pred ]
        cur_x_test = np.concatenate(cur_x_test, axis=1)
        print("Shape of new x_test: %s" %str(cur_x_test.shape))

        #get latest time stamp
        latest_time_delta = get_latest_time_delta(cur_x_test,time_ft_idx,scaler, MINMAX, NUM_LOG_NORMS)
        if FORCE_TIME_DELTA_ABS_VAL==True:
            latest_time_delta = np.absolute(latest_time_delta)

        if latest_time_delta >= ending_time_delta:
            print("\n%.4f >= %.4f" %(latest_time_delta, ending_time_delta))
            print("Ending simulation")
            break

        if num_loops>=STOP_SIMULATION_ITERATION:
            print("\nReached sim limit of %d. Breaking!"%STOP_SIMULATION_ITERATION)
            break

        num_loops+=OUTPUT_SIZE

    y_pred = np.asarray(y_pred)
    y_pred = y_pred.reshape((y_pred.shape[0], y_pred.shape[2],y_pred.shape[3]))
    print("\nFinal shape of y_pred: %s" %str(y_pred.shape))
    return y_pred


# from sklearn.utils import check_arrays
def mean_absolute_percentage_error(y_true, y_pred):
    # y_true, y_pred = check_arrays(y_true, y_pred)

    ## Note: does not handle mix 1d representation
    #if _is_1d(y_true):
    #    y_true, y_pred = _check_1d_array(y_true, y_pred)

    try:
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    except ZeroDivisionError:
        return np.inf

def count_unique_users(users):

    users = np.round(users, 0).astype("int32")
    users = pd.Series(users)
    nunique_users = users.nunique()
    return nunique_users

def get_df_result_error_data(y_test,y_pred ,fts,tag, ignore_fts):

    #get total volume results
    predicted_volume_list = []
    actual_volume_list = []

    rmse_list = []
    mape_list = []

    new_fts = []
    for ft in fts:
        if ft not in ignore_fts:
            new_fts.append(ft)

    fts = new_fts




    for i,ft in enumerate(fts):
        cur_y_test = y_test[ft]
        cur_y_pred = y_pred[ft]

        if ft=="nodeUserID":
            print(ft)
            actual_volume = count_unique_users(cur_y_test)
            pred_volume = count_unique_users(cur_y_pred)
        else:
            actual_volume = np.sum(cur_y_test)
            pred_volume = np.sum(cur_y_pred)
        print("\n%s"%ft)
        print("Pred vol: %.4f, Actual vol: %.4f"%(pred_volume, actual_volume))
        predicted_volume_list.append(pred_volume)
        actual_volume_list.append(actual_volume)

        #rmse
        cur_rmse = mean_squared_error([actual_volume], [pred_volume], squared=False)
        rmse_list.append(cur_rmse)

        #mape
        cur_mape = mean_absolute_percentage_error(actual_volume, pred_volume)
        mape_list.append(cur_mape)

    #pred dataframe
    result_df = pd.DataFrame(data={"feature":fts, tag + "_pred": predicted_volume_list, "actual":actual_volume_list, tag + "_rmse":rmse_list, tag + "_mape":mape_list})
    result_df = result_df[["feature", "actual", tag + "_pred",tag + "_mape" ,tag + "_rmse"]]
    result_df = result_df.sort_values(tag + "_mape", ascending=True)

    return result_df

def get_result_error_data(y_test,y_pred ,fts,tag, ignore_fts):

    #get total volume results
    predicted_volume_list = []
    actual_volume_list = []

    rmse_list = []
    mape_list = []

    new_fts = []
    for ft in fts:
        if ft not in ignore_fts:
            new_fts.append(ft)

    fts = new_fts

    for i,ft in enumerate(fts):
        cur_y_test = y_test[:,i]
        cur_y_pred = y_pred[:,i]

        if ft=="nodeUserID":
            new_cur_y_test = []
            new_cur_y_pred = []
            cur_y_test = cur_y_test.reshape((cur_y_test.shape[0], cur_y_test.shape[1] * cur_y_test.shape[2]))
            cur_y_pred = cur_y_pred.reshape((cur_y_pred.shape[0], cur_y_pred.shape[1] * cur_y_pred.shape[2]))
            for j in range(cur_y_test.shape[0]):
                a = cur_y_test[j]
                p = cur_y_pred[j]
                a = count_unique_users(a)
                p = count_unique_users(p)
                new_cur_y_test.append(a)
                new_cur_y_pred.append(p)
            cur_y_test = np.asarray(cur_y_test)
            cur_y_pred = np.asarray(cur_y_pred)
        #rmse
        cur_rmse = mean_squared_error(cur_y_test, cur_y_pred, squared=False)
        rmse_list.append(cur_rmse)

        #mape
        cur_mape = mean_absolute_percentage_error(cur_y_test, cur_y_pred)
        mape_list.append(cur_mape)

    #pred dataframe
    result_df = pd.DataFrame(data={"feature":fts, tag + "_rmse":rmse_list, tag + "_mape":mape_list})
    result_df = result_df[["feature",tag + "_mape" ,tag + "_rmse"]]
    result_df = result_df.sort_values(tag + "_mape", ascending=True)

    return result_df


def count_infoIDs_in_cascade_df(infoIDs, cascade_df):

	#number of infoIDs
	num_cascade_infoIDs = 0
	for infoID in infoIDs:
		try:
			num_cascade_infoIDs+=cascade_df[infoID].sum()
		except KeyError:
			print("KeyError: %s" %infoID)
			continue
	return num_cascade_infoIDs

def get_sliding_window_dfs(orig_df, INPUT_SIZE, OUTPUT_SIZE):
    NUM_ELEMENTS_PER_WINDOW = (INPUT_SIZE + OUTPUT_SIZE)

    #save dfs here
    dfs = []
    LOOPS = NUM_ELEMENTS_PER_WINDOW
    for i in range(LOOPS):
        df = orig_df.copy()
        num_rows = df.shape[0]
        tail_to_remove = i
        df = df.tail(num_rows - tail_to_remove)
        num_rows = df.shape[0]
        head_to_remove = num_rows%NUM_ELEMENTS_PER_WINDOW
        df = df.head(num_rows - head_to_remove)
        df = df.reset_index(drop=True)
        print("Iter %d, df before agg: "%i)
        print(df)

        dfs.append(df.copy())

    return dfs

def convert_df_to_sliding_window_form_diff_x_and_y_fts_SAME_PERIOD(df, INPUT_SIZE, OUTPUT_SIZE, input_features, target_fts,STRIDE=1,MOD_NUM=10000):

    SEQUENCE_SIZE = INPUT_SIZE + OUTPUT_SIZE
    TOTAL_EVENTS = df.shape[0]
    NUM_LOOPS = TOTAL_EVENTS - SEQUENCE_SIZE + 1
    #print("NUM_LOOPS: %d" %NUM_LOOPS)

    #get arrays
    x_data_arrays = df[input_features].values
    y_data_arrays = df[target_fts].values
    #print("\ndata_arrays shape:" )
    #print(x_data_arrays.shape)
    #print(y_data_arrays.shape)
    x_data = []
    y_data = []

    for i in range(NUM_LOOPS):

        x_cur_seq = x_data_arrays[i:i+SEQUENCE_SIZE]
        y_cur_seq = y_data_arrays[i:i+SEQUENCE_SIZE]

        x_seq = x_cur_seq[OUTPUT_SIZE:INPUT_SIZE + OUTPUT_SIZE, :]
        y_seq = y_cur_seq[INPUT_SIZE:INPUT_SIZE+OUTPUT_SIZE, :]
        #if i%MOD_NUM == 0:
            #print("cur_seq shapes: " )
            #print(x_cur_seq.shape)
            #print(y_cur_seq.shape)
            #print("\nCurrent loop: %d of %d" %((i+1), NUM_LOOPS))
            #print("x shape: %s" %str(x_seq.shape))
            #print("y shape: %s" %str(y_seq.shape))
        x_data.append(x_seq)
        y_data.append(y_seq)

    #final data
    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)

    return x_data,y_data

def convert_df_to_sliding_window_form_diff_x_and_y_fts_no_print(df, INPUT_SIZE, OUTPUT_SIZE, input_features, target_fts,STRIDE=1,MOD_NUM=10000):

    SEQUENCE_SIZE = INPUT_SIZE + OUTPUT_SIZE
    TOTAL_EVENTS = df.shape[0]
    NUM_LOOPS = TOTAL_EVENTS - SEQUENCE_SIZE + 1
    #print("NUM_LOOPS: %d" %NUM_LOOPS)

    #get arrays
    x_data_arrays = df[input_features].values
    y_data_arrays = df[target_fts].values
    #print("\ndata_arrays shape:" )
    #print(x_data_arrays.shape)
    #print(y_data_arrays.shape)
    x_data = []
    y_data = []

    for i in range(NUM_LOOPS):

        x_cur_seq = x_data_arrays[i:i+SEQUENCE_SIZE]
        y_cur_seq = y_data_arrays[i:i+SEQUENCE_SIZE]

        x_seq = x_cur_seq[:INPUT_SIZE, :]
        y_seq = y_cur_seq[INPUT_SIZE:INPUT_SIZE+OUTPUT_SIZE, :]
        #if i%MOD_NUM == 0:
            #print("cur_seq shapes: " )
            #print(x_cur_seq.shape)
            #print(y_cur_seq.shape)
            #print("\nCurrent loop: %d of %d" %((i+1), NUM_LOOPS))
            #print("x shape: %s" %str(x_seq.shape))
            #print("y shape: %s" %str(y_seq.shape))
        x_data.append(x_seq)
        y_data.append(y_seq)

    #final data
    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)

    return x_data,y_data

def XGBOOST_convert_df_to_sliding_window_form_diff_x_and_y_fts_no_print_with_idx_dict(df, static_input_fts,INPUT_SIZE, OUTPUT_SIZE, input_features, target_fts,STRIDE=1,MOD_NUM=10000):

    ft_to_idx_dict = {}

    SEQUENCE_SIZE = INPUT_SIZE + OUTPUT_SIZE
    TOTAL_EVENTS = df.shape[0]
    NUM_LOOPS = TOTAL_EVENTS - SEQUENCE_SIZE + 1
    #print("NUM_LOOPS: %d" %NUM_LOOPS)

    dynamic_input_fts = []
    for ft in input_features:
        if ft not in static_input_fts:
            dynamic_input_fts.append(ft)
    print("\ndynamic_input_fts")
    print(dynamic_input_fts)

    #get arrays
    # dynamic_x_data_arrays = df[dynamic_input_fts].values
    # static_x_data_arrays = df[static_input_fts].values
    x_data_arrays = df[input_features].values
    for ft_idx,ft in enumerate(input_features):
        ft_to_idx_dict[ft]=ft_idx
    y_data_arrays = df[target_fts].values
    #print("\ndata_arrays shape:" )
    #print(x_data_arrays.shape)
    #print(y_data_arrays.shape)
    x_data = []
    y_data = []

    for i in range(NUM_LOOPS):

        x_cur_seq = x_data_arrays[i:i+SEQUENCE_SIZE]
        y_cur_seq = y_data_arrays[i:i+SEQUENCE_SIZE]

        x_seq = x_cur_seq[:INPUT_SIZE, :]
        y_seq = y_cur_seq[INPUT_SIZE:INPUT_SIZE+OUTPUT_SIZE, :]
        #if i%MOD_NUM == 0:
            #print("cur_seq shapes: " )
            #print(x_cur_seq.shape)
            #print(y_cur_seq.shape)
            #print("\nCurrent loop: %d of %d" %((i+1), NUM_LOOPS))
            #print("x shape: %s" %str(x_seq.shape))
            #print("y shape: %s" %str(y_seq.shape))
        x_data.append(x_seq)
        y_data.append(y_seq)

    #final data
    x_data = np.asarray(x_data)
    # print("\nx_data shape")
    # print(x_data.shape)
    # sys.exit(0)

    #reshape
    x_data = x_data.reshape((x_data.shape[0], x_data.shape[1]*x_data.shape[2]))

    extracted_static_features = []
    # for


    y_data = np.asarray(y_data)

    return x_data,y_data,ft_to_idx_dict

def get_ft_to_idx_dict(input_features):
    ft_to_idx_dict = {}
    for ft_idx,ft in enumerate(input_features):
        ft_to_idx_dict[ft]=ft_idx
    return ft_to_idx_dict

def convert_df_to_sliding_window_form_diff_x_and_y_fts_no_print_with_idx_dict(df, INPUT_SIZE, OUTPUT_SIZE, input_features, target_fts,STRIDE=1,MOD_NUM=10000):

    ft_to_idx_dict = {}

    SEQUENCE_SIZE = INPUT_SIZE + OUTPUT_SIZE
    TOTAL_EVENTS = df.shape[0]
    NUM_LOOPS = TOTAL_EVENTS - SEQUENCE_SIZE + 1
    #print("NUM_LOOPS: %d" %NUM_LOOPS)

    #get arrays
    x_data_arrays = df[input_features].values
    for ft_idx,ft in enumerate(input_features):
        ft_to_idx_dict[ft]=ft_idx
    y_data_arrays = df[target_fts].values
    #print("\ndata_arrays shape:" )
    #print(x_data_arrays.shape)
    #print(y_data_arrays.shape)
    x_data = []
    y_data = []

    for i in range(NUM_LOOPS):

        x_cur_seq = x_data_arrays[i:i+SEQUENCE_SIZE]
        y_cur_seq = y_data_arrays[i:i+SEQUENCE_SIZE]

        x_seq = x_cur_seq[:INPUT_SIZE, :]
        y_seq = y_cur_seq[INPUT_SIZE:INPUT_SIZE+OUTPUT_SIZE, :]
        #if i%MOD_NUM == 0:
            #print("cur_seq shapes: " )
            #print(x_cur_seq.shape)
            #print(y_cur_seq.shape)
            #print("\nCurrent loop: %d of %d" %((i+1), NUM_LOOPS))
            #print("x shape: %s" %str(x_seq.shape))
            #print("y shape: %s" %str(y_seq.shape))
        x_data.append(x_seq)
        y_data.append(y_seq)

    #final data
    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)

    return x_data,y_data,ft_to_idx_dict

def make_arch_tuple_list(LSTM_UNIT_LIST, DROPOUT):

    ARCH_TUPLE_LIST = []
    for i,lstm_units in enumerate(LSTM_UNIT_LIST):
        if i == len(LSTM_UNIT_LIST)-1:
            RETURN_SEQ = False
        else:
            RETURN_SEQ = True
        lstm_tuple = ("lstm", lstm_units,RETURN_SEQ)
        ARCH_TUPLE_LIST.append(lstm_tuple)
        if (DROPOUT > 0) and (i < len(LSTM_UNIT_LIST)-1):
            drop_tuple = ("dropout", DROPOUT)
            ARCH_TUPLE_LIST.append(drop_tuple)
    print("\nARCH_TUPLE_LIST")
    print(ARCH_TUPLE_LIST)


    return ARCH_TUPLE_LIST

def make_fully_connected_arch_tuple_list(DENSE_UNIT_LIST, DROPOUT):

    ARCH_TUPLE_LIST = []
    for i,dense_units in enumerate(DENSE_UNIT_LIST):
        # if i == len(FC_UNIT_LIST)-1:
        #     RETURN_SEQ = False
        # else:
        #     RETURN_SEQ = True
        dense_tuple = ("dense", dense_units)
        ARCH_TUPLE_LIST.append(dense_tuple)
        if (DROPOUT > 0) and (i < len(DENSE_UNIT_LIST)-1):
            drop_tuple = ("dropout", DROPOUT)
            ARCH_TUPLE_LIST.append(drop_tuple)
    print("\nARCH_TUPLE_LIST")
    print(ARCH_TUPLE_LIST)


    return ARCH_TUPLE_LIST

def add_more_dropout(ARCH_TUPLE_LIST, MORE_DROPOUT):

    if MORE_DROPOUT == False:
        return ARCH_TUPLE_LIST

    for ARCH_TUPLE in ARCH_TUPLE_LIST:
        print(ARCH_TUPLE)

    return ARCH_TUPLE_LIST

# def get_optimizer_v2_dr_fix(OPT_STR, LR, DR,EPOCHS):

#     #setup optimizer/learning rate
#     if OPT_STR=="Adam":
#         OPTIMIZER=Adam(learning_rate=LR,decay=DR)
#     elif OPT_STR=="SGD":
#         OPTIMIZER=SGD(learning_rate=LR,decay=DR)
#     elif (OPT_STR=="RMSprop") or (OPT_STR=="RMSProp"):
#         OPTIMIZER=RMSprop(learning_rate=LR)
#     elif OPT_STR=="Adadelta":
#         OPTIMIZER=Adadelta(learning_rate=LR,decay=DR)
#     else:
#         print("Please set up your new optimizer %s" %OPT_STR)

#     return OPTIMIZER

def get_optimizer(OPT_STR, LR, DR):

    #setup optimizer/learning rate
    if OPT_STR=="Adam":
        OPTIMIZER=Adam(learning_rate=LR,decay=DR)
    elif OPT_STR=="SGD":
        OPTIMIZER=SGD(learning_rate=LR,decay=DR,momentum=0.9)
    elif (OPT_STR=="RMSprop") or (OPT_STR=="RMSProp"):
        OPTIMIZER=RMSprop(learning_rate=LR)
    elif OPT_STR=="Adadelta":
        OPTIMIZER=Adadelta(learning_rate=LR,decay=DR)
    else:
        print("Please set up your new optimizer %s" %OPT_STR)

    return OPTIMIZER

def get_aux_fts(aux_ft_dir, GET_AUX_ACTION_FTS,GET_AUX_INFO_ID_FTS,GET_AUX_USER_AGE_FTS):

    aux_fp = aux_ft_dir + "Aux-Features.csv"
    aux_df = pd.read_csv(aux_fp)
    print(aux_df)

    keep_aux_cols = ["nodeTime"]

    if GET_AUX_ACTION_FTS == True:
        aux_action_fts = get_fts_from_fp(aux_ft_dir + "action_fts.txt")
        keep_aux_cols+=aux_action_fts
    if GET_AUX_INFO_ID_FTS == True:
        aux_infoID_fts = get_fts_from_fp(aux_ft_dir + "infoID_fts.txt")
        keep_aux_cols+=aux_infoID_fts
    if GET_AUX_USER_AGE_FTS == True:
        aux_user_age_fts = get_fts_from_fp(aux_ft_dir + "user_age_fts.txt")
        keep_aux_cols+=aux_user_age_fts

    #get aux fts
    aux_df = aux_df[keep_aux_cols]
    aux_cols = list(aux_df)
    for col in aux_cols:
        if col != "nodeTime":
            aux_df = aux_df.rename(columns={col:"aux_"+col})
    print(aux_df)

    aux_fts = list(aux_df)
    aux_fts.remove("nodeTime")


    return aux_df,aux_fts

def get_aux_fts_v2_platform_options(aux_ft_dir,
                                        GET_AUX_YOUTUBE_ACTION_FTS ,
                                        GET_AUX_TWITTER_ACTION_FTS ,
                                        GET_AUX_YOUTUBE_INFO_ID_FTS,
                                        GET_AUX_TWITTER_INFO_ID_FTS,
                                        GET_AUX_YOUTUBE_USER_AGE_FTS,
                                        GET_AUX_TWITTER_USER_AGE_FTS ):

    aux_fp = aux_ft_dir + "Aux-Features.csv"
    aux_df = pd.read_csv(aux_fp)
    print(aux_df)

    keep_aux_cols = ["nodeTime"]

    #aux action fts
    new_aux_action_fts = []
    aux_action_fts = get_fts_from_fp(aux_ft_dir + "action_fts.txt")
    for ft in aux_action_fts:
        if (GET_AUX_YOUTUBE_ACTION_FTS == True) and ("youtube" in ft):
            new_aux_action_fts.append(ft)
        if (GET_AUX_TWITTER_ACTION_FTS == True) and ("twitter" in ft):
            new_aux_action_fts.append(ft)
    aux_action_fts = new_aux_action_fts

    #aux_infoID_fts
    new_aux_infoID_fts = []
    aux_infoID_fts = get_fts_from_fp(aux_ft_dir + "infoID_fts.txt")
    for ft in aux_infoID_fts:
        if (GET_AUX_YOUTUBE_INFO_ID_FTS == True) and ("youtube" in ft):
            new_aux_infoID_fts.append(ft)
        if (GET_AUX_TWITTER_INFO_ID_FTS == True) and ("twitter" in ft):
            new_aux_infoID_fts.append(ft)
    aux_infoID_fts = new_aux_infoID_fts

    #aux_user_age_fts
    new_aux_user_age_fts = []
    aux_user_age_fts = get_fts_from_fp(aux_ft_dir + "user_age_fts.txt")
    for ft in aux_user_age_fts:
        if (GET_AUX_YOUTUBE_USER_AGE_FTS == True) and ("youtube" in ft):
            new_aux_user_age_fts.append(ft)
        if (GET_AUX_TWITTER_USER_AGE_FTS == True) and ("twitter" in ft):
            new_aux_user_age_fts.append(ft)
    aux_user_age_fts = new_aux_user_age_fts

    keep_aux_cols+=aux_action_fts
    keep_aux_cols+=aux_infoID_fts
    keep_aux_cols+=aux_user_age_fts

    #get aux fts
    aux_df = aux_df[keep_aux_cols]
    aux_cols = list(aux_df)
    for col in aux_cols:
        if col != "nodeTime":
            aux_df = aux_df.rename(columns={col:"aux_"+col})
    print(aux_df)

    aux_fts = list(aux_df)
    aux_fts.remove("nodeTime")


    return aux_df,aux_fts

def alter_df_GRAN(df, GRAN, input_fts, sum_fts, avg_fts):
    print("\nAltering df granularity to %s...."%GRAN)
    fts = list(input_fts)
    keep_cols = list(["nodeTime"] + fts)
    df = df[keep_cols]
    df["nodeTime"] = pd.to_datetime(df["nodeTime"], utc=True)
    df["nodeTime"] = df["nodeTime"].dt.floor(GRAN)

    for ft in sum_fts:
        if ft in list(df):
            df[ft] = df.groupby("nodeTime")[ft].transform("sum")
    for ft in avg_fts:
        if ft in list(df):
            df[ft] = df.groupby("nodeTime")[ft].transform("mean")

    df = df.drop_duplicates("nodeTime").reset_index(drop=True)
    print("\nAltered gran df")
    print(df)
    return df

def alter_df_GRAN_v3_fix_duplicate_issue(df, GRAN, input_fts, target_fts,sum_fts, avg_fts):
    # print("\nAltering df granularity to %s...."%GRAN)
    keep_cols = list(["nodeTime"] + input_fts)
    for ft in target_fts:
        if ft not in keep_cols:
            keep_cols.append(ft)


    # keep_cols = list(set(["nodeTime"] + input_fts + target_fts))
    df = df[keep_cols]
    df["nodeTime"] = pd.to_datetime(df["nodeTime"], utc=True)
    df["nodeTime"] = df["nodeTime"].dt.floor(GRAN)

    for ft in sum_fts:
        if ft in list(df):
            df[ft] = df.groupby("nodeTime")[ft].transform("sum")
    for ft in avg_fts:
        if ft in list(df):
            df[ft] = df.groupby("nodeTime")[ft].transform("mean")

    df = df.drop_duplicates("nodeTime").reset_index(drop=True)
    # print("\nAltered gran df")
    # print(df)
    return df

def alter_df_GRAN_v2(df, GRAN, input_fts, target_fts,sum_fts, avg_fts):
    # print("\nAltering df granularity to %s...."%GRAN)
    keep_cols = list(["nodeTime"] + input_fts + target_fts)
    df = df[keep_cols]
    df["nodeTime"] = pd.to_datetime(df["nodeTime"], utc=True)
    df["nodeTime"] = df["nodeTime"].dt.floor(GRAN)

    for ft in sum_fts:
        if ft in list(df):
            df[ft] = df.groupby("nodeTime")[ft].transform("sum")
    for ft in avg_fts:
        if ft in list(df):
            df[ft] = df.groupby("nodeTime")[ft].transform("mean")

    df = df.drop_duplicates("nodeTime").reset_index(drop=True)
    # print("\nAltered gran df")
    # print(df)
    return df

def convert_df_to_test_sliding_window_form_diff_x_and_y_fts_no_print_v2_diff_io_grans(ARRAY_IO_TUPLE,EXTRACTION_IO_TUPLE,df,sum_fts ,avg_fts , INPUT_GRAN,OUTPUT_GRAN ,input_features, target_fts,MOD_NUM=10):

    INPUT_SIZE = ARRAY_IO_TUPLE[0]
    OUTPUT_SIZE = ARRAY_IO_TUPLE[1]

    #config fts
    xdf = df[["nodeTime"]+input_features]
    ydf = df[["nodeTime"]+target_fts]

    xdf = alter_df_GRAN(xdf, INPUT_GRAN, input_features, sum_fts, avg_fts)
    ydf = alter_df_GRAN(ydf, OUTPUT_GRAN, target_fts, sum_fts, avg_fts)
    print("\nDfs after gran change")
    print("xdf")
    print(xdf)
    print("\nydf")
    print(ydf)

    SEQUENCE_SIZE = INPUT_SIZE + OUTPUT_SIZE
    TOTAL_EVENTS = df.shape[0] - INPUT_SIZE
    NUM_LOOPS = int(TOTAL_EVENTS/OUTPUT_SIZE)
    #print("NUM_LOOPS: %d" %NUM_LOOPS)

    #get arrays
    #get arrays
    x_data_arrays = xdf[input_features].values
    y_data_arrays = ydf[target_fts].values
    print("\ndata_arrays shape:" )
    print(x_data_arrays.shape)
    print(y_data_arrays.shape)
    sys.exit(0)
    x_data = []
    y_data = []

    for i in range(NUM_LOOPS):
        start = i *OUTPUT_SIZE
        end = start + SEQUENCE_SIZE
        # print("\nIdx %d: start: %d, end: %d" %(i, start,end))

        x_cur_seq = x_data_arrays[start:end, :]
        y_cur_seq = y_data_arrays[start:end, :]

        x_seq = np.asarray(x_cur_seq[:INPUT_SIZE, :])
        # y_seq = np.asarray(y_cur_seq[INPUT_SIZE:INPUT_SIZE+OUTPUT_SIZE, :])
        y_seq = np.asarray(y_cur_seq[:OUTPUT_SIZE, :])
        # if i%MOD_NUM == 0:
        #     # print("cur_seq shape: %s" %str(cur_seq.shape))
        #     print("\nCurrent loop: %d of %d" %((i+1), NUM_LOOPS))
        #     print("x shape: %s" %str(x_seq.shape))
        #     print("y shape: %s" %str(y_seq.shape))
        x_data.append(x_seq)
        y_data.append(y_seq)

    #final data
    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)

    return x_data,y_data

def convert_df_to_sliding_window_form_diff_x_and_y_fts_no_print_with_idx_dict_v2_diff_io_grans(sum_fts ,avg_fts ,INPUT_GRAN,OUTPUT_GRAN ,df, EXTRACTION_IO_TUPLE,ARRAY_IO_TUPLE, input_features, target_fts,STRIDE=1,MOD_NUM=10000):

    ft_to_idx_dict = {}

    INPUT_SIZE = ARRAY_IO_TUPLE[0]
    OUTPUT_SIZE = ARRAY_IO_TUPLE[1]

    EXTRACT_OUTPUT_SIZE= EXTRACTION_IO_TUPLE[1]

    SEQUENCE_SIZE = INPUT_SIZE + OUTPUT_SIZE
    TOTAL_EVENTS = df.shape[0]
    NUM_LOOPS = TOTAL_EVENTS - SEQUENCE_SIZE + 1

    #print("NUM_LOOPS: %d" %NUM_LOOPS)

    if INPUT_GRAN == "H":
        INPUT_GRAN_MAG =1
    else:
        INPUT_GRAN_MAG = int(INPUT_GRAN[:-1])

    if OUTPUT_GRAN == "H":
        OUTPUT_GRAN_MAG =1
    else:
        OUTPUT_GRAN_MAG = int(OUTPUT_GRAN[:-1])

    INPUT_SEQ_SIZE = INPUT_SIZE + INPUT_GRAN_MAG * OUTPUT_GRAN_MAG
    print("\nINPUT_SEQ_SIZE: %d"%INPUT_SEQ_SIZE)
    OUTPUT_SEQ_SIZE = int((INPUT_SIZE + EXTRACT_OUTPUT_SIZE)/OUTPUT_GRAN_MAG )
    print("\nOUTPUT_SEQ_SIZE: %d"%OUTPUT_SEQ_SIZE)

    # sys.exit(0)

    #config fts
    xdf = df[["nodeTime"]+input_features]
    ydf = df[["nodeTime"]+target_fts]

    NUM_LOOPS = ydf.shape[0] - OUTPUT_SEQ_SIZE + 1
    print("\nnum loops: %d"%NUM_LOOPS)

    # xfts = list(xdf)
    # xfts.remove("nodeTime")
    # yfts = list(ydf)
    # yfts.remove("nodeTime")

    xdf = alter_df_GRAN(xdf, INPUT_GRAN, input_features, sum_fts, avg_fts)
    ydf = alter_df_GRAN(ydf, OUTPUT_GRAN, target_fts, sum_fts, avg_fts)
    # print("\nDfs after gran change")
    # print("xdf")
    # print(xdf)
    # print("\nydf")
    # print(ydf)

    #get arrays
    x_data_arrays = xdf[input_features].values
    for ft_idx,ft in enumerate(input_features):
        ft_to_idx_dict[ft]=ft_idx
    y_data_arrays = ydf[target_fts].values
    #print("\ndata_arrays shape:" )
    #print(x_data_arrays.shape)
    #print(y_data_arrays.shape)
    x_data = []
    y_data = []

    for i in range(NUM_LOOPS):

        # x_cur_seq = x_data_arrays[i:i+SEQUENCE_SIZE]
        # y_cur_seq = y_data_arrays[i:i+SEQUENCE_SIZE]
        x_cur_seq = x_data_arrays[i:i+INPUT_SEQ_SIZE]
        y_cur_seq = y_data_arrays[i:i+OUTPUT_SEQ_SIZE]

        x_seq = x_cur_seq[:INPUT_SIZE, :]
        # y_seq = y_cur_seq[INPUT_SIZE:INPUT_SIZE+OUTPUT_SIZE, :]
        y_seq = y_cur_seq[-OUTPUT_SIZE:, :]
        #if i%MOD_NUM == 0:
            #print("cur_seq shapes: " )
            #print(x_cur_seq.shape)
            #print(y_cur_seq.shape)
            #print("\nCurrent loop: %d of %d" %((i+1), NUM_LOOPS))
            #print("x shape: %s" %str(x_seq.shape))
            #print("y shape: %s" %str(y_seq.shape))
        x_data.append(x_seq)
        y_data.append(y_seq)

    #final data
    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)
    print("\nfinal shapes")
    print(x_data.shape)
    print(y_data.shape)
    print(x_data)
    print(y_data)
    sys.exit(0)

    return x_data,y_data,ft_to_idx_dict

def convert_df_to_sliding_window_form_diff_x_and_y_fts(df, INPUT_SIZE, OUTPUT_SIZE, input_features, target_fts,STRIDE=1,MOD_NUM=10000):

    SEQUENCE_SIZE = INPUT_SIZE + OUTPUT_SIZE
    TOTAL_EVENTS = df.shape[0]
    NUM_LOOPS = TOTAL_EVENTS - SEQUENCE_SIZE + 1
    print("NUM_LOOPS: %d" %NUM_LOOPS)

    #get arrays
    x_data_arrays = df[input_features].values
    y_data_arrays = df[target_fts].values
    print("\ndata_arrays shape:" )
    print(x_data_arrays.shape)
    print(y_data_arrays.shape)
    x_data = []
    y_data = []

    for i in range(NUM_LOOPS):

        x_cur_seq = x_data_arrays[i:i+SEQUENCE_SIZE]
        y_cur_seq = y_data_arrays[i:i+SEQUENCE_SIZE]

        x_seq = x_cur_seq[:INPUT_SIZE, :]
        y_seq = y_cur_seq[INPUT_SIZE:INPUT_SIZE+OUTPUT_SIZE, :]
        if i%MOD_NUM == 0:
            print("cur_seq shapes: " )
            print(x_cur_seq.shape)
            print(y_cur_seq.shape)
            print("\nCurrent loop: %d of %d" %((i+1), NUM_LOOPS))
            print("x shape: %s" %str(x_seq.shape))
            print("y shape: %s" %str(y_seq.shape))
        x_data.append(x_seq)
        y_data.append(y_seq)

    #final data
    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)

    return x_data,y_data


def convert_df_to_sliding_window_form(df, INPUT_SIZE, OUTPUT_SIZE,STRIDE=1,MOD_NUM=10000):

    SEQUENCE_SIZE = INPUT_SIZE + OUTPUT_SIZE
    TOTAL_EVENTS = df.shape[0]
    NUM_LOOPS = TOTAL_EVENTS - SEQUENCE_SIZE + 1
    print("NUM_LOOPS: %d" %NUM_LOOPS)

    #get arrays
    data_arrays = df.values
    print("\ndata_arrays shape: %s" %str(data_arrays.shape))
    x_data = []
    y_data = []

    for i in range(NUM_LOOPS):

        cur_seq = data_arrays[i:i+SEQUENCE_SIZE]

        x_seq = cur_seq[:INPUT_SIZE, :]
        y_seq = cur_seq[INPUT_SIZE:INPUT_SIZE+OUTPUT_SIZE, :]
        if i%MOD_NUM == 0:
            print("cur_seq shape: %s" %str(cur_seq.shape))
            print("\nCurrent loop: %d of %d" %((i+1), NUM_LOOPS))
            print("x shape: %s" %str(x_seq.shape))
            print("y shape: %s" %str(y_seq.shape))
        x_data.append(x_seq)
        y_data.append(y_seq)

    #final data
    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)

    return x_data,y_data

def convert_df_to_test_sliding_window_form(df, INPUT_SIZE, OUTPUT_SIZE,MOD_NUM=10000):

    SEQUENCE_SIZE = INPUT_SIZE + OUTPUT_SIZE
    TOTAL_EVENTS = df.shape[0] - INPUT_SIZE
    NUM_LOOPS = int(TOTAL_EVENTS/OUTPUT_SIZE)
    print("NUM_LOOPS: %d" %NUM_LOOPS)

    #get arrays
    data_arrays = df.values
    print("\ndata_arrays shape: %s" %str(data_arrays.shape))
    x_data = []
    y_data = []

    for i in range(NUM_LOOPS):
        start = i *OUTPUT_SIZE
        end = start + SEQUENCE_SIZE
        # print("\nIdx %d: start: %d, end: %d" %(i, start,end))

        cur_seq = data_arrays[start:end, :]

        x_seq = np.asarray(cur_seq[:INPUT_SIZE, :])
        y_seq = np.asarray(cur_seq[INPUT_SIZE:INPUT_SIZE+OUTPUT_SIZE, :])
        if i%MOD_NUM == 0:
            print("cur_seq shape: %s" %str(cur_seq.shape))
            print("\nCurrent loop: %d of %d" %((i+1), NUM_LOOPS))
            print("x shape: %s" %str(x_seq.shape))
            print("y shape: %s" %str(y_seq.shape))
        x_data.append(x_seq)
        y_data.append(y_seq)

    #final data
    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)

    return x_data,y_data

def convert_df_to_test_sliding_window_form_diff_x_and_y_fts(df, INPUT_SIZE, OUTPUT_SIZE,input_features, target_fts,MOD_NUM=10):

    SEQUENCE_SIZE = INPUT_SIZE + OUTPUT_SIZE
    TOTAL_EVENTS = df.shape[0] - INPUT_SIZE
    NUM_LOOPS = int(TOTAL_EVENTS/OUTPUT_SIZE)
    print("NUM_LOOPS: %d" %NUM_LOOPS)

    #get arrays
    #get arrays
    x_data_arrays = df[input_features].values
    y_data_arrays = df[target_fts].values
    print("\ndata_arrays shape:" )
    print(x_data_arrays.shape)
    print(y_data_arrays.shape)
    x_data = []
    y_data = []

    for i in range(NUM_LOOPS):
        start = i *OUTPUT_SIZE
        end = start + SEQUENCE_SIZE
        # print("\nIdx %d: start: %d, end: %d" %(i, start,end))

        x_cur_seq = x_data_arrays[start:end, :]
        y_cur_seq = y_data_arrays[start:end, :]

        x_seq = np.asarray(x_cur_seq[:INPUT_SIZE, :])
        y_seq = np.asarray(y_cur_seq[INPUT_SIZE:INPUT_SIZE+OUTPUT_SIZE, :])
        if i%MOD_NUM == 0:
            # print("cur_seq shape: %s" %str(cur_seq.shape))
            print("\nCurrent loop: %d of %d" %((i+1), NUM_LOOPS))
            print("x shape: %s" %str(x_seq.shape))
            print("y shape: %s" %str(y_seq.shape))
        x_data.append(x_seq)
        y_data.append(y_seq)

    #final data
    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)

    return x_data,y_data

def convert_df_to_test_sliding_window_form_diff_x_and_y_fts_no_print(df, INPUT_SIZE, OUTPUT_SIZE,input_features, target_fts,MOD_NUM=10):

    SEQUENCE_SIZE = INPUT_SIZE + OUTPUT_SIZE
    TOTAL_EVENTS = df.shape[0] - INPUT_SIZE
    NUM_LOOPS = int(TOTAL_EVENTS/OUTPUT_SIZE)
    #print("NUM_LOOPS: %d" %NUM_LOOPS)

    #get arrays
    #get arrays
    x_data_arrays = df[input_features].values
    y_data_arrays = df[target_fts].values
    #print("\ndata_arrays shape:" )
    #print(x_data_arrays.shape)
    #print(y_data_arrays.shape)
    x_data = []
    y_data = []

    for i in range(NUM_LOOPS):
        start = i *OUTPUT_SIZE
        end = start + SEQUENCE_SIZE
        # print("\nIdx %d: start: %d, end: %d" %(i, start,end))

        x_cur_seq = x_data_arrays[start:end, :]
        y_cur_seq = y_data_arrays[start:end, :]

        x_seq = np.asarray(x_cur_seq[:INPUT_SIZE, :])
        y_seq = np.asarray(y_cur_seq[INPUT_SIZE:INPUT_SIZE+OUTPUT_SIZE, :])
        # if i%MOD_NUM == 0:
        #     # print("cur_seq shape: %s" %str(cur_seq.shape))
        #     print("\nCurrent loop: %d of %d" %((i+1), NUM_LOOPS))
        #     print("x shape: %s" %str(x_seq.shape))
        #     print("y shape: %s" %str(y_seq.shape))
        x_data.append(x_seq)
        y_data.append(y_seq)

    #final data
    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)

    return x_data,y_data

def convert_df_to_test_sliding_window_form_diff_x_and_y_fts_SAME_PERIOD(df, INPUT_SIZE, OUTPUT_SIZE,input_features, target_fts,MOD_NUM=10):

    SEQUENCE_SIZE = INPUT_SIZE + OUTPUT_SIZE
    TOTAL_EVENTS = df.shape[0] - INPUT_SIZE
    NUM_LOOPS = int(TOTAL_EVENTS/OUTPUT_SIZE)
    #print("NUM_LOOPS: %d" %NUM_LOOPS)

    #get arrays
    #get arrays
    x_data_arrays = df[input_features].values
    y_data_arrays = df[target_fts].values
    #print("\ndata_arrays shape:" )
    #print(x_data_arrays.shape)
    #print(y_data_arrays.shape)
    x_data = []
    y_data = []

    for i in range(NUM_LOOPS):
        start = i *OUTPUT_SIZE
        end = start + SEQUENCE_SIZE
        # print("\nIdx %d: start: %d, end: %d" %(i, start,end))

        x_cur_seq = x_data_arrays[start:end, :]
        y_cur_seq = y_data_arrays[start:end, :]

        x_seq = np.asarray(x_cur_seq[OUTPUT_SIZE:INPUT_SIZE + OUTPUT_SIZE, :])
        y_seq = np.asarray(y_cur_seq[INPUT_SIZE:INPUT_SIZE+OUTPUT_SIZE, :])
        # if i%MOD_NUM == 0:
        #     # print("cur_seq shape: %s" %str(cur_seq.shape))
        #     print("\nCurrent loop: %d of %d" %((i+1), NUM_LOOPS))
        #     print("x shape: %s" %str(x_seq.shape))
        #     print("y shape: %s" %str(y_seq.shape))
        x_data.append(x_seq)
        y_data.append(y_seq)

    #final data
    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)

    return x_data,y_data

def convert_df_to_test_sliding_window_form_diff_x_ONLY_fts_SAME_PERIOD(df, INPUT_SIZE, OUTPUT_SIZE,input_features, MOD_NUM=10):

    SEQUENCE_SIZE = INPUT_SIZE + OUTPUT_SIZE
    TOTAL_EVENTS = df.shape[0] - INPUT_SIZE
    NUM_LOOPS = int(TOTAL_EVENTS/OUTPUT_SIZE)
    #print("NUM_LOOPS: %d" %NUM_LOOPS)

    #get arrays
    #get arrays
    x_data_arrays = df[input_features].values
    #print("\ndata_arrays shape:" )
    #print(x_data_arrays.shape)
    #print(y_data_arrays.shape)
    x_data = []

    for i in range(NUM_LOOPS):
        start = i *OUTPUT_SIZE
        end = start + SEQUENCE_SIZE
        # print("\nIdx %d: start: %d, end: %d" %(i, start,end))

        x_cur_seq = x_data_arrays[start:end, :]

        x_seq = np.asarray(x_cur_seq[OUTPUT_SIZE:INPUT_SIZE + OUTPUT_SIZE, :])
        # if i%MOD_NUM == 0:
        #     # print("cur_seq shape: %s" %str(cur_seq.shape))
        #     print("\nCurrent loop: %d of %d" %((i+1), NUM_LOOPS))
        #     print("x shape: %s" %str(x_seq.shape))
        #     print("y shape: %s" %str(y_seq.shape))
        x_data.append(x_seq)

    #final data
    x_data = np.asarray(x_data)

    return x_data


def convert_df_to_test_sliding_window_form_FIXED(df, INPUT_SIZE, OUTPUT_SIZE,MOD_NUM=10000):

    SEQUENCE_SIZE = INPUT_SIZE + OUTPUT_SIZE
    TOTAL_EVENTS = df.shape[0] - INPUT_SIZE
    NUM_LOOPS = int(TOTAL_EVENTS/OUTPUT_SIZE)
    print("NUM_LOOPS: %d" %NUM_LOOPS)

    #get num samples
    num_rows = df.shape[0]
    num_fts = len(list(df))
    num_chopped_rows = num_rows - INPUT_SIZE
    print("\nnum_rows: %d" %num_rows)
    print("num_chopped_rows: %d" %num_chopped_rows)
    print("\nNum fts: %d" %num_fts)

    #chop off for x and y test
    x_test = df.values[:-INPUT_SIZE,:]
    y_test = df.values[INPUT_SIZE:,:]

    x_test = x_test.reshape((int(num_chopped_rows/INPUT_SIZE), INPUT_SIZE, num_fts))
    y_test = y_test.reshape((int(num_chopped_rows/INPUT_SIZE), INPUT_SIZE, num_fts))
    y_test = y_test[:, :OUTPUT_SIZE, :]

    print("\nx and y test shapes")
    print(x_test.shape)
    print(y_test.shape)

    return x_test,y_test

def config_df_by_daily_dates(df,start_date,end_date,time_col="nodeTime"):
    df[time_col]=pd.to_datetime(df[time_col], utc=True)
    df = df.sort_values(time_col)
    df["temp_dates"] = df[time_col].dt.floor("D")
    df=df.set_index("temp_dates")
    df = df[(df.index >= start_date) & (df.index <= end_date)]
    df=df.reset_index(drop=True)
    return df

def split_dfs_into_train_and_test_v2_5_26(df, test_start, test_end,IO_TUPLE):

    #get io
    INPUT_SIZE = IO_TUPLE[0]
    OUTPUT_SIZE = IO_TUPLE[1]

    #insert ids
    if "nodeID" not in list(df):
        df["nodeID"] = [i for i in range(df.shape[0])]

    #get test df
    test_df = config_df_by_daily_dates(df, test_start, test_end,"nodeTime")
    print(test_df["nodeTime"])

    #get test nodes
    test_nodeIDs = list(test_df["nodeID"])
    print("\nNum test nodeiDs %d"%len(test_nodeIDs))

    #get train df
    test_size = test_df.shape[0]
    train_size = df.shape[0] - test_size
    print("\ntrain and test size: %d , %d"%(train_size, test_size))
    print("IO: %s" %str(IO_TUPLE))

    #fix train
    train_df = df.head(train_size)
    print("\nfixed train df after head")
    print(train_df)

    #make sure you have enough time steps
    if OUTPUT_SIZE > test_size:
        print("\nError: OUTPUT_SIZE > test_size")
        sys.exit(0)

    #do we have enough data to make samples?
    if test_size%OUTPUT_SIZE != 0:
        num_leftovers =OUTPUT_SIZE -( test_size%OUTPUT_SIZE)
    else:
        num_leftovers = 0
    print("\nnum_leftovers: %d" %num_leftovers)
    print("Need to take %d rows from train data"%num_leftovers)

    #add to test
    train_tail = train_df.tail(num_leftovers)
    train_df = train_df.head(train_size - num_leftovers)
    test_df = pd.concat([train_tail, test_df])
    test_df = test_df.sort_values("nodeTime").reset_index(drop=True)
    print("\ntest df after getting train tail")
    print(test_df)
    print("\ntest_df nodeTime")
    print(test_df["nodeTime"])
    print("\ntrain_df nodeTime")
    print(train_df["nodeTime"])

    #now we need to get the init condit
    init_condit = train_df.tail(INPUT_SIZE)
    print("\ninit_condit")
    print(init_condit)
    # sys.exit(0)
    test_df = pd.concat([init_condit, test_df])
    test_df = test_df.sort_values("nodeTime").reset_index(drop=True)
    train_df = train_df.head(train_df.shape[0] - init_condit.shape[0])
    print("\ntrain df nodeTime after ic change")
    print(train_df["nodeTime"])
    print("\ntest df nodeTime after ic change")
    print(test_df["nodeTime"])

    # sys.exit(0)

    #check
    gt_size = df.shape[0]
    my_size = test_df.shape[0] + train_df.shape[0]
    if my_size != gt_size:
        print("my size and gt size: %d, %d"%(my_size, gt_size))
        print("my_size != gt_size")
        print(sys.exit(0))

    test_size_without_init = test_df.shape[0] - INPUT_SIZE
    print("\ntest_size_without_init: %d" %test_size_without_init)
    test_remainder = test_size_without_init%OUTPUT_SIZE
    if test_remainder != 0:
        print("test_remainder != 0")
        sys.exit(0)

    print("Dfs are ok!")
    # sys.exit(0)

    return train_df,test_df

def split_dfs_into_train_and_test_v2_5_26_no_print_JULY14_BACKUP(df, test_start, test_end,IO_TUPLE):

    #get io
    INPUT_SIZE = IO_TUPLE[0]
    OUTPUT_SIZE = IO_TUPLE[1]

    #insert ids
    if "nodeID" not in list(df):
        df["nodeID"] = [i for i in range(df.shape[0])]

    #get test df
    test_df = config_df_by_daily_dates(df, test_start, test_end,"nodeTime")
    print(test_df["nodeTime"])
    # sys.exit(0)

    #get test nodes
    test_nodeIDs = list(test_df["nodeID"])
    #print("\nNum test nodeiDs %d"%len(test_nodeIDs))

    #get train df
    test_size = test_df.shape[0]
    train_size = df.shape[0] - test_size
    #print("\ntrain and test size: %d , %d"%(train_size, test_size))
    #print("IO: %s" %str(IO_TUPLE))

    #fix train
    train_df = df.head(train_size)
    #print("\nfixed train df after head")
    #print(train_df)

    print("\ntrain")
    print(train_df)

    print("\ntest_df")
    print(test_df)

    sys.exit(0)

    #make sure you have enough time steps
    if OUTPUT_SIZE > test_size:
        print(OUTPUT_SIZE)
        print(test_size)
        print("\nError: OUTPUT_SIZE > test_size")
        sys.exit(0)

    #do we have enough data to make samples?
    if test_size%OUTPUT_SIZE != 0:
        num_leftovers =OUTPUT_SIZE -( test_size%OUTPUT_SIZE)
    else:
        num_leftovers = 0
    #print("\nnum_leftovers: %d" %num_leftovers)
    #print("Need to take %d rows from train data"%num_leftovers)

    #add to test
    train_tail = train_df.tail(num_leftovers)
    train_df = train_df.head(train_size - num_leftovers)
    test_df = pd.concat([train_tail, test_df])
    test_df = test_df.sort_values("nodeTime").reset_index(drop=True)
    #print("\ntest df after getting train tail")
    #print(test_df)
    #print("\ntest_df nodeTime")
    #print(test_df["nodeTime"])
    #print("\ntrain_df nodeTime")
    #print(train_df["nodeTime"])

    #now we need to get the init condit
    init_condit = train_df.tail(INPUT_SIZE)
    #print("\ninit_condit")
    #print(init_condit)
    # sys.exit(0)
    test_df = pd.concat([init_condit, test_df])
    test_df = test_df.sort_values("nodeTime").reset_index(drop=True)
    train_df = train_df.head(train_df.shape[0] - init_condit.shape[0])
    #print("\ntrain df nodeTime after ic change")
    #print(train_df["nodeTime"])
    #print("\ntest df nodeTime after ic change")
    #print(test_df["nodeTime"])

    # sys.exit(0)

    #check
    gt_size = df.shape[0]
    my_size = test_df.shape[0] + train_df.shape[0]
    if my_size != gt_size:
        print("my size and gt size: %d, %d"%(my_size, gt_size))
        print("my_size != gt_size")
        print(sys.exit(0))

    test_size_without_init = test_df.shape[0] - INPUT_SIZE
    #print("\ntest_size_without_init: %d" %test_size_without_init)
    test_remainder = test_size_without_init%OUTPUT_SIZE
    if test_remainder != 0:
        print("test_remainder != 0")
        sys.exit(0)

    #print("Dfs are ok!")
    # sys.exit(0)

    return train_df,test_df

def split_dfs_into_train_and_test_v2_5_26_no_print(df, test_start, test_end,IO_TUPLE):

    #get io
    INPUT_SIZE = IO_TUPLE[0]
    OUTPUT_SIZE = IO_TUPLE[1]

    #sort
    df["nodeTime"] = pd.to_datetime(df["nodeTime"], utc=True)
    df = df.sort_values("nodeTime").reset_index(drop=True)

    train_start = df["nodeTime"].iloc[0]
    df = config_df_by_daily_dates(df, train_start, test_end,"nodeTime")

    #insert ids
    if "nodeID" not in list(df):
        df["nodeID"] = [i for i in range(df.shape[0])]


    #get test df
    test_df = config_df_by_daily_dates(df, test_start, test_end,"nodeTime")
    # print(test_df["nodeTime"])
    # sys.exit(0)

    #get test nodes
    test_nodeIDs = list(test_df["nodeID"])
    #print("\nNum test nodeiDs %d"%len(test_nodeIDs))

    #get train df
    test_size = test_df.shape[0]
    train_size = df.shape[0] - test_size
    #print("\ntrain and test size: %d , %d"%(train_size, test_size))
    #print("IO: %s" %str(IO_TUPLE))

    #fix train
    train_df = df.head(train_size)
    #print("\nfixed train df after head")
    #print(train_df)

    # print("\ntrain")
    # print(train_df["nodeTime"])

    # print("\ntest_df")
    # print(test_df["nodeTime"])

    # sys.exit(0)

    #make sure you have enough time steps
    if OUTPUT_SIZE > test_size:
        print(OUTPUT_SIZE)
        print(test_size)
        print("\nError: OUTPUT_SIZE > test_size")
        sys.exit(0)

    #do we have enough data to make samples?
    if test_size%OUTPUT_SIZE != 0:
        num_leftovers =OUTPUT_SIZE -( test_size%OUTPUT_SIZE)
    else:
        num_leftovers = 0
    #print("\nnum_leftovers: %d" %num_leftovers)
    #print("Need to take %d rows from train data"%num_leftovers)

    #add to test
    train_tail = train_df.tail(num_leftovers)
    train_df = train_df.head(train_size - num_leftovers)
    test_df = pd.concat([train_tail, test_df])
    test_df = test_df.sort_values("nodeTime").reset_index(drop=True)
    #print("\ntest df after getting train tail")
    #print(test_df)
    #print("\ntest_df nodeTime")
    #print(test_df["nodeTime"])
    #print("\ntrain_df nodeTime")
    #print(train_df["nodeTime"])

    #now we need to get the init condit
    init_condit = train_df.tail(INPUT_SIZE)
    #print("\ninit_condit")
    #print(init_condit)
    # sys.exit(0)
    test_df = pd.concat([init_condit, test_df])
    test_df = test_df.sort_values("nodeTime").reset_index(drop=True)
    train_df = train_df.head(train_df.shape[0] - init_condit.shape[0])
    #print("\ntrain df nodeTime after ic change")
    #print(train_df["nodeTime"])
    #print("\ntest df nodeTime after ic change")
    #print(test_df["nodeTime"])

    # sys.exit(0)

    #check
    gt_size = df.shape[0]
    my_size = test_df.shape[0] + train_df.shape[0]
    if my_size != gt_size:
        print("my size and gt size: %d, %d"%(my_size, gt_size))
        print("my_size != gt_size")
        print(sys.exit(0))

    test_size_without_init = test_df.shape[0] - INPUT_SIZE
    #print("\ntest_size_without_init: %d" %test_size_without_init)
    test_remainder = test_size_without_init%OUTPUT_SIZE
    if test_remainder != 0:
        print("test_remainder != 0")
        sys.exit(0)

    #print("Dfs are ok!")
    # sys.exit(0)

    return train_df,test_df

def split_dfs_into_train_and_test(df, test_start, test_end,IO_TUPLE):

    #get test df
    test_df = config_df_by_daily_dates(df, test_start, test_end,"nodeTime")
    print(test_df["nodeTime"])


    #get test nodes
    test_nodeIDs = list(test_df["nodeID"])
    print("\nNum test nodeiDs %d"%len(test_nodeIDs))

    #we need to get the number of test records for a perfect fit
    sequence_size = IO_TUPLE[0] + IO_TUPLE[1]
    print("seq size")
    print(sequence_size)

    #get perfect fit so far
    num_test_records = test_df.shape[0]
    num_sequences_in_test_sequence_so_far = num_test_records/sequence_size
    print("\nnum_sequences_in_test_sequence_so_far: %d"%num_sequences_in_test_sequence_so_far)

    #get remainder
    remaining_events = num_test_records%sequence_size
    print("\nremaining_events: %d"%remaining_events)

    #get amount needed for perfect fit
    needed_leftovers = sequence_size - remaining_events
    print("needed_leftovers: %d"%needed_leftovers)

    #get full leftovers
    print("\nGetting leftovers...")
    leftovers_df = df.head(df.shape[0] - test_df.shape[0])
    num_leftovers = leftovers_df.shape[0]
    print("num_leftovers: %d"%num_leftovers)

    #put last bit of leftovers into test df
    removed_leftover_bit = leftovers_df.tail(needed_leftovers)
    test_df = pd.concat([removed_leftover_bit, test_df]).reset_index(drop=True)
    leftovers_df = leftovers_df.head(leftovers_df.shape[0] - needed_leftovers)

    print("\nNew test df")
    print(test_df)

    print("\nNew leftovers")
    print(leftovers_df)

    # sys.exit(0)

    #verify that it worked
    num_test_df_records = test_df.shape[0]
    res = num_test_df_records%sequence_size
    if res !=0:
        print("num_test_df_records mod sequence_size !=0")
        print(res)
        sys.exit(0)
    print("Correctly retrieved y test info... Now we need the initial condition...")

    initial_condition_size = IO_TUPLE[0]
    print("\ninitial_condition_size: %d" %initial_condition_size)

    init_condit = leftovers_df.tail(initial_condition_size)
    print("\ninit_condit")
    print(init_condit)

    #now remove and add to test
    leftovers_df = leftovers_df.head(leftovers_df.shape[0] - initial_condition_size)
    test_df = pd.concat([init_condit, test_df]).reset_index(drop=True)

    print("\nnew leftovers")
    print(leftovers_df)
    print("\ntest df with init condit")
    print(test_df)

    #verify
    num_leftovers = leftovers_df.shape[0]
    num_test_records = test_df.shape[0]
    total_records = num_leftovers + num_test_records
    if total_records != df.shape[0]:
        print("error total_records != df.shape[0]")
        print("total: %d"%total_records)
        print("original: %d"%df.shape[0])
    else:
        print("\nCorrectly created records")

    return leftovers_df,test_df

def split_dfs_into_train_and_test_no_print(df, test_start, test_end,IO_TUPLE):

	#get test df
	test_df = config_df_by_daily_dates(df, test_start, test_end,"nodeTime")

	#get test nodes
	test_nodeIDs = list(test_df["nodeID"])
	#print("\nNum test nodeiDs %d"%len(test_nodeIDs))

	#we need to get the number of test records for a perfect fit
	sequence_size = IO_TUPLE[0] + IO_TUPLE[1]
	#print("seq size")
	#print(sequence_size)

	#get perfect fit so far
	num_test_records = test_df.shape[0]
	num_sequences_in_test_sequence_so_far = num_test_records/sequence_size
	#print("\nnum_sequences_in_test_sequence_so_far: %d"%num_sequences_in_test_sequence_so_far)

	#get remainder
	remaining_events = num_test_records%sequence_size
	#print("\nremaining_events: %d"%remaining_events)

	#get amount needed for perfect fit
	needed_leftovers = sequence_size - remaining_events
	#print("needed_leftovers: %d"%needed_leftovers)

	#get full leftovers
	#print("\nGetting leftovers...")
	leftovers_df = df.head(df.shape[0] - test_df.shape[0])
	num_leftovers = leftovers_df.shape[0]
	#print("num_leftovers: %d"%num_leftovers)

	#put last bit of leftovers into test df
	removed_leftover_bit = leftovers_df.tail(needed_leftovers)
	test_df = pd.concat([removed_leftover_bit, test_df]).reset_index(drop=True)
	leftovers_df = leftovers_df.head(leftovers_df.shape[0] - needed_leftovers)

	#print("\nNew test df")
	#print(test_df)

	#print("\nNew leftovers")
	#print(leftovers_df)

	#verify that it worked
	num_test_df_records = test_df.shape[0]
	res = num_test_df_records%sequence_size
	if res !=0:
		print("num_test_df_records mod sequence_size !=0")
		print(res)
		sys.exit(0)
	#print("Correctly retrieved y test info... Now we need the initial condition...")

	initial_condition_size = IO_TUPLE[0]
	#print("\ninitial_condition_size: %d" %initial_condition_size)

	init_condit = leftovers_df.tail(initial_condition_size)
	#print("\ninit_condit")
	#print(init_condit)

	#now remove and add to test
	leftovers_df = leftovers_df.head(leftovers_df.shape[0] - initial_condition_size)
	test_df = pd.concat([init_condit, test_df]).reset_index(drop=True)

	#print("\nnew leftovers")
	#print(leftovers_df)
	#print("\ntest df with init condit")
	#print(test_df)

	#verify
	num_leftovers = leftovers_df.shape[0]
	num_test_records = test_df.shape[0]
	total_records = num_leftovers + num_test_records
	if total_records != df.shape[0]:
		print("error total_records != df.shape[0]")
		print("total: %d"%total_records)
		print("original: %d"%df.shape[0])
	# else:
	# 	print("\nCorrectly created records")

	return leftovers_df,test_df

def split_dfs_into_train_and_test_backup(df, test_start, test_end,IO_TUPLE):

	#get test df
	test_df = config_df_by_dates(df, test_start, test_end,"nodeTime")

	#get test nodes
	test_nodeIDs = list(test_df["nodeID"])
	print("\nNum test nodeiDs %d"%len(test_nodeIDs))

	#get leftover df
	leftover_df = df.head(df.shape[0] - test_df.shape[0])

	#verify split
	print("\ntest df")
	print(test_df)
	print("\nleftovers")
	print(leftover_df)
	num_test_records = test_df.shape[0]
	num_leftover_records = leftover_df.shape[0]
	num_df_records = df.shape[0]
	print("\nNum test records: %d" %num_test_records)
	print("Num leftover records: %d" %num_leftover_records)
	print("Num original records: %d" %num_df_records)
	total_records = leftover_df.shape[0] + test_df.shape[0]
	if total_records != num_df_records:
		print("\nError! total_records != num_df_records")
		print(total_records)
		print(num_df_records)
		sys.exit(0)

	#we need to take the last bit of records from leftovers and add it to the test df


	return

def make_output_dir_from_params(PARAM_VALS, main_output_dir,ARCH_TUPLE):

  PARAM_TAG = ""
  for p in PARAM_VALS:
    if str(p) == str(ARCH_TUPLE):
      for cur_tuple in ARCH_TUPLE:
        for e in cur_tuple:
         PARAM_TAG = PARAM_TAG + "-" + str(e)
    else:
      PARAM_TAG = PARAM_TAG + "-" + str(p)
  print(PARAM_TAG)
  output_dir =  main_output_dir + PARAM_TAG + "/"
  if not os.path.exists(output_dir):
      os.makedirs(output_dir)

  return output_dir

def make_output_dir_from_params_v2(PARAM_VALS, main_output_dir):

  PARAM_TAG = ""
  for p in PARAM_VALS:
      PARAM_TAG = PARAM_TAG + "-" + str(p)
  print(PARAM_TAG)
  output_dir =  main_output_dir + PARAM_TAG + "/"
  if not os.path.exists(output_dir):
      os.makedirs(output_dir)

  return output_dir

def make_param_tag(PARAM_VALS):

    PARAM_TAG = ""
    for p in PARAM_VALS:
        PARAM_TAG = PARAM_TAG + "-" + str(p)

    PARAM_TAG=PARAM_TAG[1:]


    return PARAM_TAG

#get fts
def get_cascade_model_fts(input_dir):
	model_fts = []
	with open(input_dir + "model-fts.txt") as f:
		for line in f:
			line = line.replace("\n","")
			model_fts.append(line)
			print(line)
	return model_fts

def get_old_and_new_users_by_cascade(df, MOD_NUM=10000):

	num_records = df.shape[0]

	#groupby cascade
	df["nodeTime"] = pd.to_datetime(df["nodeTime"], utc=True)
	df = df.sort_values("nodeTime")

	temp = df[["rootID","nodeUserID"]].drop_duplicates().reset_index(drop=True)

	#get user list
	users = list(temp["nodeUserID"])

	#make seen user set
	seen_user_set = set()

	num_users = len(users)
	is_old_user_list = [0 for i in range(num_users)]
	is_new_user_list = [0 for i in range(num_users)]

	for i,user in enumerate(users):
		if i%MOD_NUM == 0:
			print("On user %d of %d" %(i, num_users))

		if user in seen_user_set:
			is_old_user_list[i] = 1
		else:
			is_new_user_list[i] = 1
			seen_user_set.add(user)

	temp["is_old_user"] = is_old_user_list
	temp["is_new_user"] = is_new_user_list

	df = df.merge(temp, on=["rootID", "nodeUserID"], how="inner")
	df = df.reset_index(drop=True)
	print(df)

	old_and_new_user_records = df[["nodeUserID","rootID", "is_old_user", "is_new_user"]].drop_duplicates().reset_index(drop=True)
	print(old_and_new_user_records)

	old_new_counts = df["is_old_user"].sum() + df["is_new_user"].sum()
	print("\nnum records: %d" %num_records)
	print("old_new_counts: %d"%old_new_counts)
	if old_new_counts != num_records:
		print("\nError! old_new_counts != num_records")
		sys.exit(0)

	return df,old_and_new_user_records



def mark_new_users(df, old_user_start, old_user_end):
	old_user_df = config_df_by_dates(df, old_user_start, old_user_end)
	old_users = set(old_user_df["nodeUserID"].unique())
	df["is_old_user"] = df["nodeUserID"].isin(old_users).astype("int32")
	df["is_new_user"] = np.logical_xor(df["is_old_user"], 1).astype("int32")
	# print(df[["is_old_user", "is_new_user"]])
	old_and_new_user_records = df[["nodeUserID", "is_old_user", "is_new_user"]].drop_duplicates().reset_index(drop=True)
	print(old_and_new_user_records)
	return df,old_and_new_user_records

#count new and old for cascade
def get_new_and_old_user_counts(df, old_user_start, old_user_end,groupby_col):
	df,old_and_new_user_records = mark_new_users(df, old_user_start, old_user_end)
	df["num_old_users"] = df.groupby([groupby_col])["is_old_user"].transform("sum")
	df["num_new_users"] = df.groupby([groupby_col])["is_new_user"].transform("sum")
	old_and_new_cols = ["num_old_users", "num_new_users"]

	return df,old_and_new_cols,old_and_new_user_records

def get_new_and_old_user_counts_by_cascade(df, groupby_col, MOD_NUM=10000):
	df,old_and_new_user_records = get_old_and_new_users_by_cascade(df, MOD_NUM=MOD_NUM)
	df["num_old_users"] = df.groupby([groupby_col])["is_old_user"].transform("sum")
	df["num_new_users"] = df.groupby([groupby_col])["is_new_user"].transform("sum")
	old_and_new_cols = ["num_old_users", "num_new_users"]

	return df,old_and_new_cols,old_and_new_user_records

def config_df_by_dates(df,start_date,end_date,time_col="nodeTime"):
    df[time_col]=pd.to_datetime(df[time_col], utc=True)
    df = df.sort_values(time_col)
    df=df.set_index(time_col)

    if (start_date != None) and (end_date != None):
        df = df[(df.index >= start_date) & (df.index <= end_date)]

    df=df.reset_index(drop=False)
    return df


def get_action_count_cols(df, actions,groupby_col):

	for action in actions:
		df[action] = df["actionType"].isin([action]).astype("int32")
		df[action] = df.groupby([groupby_col])[action].transform("sum")

	#verify
	temp = df[[groupby_col] + actions].drop_duplicates()
	total_records = df.shape[0]
	num_actions = 0
	for action in actions:
		num_actions+=temp[action].sum()
	print("\nTotal records: %d" %total_records)
	print("Total actions: %d" %num_actions)
	if total_records != num_actions:
		print("Error! total_records != num_actions")
		sys.exit(0)
	else:
		print("Actions and records match!")

	return df

def get_infoID_count_cols(df, infoIDs,groupby_col):

	target_count = df[df["informationID"].isin(infoIDs)].shape[0]
	actual_count = 0
	for infoID in infoIDs:
		df[infoID] = df["informationID"].isin([infoID]).astype("int32")
		df[infoID] = df.groupby([groupby_col])[infoID].transform("sum")
		# actual_count+=df[infoID].sum()
		# final_cols.append(infoID)

	temp = df[["rootID"]+ infoIDs].drop_duplicates()
	for infoID in infoIDs:
		actual_count+=temp[infoID].sum()

	print("\ntarget count: %d, actual: %d" %(target_count, actual_count))
	if target_count != actual_count:
		print("Error! target_count != actual_count")
		sys.exit(0)

	return df,target_count

def get_cascade_nodeTime_delta_col(df, initial_date):
	print("\nGetting cascade nodeTime delta col from %s" %initial_date)
	df["nodeTime"] = pd.to_datetime(df["nodeTime"], utc=True)
	df = df.sort_values("nodeTime").reset_index(drop=True)

	initial_date = pd.to_datetime(initial_date, utc=True)
	df["cascade_start_nodeTime_delta"] = df["nodeTime"] - initial_date
	df["cascade_start_nodeTime_delta"] = df["cascade_start_nodeTime_delta"]/pd.to_timedelta(1, unit='D')

	return df

def get_param_df_and_output_dir(main_output_dir, PARAMS, PARAM_NAMES):
	print("\nCreating and saving param df...")
	output_tag = ""
	PARAM_DF = pd.DataFrame(data={"param":PARAM_NAMES, "param_vals":PARAMS})
	PARAM_DF = PARAM_DF[["param", "param_vals"]]
	print(PARAM_DF)

	for p in PARAMS:
		output_tag=output_tag + str(p) + "-"
	output_tag = output_tag[:-1]
	output_dir = main_output_dir + output_tag + "/"
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	output_fp = output_dir + "params.csv"
	PARAM_DF.to_csv(output_fp)
	print(output_fp)

	return PARAM_DF,output_dir

def rank_users_by_age(df):

	#make user df
	user_df = df.copy()
	user_df["nodeTime"] = pd.to_datetime(df["nodeTime"], utc=True)
	user_df = user_df.sort_values("nodeTime")
	user_df = user_df[["nodeUserID"]].drop_duplicates(keep="first").reset_index(drop=True)

	#rank users
	user_df["original_nodeUserID"] = user_df["nodeUserID"].copy()
	user_df["nodeUserID"] = [(i+1) for i in range(user_df["nodeUserID"].shape[0])]

	#merge with original
	df["original_nodeUserID"] = df["nodeUserID"].copy()
	df = df.drop(["nodeUserID"], axis=1)
	print("\ndf size before merge")
	print(df.shape)
	df = pd.merge(user_df,df,on="original_nodeUserID",how="inner")
	print("\ndf size after merge")
	print(df.shape)
	# sys.exit(0)

	#make sure it worked
	nunique_original_users = df["original_nodeUserID"].nunique()
	nunique_ints = df["nodeUserID"].nunique()
	print("\nnunique_original_users: %d" %nunique_original_users)
	print("nunique_ints: %d" %nunique_ints)
	if nunique_ints != nunique_original_users:
		print("\nError! nunique_ints != nunique_original_users")
		sys.exit(0)

	#encode ints
	int_to_user_dict = pd.Series(user_df.original_nodeUserID.values,index=user_df.nodeUserID).to_dict()
	user_to_int_dict = pd.Series(user_df.nodeUserID.values,index=user_df.original_nodeUserID).to_dict()

	i = 0
	for user, user_int in user_to_int_dict.items():
		print("User: %s, int: %s" %(user, user_int))
		i+=1
		if i==10:
			break

	i = 0
	for user_int,user in int_to_user_dict.items():
		print("User: %s, int: %s" %(user, user_int))
		i+=1
		if i==10:
			break

	#re-sort
	df["nodeTime"] = pd.to_datetime(df["nodeTime"], utc=True)
	df = df.sort_values("nodeTime").reset_index(drop=True)

	return df,int_to_user_dict,user_to_int_dict

def rank_users_by_activity(df):
	user_df = df[["nodeUserID"]]
	user_df["num_actions"] = user_df.groupby(["nodeUserID"])["nodeUserID"].transform("size")
	user_df = user_df[["nodeUserID", "num_actions"]].drop_duplicates().reset_index(drop=True)
	user_df = user_df.sort_values("num_actions", ascending=True).reset_index(drop=True)

	#rank users
	user_df["original_nodeUserID"] = user_df["nodeUserID"].copy()
	user_df["nodeUserID"] = [(i+1) for i in range(user_df["nodeUserID"].shape[0])]
	df["original_nodeUserID"] = df["nodeUserID"].copy()
	df = df.drop(["nodeUserID"], axis=1)
	df = pd.merge(user_df,df,on="original_nodeUserID",how="inner")

	int_to_user_dict = pd.Series(user_df.original_nodeUserID.values,index=user_df.nodeUserID).to_dict()
	user_to_int_dict = pd.Series(user_df.nodeUserID.values,index=user_df.original_nodeUserID).to_dict()

	i = 0
	for user, user_int in user_to_int_dict.items():
		print("User: %s, int: %s" %(user, user_int))
		i+=1
		if i==10:
			break

	i = 0
	for user_int,user in int_to_user_dict.items():
		print("User: %s, int: %s" %(user, user_int))
		i+=1
		if i==10:
			break

	#re-sort
	df["nodeTime"] = pd.to_datetime(df["nodeTime"], utc=True)
	df = df.sort_values("nodeTime").reset_index(drop=True)

	return df,int_to_user_dict,user_to_int_dict

#get root function
def get_rootUserID_col(df):

    print("Making nodeID to user dict...")
    nodeID_to_nodeUserID_dict = pd.Series(df.nodeUserID.values,index=df.nodeID).to_dict()
    print("Made nodeID dict. Num elements: %d" %len(nodeID_to_nodeUserID_dict.keys()))

    #I want to have a dict where I can put in the rootID  and get the rootIDUser
    rootIDs = list(df["rootID"])
    # infoIDs = list(df["informationID"])
    rootUserIDs = []
    print("Getting rootIDUsers...")
    for rootID in rootIDs:
        try:
            rootUserID = nodeID_to_nodeUserID_dict[rootID]
        except:
            rootUserID = "missing_rootUserID"
            # rootUserID = rootUserID + concat_tag + infoID

        rootUserIDs.append(rootUserID)
    df["rootUserID"] = rootUserIDs
    print(df["rootUserID"])

    return df

#get root function
def get_parentUserID_col(df):

    print("Making nodeID to user dict...")
    nodeID_to_nodeUserID_dict = pd.Series(df.nodeUserID.values,index=df.nodeID).to_dict()
    print("Made nodeID dict. Num elements: %d" %len(nodeID_to_nodeUserID_dict.keys()))

    #I want to have a dict where I can put in the parentID  and get the parentIDUser
    parentIDs = list(df["parentID"])
    # infoIDs = list(df["informationID"])
    parentUserIDs = []
    print("Getting parentIDUsers...")
    for parentID in parentIDs:
        try:
            parentUserID = nodeID_to_nodeUserID_dict[parentID]
        except:
            parentUserID = "missing_parentUserID"
            # parentUserID = parentUserID + concat_tag + infoID

        parentUserIDs.append(parentUserID)
    df["parentUserID"] = parentUserIDs
    print(df["parentUserID"])

    return df

def get_parentUserID_col_v2_pnnl_version(df, parent_node_col="parentID", node_col="nodeID", root_node_col="rootID", user_col="nodeUserID"):
    """
    :return: adds parentUserID column with user id of the parent if it exits in df
    if it doesn't exist, uses the user id of the root instead
    if both doesn't exist: NaN
    """
    tweet_uids = pd.Series(df[user_col].values, index=df[node_col]).to_dict()

    df['parentUserID'] = df[parent_node_col].map(tweet_uids)

    df.loc[(df[root_node_col] != df[node_col]) & (df['parentUserID'].isnull()), 'parentUserID'] = \
        df[(df[root_node_col] != df[node_col]) & (df['parentUserID'].isnull())][root_node_col].map(tweet_uids)

    # df = df[df['nodeUserID'] != df['parentUserID']]

    df['parentUserID'] = df['parentUserID'].fillna("missing_parentUserID")

    return df


def get_parentUserIDs_and_rootUserIDs_and_kick_out_missing_root_cascades(df):
    print("Getting root users and parent users...")
    df = get_rootUserID_col(df)
    df = get_parentUserID_col(df)
    parent_filter_df = df[df["parentUserID"] == "missing_parentUserID"]
    root_filter_df = df[df["rootUserID"] == "missing_rootUserID"]
    rootIDs_to_kickout = set(list(parent_filter_df["rootID"]) + list(root_filter_df["rootID"]))
    df = df[~df["rootID"].isin(rootIDs_to_kickout)]
    df = df.reset_index(drop=True)
    return df

def get_parentUserIDs_and_rootUserIDs_and_kick_out_partial_cascades(df):
	print("Getting root users and parent users...")
	df = get_rootUserID_col(df)
	df = get_parentUserID_col(df)
	parent_filter_df = df[df["parentUserID"] == "missing_parentUserID"]
	root_filter_df = df[df["rootUserID"] == "missing_rootUserID"]
	rootIDs_to_kickout = set(list(parent_filter_df["rootID"]) + list(root_filter_df["rootID"]))
	df = df[~df["rootID"].isin(rootIDs_to_kickout)]
	df = df.reset_index(drop=True)
	return df

# def mark_new_users(df, old_user_start, old_user_end):
# 	old_user_df = config_df_by_dates(df, old_user_start, old_user_end)
# 	old_users = set(old_user_df["nodeUserID"].unique())
# 	df["is_old_user"] = df["nodeUserID"].isin(old_users).astype("int32")
# 	df["is_new_user"] = np.logical_xor(df["is_old_user"], 1).astype("int32")
# 	# print(df[["is_old_user", "is_new_user"]])
# 	old_and_new_user_records = df[["nodeUserID", "is_old_user", "is_new_user"]].drop_duplicates().reset_index(drop=True)
# 	print(old_and_new_user_records)
# 	return df,old_and_new_user_records

def replace_new_users_with_tags(df):

	new_userIDs = []
	users = df["nodeUserID"]
	new_user_mask = df["is_new_user"]
	for i in range(len(users)):
		is_new_user = new_user_mask[i]
		user = users[i]
		if is_new_user == 1:
			new_userIDs.append("new_user")
		else:
			new_userIDs.append(user)
	df["really_original_nodeUserID"] = df["nodeUserID"].copy()
	df["nodeUserID"] = new_userIDs

	new_user_original_records = df[["really_original_nodeUserID", "nodeUserID"]].drop_duplicates().reset_index(drop=True)

	return df,new_user_original_records

def get_user_influence_col(df,MOD_NUM=1000):

    df = get_parentUserID_col(df)
    df = get_rootUserID_col(df)
    # df = remove_missing_root_users_from_df(df)
    # df = remove_missing_parent_users_from_df(df)

    #make user influence dict
    user_inf_dict = {}
    node_users = df["nodeUserID"]
    root_users = df["rootUserID"]
    parent_users = df["parentUserID"]

    #num recs
    num_records = df.shape[0]

    for i in range(num_records):

        node_user = node_users[i]
        parent_user = parent_users[i]
        root_user = root_users[i]

        if node_user not in user_inf_dict:
            user_inf_dict[node_user] = 0

        if parent_user not in user_inf_dict:
            user_inf_dict[parent_user] = 0

        if root_user not in user_inf_dict:
            user_inf_dict[root_user] = 0

        if (root_user != node_user):
            user_inf_dict[root_user] +=1

        if (parent_user != root_user):
            user_inf_dict[parent_user] +=1

        if i%MOD_NUM == 0:
            print("Processed user %d of %d infl. info" %(i, num_records))

    #now map counts to a user
    print("\n\nMapping influence counts to nodeUserIDs...")
    df["user_influence"] = df["nodeUserID"].map(user_inf_dict)
    # df = df.fillna(0)
    # print(df["user_influence"].drop_duplicates().value_counts())
    print("Done")

    return df

def make_test_set_pred_and_eval(output_dir,x_eval_tag,infoIDs, desired_actions,pair_train_and_test_array_dict,EXTRACT_OUTPUT_SIZE,ft_to_idx_dict,target_ft,RESCALE_TARGET,NUM_TARGET_LOG_NORMS,ZERO_OUT_NEG_RESULTS,RESCALE,scaler,y_scaler,model ,OUTPUT_SIZE,actions):

    create_output_dir(output_dir)
    #make pred dict
    baseline_pred_dict = get_m1_shifted_baseline_pred_dict_v3_x_array_option(infoIDs, desired_actions,pair_train_and_test_array_dict,EXTRACT_OUTPUT_SIZE,ft_to_idx_dict,target_ft,x_eval_tag )

    num_pairs = len(actions) * len(infoIDs)

    #predict!
    i = 0
    pair_pred_dict = {}
    for infoID in infoIDs:
        pair_pred_dict[infoID] = {}
        for action in desired_actions:
            x_array = pair_train_and_test_array_dict[infoID][action][x_eval_tag]
            #y_test = pair_train_and_test_array_dict[infoID][action]["y_test"]

            if RESCALE == True:
                x_array =normalize_data_with_scaler(x_array, scaler)

            # if NUM_TARGET_LOG_NORMS == True:
            #   y_test = np.log1p(y_test)

            y_pred = model.predict(x_array)

            if RESCALE_TARGET == True:
                old_shape = y_pred.shape
                # y_pred = y_pred.reshape((y_pred.shape[0], 1, 1))
                y_pred = y_pred.reshape((y_pred.shape[0], OUTPUT_SIZE, 1))
                y_pred =denormalize_single_array(y_pred, y_scaler)
                y_pred = y_pred.reshape(old_shape)

            for j in range(NUM_TARGET_LOG_NORMS):
                y_pred = np.expm1(y_pred)

            if ZERO_OUT_NEG_RESULTS == True:
                y_pred[y_pred<0]=0

            pair_pred_dict[infoID][action] = y_pred
            i+=1
            print("Got pair df %d of %d" %(i, num_pairs))

    # =================================== score results ===================================
    vam_pred_tag = "VAM"
    baseline_pred_tag = "Shifted-Baseline"
    with open(output_dir + "pair_pred_dict", 'wb') as handle:
        pickle.dump(pair_pred_dict, handle)

    with open(output_dir + "pair_train_and_test_array_dict", 'wb') as handle:
        pickle.dump(pair_train_and_test_array_dict, handle)

    # ========================== result dfs ====================================================
    # vam_result_df = score_pred_results_v2(pair_pred_dict,pair_train_and_test_array_dict,desired_actions, infoIDs,output_dir,vam_pred_tag)
    if x_eval_tag == "x_test":
        y_eval_tag = "y_test"
    else:
        y_eval_tag = "y_val"
    vam_result_df =score_pred_results_v3_with_array_option(pair_pred_dict,pair_train_and_test_array_dict,desired_actions, infoIDs,output_dir,vam_pred_tag,y_eval_tag)
    # vam_infoID_result_df = get_infoID_combined_result(vam_result_df,vam_pred_tag,output_dir)
    # vam_action_result_df = action_result_df(vam_result_df, output_dir,vam_pred_tag)

    # ========================== baseline dfs ====================================================
    #baseline_result_df = score_pred_results_v2(baseline_pred_dict,pair_train_and_test_array_dict,desired_actions, infoIDs,output_dir,baseline_pred_tag)
    baseline_result_df =score_pred_results_v3_with_array_option(baseline_pred_dict,pair_train_and_test_array_dict,desired_actions, infoIDs,output_dir,baseline_pred_tag,y_eval_tag)
    # baseline_infoID_result_df = get_infoID_combined_result(baseline_result_df,baseline_pred_tag,output_dir)
    # baseline_action_result_df = action_result_df(baseline_result_df, output_dir,baseline_pred_tag)

    #comp info
    merge_cols = ["infoID", "action"]
    rename_cols = ["mape", "rmse"]
    vam_tag = "VAM"
    baseline_tag = "Shifted-Baseline"
    metric_of_interest="rmse"

    #get comp info
    comp_df,winner_tag = m1_make_comp_df(vam_result_df, baseline_result_df,merge_cols,rename_cols,vam_tag,baseline_tag,metric_of_interest)
    summary_df = m1_get_model_win_info(comp_df,winner_tag)

    output_fp = output_dir + "NN-vs-Baseline-Comps.csv"
    comp_df.to_csv(output_fp, index=False)
    print(output_fp)

    output_fp = output_dir + "Summary-Baseline-Comps.csv"
    summary_df.to_csv(output_fp, index=False)
    print(output_fp)

# score_pred_results_v3_with_array_option_v4_timestep_dim

def make_test_set_pred_and_eval_v4_with_timestep_dim(output_dir,x_eval_tag,infoIDs, desired_actions,pair_train_and_test_array_dict,EXTRACT_OUTPUT_SIZE,ft_to_idx_dict,target_ft,RESCALE_TARGET,NUM_TARGET_LOG_NORMS,ZERO_OUT_NEG_RESULTS,RESCALE,scaler,y_scaler,model ,OUTPUT_SIZE):

    create_output_dir(output_dir)
    #make pred dict
    baseline_pred_dict = get_m1_shifted_baseline_pred_dict_v3_x_array_option(infoIDs, desired_actions,pair_train_and_test_array_dict,EXTRACT_OUTPUT_SIZE,ft_to_idx_dict,target_ft,x_eval_tag )

    num_pairs = len(desired_actions) * len(infoIDs)

    #predict!
    i = 0
    pair_pred_dict = {}
    for infoID in infoIDs:
        pair_pred_dict[infoID] = {}
        for action in desired_actions:
            x_array = pair_train_and_test_array_dict[infoID][action][x_eval_tag]
            #y_test = pair_train_and_test_array_dict[infoID][action]["y_test"]

            if RESCALE == True:
                x_array =normalize_data_with_scaler(x_array, scaler)

            # if NUM_TARGET_LOG_NORMS == True:
            #   y_test = np.log1p(y_test)

            y_pred = model.predict(x_array)

            if RESCALE_TARGET == True:
                old_shape = y_pred.shape
                # y_pred = y_pred.reshape((y_pred.shape[0], 1, 1))
                y_pred = y_pred.reshape((y_pred.shape[0], OUTPUT_SIZE, 1))
                y_pred =denormalize_single_array(y_pred, y_scaler)
                y_pred = y_pred.reshape(old_shape)

            for j in range(NUM_TARGET_LOG_NORMS):
                y_pred = np.expm1(y_pred)

            if ZERO_OUT_NEG_RESULTS == True:
                y_pred[y_pred<0]=0

            pair_pred_dict[infoID][action] = y_pred
            i+=1
            print("Got pair df %d of %d" %(i, num_pairs))

    # =================================== score results ===================================
    vam_pred_tag = "VAM"
    baseline_pred_tag = "Shifted-Baseline"
    with open(output_dir + "pair_pred_dict", 'wb') as handle:
        pickle.dump(pair_pred_dict, handle)

    with open(output_dir + "pair_train_and_test_array_dict", 'wb') as handle:
        pickle.dump(pair_train_and_test_array_dict, handle)

    # ========================== result dfs ====================================================
    # vam_result_df = score_pred_results_v2(pair_pred_dict,pair_train_and_test_array_dict,desired_actions, infoIDs,output_dir,vam_pred_tag)
    if x_eval_tag == "x_test":
        y_eval_tag = "y_test"

    if x_eval_tag == "x_val":
        y_eval_tag = "y_val"

    if x_eval_tag == "x_test_sliding_window":
        y_eval_tag = "y_test_sliding_window"

    if x_eval_tag == "x_val_sliding_window":
        y_eval_tag = "y_val_sliding_window"
    vam_result_df =score_pred_results_v3_with_array_option_v4_timestep_dim(pair_pred_dict,pair_train_and_test_array_dict,desired_actions, infoIDs,output_dir,vam_pred_tag,y_eval_tag)
    # vam_infoID_result_df = get_infoID_combined_result(vam_result_df,vam_pred_tag,output_dir)
    # vam_action_result_df = action_result_df(vam_result_df, output_dir,vam_pred_tag)

    # ========================== baseline dfs ====================================================
    #baseline_result_df = score_pred_results_v2(baseline_pred_dict,pair_train_and_test_array_dict,desired_actions, infoIDs,output_dir,baseline_pred_tag)
    baseline_result_df =score_pred_results_v3_with_array_option_v4_timestep_dim(baseline_pred_dict,pair_train_and_test_array_dict,desired_actions, infoIDs,output_dir,baseline_pred_tag,y_eval_tag)
    # baseline_infoID_result_df = get_infoID_combined_result(baseline_result_df,baseline_pred_tag,output_dir)
    # baseline_action_result_df = action_result_df(baseline_result_df, output_dir,baseline_pred_tag)

    #=========================== with timestep info ===========================
    #comp info
    merge_cols = ["infoID", "action","timestep"]
    rename_cols = ["mape", "rmse"]
    vam_tag = "VAM"
    baseline_tag = "Shifted-Baseline"
    metric_of_interest="rmse"

    #get comp info
    comp_df,winner_tag = m1_make_comp_df(vam_result_df, baseline_result_df,merge_cols,rename_cols,vam_tag,baseline_tag,metric_of_interest)
    summary_df = m1_get_model_win_info(comp_df,winner_tag)

    output_fp = output_dir + "NN-vs-Baseline-Comps-with-timestep-info.csv"
    comp_df.to_csv(output_fp, index=False)
    print(output_fp)

    output_fp = output_dir + "Summary-Baseline-Comps-with-timestep-info.csv"
    summary_df.to_csv(output_fp, index=False)
    print(output_fp)

    #=========================== without timestep info ===========================
    vam_result_df =score_pred_results_v3_with_array_option(pair_pred_dict,pair_train_and_test_array_dict,desired_actions, infoIDs,output_dir,vam_pred_tag,y_eval_tag)
    baseline_result_df =score_pred_results_v3_with_array_option(baseline_pred_dict,pair_train_and_test_array_dict,desired_actions, infoIDs,output_dir,baseline_pred_tag,y_eval_tag)
    merge_cols = ["infoID", "action"]

    #get comp info
    comp_df,winner_tag = m1_make_comp_df(vam_result_df, baseline_result_df,merge_cols,rename_cols,vam_tag,baseline_tag,metric_of_interest)
    summary_df = m1_get_model_win_info(comp_df,winner_tag)

    output_fp = output_dir + "NN-vs-Baseline-Comps-without-timestep-info.csv"
    comp_df.to_csv(output_fp, index=False)
    print(output_fp)

    output_fp = output_dir + "Summary-Baseline-Comps-without-timestep-info.csv"
    summary_df.to_csv(output_fp, index=False)
    print(output_fp)


def make_test_set_pred_and_eval_v2(output_dir,x_eval_tag,infoIDs, desired_actions,pair_train_and_test_array_dict,EXTRACT_OUTPUT_SIZE,ft_to_idx_dict,target_ft,RESCALE_TARGET,NUM_TARGET_LOG_NORMS,ZERO_OUT_NEG_RESULTS,RESCALE,scaler,y_scaler,model ,OUTPUT_SIZE):

    create_output_dir(output_dir)
    #make pred dict
    baseline_pred_dict = get_m1_shifted_baseline_pred_dict_v3_x_array_option(infoIDs, desired_actions,pair_train_and_test_array_dict,EXTRACT_OUTPUT_SIZE,ft_to_idx_dict,target_ft,x_eval_tag )

    num_pairs = len(desired_actions) * len(infoIDs)

    #predict!
    i = 0
    pair_pred_dict = {}
    for infoID in infoIDs:
        pair_pred_dict[infoID] = {}
        for action in desired_actions:
            x_array = pair_train_and_test_array_dict[infoID][action][x_eval_tag]
            #y_test = pair_train_and_test_array_dict[infoID][action]["y_test"]

            if RESCALE == True:
                x_array =normalize_data_with_scaler(x_array, scaler)

            # if NUM_TARGET_LOG_NORMS == True:
            #   y_test = np.log1p(y_test)

            y_pred = model.predict(x_array)

            if RESCALE_TARGET == True:
                old_shape = y_pred.shape
                # y_pred = y_pred.reshape((y_pred.shape[0], 1, 1))
                y_pred = y_pred.reshape((y_pred.shape[0], OUTPUT_SIZE, 1))
                y_pred =denormalize_single_array(y_pred, y_scaler)
                y_pred = y_pred.reshape(old_shape)

            for j in range(NUM_TARGET_LOG_NORMS):
                y_pred = np.expm1(y_pred)

            if ZERO_OUT_NEG_RESULTS == True:
                y_pred[y_pred<0]=0

            pair_pred_dict[infoID][action] = y_pred
            i+=1
            print("Got pair df %d of %d" %(i, num_pairs))

    # =================================== score results ===================================
    vam_pred_tag = "VAM"
    baseline_pred_tag = "Shifted-Baseline"
    with open(output_dir + "pair_pred_dict", 'wb') as handle:
        pickle.dump(pair_pred_dict, handle)

    with open(output_dir + "pair_train_and_test_array_dict", 'wb') as handle:
        pickle.dump(pair_train_and_test_array_dict, handle)

    # ========================== result dfs ====================================================
    # vam_result_df = score_pred_results_v2(pair_pred_dict,pair_train_and_test_array_dict,desired_actions, infoIDs,output_dir,vam_pred_tag)
    if x_eval_tag == "x_test":
        y_eval_tag = "y_test"

    if x_eval_tag == "x_val":
        y_eval_tag = "y_val"

    if x_eval_tag == "x_test_sliding_window":
        y_eval_tag = "y_test_sliding_window"

    if x_eval_tag == "x_val_sliding_window":
        y_eval_tag = "y_val_sliding_window"
    vam_result_df =score_pred_results_v3_with_array_option(pair_pred_dict,pair_train_and_test_array_dict,desired_actions, infoIDs,output_dir,vam_pred_tag,y_eval_tag)
    # vam_infoID_result_df = get_infoID_combined_result(vam_result_df,vam_pred_tag,output_dir)
    # vam_action_result_df = action_result_df(vam_result_df, output_dir,vam_pred_tag)

    # ========================== baseline dfs ====================================================
    #baseline_result_df = score_pred_results_v2(baseline_pred_dict,pair_train_and_test_array_dict,desired_actions, infoIDs,output_dir,baseline_pred_tag)
    baseline_result_df =score_pred_results_v3_with_array_option(baseline_pred_dict,pair_train_and_test_array_dict,desired_actions, infoIDs,output_dir,baseline_pred_tag,y_eval_tag)
    # baseline_infoID_result_df = get_infoID_combined_result(baseline_result_df,baseline_pred_tag,output_dir)
    # baseline_action_result_df = action_result_df(baseline_result_df, output_dir,baseline_pred_tag)

    #comp info
    merge_cols = ["infoID", "action"]
    rename_cols = ["mape", "rmse"]
    vam_tag = "VAM"
    baseline_tag = "Shifted-Baseline"
    metric_of_interest="rmse"

    #get comp info
    comp_df,winner_tag = m1_make_comp_df(vam_result_df, baseline_result_df,merge_cols,rename_cols,vam_tag,baseline_tag,metric_of_interest)
    summary_df = m1_get_model_win_info(comp_df,winner_tag)

    output_fp = output_dir + "NN-vs-Baseline-Comps.csv"
    comp_df.to_csv(output_fp, index=False)
    print(output_fp)

    output_fp = output_dir + "Summary-Baseline-Comps.csv"
    summary_df.to_csv(output_fp, index=False)
    print(output_fp)

def make_test_set_pred_and_eval_v3_XGBOOST_FEATURES(flat_ft_to_idx_dict,flat_idx_to_ft_dict,flattened_fts,output_dir,x_eval_tag,infoIDs, desired_actions,pair_train_and_test_array_dict,EXTRACT_OUTPUT_SIZE,ft_to_idx_dict,target_ft,RESCALE_TARGET,NUM_TARGET_LOG_NORMS,ZERO_OUT_NEG_RESULTS,RESCALE,scaler,y_scaler,model ,OUTPUT_SIZE):

    create_output_dir(output_dir)
    #make pred dict
    baseline_pred_dict = get_m1_shifted_baseline_pred_dict_v4_x_array_option(infoIDs,flat_ft_to_idx_dict,flat_idx_to_ft_dict,flattened_fts,  desired_actions,pair_train_and_test_array_dict,EXTRACT_OUTPUT_SIZE,ft_to_idx_dict,target_ft,x_eval_tag )

    num_pairs = len(desired_actions) * len(infoIDs)

    #predict!
    i = 0
    pair_pred_dict = {}
    for infoID in infoIDs:
        pair_pred_dict[infoID] = {}
        for action in desired_actions:
            x_array = pair_train_and_test_array_dict[infoID][action][x_eval_tag]
            #y_test = pair_train_and_test_array_dict[infoID][action]["y_test"]

            if RESCALE == True:
                x_array =normalize_data_with_scaler(x_array, scaler)

            # if NUM_TARGET_LOG_NORMS == True:
            #   y_test = np.log1p(y_test)

            y_pred = model.predict(x_array)

            if RESCALE_TARGET == True:
                old_shape = y_pred.shape
                # y_pred = y_pred.reshape((y_pred.shape[0], 1, 1))
                y_pred = y_pred.reshape((y_pred.shape[0], OUTPUT_SIZE, 1))
                y_pred =denormalize_single_array(y_pred, y_scaler)
                y_pred = y_pred.reshape(old_shape)

            for j in range(NUM_TARGET_LOG_NORMS):
                y_pred = np.expm1(y_pred)

            if ZERO_OUT_NEG_RESULTS == True:
                y_pred[y_pred<0]=0

            pair_pred_dict[infoID][action] = y_pred
            i+=1
            print("Got pair df %d of %d" %(i, num_pairs))

    # =================================== score results ===================================
    vam_pred_tag = "VAM"
    baseline_pred_tag = "Shifted-Baseline"
    with open(output_dir + "pair_pred_dict", 'wb') as handle:
        pickle.dump(pair_pred_dict, handle)

    with open(output_dir + "pair_train_and_test_array_dict", 'wb') as handle:
        pickle.dump(pair_train_and_test_array_dict, handle)

    with open(output_dir + "baseline_pred_dict", "wb")as handle:
        pickle.dump(baseline_pred_dict, handle)
    # ========================== result dfs ====================================================
    # vam_result_df = score_pred_results_v2(pair_pred_dict,pair_train_and_test_array_dict,desired_actions, infoIDs,output_dir,vam_pred_tag)
    if x_eval_tag == "x_test":
        y_eval_tag = "y_test"

    if x_eval_tag == "x_val":
        y_eval_tag = "y_val"

    if x_eval_tag == "x_test_sliding_window":
        y_eval_tag = "y_test_sliding_window"

    if x_eval_tag == "x_val_sliding_window":
        y_eval_tag = "y_val_sliding_window"
    vam_result_df =score_pred_results_v3_with_array_option(pair_pred_dict,pair_train_and_test_array_dict,desired_actions, infoIDs,output_dir,vam_pred_tag,y_eval_tag)
    # vam_infoID_result_df = get_infoID_combined_result(vam_result_df,vam_pred_tag,output_dir)
    # vam_action_result_df = action_result_df(vam_result_df, output_dir,vam_pred_tag)

    # ========================== baseline dfs ====================================================
    #baseline_result_df = score_pred_results_v2(baseline_pred_dict,pair_train_and_test_array_dict,desired_actions, infoIDs,output_dir,baseline_pred_tag)
    baseline_result_df =score_pred_results_v3_with_array_option(baseline_pred_dict,pair_train_and_test_array_dict,desired_actions, infoIDs,output_dir,baseline_pred_tag,y_eval_tag)
    # baseline_infoID_result_df = get_infoID_combined_result(baseline_result_df,baseline_pred_tag,output_dir)
    # baseline_action_result_df = action_result_df(baseline_result_df, output_dir,baseline_pred_tag)

    #comp info
    merge_cols = ["infoID", "action"]
    rename_cols = ["mape", "rmse"]
    vam_tag = "VAM"
    baseline_tag = "Shifted-Baseline"
    metric_of_interest="rmse"

    #get comp info
    comp_df,winner_tag = m1_make_comp_df(vam_result_df, baseline_result_df,merge_cols,rename_cols,vam_tag,baseline_tag,metric_of_interest)
    summary_df = m1_get_model_win_info(comp_df,winner_tag)

    output_fp = output_dir + "NN-vs-Baseline-Comps.csv"
    comp_df.to_csv(output_fp, index=False)
    print(output_fp)

    output_fp = output_dir + "Summary-Baseline-Comps.csv"
    summary_df.to_csv(output_fp, index=False)
    print(output_fp)

def get_pair_train_and_test_array_dict_with_certain_fts_XGBOOST_VERSION(pair_train_and_test_df_dict, infoIDs, desired_actions,INPUT_SIZE, EXTRACT_OUTPUT_SIZE, dynamic_fts, target_fts,static_fts,GET_1HOT_INFO_ID_FTS, GET_1HOT_ACTION_FTS):
        #get num combos
    num_pairs = len(desired_actions) * len(infoIDs)
    print("\n%d pairs"%num_pairs)

    pair_train_and_test_array_dict = {}
    i = 0
    for infoID in infoIDs:
        pair_train_and_test_array_dict[infoID] = {}
        for action in desired_actions:
            pair_train_and_test_array_dict[infoID][action] = {}

            #get dfs
            train_df = pair_train_and_test_df_dict[infoID][action]["train_df"]
            val_df = pair_train_and_test_df_dict[infoID][action]["val_df"]
            test_df = pair_train_and_test_df_dict[infoID][action]["test_df"]

            #x_train,y_train,ft_to_idx_dict =XGBOOST_convert_df_to_sliding_window_form_diff_x_and_y_fts_no_print_with_idx_dict(train_df,static_features, INPUT_SIZE, EXTRACT_OUTPUT_SIZE, input_fts, target_fts)
            #x_train,y_train,ft_to_idx_dict =convert_df_to_sliding_window_form_diff_x_and_y_fts_no_print_with_idx_dict(train_df, INPUT_SIZE, EXTRACT_OUTPUT_SIZE, input_fts, target_fts)
            x_train,y_train,ft_to_idx_dict =convert_df_to_sliding_window_form_diff_x_and_y_fts_no_print_with_idx_dict(train_df, INPUT_SIZE, EXTRACT_OUTPUT_SIZE, dynamic_fts, target_fts)
            x_train,flat_ft_to_idx_dict,flat_idx_to_ft_dict,flattened_fts = insert_static_1hot_fts( action,infoID,x_train, dynamic_fts,ft_to_idx_dict ,static_fts,infoIDs,desired_actions,GET_1HOT_INFO_ID_FTS, GET_1HOT_ACTION_FTS)
            print("\nx and y train arrays")
            print(x_train.shape)
            print(y_train.shape)



            #agg y
            y_train = agg_y_array(y_train)

            #get val arrays
            x_val,y_val = convert_df_to_test_sliding_window_form_diff_x_and_y_fts_no_print(val_df, INPUT_SIZE, EXTRACT_OUTPUT_SIZE,dynamic_fts, target_fts,MOD_NUM=1000)
            x_val,_,_,_ = insert_static_1hot_fts(action,infoID,x_val, dynamic_fts,ft_to_idx_dict ,static_fts,infoIDs,desired_actions,GET_1HOT_INFO_ID_FTS, GET_1HOT_ACTION_FTS)
            y_val = agg_y_array(y_val)
            print("\nx and y val arrays")
            print(x_val.shape)
            print(y_val.shape)



            #sliding window train
            x_val_sliding_window,y_val_sliding_window,ft_to_idx_dict =convert_df_to_sliding_window_form_diff_x_and_y_fts_no_print_with_idx_dict(val_df, INPUT_SIZE, EXTRACT_OUTPUT_SIZE, dynamic_fts, target_fts)
            x_val_sliding_window,_,_,_ = insert_static_1hot_fts(action,infoID,x_val_sliding_window, dynamic_fts,ft_to_idx_dict ,static_fts,infoIDs,desired_actions,GET_1HOT_INFO_ID_FTS, GET_1HOT_ACTION_FTS)
            y_val_sliding_window = agg_y_array(y_val_sliding_window)
            print("\nx and y val sliding arrays")
            print(x_val_sliding_window.shape)
            print(y_val_sliding_window.shape)

            #get test arrays
            x_test,y_test = convert_df_to_test_sliding_window_form_diff_x_and_y_fts_no_print(test_df, INPUT_SIZE, EXTRACT_OUTPUT_SIZE,dynamic_fts, target_fts,MOD_NUM=1000)
            x_test,_,_,_ = insert_static_1hot_fts(action,infoID,x_test, dynamic_fts,ft_to_idx_dict ,static_fts,infoIDs,desired_actions,GET_1HOT_INFO_ID_FTS, GET_1HOT_ACTION_FTS)
            y_test = agg_y_array(y_test)
            print("\ny_test shape")
            print(y_test.shape)

            #sliding window test
            x_test_sliding_window,y_test_sliding_window,ft_to_idx_dict =convert_df_to_sliding_window_form_diff_x_and_y_fts_no_print_with_idx_dict(test_df, INPUT_SIZE, EXTRACT_OUTPUT_SIZE, dynamic_fts, target_fts)
            x_test_sliding_window,_,_,_ = insert_static_1hot_fts(action,infoID,x_test_sliding_window, dynamic_fts,ft_to_idx_dict ,static_fts,infoIDs,desired_actions,GET_1HOT_INFO_ID_FTS, GET_1HOT_ACTION_FTS)
            y_test_sliding_window = agg_y_array(y_test_sliding_window)
            print("\nx and y test sliding arrays")
            print(x_test_sliding_window.shape)
            print(y_test_sliding_window.shape)
            # sys.exit(0)



            #data info
            pair_train_and_test_array_dict[infoID][action]["x_train"] = x_train
            pair_train_and_test_array_dict[infoID][action]["y_train"] = y_train
            pair_train_and_test_array_dict[infoID][action]["x_val"] = x_val
            pair_train_and_test_array_dict[infoID][action]["y_val"] = y_val
            pair_train_and_test_array_dict[infoID][action]["x_test"] = x_test
            pair_train_and_test_array_dict[infoID][action]["y_test"] = y_test

            pair_train_and_test_array_dict[infoID][action]["x_val_sliding_window"] = x_val_sliding_window
            pair_train_and_test_array_dict[infoID][action]["y_val_sliding_window"] = y_val_sliding_window

            pair_train_and_test_array_dict[infoID][action]["x_test_sliding_window"] = x_test_sliding_window
            pair_train_and_test_array_dict[infoID][action]["y_test_sliding_window"] = y_test_sliding_window

            i+=1
            print("Got pair array %d of %d" %(i, num_pairs))

    return pair_train_and_test_array_dict,flat_ft_to_idx_dict,flat_idx_to_ft_dict,flattened_fts

def get_pair_train_and_test_array_dict_with_certain_fts(pair_train_and_test_df_dict, infoIDs, desired_actions,INPUT_SIZE, EXTRACT_OUTPUT_SIZE, input_fts, target_fts):
    pair_train_and_test_array_dict = {}
    #get num combos
    num_pairs = len(desired_actions) * len(infoIDs)
    print("\n%d pairs"%num_pairs)
    i = 0
    for infoID in infoIDs:
        pair_train_and_test_array_dict[infoID] = {}
        for action in desired_actions:
            pair_train_and_test_array_dict[infoID][action] = {}

            #get dfs
            train_df = pair_train_and_test_df_dict[infoID][action]["train_df"]
            val_df = pair_train_and_test_df_dict[infoID][action]["val_df"]
            test_df = pair_train_and_test_df_dict[infoID][action]["test_df"]

            x_train,y_train,ft_to_idx_dict =convert_df_to_sliding_window_form_diff_x_and_y_fts_no_print_with_idx_dict(train_df, INPUT_SIZE, EXTRACT_OUTPUT_SIZE, input_fts, target_fts)
            print("\nx and y train arrays")
            print(x_train.shape)
            print(y_train.shape)

            #agg y
            y_train = agg_y_array(y_train)

            #get val arrays
            x_val,y_val = convert_df_to_test_sliding_window_form_diff_x_and_y_fts_no_print(val_df, INPUT_SIZE, EXTRACT_OUTPUT_SIZE,input_fts, target_fts,MOD_NUM=1000)
            y_val = agg_y_array(y_val)
            print("\ny_val shape")
            print(y_val.shape)

            #sliding window train
            x_val_sliding_window,y_val_sliding_window,ft_to_idx_dict =convert_df_to_sliding_window_form_diff_x_and_y_fts_no_print_with_idx_dict(val_df, INPUT_SIZE, EXTRACT_OUTPUT_SIZE, input_fts, target_fts)
            y_val_sliding_window = agg_y_array(y_val_sliding_window)
            print("\nx and y val sliding arrays")
            print(x_val_sliding_window.shape)
            print(y_val_sliding_window.shape)

            #get test arrays
            x_test,y_test = convert_df_to_test_sliding_window_form_diff_x_and_y_fts_no_print(test_df, INPUT_SIZE, EXTRACT_OUTPUT_SIZE,input_fts, target_fts,MOD_NUM=1000)
            y_test = agg_y_array(y_test)
            print("\ny_test shape")
            print(y_test.shape)

            #sliding window test
            x_test_sliding_window,y_test_sliding_window,ft_to_idx_dict =convert_df_to_sliding_window_form_diff_x_and_y_fts_no_print_with_idx_dict(test_df, INPUT_SIZE, EXTRACT_OUTPUT_SIZE, input_fts, target_fts)
            y_test_sliding_window = agg_y_array(y_test_sliding_window)
            print("\nx and y test sliding arrays")
            print(x_test_sliding_window.shape)
            print(y_test_sliding_window.shape)



            #data info
            pair_train_and_test_array_dict[infoID][action]["x_train"] = x_train
            pair_train_and_test_array_dict[infoID][action]["y_train"] = y_train
            pair_train_and_test_array_dict[infoID][action]["x_val"] = x_val
            pair_train_and_test_array_dict[infoID][action]["y_val"] = y_val
            pair_train_and_test_array_dict[infoID][action]["x_test"] = x_test
            pair_train_and_test_array_dict[infoID][action]["y_test"] = y_test

            pair_train_and_test_array_dict[infoID][action]["x_val_sliding_window"] = x_val_sliding_window
            pair_train_and_test_array_dict[infoID][action]["y_val_sliding_window"] = y_val_sliding_window

            pair_train_and_test_array_dict[infoID][action]["x_test_sliding_window"] = x_test_sliding_window
            pair_train_and_test_array_dict[infoID][action]["y_test_sliding_window"] = y_test_sliding_window

            i+=1
            print("Got pair array %d of %d" %(i, num_pairs))

    print("\nx and y train sizes")
    print(x_train.shape)
    print(y_train.shape)

    print("\nx and y val sizes")
    print(x_val.shape)
    print(y_val.shape)

    print("\nx and y test sizes")
    print(x_test.shape)
    print(y_test.shape)

    print("\nx and y val sliding window sizes")
    print(x_val_sliding_window.shape)
    print(y_val_sliding_window.shape)

    print("\nx and y test sliding window sizes")
    print(x_test_sliding_window.shape)
    print(y_test_sliding_window.shape)

    return pair_train_and_test_array_dict,ft_to_idx_dict

def make_test_set_pred_and_eval_v5_with_timestep_dim_and_platform_history_option(output_dir,x_eval_tag,infoIDs, desired_actions,pair_train_and_test_array_dict,EXTRACT_OUTPUT_SIZE,ft_to_idx_dict,target_ft,RESCALE_TARGET,NUM_TARGET_LOG_NORMS,ZERO_OUT_NEG_RESULTS,RESCALE,scaler,y_scaler,model ,OUTPUT_SIZE,platform_history_pair_train_and_test_array_dict,platform_history_ft_to_idx_dict,USE_PLATFORM_HISTORY):

    create_output_dir(output_dir)
    #make pred dict
    #platform_history_pair_train_and_test_array_dict,platform_history_ft_to_idx_dict
    if USE_PLATFORM_HISTORY == False:
        baseline_pred_dict = get_m1_shifted_baseline_pred_dict_v3_x_array_option(infoIDs, desired_actions,platform_history_pair_train_and_test_array_dict,EXTRACT_OUTPUT_SIZE,platform_history_ft_to_idx_dict,target_ft,x_eval_tag )
    else:
        baseline_pred_dict = get_m1_shifted_baseline_pred_dict_v3_x_array_option(infoIDs, desired_actions,pair_train_and_test_array_dict,EXTRACT_OUTPUT_SIZE,ft_to_idx_dict,target_ft,x_eval_tag )

    num_pairs = len(desired_actions) * len(infoIDs)

    #predict!
    i = 0
    pair_pred_dict = {}
    for infoID in infoIDs:
        pair_pred_dict[infoID] = {}
        for action in desired_actions:
            x_array = pair_train_and_test_array_dict[infoID][action][x_eval_tag]
            #y_test = pair_train_and_test_array_dict[infoID][action]["y_test"]

            if RESCALE == True:
                x_array =normalize_data_with_scaler(x_array, scaler)

            # if NUM_TARGET_LOG_NORMS == True:
            #   y_test = np.log1p(y_test)

            y_pred = model.predict(x_array)

            if RESCALE_TARGET == True:
                old_shape = y_pred.shape
                # y_pred = y_pred.reshape((y_pred.shape[0], 1, 1))
                y_pred = y_pred.reshape((y_pred.shape[0], OUTPUT_SIZE, 1))
                y_pred =denormalize_single_array(y_pred, y_scaler)
                y_pred = y_pred.reshape(old_shape)

            for j in range(NUM_TARGET_LOG_NORMS):
                y_pred = np.expm1(y_pred)

            if ZERO_OUT_NEG_RESULTS == True:
                y_pred[y_pred<0]=0

            pair_pred_dict[infoID][action] = y_pred
            i+=1
            print("Got pair df %d of %d" %(i, num_pairs))

    # =================================== score results ===================================
    vam_pred_tag = "VAM"
    baseline_pred_tag = "Shifted-Baseline"
    with open(output_dir + "pair_pred_dict", 'wb') as handle:
        pickle.dump(pair_pred_dict, handle)

    with open(output_dir + "pair_train_and_test_array_dict", 'wb') as handle:
        pickle.dump(pair_train_and_test_array_dict, handle)

    with open(output_dir + "baseline_pred_dict", "wb")as handle:
        pickle.dump(baseline_pred_dict, handle)

    # ========================== result dfs ====================================================
    # vam_result_df = score_pred_results_v2(pair_pred_dict,pair_train_and_test_array_dict,desired_actions, infoIDs,output_dir,vam_pred_tag)
    if x_eval_tag == "x_test":
        y_eval_tag = "y_test"

    if x_eval_tag == "x_val":
        y_eval_tag = "y_val"

    if x_eval_tag == "x_test_sliding_window":
        y_eval_tag = "y_test_sliding_window"

    if x_eval_tag == "x_val_sliding_window":
        y_eval_tag = "y_val_sliding_window"

    cur_output_dir = output_dir + "With-Timestep-Info/"
    create_output_dir(cur_output_dir)

    vam_result_df =score_pred_results_v3_with_array_option_v4_timestep_dim(pair_pred_dict,pair_train_and_test_array_dict,desired_actions, infoIDs,cur_output_dir,vam_pred_tag,y_eval_tag)
    # vam_infoID_result_df = get_infoID_combined_result(vam_result_df,vam_pred_tag,output_dir)
    # vam_action_result_df = action_result_df(vam_result_df, output_dir,vam_pred_tag)

    # ========================== baseline dfs ====================================================
    #baseline_result_df = score_pred_results_v2(baseline_pred_dict,pair_train_and_test_array_dict,desired_actions, infoIDs,output_dir,baseline_pred_tag)
    baseline_result_df =score_pred_results_v3_with_array_option_v4_timestep_dim(baseline_pred_dict,pair_train_and_test_array_dict,desired_actions, infoIDs,cur_output_dir,baseline_pred_tag,y_eval_tag)
    # baseline_infoID_result_df = get_infoID_combined_result(baseline_result_df,baseline_pred_tag,output_dir)
    # baseline_action_result_df = action_result_df(baseline_result_df, output_dir,baseline_pred_tag)

    #=========================== with timestep info ===========================
    #comp info
    merge_cols = ["infoID", "action","timestep"]
    rename_cols = ["mape", "rmse"]
    vam_tag = "VAM"
    baseline_tag = "Shifted-Baseline"
    metric_of_interest="rmse"

    #get comp info
    comp_df,winner_tag = m1_make_comp_df(vam_result_df, baseline_result_df,merge_cols,rename_cols,vam_tag,baseline_tag,metric_of_interest)
    summary_df = m1_get_model_win_info(comp_df,winner_tag)

    output_fp =cur_output_dir + "NN-vs-Baseline-Comps.csv"
    comp_df.to_csv(output_fp, index=False)
    print(output_fp)

    output_fp =cur_output_dir + "Summary-Baseline-Comps.csv"
    summary_df.to_csv(output_fp, index=False)
    print(output_fp)

    #=========================== without timestep info ===========================
    cur_output_dir =output_dir + "Without-Timestep-Info/"
    create_output_dir(cur_output_dir)
    vam_result_df =score_pred_results_v3_with_array_option(pair_pred_dict,pair_train_and_test_array_dict,desired_actions, infoIDs,cur_output_dir,vam_pred_tag,y_eval_tag)
    baseline_result_df =score_pred_results_v3_with_array_option(baseline_pred_dict,pair_train_and_test_array_dict,desired_actions, infoIDs,cur_output_dir,baseline_pred_tag,y_eval_tag)
    merge_cols = ["infoID", "action"]

    #get comp info
    comp_df,winner_tag = m1_make_comp_df(vam_result_df, baseline_result_df,merge_cols,rename_cols,vam_tag,baseline_tag,metric_of_interest)
    summary_df = m1_get_model_win_info(comp_df,winner_tag)

    output_fp =cur_output_dir + "NN-vs-Baseline-Comps.csv"
    comp_df.to_csv(output_fp, index=False)
    print(output_fp)

    output_fp =cur_output_dir + "Summary-Baseline-Comps.csv"
    summary_df.to_csv(output_fp, index=False)
    print(output_fp)

def make_test_set_pred_and_eval_v4_XGBOOST_FEATURES_NO_PLATFORM_OPTION(flat_ft_to_idx_dict,flat_idx_to_ft_dict,flattened_fts,output_dir,x_eval_tag,infoIDs, desired_actions,pair_train_and_test_array_dict,EXTRACT_OUTPUT_SIZE,target_ft,RESCALE_TARGET,NUM_TARGET_LOG_NORMS,ZERO_OUT_NEG_RESULTS,RESCALE,scaler,y_scaler,model ,OUTPUT_SIZE,platform_history_pair_train_and_test_array_dict,platform_history_flat_ft_to_idx_dict,platform_history_flat_idx_to_ft_dict,platform_history_flattened_fts,USE_PLATFORM_HISTORY):

    create_output_dir(output_dir)
    #make pred dict

    if USE_PLATFORM_HISTORY == True:
        baseline_pred_dict = get_m1_shifted_baseline_pred_dict_v4_x_array_option(infoIDs,flat_ft_to_idx_dict,flat_idx_to_ft_dict,flattened_fts,  desired_actions,pair_train_and_test_array_dict,EXTRACT_OUTPUT_SIZE,None,target_ft,x_eval_tag )
    else:
        baseline_pred_dict = get_m1_shifted_baseline_pred_dict_v4_x_array_option(infoIDs,platform_history_flat_ft_to_idx_dict,platform_history_flat_idx_to_ft_dict,platform_history_flattened_fts,  desired_actions,platform_history_pair_train_and_test_array_dict,EXTRACT_OUTPUT_SIZE,None,target_ft,x_eval_tag )

    num_pairs = len(desired_actions) * len(infoIDs)

    #predict!
    i = 0
    pair_pred_dict = {}
    for infoID in infoIDs:
        pair_pred_dict[infoID] = {}
        for action in desired_actions:
            x_array = pair_train_and_test_array_dict[infoID][action][x_eval_tag]
            #y_test = pair_train_and_test_array_dict[infoID][action]["y_test"]

            if RESCALE == True:
                x_array =normalize_data_with_scaler(x_array, scaler)

            # if NUM_TARGET_LOG_NORMS == True:
            #   y_test = np.log1p(y_test)

            y_pred = model.predict(x_array)

            if RESCALE_TARGET == True:
                old_shape = y_pred.shape
                # y_pred = y_pred.reshape((y_pred.shape[0], 1, 1))
                y_pred = y_pred.reshape((y_pred.shape[0], OUTPUT_SIZE, 1))
                y_pred =denormalize_single_array(y_pred, y_scaler)
                y_pred = y_pred.reshape(old_shape)

            for j in range(NUM_TARGET_LOG_NORMS):
                y_pred = np.expm1(y_pred)

            if ZERO_OUT_NEG_RESULTS == True:
                y_pred[y_pred<0]=0

            pair_pred_dict[infoID][action] = y_pred
            i+=1
            print("Got pair df %d of %d" %(i, num_pairs))

    # =================================== score results ===================================
    vam_pred_tag = "VAM"
    baseline_pred_tag = "Shifted-Baseline"
    with open(output_dir + "pair_pred_dict", 'wb') as handle:
        pickle.dump(pair_pred_dict, handle)

    with open(output_dir + "pair_train_and_test_array_dict", 'wb') as handle:
        pickle.dump(pair_train_and_test_array_dict, handle)

    with open(output_dir + "baseline_pred_dict", "wb")as handle:
        pickle.dump(baseline_pred_dict, handle)
    # ========================== result dfs ====================================================
    # vam_result_df = score_pred_results_v2(pair_pred_dict,pair_train_and_test_array_dict,desired_actions, infoIDs,output_dir,vam_pred_tag)
    if x_eval_tag == "x_test":
        y_eval_tag = "y_test"

    if x_eval_tag == "x_val":
        y_eval_tag = "y_val"

    if x_eval_tag == "x_test_sliding_window":
        y_eval_tag = "y_test_sliding_window"

    if x_eval_tag == "x_val_sliding_window":
        y_eval_tag = "y_val_sliding_window"
    # vam_result_df =score_pred_results_v3_with_array_option(pair_pred_dict,pair_train_and_test_array_dict,desired_actions, infoIDs,output_dir,vam_pred_tag,y_eval_tag)
    # # vam_infoID_result_df = get_infoID_combined_result(vam_result_df,vam_pred_tag,output_dir)
    # # vam_action_result_df = action_result_df(vam_result_df, output_dir,vam_pred_tag)

    # # ========================== baseline dfs ====================================================
    # #baseline_result_df = score_pred_results_v2(baseline_pred_dict,pair_train_and_test_array_dict,desired_actions, infoIDs,output_dir,baseline_pred_tag)
    # baseline_result_df =score_pred_results_v3_with_array_option(baseline_pred_dict,pair_train_and_test_array_dict,desired_actions, infoIDs,output_dir,baseline_pred_tag,y_eval_tag)

    cur_output_dir = output_dir + "With-Timestep-Info/"
    create_output_dir(cur_output_dir)

    vam_result_df =score_pred_results_v3_with_array_option_v4_timestep_dim(pair_pred_dict,pair_train_and_test_array_dict,desired_actions, infoIDs,cur_output_dir,vam_pred_tag,y_eval_tag)
    # vam_infoID_result_df = get_infoID_combined_result(vam_result_df,vam_pred_tag,output_dir)
    # vam_action_result_df = action_result_df(vam_result_df, output_dir,vam_pred_tag)

    # ========================== baseline dfs ====================================================
    #baseline_result_df = score_pred_results_v2(baseline_pred_dict,pair_train_and_test_array_dict,desired_actions, infoIDs,output_dir,baseline_pred_tag)
    baseline_result_df =score_pred_results_v3_with_array_option_v4_timestep_dim(baseline_pred_dict,pair_train_and_test_array_dict,desired_actions, infoIDs,cur_output_dir,baseline_pred_tag,y_eval_tag)

     #=========================== with timestep info ===========================
    #comp info
    merge_cols = ["infoID", "action","timestep"]
    rename_cols = ["mape", "rmse"]
    vam_tag = "VAM"
    baseline_tag = "Shifted-Baseline"
    metric_of_interest="rmse"

    #get comp info
    comp_df,winner_tag = m1_make_comp_df(vam_result_df, baseline_result_df,merge_cols,rename_cols,vam_tag,baseline_tag,metric_of_interest)
    summary_df = m1_get_model_win_info(comp_df,winner_tag)

    output_fp =cur_output_dir + "NN-vs-Baseline-Comps.csv"
    comp_df.to_csv(output_fp, index=False)
    print(output_fp)

    output_fp =cur_output_dir + "Summary-Baseline-Comps.csv"
    summary_df.to_csv(output_fp, index=False)
    print(output_fp)

    #=========================== without timestep info ===========================
    cur_output_dir =output_dir + "Without-Timestep-Info/"
    create_output_dir(cur_output_dir)
    vam_result_df =score_pred_results_v3_with_array_option(pair_pred_dict,pair_train_and_test_array_dict,desired_actions, infoIDs,cur_output_dir,vam_pred_tag,y_eval_tag)
    baseline_result_df =score_pred_results_v3_with_array_option(baseline_pred_dict,pair_train_and_test_array_dict,desired_actions, infoIDs,cur_output_dir,baseline_pred_tag,y_eval_tag)
    merge_cols = ["infoID", "action"]

    #get comp info
    comp_df,winner_tag = m1_make_comp_df(vam_result_df, baseline_result_df,merge_cols,rename_cols,vam_tag,baseline_tag,metric_of_interest)
    summary_df = m1_get_model_win_info(comp_df,winner_tag)

    output_fp =cur_output_dir + "NN-vs-Baseline-Comps.csv"
    comp_df.to_csv(output_fp, index=False)
    print(output_fp)

    output_fp =cur_output_dir + "Summary-Baseline-Comps.csv"
    summary_df.to_csv(output_fp, index=False)
    print(output_fp)

def make_test_set_pred_and_eval_v4_XGBOOST_FEATURES_INPUT_LOGNORM(NUM_INPUT_LOG_NORMS,flat_ft_to_idx_dict,flat_idx_to_ft_dict,flattened_fts,output_dir,x_eval_tag,infoIDs, desired_actions,pair_train_and_test_array_dict,EXTRACT_OUTPUT_SIZE,ft_to_idx_dict,target_ft,RESCALE_TARGET,NUM_TARGET_LOG_NORMS,ZERO_OUT_NEG_RESULTS,RESCALE,scaler,y_scaler,model ,OUTPUT_SIZE):

    create_output_dir(output_dir)
    #make pred dict
    baseline_pred_dict = get_m1_shifted_baseline_pred_dict_v4_x_array_option(infoIDs,flat_ft_to_idx_dict,flat_idx_to_ft_dict,flattened_fts,  desired_actions,pair_train_and_test_array_dict,EXTRACT_OUTPUT_SIZE,ft_to_idx_dict,target_ft,x_eval_tag )

    num_pairs = len(desired_actions) * len(infoIDs)

    #predict!
    i = 0
    pair_pred_dict = {}
    for infoID in infoIDs:
        pair_pred_dict[infoID] = {}
        for action in desired_actions:
            x_array = pair_train_and_test_array_dict[infoID][action][x_eval_tag]
            #y_test = pair_train_and_test_array_dict[infoID][action]["y_test"]

            for j in range(NUM_INPUT_LOG_NORMS):
                x_array = np.log1p(x_array)

            if RESCALE == True:
                x_array =normalize_data_with_scaler(x_array, scaler)



            print("\n%s"%x_eval_tag)
            print(x_array)

            # if NUM_TARGET_LOG_NORMS == True:
            #   y_test = np.log1p(y_test)

            y_pred = model.predict(x_array)

            if RESCALE_TARGET == True:
                old_shape = y_pred.shape
                # y_pred = y_pred.reshape((y_pred.shape[0], 1, 1))
                y_pred = y_pred.reshape((y_pred.shape[0], OUTPUT_SIZE, 1))
                y_pred =denormalize_single_array(y_pred, y_scaler)
                y_pred = y_pred.reshape(old_shape)

            for j in range(NUM_TARGET_LOG_NORMS):
                y_pred = np.expm1(y_pred)

            if ZERO_OUT_NEG_RESULTS == True:
                y_pred[y_pred<0]=0

            pair_pred_dict[infoID][action] = y_pred
            i+=1
            print("Got pair df %d of %d" %(i, num_pairs))

    # =================================== score results ===================================
    vam_pred_tag = "VAM"
    baseline_pred_tag = "Shifted-Baseline"
    with open(output_dir + "pair_pred_dict", 'wb') as handle:
        pickle.dump(pair_pred_dict, handle)

    with open(output_dir + "pair_train_and_test_array_dict", 'wb') as handle:
        pickle.dump(pair_train_and_test_array_dict, handle)

    with open(output_dir + "baseline_pred_dict", "wb")as handle:
        pickle.dump(baseline_pred_dict, handle)
    # ========================== result dfs ====================================================
    # vam_result_df = score_pred_results_v2(pair_pred_dict,pair_train_and_test_array_dict,desired_actions, infoIDs,output_dir,vam_pred_tag)
    if x_eval_tag == "x_test":
        y_eval_tag = "y_test"

    if x_eval_tag == "x_val":
        y_eval_tag = "y_val"

    if x_eval_tag == "x_test_sliding_window":
        y_eval_tag = "y_test_sliding_window"

    if x_eval_tag == "x_val_sliding_window":
        y_eval_tag = "y_val_sliding_window"
    vam_result_df =score_pred_results_v3_with_array_option(pair_pred_dict,pair_train_and_test_array_dict,desired_actions, infoIDs,output_dir,vam_pred_tag,y_eval_tag)
    # vam_infoID_result_df = get_infoID_combined_result(vam_result_df,vam_pred_tag,output_dir)
    # vam_action_result_df = action_result_df(vam_result_df, output_dir,vam_pred_tag)

    # ========================== baseline dfs ====================================================
    #baseline_result_df = score_pred_results_v2(baseline_pred_dict,pair_train_and_test_array_dict,desired_actions, infoIDs,output_dir,baseline_pred_tag)
    baseline_result_df =score_pred_results_v3_with_array_option(baseline_pred_dict,pair_train_and_test_array_dict,desired_actions, infoIDs,output_dir,baseline_pred_tag,y_eval_tag)
    # baseline_infoID_result_df = get_infoID_combined_result(baseline_result_df,baseline_pred_tag,output_dir)
    # baseline_action_result_df = action_result_df(baseline_result_df, output_dir,baseline_pred_tag)

    #comp info
    merge_cols = ["infoID", "action"]
    rename_cols = ["mape", "rmse"]
    vam_tag = "VAM"
    baseline_tag = "Shifted-Baseline"
    metric_of_interest="rmse"

    #get comp info
    comp_df,winner_tag = m1_make_comp_df(vam_result_df, baseline_result_df,merge_cols,rename_cols,vam_tag,baseline_tag,metric_of_interest)
    summary_df = m1_get_model_win_info(comp_df,winner_tag)

    output_fp = output_dir + "NN-vs-Baseline-Comps.csv"
    comp_df.to_csv(output_fp, index=False)
    print(output_fp)

    output_fp = output_dir + "Summary-Baseline-Comps.csv"
    summary_df.to_csv(output_fp, index=False)
    print(output_fp)

def make_test_set_pred_and_eval_PROPER_LSTM(output_dir,x_eval_tag,infoIDs, desired_actions,pair_train_and_test_array_dict,EXTRACT_OUTPUT_SIZE,ft_to_idx_dict,target_ft,RESCALE_TARGET,NUM_TARGET_LOG_NORMS,ZERO_OUT_NEG_RESULTS,RESCALE,scaler,y_scaler,model ,OUTPUT_SIZE,NUM_INPUT_LOG_NORMS):

    create_output_dir(output_dir)
    #make pred dict
    baseline_pred_dict = get_m1_shifted_baseline_pred_dict_v3_x_array_option(infoIDs, desired_actions,pair_train_and_test_array_dict,EXTRACT_OUTPUT_SIZE,ft_to_idx_dict,target_ft,x_eval_tag )

    num_pairs = len(desired_actions) * len(infoIDs)

    x_eval_tag_to_static_tag_dict = {
    "dynamic_x_test" : "static_x_test",
    "dynamic_x_val" : "static_x_val",
    "dynamic_x_test_sliding_window" : "static_x_test_sliding_window",
    "dynamic_x_val_sliding_window" : "static_x_val_sliding_window"
    }

    x_static_tag = x_eval_tag_to_static_tag_dict[x_eval_tag]


    #predict!
    i = 0
    pair_pred_dict = {}
    for infoID in infoIDs:
        pair_pred_dict[infoID] = {}
        for action in desired_actions:
            dynamic_x_array = pair_train_and_test_array_dict[infoID][action][x_eval_tag]
            static_x_array = pair_train_and_test_array_dict[infoID][action][x_static_tag]

            if RESCALE == True:
                dynamic_x_array =normalize_data_with_scaler(dynamic_x_array, scaler)

            x_array_list = [dynamic_x_array,static_x_array]


            # if NUM_TARGET_LOG_NORMS == True:
            #   y_test = np.log1p(y_test)

            y_pred = model.predict(x_array_list)

            if RESCALE_TARGET == True:
                old_shape = y_pred.shape
                # y_pred = y_pred.reshape((y_pred.shape[0], 1, 1))
                y_pred = y_pred.reshape((y_pred.shape[0], OUTPUT_SIZE, 1))
                y_pred =denormalize_single_array(y_pred, y_scaler)
                y_pred = y_pred.reshape(old_shape)

            for j in range(NUM_TARGET_LOG_NORMS):
                y_pred = np.expm1(y_pred)

            if ZERO_OUT_NEG_RESULTS == True:
                y_pred[y_pred<0]=0

            pair_pred_dict[infoID][action] = y_pred
            i+=1
            print("Got pair df %d of %d" %(i, num_pairs))

    # =================================== score results ===================================
    vam_pred_tag = "VAM"
    baseline_pred_tag = "Shifted-Baseline"
    with open(output_dir + "pair_pred_dict", 'wb') as handle:
        pickle.dump(pair_pred_dict, handle)

    with open(output_dir + "pair_train_and_test_array_dict", 'wb') as handle:
        pickle.dump(pair_train_and_test_array_dict, handle)

    # ========================== result dfs ====================================================
    # vam_result_df = score_pred_results_v2(pair_pred_dict,pair_train_and_test_array_dict,desired_actions, infoIDs,output_dir,vam_pred_tag)
    if x_eval_tag == "dynamic_x_test":
        y_eval_tag = "y_test"

    if x_eval_tag == "dynamic_x_val":
        y_eval_tag = "y_val"

    if x_eval_tag == "dynamic_x_test_sliding_window":
        y_eval_tag = "y_test_sliding_window"

    if x_eval_tag == "dynamic_x_val_sliding_window":
        y_eval_tag = "y_val_sliding_window"
    vam_result_df =score_pred_results_v3_with_array_option(pair_pred_dict,pair_train_and_test_array_dict,desired_actions, infoIDs,output_dir,vam_pred_tag,y_eval_tag)
    # vam_infoID_result_df = get_infoID_combined_result(vam_result_df,vam_pred_tag,output_dir)
    # vam_action_result_df = action_result_df(vam_result_df, output_dir,vam_pred_tag)

    # ========================== baseline dfs ====================================================
    #baseline_result_df = score_pred_results_v2(baseline_pred_dict,pair_train_and_test_array_dict,desired_actions, infoIDs,output_dir,baseline_pred_tag)
    baseline_result_df =score_pred_results_v3_with_array_option(baseline_pred_dict,pair_train_and_test_array_dict,desired_actions, infoIDs,output_dir,baseline_pred_tag,y_eval_tag)
    # baseline_infoID_result_df = get_infoID_combined_result(baseline_result_df,baseline_pred_tag,output_dir)
    # baseline_action_result_df = action_result_df(baseline_result_df, output_dir,baseline_pred_tag)

    #comp info
    merge_cols = ["infoID", "action"]
    rename_cols = ["mape", "rmse"]
    vam_tag = "VAM"
    baseline_tag = "Shifted-Baseline"
    metric_of_interest="rmse"

    #get comp info
    comp_df,winner_tag = m1_make_comp_df(vam_result_df, baseline_result_df,merge_cols,rename_cols,vam_tag,baseline_tag,metric_of_interest)
    summary_df = m1_get_model_win_info(comp_df,winner_tag)

    output_fp = output_dir + "NN-vs-Baseline-Comps.csv"
    comp_df.to_csv(output_fp, index=False)
    print(output_fp)

    output_fp = output_dir + "Summary-Baseline-Comps.csv"
    summary_df.to_csv(output_fp, index=False)
    print(output_fp)

def make_test_set_pred_and_eval_v5_XGBOOST_SAVE_DICT_OPTION(flat_ft_to_idx_dict,flat_idx_to_ft_dict,flattened_fts,output_dir,x_eval_tag,infoIDs, desired_actions,pair_train_and_test_array_dict,EXTRACT_OUTPUT_SIZE,target_ft,RESCALE_TARGET,NUM_TARGET_LOG_NORMS,ZERO_OUT_NEG_RESULTS,RESCALE,scaler,y_scaler,model ,OUTPUT_SIZE,platform_history_pair_train_and_test_array_dict,platform_history_flat_ft_to_idx_dict,platform_history_flat_idx_to_ft_dict,platform_history_flattened_fts,USE_PLATFORM_HISTORY,SAVE_DICTS=False):

    create_output_dir(output_dir)
    #make pred dict

    if USE_PLATFORM_HISTORY == True:
        baseline_pred_dict = get_m1_shifted_baseline_pred_dict_v4_x_array_option(infoIDs,flat_ft_to_idx_dict,flat_idx_to_ft_dict,flattened_fts,  desired_actions,pair_train_and_test_array_dict,EXTRACT_OUTPUT_SIZE,None,target_ft,x_eval_tag )
    else:
        baseline_pred_dict = get_m1_shifted_baseline_pred_dict_v4_x_array_option(infoIDs,platform_history_flat_ft_to_idx_dict,platform_history_flat_idx_to_ft_dict,platform_history_flattened_fts,  desired_actions,platform_history_pair_train_and_test_array_dict,EXTRACT_OUTPUT_SIZE,None,target_ft,x_eval_tag )

    num_pairs = len(desired_actions) * len(infoIDs)

    #predict!
    i = 0
    pair_pred_dict = {}
    for infoID in infoIDs:
        pair_pred_dict[infoID] = {}
        for action in desired_actions:
            x_array = pair_train_and_test_array_dict[infoID][action][x_eval_tag]
            #y_test = pair_train_and_test_array_dict[infoID][action]["y_test"]

            if RESCALE == True:
                x_array =normalize_data_with_scaler(x_array, scaler)

            # if NUM_TARGET_LOG_NORMS == True:
            #   y_test = np.log1p(y_test)

            y_pred = model.predict(x_array)

            if RESCALE_TARGET == True:
                old_shape = y_pred.shape
                # y_pred = y_pred.reshape((y_pred.shape[0], 1, 1))
                y_pred = y_pred.reshape((y_pred.shape[0], OUTPUT_SIZE, 1))
                y_pred =denormalize_single_array(y_pred, y_scaler)
                y_pred = y_pred.reshape(old_shape)

            for j in range(NUM_TARGET_LOG_NORMS):
                y_pred = np.expm1(y_pred)

            if ZERO_OUT_NEG_RESULTS == True:
                y_pred[y_pred<0]=0

            pair_pred_dict[infoID][action] = y_pred
            i+=1
            print("Got pair df %d of %d" %(i, num_pairs))

    # =================================== score results ===================================
    vam_pred_tag = "VAM"
    baseline_pred_tag = "Shifted-Baseline"

    if SAVE_DICTS==True:
        with open(output_dir + "pair_pred_dict", 'wb') as handle:
            pickle.dump(pair_pred_dict, handle)

        with open(output_dir + "pair_train_and_test_array_dict", 'wb') as handle:
            pickle.dump(pair_train_and_test_array_dict, handle)

        with open(output_dir + "baseline_pred_dict", "wb")as handle:
            pickle.dump(baseline_pred_dict, handle)
    # ========================== result dfs ====================================================
    # vam_result_df = score_pred_results_v2(pair_pred_dict,pair_train_and_test_array_dict,desired_actions, infoIDs,output_dir,vam_pred_tag)
    if x_eval_tag == "x_test":
        y_eval_tag = "y_test"

    if x_eval_tag == "x_val":
        y_eval_tag = "y_val"

    if x_eval_tag == "x_test_sliding_window":
        y_eval_tag = "y_test_sliding_window"

    if x_eval_tag == "x_val_sliding_window":
        y_eval_tag = "y_val_sliding_window"
    # vam_result_df =score_pred_results_v3_with_array_option(pair_pred_dict,pair_train_and_test_array_dict,desired_actions, infoIDs,output_dir,vam_pred_tag,y_eval_tag)
    # # vam_infoID_result_df = get_infoID_combined_result(vam_result_df,vam_pred_tag,output_dir)
    # # vam_action_result_df = action_result_df(vam_result_df, output_dir,vam_pred_tag)

    # # ========================== baseline dfs ====================================================
    # #baseline_result_df = score_pred_results_v2(baseline_pred_dict,pair_train_and_test_array_dict,desired_actions, infoIDs,output_dir,baseline_pred_tag)
    # baseline_result_df =score_pred_results_v3_with_array_option(baseline_pred_dict,pair_train_and_test_array_dict,desired_actions, infoIDs,output_dir,baseline_pred_tag,y_eval_tag)

    cur_output_dir = output_dir + "With-Timestep-Info/"
    create_output_dir(cur_output_dir)

    vam_result_df =score_pred_results_v3_with_array_option_v4_timestep_dim(pair_pred_dict,pair_train_and_test_array_dict,desired_actions, infoIDs,cur_output_dir,vam_pred_tag,y_eval_tag)
    # vam_infoID_result_df = get_infoID_combined_result(vam_result_df,vam_pred_tag,output_dir)
    # vam_action_result_df = action_result_df(vam_result_df, output_dir,vam_pred_tag)

    # ========================== baseline dfs ====================================================
    #baseline_result_df = score_pred_results_v2(baseline_pred_dict,pair_train_and_test_array_dict,desired_actions, infoIDs,output_dir,baseline_pred_tag)
    baseline_result_df =score_pred_results_v3_with_array_option_v4_timestep_dim(baseline_pred_dict,pair_train_and_test_array_dict,desired_actions, infoIDs,cur_output_dir,baseline_pred_tag,y_eval_tag)

     #=========================== with timestep info ===========================
    #comp info
    merge_cols = ["infoID", "action","timestep"]
    rename_cols = ["mape", "rmse"]
    vam_tag = "VAM"
    baseline_tag = "Shifted-Baseline"
    metric_of_interest="rmse"

    #get comp info
    comp_df,winner_tag = m1_make_comp_df(vam_result_df, baseline_result_df,merge_cols,rename_cols,vam_tag,baseline_tag,metric_of_interest)
    summary_df = m1_get_model_win_info(comp_df,winner_tag)

    output_fp =cur_output_dir + "NN-vs-Baseline-Comps.csv"
    comp_df.to_csv(output_fp, index=False)
    print(output_fp)

    output_fp =cur_output_dir + "Summary-Baseline-Comps.csv"
    summary_df.to_csv(output_fp, index=False)
    print(output_fp)

    #=========================== without timestep info ===========================
    cur_output_dir =output_dir + "Without-Timestep-Info/"
    create_output_dir(cur_output_dir)
    vam_result_df =score_pred_results_v3_with_array_option(pair_pred_dict,pair_train_and_test_array_dict,desired_actions, infoIDs,cur_output_dir,vam_pred_tag,y_eval_tag)
    baseline_result_df =score_pred_results_v3_with_array_option(baseline_pred_dict,pair_train_and_test_array_dict,desired_actions, infoIDs,cur_output_dir,baseline_pred_tag,y_eval_tag)
    merge_cols = ["infoID", "action"]

    #get comp info
    comp_df,winner_tag = m1_make_comp_df(vam_result_df, baseline_result_df,merge_cols,rename_cols,vam_tag,baseline_tag,metric_of_interest)
    summary_df = m1_get_model_win_info(comp_df,winner_tag)

    output_fp =cur_output_dir + "NN-vs-Baseline-Comps.csv"
    comp_df.to_csv(output_fp, index=False)
    print(output_fp)

    output_fp =cur_output_dir + "Summary-Baseline-Comps.csv"
    summary_df.to_csv(output_fp, index=False)
    print(output_fp)