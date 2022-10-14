import pandas as pd
import json
import glob
import numpy as np
import sys,os
import pickle
import ast
from functools import *
from keras.layers import Bidirectional
from scipy.stats import norm
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import datetime
from datetime import datetime
from datetime import timedelta, date, datetime

import gc
gc.collect()

from scipy.stats.stats import pearsonr
# from PND_DF_funcs_for_nn_exp import *

from keras.optimizers import SGD,Adam,RMSprop
from sklearn.preprocessing import MinMaxScaler,StandardScaler
# from sklearn.externals import joblib
from ast import literal_eval

from keras.models import Sequential
from keras import backend as K
from keras import optimizers
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, MaxPooling2D
from keras.layers import Dense, LSTM, TimeDistributed, Input, Dropout,Reshape,Flatten,Concatenate,concatenate
from keras.models import Sequential
import tensorflow as tf
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
import json

from keras.models import Model
from keras.layers import Input

from dateutil.relativedelta import relativedelta
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping

def get_and_preprocess_train_val_and_test_dfs_v3_fixed_nodeTime_deltas(df, y_test_start, y_test_end, rain_start,train_end,val_start,val_end,test_start,test_end, EMBED_COLS, TAG_TOKENS, NUM_LOG_NORMS,output_order,INPUT_SIZE,OUTPUT_SIZE,VAL_PERCENTAGE, INITIAL_DATE,time_gran="seconds"):

    time_gran_list = ["seconds", "minutes", "hours", "days"]

    #make a dict with test info
    TEST_INFO_DICT = {}


    #first of all make sure we're sorted
    df["nodeTime"] = pd.to_datetime(df["nodeTime"], utc=True)
    df = df.sort_values("nodeTime").reset_index(drop=True)
    df =config_df_by_dates(df,INITIAL_DATE, y_test_end)

    #change the nodetime stuff here
    if "nodeTime" in output_order:
        df,INITIAL_DATE_FOR_TIME_DELTA_MANIPULATION,ORIGINAL_DATETIME_SERIES = convert_datetimes_to_timedeltas(df,time_gran)

        print("\n\ndf after manipulation: ")
        print(df)
        df["original_nodeTime"] = ORIGINAL_DATETIME_SERIES


        verify_timedeltas(df, INITIAL_DATE_FOR_TIME_DELTA_MANIPULATION, time_gran)
        print("\n\ndf after time delta verification: ")
        print(df)

    # sys.exit(0)


    #first let's get the test df
    y_test_df = df.copy()
    y_test_df =config_df_by_dates(y_test_df,y_test_start, y_test_end,time_col="original_nodeTime")
    print(y_test_df)

    #get size
    ytdf_size = y_test_df.shape[0]
    print("Y test df is of size: %d" %ytdf_size)

    #figure out how the right io size for our test data
    SEQUENCE_SIZE = INPUT_SIZE + OUTPUT_SIZE
    NUM_FULL_SEQUENCES = int(ytdf_size/SEQUENCE_SIZE)
    REMAINING_TIMESTEPS = ytdf_size%SEQUENCE_SIZE
    TIMESTEPS_NEEDED_TO_CAPTURE_ALL_DATES = SEQUENCE_SIZE - REMAINING_TIMESTEPS
    TIMESTEPS_NEEDED_TO_HAVE_AN_INIT_CONDIT = INPUT_SIZE
    TOTAL_EXTRA_ROWS_NEEDED_FROM_DF = TIMESTEPS_NEEDED_TO_HAVE_AN_INIT_CONDIT + TIMESTEPS_NEEDED_TO_CAPTURE_ALL_DATES
    TOTAL_TAIL_SIZE_NEEDED = ytdf_size + TOTAL_EXTRA_ROWS_NEEDED_FROM_DF
    print("\nTest seq info")
    print("\ntest df has this many rows aka time steps: %d" %ytdf_size)
    print("IO size: %d to %d" %(INPUT_SIZE, OUTPUT_SIZE))
    print("Seq size: %d" %SEQUENCE_SIZE)
    print("Num full sequences %d"%NUM_FULL_SEQUENCES)
    print("REMAINING_TIMESTEPS: %d" %REMAINING_TIMESTEPS)
    print("We need this many rows added on from the past: %d"%TIMESTEPS_NEEDED_TO_CAPTURE_ALL_DATES)
    print("TIMESTEPS_NEEDED_TO_HAVE_AN_INIT_CONDIT: %d"%TIMESTEPS_NEEDED_TO_HAVE_AN_INIT_CONDIT)
    print("TOTAL_EXTRA_ROWS_NEEDED_FROM_DF: %d"%TOTAL_EXTRA_ROWS_NEEDED_FROM_DF)
    print("TOTAL_TAIL_SIZE_NEEDED: %d" %TOTAL_TAIL_SIZE_NEEDED)
    # sys.exit(0)


    #get test tail
    test_with_init_condit_tail = df.copy()
    test_with_init_condit_tail = test_with_init_condit_tail.tail(TOTAL_TAIL_SIZE_NEEDED).reset_index(drop=True)
    test_df = test_with_init_condit_tail.copy()
    print(test_df)
    # sys.exit(0)

    #GET init condit head
    init_condit_head = test_with_init_condit_tail.head(INPUT_SIZE).copy()
    init_condit_start = init_condit_head["nodeTime"].iloc[0]
    init_condit_end = init_condit_head["nodeTime"].iloc[-1]
    TEST_INFO_DICT["init_condit_start"] = init_condit_start
    TEST_INFO_DICT["init_condit_end"] = init_condit_end

    dict_keys = ["init_condit_start", "init_condit_end"]





       #make dfs
    leftover_df = df.copy()
    leftover_df =leftover_df.head(leftover_df.shape[0] - TOTAL_TAIL_SIZE_NEEDED)

    #get split sizes
    leftover_size = leftover_df.shape[0]
    print("Num leftover_size rows: %d" %leftover_size)

    val_size = int(leftover_size * VAL_PERCENTAGE)
    print("Num val rows: %d" %val_size)

    train_size = leftover_size - val_size
    print("Num train rows: %d" %train_size)

    #get train and val dfs
    train_df = leftover_df.head(train_size).copy()
    val_df = leftover_df.tail(val_size).copy()



    # val_df = df.copy()
    # val_df =config_df_by_dates(val_df,val_start,val_end)

    # test_df = df.copy()
    # test_df =config_df_by_dates(test_df,test_start,test_end)

    # print(train_df[NONEMBED_COLS])
    # print(test_df[NONEMBED_COLS])
    # sys.exit(0)

    # train_df = train_df[output_order]
    print(train_df)
    print(test_df)
    # sys.exit(0)

    if "nodeTime" in output_order:
    #     #convert nodeTime to offset
    #     train_df = convert_nodeTime_to_offset(train_df, INITIAL_DATE,time_gran=time_gran)
    #     val_df = convert_nodeTime_to_offset(val_df, INITIAL_DATE,time_gran=time_gran)
    #     test_df = convert_nodeTime_to_offset(test_df, INITIAL_DATE,time_gran=time_gran)

        for i in range(NUM_LOG_NORMS):
            train_df["nodeTime"] = np.log1p(train_df["nodeTime"])
            val_df["nodeTime"] = np.log1p(val_df["nodeTime"])
            test_df["nodeTime"] = np.log1p(test_df["nodeTime"])

    #     print(train_df["nodeTime"])
    #     print(val_df["nodeTime"])
    #     print(test_df["nodeTime"])
    #     # sys.exit(0)

    if TAG_TOKENS == True:
        for col in EMBED_COLS:
            train_df[col] = "<" + col + ">" + train_df[col].astype(str)
            val_df[col] = "<" + col + ">" + val_df[col].astype(str)
            test_df[col] = "<" + col + ">" + test_df[col].astype(str)
    # print(train_df[output_order])

    #convert to strings
    for col in EMBED_COLS:
        train_df[col] = train_df[col].astype(str)
        val_df[col] = val_df[col].astype(str)
        test_df[col] = test_df[col].astype(str)

    print(train_df)
    print(val_df)
    print(test_df)

    #get val and test initial dates
    TRAIN_INITIAL_DATE_FOR_TIME_DELTA_MANIPULATION = INITIAL_DATE_FOR_TIME_DELTA_MANIPULATION
    VAL_INITIAL_DATE_FOR_TIME_DELTA_MANIPULATION = val_df["original_nodeTime"].iloc[INPUT_SIZE-1]
    TEST_INITIAL_DATE_FOR_TIME_DELTA_MANIPULATION = test_df["original_nodeTime"].iloc[INPUT_SIZE-1]


    return train_df, val_df, test_df,ORIGINAL_DATETIME_SERIES,TRAIN_INITIAL_DATE_FOR_TIME_DELTA_MANIPULATION,VAL_INITIAL_DATE_FOR_TIME_DELTA_MANIPULATION,TEST_INITIAL_DATE_FOR_TIME_DELTA_MANIPULATION


def config_df_by_daily_dates(df,start_date,end_date,time_col="nodeTime"):
    df[time_col]=pd.to_datetime(df[time_col], utc=True)
    df = df.sort_values(time_col)
    df["temp_dates"] = df[time_col].dt.floor("D")
    df=df.set_index("temp_dates")
    df = df[(df.index >= start_date) & (df.index <= end_date)]
    df=df.reset_index(drop=True)
    return df

# def get_data_arrays(df, desired_cols,INPUT_SIZE,OUTPUT_SIZE, overall_start, overall_end, y_test_start, y_test_end,GRAN, GRAN_UNIT,NUM_OF_GRAN_UNIT,TIME_FEATURES_TO_GET):
#     df = df[desired_cols]
#     df = df.reset_index(drop=True)

#     df["nodeTime"] = pd.to_datetime(df["nodeTime"], utc=True)
#     df = df.sort_values("nodeTime")
#     print(df)
#     # sys.exit(0)





#     #gran
#     # GRAN_UNIT = "D"

#     def aggregate_cols_by_gran(df, overall_start, overall_end, GRAN):

#         df = config_df_by_dates(df,overall_start,overall_end,time_col="nodeTime")
#         df["nodeTime"] = df["nodeTime"].dt.floor(GRAN)

#         new_cols = ["nodeTime"]
#         actions = list(df)
#         actions.remove("nodeTime")
#         for action in actions:
#             new_col = "num_%ss"%action
#             new_cols.append(new_col)
#             df[new_col] = df.groupby(["nodeTime"])[action].transform("sum")

#         #get data df
#         df = df[new_cols].drop_duplicates().reset_index(drop=True)

#         if "nodeTime" in new_cols:
#             new_cols.remove("nodeTime")

#         return df,new_cols


#     #get basic daily df
#     df,reg_fts = aggregate_cols_by_gran(df, overall_start, overall_end, GRAN)
#     print(df)




#     #get ft list
#     fts = list(reg_fts + TIME_FEATURES_TO_GET)
#     if "nodeTime" not in fts:
#         fts = ["nodeTime"] + fts

#     #get time fts
#     df = insert_time_features_into_df_v2(df,  TIME_FEATURES_TO_GET, GRAN,timecol="nodeTime")
#     print(df)
#     # sys.exit(0)

#     # #split df into test and leftovers
#     test_df, leftovers_df = split_df_into_test_and_leftovers_v2(df,INPUT_SIZE,OUTPUT_SIZE, overall_start, overall_end, y_test_start, y_test_end,GRAN, GRAN_UNIT,NUM_OF_GRAN_UNIT,reg_fts)

#     print("\nleftovers_df")
#     print(leftovers_df)

#     print("\ntest_df")
#     print(test_df)

#     # sys.exit(0)

#     #get sliding window dfs
#     sliding_window_dfs = get_sliding_window_dfs_at_gran_unit_level(leftovers_df, INPUT_SIZE, OUTPUT_SIZE,GRAN, GRAN_UNIT,NUM_OF_GRAN_UNIT,reg_fts)

#     #convert dfs to arrays




#     x_test, y_test, test_sample_idx_to_nodeTime_dict = get_data_arrays_without_sliding_window(test_df, INPUT_SIZE, OUTPUT_SIZE, fts,"test")
#     x_train, y_train,train_sample_idx_to_nodeTime_dict = convert_df_list_to_arrays_with_time_fts(sliding_window_dfs, INPUT_SIZE, OUTPUT_SIZE, fts)


#     #now remove nodeTime fts
#     if "nodeTime" in fts:
#         fts.remove("nodeTime")

#     #get rid of time fts in y out
#     if INCLUDE_TIME_FEATURES == False:
#         x_train= trim_time_fts_from_y_array(x_train, reg_fts)
#         x_test= trim_time_fts_from_y_array(x_test, reg_fts)

#     y_train= trim_time_fts_from_y_array(y_train, reg_fts)
#     y_test= trim_time_fts_from_y_array(y_test, reg_fts)

#     print("\nx_train shape")
#     print(x_train.shape)
#     print("y_train shape")
#     print(y_train.shape)

#     print("\nx_test shape")
#     print(x_test.shape)
#     print("y_test shape")
#     print(y_test.shape)

#     #convert to float
#     x_train = x_train.astype("float32")
#     y_train = y_train.astype("float32")
#     x_test = x_test.astype("float32")
#     y_test = y_test.astype("float32")

#     # sys.exit(0)
#     return x_test, y_test, test_sample_idx_to_nodeTime_dict,x_train, y_train,train_sample_idx_to_nodeTime_dict

def denormalize_train_and_test(x_train, x_test, scaler):
    old_train_shape=x_train.shape

    old_test_shape=x_test.shape

    x_train=x_train.reshape(x_train.shape[0]*x_train.shape[1],x_train.shape[2])

    x_test=x_test.reshape(x_test.shape[0]*x_test.shape[1],x_test.shape[2])


    x_train=scaler.inverse_transform(x_train)

    x_test=scaler.inverse_transform(x_test)

    x_train=x_train.reshape(old_train_shape)

    x_test=x_test.reshape(old_test_shape)

    return x_train, x_test

def save_as_json_v2(history_dict,OUTPUT_FP):
    print("Saving %s..."%OUTPUT_FP)
    for key, cur_list in history_dict.items():
        cur_list = list(cur_list)
        for i in range(len(cur_list)):
            # print(history_dict[key][i])
            history_dict[key][i] = float(history_dict[key][i])
    with open(OUTPUT_FP, "w") as f:
        # json.dump(str(history_dict), f)
        json.dump(history_dict, f)
    print("Saved %s"%OUTPUT_FP)

def adjust_df_by_end_date(df,end_date,time_col="nodeTime"):
    # df=pd.read_csv(INPUT_FP)
    preshape=str(df.shape)
    df=df.set_index(time_col)
    # df = df[(df.index >= start_date) & (df.index <= end_date)]
    df = df[(df.index <= end_date)]
    # df['created_at'] =pd.to_datetime(df['created_at']).tz_localize("utc")
    df=df.reset_index(drop=False)

    postshape=str(df.shape)

    print("Shape before date adjustment: %s, shape after: %s" %(preshape, postshape))
    return df


def verify_x_and_y_timesteps(X_ARRAY_TIMESTEPS_IN_A_SEQUENCE, Y_ARRAY_TIMESTEPS_IN_A_SEQUENCE):
    if Y_ARRAY_TIMESTEPS_IN_A_SEQUENCE > X_ARRAY_TIMESTEPS_IN_A_SEQUENCE:
        print("Error! Y_ARRAY_TIMESTEPS_IN_A_SEQUENCE (%d) > X_ARRAY_TIMESTEPS_IN_A_SEQUENCE (%d). Terminating.")%(X_ARRAY_TIMESTEPS_IN_A_SEQUENCE,Y_ARRAY_TIMESTEPS_IN_A_SEQUENCE)
        sys.exit(0)
    if X_ARRAY_TIMESTEPS_IN_A_SEQUENCE%Y_ARRAY_TIMESTEPS_IN_A_SEQUENCE !=0:
        print("Error! Y_ARRAY_TIMESTEPS_IN_A_SEQUENCE (%d) is not a factor of X_ARRAY_TIMESTEPS_IN_A_SEQUENCE (%d). Terminating.")%(Y_ARRAY_TIMESTEPS_IN_A_SEQUENCE,X_ARRAY_TIMESTEPS_IN_A_SEQUENCE)
        sys.exit(0)

def get_best_coins():
    #best coins
    coin_names=[
    # "omni",
    "electroneum",
    # "indorse",
    "agrello",
    # "genesis vision",
    # "agoras_tokens",
    # "peercoin",
    # "tether",
    # "bitcoin_diamond"
    ]

    return coin_names

def get_twitter_gen_fts():

    twitter_gen_fts=[
    'num_tweet_mentions_of_<coin_name>_primary_feature_twitter',
    # 'num_retweet_mentions_of_<coin_name>_primary_feature_twitter',
    # 'num_quote_mentions_of_<coin_name>_primary_feature_twitter',
    # 'num_reply_mentions_of_<coin_name>_primary_feature_twitter',

    # 'num_new_users_<coin_name>_primary_feature_twitter',
    # 'num_old_users_<coin_name>_primary_feature_twitter',

    # 'avg_le_score_<coin_name>_primary_feature_twitter',
    # 'avg_tfidf_score_<coin_name>_primary_feature_twitter',
    # 'avg_sentiment_polarity_<coin_name>_primary_feature_twitter'
    ]

    return twitter_gen_fts

def get_telegram_gen_fts():
    telegram_gen_fts=[
    #telegram user fts
    # 'num_new_users_<coin_name>_primary_feature_telegram',
    # 'num_old_users_<coin_name>_primary_feature_telegram',

    #telegram comm fts
    'num_group_messages_<coin_name>_primary_feature_telegram',
    'num_channel_messages_<coin_name>_primary_feature_telegram',

    # #telegram user fts
    # 'num_new_users_<coin_name>_secondary_categories_total_telegram',
    # 'num_old_users_<coin_name>_secondary_categories_total_telegram',

    # #telegram comm fts
    # 'num_group_messages_<coin_name>_secondary_categories_total_telegram',
    # 'num_channel_messages_<coin_name>_secondary_categories_total_telegram'
    ]

    return telegram_gen_fts

def get_nazim_telegram_fts():
    nazim_telegram_fts=[
    'tfidf',
    '#common_tags',
    'post_count',
    'sentiment',
    'viewedby',
    'text_similarity',
    's_post',
    'channel_count',
    'r_post',
    '#hashtags',
    'log_entropy',
    ]

    return nazim_telegram_fts

def get_full_2_year_coins():
    # #full 2 year coins
    coin_names=[
    'tether',
    'bancor_network_token',
    'peercoin',
    'chaincoin',
    'bytecent',
    'bean_cash',
    'pesetacoin',
    'magi_coin',
    'paycoin',
    'agoras_tokens',
    'stealth',
    ]

    return coin_names

def get_october_start_coins():
    #coins starting from october
    coin_names=[
    'omni',
    'electroneum',
    'agrello',
    'genesis vision',
    'tether',
    'ubiq',
    'indorse',
    'bitcoin_diamond',
    'bancor_network_token',
    'peercoin',
    'chaincoin',
    'quantum_resistant_ledger',
    'blockmason_credit_protocol',
    'bytecent',
    'bean_cash',
    'pesetacoin',
    'magi_coin',
    'paycoin',
    'agoras_tokens',
    'stealth',
    ]

    return coin_names

def get_all_coins():

    coin_names=[

    #dry run 7
    "omni",
    "peercoin",
    "ecoin",
    "indorse",
    "chaincoin",
    "bancor_network_token",
    "genesis vision",

    #other 18
    "version",
    "chill_coin",
    "pesetacoin",
    "blockmason_credit_protocol",
    "agoras_tokens",
    "tether",
    "stealth",
    "quantum_resistant_ledger",
    "paycoin",
    "granitecoin",
    "bitcoin_diamond",
    "magi_coin",
    "bytecent",
    "vcash",
    "bean_cash",
    "electroneum",
    "agrello",
    "ubiq"
    ]

    return coin_names


def insert_dows_into_2d_array(my_2d_array, initial_dow):
    all_subarrays=[]
    cur_dow=initial_dow
    for i in range(my_2d_array.shape[0]):
        dow_list=[0 for j in range(7)]
        dow_list[cur_dow]=1
        dow_array=np.asarray(dow_list)
        cur_feature_subarray=np.concatenate([my_2d_array[i],dow_array])
        all_subarrays.append(cur_feature_subarray)
        cur_dow=(cur_dow+1)%7
    my_2d_array=np.asarray(all_subarrays)
    print("My 2d array shape: %s" %str(my_2d_array.shape))
    return my_2d_array


def config_df_by_dates(df,start_date,end_date,time_col="created_at"):
    df[time_col]=pd.to_datetime(df[time_col], utc=True)
    df = df.sort_values(time_col)
    df=df.set_index(time_col)
    df = df[(df.index >= start_date) & (df.index <= end_date)]
    df=df.reset_index(drop=False)
    return df

def verify_features(FEATURE_DF, fts):
    actual_fts=list(FEATURE_DF)
    verified_fts=[]
    for ft in fts:
        if ft in actual_fts:
            verified_fts.append(ft)
    return verified_fts

def configure_coin_fts(coin, gen_fts, str_to_replace="<coin_name>"):
    new_ft_list=[]
    for gf in gen_fts:
        new_ft=gf.replace(str_to_replace, coin)
        # print(new_ft)
        new_ft_list.append(new_ft)

    return new_ft_list

def get_feature_df_from_fp_list(all_fps, start_date,end_date,keep_time_col=False,time_col="created_at"):
    all_dfs=[]
    for fp in all_fps:
        try:
            df=pd.read_csv(fp)
            df[time_col]=pd.to_datetime(df[time_col], utc=True)
            all_dfs.append(df)
        except FileNotFoundError:
            print("%s does not exist. Moving on." %fp)
            if len(all_fps)==1:
                return pd.DataFrame(data={"blah":[]}),start_date,end_date

def rename_df_columns_with_tags(df, tag, time_col="created_at"):
    fts=list(df)
    if time_col in fts:
        fts.remove(time_col)

    for ft in fts:
        df=df.rename(columns={ft:ft+tag})

    return df

def get_feature_df_from_fp_list_and_fix_column_names(all_fps,fp_tag_dict,fp_ft_list_dict, start_date,end_date,keep_time_col=False,time_col="created_at"):
    all_dfs=[]
    for fp in all_fps:
        try:
            df=pd.read_csv(fp)
            desired_fts=fp_ft_list_dict[fp]
            desired_fts=[time_col] + desired_fts
            existing_fts=list(df)
            for ft in existing_fts:
                if ft not in desired_fts:
                    df=df.drop(ft, axis=1)


            # df=df[fts]
            df[time_col]=pd.to_datetime(df[time_col], utc=True)
            df=df.sort_values([time_col])
            tag=fp_tag_dict[fp]
            df=rename_df_columns_with_tags(df, tag)
            all_dfs.append(df)
        except FileNotFoundError:
            print("%s does not exist. Moving on." %fp)
            if len(all_fps)==1:
                return pd.DataFrame(data={"blah":[]}),start_date,end_date


    df = reduce(lambda left,right: pd.merge(left,right,on=[time_col],how="inner"), all_dfs)
    for ft in list(df):
        if "Unnamed" in ft:
            df=df.drop(ft, axis=1)
            print("Dropped %s"%ft)

    if start_date==-1:
        start_date=str(df[time_col].iloc[0])
        start_date=start_date.split(" ")[0]

    if end_date==-1:
        end_date=str(df[time_col].iloc[-1])
        end_date=end_date.split(" ")[0]

    df=df.set_index(time_col)
    df = df[(df.index >= start_date) & (df.index <= end_date)]
    df=df.reset_index(drop=False)

    if keep_time_col==False:
        df=df.drop(time_col, axis=1)
    return df,start_date,end_date




def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# def mape(y_true, y_pred):
#     y_true, y_pred = np.array(y_true), np.array(y_pred)
#     return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def mae(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred)))

def convert_to_differences(ORIGINAL_DATA, interval=1):
    ORIGINAL_SHAPE=ORIGINAL_DATA.shape

    print("Converting to differences. Original shape of data: %s" %str(ORIGINAL_SHAPE))

    DIFF=[]
    for i in range(interval, ORIGINAL_DATA.shape[0]):
        for j in range(ORIGINAL_DATA.shape[1]):
            value=ORIGINAL_DATA[i][j] -ORIGINAL_DATA[i-interval][j]
            DIFF.append(value)
    DIFF_ARRAY=np.asarray(DIFF)


    DIFF_ARRAY_NEW_SHAPE=(ORIGINAL_SHAPE[0]-1,ORIGINAL_SHAPE[1])

    DIFF_ARRAY=DIFF_ARRAY.reshape(DIFF_ARRAY_NEW_SHAPE)
    # ORIGINAL_DATA=ORIGINAL_DATA[1:,:]
    print("Shape of new DIFF_ARRAY: %s" %str(DIFF_ARRAY.shape))
    # print("Shape of new ORIGINAL_DATA: %s" %str(ORIGINAL_DATA.shape))

    # FEATURE_IDX=1

    # # print("Sanity check differences array")
    # # for i in range(500,510):
    # #     target_value=ORIGINAL_DATA[i+1][FEATURE_IDX]
    #     delta_to_reach_target=DIFF_ARRAY[i][FEATURE_IDX]
    #     starting_value=ORIGINAL_DATA[i][FEATURE_IDX]
    #     what_we_actually_got=delta_to_reach_target+starting_value
    #     print("Day %d, our target: %.4f" %(i+1, target_value))
    #     print("Day %d, what_we_actually_got: %.4f" %(i+1, what_we_actually_got))

    # DIFF_ARRAY=DIFF_ARRAY.reshape(ORIGINAL_SHAPE)

    return DIFF_ARRAY
    # return DIFF_ARRAY

def get_target_value_and_difference_arrays(ORIGINAL_DATA,X_ARRAY_TIMESTEPS_IN_A_SEQUENCE,Y_ARRAY_TIMESTEPS_IN_A_SEQUENCE,interval=1):
    TARGET_DIFF_DATA=convert_to_differences(ORIGINAL_DATA, interval)
    TARGET_VALUE_DATA=ORIGINAL_DATA[1:,:]
    SOURCE_DATA=ORIGINAL_DATA[:-1,:]
    # print("Shape of new DIFF_ARRAY: %s" %str(DIFF_ARRAY.shape))
    print("\nShape of old TARGET_DIFF_DATA: %s" %str(TARGET_DIFF_DATA.shape))
    print("Shape of old TARGET_VALUE_DATA: %s" %str(TARGET_VALUE_DATA.shape))
    print("Shape of old SOURCE_DATA: %s" %str(SOURCE_DATA.shape))

    NEW_TARGET_DIFF_DATA=[]
    NEW_TARGET_VALUE_DATA=[]
    NEW_SOURCE_DATA=[]

    NUM_LOOPS=TARGET_DIFF_DATA.shape[0]

    x_start_idx=0
    x_end_idx=X_ARRAY_TIMESTEPS_IN_A_SEQUENCE

    y_start_idx=x_end_idx-1
    y_end_idx =y_start_idx+ Y_ARRAY_TIMESTEPS_IN_A_SEQUENCE

    for i in range(NUM_LOOPS):
        TEMP_X=SOURCE_DATA[x_start_idx:x_end_idx,:]
        NEW_SOURCE_DATA.append(TEMP_X)

        TEMP_Y_VALUES=np.asarray(TARGET_VALUE_DATA[y_start_idx:y_end_idx,:])
        # print("\nShape of TEMP_Y_VALUES: %s" %str(TEMP_Y_VALUES.shape))
        NEW_TARGET_VALUE_DATA.append(TEMP_Y_VALUES)

        TEMP_Y_DELTAS=np.asarray(TARGET_DIFF_DATA[y_start_idx:y_end_idx,:])
        NEW_TARGET_DIFF_DATA.append(TEMP_Y_DELTAS)

        x_start_idx +=X_ARRAY_TIMESTEPS_IN_A_SEQUENCE
        x_end_idx+=X_ARRAY_TIMESTEPS_IN_A_SEQUENCE

        y_start_idx=x_end_idx-1
        y_end_idx =y_start_idx+ Y_ARRAY_TIMESTEPS_IN_A_SEQUENCE

        if x_start_idx >=NUM_LOOPS:
            break
        if x_end_idx >=NUM_LOOPS:
            break
        if y_start_idx >=NUM_LOOPS:
            break
        if y_end_idx >=NUM_LOOPS:
            break

    NEW_TARGET_DIFF_DATA=np.asarray(NEW_TARGET_DIFF_DATA)
    NEW_TARGET_VALUE_DATA=np.asarray(NEW_TARGET_VALUE_DATA)
    NEW_SOURCE_DATA=np.asarray(NEW_SOURCE_DATA)

    # print(NEW_SOURCE_DATA)
    # print(NEW_TARGET_VALUE_DATA)

    print("\nShape of new TARGET_DIFF_DATA: %s" %str(NEW_TARGET_DIFF_DATA.shape))
    print("Shape of new TARGET_VALUE_DATA: %s" %str(NEW_TARGET_VALUE_DATA.shape))
    print("Shape of new SOURCE_DATA: %s" %str(NEW_SOURCE_DATA.shape))

    x_sample_num=NEW_SOURCE_DATA.shape[0]
    y_sample_num=NEW_TARGET_DIFF_DATA.shape[0]

    MIN_SAMPLES=min([x_sample_num,y_sample_num])

    NEW_TARGET_DIFF_DATA=NEW_TARGET_DIFF_DATA[:MIN_SAMPLES,:].astype("float32")
    NEW_TARGET_VALUE_DATA=NEW_TARGET_VALUE_DATA[:MIN_SAMPLES,:].astype("float32")
    NEW_SOURCE_DATA=NEW_SOURCE_DATA[:MIN_SAMPLES,:].astype("float32")

    print("\nShape of final TARGET_DIFF_DATA: %s" %str(NEW_TARGET_DIFF_DATA.shape))
    print("Shape of final TARGET_VALUE_DATA: %s" %str(NEW_TARGET_VALUE_DATA.shape))
    print("Shape of final SOURCE_DATA: %s" %str(NEW_SOURCE_DATA.shape))


    return  NEW_SOURCE_DATA,NEW_TARGET_VALUE_DATA,NEW_TARGET_DIFF_DATA

def denormalize_data(x_train, x_val, x_test, scaler):
    old_train_shape=x_train.shape
    old_val_shape=x_val.shape
    old_test_shape=x_test.shape

    if len(x_train.shape) == 3:
    	x_train=x_train.reshape(x_train.shape[0]*x_train.shape[1],x_train.shape[2])

    if len(x_val.shape) == 3:
    	x_val=x_val.reshape(x_val.shape[0]*x_val.shape[1],x_val.shape[2])

    if len(x_test.shape) ==3:
    	x_test=x_test.reshape(x_test.shape[0]*x_test.shape[1],x_test.shape[2])


    x_train=scaler.inverse_transform(x_train)
    x_val=scaler.inverse_transform(x_val)
    x_test=scaler.inverse_transform(x_test)

    x_train=x_train.reshape(old_train_shape)
    x_val=x_val.reshape(old_val_shape)
    x_test=x_test.reshape(old_test_shape)

    return x_train, x_val, x_test





def simulate_on_initial_condition(x_test, y_test_as_deltas,y_test_as_values,model,scaler, num_sequences_to_simulate, SANITY_CHECK=False):
    #sanity checking tag means these are arrays we should not have access to during the challenge

    #DIFFERENCE THRESHOLD
    #SANITY CHECK PARAM
    DIFFERENCE_THRESHOLD=0.1

    num_timesteps_in_a_sequence=x_test.shape[1]

    #create init condit array
    x_init_condit=x_test[:1,:,:]

    #get first sequence of predictions
    prediction_deltas=model.predict(x_init_condit)

    #make x_init_condit 2d
    x_init_condit=x_init_condit.reshape((x_init_condit.shape[0]*x_init_condit.shape[1],x_init_condit.shape[2]))

    ######################SANITY CHECKING###########################
    #make y arrays 2d
    y_test_as_values=y_test_as_values.reshape((y_test_as_values.shape[0]*y_test_as_values.shape[1],y_test_as_values.shape[2]))
    y_test_as_deltas=y_test_as_deltas.reshape((y_test_as_deltas.shape[0]*y_test_as_deltas.shape[1],y_test_as_deltas.shape[2]))
    ################################################################

    #make pred array 2d
    prediction_deltas=prediction_deltas.reshape((prediction_deltas.shape[0]*prediction_deltas.shape[1],prediction_deltas.shape[2]))
    print("prediction_deltas shape: %s" %str(prediction_deltas.shape))

    #current starting point = last day in the sequence
    cur_x_test_base_vals_for_pred=x_init_condit[num_timesteps_in_a_sequence-1:, ]



    #we must inverse transform our initial condition so the deltas can be added to it properly
    cur_x_test_base_vals_for_pred=scaler.inverse_transform(cur_x_test_base_vals_for_pred)
    cur_x_test_base_vals_for_pred=cur_x_test_base_vals_for_pred.flatten()
    print("Cur base shape: %s" %str(cur_x_test_base_vals_for_pred.shape))

    #FOR SANITY CHECKING
    gt_cur_x_test_base_vals=cur_x_test_base_vals_for_pred.copy()

    #save all predicted sequences
    all_predicted_sequences=[]

    #get fts in a time step
    for i in range(num_sequences_to_simulate):

        #save your new cur sequence of predictions here
        cur_sequence_of_calculated_timestep_vals_not_deltas=[]

        ###########SANITY CHECKING#############
        #for sanity checking
        start_idx=i*num_timesteps_in_a_sequence
        end_idx=start_idx+num_timesteps_in_a_sequence

        #get the current week we are sanity checking
        #getting 7 days at a time
        cur_gt_delta_sequence=y_test_as_deltas[start_idx:end_idx,:]
        cur_gt_value_sequence=y_test_as_values[start_idx:end_idx,:]

        #make sure arrays have right shape
        print("cur_gt_delta_sequence shape: %s" %str(cur_gt_delta_sequence.shape))
        print("cur_gt_value_sequence: %s" %str(cur_gt_value_sequence.shape))
        ######################################

        #might have to add back 1
        for j in range(num_timesteps_in_a_sequence):

            #get cur timestep of predicted deltas
            cur_predicted_timestep_array_of_deltas_to_add=prediction_deltas[j]
            print("cur_predicted_timestep_array_of_deltas_to_add shape: %s" %str(cur_predicted_timestep_array_of_deltas_to_add.shape))

            #add pred deltas
            new_pred_array=cur_x_test_base_vals_for_pred +  cur_predicted_timestep_array_of_deltas_to_add
            print("new_pred_array: %s" %str(new_pred_array.shape))
            cur_sequence_of_calculated_timestep_vals_not_deltas.append(new_pred_array)
            cur_x_test_base_vals_for_pred=new_pred_array.copy()

            #normalize this and make it the current set of values to be added with deltas next
            # cur_x_test_base_vals_for_pred = new_pred_array.copy().reshape((1,new_pred_array.shape[0]))
            # cur_x_test_base_vals_for_pred=scaler.transform(cur_x_test_base_vals_for_pred).flatten()

            ############SANITY CHECK#######################################
            #sanity check with gt
            cur_gt_timestep_array_of_deltas_to_add = cur_gt_delta_sequence[j]
            sanity_check_gt_array=gt_cur_x_test_base_vals + cur_gt_timestep_array_of_deltas_to_add
            print("cur_predicted_timestep_array_of_deltas_to_add shape: %s" %str(cur_predicted_timestep_array_of_deltas_to_add.shape))
            print("sanity_check_gt_array: %s" %str(sanity_check_gt_array.shape))
            gt_cur_x_test_base_vals=sanity_check_gt_array.copy()
            #normalize this and make it the current set of values to be added with deltas next
            # gt_cur_x_test_base_vals = sanity_check_gt_array.copy().reshape((1,sanity_check_gt_array.shape[0]))
            # gt_cur_x_test_base_vals=scaler.transform(gt_cur_x_test_base_vals).flatten()
            ############SANITY CHECK######################################

            #get correct answer for comparison
            cur_gt_timestep_array_of_target_vals=cur_gt_value_sequence[j]

            #check differences to make sure we did this right
            for k in range(sanity_check_gt_array.shape[0]):
                difference=abs(sanity_check_gt_array[k] - cur_gt_timestep_array_of_target_vals[k])


                if difference > DIFFERENCE_THRESHOLD and SANITY_CHECK==True:
                    print("\nCalculated:")
                    print(sanity_check_gt_array[k] )
                    print("Expected:")
                    print(cur_gt_timestep_array_of_target_vals[k])
                    print("Error! Your simulation logic is off...")
                    sys.exit(0)
                # else:
                #     print("\nCalculated:")
                #     print(sanity_check_gt_array[k] )
                #     print("Expected:")
                #     print(cur_gt_timestep_array_of_target_vals[k])
                #     print("Correct")
            print("Adding back was successful")
            # sys.exit(0)

        #at the end of an iteration i, we must make a sequence array out of all the timestep features we just calculated
        #then, we must re-normalize them to make a new x_test array
        new_x_test = np.asarray(cur_sequence_of_calculated_timestep_vals_not_deltas)
        print("Shape of new_x_test: %s" %str(new_x_test.shape))

        #Do not normalize, we want these actual count results
        all_predicted_sequences.append(new_x_test.copy())
        print("Current number of predicted sequences: %d out of %d" %(len(all_predicted_sequences),num_sequences_to_simulate))

        #Now we have to normalize x test
        new_x_test=scaler.transform(new_x_test).reshape((1,new_x_test.shape[0],new_x_test.shape[1] ))

        #predict
        prediction_deltas=model.predict(new_x_test)

        #make pred array 2d
        prediction_deltas=prediction_deltas.reshape((prediction_deltas.shape[0]*prediction_deltas.shape[1],prediction_deltas.shape[2]))

        #current starting point = last day in the sequence
        new_x_test=new_x_test.reshape((new_x_test.shape[1],new_x_test.shape[2] ))
        cur_x_test_base_vals_for_pred=new_x_test[num_timesteps_in_a_sequence-1:, ]

        #we must inverse transform our new text x condition so the deltas can be added to it properly
        cur_x_test_base_vals_for_pred=scaler.inverse_transform(cur_x_test_base_vals_for_pred)
        cur_x_test_base_vals_for_pred=cur_x_test_base_vals_for_pred.flatten()
        print("Cur base shape: %s" %str(cur_x_test_base_vals_for_pred.shape))

    #convert our new predicted list into array
    all_predicted_sequences=np.asarray(all_predicted_sequences)
    print("all_predicted_sequences shape: %s" %str(all_predicted_sequences.shape))
    return all_predicted_sequences

def sanity_check_differences_in_2d_subarray(SOURCE_2D_ARRAY,TARGET_DIFF_2D_ARRAY,TARGET_VALUE_2D_ARRAY,X_ARRAY_TIMESTEPS_IN_A_SEQUENCE,CUR_CHECK_IDX,NUM_CHECKS,LIMIT=3,DIFFERENCE_THRESHOLD=0.1):
    print("On sanity check %d of %d" %((CUR_CHECK_IDX+1), NUM_CHECKS))
    NUM_LOOPS=LIMIT

    if LIMIT==-1:
        NUM_LOOPS=SOURCE_2D_ARRAY.shape[0]

    if LIMIT > SOURCE_2D_ARRAY.shape[0]:
        NUM_LOOPS=SOURCE_2D_ARRAY.shape[0]

    # TEMP_SOURCE_2D_ARRAY=SOURCE_2D_ARRAY
    # TEMP_TARGET_VALUE_2D_ARRAY=TARGET_VALUE_2D_ARRAY.flatten()
    # TEMP_TARGET_DIFF_2D_ARRAY=TARGET_DIFF_2D_ARRAY.flatten()
    # print("\nShape of TEMP_SOURCE_2D_ARRAY: %s" %str(TEMP_SOURCE_2D_ARRAY.shape))
    # print("Shape of TEMP_TARGET_VALUE_2D_ARRAY: %s" %str(TEMP_TARGET_VALUE_2D_ARRAY.shape))
    # print("Shape of TEMP_TARGET_DIFF_2D_ARRAY: %s" %str(TEMP_TARGET_DIFF_2D_ARRAY.shape))

    for i in range(NUM_LOOPS):
        # if i%9==0:
        #     print("On sanity check: %d" %(i+1))
        cur_base=SOURCE_2D_ARRAY[i][-1]
        for j in range(TARGET_DIFF_2D_ARRAY[i].shape[0]):
            cur_target=TARGET_VALUE_2D_ARRAY[i][j]
            cur_delta=TARGET_DIFF_2D_ARRAY[i][j]
            attempt=cur_base+ cur_delta
            # print("Correct target value: %.4f, my attempt: %.4f" %(cur_target,attempt))

            if (attempt != cur_target):
                if abs(attempt-cur_target) > DIFFERENCE_THRESHOLD:
                    print("Error!")
                    print("attempt != cur_target")
                    print("%.4f != %.4f"%(attempt,cur_target))
                    print(SOURCE_2D_ARRAY[i])
                    print(SOURCE_2D_ARRAY[i+1])
                    print(TARGET_DIFF_2D_ARRAY[i])

                    #uncomment this for sanity checking
                    sys.exit(0)
            cur_base=attempt



def sanity_check_all_arrays(SOURCE_DATA,TARGET_VALUE_DATA,TARGET_DIFF_DATA,X_ARRAY_TIMESTEPS_IN_A_SEQUENCE,LIMIT):
    # NUM_LOOPS=LIMIT
    # if LIMIT > SOURCE_DATA.shape[0]:
    #     NUM_LOOPS=SOURCE_DATA.shape[0]
    NUM_FEATURES=SOURCE_DATA.shape[2]

    ALL_SOURCE_SUBARRAYS=[]
    ALL_TARGET_DIFF_SUBARRAYS=[]
    ALL_TARGET_VALUE_SUBARRAYS=[]

    for i in range(NUM_FEATURES):
        SOURCE_SUBARRAY=SOURCE_DATA[:,:,i]
        SOURCE_SUBARRAY=SOURCE_SUBARRAY.reshape((SOURCE_SUBARRAY.shape[0],SOURCE_SUBARRAY.shape[1]))
        ALL_SOURCE_SUBARRAYS.append(SOURCE_SUBARRAY)

        TARGET_VALUE_SUBARRAY=TARGET_VALUE_DATA[:,:,i]
        TARGET_VALUE_SUBARRAY=TARGET_VALUE_SUBARRAY.reshape((TARGET_VALUE_SUBARRAY.shape[0],TARGET_VALUE_SUBARRAY.shape[1]))
        ALL_TARGET_VALUE_SUBARRAYS.append(TARGET_VALUE_SUBARRAY)

        TARGET_DIFF_SUBARRAY=TARGET_DIFF_DATA[:,:,i]
        TARGET_DIFF_SUBARRAY=TARGET_DIFF_SUBARRAY.reshape((TARGET_DIFF_SUBARRAY.shape[0],TARGET_DIFF_SUBARRAY.shape[1]))
        ALL_TARGET_DIFF_SUBARRAYS.append(TARGET_DIFF_SUBARRAY)

    ALL_SOURCE_SUBARRAYS=np.asarray(ALL_SOURCE_SUBARRAYS)
    ALL_TARGET_VALUE_SUBARRAYS=np.asarray(ALL_TARGET_VALUE_SUBARRAYS)
    ALL_TARGET_DIFF_SUBARRAYS=np.asarray(ALL_TARGET_DIFF_SUBARRAYS)

    print("\nShape of ALL_SOURCE_SUBARRAYS: %s" %str(ALL_SOURCE_SUBARRAYS.shape))
    print("Shape of final ALL_TARGET_VALUE_SUBARRAYS: %s" %str(ALL_TARGET_VALUE_SUBARRAYS.shape))
    print("Shape of ALL_TARGET_DIFF_SUBARRAYS: %s" %str(ALL_TARGET_DIFF_SUBARRAYS.shape))

    NUM_LOOPS=ALL_SOURCE_SUBARRAYS.shape[0]
    for i in range(NUM_LOOPS):
    # for i in range(1):
        SOURCE_2D_ARRAY=ALL_SOURCE_SUBARRAYS[i].reshape((ALL_SOURCE_SUBARRAYS.shape[1],ALL_SOURCE_SUBARRAYS.shape[2]))
        TARGET_DIFF_2D_ARRAY=ALL_TARGET_DIFF_SUBARRAYS[i].reshape((ALL_TARGET_DIFF_SUBARRAYS.shape[1],ALL_TARGET_DIFF_SUBARRAYS.shape[2]))
        TARGET_VALUE_2D_ARRAY=ALL_TARGET_VALUE_SUBARRAYS[i].reshape((ALL_TARGET_VALUE_SUBARRAYS.shape[1],ALL_TARGET_VALUE_SUBARRAYS.shape[2]))
        sanity_check_differences_in_2d_subarray(SOURCE_2D_ARRAY,TARGET_DIFF_2D_ARRAY,TARGET_VALUE_2D_ARRAY,X_ARRAY_TIMESTEPS_IN_A_SEQUENCE,i,NUM_LOOPS,LIMIT)










def get_feature_df(INPUT_FP,start_date,end_date):
    df=pd.read_csv(INPUT_FP)
    df["created_at"]=pd.to_datetime(df["created_at"], utc=True)
    preshape=str(df.shape)

    df=df.set_index("created_at")
    df = df[(df.index >= start_date) & (df.index <= end_date)]
    df=df.reset_index(drop=False)

    postshape=str(df.shape)

    return df

def get_proper_header_list(DF):
    HEADER_LIST=list(DF)
    if "created_at" in HEADER_LIST:
        HEADER_LIST.remove("created_at")
    if "Unnamed: 0" in HEADER_LIST:
        HEADER_LIST.remove("Unnamed: 0")
    print("Number of features to predict: %d" %len(HEADER_LIST))
    return HEADER_LIST

def convert_feature_df_to_feature_array(ALL_FEATURE_DFS):
    ALL_ARRAYS_LIST=[]
    for TEMP_DF in ALL_FEATURE_DFS:
        FEATURE_ARRAY=np.asarray(TEMP_DF).astype('float32')
        ALL_ARRAYS_LIST.append(FEATURE_ARRAY)
    FEATURE_ARRAY=np.concatenate(ALL_ARRAYS_LIST, axis=1)
    # FEATURE_ARRAY=np.expand_dims(FEATURE_ARRAY,axis=2)
    # print("Shape of feature array: %s" %str(FEATURE_ARRAY.shape))
    return FEATURE_ARRAY

def get_proper_feature_and_target_arrays(FEATURE_ARRAY,TARGET_ARRAY_AS_VALUES,TARGET_ARRAY_AS_DELTAS,X_ARRAY_TIMESTEPS_IN_A_SEQUENCE,Y_ARRAY_TIMESTEPS_IN_A_SEQUENCE):
    NUM_EVENTS_TO_PREDICT=FEATURE_ARRAY.shape[-1]
    TOTAL_NUM_DAYS=FEATURE_ARRAY.shape[0]
    # NUM_DAYS_TO_REMOVE_FOR_CLEAN_FIT_FOR_ARRAY_X
    NUM_DAYS_TO_REMOVE_FOR_CLEAN_FIT_FOR_ARRAY_X=TOTAL_NUM_DAYS%X_ARRAY_TIMESTEPS_IN_A_SEQUENCE
    print("NUM_DAYS_TO_REMOVE_FOR_CLEAN_FIT_FOR_ARRAY_X OF TIMESTEPS INTO TOTAL DAYS: %d" %NUM_DAYS_TO_REMOVE_FOR_CLEAN_FIT_FOR_ARRAY_X)
    # NUMBER_OF_DAYS_NEEDED_FOR_CLEAN_FIT_FOR_ARRAY_X
    NUMBER_OF_DAYS_NEEDED_FOR_CLEAN_FIT_FOR_ARRAY_X=TOTAL_NUM_DAYS-NUM_DAYS_TO_REMOVE_FOR_CLEAN_FIT_FOR_ARRAY_X
    print("NUMBER_OF_DAYS_NEEDED_FOR_CLEAN_FIT_FOR_ARRAY_X OF TIMESTEPS INTO TOTAL DAYS: %d" %NUMBER_OF_DAYS_NEEDED_FOR_CLEAN_FIT_FOR_ARRAY_X)

    TOTAL_SEQUENCES=int(NUMBER_OF_DAYS_NEEDED_FOR_CLEAN_FIT_FOR_ARRAY_X/X_ARRAY_TIMESTEPS_IN_A_SEQUENCE)
    print("TOTAL TIMESTEPS: %d" %TOTAL_SEQUENCES)

    print("Now we must re-structure the feature and platform arrays")
    OFFSET=NUM_DAYS_TO_REMOVE_FOR_CLEAN_FIT_FOR_ARRAY_X

    FEATURE_ARRAY=FEATURE_ARRAY[OFFSET:,:]
    TARGET_ARRAY_AS_VALUES=TARGET_ARRAY_AS_VALUES[OFFSET:,:]
    TARGET_ARRAY_AS_DELTAS=TARGET_ARRAY_AS_DELTAS[OFFSET-1:,:]

    print("Shape of FEATURE_ARRAY after cutting out first %d days : %s" %(NUM_DAYS_TO_REMOVE_FOR_CLEAN_FIT_FOR_ARRAY_X,str(FEATURE_ARRAY.shape)))
    print("Shape of TARGET_ARRAY_AS_VALUES after cutting out first %d days : %s" %(NUM_DAYS_TO_REMOVE_FOR_CLEAN_FIT_FOR_ARRAY_X,str(TARGET_ARRAY_AS_VALUES.shape)))
    print("Shape of TARGET_ARRAY_AS_DELTAS after cutting out first %d days : %s" %(NUM_DAYS_TO_REMOVE_FOR_CLEAN_FIT_FOR_ARRAY_X-1,str(TARGET_ARRAY_AS_DELTAS.shape)))

    print("Now putting vector X into the proper shape.")
    NEW_FEATURE_ARRAY_SHAPE=(TOTAL_SEQUENCES, X_ARRAY_TIMESTEPS_IN_A_SEQUENCE,NUM_EVENTS_TO_PREDICT)
    print(NEW_FEATURE_ARRAY_SHAPE)
    # sys.exit(0)
    FEATURE_ARRAY= FEATURE_ARRAY.reshape(NEW_FEATURE_ARRAY_SHAPE)
    print("Shape of FEATURE_ARRAY after reshaping: %s" %str(FEATURE_ARRAY.shape))

    NEW_TARGET_ARRAY_SHAPE=(TOTAL_SEQUENCES, X_ARRAY_TIMESTEPS_IN_A_SEQUENCE,NUM_EVENTS_TO_PREDICT)

    TARGET_ARRAY_AS_VALUES=TARGET_ARRAY_AS_VALUES.reshape(NEW_TARGET_ARRAY_SHAPE)
    TARGET_ARRAY_AS_DELTAS=TARGET_ARRAY_AS_DELTAS.reshape(NEW_TARGET_ARRAY_SHAPE)


    print("Shape of TARGET_ARRAY_AS_VALUES after reshaping: %s" %str(TARGET_ARRAY_AS_VALUES.shape))
    print("Shape of TARGET_ARRAY_AS_DELTAS after reshaping: %s" %str(TARGET_ARRAY_AS_DELTAS.shape))

    print("Now reconfiguring vector Y and putting it into the proper shape.")
    TEMP_TARGET_LIST_AS_VALUES=[]
    TEMP_TARGET_LIST_AS_DELTAS=[]
    for i in range(FEATURE_ARRAY.shape[0]-1):
        TEMP_TARGET_LIST_AS_VALUES.append(TARGET_ARRAY_AS_VALUES[i+1][:Y_ARRAY_TIMESTEPS_IN_A_SEQUENCE])
        TEMP_TARGET_LIST_AS_DELTAS.append(TARGET_ARRAY_AS_DELTAS[i+1][:Y_ARRAY_TIMESTEPS_IN_A_SEQUENCE])

    OLD_TARGET_ARRAY_AS_VALUES=np.copy(TARGET_ARRAY_AS_VALUES)
    OLD_TARGET_ARRAY_AS_DELTAS=np.copy(TARGET_ARRAY_AS_DELTAS)

    TARGET_ARRAY_AS_VALUES=np.asarray(TEMP_TARGET_LIST_AS_VALUES)
    TARGET_ARRAY_AS_DELTAS=np.asarray(TEMP_TARGET_LIST_AS_DELTAS)

    FEATURE_ARRAY=FEATURE_ARRAY[:-1,:,:]


    print("\nShape of FEATURE_ARRAY after reshaping: %s" %str(FEATURE_ARRAY.shape))
    print("Shape of TARGET_ARRAY_AS_VALUES after reconfiguring: %s" %str(TARGET_ARRAY_AS_VALUES.shape))
    print("Shape of TARGET_ARRAY_AS_DELTAS after reconfiguring: %s" %str(TARGET_ARRAY_AS_DELTAS.shape))




    return FEATURE_ARRAY, TARGET_ARRAY_AS_VALUES, TARGET_ARRAY_AS_DELTAS

def get_train_val_and_test_arrays(FEATURE_ARRAY,TARGET_ARRAY_AS_VALUES, TARGET_ARRAY_AS_DELTAS,VAL_PERCENTAGE,TEST_PERCENTAGE):
    NUM_SAMPLES=FEATURE_ARRAY.shape[0]
    NUM_VAL_SAMPLES=int(NUM_SAMPLES * VAL_PERCENTAGE)
    NUM_TESTING_SAMPLES=int(NUM_SAMPLES *TEST_PERCENTAGE)
    NUM_TRAINING_SAMPLES= NUM_SAMPLES - NUM_VAL_SAMPLES - NUM_TESTING_SAMPLES

    if NUM_TRAINING_SAMPLES >=3 and NUM_TESTING_SAMPLES ==0 and NUM_VAL_SAMPLES==0:
        NUM_TRAINING_SAMPLES=NUM_TRAINING_SAMPLES -2
        NUM_VAL_SAMPLES=1
        NUM_TESTING_SAMPLES=1

    if NUM_TESTING_SAMPLES ==0:
        if NUM_VAL_SAMPLES > 2:
            NUM_TESTING_SAMPLES +=1
            NUM_VAL_SAMPLES -= 1
        else:
            NUM_TRAINING_SAMPLES -=1
            NUM_TESTING_SAMPLES+=1

    print("There are %d samples" %NUM_SAMPLES)
    print("%d samples will be used training" %NUM_TRAINING_SAMPLES)
    print("%d samples will be used for validation" %NUM_VAL_SAMPLES)
    print("%d samples will be used for testing" %NUM_TESTING_SAMPLES)

    x_train=FEATURE_ARRAY[:NUM_TRAINING_SAMPLES, :, :]
    y_train_as_values=TARGET_ARRAY_AS_VALUES[:NUM_TRAINING_SAMPLES, :, :]
    y_train_as_deltas=TARGET_ARRAY_AS_DELTAS[:NUM_TRAINING_SAMPLES, :, :]

    x_val= FEATURE_ARRAY[NUM_TRAINING_SAMPLES:NUM_TRAINING_SAMPLES + NUM_VAL_SAMPLES, :,:]
    y_val_as_values= TARGET_ARRAY_AS_VALUES[NUM_TRAINING_SAMPLES:NUM_TRAINING_SAMPLES + NUM_VAL_SAMPLES, :,:]
    y_val_as_deltas= TARGET_ARRAY_AS_DELTAS[NUM_TRAINING_SAMPLES:NUM_TRAINING_SAMPLES + NUM_VAL_SAMPLES, :,:]


    x_test= FEATURE_ARRAY[NUM_TRAINING_SAMPLES + NUM_VAL_SAMPLES:,:,:]
    y_test_as_values= TARGET_ARRAY_AS_VALUES[NUM_TRAINING_SAMPLES + NUM_VAL_SAMPLES:,:,:]
    y_test_as_deltas= TARGET_ARRAY_AS_DELTAS[NUM_TRAINING_SAMPLES + NUM_VAL_SAMPLES:,:,:]

    print("x_train: shape:")
    print (x_train.shape)
    print("x_val shape:")
    print (x_val.shape)
    print("x_test shape:")
    print (x_test.shape)

    print("y_train_as_values: shape:")
    print (y_train_as_values.shape)
    print("y_train_as_deltas: shape:")
    print (y_train_as_deltas.shape)

    print("y_val_as_values shape:")
    print (y_val_as_values.shape)
    print("y_val_as_deltas shape:")
    print (y_val_as_deltas.shape)


    print("y_test_as_values shape:")
    print (y_test_as_values.shape)
    print("y_test_as_deltas shape:")
    print (y_test_as_deltas.shape)

    # sys.exit(0)

    return x_train, x_val, x_test, y_train_as_values,y_train_as_deltas, y_val_as_values, y_val_as_deltas, y_test_as_values,y_test_as_deltas

def get_train_val_and_test_arrays_with_date_info(FEATURE_ARRAY,TARGET_ARRAY_AS_VALUES, TARGET_ARRAY_AS_DELTAS,VAL_PERCENTAGE,TEST_PERCENTAGE,output_dir,start_date,end_date, GRANULARITY,X_ARRAY_TIMESTEPS_IN_A_SEQUENCE,Y_ARRAY_TIMESTEPS_IN_A_SEQUENCE):
    NUM_SAMPLES=FEATURE_ARRAY.shape[0]
    NUM_VAL_SAMPLES=int(NUM_SAMPLES * VAL_PERCENTAGE)
    NUM_TESTING_SAMPLES=int(NUM_SAMPLES *TEST_PERCENTAGE)
    NUM_TRAINING_SAMPLES= NUM_SAMPLES - NUM_VAL_SAMPLES - NUM_TESTING_SAMPLES

    if NUM_TRAINING_SAMPLES >=3 and NUM_TESTING_SAMPLES ==0 and NUM_VAL_SAMPLES==0:
        NUM_TRAINING_SAMPLES=NUM_TRAINING_SAMPLES -2
        NUM_VAL_SAMPLES=1
        NUM_TESTING_SAMPLES=1

    if NUM_TESTING_SAMPLES ==0:
        if NUM_VAL_SAMPLES > 2:
            NUM_TESTING_SAMPLES +=1
            NUM_VAL_SAMPLES -= 1
        else:
            NUM_TRAINING_SAMPLES -=1
            NUM_TESTING_SAMPLES+=1

    date_fp=output_dir + "Coin-Date-Info.txt"
    f=open(date_fp, "w")

    #setup dates
    dates=list(pd.date_range(start_date, end_date, freq=GRANULARITY))

    #SETUP INDICES
    xtrain_start=0
    xtrain_end = NUM_TRAINING_SAMPLES * X_ARRAY_TIMESTEPS_IN_A_SEQUENCE

    xval_start=xtrain_end
    xval_end = xval_start + NUM_VAL_SAMPLES* X_ARRAY_TIMESTEPS_IN_A_SEQUENCE

    xtest_start = xval_end
    xtest_end = xtest_start + NUM_TESTING_SAMPLES * X_ARRAY_TIMESTEPS_IN_A_SEQUENCE

    #get dates
    x_train_dates = dates[xtrain_start:xtrain_end]
    x_val_dates = dates[xval_start:xval_end]
    x_test_dates = dates[xtest_start:xtest_end]

    ytrain_start=X_ARRAY_TIMESTEPS_IN_A_SEQUENCE
    ytrain_end = ytrain_start + NUM_TRAINING_SAMPLES * X_ARRAY_TIMESTEPS_IN_A_SEQUENCE

    yval_start=ytrain_end
    yval_end = yval_start + NUM_VAL_SAMPLES* X_ARRAY_TIMESTEPS_IN_A_SEQUENCE

    ytest_start = yval_end
    ytest_end = ytest_start + NUM_TESTING_SAMPLES * X_ARRAY_TIMESTEPS_IN_A_SEQUENCE

    #get dates
    y_train_dates = dates[ytrain_start:ytrain_end]
    y_val_dates = dates[yval_start:yval_end]
    y_test_dates = dates[ytest_start:ytest_end]

    #make date dfs
    train_dates_df=pd.DataFrame(data={"x_train_date":x_train_dates, "y_train_date":y_train_dates})
    val_dates_df=pd.DataFrame(data={"x_val_date":x_val_dates, "y_val_date":y_val_dates})
    test_dates_df=pd.DataFrame(data={"x_test_date":x_test_dates, "y_test_date":y_test_dates})
    print("DATES:")
    print(train_dates_df)
    print(val_dates_df)
    print(test_dates_df)

    #save them
    train_fp = output_dir + "training-dates.csv"
    train_dates_df.to_csv(train_fp)
    print("Saved training dates to %s" %train_fp)

    val_fp = output_dir + "val-dates.csv"
    val_dates_df.to_csv(val_fp)
    print("Saved val dates to %s" %val_fp)

    test_fp = output_dir + "testing-dates.csv"
    test_dates_df.to_csv(test_fp)
    print("Saved testing dates to %s" %test_fp)

    screen_and_file_print("There are %d samples" %NUM_SAMPLES, f)
    screen_and_file_print("%d samples will be used training" %NUM_TRAINING_SAMPLES, f)
    screen_and_file_print("%d samples will be used for validation" %NUM_VAL_SAMPLES, f)
    screen_and_file_print("%d samples will be used for testing" %NUM_TESTING_SAMPLES, f)

    x_train=FEATURE_ARRAY[:NUM_TRAINING_SAMPLES, :, :]
    y_train_as_values=TARGET_ARRAY_AS_VALUES[:NUM_TRAINING_SAMPLES, :, :]
    y_train_as_deltas=TARGET_ARRAY_AS_DELTAS[:NUM_TRAINING_SAMPLES, :, :]

    x_val= FEATURE_ARRAY[NUM_TRAINING_SAMPLES:NUM_TRAINING_SAMPLES + NUM_VAL_SAMPLES, :,:]
    y_val_as_values= TARGET_ARRAY_AS_VALUES[NUM_TRAINING_SAMPLES:NUM_TRAINING_SAMPLES + NUM_VAL_SAMPLES, :,:]
    y_val_as_deltas= TARGET_ARRAY_AS_DELTAS[NUM_TRAINING_SAMPLES:NUM_TRAINING_SAMPLES + NUM_VAL_SAMPLES, :,:]


    x_test= FEATURE_ARRAY[NUM_TRAINING_SAMPLES + NUM_VAL_SAMPLES:,:,:]
    y_test_as_values= TARGET_ARRAY_AS_VALUES[NUM_TRAINING_SAMPLES + NUM_VAL_SAMPLES:,:,:]
    y_test_as_deltas= TARGET_ARRAY_AS_DELTAS[NUM_TRAINING_SAMPLES + NUM_VAL_SAMPLES:,:,:]

    print("x_train: shape:")
    print (x_train.shape)
    print("x_val shape:")
    print (x_val.shape)
    print("x_test shape:")
    print (x_test.shape)

    print("y_train_as_values: shape:")
    print (y_train_as_values.shape)
    print("y_train_as_deltas: shape:")
    print (y_train_as_deltas.shape)

    print("y_val_as_values shape:")
    print (y_val_as_values.shape)
    print("y_val_as_deltas shape:")
    print (y_val_as_deltas.shape)


    print("y_test_as_values shape:")
    print (y_test_as_values.shape)
    print("y_test_as_deltas shape:")
    print (y_test_as_deltas.shape)



    # sys.exit(0)

    return x_train, x_val, x_test, y_train_as_values,y_train_as_deltas, y_val_as_values, y_val_as_deltas, y_test_as_values,y_test_as_deltas


def get_training_array(FEATURE_ARRAY,TARGET_ARRAY_AS_VALUES, TARGET_ARRAY_AS_DELTAS):
    NUM_SAMPLES=FEATURE_ARRAY.shape[0]
    # NUM_VAL_SAMPLES=int(NUM_SAMPLES * VAL_PERCENTAGE)
    # NUM_TESTING_SAMPLES=int(NUM_SAMPLES *TEST_PERCENTAGE)
    # NUM_TRAINING_SAMPLES= NUM_SAMPLES - NUM_VAL_SAMPLES - NUM_TESTING_SAMPLES

    # if NUM_TRAINING_SAMPLES >=3 and NUM_TESTING_SAMPLES ==0 and NUM_VAL_SAMPLES==0:
    #     NUM_TRAINING_SAMPLES=NUM_TRAINING_SAMPLES -2
    #     NUM_VAL_SAMPLES=1
    #     NUM_TESTING_SAMPLES=1

    print("There are %d samples" %NUM_SAMPLES)
    # print("%d samples will be used training" %NUM_TRAINING_SAMPLES)
    # print("%d samples will be used for validation" %NUM_VAL_SAMPLES)
    # print("%d samples will be used for testing" %NUM_TESTING_SAMPLES)

    x_train=FEATURE_ARRAY
    y_train_as_values=TARGET_ARRAY_AS_VALUES
    y_train_as_deltas=TARGET_ARRAY_AS_DELTAS



    print("x_train: shape:")
    print (x_train.shape)


    print("y_train_as_values: shape:")
    print (y_train_as_values.shape)
    print("y_train_as_deltas: shape:")
    print (y_train_as_deltas.shape)

    return x_train, y_train_as_values,y_train_as_deltas

def cp3_get_array_from_feature_df_DAILY(FEATURE_DF,LOG1_FLAG,LOG2_FLAG,start,
    LOG_ADD_COEFF,X_ARRAY_TIMESTEPS_IN_A_SEQUENCE,Y_ARRAY_TIMESTEPS_IN_A_SEQUENCE):
    #make sure df is nonempty:
    if FEATURE_DF.shape[0]==0:
        print("%s has no features. Skipping...")
        return

    #get all the fts we want to use
    fts=list(FEATURE_DF)
    fts=verify_features(FEATURE_DF, fts)

    #config df with verified fts
    FEATURE_DF=FEATURE_DF[fts]
    if "created_at" in list(FEATURE_DF):
        FEATURE_DF=FEATURE_DF.drop("created_at", axis=1)
    FEATURES_OF_INTEREST=get_proper_header_list(FEATURE_DF)
    FEATURE_DF=FEATURE_DF[FEATURES_OF_INTEREST]

    #log normalize
    if LOG1_FLAG==True:
        FEATURE_DF=np.log1p(FEATURE_DF + LOG_ADD_COEFF)
    if LOG2_FLAG==True:
        FEATURE_DF=np.log1p(FEATURE_DF + LOG_ADD_COEFF)
        print("np log1p on df")
        print(FEATURE_DF)

    #show our fts
    fts=list(FEATURE_DF)
    for ft in fts:
        print(ft)

    #get the starting day of the week
    start_date_obj=pd.to_datetime(start, utc=True)
    start_dow=start_date_obj.weekday()
    initial_dow=start_dow
    print("Starting DOW: %s" %str(start_dow))

    NUM_FEATURES_TO_PREDICT=len(FEATURES_OF_INTEREST)

    #convert all feature dfs to arrays
    TEMP_DF=FEATURE_DF[FEATURES_OF_INTEREST].copy()
    ALL_FEATURE_DFS=[TEMP_DF]
    FEATURE_ARRAY=convert_feature_df_to_feature_array(ALL_FEATURE_DFS)
    del TEMP_DF
    gc.collect()

    ############################## DATA PREPROCESSING ####################################
    #create source and target arrays
    SOURCE_DATA,TARGET_VALUE_DATA,TARGET_DIFF_DATA=get_target_value_and_difference_arrays(FEATURE_ARRAY,X_ARRAY_TIMESTEPS_IN_A_SEQUENCE,Y_ARRAY_TIMESTEPS_IN_A_SEQUENCE)

    # #uncomment this to sanity check
    # # sanity_check_all_arrays(SOURCE_DATA,TARGET_VALUE_DATA,TARGET_DIFF_DATA,X_ARRAY_TIMESTEPS_IN_A_SEQUENCE,LIMIT=-1)
    # # sys.exit(0)
    x_train, y_train_as_values,y_train_as_deltas = get_training_array(SOURCE_DATA,TARGET_VALUE_DATA,TARGET_DIFF_DATA)

    return x_train, y_train_as_values,y_train_as_deltas,initial_dow

def cp3_get_training_array_from_feature_df_DAILY(FEATURE_DF,LOG1_FLAG,LOG2_FLAG,train_start,LSTM_MULT_COEFFICIENT,
    LOG_ADD_COEFF,X_ARRAY_TIMESTEPS_IN_A_SEQUENCE,Y_ARRAY_TIMESTEPS_IN_A_SEQUENCE):
    #make sure df is nonempty:
    if FEATURE_DF.shape[0]==0:
        print("%s has no features. Skipping...")
        return

    #get all the fts we want to use
    fts=list(FEATURE_DF)
    fts=verify_features(FEATURE_DF, fts)

    #if debug is true, use less fts
    # if DEBUG==True:
    #   fts=fts[:4]

    #config df with verified fts
    FEATURE_DF=FEATURE_DF[fts]
    if "created_at" in list(FEATURE_DF):
        FEATURE_DF=FEATURE_DF.drop("created_at", axis=1)
    FEATURES_OF_INTEREST=get_proper_header_list(FEATURE_DF)
    FEATURE_DF=FEATURE_DF[FEATURES_OF_INTEREST]

    #log normalize
    if LOG1_FLAG==True:
        FEATURE_DF=np.log1p(FEATURE_DF + LOG_ADD_COEFF)
    if LOG2_FLAG==True:
        FEATURE_DF=np.log1p(FEATURE_DF + LOG_ADD_COEFF)
        print("np log1p on df")
        print(FEATURE_DF)

    #show our fts
    fts=list(FEATURE_DF)
    for ft in fts:
        print(ft)


        #get the starting day of the week
    start_date_obj=pd.to_datetime(train_start, utc=True)
    start_dow=start_date_obj.weekday()
    initial_dow=start_dow
    print("Starting DOW: %s" %str(start_dow))

    NUM_FEATURES_TO_PREDICT=len(FEATURES_OF_INTEREST)

    #Setup lstm units
    LSTM_UNITS=int(NUM_FEATURES_TO_PREDICT * LSTM_MULT_COEFFICIENT)
    if LSTM_UNITS==0:
        LSTM_UNITS=1


    for ft in FEATURES_OF_INTEREST:
        print(ft)



    #convert all feature dfs to arrays
    TEMP_DF=FEATURE_DF[FEATURES_OF_INTEREST].copy()
    ALL_FEATURE_DFS=[TEMP_DF]
    FEATURE_ARRAY=convert_feature_df_to_feature_array(ALL_FEATURE_DFS)
    del TEMP_DF
    gc.collect()

    ############################## DATA PREPROCESSING ####################################
    #create source and target arrays
    SOURCE_DATA,TARGET_VALUE_DATA,TARGET_DIFF_DATA=get_target_value_and_difference_arrays(FEATURE_ARRAY,X_ARRAY_TIMESTEPS_IN_A_SEQUENCE,Y_ARRAY_TIMESTEPS_IN_A_SEQUENCE)

    # #uncomment this to sanity check
    # # sanity_check_all_arrays(SOURCE_DATA,TARGET_VALUE_DATA,TARGET_DIFF_DATA,X_ARRAY_TIMESTEPS_IN_A_SEQUENCE,LIMIT=-1)
    # # sys.exit(0)
    x_train, y_train_as_values,y_train_as_deltas = get_training_array(SOURCE_DATA,TARGET_VALUE_DATA,TARGET_DIFF_DATA)

    return x_train, y_train_as_values,y_train_as_deltas,LSTM_UNITS,initial_dow


def cp3_get_testing_array_from_feature_df_DAILY(FEATURE_DF,LOG1_FLAG,LOG2_FLAG,test_start,
    LOG_ADD_COEFF,X_ARRAY_TIMESTEPS_IN_A_SEQUENCE,Y_ARRAY_TIMESTEPS_IN_A_SEQUENCE):
    #make sure df is nonempty:
    if FEATURE_DF.shape[0]==0:
        print("%s has no features. Skipping...")
        return

    #get all the fts we want to use
    fts=list(FEATURE_DF)
    fts=verify_features(FEATURE_DF, fts)

    #if debug is true, use less fts
    # if DEBUG==True:
    #   fts=fts[:4]

    #config df with verified fts
    FEATURE_DF=FEATURE_DF[fts]
    if "created_at" in list(FEATURE_DF):
        FEATURE_DF=FEATURE_DF.drop("created_at", axis=1)
    FEATURES_OF_INTEREST=get_proper_header_list(FEATURE_DF)
    FEATURE_DF=FEATURE_DF[FEATURES_OF_INTEREST]

    #log normalize
    if LOG1_FLAG==True:
        FEATURE_DF=np.log1p(FEATURE_DF + LOG_ADD_COEFF)
    if LOG2_FLAG==True:
        FEATURE_DF=np.log1p(FEATURE_DF + LOG_ADD_COEFF)
        print("np log1p on df")
        print(FEATURE_DF)

    #show our fts
    fts=list(FEATURE_DF)
    for ft in fts:
        print(ft)


        #get the starting day of the week
    start_date_obj=pd.to_datetime(test_start, utc=True)
    start_dow=start_date_obj.weekday()
    initial_dow=start_dow
    print("Starting DOW: %s" %str(start_dow))

    NUM_FEATURES_TO_PREDICT=len(FEATURES_OF_INTEREST)



    for ft in FEATURES_OF_INTEREST:
        print(ft)



    #convert all feature dfs to arrays
    TEMP_DF=FEATURE_DF[FEATURES_OF_INTEREST].copy()
    ALL_FEATURE_DFS=[TEMP_DF]
    FEATURE_ARRAY=convert_feature_df_to_feature_array(ALL_FEATURE_DFS)
    del TEMP_DF
    gc.collect()

    ############################## DATA PREPROCESSING ####################################
    #create source and target arrays
    SOURCE_DATA,TARGET_VALUE_DATA,TARGET_DIFF_DATA=get_target_value_and_difference_arrays(FEATURE_ARRAY,X_ARRAY_TIMESTEPS_IN_A_SEQUENCE,Y_ARRAY_TIMESTEPS_IN_A_SEQUENCE)

    # #uncomment this to sanity check
    # # sanity_check_all_arrays(SOURCE_DATA,TARGET_VALUE_DATA,TARGET_DIFF_DATA,X_ARRAY_TIMESTEPS_IN_A_SEQUENCE,LIMIT=-1)
    # # sys.exit(0)
    x_test, y_test_as_values,y_test_as_deltas = get_training_array(SOURCE_DATA,TARGET_VALUE_DATA,TARGET_DIFF_DATA)

    return x_test, y_test_as_values,y_test_as_deltas,initial_dow

def get_training_array_from_feature_df(FEATURE_DF,LOG1_FLAG,LOG2_FLAG,train_start,coin_names,coin,kept_coins,coin_ft_dict ,LSTM_MULT_COEFFICIENT,
    LOG_ADD_COEFF,NUM_FTS_TO_ENFORCE,ALL_POTENTIAL_FTS,X_ARRAY_TIMESTEPS_IN_A_SEQUENCE,Y_ARRAY_TIMESTEPS_IN_A_SEQUENCE):
    #make sure df is nonempty:
    if FEATURE_DF.shape[0]==0:
        print("%s has no features. Skipping...")
        return

    #get all the fts we want to use
    fts=list(FEATURE_DF)
    fts=verify_features(FEATURE_DF, fts)

    #if debug is true, use less fts
    # if DEBUG==True:
    #   fts=fts[:4]

    #config df with verified fts
    FEATURE_DF=FEATURE_DF[fts]
    if "created_at" in list(FEATURE_DF):
        FEATURE_DF=FEATURE_DF.drop("created_at", axis=1)
    FEATURES_OF_INTEREST=get_proper_header_list(FEATURE_DF)
    FEATURE_DF=FEATURE_DF[FEATURES_OF_INTEREST]

    #log normalize
    if LOG1_FLAG==True:
        FEATURE_DF=np.log1p(FEATURE_DF + LOG_ADD_COEFF)
    if LOG2_FLAG==True:
        FEATURE_DF=np.log1p(FEATURE_DF + LOG_ADD_COEFF)
        print("np log1p on df")
        print(FEATURE_DF)

    #show our fts
    fts=list(FEATURE_DF)
    for ft in fts:
        print(ft)

    #get the starting day of the week
    # start_date_obj=pd.to_datetime(train_start, utc=True)
    # start_month=start_date_obj.month()
    initial_month=int(train_start[5:7])
    print("Starting month: %s" %str(initial_month))
    # sys.exit(0)

    #print time info
    # print(start_date)
    # print(end_date)

    NUM_FEATURES_TO_PREDICT=len(FEATURES_OF_INTEREST)

    #Setup lstm units
    LSTM_UNITS=int(NUM_FEATURES_TO_PREDICT * LSTM_MULT_COEFFICIENT)
    if LSTM_UNITS==0:
        LSTM_UNITS=1

    #add features to dict for particular coin
    coin_ft_dict[coin]=FEATURES_OF_INTEREST
    print("Coin features for %s" %coin)
    for ft in FEATURES_OF_INTEREST:
        print(ft)

    if (NUM_FTS_TO_ENFORCE != len(FEATURES_OF_INTEREST))and (len(coin_names)>1):
        print("NUM_FTS_TO_ENFORCE != len(FEATURES_OF_INTEREST)")
        print("%d != %d"%(NUM_FTS_TO_ENFORCE, len(FEATURES_OF_INTEREST)))
        return
    else:
        kept_coins.append(coin)
        print("FEATURES IN KEPT COIN:")
        ENFORCED_FEATURES_OF_INTEREST = ALL_POTENTIAL_FTS
        print(ENFORCED_FEATURES_OF_INTEREST)

    #convert all feature dfs to arrays
    TEMP_DF=FEATURE_DF[FEATURES_OF_INTEREST].copy()
    ALL_FEATURE_DFS=[TEMP_DF]
    FEATURE_ARRAY=convert_feature_df_to_feature_array(ALL_FEATURE_DFS)
    del TEMP_DF
    gc.collect()

    ############################## DATA PREPROCESSING ####################################
    #create source and target arrays
    SOURCE_DATA,TARGET_VALUE_DATA,TARGET_DIFF_DATA=get_target_value_and_difference_arrays(FEATURE_ARRAY,X_ARRAY_TIMESTEPS_IN_A_SEQUENCE,Y_ARRAY_TIMESTEPS_IN_A_SEQUENCE)

    # #uncomment this to sanity check
    # # sanity_check_all_arrays(SOURCE_DATA,TARGET_VALUE_DATA,TARGET_DIFF_DATA,X_ARRAY_TIMESTEPS_IN_A_SEQUENCE,LIMIT=-1)
    # # sys.exit(0)
    x_train, y_train_as_values,y_train_as_deltas = get_training_array(SOURCE_DATA,TARGET_VALUE_DATA,TARGET_DIFF_DATA)

    return x_train, y_train_as_values,y_train_as_deltas,LSTM_UNITS,kept_coins,coin_ft_dict,ENFORCED_FEATURES_OF_INTEREST,initial_month

def get_x_and_y_arrays_for_2d_input(FEATURE_ARRAY,TARGET_ARRAY_AS_VALUES, TARGET_ARRAY_AS_DELTAS,VAL_PERCENTAGE,TEST_PERCENTAGE):


    NUM_SAMPLES=FEATURE_ARRAY.shape[0]
    NUM_VAL_SAMPLES=int(NUM_SAMPLES * VAL_PERCENTAGE)
    NUM_TESTING_SAMPLES=int(NUM_SAMPLES *TEST_PERCENTAGE)
    NUM_TRAINING_SAMPLES= NUM_SAMPLES - NUM_VAL_SAMPLES - NUM_TESTING_SAMPLES

    if NUM_TRAINING_SAMPLES >=3 and NUM_TESTING_SAMPLES ==0 and NUM_VAL_SAMPLES==0:
        NUM_TRAINING_SAMPLES=NUM_TRAINING_SAMPLES -2
        NUM_VAL_SAMPLES=1
        NUM_TESTING_SAMPLES=1

    print("There are %d samples" %NUM_SAMPLES)
    print("%d samples will be used training" %NUM_TRAINING_SAMPLES)
    print("%d samples will be used for validation" %NUM_VAL_SAMPLES)
    print("%d samples will be used for testing" %NUM_TESTING_SAMPLES)

    x_train=FEATURE_ARRAY[:NUM_TRAINING_SAMPLES, :]
    y_train_as_values=TARGET_ARRAY_AS_VALUES[:NUM_TRAINING_SAMPLES, :]
    y_train_as_deltas=TARGET_ARRAY_AS_DELTAS[:NUM_TRAINING_SAMPLES, :]

    x_val= FEATURE_ARRAY[NUM_TRAINING_SAMPLES:NUM_TRAINING_SAMPLES + NUM_VAL_SAMPLES, :]
    y_val_as_values= TARGET_ARRAY_AS_VALUES[NUM_TRAINING_SAMPLES:NUM_TRAINING_SAMPLES + NUM_VAL_SAMPLES, :]
    y_val_as_deltas= TARGET_ARRAY_AS_DELTAS[NUM_TRAINING_SAMPLES:NUM_TRAINING_SAMPLES + NUM_VAL_SAMPLES, :]


    x_test= FEATURE_ARRAY[NUM_TRAINING_SAMPLES + NUM_VAL_SAMPLES:,:]
    y_test_as_values= TARGET_ARRAY_AS_VALUES[NUM_TRAINING_SAMPLES + NUM_VAL_SAMPLES:,:]
    y_test_as_deltas= TARGET_ARRAY_AS_DELTAS[NUM_TRAINING_SAMPLES + NUM_VAL_SAMPLES:,:]

    print("\nx_train: shape:")
    print (x_train.shape)
    print("y_train_as_values: shape:")
    print (y_train_as_values.shape)
    print("y_train_as_deltas: shape:")
    print (y_train_as_deltas.shape)


    print("\nx_val shape:")
    print (x_val.shape)
    print("y_val_as_values shape:")
    print (y_val_as_values.shape)
    print("y_val_as_deltas shape:")
    print (y_val_as_deltas.shape)

    print("\nx_test shape:")
    print (x_test.shape)
    print("y_test_as_values shape:")
    print (y_test_as_values.shape)
    print("y_test_as_deltas shape:")
    print (y_test_as_deltas.shape)

    if VAL_PERCENTAGE==0 and TEST_PERCENTAGE ==0:
        return x_train, y_train_as_values,y_train_as_deltas

    if VAL_PERCENTAGE == 0:
        return x_train, x_test, y_train_as_values,y_train_as_deltas, y_test_as_values,y_test_as_deltas

    if TEST_PERCENTAGE == 0:
        return x_train, x_val, y_train_as_values,y_train_as_deltas, y_val_as_values, y_val_as_deltas




    return x_train, x_val, x_test, y_train_as_values,y_train_as_deltas, y_val_as_values, y_val_as_deltas, y_test_as_values,y_test_as_deltas

def normalize_train_and_test_data(x_train, x_test, feature_range=(0,1)):
    scaler=MinMaxScaler(feature_range)

    old_train_shape=x_train.shape

    old_test_shape=x_test.shape

    if len(old_train_shape) == 3:
        x_train=x_train.reshape(x_train.shape[0]*x_train.shape[1],x_train.shape[2])

        x_test=x_test.reshape(x_test.shape[0]*x_test.shape[1],x_test.shape[2])

    scaler.fit(x_train)
    print(scaler.data_max_)

    x_train=scaler.transform(x_train)

    x_test=scaler.transform(x_test)

    x_train=x_train.reshape(old_train_shape)

    x_test=x_test.reshape(old_test_shape)

    # print("Shape of x_train: %s" %str(x_train.shape))
    # print("Shape of x_val: %s" %str(x_val.shape))
    # print("Shape of x_test: %s" %str(x_test.shape))

    # sys.exit(0)

    return x_train, x_test, scaler

def normalize_single_train_only_data_v2_standard_option(x_train, SCALER_TYPE,feature_range=(0,1)):

    if SCALER_TYPE != "Standard":
        scaler=MinMaxScaler(feature_range)
    else:
        scaler = StandardScaler()

    old_train_shape=x_train.shape


    if len(old_train_shape) == 3:
        x_train=x_train.reshape(x_train.shape[0]*x_train.shape[1],x_train.shape[2])


    scaler.fit(x_train)
    if SCALER_TYPE != "Standard":
        print(scaler.data_max_)

    x_train=scaler.transform(x_train)


    x_train=x_train.reshape(old_train_shape)


    return x_train, scaler

def normalize_train_and_test_data_v2_standard_option(x_train, x_test, SCALER_TYPE,feature_range=(0,1)):

    if SCALER_TYPE != "Standard":
        scaler=MinMaxScaler(feature_range)
    else:
        scaler = StandardScaler()

    old_train_shape=x_train.shape

    old_test_shape=x_test.shape

    if len(old_train_shape) == 3:
        x_train=x_train.reshape(x_train.shape[0]*x_train.shape[1],x_train.shape[2])

        x_test=x_test.reshape(x_test.shape[0]*x_test.shape[1],x_test.shape[2])

    scaler.fit(x_train)
    if SCALER_TYPE != "Standard":
        print(scaler.data_max_)

    x_train=scaler.transform(x_train)

    x_test=scaler.transform(x_test)

    x_train=x_train.reshape(old_train_shape)

    x_test=x_test.reshape(old_test_shape)

    # print("Shape of x_train: %s" %str(x_train.shape))
    # print("Shape of x_val: %s" %str(x_val.shape))
    # print("Shape of x_test: %s" %str(x_test.shape))

    # sys.exit(0)

    return x_train, x_test, scaler

# def normalize_single_array_with_scaler_v2_standard_option(x,scaler):
#     old_shape=x.shape

#     #reshape for scalar form
#     x=x.reshape(x.shape[0]*x.shape[1],x.shape[2])


#     #fit scaler
#     x=scaler.transform(x)

#     #return to prev shape
#     x=x.reshape(old_shape)


#     return x



def normalize_data(x_train, x_val, x_test, feature_range=(0,1)):
    scaler=MinMaxScaler(feature_range)

    old_train_shape=x_train.shape
    old_val_shape=x_val.shape
    old_test_shape=x_test.shape

    # old_data_shape_dict={
    # "old_train_shape":old_train_shape,
    # "old_val_shape" : old_val_shape,
    # "old_test_shape":old_test_shape
    # }

    # x_train=x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2])
    # x_val=x_val.reshape(x_val.shape[0],x_val.shape[1]*x_val.shape[2])
    # x_test=x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2])

    if len(old_train_shape) == 3:
    	x_train=x_train.reshape(x_train.shape[0]*x_train.shape[1],x_train.shape[2])
    	x_val=x_val.reshape(x_val.shape[0]*x_val.shape[1],x_val.shape[2])
    	x_test=x_test.reshape(x_test.shape[0]*x_test.shape[1],x_test.shape[2])

    print(scaler.fit(x_train))
    print(scaler.data_max_)

    x_train=scaler.transform(x_train)
    x_val=scaler.transform(x_val)
    x_test=scaler.transform(x_test)

    x_train=x_train.reshape(old_train_shape)
    x_val=x_val.reshape(old_val_shape)
    x_test=x_test.reshape(old_test_shape)

    # print("Shape of x_train: %s" %str(x_train.shape))
    # print("Shape of x_val: %s" %str(x_val.shape))
    # print("Shape of x_test: %s" %str(x_test.shape))

    # sys.exit(0)

    return x_train, x_val, x_test, scaler

def reshape_y_test_and_pred_by_features(y_test,predictions):
    NUM_FEATURES_TO_PREDICT=y_test.shape[2]
    new_dim0=y_test.shape[0] * y_test.shape[1]
    new_dim1=NUM_FEATURES_TO_PREDICT

    y_test=y_test.reshape((new_dim0,new_dim1))
    predictions=predictions.reshape((new_dim0,new_dim1))

    y_test=np.transpose(y_test)
    predictions=np.transpose(predictions)
    print("New y_test shape: %s" %str(y_test.shape))
    print("New predictions shape: %s" %str(predictions.shape))
    return y_test,predictions

def reshape_y_array(y,tag):

    NUM_FEATURES_TO_PREDICT=y.shape[2]
    new_dim0=y.shape[0] * y.shape[1]
    new_dim1=NUM_FEATURES_TO_PREDICT
    y=y.reshape((new_dim0,new_dim1))
    y=np.transpose(y)
    print("New %s shape: %s" %(tag,str(y.shape)))
    return y


def reshape_y_test_and_model_pred_and_pseudo_pred_by_features(y_test,predictions,pseudo_predictions):
    NUM_FEATURES_TO_PREDICT=y_test.shape[2]
    new_dim0=y_test.shape[0] * y_test.shape[1]
    new_dim1=NUM_FEATURES_TO_PREDICT

    y_test=y_test.reshape((new_dim0,new_dim1))
    predictions=predictions.reshape((new_dim0,new_dim1))
    pseudo_predictions=pseudo_predictions.reshape((new_dim0,new_dim1))

    y_test=np.transpose(y_test)
    predictions=np.transpose(predictions)
    pseudo_predictions=np.transpose(pseudo_predictions)
    print("New y_test shape: %s" %str(y_test.shape))
    print("New predictions shape: %s" %str(predictions.shape))
    print("New pseudo_predictions shape: %s" %str(pseudo_predictions.shape))
    return y_test,predictions,pseudo_predictions

def properly_denormalize_2d_data(my_2d_array, scaler):
    print("Original 2d array shape: %s" %str(my_2d_array))
    my_2d_array = np.transpose(my_2d_array)

    print("2d array shape after transposing: %s" %str(my_2d_array))
    my_2d_array = scaler.inverse_transform(my_2d_array)
    my_2d_array = np.transpose(my_2d_array)

    print("2d array shape after inverse transforming and transposing: %s" %str(my_2d_array))
    return my_2d_array

def properly_normalize_2d_data(my_2d_array, scaler):
    print("Original 2d array shape: %s" %str(my_2d_array))
    my_2d_array = np.transpose(my_2d_array)

    print("2d array shape after transposing: %s" %str(my_2d_array))
    my_2d_array = scaler.transform(my_2d_array)
    my_2d_array = np.transpose(my_2d_array)

    print("2d array shape after inverse transforming and transposing: %s" %str(my_2d_array))
    return my_2d_array


# def properly_normalize_data(x_train, x_val, x_test, feature_range=(0,1)):
#     scaler=MinMaxScaler(feature_range)

#     old_train_shape=x_train.shape
#     old_val_shape=x_val.shape
#     old_test_shape=x_test.shape

#     # old_data_shape_dict={
#     # "old_train_shape":old_train_shape,
#     # "old_val_shape" : old_val_shape,
#     # "old_test_shape":old_test_shape
#     # }

#     x_train=x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2])
#     x_val=x_val.reshape(x_val.shape[0],x_val.shape[1]*x_val.shape[2])
#     x_test=x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2])

#     #samples over days, features
#     #features, feature count over day


#     # x_train=x_train.reshape(x_train.shape[0]*x_train.shape[1],x_train.shape[2])
#     # x_val=x_val.reshape(x_val.shape[0]*x_val.shape[1],x_val.shape[2])
#     # x_test=x_test.reshape(x_test.shape[0]*x_test.shape[1],x_test.shape[2])

#     #transpose for scaler
#     x_train=np.transpose(x_train)
#     x_val=np.transpose(x_val)
#     x_test=np.transpose(x_test)

#     print(scaler.fit(x_train))
#     print(scaler.data_max_)
#     print("scaler.data_max_.shape")
#     print(scaler.data_max_.shape)

#     x_train=scaler.transform(x_train)
#     x_val=scaler.transform(x_val)
#     x_test=scaler.transform(x_test)

#     #detranspose for scaler
#     x_train=np.transpose(x_train)
#     x_val=np.transpose(x_val)
#     x_test=np.transpose(x_test)

#     x_train=x_train.reshape(old_train_shape)
#     x_val=x_val.reshape(old_val_shape)
#     x_test=x_test.reshape(old_test_shape)

#     # print("Shape of x_train: %s" %str(x_train.shape))
#     # print("Shape of x_val: %s" %str(x_val.shape))
#     # print("Shape of x_test: %s" %str(x_test.shape))

#     # sys.exit(0)

#     return x_train, x_val, x_test, scaler

def properly_normalize_data(x_train, x_val, x_test, feature_range=(0,1)):
    scaler=MinMaxScaler(feature_range)

    old_train_shape=x_train.shape
    old_val_shape=x_val.shape
    old_test_shape=x_test.shape

    # old_data_shape_dict={
    # "old_train_shape":old_train_shape,
    # "old_val_shape" : old_val_shape,
    # "old_test_shape":old_test_shape
    # }

    # x_train=x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2])
    # x_val=x_val.reshape(x_val.shape[0],x_val.shape[1]*x_val.shape[2])
    # x_test=x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2])

    if len(old_train_shape) == 3:
        # x_train=x_train.reshape(x_train.shape[0]*x_train.shape[1],x_train.shape[2])
        # x_val=x_val.reshape(x_val.shape[0]*x_val.shape[1],x_val.shape[2])
        # x_test=x_test.reshape(x_test.shape[0]*x_test.shape[1],x_test.shape[2])
        x_train=x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2])
        x_val=x_val.reshape(x_val.shape[0],x_val.shape[1]*x_val.shape[2])
        x_test=x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2])

    print(scaler.fit(x_train))
    print(scaler.data_max_)

    x_train=scaler.transform(x_train)
    x_val=scaler.transform(x_val)
    x_test=scaler.transform(x_test)

    x_train=x_train.reshape(old_train_shape)
    x_val=x_val.reshape(old_val_shape)
    x_test=x_test.reshape(old_test_shape)

    # print("Shape of x_train: %s" %str(x_train.shape))
    # print("Shape of x_val: %s" %str(x_val.shape))
    # print("Shape of x_test: %s" %str(x_test.shape))

    # sys.exit(0)

    return x_train, x_val, x_test, scaler

def normalize_single_array_with_scaler(x,scaler):
    old_shape=x.shape

    #reshape for scalar form
    if len(old_shape) == 3:
        x=x.reshape(x.shape[0]*x.shape[1],x.shape[2])


    #fit scaler
    x=scaler.transform(x)

    #return to prev shape
    x=x.reshape(old_shape)


    return x

def save_scaler(scaler,SCALER_OUTPUT_FP):
    print("Saving scaler")
    joblib.dump(scaler, SCALER_OUTPUT_FP)
    print("Successfully saved scaler in %s."%SCALER_OUTPUT_FP)

def load_scaler(SCALER_INPUT_FP):
    scaler = joblib.load(SCALER_INPUT_FP)
    print("Successfully loaded scaler in %s."%SCALER_INPUT_FP)
    return scaler

def denormalize_data(x_train, x_val, x_test, scaler):
    old_train_shape=x_train.shape
    old_val_shape=x_val.shape
    old_test_shape=x_test.shape

    x_train=x_train.reshape(x_train.shape[0]*x_train.shape[1],x_train.shape[2])
    x_val=x_val.reshape(x_val.shape[0]*x_val.shape[1],x_val.shape[2])
    x_test=x_test.reshape(x_test.shape[0]*x_test.shape[1],x_test.shape[2])


    x_train=scaler.inverse_transform(x_train)
    x_val=scaler.inverse_transform(x_val)
    x_test=scaler.inverse_transform(x_test)

    x_train=x_train.reshape(old_train_shape)
    x_val=x_val.reshape(old_val_shape)
    x_test=x_test.reshape(old_test_shape)

    return x_train, x_val, x_test

def denormalize_single_array(data, scaler):
    old_data_shape=data.shape
    data=data.reshape(data.shape[0]*data.shape[1],data.shape[2])
    data=scaler.inverse_transform(data)
    data=data.reshape(old_data_shape)
    return data

def create_basic_lstm(NUM_LSTM_LAYERS,LEARNING_RATE,OPTIMIZER,INPUT_SHAPE,LSTM_UNITS,LOSS,NUM_OUTPUTS=1):
    model = Sequential()
    model.add(LSTM(LSTM_UNITS, return_sequences=True, input_shape=INPUT_SHAPE))
    for LAYER in range(NUM_LSTM_LAYERS-1):
        model.add(LSTM(LSTM_UNITS, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(NUM_OUTPUTS)))
    # model.add(Dense(NUM_OUTPUTS))
    model.compile(loss=LOSS, optimizer=OPTIMIZER)

    print(model.summary())
    return model

def create_output_dir(OUTPUT_DIR):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print("Created %s" %OUTPUT_DIR)
    else:
        print("%s already exists"%OUTPUT_DIR)

def screen_and_file_print(output_str,file_ptr):
    print(output_str)
    file_ptr.write("%s\n"%output_str)


def get_date_from_startdate_by_granularity(start_date, num_timesteps,GRANULARITY):
    if GRANULARITY=="D":
        return start_date + timedelta(days=num_timesteps)
    elif GRANULARITY=="H":
        return start_date + timedelta(hours=num_timesteps)
    elif GRANULARITY=="min":
        return start_date + timedelta(minutes=num_timesteps)
    else:
        return start_date + timedelta(seconds=num_timesteps)

def merge_dicts(dict1, dict2):
    new_dict=dict2.copy()
    for key,val in dict1.items():
        new_dict[key]=val
        return new_dict

def create_param_dict(BATCH_SIZE, EPOCHS,NUM_LSTM_LAYERS,LOSS,LEARNING_RATE,DECAY_RATE,OPTIMIZER_STR,LSTM_UNITS):
    PARAM_DICT={}
    PARAM_DICT["batch_size"]=BATCH_SIZE
    PARAM_DICT["epochs"]=EPOCHS
    PARAM_DICT["num_lstm_layers"]=NUM_LSTM_LAYERS
    PARAM_DICT["loss"]=LOSS
    PARAM_DICT["learning_rate"]=LEARNING_RATE
    PARAM_DICT["decay_rate"]=DECAY_RATE
    PARAM_DICT["optimizer"]=OPTIMIZER_STR
    PARAM_DICT["lstm_units_per_layer"]=LSTM_UNITS
    return PARAM_DICT

def create_param_dict2(BATCH_SIZE, EPOCHS,LOSS,LEARNING_RATE,DECAY_RATE,OPTIMIZER_STR,LSTM_UNITS):
    PARAM_DICT={}
    PARAM_DICT["batch_size"]=BATCH_SIZE
    PARAM_DICT["epochs"]=EPOCHS
    PARAM_DICT["loss"]=LOSS
    PARAM_DICT["learning_rate"]=LEARNING_RATE
    PARAM_DICT["decay_rate"]=DECAY_RATE
    PARAM_DICT["optimizer"]=OPTIMIZER_STR
    PARAM_DICT["lstm_units_per_layer"]=LSTM_UNITS
    return PARAM_DICT

# def create_general_info_dict(VERSION_TAG,ABBREV_TO_PARAM_DICT,ARRAY_TIME_INFO_DICT,start_date,end_date,GRANULARITY,EXPERIMENT_TAG,PARAM_DICT ):
#     if GRANULARITY=="D":
#         gran_str="days"
#     elif GRANULARITY=="H":
#         gran_str="hours"
#     elif GRANULARITY=="min":
#         gran_str="minutes"
#     else:
#         gran_str="seconds"

#     training_timesteps=ARRAY_TIME_INFO_DICT["x_train_shape"][0] * ARRAY_TIME_INFO_DICT["x_train_shape"][1]
#     training_start_date=start_date
#     training_end_date=get_date_from_startdate_by_granularity(training_start_date, training_timesteps,GRANULARITY)
#     # output_str="Training started at %s and lasted %d %ss."%(str(training_start_date), training_timesteps, gran_str)
#     # screen_and_file_print(output_str, f)

#     val_timesteps=ARRAY_TIME_INFO_DICT["x_val_shape"][0]* ARRAY_TIME_INFO_DICT["x_val_shape"][1]
#     val_start_date=get_date_from_startdate_by_granularity(training_end_date, 1,GRANULARITY)
#     val_end_date=get_date_from_startdate_by_granularity(val_start_date, val_timesteps,GRANULARITY)
#     # output_str="Validation started at %s and lasted %d %ss."%(str(val_start_date), val_timesteps, gran_str)
#     # screen_and_file_print(output_str, f)

#     test_timesteps=ARRAY_TIME_INFO_DICT["x_test_shape"][0]* ARRAY_TIME_INFO_DICT["x_test_shape"][1]
#     test_start_date=get_date_from_startdate_by_granularity(val_end_date, 1,GRANULARITY)
#     test_end_date=get_date_from_startdate_by_granularity(test_start_date, test_timesteps,GRANULARITY)
#     # output_str="Testing started at %s and lasted %d %ss."%(str(test_start_date), test_timesteps, gran_str)
#     # screen_and_file_print(output_str,f )

#     GEN_INFO_DICT=PARAM_DICT.copy()
#     GEN_INFO_DICT["granularity"]=gran_str
#     GEN_INFO_DICT["training_start_date"]=training_start_date
#     GEN_INFO_DICT["training_end_date"]=training_end_date



#     GEN_INFO_DICT["val_start_date"]=val_start_date
#     GEN_INFO_DICT["val_end_date"]=val_end_date

#     GEN_INFO_DICT["test_start_date"]=test_start_date
#     GEN_INFO_DICT["test_end_date"]=test_end_date

#     GEN_INFO_DICT["training_samples"]=ARRAY_TIME_INFO_DICT["x_train_shape"].shape[0]
#     GEN_INFO_DICT["training_timesteps_per_sample"]=ARRAY_TIME_INFO_DICT["x_train_shape"].shape[1]
#     GEN_INFO_DICT["training_features_per_timestep"]=ARRAY_TIME_INFO_DICT["x_train_shape"].shape[2]
#     GEN_INFO_DICT["training_total_features_per_sample"]=ARRAY_TIME_INFO_DICT["x_train_shape"].shape[0] * ARRAY_TIME_INFO_DICT["x_train_shape"].shape[1]
#     GEN_INFO_DICT["feature_timesteps_and_target_timesteps"]=EXPERIMENT_TAG+"-in-%s"%gran_str
#     GEN_INFO_DICT["x_training_keras_shape"]=ARRAY_TIME_INFO_DICT["x_train_shape"]
#     GEN_INFO_DICT["y_training_keras_shape"]=ARRAY_TIME_INFO_DICT["y_train_as_values_shape"]

#     GEN_INFO_DICT["val_samples"]=ARRAY_TIME_INFO_DICT["x_val_shape"].shape[0]
#     GEN_INFO_DICT["val_timesteps_per_sample"]=ARRAY_TIME_INFO_DICT["x_val_shape"].shape[1]
#     GEN_INFO_DICT["val_features_per_timestep"]=ARRAY_TIME_INFO_DICT["x_val_shape"].shape[2]
#     GEN_INFO_DICT["val_total_features_per_sample"]=ARRAY_TIME_INFO_DICT["x_val_shape"].shape[0] * ARRAY_TIME_INFO_DICT["x_val_shape"].shape[1]
#     GEN_INFO_DICT["x_val_keras_shape"]=ARRAY_TIME_INFO_DICT["x_val_shape"]
#     GEN_INFO_DICT["y_training_keras_shape"]=ARRAY_TIME_INFO_DICT["y_val_as_values_shape"]

#     GEN_INFO_DICT["test_samples"]=ARRAY_TIME_INFO_DICT["x_test_shape"].shape[0]
#     GEN_INFO_DICT["test_timesteps_per_sample"]=ARRAY_TIME_INFO_DICT["x_test_shape"].shape[1]
#     GEN_INFO_DICT["test_features_per_timestep"]=ARRAY_TIME_INFO_DICT["x_test_shape"].shape[2]
#     GEN_INFO_DICT["test_total_features_per_sample"]=ARRAY_TIME_INFO_DICT["x_test_shape"].shape[0] * ARRAY_TIME_INFO_DICT["x_test_shape"].shape[1]
#     GEN_INFO_DICT["test_keras_shape"]=ARRAY_TIME_INFO_DICT["x_test_shape"]
#     GEN_INFO_DICT["test_keras_shape"]=ARRAY_TIME_INFO_DICT["y_test_as_values_shape"]

#     return GEN_INFO_DICT

# def record_general_info()

def create_general_info_dict(VERSION_TAG,ABBREV_TO_PARAM_DICT,ARRAY_TIME_INFO_DICT,start_date,end_date,GRANULARITY,EXPERIMENT_TAG,PARAM_DICT,NUM_FEATURES_TO_PREDICT ):
    if GRANULARITY=="D":
        gran_str="days"
    elif GRANULARITY=="H":
        gran_str="hours"
    elif GRANULARITY=="min":
        gran_str="minutes"
    else:
        gran_str="seconds"

    ABBREV_TO_PARAM_DICT={
    "GRAN":"Granularity",
    "SD":"start_date",
    "ED":"end_date",
    "XS":"X_ARRAY_TIMESTEPS_IN_A_SEQUENCE",
    "YS":"Y_ARRAY_TIMESTEPS_IN_A_SEQUENCE",
    "BS":"Batch size",
    "E":"Epochs",
    # "NLL":"Number of LSTM Layers",
    "LOSS":"Loss",
    "VP": "Validation set percentage out of training set",
    "TP": "Test set percentage out of training set",
    "NF":"Number of features"
    }

    actual_timesteps_represented_in_x_train=int(ARRAY_TIME_INFO_DICT["x_train_shape"][2]/NUM_FEATURES_TO_PREDICT)
    actual_timesteps_represented_in_y_train=int(ARRAY_TIME_INFO_DICT["y_train_as_values_shape"][2]/NUM_FEATURES_TO_PREDICT)

    actual_timesteps_represented_in_x_val=int(ARRAY_TIME_INFO_DICT["x_val_shape"][2]/NUM_FEATURES_TO_PREDICT)
    actual_timesteps_represented_in_y_val=int(ARRAY_TIME_INFO_DICT["y_val_as_values_shape"][2]/NUM_FEATURES_TO_PREDICT)

    actual_timesteps_represented_in_x_test=int(ARRAY_TIME_INFO_DICT["x_test_shape"][2]/NUM_FEATURES_TO_PREDICT)
    actual_timesteps_represented_in_y_test=int(ARRAY_TIME_INFO_DICT["y_test_as_values_shape"][2]/NUM_FEATURES_TO_PREDICT)

    training_timesteps=ARRAY_TIME_INFO_DICT["x_train_shape"][0] * ARRAY_TIME_INFO_DICT["x_train_shape"][1] * actual_timesteps_represented_in_x_train

    #make start date object
    start_date_list=start_date.split("-")
    syear=int(start_date_list[0])
    smonth=int(start_date_list[1])
    sday=int(start_date_list[2])

    training_start_date=date(syear,smonth,sday)
    # training_end_date=get_date_from_startdate_by_granularity(training_start_date, training_timesteps,GRANULARITY)
    # output_str="Training started at %s and lasted %d %ss."%(str(training_start_date), training_timesteps, gran_str)
    # screen_and_file_print(output_str, f)

    val_timesteps=ARRAY_TIME_INFO_DICT["x_val_shape"][0]* ARRAY_TIME_INFO_DICT["x_val_shape"][1] *actual_timesteps_represented_in_x_val
    # val_start_date=get_date_from_startdate_by_granularity(training_end_date, 1,GRANULARITY)
    # val_end_date=get_date_from_startdate_by_granularity(val_start_date, val_timesteps,GRANULARITY)
    # output_str="Validation started at %s and lasted %d %ss."%(str(val_start_date), val_timesteps, gran_str)
    # screen_and_file_print(output_str, f)

    test_timesteps=ARRAY_TIME_INFO_DICT["x_test_shape"][0]* ARRAY_TIME_INFO_DICT["x_test_shape"][1] * actual_timesteps_represented_in_x_test
    # test_start_date=get_date_from_startdate_by_granularity(val_end_date, 1,GRANULARITY)
    # test_end_date=get_date_from_startdate_by_granularity(test_start_date, test_timesteps,GRANULARITY)
    # output_str="Testing started at %s and lasted %d %ss."%(str(test_start_date), test_timesteps, gran_str)
    # screen_and_file_print(output_str,f )

    GEN_INFO_DICT=PARAM_DICT.copy()
    GEN_INFO_DICT["granularity"]=gran_str
    # GEN_INFO_DICT["training_start_date"]=training_start_date
    # GEN_INFO_DICT["training_end_date"]=training_end_date
    # # GEN_INFO_DICT["ft_timesteps_and_target_timesteps"]=EXPERIMENT_TAG+"-in-%s"%gran_str

    # GEN_INFO_DICT["val_start_date"]=val_start_date
    # GEN_INFO_DICT["val_end_date"]=val_end_date

    # GEN_INFO_DICT["test_start_date"]=test_start_date
    # GEN_INFO_DICT["test_end_date"]=test_end_date

    GEN_INFO_DICT["training_samples"]=ARRAY_TIME_INFO_DICT["x_train_shape"][0]

    GEN_INFO_DICT["x_train_keras_shape"]=ARRAY_TIME_INFO_DICT["x_train_shape"]
    GEN_INFO_DICT["y_train_keras_shape"]=ARRAY_TIME_INFO_DICT["y_train_as_values_shape"]

    GEN_INFO_DICT["keras_x_train_timesteps_per_sample"]=ARRAY_TIME_INFO_DICT["x_train_shape"][1]
    GEN_INFO_DICT["keras_y_train_timesteps_per_sample"]=ARRAY_TIME_INFO_DICT["y_train_as_values_shape"][1]

    GEN_INFO_DICT["x_train_actual_timesteps"]=actual_timesteps_represented_in_x_train
    GEN_INFO_DICT["y_train_actual_timesteps"]= actual_timesteps_represented_in_y_train

    GEN_INFO_DICT["x_training_features_per_timestep"]=ARRAY_TIME_INFO_DICT["x_train_shape"][2]
    GEN_INFO_DICT["y_training_features_per_timestep"]=ARRAY_TIME_INFO_DICT["y_train_as_values_shape"][2]

    GEN_INFO_DICT["x_train_total_features_per_sample"]=ARRAY_TIME_INFO_DICT["x_train_shape"][1] * ARRAY_TIME_INFO_DICT["x_train_shape"][2]
    GEN_INFO_DICT["y_train_total_features_per_sample"]=ARRAY_TIME_INFO_DICT["y_train_as_values_shape"][1] * ARRAY_TIME_INFO_DICT["y_train_as_values_shape"][2]


    GEN_INFO_DICT["feature_timesteps_and_target_timesteps"]=EXPERIMENT_TAG+"-in-%s"%gran_str

    GEN_INFO_DICT["x_val_keras_shape"]=ARRAY_TIME_INFO_DICT["x_val_shape"]
    GEN_INFO_DICT["y_val_keras_shape"]=ARRAY_TIME_INFO_DICT["y_val_as_values_shape"]


    GEN_INFO_DICT["val_samples"]=ARRAY_TIME_INFO_DICT["x_val_shape"][0]

    GEN_INFO_DICT["keras_x_val_timesteps_per_sample"]=ARRAY_TIME_INFO_DICT["x_val_shape"][1]
    GEN_INFO_DICT["keras_y_val_timesteps_per_sample"]=ARRAY_TIME_INFO_DICT["y_val_as_values_shape"][1]

    GEN_INFO_DICT["x_val_actual_timesteps"]=actual_timesteps_represented_in_x_val
    GEN_INFO_DICT["y_val_actual_timesteps"]= actual_timesteps_represented_in_y_val

    GEN_INFO_DICT["x_val_features_per_timestep"]=ARRAY_TIME_INFO_DICT["x_val_shape"][2]
    GEN_INFO_DICT["y_val_features_per_timestep"]=ARRAY_TIME_INFO_DICT["y_val_as_values_shape"][2]

    GEN_INFO_DICT["x_val_total_features_per_sample"]=ARRAY_TIME_INFO_DICT["x_val_shape"][1] * ARRAY_TIME_INFO_DICT["x_val_shape"][2]
    GEN_INFO_DICT["y_val_total_features_per_sample"]=ARRAY_TIME_INFO_DICT["y_val_as_values_shape"][1] * ARRAY_TIME_INFO_DICT["y_val_as_values_shape"][2]

    GEN_INFO_DICT["x_val_keras_shape"]=ARRAY_TIME_INFO_DICT["x_val_shape"]
    GEN_INFO_DICT["y_val_keras_shape"]=ARRAY_TIME_INFO_DICT["y_val_as_values_shape"]


    GEN_INFO_DICT["val_samples"]=ARRAY_TIME_INFO_DICT["x_val_shape"][0]

    GEN_INFO_DICT["keras_x_val_timesteps_per_sample"]=ARRAY_TIME_INFO_DICT["x_val_shape"][1]
    GEN_INFO_DICT["keras_y_val_timesteps_per_sample"]=ARRAY_TIME_INFO_DICT["y_val_as_values_shape"][1]

    GEN_INFO_DICT["x_val_actual_timesteps"]=actual_timesteps_represented_in_x_val
    GEN_INFO_DICT["y_val_actual_timesteps"]= actual_timesteps_represented_in_y_val

    GEN_INFO_DICT["x_val_features_per_timestep"]=ARRAY_TIME_INFO_DICT["x_val_shape"][2]
    GEN_INFO_DICT["y_val_features_per_timestep"]=ARRAY_TIME_INFO_DICT["y_val_as_values_shape"][2]

    GEN_INFO_DICT["x_val_total_features_per_sample"]=ARRAY_TIME_INFO_DICT["x_val_shape"][1] * ARRAY_TIME_INFO_DICT["x_val_shape"][2]
    GEN_INFO_DICT["y_val_total_features_per_sample"]=ARRAY_TIME_INFO_DICT["y_val_as_values_shape"][1] * ARRAY_TIME_INFO_DICT["y_val_as_values_shape"][2]

    GEN_INFO_DICT["test_samples"]=ARRAY_TIME_INFO_DICT["x_test_shape"][0]

    GEN_INFO_DICT["keras_x_test_timesteps_per_sample"]=ARRAY_TIME_INFO_DICT["x_test_shape"][1]
    GEN_INFO_DICT["keras_y_test_timesteps_per_sample"]=ARRAY_TIME_INFO_DICT["y_test_as_values_shape"][1]

    GEN_INFO_DICT["x_test_actual_timesteps"]=actual_timesteps_represented_in_x_val
    GEN_INFO_DICT["y_test_actual_timesteps"]= actual_timesteps_represented_in_y_val

    GEN_INFO_DICT["x_test_features_per_timestep"]=ARRAY_TIME_INFO_DICT["x_test_shape"][2]
    GEN_INFO_DICT["y_test_features_per_timestep"]=ARRAY_TIME_INFO_DICT["y_test_as_values_shape"][2]

    GEN_INFO_DICT["x_test_total_features_per_sample"]=ARRAY_TIME_INFO_DICT["x_test_shape"][1] * ARRAY_TIME_INFO_DICT["x_test_shape"][2]
    GEN_INFO_DICT["y_test_total_features_per_sample"]=ARRAY_TIME_INFO_DICT["y_test_as_values_shape"][1] * ARRAY_TIME_INFO_DICT["y_test_as_values_shape"][2]

    return GEN_INFO_DICT

def create_general_info_dict_no_val(VERSION_TAG,ABBREV_TO_PARAM_DICT,ARRAY_TIME_INFO_DICT,start_date,end_date,GRANULARITY,EXPERIMENT_TAG,PARAM_DICT,NUM_FEATURES_TO_PREDICT ):
    if GRANULARITY=="D":
        gran_str="days"
    elif GRANULARITY=="H":
        gran_str="hours"
    elif GRANULARITY=="min":
        gran_str="minutes"
    else:
        gran_str="seconds"

    ABBREV_TO_PARAM_DICT={
    "GRAN":"Granularity",
    "SD":"start_date",
    "ED":"end_date",
    "XS":"X_ARRAY_TIMESTEPS_IN_A_SEQUENCE",
    "YS":"Y_ARRAY_TIMESTEPS_IN_A_SEQUENCE",
    "BS":"Batch size",
    "E":"Epochs",
    # "NLL":"Number of LSTM Layers",
    "LOSS":"Loss",
    "VP": "Validation set percentage out of training set",
    "TP": "Test set percentage out of training set",
    "NF":"Number of features"
    }

    actual_timesteps_represented_in_x_train=int(ARRAY_TIME_INFO_DICT["x_train_shape"][2]/NUM_FEATURES_TO_PREDICT)
    actual_timesteps_represented_in_y_train=int(ARRAY_TIME_INFO_DICT["y_train_as_values_shape"][2]/NUM_FEATURES_TO_PREDICT)

    # actual_timesteps_represented_in_x_val=int(ARRAY_TIME_INFO_DICT["x_val_shape"][2]/NUM_FEATURES_TO_PREDICT)
    # actual_timesteps_represented_in_y_val=int(ARRAY_TIME_INFO_DICT["y_val_as_values_shape"][2]/NUM_FEATURES_TO_PREDICT)

    actual_timesteps_represented_in_x_test=int(ARRAY_TIME_INFO_DICT["x_test_shape"][2]/NUM_FEATURES_TO_PREDICT)
    actual_timesteps_represented_in_y_test=int(ARRAY_TIME_INFO_DICT["y_test_as_values_shape"][2]/NUM_FEATURES_TO_PREDICT)

    training_timesteps=ARRAY_TIME_INFO_DICT["x_train_shape"][0] * ARRAY_TIME_INFO_DICT["x_train_shape"][1] * actual_timesteps_represented_in_x_train

    #make start date object
    start_date_list=start_date.split("-")
    syear=int(start_date_list[0])
    smonth=int(start_date_list[1])
    sday=int(start_date_list[2])

    training_start_date=date(syear,smonth,sday)
    # training_end_date=get_date_from_startdate_by_granularity(training_start_date, training_timesteps,GRANULARITY)
    # output_str="Training started at %s and lasted %d %ss."%(str(training_start_date), training_timesteps, gran_str)
    # screen_and_file_print(output_str, f)

    # val_timesteps=ARRAY_TIME_INFO_DICT["x_val_shape"][0]* ARRAY_TIME_INFO_DICT["x_val_shape"][1] *actual_timesteps_represented_in_x_val
    # val_start_date=get_date_from_startdate_by_granularity(training_end_date, 1,GRANULARITY)
    # val_end_date=get_date_from_startdate_by_granularity(val_start_date, val_timesteps,GRANULARITY)
    # output_str="Validation started at %s and lasted %d %ss."%(str(val_start_date), val_timesteps, gran_str)
    # screen_and_file_print(output_str, f)

    test_timesteps=ARRAY_TIME_INFO_DICT["x_test_shape"][0]* ARRAY_TIME_INFO_DICT["x_test_shape"][1] * actual_timesteps_represented_in_x_test
    # test_start_date=get_date_from_startdate_by_granularity(val_end_date, 1,GRANULARITY)
    # test_end_date=get_date_from_startdate_by_granularity(test_start_date, test_timesteps,GRANULARITY)
    # output_str="Testing started at %s and lasted %d %ss."%(str(test_start_date), test_timesteps, gran_str)
    # screen_and_file_print(output_str,f )

    GEN_INFO_DICT=PARAM_DICT.copy()
    GEN_INFO_DICT["granularity"]=gran_str
    # GEN_INFO_DICT["training_start_date"]=training_start_date
    # GEN_INFO_DICT["training_end_date"]=training_end_date
    # # GEN_INFO_DICT["ft_timesteps_and_target_timesteps"]=EXPERIMENT_TAG+"-in-%s"%gran_str

    # GEN_INFO_DICT["val_start_date"]=val_start_date
    # GEN_INFO_DICT["val_end_date"]=val_end_date

    # GEN_INFO_DICT["test_start_date"]=test_start_date
    # GEN_INFO_DICT["test_end_date"]=test_end_date

    GEN_INFO_DICT["training_samples"]=ARRAY_TIME_INFO_DICT["x_train_shape"][0]

    GEN_INFO_DICT["x_train_keras_shape"]=ARRAY_TIME_INFO_DICT["x_train_shape"]
    GEN_INFO_DICT["y_train_keras_shape"]=ARRAY_TIME_INFO_DICT["y_train_as_values_shape"]

    GEN_INFO_DICT["keras_x_train_timesteps_per_sample"]=ARRAY_TIME_INFO_DICT["x_train_shape"][1]
    GEN_INFO_DICT["keras_y_train_timesteps_per_sample"]=ARRAY_TIME_INFO_DICT["y_train_as_values_shape"][1]

    GEN_INFO_DICT["x_train_actual_timesteps"]=actual_timesteps_represented_in_x_train
    GEN_INFO_DICT["y_train_actual_timesteps"]= actual_timesteps_represented_in_y_train

    GEN_INFO_DICT["x_training_features_per_timestep"]=ARRAY_TIME_INFO_DICT["x_train_shape"][2]
    GEN_INFO_DICT["y_training_features_per_timestep"]=ARRAY_TIME_INFO_DICT["y_train_as_values_shape"][2]

    GEN_INFO_DICT["x_train_total_features_per_sample"]=ARRAY_TIME_INFO_DICT["x_train_shape"][1] * ARRAY_TIME_INFO_DICT["x_train_shape"][2]
    GEN_INFO_DICT["y_train_total_features_per_sample"]=ARRAY_TIME_INFO_DICT["y_train_as_values_shape"][1] * ARRAY_TIME_INFO_DICT["y_train_as_values_shape"][2]


    GEN_INFO_DICT["feature_timesteps_and_target_timesteps"]=EXPERIMENT_TAG+"-in-%s"%gran_str

    # GEN_INFO_DICT["x_val_keras_shape"]=ARRAY_TIME_INFO_DICT["x_val_shape"]
    # GEN_INFO_DICT["y_val_keras_shape"]=ARRAY_TIME_INFO_DICT["y_val_as_values_shape"]


    # GEN_INFO_DICT["val_samples"]=ARRAY_TIME_INFO_DICT["x_val_shape"][0]

    # GEN_INFO_DICT["keras_x_val_timesteps_per_sample"]=ARRAY_TIME_INFO_DICT["x_val_shape"][1]
    # GEN_INFO_DICT["keras_y_val_timesteps_per_sample"]=ARRAY_TIME_INFO_DICT["y_val_as_values_shape"][1]

    # GEN_INFO_DICT["x_val_actual_timesteps"]=actual_timesteps_represented_in_x_val
    # GEN_INFO_DICT["y_val_actual_timesteps"]= actual_timesteps_represented_in_y_val

    # GEN_INFO_DICT["x_val_features_per_timestep"]=ARRAY_TIME_INFO_DICT["x_val_shape"][2]
    # GEN_INFO_DICT["y_val_features_per_timestep"]=ARRAY_TIME_INFO_DICT["y_val_as_values_shape"][2]

    # GEN_INFO_DICT["x_val_total_features_per_sample"]=ARRAY_TIME_INFO_DICT["x_val_shape"][1] * ARRAY_TIME_INFO_DICT["x_val_shape"][2]
    # GEN_INFO_DICT["y_val_total_features_per_sample"]=ARRAY_TIME_INFO_DICT["y_val_as_values_shape"][1] * ARRAY_TIME_INFO_DICT["y_val_as_values_shape"][2]

    # GEN_INFO_DICT["x_val_keras_shape"]=ARRAY_TIME_INFO_DICT["x_val_shape"]
    # GEN_INFO_DICT["y_val_keras_shape"]=ARRAY_TIME_INFO_DICT["y_val_as_values_shape"]


    # GEN_INFO_DICT["val_samples"]=ARRAY_TIME_INFO_DICT["x_val_shape"][0]

    # GEN_INFO_DICT["keras_x_val_timesteps_per_sample"]=ARRAY_TIME_INFO_DICT["x_val_shape"][1]
    # GEN_INFO_DICT["keras_y_val_timesteps_per_sample"]=ARRAY_TIME_INFO_DICT["y_val_as_values_shape"][1]

    # GEN_INFO_DICT["x_val_actual_timesteps"]=actual_timesteps_represented_in_x_val
    # GEN_INFO_DICT["y_val_actual_timesteps"]= actual_timesteps_represented_in_y_val

    # GEN_INFO_DICT["x_val_features_per_timestep"]=ARRAY_TIME_INFO_DICT["x_val_shape"][2]
    # GEN_INFO_DICT["y_val_features_per_timestep"]=ARRAY_TIME_INFO_DICT["y_val_as_values_shape"][2]

    # GEN_INFO_DICT["x_val_total_features_per_sample"]=ARRAY_TIME_INFO_DICT["x_val_shape"][1] * ARRAY_TIME_INFO_DICT["x_val_shape"][2]
    # GEN_INFO_DICT["y_val_total_features_per_sample"]=ARRAY_TIME_INFO_DICT["y_val_as_values_shape"][1] * ARRAY_TIME_INFO_DICT["y_val_as_values_shape"][2]

    GEN_INFO_DICT["test_samples"]=ARRAY_TIME_INFO_DICT["x_test_shape"][0]

    GEN_INFO_DICT["keras_x_test_timesteps_per_sample"]=ARRAY_TIME_INFO_DICT["x_test_shape"][1]
    GEN_INFO_DICT["keras_y_test_timesteps_per_sample"]=ARRAY_TIME_INFO_DICT["y_test_as_values_shape"][1]

    # GEN_INFO_DICT["x_test_actual_timesteps"]=actual_timesteps_represented_in_x_val
    # GEN_INFO_DICT["y_test_actual_timesteps"]= actual_timesteps_represented_in_y_val

    GEN_INFO_DICT["x_test_features_per_timestep"]=ARRAY_TIME_INFO_DICT["x_test_shape"][2]
    GEN_INFO_DICT["y_test_features_per_timestep"]=ARRAY_TIME_INFO_DICT["y_test_as_values_shape"][2]

    GEN_INFO_DICT["x_test_total_features_per_sample"]=ARRAY_TIME_INFO_DICT["x_test_shape"][1] * ARRAY_TIME_INFO_DICT["x_test_shape"][2]
    GEN_INFO_DICT["y_test_total_features_per_sample"]=ARRAY_TIME_INFO_DICT["y_test_as_values_shape"][1] * ARRAY_TIME_INFO_DICT["y_test_as_values_shape"][2]

    return GEN_INFO_DICT

def create_general_info_dict_without_abbrev_dict_input(VERSION_TAG,ARRAY_TIME_INFO_DICT,start_date,end_date,GRANULARITY,EXPERIMENT_TAG,PARAM_DICT,NUM_FEATURES_TO_PREDICT ):
    if GRANULARITY=="D":
        gran_str="days"
    elif GRANULARITY=="H":
        gran_str="hours"
    elif GRANULARITY=="min":
        gran_str="minutes"
    else:
        gran_str="seconds"

    ABBREV_TO_PARAM_DICT={
    "GRAN":"Granularity",
    "SD":"start_date",
    "ED":"end_date",
    "XS":"X_ARRAY_TIMESTEPS_IN_A_SEQUENCE",
    "YS":"Y_ARRAY_TIMESTEPS_IN_A_SEQUENCE",
    "BS":"Batch size",
    "E":"Epochs",
    "NLL":"Number of LSTM Layers",
    "LOSS":"Loss",
    "VP": "Validation set percentage out of training set",
    "TP": "Test set percentage out of training set",
    "NF":"Number of features"
    }

    actual_timesteps_represented_in_x_train=int(ARRAY_TIME_INFO_DICT["x_train_shape"][2]/NUM_FEATURES_TO_PREDICT)
    actual_timesteps_represented_in_y_train=int(ARRAY_TIME_INFO_DICT["y_train_as_values_shape"][2]/NUM_FEATURES_TO_PREDICT)

    actual_timesteps_represented_in_x_val=int(ARRAY_TIME_INFO_DICT["x_val_shape"][2]/NUM_FEATURES_TO_PREDICT)
    actual_timesteps_represented_in_y_val=int(ARRAY_TIME_INFO_DICT["y_val_as_values_shape"][2]/NUM_FEATURES_TO_PREDICT)

    actual_timesteps_represented_in_x_test=int(ARRAY_TIME_INFO_DICT["x_test_shape"][2]/NUM_FEATURES_TO_PREDICT)
    actual_timesteps_represented_in_y_test=int(ARRAY_TIME_INFO_DICT["y_test_as_values_shape"][2]/NUM_FEATURES_TO_PREDICT)

    training_timesteps=ARRAY_TIME_INFO_DICT["x_train_shape"][0] * ARRAY_TIME_INFO_DICT["x_train_shape"][1] * actual_timesteps_represented_in_x_train

    #make start date object
    start_date_list=start_date.split("-")
    syear=int(start_date_list[0])
    smonth=int(start_date_list[1])
    sday=int(start_date_list[2])

    training_start_date=date(syear,smonth,sday)
    # training_end_date=get_date_from_startdate_by_granularity(training_start_date, training_timesteps,GRANULARITY)
    # output_str="Training started at %s and lasted %d %ss."%(str(training_start_date), training_timesteps, gran_str)
    # screen_and_file_print(output_str, f)

    val_timesteps=ARRAY_TIME_INFO_DICT["x_val_shape"][0]* ARRAY_TIME_INFO_DICT["x_val_shape"][1] *actual_timesteps_represented_in_x_val
    # val_start_date=get_date_from_startdate_by_granularity(training_end_date, 1,GRANULARITY)
    # val_end_date=get_date_from_startdate_by_granularity(val_start_date, val_timesteps,GRANULARITY)
    # output_str="Validation started at %s and lasted %d %ss."%(str(val_start_date), val_timesteps, gran_str)
    # screen_and_file_print(output_str, f)

    test_timesteps=ARRAY_TIME_INFO_DICT["x_test_shape"][0]* ARRAY_TIME_INFO_DICT["x_test_shape"][1] * actual_timesteps_represented_in_x_test
    # test_start_date=get_date_from_startdate_by_granularity(val_end_date, 1,GRANULARITY)
    # test_end_date=get_date_from_startdate_by_granularity(test_start_date, test_timesteps,GRANULARITY)
    # output_str="Testing started at %s and lasted %d %ss."%(str(test_start_date), test_timesteps, gran_str)
    # screen_and_file_print(output_str,f )

    GEN_INFO_DICT=PARAM_DICT.copy()
    GEN_INFO_DICT["granularity"]=gran_str
    # GEN_INFO_DICT["training_start_date"]=training_start_date
    # GEN_INFO_DICT["training_end_date"]=training_end_date
    # # GEN_INFO_DICT["ft_timesteps_and_target_timesteps"]=EXPERIMENT_TAG+"-in-%s"%gran_str

    # GEN_INFO_DICT["val_start_date"]=val_start_date
    # GEN_INFO_DICT["val_end_date"]=val_end_date

    # GEN_INFO_DICT["test_start_date"]=test_start_date
    # GEN_INFO_DICT["test_end_date"]=test_end_date

    GEN_INFO_DICT["training_samples"]=ARRAY_TIME_INFO_DICT["x_train_shape"][0]

    GEN_INFO_DICT["x_train_keras_shape"]=ARRAY_TIME_INFO_DICT["x_train_shape"]
    GEN_INFO_DICT["y_train_keras_shape"]=ARRAY_TIME_INFO_DICT["y_train_as_values_shape"]

    GEN_INFO_DICT["keras_x_train_timesteps_per_sample"]=ARRAY_TIME_INFO_DICT["x_train_shape"][1]
    GEN_INFO_DICT["keras_y_train_timesteps_per_sample"]=ARRAY_TIME_INFO_DICT["y_train_as_values_shape"][1]

    GEN_INFO_DICT["x_train_actual_timesteps"]=actual_timesteps_represented_in_x_train
    GEN_INFO_DICT["y_train_actual_timesteps"]= actual_timesteps_represented_in_y_train

    GEN_INFO_DICT["x_training_features_per_timestep"]=ARRAY_TIME_INFO_DICT["x_train_shape"][2]
    GEN_INFO_DICT["y_training_features_per_timestep"]=ARRAY_TIME_INFO_DICT["y_train_as_values_shape"][2]

    GEN_INFO_DICT["x_train_total_features_per_sample"]=ARRAY_TIME_INFO_DICT["x_train_shape"][1] * ARRAY_TIME_INFO_DICT["x_train_shape"][2]
    GEN_INFO_DICT["y_train_total_features_per_sample"]=ARRAY_TIME_INFO_DICT["y_train_as_values_shape"][1] * ARRAY_TIME_INFO_DICT["y_train_as_values_shape"][2]


    GEN_INFO_DICT["feature_timesteps_and_target_timesteps"]=EXPERIMENT_TAG+"-in-%s"%gran_str

    GEN_INFO_DICT["x_val_keras_shape"]=ARRAY_TIME_INFO_DICT["x_val_shape"]
    GEN_INFO_DICT["y_val_keras_shape"]=ARRAY_TIME_INFO_DICT["y_val_as_values_shape"]


    GEN_INFO_DICT["val_samples"]=ARRAY_TIME_INFO_DICT["x_val_shape"][0]

    GEN_INFO_DICT["keras_x_val_timesteps_per_sample"]=ARRAY_TIME_INFO_DICT["x_val_shape"][1]
    GEN_INFO_DICT["keras_y_val_timesteps_per_sample"]=ARRAY_TIME_INFO_DICT["y_val_as_values_shape"][1]

    GEN_INFO_DICT["x_val_actual_timesteps"]=actual_timesteps_represented_in_x_val
    GEN_INFO_DICT["y_val_actual_timesteps"]= actual_timesteps_represented_in_y_val

    GEN_INFO_DICT["x_val_features_per_timestep"]=ARRAY_TIME_INFO_DICT["x_val_shape"][2]
    GEN_INFO_DICT["y_val_features_per_timestep"]=ARRAY_TIME_INFO_DICT["y_val_as_values_shape"][2]

    GEN_INFO_DICT["x_val_total_features_per_sample"]=ARRAY_TIME_INFO_DICT["x_val_shape"][1] * ARRAY_TIME_INFO_DICT["x_val_shape"][2]
    GEN_INFO_DICT["y_val_total_features_per_sample"]=ARRAY_TIME_INFO_DICT["y_val_as_values_shape"][1] * ARRAY_TIME_INFO_DICT["y_val_as_values_shape"][2]

    GEN_INFO_DICT["x_val_keras_shape"]=ARRAY_TIME_INFO_DICT["x_val_shape"]
    GEN_INFO_DICT["y_val_keras_shape"]=ARRAY_TIME_INFO_DICT["y_val_as_values_shape"]


    GEN_INFO_DICT["val_samples"]=ARRAY_TIME_INFO_DICT["x_val_shape"][0]

    GEN_INFO_DICT["keras_x_val_timesteps_per_sample"]=ARRAY_TIME_INFO_DICT["x_val_shape"][1]
    GEN_INFO_DICT["keras_y_val_timesteps_per_sample"]=ARRAY_TIME_INFO_DICT["y_val_as_values_shape"][1]

    GEN_INFO_DICT["x_val_actual_timesteps"]=actual_timesteps_represented_in_x_val
    GEN_INFO_DICT["y_val_actual_timesteps"]= actual_timesteps_represented_in_y_val

    GEN_INFO_DICT["x_val_features_per_timestep"]=ARRAY_TIME_INFO_DICT["x_val_shape"][2]
    GEN_INFO_DICT["y_val_features_per_timestep"]=ARRAY_TIME_INFO_DICT["y_val_as_values_shape"][2]

    GEN_INFO_DICT["x_val_total_features_per_sample"]=ARRAY_TIME_INFO_DICT["x_val_shape"][1] * ARRAY_TIME_INFO_DICT["x_val_shape"][2]
    GEN_INFO_DICT["y_val_total_features_per_sample"]=ARRAY_TIME_INFO_DICT["y_val_as_values_shape"][1] * ARRAY_TIME_INFO_DICT["y_val_as_values_shape"][2]

    GEN_INFO_DICT["test_samples"]=ARRAY_TIME_INFO_DICT["x_test_shape"][0]

    GEN_INFO_DICT["keras_x_test_timesteps_per_sample"]=ARRAY_TIME_INFO_DICT["x_test_shape"][1]
    GEN_INFO_DICT["keras_y_test_timesteps_per_sample"]=ARRAY_TIME_INFO_DICT["y_test_as_values_shape"][1]

    GEN_INFO_DICT["x_test_actual_timesteps"]=actual_timesteps_represented_in_x_val
    GEN_INFO_DICT["y_test_actual_timesteps"]= actual_timesteps_represented_in_y_val

    GEN_INFO_DICT["x_test_features_per_timestep"]=ARRAY_TIME_INFO_DICT["x_test_shape"][2]
    GEN_INFO_DICT["y_test_features_per_timestep"]=ARRAY_TIME_INFO_DICT["y_test_as_values_shape"][2]

    GEN_INFO_DICT["x_test_total_features_per_sample"]=ARRAY_TIME_INFO_DICT["x_test_shape"][1] * ARRAY_TIME_INFO_DICT["x_test_shape"][2]
    GEN_INFO_DICT["y_test_total_features_per_sample"]=ARRAY_TIME_INFO_DICT["y_test_as_values_shape"][1] * ARRAY_TIME_INFO_DICT["y_test_as_values_shape"][2]

    return GEN_INFO_DICT

def FIXED_create_general_info_dict_without_abbrev_dict_input(VERSION_TAG,ARRAY_TIME_INFO_DICT,start_date,end_date,GRANULARITY,EXPERIMENT_TAG,PARAM_DICT,NUM_FEATURES_TO_PREDICT ):
    if GRANULARITY=="D":
        gran_str="days"
    elif GRANULARITY=="H":
        gran_str="hours"
    elif GRANULARITY=="min":
        gran_str="minutes"
    else:
        gran_str="seconds"

    ABBREV_TO_PARAM_DICT={
    "GRAN":"Granularity",
    "SD":"start_date",
    "ED":"end_date",
    "XS":"X_ARRAY_TIMESTEPS_IN_A_SEQUENCE",
    "YS":"Y_ARRAY_TIMESTEPS_IN_A_SEQUENCE",
    "BS":"Batch size",
    "E":"Epochs",
    "NLL":"Number of LSTM Layers",
    "LOSS":"Loss",
    "VP": "Validation set percentage out of training set",
    "TP": "Test set percentage out of training set",
    "NF":"Number of features"
    }

    actual_timesteps_represented_in_x_train=int(ARRAY_TIME_INFO_DICT["x_train_shape"][2]/NUM_FEATURES_TO_PREDICT)
    actual_timesteps_represented_in_y_train=int(ARRAY_TIME_INFO_DICT["y_train_as_values_shape"][2]/NUM_FEATURES_TO_PREDICT)

    actual_timesteps_represented_in_x_val=int(ARRAY_TIME_INFO_DICT["x_val_shape"][2]/NUM_FEATURES_TO_PREDICT)
    actual_timesteps_represented_in_y_val=int(ARRAY_TIME_INFO_DICT["y_val_as_values_shape"][2]/NUM_FEATURES_TO_PREDICT)

    actual_timesteps_represented_in_x_test=int(ARRAY_TIME_INFO_DICT["x_test_shape"][2]/NUM_FEATURES_TO_PREDICT)
    actual_timesteps_represented_in_y_test=int(ARRAY_TIME_INFO_DICT["y_test_as_values_shape"][2]/NUM_FEATURES_TO_PREDICT)

    training_timesteps=ARRAY_TIME_INFO_DICT["x_train_shape"][0] * ARRAY_TIME_INFO_DICT["x_train_shape"][1] * actual_timesteps_represented_in_x_train

    #make start date object
    start_date_list=start_date.split("-")
    syear=int(start_date_list[0])
    smonth=int(start_date_list[1])
    sday=int(start_date_list[2])

    training_start_date=date(syear,smonth,sday)
    # training_end_date=get_date_from_startdate_by_granularity(training_start_date, training_timesteps,GRANULARITY)
    # output_str="Training started at %s and lasted %d %ss."%(str(training_start_date), training_timesteps, gran_str)
    # screen_and_file_print(output_str, f)

    val_timesteps=ARRAY_TIME_INFO_DICT["x_val_shape"][0]* ARRAY_TIME_INFO_DICT["x_val_shape"][1] *actual_timesteps_represented_in_x_val
    # val_start_date=get_date_from_startdate_by_granularity(training_end_date, 1,GRANULARITY)
    # val_end_date=get_date_from_startdate_by_granularity(val_start_date, val_timesteps,GRANULARITY)
    # output_str="Validation started at %s and lasted %d %ss."%(str(val_start_date), val_timesteps, gran_str)
    # screen_and_file_print(output_str, f)

    test_timesteps=ARRAY_TIME_INFO_DICT["x_test_shape"][0]* ARRAY_TIME_INFO_DICT["x_test_shape"][1] * actual_timesteps_represented_in_x_test
    # test_start_date=get_date_from_startdate_by_granularity(val_end_date, 1,GRANULARITY)
    # test_end_date=get_date_from_startdate_by_granularity(test_start_date, test_timesteps,GRANULARITY)
    # output_str="Testing started at %s and lasted %d %ss."%(str(test_start_date), test_timesteps, gran_str)
    # screen_and_file_print(output_str,f )

    GEN_INFO_DICT=PARAM_DICT.copy()
    GEN_INFO_DICT["granularity"]=gran_str
    # GEN_INFO_DICT["training_start_date"]=training_start_date
    # GEN_INFO_DICT["training_end_date"]=training_end_date
    # # GEN_INFO_DICT["ft_timesteps_and_target_timesteps"]=EXPERIMENT_TAG+"-in-%s"%gran_str

    # GEN_INFO_DICT["val_start_date"]=val_start_date
    # GEN_INFO_DICT["val_end_date"]=val_end_date

    # GEN_INFO_DICT["test_start_date"]=test_start_date
    # GEN_INFO_DICT["test_end_date"]=test_end_date

    GEN_INFO_DICT["training_samples"]=ARRAY_TIME_INFO_DICT["x_train_shape"][0]

    GEN_INFO_DICT["x_train_keras_shape"]=ARRAY_TIME_INFO_DICT["x_train_shape"]
    GEN_INFO_DICT["y_train_keras_shape"]=ARRAY_TIME_INFO_DICT["y_train_as_values_shape"]

    GEN_INFO_DICT["keras_x_train_timesteps_per_sample"]=ARRAY_TIME_INFO_DICT["x_train_shape"][1]
    GEN_INFO_DICT["keras_y_train_timesteps_per_sample"]=ARRAY_TIME_INFO_DICT["y_train_as_values_shape"][1]

    GEN_INFO_DICT["x_train_actual_timesteps"]=actual_timesteps_represented_in_x_train
    GEN_INFO_DICT["y_train_actual_timesteps"]= actual_timesteps_represented_in_y_train

    GEN_INFO_DICT["x_training_features_per_timestep"]=ARRAY_TIME_INFO_DICT["x_train_shape"][2]
    GEN_INFO_DICT["y_training_features_per_timestep"]=ARRAY_TIME_INFO_DICT["y_train_as_values_shape"][2]

    GEN_INFO_DICT["x_train_total_features_per_sample"]=ARRAY_TIME_INFO_DICT["x_train_shape"][1] * ARRAY_TIME_INFO_DICT["x_train_shape"][2]
    GEN_INFO_DICT["y_train_total_features_per_sample"]=ARRAY_TIME_INFO_DICT["y_train_as_values_shape"][1] * ARRAY_TIME_INFO_DICT["y_train_as_values_shape"][2]


    GEN_INFO_DICT["feature_timesteps_and_target_timesteps"]=EXPERIMENT_TAG+"-in-%s"%gran_str

    GEN_INFO_DICT["x_val_keras_shape"]=ARRAY_TIME_INFO_DICT["x_val_shape"]
    GEN_INFO_DICT["y_val_keras_shape"]=ARRAY_TIME_INFO_DICT["y_val_as_values_shape"]


    GEN_INFO_DICT["val_samples"]=ARRAY_TIME_INFO_DICT["x_val_shape"][0]

    GEN_INFO_DICT["keras_x_val_timesteps_per_sample"]=ARRAY_TIME_INFO_DICT["x_val_shape"][1]
    GEN_INFO_DICT["keras_y_val_timesteps_per_sample"]=ARRAY_TIME_INFO_DICT["y_val_as_values_shape"][1]

    GEN_INFO_DICT["x_val_actual_timesteps"]=actual_timesteps_represented_in_x_val
    GEN_INFO_DICT["y_val_actual_timesteps"]= actual_timesteps_represented_in_y_val

    GEN_INFO_DICT["x_val_features_per_timestep"]=ARRAY_TIME_INFO_DICT["x_val_shape"][2]
    GEN_INFO_DICT["y_val_features_per_timestep"]=ARRAY_TIME_INFO_DICT["y_val_as_values_shape"][2]

    GEN_INFO_DICT["x_val_total_features_per_sample"]=ARRAY_TIME_INFO_DICT["x_val_shape"][1] * ARRAY_TIME_INFO_DICT["x_val_shape"][2]
    GEN_INFO_DICT["y_val_total_features_per_sample"]=ARRAY_TIME_INFO_DICT["y_val_as_values_shape"][1] * ARRAY_TIME_INFO_DICT["y_val_as_values_shape"][2]

    GEN_INFO_DICT["x_val_keras_shape"]=ARRAY_TIME_INFO_DICT["x_val_shape"]
    GEN_INFO_DICT["y_val_keras_shape"]=ARRAY_TIME_INFO_DICT["y_val_as_values_shape"]


    GEN_INFO_DICT["val_samples"]=ARRAY_TIME_INFO_DICT["x_val_shape"][0]

    GEN_INFO_DICT["keras_x_val_timesteps_per_sample"]=ARRAY_TIME_INFO_DICT["x_val_shape"][1]
    GEN_INFO_DICT["keras_y_val_timesteps_per_sample"]=ARRAY_TIME_INFO_DICT["y_val_as_values_shape"][1]

    GEN_INFO_DICT["x_val_actual_timesteps"]=actual_timesteps_represented_in_x_val
    GEN_INFO_DICT["y_val_actual_timesteps"]= actual_timesteps_represented_in_y_val

    GEN_INFO_DICT["x_val_features_per_timestep"]=ARRAY_TIME_INFO_DICT["x_val_shape"][2]
    GEN_INFO_DICT["y_val_features_per_timestep"]=ARRAY_TIME_INFO_DICT["y_val_as_values_shape"][2]

    GEN_INFO_DICT["x_val_total_features_per_sample"]=ARRAY_TIME_INFO_DICT["x_val_shape"][1] * ARRAY_TIME_INFO_DICT["x_val_shape"][2]
    GEN_INFO_DICT["y_val_total_features_per_sample"]=ARRAY_TIME_INFO_DICT["y_val_as_values_shape"][1] * ARRAY_TIME_INFO_DICT["y_val_as_values_shape"][2]

    GEN_INFO_DICT["test_samples"]=ARRAY_TIME_INFO_DICT["x_test_shape"][0]

    GEN_INFO_DICT["keras_x_test_timesteps_per_sample"]=ARRAY_TIME_INFO_DICT["x_test_shape"][1]
    GEN_INFO_DICT["keras_y_test_timesteps_per_sample"]=ARRAY_TIME_INFO_DICT["y_test_as_values_shape"][1]

    GEN_INFO_DICT["x_test_actual_timesteps"]=actual_timesteps_represented_in_x_val
    GEN_INFO_DICT["y_test_actual_timesteps"]= actual_timesteps_represented_in_y_val

    GEN_INFO_DICT["x_test_features_per_timestep"]=ARRAY_TIME_INFO_DICT["x_test_shape"][2]
    GEN_INFO_DICT["y_test_features_per_timestep"]=ARRAY_TIME_INFO_DICT["y_test_as_values_shape"][2]

    GEN_INFO_DICT["x_test_total_features_per_sample"]=ARRAY_TIME_INFO_DICT["x_test_shape"][1] * ARRAY_TIME_INFO_DICT["x_test_shape"][2]
    GEN_INFO_DICT["y_test_total_features_per_sample"]=ARRAY_TIME_INFO_DICT["y_test_as_values_shape"][1] * ARRAY_TIME_INFO_DICT["y_test_as_values_shape"][2]

    return GEN_INFO_DICT


def record_general_dict(GEN_INFO_DICT,MAIN_RESULT_DIR,ALL_FEATURES_TO_PREDICT,TIME_DICT):
    OUTPUT_FP=MAIN_RESULT_DIR+"All-Params.txt"
    f=open(OUTPUT_FP, "w+")
    for key,val in GEN_INFO_DICT.items():
        output_str="%s : %s"%(key,val)
        screen_and_file_print(output_str,f)

    output_str="\nAll features: "
    screen_and_file_print(output_str,f)
    for FEATURE in ALL_FEATURES_TO_PREDICT:
        screen_and_file_print(FEATURE,f)


    screen_and_file_print("\n----------Time Info-----------\n",f)
    for key,val in TIME_DICT.items():
        output_str="%s : %s"%(key,val)
        screen_and_file_print(output_str,f)



def record_params(OUTPUT_DIR, VERSION_TAG,ABBREV_TO_PARAM_DICT,ARRAY_TIME_INFO_DICT,start_date,end_date,GRANULARITY):
    # ARRAY_TIME_INFO_DICT["x_train_shape"]=x_train.shape
    # ARRAY_TIME_INFO_DICT["x_val_shape"]=x_val.shape
    # ARRAY_TIME_INFO_DICT["x_test_shape"]=x_test.shape

    # ARRAY_TIME_INFO_DICT["y_train_as_values_shape"]=y_train_as_values.shape
    # ARRAY_TIME_INFO_DICT["y_train_as_deltas_shape"]=y_train_as_deltas.shape
    # ARRAY_TIME_INFO_DICT["y_val_as_values_shape"]=y_val_as_values.shape
    # ARRAY_TIME_INFO_DICT["y_val_as_deltas_shape"]=y_val_as_deltas.shape
    # ARRAY_TIME_INFO_DICT["y_test_as_values_shape"]=y_test_as_values.shape
    # ARRAY_TIME_INFO_DICT["y_test_as_deltas_shape"]=y_test_as_deltas.shape
    if GRANULARITY=="D":
        gran_str="day"
    elif GRANULARITY=="H":
        gran_str="hour"
    elif GRANULARITY=="min":
        gran_str="minute"
    else:
        gran_str="second"

    OUTPUT_FP=OUTPUT_DIR+"Parameters.txt"

    f=open(OUTPUT_FP, "w+")
    VTAG_LIST=VERSION_TAG.split("_")
    print(VTAG_LIST)

    for SUB_STR in VTAG_LIST:
        SUB_STR_LIST=SUB_STR.split("-",1)
        print(SUB_STR_LIST)
        ABBREV=SUB_STR_LIST[0]
        PARAM_NAME=ABBREV_TO_PARAM_DICT[ABBREV]

        PARAM_VALUE=SUB_STR_LIST[1]
        output_str="%s: %s"%(PARAM_NAME, PARAM_VALUE)

        screen_and_file_print(output_str,f)

    for key,value in ARRAY_TIME_INFO_DICT.items():
        output_str="%s : %s" %(str(key), str(value))
        screen_and_file_print(output_str,f)

    training_timesteps=ARRAY_TIME_INFO_DICT["x_train_shape"][0] * ARRAY_TIME_INFO_DICT["x_train_shape"][1]
    training_start_date=start_date
    training_end_date=get_date_from_startdate_by_granularity(training_start_date, training_timesteps,GRANULARITY)
    output_str="Training started at %s and lasted %d %ss."%(str(training_start_date), training_timesteps, gran_str)
    screen_and_file_print(output_str, f)

    val_timesteps=ARRAY_TIME_INFO_DICT["x_val_shape"][0]* ARRAY_TIME_INFO_DICT["x_val_shape"][1]
    val_start_date=get_date_from_startdate_by_granularity(training_end_date, 1,GRANULARITY)
    val_end_date=get_date_from_startdate_by_granularity(val_start_date, val_timesteps,GRANULARITY)
    output_str="Validation started at %s and lasted %d %ss."%(str(val_start_date), val_timesteps, gran_str)
    screen_and_file_print(output_str, f)

    test_timesteps=ARRAY_TIME_INFO_DICT["x_test_shape"][0]* ARRAY_TIME_INFO_DICT["x_test_shape"][1]
    test_start_date=get_date_from_startdate_by_granularity(val_end_date, 1,GRANULARITY)
    test_end_date=get_date_from_startdate_by_granularity(test_start_date, test_timesteps,GRANULARITY)
    output_str="Testing started at %s and lasted %d %ss."%(str(test_start_date), test_timesteps, gran_str)
    screen_and_file_print(output_str,f )



def create_lstm_with_var_layers(LEARNING_RATE,OPTIMIZER,INPUT_SHAPE,LSTM_UNITS,LOSS,NUM_LAYERS,NUM_OUTPUTS=1):
    model = Sequential()
    model.add(LSTM(LSTM_UNITS, return_sequences=True, input_shape=INPUT_SHAPE))
    for i in range(NUM_LAYERS-1):
        model.add(LSTM(LSTM_UNITS, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(Dense(NUM_OUTPUTS))
    model.compile(loss=LOSS, optimizer=OPTIMIZER)

    print(model.summary())
    return model



# def create_non_sequential_lstm(LEARNING_RATE,OPTIMIZER,INPUT_SHAPE,LSTM_UNITS,LOSS,NUM_LAYERS,NUM_OUTPUTS=1):
#     inputs1 = Input(shape=INPUT_SHAPE)
#     lstm_layer1 = LSTM(LSTM_UNITS, return_sequences=True)(INPUT_SHAPE)
#     model=Model(inputs=lstm_layer1, outputs=lstm_layer1)
#     prev_lstm_layer=lstm_layer1
#     for i in range(NUM_LAYERS-1):
#         cur_lstm_layer=


def smape_old(A, F):
    ACTUAL_LIST=A.flatten().tolist()
    FORECAST_LIST=F.flatten().tolist()

    A=[]
    F=[]
    # NUM_DOUBLE_ZEROS=0
    # print("Getting smape...")
    for actual,forecast in zip(ACTUAL_LIST,FORECAST_LIST):
        if (actual ==0) and (forecast == 0):
            A.append(1)
            F.append(1)
        else:
            A.append(actual)
            F.append(forecast)
            # NUM_DOUBLE_ZEROS+=1

    A=np.asarray(A)
    F=np.asarray(F)
    SMAPE=100.0/A.shape[0] * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))
    # SMAPE=100.0/A.shape[0] * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)) + np.finfo(float).eps)
    # print("SMAPE: %.4f%%"%SMAPE)
    # print("Number of double 0's: %d" %NUM_DOUBLE_ZEROS)

    return SMAPE

# def smape(A, F):
#     A=np.asarray(A).flatten()
#     F=np.asarray(F).flatten()
#     SMAPE=100.0/A.shape[0] * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)) + np.finfo(float).eps)
#     return SMAPE

def smape_round(A, F):
    A=np.asarray(A).flatten().round(0)
    F=np.asarray(F).flatten().round(0)
    SMAPE=100.0/A.shape[0] * np.sum(2 * np.abs(F - A ) / (np.abs(A) + np.abs(F) + np.finfo(float).eps))
    return SMAPE

def smape(A, F):
    A=np.asarray(A).flatten()
    F=np.asarray(F).flatten()
    SMAPE=100.0/A.shape[0] * np.sum(2 * np.abs(F - A ) / (np.abs(A) + np.abs(F) + np.finfo(float).eps))
    return SMAPE

def smape_100(A, F):
    A=np.asarray([A]).flatten()
    F=np.asarray([F]).flatten()
    SMAPE=100.0/A.shape[0] * np.sum(2 * np.abs(F - A ) / (np.abs(A) + np.abs(F) + np.finfo(float).eps))
    return SMAPE/2.0

def test_model(model,RESULTS_DIR,x_test,y_test):
    #make predictions on test data
    predictions=model.predict(x_test)
    print("Shape of predictions: %s"%str(predictions.shape))
    PREDICTION_OUTPUT_FP=RESULTS_DIR+"predictions"
    np.save(PREDICTION_OUTPUT_FP,predictions)
    return predictions

def output_model_summary(model,RESULTS_DIR):
    orig_stdout = sys.stdout
    mf = open(RESULTS_DIR+ 'Model-Summary.txt', 'w')
    sys.stdout = mf
    print(model.summary())
    sys.stdout = orig_stdout
    mf.close()

def create_checkpoint_callback(MODEL_OUTPUT_DIR):
    checkpoint_filepath=MODEL_OUTPUT_DIR+"p2_model.best.hdf5"
    checkpoint = ModelCheckpoint(checkpoint_filepath, monitor="val_loss", verbose=1, save_best_only=True, mode='auto')
    print("Created checkpoint callback")
    print(checkpoint)
    return checkpoint,checkpoint_filepath



# def reconfigure_x_array_shape_for_training(x,y):
#     difference=x.shape[1] - y.shape[1]

#     if y.shape[1] ==1:
#         mult_factor = difference +1
#     else:
#         mult_factor = difference

#     xnew_dim1=y.shape[1]
#     xnew_dim2=x.shape[2] * mult_factor
#     x=x.reshape((x.shape[0], xnew_dim1, xnew_dim2))
#     return x
# def convert_3d_time_series_array_to_2d(myarray):
#     myarray = myarray.reshape((myarray.shape[0],  myarray.shape[1] * myarray.shape[2]))
#     return myarray

def reconfigure_x_array_shape_for_training(x,y,x_str):
    product_to_be_maintained=x.shape[1] * x.shape[2]
    new_xdim1=x.shape[1]
    new_xdim2=int(product_to_be_maintained/new_xdim1)
    x=x.reshape((x.shape[0],new_xdim1, new_xdim2))
    print("New %s shape after reshaping: %s"%(x_str, str(x.shape)))
    return x


def reconfigure_all_x_array_shapes_for_training(x_train, x_val, x_test, y):
    x_train=reconfigure_x_array_shape_for_training(x_train,y, "x_train")
    x_val=reconfigure_x_array_shape_for_training(x_val,y, "x_val")
    x_test=reconfigure_x_array_shape_for_training(x_test,y, "x_test")
    return x_train,x_val,x_test

def reconfigure_all_x_array_shapes_for_training(x_train, x_val, x_test, y):
    x_train=reconfigure_x_array_shape_for_training(x_train,y, "x_train")
    x_val=reconfigure_x_array_shape_for_training(x_val,y, "x_val")
    x_test=reconfigure_x_array_shape_for_training(x_test,y, "x_test")
    return x_train,x_val,x_test


def reshape_y_test_and_pred_by_features(y_test,predictions):
    NUM_FEATURES_TO_PREDICT=y_test.shape[2]
    new_dim0=y_test.shape[0] * y_test.shape[1]
    new_dim1=NUM_FEATURES_TO_PREDICT

    y_test=y_test.reshape((new_dim0,new_dim1))
    predictions=predictions.reshape((new_dim0,new_dim1))

    y_test=np.transpose(y_test)
    predictions=np.transpose(predictions)
    print("New y_test shape: %s" %str(y_test.shape))
    print("New predictions shape: %s" %str(predictions.shape))
    return y_test,predictions

# def get_wsmape(y_test,predictions):

def get_weight_dataframe(y_test, FEATURES_OF_INTEREST):
    WEIGHT_DICT={}
    y_test=np.absolute(y_test)
    SUMS=np.sum(y_test, axis=1)
    print(SUMS)
    WEIGHTS=SUMS/SUMS.sum(axis=0)
    print(WEIGHTS)
    print(WEIGHTS.sum())

    WEIGHT_DICT["feature"]=FEATURES_OF_INTEREST
    WEIGHT_DICT["weight"]=WEIGHTS

    WEIGHT_DF=pd.DataFrame(data=WEIGHT_DICT)
    WEIGHT_DF=WEIGHT_DF.sort_values(by=["weight"], ascending=False)
    return WEIGHT_DF


def wsmape(y_test, predictions,WEIGHT_DF):
    WEIGHTS=list(WEIGHT_DF["weight"])
    WSMAPE=0
    for i in range(y_test.shape[0]):
        CUR_SMAPE=smape(y_test[i], predictions[i])
        WSMAPE+= (CUR_SMAPE * WEIGHTS[i])
    # print("WSMAPE: %.4f%%" %WSMAPE)
    return WSMAPE

def evaluate_predictions(y_test, predictions,WEIGHT_DF,RESULTS_DIR):
    #evaluate predictions
    MAE=mae(y_test,predictions)
    WSMAPE=wsmape(y_test,predictions,WEIGHT_DF)
    UNWEIGHTED_SMAPE=smape(y_test,predictions)

    #save evaluation to a file
    EVAL_FP=RESULTS_DIR+"Results.txt"
    f=open(EVAL_FP, "w")

    output_str="MAE: %.4f"%MAE
    screen_and_file_print(output_str,f)

    output_str="WSMAPE: %.2f%%"%WSMAPE
    screen_and_file_print(output_str,f)

    output_str="UNWEIGHTED_SMAPE: %.2f%%"%UNWEIGHTED_SMAPE
    screen_and_file_print(output_str,f)

    f.close()
    return MAE,WSMAPE

def get_model_pred_and_shifted_pred_comparison_df(model_smape_df,pseudo_pred_smape_df, RESULTS_DIR):
    print("Combining model pred and shifted pred smape dfs...")
    model_smape_df["pred_type"]="model_pred"
    pseudo_pred_smape_df["pred_type"]="pseudo_pred"
    smape_df=pd.concat([model_smape_df, pseudo_pred_smape_df])

    #now we must sort by each feature
    fts=smape_df["feature"].unique()
    all_temp_dfs=[]
    for ft in fts:
        temp_df=smape_df.copy()
        temp_df=temp_df[temp_df["feature"]==ft]
        temp_df=temp_df.sort_values("smape", ascending=True)
        temp_df["winning_pred_type"]=[1,0]
        all_temp_dfs.append(temp_df)

    smape_df=pd.concat(all_temp_dfs).reset_index(drop=True)
    output_fp=RESULTS_DIR+"All-Prediction-Types-SMAPEs.csv"
    smape_df.to_csv(output_fp, index=False)
    return smape_df

def get_smape_per_feature(y_test, predictions,ALL_FEATURES_TO_PREDICT,RESULTS_DIR,FAKE_PRED_FLAG=False):

    #save smapes here
    all_smapes=[]
    for i in range(y_test.shape[0]):
        cur_ft=ALL_FEATURES_TO_PREDICT[i]
        print("Getting smape for %s"%cur_ft)

        #cur smape
        smape_result=smape(y_test[i],predictions[i])

        all_smapes.append(smape_result)
        print("Smape for %s: %.4f%%"%(cur_ft,smape_result))

    #make smape df
    smape_df=pd.DataFrame(data={"feature":ALL_FEATURES_TO_PREDICT, "smape":all_smapes})

    smape_df=smape_df.sort_values(["smape"])

    print(smape_df)

    output_fp=RESULTS_DIR+"All-Feature-SMAPEs.csv"
    if FAKE_PRED_FLAG==True:
        output_fp=RESULTS_DIR+"Prediction-By-Shifting-All-Feature-SMAPEs.csv"


    create_output_dir(RESULTS_DIR)
    smape_df.to_csv(output_fp, index=False)
    return smape_df

def get_smape_df(PRED_DF, GTDF,RESULTS_DIR):
    fts=list(PRED_DF)
    if "created_at" in fts:
        fts.remove("created_at")

    #save smapes here
    all_smapes=[]
    proper_fts=[]

    for ft in fts:

        #cur smape
        if ft in list(GTDF):
            smape_result=smape(PRED_DF[ft],GTDF[ft])
            all_smapes.append(smape_result)
            print("Smape for %s: %.4f%%"%(ft,smape_result))
        else:
            fts.remove(ft)
            print("Removed %s" %ft)
            continue




    print("Num fts: %d" %len(fts))
    print("Num smapes: %d" %len(all_smapes))
    print(fts)
    smape_df=pd.DataFrame(data={"feature":fts, "smape":all_smapes})

    smape_df=smape_df.sort_values(["smape"])

    print(smape_df)

    output_fp=RESULTS_DIR+"All-Feature-SMAPEs.csv"
    create_output_dir(RESULTS_DIR)
    smape_df.to_csv(output_fp, index=False)
    return smape_df

def save_get_gtdf_and_pred_df(GTDF,PRED_DF,RESULTS_DIR):
    OUTPUT_DIR=RESULTS_DIR+"Ground-Truth-and-Prediction-Dataframes/"
    create_output_dir(OUTPUT_DIR)
    PRED_FP=OUTPUT_DIR+"Predictions.csv"
    PRED_DF.to_csv(PRED_FP, index=False)
    GTFP=OUTPUT_DIR+"Ground-Truth.csv"
    GTDF.to_csv(GTFP, index=False)
    print("Done saving ground truth and prediction dataframes!")

def save_y_deltas_and_both_prediction_types(y_test_as_deltas,predictions,predictions_by_shifting,ALL_FEATURES_TO_PREDICT,RESULTS_DIR):
    GT_DICT={}
    PRED_DICT={}
    FAKE_PRED_DICT={}
    print("Saving y_deltas and predictions")

    for i in range(y_test_as_deltas.shape[0]):
        print("Creating %s column" %(ALL_FEATURES_TO_PREDICT[i]))
        GT_DICT[ALL_FEATURES_TO_PREDICT[i]]=y_test_as_deltas[i]
        PRED_DICT[ALL_FEATURES_TO_PREDICT[i]]=predictions[i]
        FAKE_PRED_DICT[ALL_FEATURES_TO_PREDICT[i]]=predictions_by_shifting[i]

    PRED_DF=pd.DataFrame(data=PRED_DICT)
    GT_DF=pd.DataFrame(data=GT_DICT)
    FAKE_PRED_DF=pd.DataFrame(data=FAKE_PRED_DICT)

    # print("pred df")
    # print(PRED_DF)

    # print("\ngtdf")
    # print(GT_DF)

    # print("\nFake pred df")
    # print(FAKE_PRED_DF)

    OUTPUT_DIR=RESULTS_DIR+"Ground-Truth-and-Prediction-Dataframes/"
    create_output_dir(OUTPUT_DIR)
    PRED_FP=OUTPUT_DIR+"Predictions.csv"
    PRED_DF.to_csv(PRED_FP, index=False)
    GTFP=OUTPUT_DIR+"Ground-Truth.csv"
    GT_DF.to_csv(GTFP, index=False)
    FPRED_FP=OUTPUT_DIR+"Predictions-By-Shifting.csv"
    FAKE_PRED_DF.to_csv(FPRED_FP, index=False)

    print("Done saving ground truth and prediction dataframes!")

    return GT_DF,PRED_DF,FAKE_PRED_DF

def save_y_deltas_and_predictions(y_test_as_deltas,predictions,ALL_FEATURES_TO_PREDICT,RESULTS_DIR):
    GT_DICT={}
    PRED_DICT={}
    print("Saving y_deltas and predictions")

    for i in range(y_test_as_deltas.shape[0]):
        print("Creating %s column" %(ALL_FEATURES_TO_PREDICT[i]))
        GT_DICT[ALL_FEATURES_TO_PREDICT[i]]=y_test_as_deltas[i]
        PRED_DICT[ALL_FEATURES_TO_PREDICT[i]]=predictions[i]

    PRED_DF=pd.DataFrame(data=PRED_DICT)
    GT_DF=pd.DataFrame(data=GT_DICT)

    print("pred df")
    print(PRED_DF)

    print("\ngtdf")
    print(GT_DF)

    OUTPUT_DIR=RESULTS_DIR+"Ground-Truth-and-Prediction-Dataframes/"
    create_output_dir(OUTPUT_DIR)
    PRED_FP=OUTPUT_DIR+"Predictions.csv"
    PRED_DF.to_csv(PRED_FP, index=False)
    GTFP=OUTPUT_DIR+"Ground-Truth.csv"
    GT_DF.to_csv(GTFP, index=False)

    print("Done saving ground truth and prediction dataframes!")

    return GT_DF,PRED_DF

def get_gtdf_and_pred_df(y_test_as_deltas,predictions,ALL_FEATURES_TO_PREDICT):
    GT_DICT={}
    PRED_DICT={}
    print("Saving y_deltas and predictions")

    for i in range(y_test_as_deltas.shape[0]):
        print("Creating %s column" %(ALL_FEATURES_TO_PREDICT[i]))
        GT_DICT[ALL_FEATURES_TO_PREDICT[i]]=y_test_as_deltas[i]
        PRED_DICT[ALL_FEATURES_TO_PREDICT[i]]=predictions[i]

    PRED_DF=pd.DataFrame(data=PRED_DICT)
    GT_DF=pd.DataFrame(data=GT_DICT)

    print(PRED_DF)
    print(GT_DF)

    return GT_DF,PRED_DF

def get_gtdf_and_pred_df(y_test_as_deltas,predictions,ALL_FEATURES_TO_PREDICT):
    GT_DICT={}
    PRED_DICT={}
    print("Saving y_deltas and predictions")

    for i in range(y_test_as_deltas.shape[0]):
        print("Creating %s column" %(ALL_FEATURES_TO_PREDICT[i]))
        GT_DICT[ALL_FEATURES_TO_PREDICT[i]]=y_test_as_deltas[i]
        PRED_DICT[ALL_FEATURES_TO_PREDICT[i]]=predictions[i]

    PRED_DF=pd.DataFrame(data=PRED_DICT)
    GT_DF=pd.DataFrame(data=GT_DICT)

    print(PRED_DF)
    print(GT_DF)

    return GT_DF,PRED_DF


def plot_loss(history, RESULTS_DIR):
    print("Plotting training and validation loss.")
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='upper left')

    LOSS_GRAPH_FP=RESULTS_DIR+"Loss-Plot.png"
    plt.savefig(LOSS_GRAPH_FP)
    plt.close()
    print("Done plotting loss.")

def plot_loss_without_validation(history, RESULTS_DIR):
    print("Plotting training and validation loss.")
    plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['training'], loc='upper left')

    LOSS_GRAPH_FP=RESULTS_DIR+"Loss-Plot.png"
    plt.savefig(LOSS_GRAPH_FP)
    plt.close()
    print("Done plotting loss.")

def plot_loss_by_model(history, RESULTS_DIR,ft_idx,ft):
    print("Plotting training and validation loss.")
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='upper left')

    LOSS_GRAPH_FP=RESULTS_DIR+"Loss-Plot-%d-%s.png"%(ft_idx,ft)
    plt.savefig(LOSS_GRAPH_FP)
    plt.close()
    print("Done plotting loss.")

def plot_loss_by_model_no_val(history, RESULTS_DIR,ft_idx,ft):
    print("Plotting training and validation loss.")
    plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['training'], loc='upper left')

    LOSS_GRAPH_FP=RESULTS_DIR+"Loss-Plot-%d-%s.png"%(ft_idx,ft)
    plt.savefig(LOSS_GRAPH_FP)
    plt.close()
    print("Done plotting loss.")


def save_as_json(OBJECT,OUTPUT_FP):
    print("Saving %s..."%OUTPUT_FP)
    with open(OUTPUT_FP, "w") as f:
        json.dump(OBJECT,f)
    print("Saved %s"%OUTPUT_FP)

def create_trial_df(ALL_TAGS,MAIN_RESULTS_DIR):
    ALL_EVAL_FILEPATHS=[]
    print("Checking for all directories")
    for TAG in ALL_TAGS:
        TAG_DIR=MAIN_RESULTS_DIR+TAG+"/"
        FINISHED_TEXT_FP=TAG_DIR +"Done.txt"
        file_exists = os.path.isfile(FINISHED_TEXT_FP)
        if not file_exists:
            print("%s does not exist. Cannot create trial df file yet."%FINISHED_TEXT_FP)
            return
        else:
            print("%s exists" %FINISHED_TEXT_FP)
            EVAL_FP=TAG_DIR+"MAE.csv"
            ALL_EVAL_FILEPATHS.append(EVAL_FP)

    ALL_DFS=[]
    for EVAL_FP in ALL_EVAL_FILEPATHS:
        TEMP_DF=pd.read_csv(EVAL_FP)
        ALL_DFS.append(TEMP_DF)
        print("Retrieved dataframe from %s" %EVAL_FP)

    FINAL_DF=pd.concat(ALL_DFS)
    FINAL_DF=FINAL_DF.sort_values(by=["mae"])
    OUTPUT_FP=MAIN_RESULTS_DIR+"All-Trial-Results.csv"
    FINAL_DF.to_csv(OUTPUT_FP, index=False)
    print("Saved all trials in one df at %s" %OUTPUT_FP)
    print(FINAL_DF)
    print("Done!")

def graph_test_and_prediction_arrays(y_test, predictions,FEATURE,GRANULARITY,OUTPUT_DIR,FAKE_PRED=False):
    print("Graphing %s" %FEATURE)
    if GRANULARITY=="D":
        gran_str="day"
    elif GRANULARITY=="H":
        gran_str="hour"
    elif GRANULARITY=="min":
        gran_str="minute"
    elif GRANULARITY=="W":
        gran_str="week"
    elif GRANULARITY=="M":
        gran_str="month"
    else:
        gran_str="second"

    create_output_dir(OUTPUT_DIR)

    fig, ax = plt.subplots()

    ax.plot(y_test,"-b",label="Ground Truth")
    # if FAKE_PRED==True:
    #     ax.plot(predictions,"+darkorange",label="Fake-Prediction")
    # else:
    ax.plot(predictions,":r",label="Prediction")

    ax.set_title("%s at %s granularity"%(FEATURE,gran_str))

    # ax.set_title("%s (in %ss) %.4f%%"%(FEATURE,gran_str,smape))
    ax.set_xlabel("%ss" %gran_str)
    ax.set_ylabel("Count")

    leg = ax.legend()
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR+"%s-%s-Graph.png"%(FEATURE,gran_str))

    plt.close()

def graph_test_and_prediction_arrays_with_smape(y_test, predictions,FEATURE,GRANULARITY,OUTPUT_DIR,smape,FAKE_PRED=False):
    print("Graphing %s" %FEATURE)
    if GRANULARITY=="D":
        gran_str="day"
    elif GRANULARITY=="H":
        gran_str="hour"
    elif GRANULARITY=="min":
        gran_str="minute"
    else:
        gran_str=GRANULARITY

    create_output_dir(OUTPUT_DIR)

    fig, ax = plt.subplots()

    ax.plot(y_test,"-b",label="Ground Truth")
    if FAKE_PRED==True:
        # ax.plot(predictions,":",color="darkorange",label="Fake-Prediction")
        ax.plot(predictions,":r",label="Fake-Prediction")
    else:
        ax.plot(predictions,":r",label="Prediction")

    ax.set_title("%s (in %ss) %.4f%%"%(FEATURE,gran_str,smape))
    ax.set_xlabel("%ss" %gran_str)
    ax.set_ylabel("Count")

    leg = ax.legend()
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR+"%s-%s-Graph-%.4f%%.png"%(FEATURE,gran_str,smape))
    plt.close()


def graph_simulated_feature(predictions, graph_output_dir,FEATURE,GRANULARITY):
    print("Graphing %s" %FEATURE)
    if GRANULARITY=="D":
        gran_str="day"
    elif GRANULARITY=="H":
        gran_str="hour"
    elif GRANULARITY=="min":
        gran_str="minute"
    else:
        gran_str="second"

    create_output_dir(graph_output_dir)

    # ax.plot(y_test,"-b",label="Ground Truth")
    fig, ax = plt.subplots()
    ax.plot(predictions,":r",label="Prediction")
    ax.set_title("%s at %s granularity"%(FEATURE,gran_str))
    ax.set_xlabel("%ss" %gran_str)
    ax.set_ylabel("Count")

    leg = ax.legend()
    fig.savefig(graph_output_dir+"%s-%s-Simulation-Graph.png"%(FEATURE,gran_str))
    plt.close()

def graph_df(df,VERSION_OUTPUT_DIR,GRANULARITY):
    graph_dir=VERSION_OUTPUT_DIR + "Graphs/"
    create_output_dir(graph_dir)

    if GRANULARITY=="D":
        gran_str="day"
    elif GRANULARITY=="H":
        gran_str="hour"
    elif GRANULARITY=="min":
        gran_str="minute"
    else:
        gran_str="second"

    fts=list(df)
    if "created_at" in fts:
        fts.remove("created_at")
    if "nodeTime" in fts:
        fts.remove("nodeTime")

    for ft in fts:
        cur_vector=df[ft]
        print("Graphing %s" %ft)

        fig, ax = plt.subplots()
        ax.plot(cur_vector,"-b")
        ax.set_title("%s at %s granularity"%(ft,gran_str))
        ax.set_xlabel("%ss" %gran_str)
        ax.set_ylabel("Count")

        leg = ax.legend()
        fig.savefig(graph_dir+"%s-%s-Graph.png"%(ft,gran_str))
        plt.close()






def graph_ground_truth_vs_pred_dfs(GTDF,PRED_DF,GRANULARITY,OUTPUT_DIR):
    create_output_dir(OUTPUT_DIR)

    FEATURES=list(GTDF)

    for FEATURE in FEATURES:
        y_test=np.asarray(GTDF[FEATURE]).astype("float32")
        predictions=np.asarray(PRED_DF[FEATURE]).astype("float32")
        graph_test_and_prediction_arrays(y_test, predictions,FEATURE,GRANULARITY,OUTPUT_DIR)

    print("Done graphing!")

def graph_ground_truth_vs_pred_dfs_with_smape(GTDF,PRED_DF,GRANULARITY,OUTPUT_DIR,smape_df,FAKE_PRED=False):
    create_output_dir(OUTPUT_DIR)

    FEATURES=list(GTDF)
    smapes=list(smape_df["smape"])

    for i,FEATURE in enumerate(FEATURES):
        cur_smape=smapes[i]
        y_test=np.asarray(GTDF[FEATURE])
        predictions=np.asarray(PRED_DF[FEATURE])
        graph_test_and_prediction_arrays_with_smape(y_test, predictions,FEATURE,GRANULARITY,OUTPUT_DIR,cur_smape,FAKE_PRED)

    print("Done graphing!")

def graph_ground_truth_vs_pred_dfs_with_smape_v2_with_smape_dict(ft_to_smape_dict,GTDF,PRED_DF,GRANULARITY,OUTPUT_DIR,smape_df,FAKE_PRED=False,timecol="nodeTime"):
    create_output_dir(OUTPUT_DIR)

    FEATURES=list(GTDF)
    # smapes=list(smape_df["smape"])

    if str(timecol) in FEATURES:
        FEATURES.remove(timecol)

    print(FEATURES)

    for i,FEATURE in enumerate(FEATURES):

        if (str(FEATURE) == "nodeTime") or (str(FEATURE) == "created_at"):
            continue

        cur_smape=ft_to_smape_dict[FEATURE]
        y_test=np.asarray(GTDF[FEATURE])
        predictions=np.asarray(PRED_DF[FEATURE])
        graph_test_and_prediction_arrays_with_smape(y_test, predictions,FEATURE,GRANULARITY,OUTPUT_DIR,cur_smape,FAKE_PRED)

    print("Done graphing!")


def save_mae_and_wsmape_df(MAE,WSMAPE,EXPERIMENT_TAG,MAIN_OUTPUT_DIR):
    RESULT_DICT={}
    RESULT_DICT["experiment_tag"]=[EXPERIMENT_TAG]
    RESULT_DICT["mae"]=[MAE]
    RESULT_DICT["wsmape"]=[WSMAPE]

    MINI_DF=pd.DataFrame(data=RESULT_DICT)

    OUTPUT_FP=MAIN_OUTPUT_DIR+EXPERIMENT_TAG+".csv"
    FINISHED_TEXT_FP=MAIN_OUTPUT_DIR +"%s.txt"%EXPERIMENT_TAG
    with open(FINISHED_TEXT_FP, "w") as f:
        f.write(OUTPUT_FP)
        print(OUTPUT_FP)

    MINI_DF.to_csv(OUTPUT_FP, index=False)

    return MINI_DF, FINISHED_TEXT_FP


def create_trial_df_from_experiment_tags(ALL_TAGS,AGG_RESULTS_DIR):
    ALL_ERROR_DFS=[]
    for TAG in ALL_TAGS:
        CUR_FILE_OF_INTEREST=AGG_RESULTS_DIR+"%s.txt"%TAG
        # file_exists = os.path.isfile(CUR_FILE_OF_INTEREST)
        try:
            f=open(CUR_FILE_OF_INTEREST, "r")
        except:
            print("%s does not exist. Cannot create trial df just yet." %CUR_FILE_OF_INTEREST)
            return

        print("%s exists" %CUR_FILE_OF_INTEREST)
        for line in f:
            ERROR_DF_DIR=line
            print(ERROR_DF_DIR)
            break
        TEMP_ERROR_DF=pd.read_csv(ERROR_DF_DIR)
        print(TEMP_ERROR_DF)
        ALL_ERROR_DFS.append(TEMP_ERROR_DF)

    FINAL_DF=pd.concat(ALL_ERROR_DFS)
    FINAL_DF=FINAL_DF.sort_values(by=["wsmape"])
    print(FINAL_DF)
    OUTPUT_FP=AGG_RESULTS_DIR+"Final-Results.csv"
    FINAL_DF.to_csv(OUTPUT_FP, index=False)
    return FINAL_DF


def get_start_date(DF, TIME_COL="created_at"):
    DF[TIME_COL] = pd.to_datetime(DF[TIME_COL])
    DF=DF.sort_values(by=[TIME_COL])
    start_date=DF[TIME_COL].iloc[0]
    return start_date

def get_end_date(DF, TIME_COL="created_at"):
    DF[TIME_COL] = pd.to_datetime(DF[TIME_COL])
    DF=DF.sort_values(by=[TIME_COL])
    end_date=DF[TIME_COL].iloc[-1]
    return end_date

def get_array_time_info_dict(start_date,end_date,x_train, x_val, x_test, y_train_as_values,y_train_as_deltas, y_val_as_values, y_val_as_deltas, y_test_as_values,y_test_as_deltas):
    ARRAY_TIME_INFO_DICT={}
    ARRAY_TIME_INFO_DICT["x_train_shape"]=x_train.shape
    ARRAY_TIME_INFO_DICT["x_val_shape"]=x_val.shape
    ARRAY_TIME_INFO_DICT["x_test_shape"]=x_test.shape

    ARRAY_TIME_INFO_DICT["y_train_as_values_shape"]=y_train_as_values.shape
    ARRAY_TIME_INFO_DICT["y_train_as_deltas_shape"]=y_train_as_deltas.shape
    ARRAY_TIME_INFO_DICT["y_val_as_values_shape"]=y_val_as_values.shape
    ARRAY_TIME_INFO_DICT["y_val_as_deltas_shape"]=y_val_as_deltas.shape
    ARRAY_TIME_INFO_DICT["y_test_as_values_shape"]=y_test_as_values.shape
    ARRAY_TIME_INFO_DICT["y_test_as_deltas_shape"]=y_test_as_deltas.shape

    return ARRAY_TIME_INFO_DICT

def get_array_time_info_dict_no_val(start_date,end_date,x_train, x_test, y_train_as_values,y_train_as_deltas, y_test_as_values,y_test_as_deltas):
    ARRAY_TIME_INFO_DICT={}
    ARRAY_TIME_INFO_DICT["x_train_shape"]=x_train.shape
    # ARRAY_TIME_INFO_DICT["x_val_shape"]=x_val.shape
    ARRAY_TIME_INFO_DICT["x_test_shape"]=x_test.shape

    ARRAY_TIME_INFO_DICT["y_train_as_values_shape"]=y_train_as_values.shape
    ARRAY_TIME_INFO_DICT["y_train_as_deltas_shape"]=y_train_as_deltas.shape
    # ARRAY_TIME_INFO_DICT["y_val_as_values_shape"]=y_val_as_values.shape
    # ARRAY_TIME_INFO_DICT["y_val_as_deltas_shape"]=y_val_as_deltas.shape
    ARRAY_TIME_INFO_DICT["y_test_as_values_shape"]=y_test_as_values.shape
    ARRAY_TIME_INFO_DICT["y_test_as_deltas_shape"]=y_test_as_deltas.shape

    return ARRAY_TIME_INFO_DICT



def convert_seconds_to_HMS(TIME_IN_SECONDS):
    hours, rem = divmod(TIME_IN_SECONDS, 3600)
    minutes, seconds = divmod(rem, 60)
    return "%d hours, %d minutes, %d seconds"%(hours,minutes,seconds)

def get_time_info_dict(TOTAL_TRAINING_TIME,TOTAL_TESTING_TIME,TOTAL_TIME_OF_PROGRAM):
    TOTAL_TRAINING_TIME_STR=convert_seconds_to_HMS(TOTAL_TRAINING_TIME)
    TOTAL_TESTING_TIME_STR=convert_seconds_to_HMS(TOTAL_TESTING_TIME)
    TOTAL_TIME_OF_PROGRAM_STR=convert_seconds_to_HMS(TOTAL_TIME_OF_PROGRAM)

    TIME_DICT={}

    TIME_DICT["total_training_time"]=TOTAL_TRAINING_TIME_STR
    TIME_DICT["total_testing_time"]=TOTAL_TESTING_TIME_STR
    TIME_DICT["total_program_time"]=TOTAL_TIME_OF_PROGRAM_STR
    return TIME_DICT

def record_time_info(TIME_DICT, MAIN_RESULT_DIR):
    OUTPUT_TIME_FP=MAIN_RESULT_DIR+"Time-Info.txt"
    f=open(OUTPUT_TIME_FP, "w+")

    for key,val in TIME_DICT.items():
        output_str="%s : %s"%(key,val)
        screen_and_file_print(output_str, f)
    f.close()

def get_result_df(MAE,WSMAPE,GEN_PARAM_DICT,TIME_DICT,EXPERIMENT_TAG ,GRANULARITY):
    if GRANULARITY=="D":
        gran_str="days"
    elif GRANULARITY=="H":
        gran_str="hours"
    elif GRANULARITY=="min":
        gran_str="minutes"
    else:
        gran_str="seconds"

    for key,val in TIME_DICT.items():
        GEN_PARAM_DICT[key]=val

    RESULT_DICT={}
    RESULT_DICT["feature_timesteps_and_target_timesteps"]=EXPERIMENT_TAG+"-in-%s"%gran_str
    RESULT_DICT["mae"]=[MAE]
    RESULT_DICT["wsmape"]=[WSMAPE]
    for key,val in GEN_PARAM_DICT.items():
        RESULT_DICT[key]=[val]

    RESULT_DF=pd.DataFrame(data=RESULT_DICT)
    print(RESULT_DF)
    return RESULT_DF

def convert_df_to_feature_array2(start_date,end_date,all_fps,X_ARRAY_TIMESTEPS_IN_A_SEQUENCE,Y_ARRAY_TIMESTEPS_IN_A_SEQUENCE,VAL_PERCENTAGE,TEST_PERCENTAGE,FEATURES_OF_INTEREST,FEATURE_DF):
    TEMP_DF=FEATURE_DF[FEATURES_OF_INTEREST].copy()
    ALL_FEATURE_DFS=[TEMP_DF]
    FEATURE_ARRAY=convert_feature_df_to_feature_array(ALL_FEATURE_DFS)
    del TEMP_DF
    gc.collect()
    #create source and target arrays
    SOURCE_DATA,TARGET_VALUE_DATA,TARGET_DIFF_DATA=get_target_value_and_difference_arrays(FEATURE_ARRAY,X_ARRAY_TIMESTEPS_IN_A_SEQUENCE,Y_ARRAY_TIMESTEPS_IN_A_SEQUENCE)

    #uncomment this to sanity check
    # sanity_check_all_arrays(SOURCE_DATA,TARGET_VALUE_DATA,TARGET_DIFF_DATA,X_ARRAY_TIMESTEPS_IN_A_SEQUENCE,LIMIT=-1)
    # sys.exit(0)

    #get train,val, and test arrays
    x_train, x_val, x_test, y_train_as_values,y_train_as_deltas, y_val_as_values, y_val_as_deltas, y_test_as_values,y_test_as_deltas=get_train_val_and_test_arrays(SOURCE_DATA,TARGET_VALUE_DATA,TARGET_DIFF_DATA,VAL_PERCENTAGE,TEST_PERCENTAGE)

    # sys.exit(0)

    # #reshape data for LSTM
    x_train,x_val,x_test=reconfigure_all_x_array_shapes_for_training(x_train, x_val, x_test, y_train_as_values)

    #normalize the x arrays with MinMax Scaler
    #Also return the scaler itself for future denormalizing
    x_train, x_val, x_test, scaler=normalize_data(x_train, x_val, x_test)

    return x_train, x_val, x_test, y_train_as_values,y_train_as_deltas, y_val_as_values, y_val_as_deltas, y_test_as_values,y_test_as_deltas,scaler

def get_test_data(start_date,end_date,all_fps,X_ARRAY_TIMESTEPS_IN_A_SEQUENCE,Y_ARRAY_TIMESTEPS_IN_A_SEQUENCE,VAL_PERCENTAGE,TEST_PERCENTAGE):
    FEATURE_DF,start_date,end_date=get_feature_df_from_fp_list(all_fps, start_date,end_date)
    if FEATURE_DF.shape[0]==0:
        print("%s has no features. Skipping...")
    print(start_date)
    print(end_date)
    fts=list(FEATURE_DF)
    for ft in fts:
        print(ft)
    # FEATURE_DF=get_feature_df(INPUT_FEATURE_FP,start_date,end_date)
    fts=verify_features(FEATURE_DF, fts)
    FEATURE_DF=FEATURE_DF[fts]
    FEATURES_OF_INTEREST=get_proper_header_list(FEATURE_DF)
    NUM_FEATURES_TO_PREDICT=len(FEATURES_OF_INTEREST)
    # LSTM_UNITS=int(NUM_FEATURES_TO_PREDICT/2)
    # LSTM_UNITS=250
    # LSTM_UNITS=500
    # LSTM_UNITS=int(NUM_FEATURES_TO_PREDICT * 5)


    #establish all features being predicted.
    ALL_FEATURES_TO_PREDICT=FEATURES_OF_INTEREST

    #setup result dir
    # MAIN_RESULT_DIR="PND-NN-WITH-PRICE/DEBUG-%s/%s-%s/Epochs-%s/Granularity-%s/%s/"%(str(DEBUG),start_date,end_date,EPOCHS,GRANULARITY,coin)
    # VERSION_TAG="GRAN-%s_SD-%s_ED-%s_XS-%d_YS-%d_BS-%d_E-%d_NLL-%d_LOSS-%s_VP-%s_TP-%s_NF-%d" \
    # %(GRANULARITY,str(start_date), str(end_date),X_ARRAY_TIMESTEPS_IN_A_SEQUENCE,
    #   Y_ARRAY_TIMESTEPS_IN_A_SEQUENCE, BATCH_SIZE, EPOCHS, NUM_LSTM_LAYERS, LOSS,
    #   str(VAL_PERCENTAGE), str(TEST_PERCENTAGE),NUM_FEATURES_TO_PREDICT)

    # print(VERSION_TAG)
    # VERSION_OUTPUT_DIR=MAIN_RESULT_DIR+VERSION_TAG+"/"
    # create_output_dir(VERSION_OUTPUT_DIR)

    #convert all feature dfs to arrays
    TEMP_DF=FEATURE_DF[FEATURES_OF_INTEREST].copy()
    ALL_FEATURE_DFS=[TEMP_DF]
    FEATURE_ARRAY=convert_feature_df_to_feature_array(ALL_FEATURE_DFS)
    del TEMP_DF
    gc.collect()

    #create source and target arrays
    SOURCE_DATA,TARGET_VALUE_DATA,TARGET_DIFF_DATA=get_target_value_and_difference_arrays(FEATURE_ARRAY,X_ARRAY_TIMESTEPS_IN_A_SEQUENCE,Y_ARRAY_TIMESTEPS_IN_A_SEQUENCE)

    #uncomment this to sanity check
    # sanity_check_all_arrays(SOURCE_DATA,TARGET_VALUE_DATA,TARGET_DIFF_DATA,X_ARRAY_TIMESTEPS_IN_A_SEQUENCE,LIMIT=-1)
    # sys.exit(0)

    #get train,val, and test arrays
    x_train, x_val, x_test, y_train_as_values,y_train_as_deltas, y_val_as_values, y_val_as_deltas, y_test_as_values,y_test_as_deltas=get_train_val_and_test_arrays(SOURCE_DATA,TARGET_VALUE_DATA,TARGET_DIFF_DATA,VAL_PERCENTAGE,TEST_PERCENTAGE)

    # sys.exit(0)

    # #reshape data for LSTM
    x_train,x_val,x_test=reconfigure_all_x_array_shapes_for_training(x_train, x_val, x_test, y_train_as_values)

    #normalize the x arrays with MinMax Scaler
    #Also return the scaler itself for future denormalizing
    x_train, x_val, x_test, scaler=normalize_data(x_train, x_val, x_test)

    return x_train, x_val, x_test, y_train_as_values,y_train_as_deltas, y_val_as_values, y_val_as_deltas, y_test_as_values,y_test_as_deltas,scaler



def calc_dates(start_date,end_date,x_train, x_val, x_test,GRANULARITY="D",n_months=14):
    #get all dates in this range
    all_timesteps=list(pd.date_range(start_date, end_date, freq=GRANULARITY))
    print("Num dates: %d" %len(all_timesteps))

    num_timesteps=x_train.shape[1]
    num_training_samples=x_train.shape[0]
    num_val_samples=x_val.shape[0]
    num_test_samples=x_test.shape[0]

    training_start_date=start_date
    start_date_list=start_date.split("-")
    syear=int(start_date_list[0])
    smonth=int(start_date_list[1])
    sday=int(start_date_list[2])

    training_start_date=date(syear,smonth,sday)
    training_end_date=training_start_date + timedelta(days=num_timesteps*num_training_samples-1)

    val_start_date=training_end_date + timedelta(days=1)
    val_end_date=val_start_date + timedelta(days=num_timesteps*num_val_samples-1)

    test_start_date=val_end_date + timedelta(days=1)
    test_end_date=test_start_date + timedelta(days=num_timesteps*num_test_samples-1)

    # print("Start date (Friday): %s" %str(start_date))
    # print("End date: %s" %str(end_date))

    print("\nx training start date: %s" %str(training_start_date))
    print("x training end date: %s" %str(training_end_date))

    print("\nx val_start_date: %s" %str(val_start_date))
    print("x val_end_date: %s" %str(val_end_date))

    print("\nx test_start_date: %s" %str(test_start_date))
    print("x test_end_date: %s" %str(test_end_date))

    x_test_jump_ahead_factor=(num_test_samples-1) * num_timesteps
    init_condit_start=test_start_date
    init_condit_end=init_condit_start + timedelta(days=num_timesteps-1)
    print("\nINITIAL CONDITION STARTS AT %s" %str(init_condit_start))
    print("INITIAL CONDITION ENDS AT %s" %str(init_condit_end))

    first_predicted_date=init_condit_end+ timedelta(days=1)
    n_months_later = init_condit_end+relativedelta(months=12)+relativedelta(months=2)+timedelta(days=1)
    all_needed_prediction_dates=list(pd.date_range(first_predicted_date, n_months_later, freq=GRANULARITY))
    num_prediction_steps=len(all_needed_prediction_dates)
    print("There are %d days we must predict" %num_prediction_steps)
    print("First day we'll predict: %s" %str(first_predicted_date))
    print("%d months later the date will be: %s" %(n_months,n_months_later))

    #get number of times to loop the simulation
    if num_prediction_steps%num_timesteps != 0:
        add_amount=1
    else:
        add_amount=0

    n_loops=int(num_prediction_steps/num_timesteps) + add_amount
    print("\nWe'll need to do %d loops for the simulation" %n_loops)

    num_predicted_dates= n_loops * num_timesteps
    print("Our code will predict %d dates" %num_predicted_dates)

    last_date_predicted=first_predicted_date + timedelta(days=num_predicted_dates)
    all_predicted_dates_with_extra_dates=list(pd.date_range(first_predicted_date, last_date_predicted, freq=GRANULARITY))
    print("Our prediction df will end at date %s" %str(last_date_predicted))

    training_start_date=date(syear,smonth,sday)+ timedelta(days=num_timesteps)
    training_end_date=training_start_date + timedelta(days=num_timesteps*num_training_samples-1)

    val_start_date=training_end_date + timedelta(days=1)
    val_end_date=val_start_date + timedelta(days=num_timesteps*num_val_samples-1)

    test_start_date=val_end_date + timedelta(days=1)
    test_end_date=test_start_date + timedelta(days=num_timesteps*num_test_samples-1)

    print("\ny training start date: %s" %str(training_start_date))
    print("y training end date: %s" %str(training_end_date))

    print("\ny val_start_date: %s" %str(val_start_date))
    print("y val_end_date: %s" %str(val_end_date))

    print("\ny test_start_date: %s" %str(test_start_date))
    print("y test_end_date: %s" %str(test_end_date))

    return n_loops,all_needed_prediction_dates,all_predicted_dates_with_extra_dates

def calc_dates(start_date,end_date,x_train, x_val, x_test,GRANULARITY="D",n_months=14):
    #get all dates in this range
    all_timesteps=list(pd.date_range(start_date, end_date, freq=GRANULARITY))
    print("Num dates: %d" %len(all_timesteps))

    num_timesteps=x_train.shape[1]
    num_training_samples=x_train.shape[0]
    num_val_samples=x_val.shape[0]
    num_test_samples=x_test.shape[0]

    training_start_date=start_date
    start_date_list=start_date.split("-")
    syear=int(start_date_list[0])
    smonth=int(start_date_list[1])
    sday=int(start_date_list[2])

    training_start_date=date(syear,smonth,sday)
    training_end_date=training_start_date + timedelta(days=num_timesteps*num_training_samples-1)

    val_start_date=training_end_date + timedelta(days=1)
    val_end_date=val_start_date + timedelta(days=num_timesteps*num_val_samples-1)

    test_start_date=val_end_date + timedelta(days=1)
    test_end_date=test_start_date + timedelta(days=num_timesteps*num_test_samples-1)

    # print("Start date (Friday): %s" %str(start_date))
    # print("End date: %s" %str(end_date))

    print("\nx training start date: %s" %str(training_start_date))
    print("x training end date: %s" %str(training_end_date))

    print("\nx val_start_date: %s" %str(val_start_date))
    print("x val_end_date: %s" %str(val_end_date))

    print("\nx test_start_date: %s" %str(test_start_date))
    print("x test_end_date: %s" %str(test_end_date))

    x_test_jump_ahead_factor=(num_test_samples-1) * num_timesteps
    init_condit_start=test_start_date
    init_condit_end=init_condit_start + timedelta(days=num_timesteps-1)
    print("\nINITIAL CONDITION STARTS AT %s" %str(init_condit_start))
    print("INITIAL CONDITION ENDS AT %s" %str(init_condit_end))

    first_predicted_date=init_condit_end+ timedelta(days=1)
    n_months_later = init_condit_end+relativedelta(months=12)+relativedelta(months=2)+timedelta(days=1)
    all_needed_prediction_dates=list(pd.date_range(first_predicted_date, n_months_later, freq=GRANULARITY))
    num_prediction_steps=len(all_needed_prediction_dates)
    print("There are %d days we must predict" %num_prediction_steps)
    print("First day we'll predict: %s" %str(first_predicted_date))
    print("%d months later the date will be: %s" %(n_months,n_months_later))

    #get number of times to loop the simulation
    if num_prediction_steps%num_timesteps != 0:
        add_amount=1
    else:
        add_amount=0

    n_loops=int(num_prediction_steps/num_timesteps) + add_amount
    print("\nWe'll need to do %d loops for the simulation" %n_loops)

    num_predicted_dates= n_loops * num_timesteps
    print("Our code will predict %d dates" %num_predicted_dates)

    last_date_predicted=first_predicted_date + timedelta(days=num_predicted_dates)
    all_predicted_dates_with_extra_dates=list(pd.date_range(first_predicted_date, last_date_predicted, freq=GRANULARITY))
    print("Our prediction df will end at date %s" %str(last_date_predicted))


    return n_loops,all_needed_prediction_dates,all_predicted_dates_with_extra_dates

def simulate_on_initial_condition_not_a_drill(x_test, model,scaler, num_sequences_to_simulate, scaler_flag=True):
    #sanity checking tag means these are arrays we should not have access to during the challenge
    print("############ Now simulating on initial condition... ################")

    #DIFFERENCE THRESHOLD
    #SANITY CHECK PARAM
    DIFFERENCE_THRESHOLD=0.1

    num_timesteps_in_a_sequence=x_test.shape[1]

    #create init condit array
    x_init_condit=x_test[:1,:,:]

    #get first sequence of predictions
    prediction_deltas=model.predict(x_init_condit)

    #make x_init_condit 2d
    x_init_condit=x_init_condit.reshape((x_init_condit.shape[0]*x_init_condit.shape[1],x_init_condit.shape[2]))

    #make pred array 2d
    prediction_deltas=prediction_deltas.reshape((prediction_deltas.shape[0]*prediction_deltas.shape[1],prediction_deltas.shape[2]))
    # print("prediction_deltas shape: %s" %str(prediction_deltas.shape))

    #current starting point = last day in the sequence
    cur_x_test_base_vals_for_pred=x_init_condit[num_timesteps_in_a_sequence-1:, ]


    if scaler_flag==True:
        #we must inverse transform our initial condition so the deltas can be added to it properly
        cur_x_test_base_vals_for_pred=scaler.inverse_transform(cur_x_test_base_vals_for_pred)


    cur_x_test_base_vals_for_pred=cur_x_test_base_vals_for_pred.flatten()
    # print("Cur base shape: %s" %str(cur_x_test_base_vals_for_pred.shape))

    #FOR SANITY CHECKING
    gt_cur_x_test_base_vals=cur_x_test_base_vals_for_pred.copy()

    #save all predicted sequences
    all_predicted_sequences=[]

    #get fts in a time step
    for i in range(num_sequences_to_simulate):

        #save your new cur sequence of predictions here
        cur_sequence_of_calculated_timestep_vals_not_deltas=[]

        #might have to add back 1
        for j in range(num_timesteps_in_a_sequence):

            #get cur timestep of predicted deltas
            cur_predicted_timestep_array_of_deltas_to_add=prediction_deltas[j]
            # print("cur_predicted_timestep_array_of_deltas_to_add shape: %s" %str(cur_predicted_timestep_array_of_deltas_to_add.shape))

            #add pred deltas
            new_pred_array=cur_x_test_base_vals_for_pred +  cur_predicted_timestep_array_of_deltas_to_add
            # print("new_pred_array: %s" %str(new_pred_array.shape))
            cur_sequence_of_calculated_timestep_vals_not_deltas.append(new_pred_array)
            cur_x_test_base_vals_for_pred=new_pred_array.copy()


        #at the end of an iteration i, we must make a sequence array out of all the timestep features we just calculated
        #then, we must re-normalize them to make a new x_test array
        new_x_test = np.asarray(cur_sequence_of_calculated_timestep_vals_not_deltas)
        # print("Shape of new_x_test: %s" %str(new_x_test.shape))

        #Do not normalize, we want these actual count results
        all_predicted_sequences.append(new_x_test.copy())
        print("Current number of predicted sequences: %d out of %d" %(len(all_predicted_sequences),num_sequences_to_simulate))

        if scaler_flag==True:
            #Now we have to normalize x test
            new_x_test=scaler.transform(new_x_test).reshape((1,new_x_test.shape[0],new_x_test.shape[1] ))

        #predict
        prediction_deltas=model.predict(new_x_test)

        #make pred array 2d
        prediction_deltas=prediction_deltas.reshape((prediction_deltas.shape[0]*prediction_deltas.shape[1],prediction_deltas.shape[2]))

        #current starting point = last day in the sequence
        new_x_test=new_x_test.reshape((new_x_test.shape[1],new_x_test.shape[2] ))
        cur_x_test_base_vals_for_pred=new_x_test[num_timesteps_in_a_sequence-1:, ]

        if scaler_flag==True:
            #we must inverse transform our new text x condition so the deltas can be added to it properly
            cur_x_test_base_vals_for_pred=scaler.inverse_transform(cur_x_test_base_vals_for_pred)
        cur_x_test_base_vals_for_pred=cur_x_test_base_vals_for_pred.flatten()
        # print("Cur base shape: %s" %str(cur_x_test_base_vals_for_pred.shape))

    #convert our new predicted list into array
    all_predicted_sequences=np.asarray(all_predicted_sequences)
    print("all_predicted_sequences shape: %s" %str(all_predicted_sequences.shape))

    print("############ Simulation complete! ################")
    return all_predicted_sequences

#date to slice df by start and end date
# def config_df_by_dates(df,start_date,end_date,time_col="nodeTime"):
#     df[time_col]=pd.to_datetime(df[time_col], utc=True)
#     df=df.set_index(time_col)
#     df = df[(df.index >= start_date) & (df.index <= end_date)]
#     df=df.reset_index(drop=False)
#     return df

def simulate_with_separate_models_and_DOW(x_test, model_list, FEATURES_OF_INTEREST,scaler, num_sequences_to_simulate,scaler_flag=True):
    #sanity checking tag means these are arrays we should not have access to during the challenge
    print("############ Now simulating on initial condition... ################")

    #setup dow dict
    dow_dict={
    0: [1,0,0,0,0,0,0],
    1: [0,1,0,0,0,0,0],
    2: [0,0,1,0,0,0,0],
    3: [0,0,0,1,0,0,0],
    4: [0,0,0,0,1,0,0],
    5: [0,0,0,0,0,1,0],
    6: [0,0,0,0,0,0,1]
    }

    x_test=x_test.astype("float32")


    #DIFFERENCE THRESHOLD
    #SANITY CHECK PARAM
    DIFFERENCE_THRESHOLD=0.1

    num_timesteps_in_a_sequence=x_test.shape[1]

    #create init condit array
    x_init_condit=x_test[:1,:,:].astype("float32")
    print("Shape of x_init_condit: %s" %str(x_init_condit.shape))

    #get first sequence of predictions
    separated_pred_deltas_per_feature=[]

    for i,model in enumerate(model_list):
        cur_ft=FEATURES_OF_INTEREST[i]
        print("Predicting %s" %cur_ft)

        cur_pred_deltas=model.predict(x_init_condit)
        separated_pred_deltas_per_feature.append(cur_pred_deltas)

    prediction_deltas=np.concatenate(separated_pred_deltas_per_feature,axis=2)
    print("\nCurrent shape of the prediction_deltas: %s" %str(prediction_deltas.shape))


    #make x_init_condit 2d
    #this array has the DOW values
    x_init_condit=x_init_condit.reshape((x_init_condit.shape[0]*x_init_condit.shape[1],x_init_condit.shape[2]))

    #make pred array 2d
    #these deltas do not have the DOW
    prediction_deltas=prediction_deltas.reshape((prediction_deltas.shape[0]*prediction_deltas.shape[1],prediction_deltas.shape[2]))
    # print("prediction_deltas shape: %s" %str(prediction_deltas.shape))

    #current starting point = last day in the sequence
    #2d array with shape (1,num_features_in_timestep+7)
    #DOW is here
    cur_x_test_base_vals_for_pred=x_init_condit[num_timesteps_in_a_sequence-1:, ]
    # print("Shape of cur_x_test_base_vals_for_pred: %s" %str(cur_x_test_base_vals_for_pred.shape))

    #we must inverse transform our initial condition so the deltas can be added to it properly
    #DOW
    if scaler_flag==True:
        cur_x_test_base_vals_for_pred=scaler.inverse_transform(cur_x_test_base_vals_for_pred)
        # cur_x_test_base_vals_for_pred = properly_denormalize_2d_data(cur_x_test_base_vals_for_pred, scaler)

    #for the base vals

    cur_x_test_base_vals_for_pred=cur_x_test_base_vals_for_pred.flatten()
    # print("Shape of flattened cur_x_test_base_vals_for_pred: %s" %str(cur_x_test_base_vals_for_pred.shape))

    #remove days of the week
    cur_dow_array=cur_x_test_base_vals_for_pred[-7:]
    # print("cur_dow_array: %s" %str(cur_dow_array))
    cur_x_test_base_vals_for_pred=cur_x_test_base_vals_for_pred[:-7]
    # print("Cur base shape: %s" %str(cur_x_test_base_vals_for_pred.shape))

    #FOR SANITY CHECKING
    # gt_cur_x_test_base_vals=cur_x_test_base_vals_for_pred.copy()

    #save all predicted sequences
    all_predicted_sequences=[]

    #get fts in a time step
    for i in range(num_sequences_to_simulate):

        #save your new cur sequence of predictions here
        cur_sequence_of_calculated_timestep_vals_not_deltas=[]

        #might have to add back 1
        for j in range(num_timesteps_in_a_sequence):

            #get cur timestep of predicted deltas
            cur_predicted_timestep_array_of_deltas_to_add=prediction_deltas[j]
            # print("cur_predicted_timestep_array_of_deltas_to_add shape: %s" %str(cur_predicted_timestep_array_of_deltas_to_add.shape))

            #add pred deltas
            new_pred_array=cur_x_test_base_vals_for_pred +  cur_predicted_timestep_array_of_deltas_to_add
            # print("new_pred_array: %s" %str(new_pred_array.shape))

            #now we must get the next dow to add back
            dow=np.argmax(cur_dow_array)
            # print("DOW is %d" %dow)

            #next dow
            next_dow=(dow + 1)%7
            # print("Next dow: %d" %next_dow)

            #next dow as array
            next_dow_as_array=dow_dict[next_dow]
            # print("Next dow as array: %s" %str(next_dow_as_array))

            #update current
            cur_dow_array=next_dow_as_array

            #before we concat, make this new pred array without the dow available for the next round of delta adding
            cur_x_test_base_vals_for_pred=new_pred_array.copy()

            #concat this next dow with new pred array
            new_pred_array=np.concatenate((new_pred_array,next_dow_as_array))

            #append new pred array with dow to our current list of predicted timestep arrays
            cur_sequence_of_calculated_timestep_vals_not_deltas.append(new_pred_array)



        #at the end of an iteration i (a sequence) , we must make a sequence array out of all the timestep features we just calculated
        #then, we must re-normalize them to make a new x_test array
        new_x_test = np.asarray(cur_sequence_of_calculated_timestep_vals_not_deltas)
        # print("Shape of new_x_test: %s" %str(new_x_test.shape))

        #Do not normalize, we want these actual count results
        all_predicted_sequences.append(new_x_test.copy())
        print("Current number of predicted sequences: %d out of %d" %(len(all_predicted_sequences),num_sequences_to_simulate))

        #Now we have to normalize x test
        if scaler_flag==True:
            new_x_test=scaler.transform(new_x_test)
            # new_x_test = properly_normalize_2d_data(new_x_test, scaler)
        new_x_test=new_x_test.reshape((1,new_x_test.shape[0],new_x_test.shape[1] ))

        #get first sequence of predictions
        separated_pred_deltas_per_feature=[]

        for i,model in enumerate(model_list):
            cur_ft=FEATURES_OF_INTEREST[i]
            print("Predicting %s" %cur_ft)

            cur_pred_deltas=model.predict(new_x_test)
            separated_pred_deltas_per_feature.append(cur_pred_deltas)

        prediction_deltas=np.concatenate(separated_pred_deltas_per_feature, axis=2)
        print("Current shape of the prediction_deltas: %s" %str(prediction_deltas.shape))


        #make pred array 2d
        prediction_deltas=prediction_deltas.reshape((prediction_deltas.shape[0]*prediction_deltas.shape[1],prediction_deltas.shape[2]))

        #current starting point = last day in the sequence
        new_x_test=new_x_test.reshape((new_x_test.shape[1],new_x_test.shape[2] ))
        cur_x_test_base_vals_for_pred=new_x_test[num_timesteps_in_a_sequence-1:, ]

        #we must inverse transform our new text x condition so the deltas can be added to it properly
        if scaler_flag==True:
            cur_x_test_base_vals_for_pred=scaler.inverse_transform(cur_x_test_base_vals_for_pred)
            # cur_x_test_base_vals_for_pred = properly_denormalize_2d_data(cur_x_test_base_vals_for_pred, scaler)
        cur_x_test_base_vals_for_pred=cur_x_test_base_vals_for_pred.flatten()
        # print("Cur base shape: %s" %str(cur_x_test_base_vals_for_pred.shape))

        #remove days of the week
        saved_dow_array=cur_x_test_base_vals_for_pred[-7:]
        print("saved_dow_array: %s" %str(saved_dow_array))
        cur_x_test_base_vals_for_pred=cur_x_test_base_vals_for_pred[:-7]

    #convert our new predicted list into array
    all_predicted_sequences=np.asarray(all_predicted_sequences)
    print("all_predicted_sequences shape")
    print(all_predicted_sequences.shape)
    all_predicted_sequences=all_predicted_sequences[:,:,:-7]
    print("all_predicted_sequences shape: %s" %str(all_predicted_sequences.shape))

    print("############ Simulation complete! ################")
    return all_predicted_sequences

def simulate_with_separate_models_and_DOW_for_aggregated_daily_model(x_test, model_list, FEATURES_OF_INTEREST,scaler, num_sequences_to_simulate,scaler_flag=True):
    #sanity checking tag means these are arrays we should not have access to during the challenge
    print("############ Now simulating on initial condition... ################")

    #setup dow dict
    dow_dict={
    0: [1,0,0,0,0,0,0],
    1: [0,1,0,0,0,0,0],
    2: [0,0,1,0,0,0,0],
    3: [0,0,0,1,0,0,0],
    4: [0,0,0,0,1,0,0],
    5: [0,0,0,0,0,1,0],
    6: [0,0,0,0,0,0,1]
    }

    x_test=x_test.astype("float32")


    #DIFFERENCE THRESHOLD
    #SANITY CHECK PARAM
    DIFFERENCE_THRESHOLD=0.1

    num_timesteps_in_a_sequence=x_test.shape[1]

    #create init condit array
    x_init_condit=x_test[:1,:,:].astype("float32")
    print("Shape of x_init_condit: %s" %str(x_init_condit.shape))

    #get first sequence of predictions
    separated_pred_deltas_per_feature=[]

    for i,model in enumerate(model_list):
        cur_ft=FEATURES_OF_INTEREST[i]
        print("Predicting %s" %cur_ft)

        cur_pred_deltas=model.predict(x_init_condit)
        separated_pred_deltas_per_feature.append(cur_pred_deltas)

    prediction_deltas=np.concatenate(separated_pred_deltas_per_feature,axis=2)
    print("\nCurrent shape of the prediction_deltas: %s" %str(prediction_deltas.shape))


    #make x_init_condit 2d
    #this array has the DOW values
    x_init_condit=x_init_condit.reshape((x_init_condit.shape[0]*x_init_condit.shape[1],x_init_condit.shape[2]))

    #make pred array 2d
    #these deltas do not have the DOW
    prediction_deltas=prediction_deltas.reshape((prediction_deltas.shape[0]*prediction_deltas.shape[1],prediction_deltas.shape[2]))
    # print("prediction_deltas shape: %s" %str(prediction_deltas.shape))

    #current starting point = last day in the sequence
    #2d array with shape (1,num_features_in_timestep+7)
    #DOW is here
    cur_x_test_base_vals_for_pred=x_init_condit[num_timesteps_in_a_sequence-1:, ]
    # print("Shape of cur_x_test_base_vals_for_pred: %s" %str(cur_x_test_base_vals_for_pred.shape))

    #we must inverse transform our initial condition so the deltas can be added to it properly
    #DOW
    if scaler_flag==True:
        cur_x_test_base_vals_for_pred=scaler.inverse_transform(cur_x_test_base_vals_for_pred)
        # cur_x_test_base_vals_for_pred = properly_denormalize_2d_data(cur_x_test_base_vals_for_pred, scaler)

    #for the base vals

    cur_x_test_base_vals_for_pred=cur_x_test_base_vals_for_pred.flatten()
    # print("Shape of flattened cur_x_test_base_vals_for_pred: %s" %str(cur_x_test_base_vals_for_pred.shape))

    #remove days of the week
    cur_dow_array=cur_x_test_base_vals_for_pred[-7:]
    # print("cur_dow_array: %s" %str(cur_dow_array))
    cur_x_test_base_vals_for_pred=cur_x_test_base_vals_for_pred[:-7]
    # print("Cur base shape: %s" %str(cur_x_test_base_vals_for_pred.shape))

    #FOR SANITY CHECKING
    # gt_cur_x_test_base_vals=cur_x_test_base_vals_for_pred.copy()

    #save all predicted sequences
    all_predicted_sequences=[]

    #get fts in a time step
    for i in range(num_sequences_to_simulate):

        #save your new cur sequence of predictions here
        cur_sequence_of_calculated_timestep_vals_not_deltas=[]

        #might have to add back 1
        for j in range(num_timesteps_in_a_sequence):

            #get cur timestep of predicted deltas
            cur_predicted_timestep_array_of_deltas_to_add=prediction_deltas[j]
            # print("cur_predicted_timestep_array_of_deltas_to_add shape: %s" %str(cur_predicted_timestep_array_of_deltas_to_add.shape))

            #add pred deltas
            new_pred_array=cur_x_test_base_vals_for_pred +  cur_predicted_timestep_array_of_deltas_to_add
            # print("new_pred_array: %s" %str(new_pred_array.shape))

            #now we must get the next dow to add back
            # dow=np.argmax(cur_dow_array)
            # print("DOW is %d" %dow)

            #next dow
            # next_dow=(dow + 1)%7
            # print("Next dow: %d" %next_dow)

            #next dow as array
            # next_dow_as_array=dow_dict[next_dow]
            next_dow_as_array = cur_dow_array
            # print("Next dow as array: %s" %str(next_dow_as_array))

            #update current
            # cur_dow_array=next_dow_as_array

            #before we concat, make this new pred array without the dow available for the next round of delta adding
            cur_x_test_base_vals_for_pred=new_pred_array.copy()

            #concat this next dow with new pred array
            new_pred_array=np.concatenate((new_pred_array,next_dow_as_array))

            #append new pred array with dow to our current list of predicted timestep arrays
            cur_sequence_of_calculated_timestep_vals_not_deltas.append(new_pred_array)



        #at the end of an iteration i (a sequence) , we must make a sequence array out of all the timestep features we just calculated
        #then, we must re-normalize them to make a new x_test array
        new_x_test = np.asarray(cur_sequence_of_calculated_timestep_vals_not_deltas)
        # print("Shape of new_x_test: %s" %str(new_x_test.shape))

        #Do not normalize, we want these actual count results
        all_predicted_sequences.append(new_x_test.copy())
        print("Current number of predicted sequences: %d out of %d" %(len(all_predicted_sequences),num_sequences_to_simulate))

        #Now we have to normalize x test
        if scaler_flag==True:
            new_x_test=scaler.transform(new_x_test)
            # new_x_test = properly_normalize_2d_data(new_x_test, scaler)
        new_x_test=new_x_test.reshape((1,new_x_test.shape[0],new_x_test.shape[1] ))

        #get first sequence of predictions
        separated_pred_deltas_per_feature=[]

        for i,model in enumerate(model_list):
            cur_ft=FEATURES_OF_INTEREST[i]
            print("Predicting %s" %cur_ft)

            cur_pred_deltas=model.predict(new_x_test)
            separated_pred_deltas_per_feature.append(cur_pred_deltas)

        prediction_deltas=np.concatenate(separated_pred_deltas_per_feature, axis=2)
        print("Current shape of the prediction_deltas: %s" %str(prediction_deltas.shape))


        #make pred array 2d
        prediction_deltas=prediction_deltas.reshape((prediction_deltas.shape[0]*prediction_deltas.shape[1],prediction_deltas.shape[2]))

        #current starting point = last day in the sequence
        new_x_test=new_x_test.reshape((new_x_test.shape[1],new_x_test.shape[2] ))
        cur_x_test_base_vals_for_pred=new_x_test[num_timesteps_in_a_sequence-1:, ]

        #we must inverse transform our new text x condition so the deltas can be added to it properly
        if scaler_flag==True:
            cur_x_test_base_vals_for_pred=scaler.inverse_transform(cur_x_test_base_vals_for_pred)
            # cur_x_test_base_vals_for_pred = properly_denormalize_2d_data(cur_x_test_base_vals_for_pred, scaler)
        cur_x_test_base_vals_for_pred=cur_x_test_base_vals_for_pred.flatten()
        # print("Cur base shape: %s" %str(cur_x_test_base_vals_for_pred.shape))

        #remove days of the week
        saved_dow_array=cur_x_test_base_vals_for_pred[-7:]
        print("saved_dow_array: %s" %str(saved_dow_array))
        cur_x_test_base_vals_for_pred=cur_x_test_base_vals_for_pred[:-7]

    #convert our new predicted list into array
    all_predicted_sequences=np.asarray(all_predicted_sequences)
    all_predicted_sequences=all_predicted_sequences[:,:,:-7]
    print("all_predicted_sequences shape: %s" %str(all_predicted_sequences.shape))

    print("############ Simulation complete! ################")
    return all_predicted_sequences

def simulate_with_separate_models_and_MOY(x_test, model_list, FEATURES_OF_INTEREST,scaler, num_sequences_to_simulate,scaler_flag=True):

    #setup moy dict
    #month of year -> moy
    moy_dict={
    0: [1,0,0,0,0,0,0,0,0,0,0,0],
    1: [0,1,0,0,0,0,0,0,0,0,0,0],
    2: [0,0,1,0,0,0,0,0,0,0,0,0],
    3: [0,0,0,1,0,0,0,0,0,0,0,0],
    4: [0,0,0,0,1,0,0,0,0,0,0,0],
    5: [0,0,0,0,0,1,0,0,0,0,0,0],
    6: [0,0,0,0,0,0,1,0,0,0,0,0],
    7: [0,0,0,0,0,0,0,1,0,0,0,0],
    8: [0,0,0,0,0,0,0,0,1,0,0,0],
    9: [0,0,0,0,0,0,0,0,0,1,0,0],
    10:[0,0,0,0,0,0,0,0,0,0,1,0],
    11:[0,0,0,0,0,0,0,0,0,0,0,1],
    }




    #sanity checking tag means these are arrays we should not have access to during the challenge
    print("############ Now simulating on initial condition... ################")

    #MOY = MONTH OF YEAR
    x_test=x_test.astype("float32")

    NUM_MONTHS=12


    #DIFFERENCE THRESHOLD
    #SANITY CHECK PARAM
    DIFFERENCE_THRESHOLD=0.1

    num_timesteps_in_a_sequence=x_test.shape[1]

    #create init condit array
    x_init_condit=x_test[:1,:,:].astype("float32")
    print("Shape of x_init_condit: %s" %str(x_init_condit.shape))

    #get first sequence of predictions
    separated_pred_deltas_per_feature=[]

    for i,model in enumerate(model_list):
        cur_ft=FEATURES_OF_INTEREST[i]
        print("Predicting %s" %cur_ft)

        cur_pred_deltas=model.predict(x_init_condit)
        separated_pred_deltas_per_feature.append(cur_pred_deltas)

    prediction_deltas=np.concatenate(separated_pred_deltas_per_feature,axis=2)
    print("\nCurrent shape of the prediction_deltas: %s" %str(prediction_deltas.shape))


    #make x_init_condit 2d
    #this array has the moy values
    x_init_condit=x_init_condit.reshape((x_init_condit.shape[0]*x_init_condit.shape[1],x_init_condit.shape[2]))

    #make pred array 2d
    #these deltas do not have the moy
    prediction_deltas=prediction_deltas.reshape((prediction_deltas.shape[0]*prediction_deltas.shape[1],prediction_deltas.shape[2]))
    # print("prediction_deltas shape: %s" %str(prediction_deltas.shape))

    #current starting point = last day in the sequence
    #2d array with shape (1,num_features_in_timestep+7)
    #moy is here
    cur_x_test_base_vals_for_pred=x_init_condit[num_timesteps_in_a_sequence-1:, ]
    # print("Shape of cur_x_test_base_vals_for_pred: %s" %str(cur_x_test_base_vals_for_pred.shape))

    #we must inverse transform our initial condition so the deltas can be added to it properly
    #moy
    if scaler_flag==True:
        cur_x_test_base_vals_for_pred=scaler.inverse_transform(cur_x_test_base_vals_for_pred)
        # cur_x_test_base_vals_for_pred = properly_denormalize_2d_data(cur_x_test_base_vals_for_pred, scaler)

    #for the base vals

    cur_x_test_base_vals_for_pred=cur_x_test_base_vals_for_pred.flatten()
    # print("Shape of flattened cur_x_test_base_vals_for_pred: %s" %str(cur_x_test_base_vals_for_pred.shape))

    #remove days of the week
    cur_moy_array=cur_x_test_base_vals_for_pred[-NUM_MONTHS:]
    # print("cur_moy_array: %s" %str(cur_moy_array))
    cur_x_test_base_vals_for_pred=cur_x_test_base_vals_for_pred[:-NUM_MONTHS]
    # print("Cur base shape: %s" %str(cur_x_test_base_vals_for_pred.shape))

    #FOR SANITY CHECKING
    # gt_cur_x_test_base_vals=cur_x_test_base_vals_for_pred.copy()

    #save all predicted sequences
    all_predicted_sequences=[]

    #get fts in a time step
    for i in range(num_sequences_to_simulate):

        #save your new cur sequence of predictions here
        cur_sequence_of_calculated_timestep_vals_not_deltas=[]

        #might have to add back 1
        for j in range(num_timesteps_in_a_sequence):

            #get cur timestep of predicted deltas
            cur_predicted_timestep_array_of_deltas_to_add=prediction_deltas[j]
            # print("cur_predicted_timestep_array_of_deltas_to_add shape: %s" %str(cur_predicted_timestep_array_of_deltas_to_add.shape))

            #add pred deltas
            new_pred_array=cur_x_test_base_vals_for_pred +  cur_predicted_timestep_array_of_deltas_to_add
            # print("new_pred_array: %s" %str(new_pred_array.shape))

            #now we must get the next moy to add back
            moy=np.argmax(cur_moy_array)
            # print("moy is %d" %moy)

            #next moy
            next_moy=(moy + 1)%NUM_MONTHS
            # print("Next moy: %d" %next_moy)

            #next moy as array
            next_moy_as_array=moy_dict[next_moy]
            # print("Next moy as array: %s" %str(next_moy_as_array))

            #update current
            cur_moy_array=next_moy_as_array

            #before we concat, make this new pred array without the moy available for the next round of delta adding
            cur_x_test_base_vals_for_pred=new_pred_array.copy()

            #concat this next moy with new pred array
            new_pred_array=np.concatenate((new_pred_array,next_moy_as_array))

            #append new pred array with moy to our current list of predicted timestep arrays
            cur_sequence_of_calculated_timestep_vals_not_deltas.append(new_pred_array)



        #at the end of an iteration i (a sequence) , we must make a sequence array out of all the timestep features we just calculated
        #then, we must re-normalize them to make a new x_test array
        new_x_test = np.asarray(cur_sequence_of_calculated_timestep_vals_not_deltas)
        # print("Shape of new_x_test: %s" %str(new_x_test.shape))

        #Do not normalize, we want these actual count results
        all_predicted_sequences.append(new_x_test.copy())
        print("Current number of predicted sequences: %d out of %d" %(len(all_predicted_sequences),num_sequences_to_simulate))

        #Now we have to normalize x test
        if scaler_flag==True:
            new_x_test=scaler.transform(new_x_test)
            # new_x_test = properly_normalize_2d_data(new_x_test, scaler)
        new_x_test=new_x_test.reshape((1,new_x_test.shape[0],new_x_test.shape[1] ))

        #get first sequence of predictions
        separated_pred_deltas_per_feature=[]

        for i,model in enumerate(model_list):
            cur_ft=FEATURES_OF_INTEREST[i]
            print("Predicting %s" %cur_ft)

            cur_pred_deltas=model.predict(new_x_test)
            separated_pred_deltas_per_feature.append(cur_pred_deltas)

        prediction_deltas=np.concatenate(separated_pred_deltas_per_feature, axis=2)
        print("Current shape of the prediction_deltas: %s" %str(prediction_deltas.shape))


        #make pred array 2d
        prediction_deltas=prediction_deltas.reshape((prediction_deltas.shape[0]*prediction_deltas.shape[1],prediction_deltas.shape[2]))

        #current starting point = last day in the sequence
        new_x_test=new_x_test.reshape((new_x_test.shape[1],new_x_test.shape[2] ))
        cur_x_test_base_vals_for_pred=new_x_test[num_timesteps_in_a_sequence-1:, ]

        #we must inverse transform our new text x condition so the deltas can be added to it properly
        if scaler_flag==True:
            cur_x_test_base_vals_for_pred=scaler.inverse_transform(cur_x_test_base_vals_for_pred)
            # cur_x_test_base_vals_for_pred = properly_denormalize_2d_data(cur_x_test_base_vals_for_pred, scaler)
        cur_x_test_base_vals_for_pred=cur_x_test_base_vals_for_pred.flatten()
        # print("Cur base shape: %s" %str(cur_x_test_base_vals_for_pred.shape))

        #remove days of the week
        saved_moy_array=cur_x_test_base_vals_for_pred[-NUM_MONTHS:]
        print("saved_moy_array: %s" %str(saved_moy_array))
        cur_x_test_base_vals_for_pred=cur_x_test_base_vals_for_pred[:-NUM_MONTHS]

    #convert our new predicted list into array
    all_predicted_sequences=np.asarray(all_predicted_sequences)
    all_predicted_sequences=all_predicted_sequences[:,:,:-NUM_MONTHS]
    print("all_predicted_sequences shape: %s" %str(all_predicted_sequences.shape))

    print("############ Simulation complete! ################")
    return all_predicted_sequences


def predict_one_target_with_GT_and_no_DOW(x_test, model,TARGET_FEATURE_LIST, TARGET_FT_INDEX_IN_SOURCE_ARRAY,scaler,num_sequences_to_simulate,scaler_flag=True):
    # print("\n############ Now simulating on initial condition... ################")

    # num_timesteps_in_a_sequence=x_test.shape[1]

    # #get first sequence of predictions
    # prediction_deltas=model.predict(x_test)
    # print("\nCurrent shape of the prediction_deltas: %s" %str(prediction_deltas.shape))

    # #reshape
    # #make x_test 2d
    # x_test=x_test.reshape((x_test.shape[0]*x_test.shape[1],x_test.shape[2]))

    # #make pred array 2d
    # prediction_deltas=prediction_deltas.reshape((prediction_deltas.shape[0]*prediction_deltas.shape[1],prediction_deltas.shape[2]))

    # #now we make values
    # predictions = []

    # for i in range(num_sequences_to_simulate):
    #     for j in range(num_timesteps_in_a_sequence):pseudo_predict_with_DOW_and_GT_and_separate_models
    print("\n############ Now simulating on initial condition... ################")

    num_timesteps_in_a_sequence=x_test.shape[1]

    #create init condit array
    x_init_condit=x_test[:1,:,:].copy().astype("float32")

    #get first sequence of predictions

    prediction_deltas=model.predict(x_init_condit)
    print("\nCurrent shape of the prediction_deltas: %s" %str(prediction_deltas.shape))

    #make x_init_condit 2d
    x_init_condit=x_init_condit.reshape((x_init_condit.shape[0]*x_init_condit.shape[1],x_init_condit.shape[2]))

    #make pred array 2d
    prediction_deltas=prediction_deltas.reshape((prediction_deltas.shape[0]*prediction_deltas.shape[1],prediction_deltas.shape[2]))
    # print("prediction_deltas shape: %s" %str(prediction_deltas.shape))

    #current starting point = last day in the sequence
    cur_x_test_base_vals_for_pred=x_init_condit[num_timesteps_in_a_sequence-1:,:].copy()


    if scaler_flag==True:
        #we must inverse transform our initial condition so the deltas can be added to it properly
        cur_x_test_base_vals_for_pred=scaler.inverse_transform(cur_x_test_base_vals_for_pred)

    #only want target val
    cur_x_test_base_vals_for_pred=cur_x_test_base_vals_for_pred[:,TARGET_FT_INDEX_IN_SOURCE_ARRAY]

    cur_x_test_base_vals_for_pred=cur_x_test_base_vals_for_pred.flatten()
    print("Cur base shape after flattening: %s" %str(cur_x_test_base_vals_for_pred.shape))

    # #FOR SANITY CHECKING
    # gt_cur_x_test_base_vals=cur_x_test_base_vals_for_pred.copy()

    #save all predicted sequences
    all_predicted_sequences=[]

    #get fts in a time step
    for i in range(num_sequences_to_simulate):

        #save your new cur sequence of predictions here
        cur_sequence_of_calculated_timestep_vals_not_deltas=[]

        #loop through all timesteps in a sequence
        for j in range(num_timesteps_in_a_sequence):

            #get cur timestep of predicted deltas
            cur_predicted_timestep_array_of_deltas_to_add=prediction_deltas[j]
            # print("cur_predicted_timestep_array_of_deltas_to_add shape: %s" %str(cur_predicted_timestep_array_of_deltas_to_add.shape))

            #add pred deltas
            new_pred_array=cur_x_test_base_vals_for_pred +  cur_predicted_timestep_array_of_deltas_to_add
            # print("new_pred_array: %s" %str(new_pred_array.shape))
            cur_sequence_of_calculated_timestep_vals_not_deltas.append(new_pred_array)
            cur_x_test_base_vals_for_pred=new_pred_array.copy()


        #at the end of an iteration i, we must make a sequence array out of all the timestep features we just calculated
        #then, we make a new x_test array
        #since we're suing ground truth from x_test, there's no need to normalize
        new_x_test = x_test[i+1:i+2,:,:].astype("float32")
        # new_x_test=new_x_test.reshape((new_x_test.shape[0]*new_x_test.shape[1],new_x_test.shape[2]))
        # print("Shape of new_x_test: %s" %str(new_x_test.shape))

        #Do not normalize, we want these actual count results
        all_predicted_sequences.append(np.asarray(cur_sequence_of_calculated_timestep_vals_not_deltas).copy())
        print("\nCurrent number of predicted sequences: %d out of %d" %(len(all_predicted_sequences),num_sequences_to_simulate))
        if len(all_predicted_sequences) == num_sequences_to_simulate:
            break

        #get first sequence of predictions
        prediction_deltas=model.predict(new_x_test)
        print("\nCurrent shape of the prediction_deltas: %s" %str(prediction_deltas.shape))

        #make pred array 2d
        prediction_deltas=prediction_deltas.reshape((prediction_deltas.shape[0]*prediction_deltas.shape[1],prediction_deltas.shape[2]))

        #current starting point = last day in the sequence
        new_x_test=new_x_test.reshape((new_x_test.shape[1],new_x_test.shape[2] ))

        cur_x_test_base_vals_for_pred=new_x_test[num_timesteps_in_a_sequence-1:,:].copy()

        if scaler_flag==True:
            #we must inverse transform our new text x condition so the deltas can be added to it properly
            cur_x_test_base_vals_for_pred=scaler.inverse_transform(cur_x_test_base_vals_for_pred)

        #only want target val
        cur_x_test_base_vals_for_pred=cur_x_test_base_vals_for_pred[:,TARGET_FT_INDEX_IN_SOURCE_ARRAY]

        cur_x_test_base_vals_for_pred=cur_x_test_base_vals_for_pred.flatten()
        print("Cur base shape after flattening: %s" %str(cur_x_test_base_vals_for_pred.shape))

    #convert our new predicted list into array
    all_predicted_sequences=np.asarray(all_predicted_sequences)
    print("all_predicted_sequences shape: %s" %str(all_predicted_sequences.shape))

    print("############ Simulation complete! ################")
    return all_predicted_sequences



def predict_with_DOW_and_GT_and_separate_models(x_test, model_list, FEATURES_OF_INTEREST, scaler, num_sequences_to_simulate,scaler_flag=True):
    #sanity checking tag means these are arrays we should not have access to during the challenge
    print("\n############ Now simulating on initial condition... ################")

    num_timesteps_in_a_sequence=x_test.shape[1]

    #create init condit array
    x_init_condit=x_test[:1,:,:].copy().astype("float32")

    #get first sequence of predictions
    separated_pred_deltas_per_feature=[]

    for i,model in enumerate(model_list):
        cur_ft=FEATURES_OF_INTEREST[i]
        print("Predicting %s" %cur_ft)

        cur_pred_deltas=model.predict(x_init_condit)
        separated_pred_deltas_per_feature.append(cur_pred_deltas)

    prediction_deltas=np.concatenate(separated_pred_deltas_per_feature,axis=2)
    print("\nCurrent shape of the prediction_deltas: %s" %str(prediction_deltas.shape))

    #make x_init_condit 2d
    x_init_condit=x_init_condit.reshape((x_init_condit.shape[0]*x_init_condit.shape[1],x_init_condit.shape[2]))

    #make pred array 2d
    prediction_deltas=prediction_deltas.reshape((prediction_deltas.shape[0]*prediction_deltas.shape[1],prediction_deltas.shape[2]))
    # print("prediction_deltas shape: %s" %str(prediction_deltas.shape))

    #current starting point = last day in the sequence
    cur_x_test_base_vals_for_pred_with_DOW=x_init_condit[num_timesteps_in_a_sequence-1:,:].copy()
    # cur_x_test_base_vals_for_pred=x_init_condit[num_timesteps_in_a_sequence-1:,:-7 ]
    # print("Shape of cur_x_test_base_vals_for_pred after removing days of the week: %s" %str(cur_x_test_base_vals_for_pred.shape))


    if scaler_flag==True:
        #we must inverse transform our initial condition so the deltas can be added to it properly
        cur_x_test_base_vals_for_pred_with_DOW=scaler.inverse_transform(cur_x_test_base_vals_for_pred_with_DOW)

    #now we must remove the DOW
    cur_x_test_base_vals_for_pred= cur_x_test_base_vals_for_pred_with_DOW[:,:-7].copy()
    print("Shape of cur_x_test_base_vals_for_pred after removing days of the week: %s" %str(cur_x_test_base_vals_for_pred.shape))

    cur_x_test_base_vals_for_pred=cur_x_test_base_vals_for_pred.flatten()
    print("Cur base shape after flattening: %s" %str(cur_x_test_base_vals_for_pred.shape))

    # #FOR SANITY CHECKING
    # gt_cur_x_test_base_vals=cur_x_test_base_vals_for_pred.copy()

    #save all predicted sequences
    all_predicted_sequences=[]

    #get fts in a time step
    for i in range(num_sequences_to_simulate):

        #save your new cur sequence of predictions here
        cur_sequence_of_calculated_timestep_vals_not_deltas=[]

        #loop through all timesteps in a sequence
        for j in range(num_timesteps_in_a_sequence):

            #get cur timestep of predicted deltas
            cur_predicted_timestep_array_of_deltas_to_add=prediction_deltas[j]
            # print("cur_predicted_timestep_array_of_deltas_to_add shape: %s" %str(cur_predicted_timestep_array_of_deltas_to_add.shape))

            #add pred deltas
            new_pred_array=cur_x_test_base_vals_for_pred +  cur_predicted_timestep_array_of_deltas_to_add
            # print("new_pred_array: %s" %str(new_pred_array.shape))
            cur_sequence_of_calculated_timestep_vals_not_deltas.append(new_pred_array)
            cur_x_test_base_vals_for_pred=new_pred_array.copy()


        #at the end of an iteration i, we must make a sequence array out of all the timestep features we just calculated
        #then, we make a new x_test array
        #since we're suing ground truth from x_test, there's no need to normalize
        new_x_test = x_test[i+1:i+2,:,:].astype("float32")
        # new_x_test=new_x_test.reshape((new_x_test.shape[0]*new_x_test.shape[1],new_x_test.shape[2]))
        # print("Shape of new_x_test: %s" %str(new_x_test.shape))

        #Do not normalize, we want these actual count results
        all_predicted_sequences.append(np.asarray(cur_sequence_of_calculated_timestep_vals_not_deltas).copy())
        print("\nCurrent number of predicted sequences: %d out of %d" %(len(all_predicted_sequences),num_sequences_to_simulate))
        if len(all_predicted_sequences) == num_sequences_to_simulate:
            break

        #Now we have to normalize x test
        # new_x_test=scaler.transform(new_x_test).reshape((1,new_x_test.shape[0],new_x_test.shape[1] ))

        # #predict
        # prediction_deltas=model.predict(new_x_test)

        #get first sequence of predictions
        separated_pred_deltas_per_feature=[]

        for i,model in enumerate(model_list):
            cur_ft=FEATURES_OF_INTEREST[i]
            print("Predicting %s" %cur_ft)

            cur_pred_deltas=model.predict(new_x_test)
            separated_pred_deltas_per_feature.append(cur_pred_deltas)

        prediction_deltas=np.concatenate(separated_pred_deltas_per_feature, axis=2)
        print("Current shape of the prediction_deltas: %s" %str(prediction_deltas.shape))

        #make pred array 2d
        prediction_deltas=prediction_deltas.reshape((prediction_deltas.shape[0]*prediction_deltas.shape[1],prediction_deltas.shape[2]))

        #current starting point = last day in the sequence
        new_x_test=new_x_test.reshape((new_x_test.shape[1],new_x_test.shape[2] ))

        cur_x_test_base_vals_for_pred_with_DOW=new_x_test[num_timesteps_in_a_sequence-1:,:].copy()

        if scaler_flag==True:
            #we must inverse transform our new text x condition so the deltas can be added to it properly
            cur_x_test_base_vals_for_pred_with_DOW=scaler.inverse_transform(cur_x_test_base_vals_for_pred_with_DOW)

        #now we must remove the DOW
        cur_x_test_base_vals_for_pred= cur_x_test_base_vals_for_pred_with_DOW[:,:-7].copy()
        print("Shape of cur_x_test_base_vals_for_pred after removing days of the week: %s" %str(cur_x_test_base_vals_for_pred.shape))

        cur_x_test_base_vals_for_pred=cur_x_test_base_vals_for_pred.flatten()
        print("Cur base shape after flattening: %s" %str(cur_x_test_base_vals_for_pred.shape))

    #convert our new predicted list into array
    all_predicted_sequences=np.asarray(all_predicted_sequences)
    print("all_predicted_sequences shape: %s" %str(all_predicted_sequences.shape))

    print("############ Simulation complete! ################")
    return all_predicted_sequences

def predict_with_NO_DOW_and_GT_and_separate_models(x_test, model_list, FEATURES_OF_INTEREST, scaler, num_sequences_to_simulate,scaler_flag=True):
    #sanity checking tag means these are arrays we should not have access to during the challenge
    print("\n############ Now simulating on initial condition... ################")

    num_timesteps_in_a_sequence=x_test.shape[1]

    #create init condit array
    x_init_condit=x_test[:1,:,:].copy().astype("float32")

    #get first sequence of predictions
    separated_pred_deltas_per_feature=[]

    for i,model in enumerate(model_list):
        cur_ft=FEATURES_OF_INTEREST[i]
        print("Predicting %s" %cur_ft)

        cur_pred_deltas=model.predict(x_init_condit)
        separated_pred_deltas_per_feature.append(cur_pred_deltas)

    prediction_deltas=np.concatenate(separated_pred_deltas_per_feature,axis=2)
    print("\nCurrent shape of the prediction_deltas: %s" %str(prediction_deltas.shape))

    #make x_init_condit 2d
    x_init_condit=x_init_condit.reshape((x_init_condit.shape[0]*x_init_condit.shape[1],x_init_condit.shape[2]))

    #make pred array 2d
    prediction_deltas=prediction_deltas.reshape((prediction_deltas.shape[0]*prediction_deltas.shape[1],prediction_deltas.shape[2]))
    # print("prediction_deltas shape: %s" %str(prediction_deltas.shape))

    #current starting point = last day in the sequence
    cur_x_test_base_vals_for_pred=x_init_condit[num_timesteps_in_a_sequence-1:,:].copy()
    # cur_x_test_base_vals_for_pred=x_init_condit[num_timesteps_in_a_sequence-1:,:-7 ]
    # print("Shape of cur_x_test_base_vals_for_pred after removing days of the week: %s" %str(cur_x_test_base_vals_for_pred.shape))


    if scaler_flag==True:
        #we must inverse transform our initial condition so the deltas can be added to it properly
        cur_x_test_base_vals_for_pred=scaler.inverse_transform(cur_x_test_base_vals_for_pred)

    cur_x_test_base_vals_for_pred=cur_x_test_base_vals_for_pred.flatten()
    print("Cur base shape after flattening: %s" %str(cur_x_test_base_vals_for_pred.shape))

    # #FOR SANITY CHECKING
    # gt_cur_x_test_base_vals=cur_x_test_base_vals_for_pred.copy()

    #save all predicted sequences
    all_predicted_sequences=[]

    #get fts in a time step
    for i in range(num_sequences_to_simulate):

        #save your new cur sequence of predictions here
        cur_sequence_of_calculated_timestep_vals_not_deltas=[]

        #loop through all timesteps in a sequence
        for j in range(num_timesteps_in_a_sequence):

            #get cur timestep of predicted deltas
            cur_predicted_timestep_array_of_deltas_to_add=prediction_deltas[j]
            # print("cur_predicted_timestep_array_of_deltas_to_add shape: %s" %str(cur_predicted_timestep_array_of_deltas_to_add.shape))

            #add pred deltas
            new_pred_array=cur_x_test_base_vals_for_pred +  cur_predicted_timestep_array_of_deltas_to_add
            # print("new_pred_array: %s" %str(new_pred_array.shape))
            cur_sequence_of_calculated_timestep_vals_not_deltas.append(new_pred_array)
            cur_x_test_base_vals_for_pred=new_pred_array.copy()


        #at the end of an iteration i, we must make a sequence array out of all the timestep features we just calculated
        #then, we make a new x_test array
        #since we're suing ground truth from x_test, there's no need to normalize
        new_x_test = x_test[i+1:i+2,:,:].astype("float32")
        # new_x_test=new_x_test.reshape((new_x_test.shape[0]*new_x_test.shape[1],new_x_test.shape[2]))
        # print("Shape of new_x_test: %s" %str(new_x_test.shape))

        #Do not normalize, we want these actual count results
        all_predicted_sequences.append(np.asarray(cur_sequence_of_calculated_timestep_vals_not_deltas).copy())
        print("\nCurrent number of predicted sequences: %d out of %d" %(len(all_predicted_sequences),num_sequences_to_simulate))
        if len(all_predicted_sequences) == num_sequences_to_simulate:
            break

        #get first sequence of predictions
        separated_pred_deltas_per_feature=[]

        for i,model in enumerate(model_list):
            cur_ft=FEATURES_OF_INTEREST[i]
            print("Predicting %s" %cur_ft)

            cur_pred_deltas=model.predict(new_x_test)
            separated_pred_deltas_per_feature.append(cur_pred_deltas)

        prediction_deltas=np.concatenate(separated_pred_deltas_per_feature, axis=2)
        print("Current shape of the prediction_deltas: %s" %str(prediction_deltas.shape))

        #make pred array 2d
        prediction_deltas=prediction_deltas.reshape((prediction_deltas.shape[0]*prediction_deltas.shape[1],prediction_deltas.shape[2]))

        #current starting point = last day in the sequence
        new_x_test=new_x_test.reshape((new_x_test.shape[1],new_x_test.shape[2] ))

        cur_x_test_base_vals_for_pred=new_x_test[num_timesteps_in_a_sequence-1:,:].copy()

        if scaler_flag==True:
            #we must inverse transform our new text x condition so the deltas can be added to it properly
            cur_x_test_base_vals_for_pred=scaler.inverse_transform(cur_x_test_base_vals_for_pred)

        cur_x_test_base_vals_for_pred=cur_x_test_base_vals_for_pred.flatten()
        print("Cur base shape after flattening: %s" %str(cur_x_test_base_vals_for_pred.shape))

    #convert our new predicted list into array
    all_predicted_sequences=np.asarray(all_predicted_sequences)
    print("all_predicted_sequences shape: %s" %str(all_predicted_sequences.shape))

    print("############ Simulation complete! ################")
    return all_predicted_sequences

def predict_with_NO_DOW_and_GT_and_separate_models_and_prop_norm(x_test, model_list, FEATURES_OF_INTEREST, scaler, num_sequences_to_simulate,scaler_flag=True):
    #sanity checking tag means these are arrays we should not have access to during the challenge
    print("\n############ Now simulating on initial condition... ################")

    num_timesteps_in_a_sequence=x_test.shape[1]

    #create init condit array
    x_init_condit=x_test[:1,:,:].copy().astype("float32")

    #get first sequence of predictions
    separated_pred_deltas_per_feature=[]

    for i,model in enumerate(model_list):
        cur_ft=FEATURES_OF_INTEREST[i]
        print("Predicting %s" %cur_ft)

        cur_pred_deltas=model.predict(x_init_condit)
        separated_pred_deltas_per_feature.append(cur_pred_deltas)

    prediction_deltas=np.concatenate(separated_pred_deltas_per_feature,axis=2)
    print("\nCurrent shape of the prediction_deltas: %s" %str(prediction_deltas.shape))

    if scaler_flag==True:
        #we must inverse transform our initial condition so the deltas can be added to it properly
        x_init_condit_orig_shape = x_init_condit.shape
        x_init_condit=x_init_condit.reshape((x_init_condit.shape[0],x_init_condit.shape[1]* x_init_condit.shape[2]))
        x_init_condit=scaler.inverse_transform(x_init_condit)
        x_init_condit=x_init_condit.reshape(x_init_condit_orig_shape)

    # #make x_init_condit 2d
    # x_init_condit=x_init_condit.reshape((x_init_condit.shape[0]*x_init_condit.shape[1],x_init_condit.shape[2]))

    #make pred array 2d
    prediction_deltas=prediction_deltas.reshape((prediction_deltas.shape[0]*prediction_deltas.shape[1],prediction_deltas.shape[2]))
    # print("prediction_deltas shape: %s" %str(prediction_deltas.shape))

    #current starting point = last day in the sequence
    cur_x_test_base_vals_for_pred=x_init_condit[:,num_timesteps_in_a_sequence-1:,:].copy()
    # cur_x_test_base_vals_for_pred=x_init_condit[num_timesteps_in_a_sequence-1:,:-7 ]
    # print("Shape of cur_x_test_base_vals_for_pred after removing days of the week: %s" %str(cur_x_test_base_vals_for_pred.shape))

    cur_x_test_base_vals_for_pred=cur_x_test_base_vals_for_pred.flatten()
    print("Cur base shape after flattening: %s" %str(cur_x_test_base_vals_for_pred.shape))

    # #FOR SANITY CHECKING
    # gt_cur_x_test_base_vals=cur_x_test_base_vals_for_pred.copy()

    #save all predicted sequences
    all_predicted_sequences=[]

    #get fts in a time step
    for i in range(num_sequences_to_simulate):

        #save your new cur sequence of predictions here
        cur_sequence_of_calculated_timestep_vals_not_deltas=[]

        #loop through all timesteps in a sequence
        for j in range(num_timesteps_in_a_sequence):

            #get cur timestep of predicted deltas
            cur_predicted_timestep_array_of_deltas_to_add=prediction_deltas[j]
            # print("cur_predicted_timestep_array_of_deltas_to_add shape: %s" %str(cur_predicted_timestep_array_of_deltas_to_add.shape))

            #add pred deltas
            new_pred_array=cur_x_test_base_vals_for_pred +  cur_predicted_timestep_array_of_deltas_to_add
            # print("new_pred_array: %s" %str(new_pred_array.shape))
            cur_sequence_of_calculated_timestep_vals_not_deltas.append(new_pred_array)
            cur_x_test_base_vals_for_pred=new_pred_array.copy()


        #at the end of an iteration i, we must make a sequence array out of all the timestep features we just calculated
        #then, we make a new x_test array
        #since we're suing ground truth from x_test, there's no need to normalize
        new_x_test = x_test[i+1:i+2,:,:].astype("float32")
        # new_x_test=new_x_test.reshape((new_x_test.shape[0]*new_x_test.shape[1],new_x_test.shape[2]))
        # print("Shape of new_x_test: %s" %str(new_x_test.shape))

        #Do not normalize, we want these actual count results
        all_predicted_sequences.append(np.asarray(cur_sequence_of_calculated_timestep_vals_not_deltas).copy())
        print("\nCurrent number of predicted sequences: %d out of %d" %(len(all_predicted_sequences),num_sequences_to_simulate))
        if len(all_predicted_sequences) == num_sequences_to_simulate:
            break

        #get first sequence of predictions
        separated_pred_deltas_per_feature=[]

        for i,model in enumerate(model_list):
            cur_ft=FEATURES_OF_INTEREST[i]
            print("Predicting %s" %cur_ft)
            cur_pred_deltas=model.predict(new_x_test)
            separated_pred_deltas_per_feature.append(cur_pred_deltas)

        prediction_deltas=np.concatenate(separated_pred_deltas_per_feature, axis=2)
        print("Current shape of the prediction_deltas: %s" %str(prediction_deltas.shape))

        #make pred array 2d
        prediction_deltas=prediction_deltas.reshape((prediction_deltas.shape[0]*prediction_deltas.shape[1],prediction_deltas.shape[2]))

        # if scaler_flag==True:
        #     #we must inverse transform our new text x condition so the deltas can be added to it properly
        #     cur_x_test_base_vals_for_pred=scaler.inverse_transform(cur_x_test_base_vals_for_pred)

        # #current starting point = last day in the sequence
        # new_x_test=new_x_test.reshape((new_x_test.shape[1],new_x_test.shape[2] ))

        if scaler_flag==True:
            #we must inverse transform our new text x condition so the deltas can be added to it properly
            new_x_test_orig_shape = new_x_test.shape
            new_x_test=new_x_test.reshape((new_x_test.shape[0],new_x_test.shape[1]* new_x_test.shape[2]))
            new_x_test=scaler.inverse_transform(new_x_test)
            new_x_test=new_x_test.reshape(new_x_test_orig_shape)


        cur_x_test_base_vals_for_pred=new_x_test[:,num_timesteps_in_a_sequence-1:,:].copy()


        cur_x_test_base_vals_for_pred=cur_x_test_base_vals_for_pred.flatten()
        print("Cur base shape after flattening: %s" %str(cur_x_test_base_vals_for_pred.shape))

    #convert our new predicted list into array
    all_predicted_sequences=np.asarray(all_predicted_sequences)
    print("all_predicted_sequences shape: %s" %str(all_predicted_sequences.shape))

    print("############ Simulation complete! ################")
    return all_predicted_sequences

def predict_with_DOW_and_GT_and_separate_models_and_prop_norm(x_test, model_list, FEATURES_OF_INTEREST, scaler, num_sequences_to_simulate,scaler_flag=True):
    #sanity checking tag means these are arrays we should not have access to during the challenge
    print("\n############ Now simulating on initial condition... ################")

    num_timesteps_in_a_sequence=x_test.shape[1]

    #create init condit array
    x_init_condit=x_test[:1,:,:].copy().astype("float32")

    #get first sequence of predictions
    separated_pred_deltas_per_feature=[]

    for i,model in enumerate(model_list):
        cur_ft=FEATURES_OF_INTEREST[i]
        print("Predicting %s" %cur_ft)

        cur_pred_deltas=model.predict(x_init_condit)
        separated_pred_deltas_per_feature.append(cur_pred_deltas)

    prediction_deltas=np.concatenate(separated_pred_deltas_per_feature,axis=2)
    print("\nCurrent shape of the prediction_deltas: %s" %str(prediction_deltas.shape))

    # #make x_init_condit 2d
    # x_init_condit=x_init_condit.reshape((x_init_condit.shape[0]*x_init_condit.shape[1],x_init_condit.shape[2]))

    #make pred array 2d
    prediction_deltas=prediction_deltas.reshape((prediction_deltas.shape[0]*prediction_deltas.shape[1],prediction_deltas.shape[2]))
    # print("prediction_deltas shape: %s" %str(prediction_deltas.shape))


    # cur_x_test_base_vals_for_pred=x_init_condit[num_timesteps_in_a_sequence-1:,:-7 ]
    # print("Shape of cur_x_test_base_vals_for_pred after removing days of the week: %s" %str(cur_x_test_base_vals_for_pred.shape))


    if scaler_flag==True:
        #we must inverse transform our initial condition so the deltas can be added to it properly
        x_init_condit_orig_shape = x_init_condit.shape
        x_init_condit=x_init_condit.reshape((x_init_condit.shape[0],x_init_condit.shape[1]* x_init_condit.shape[2]))
        x_init_condit=scaler.inverse_transform(x_init_condit)
        x_init_condit=x_init_condit.reshape(x_init_condit_orig_shape)

    #current starting point = last day in the sequence
    cur_x_test_base_vals_for_pred_with_DOW=x_init_condit[:,num_timesteps_in_a_sequence-1:,:].copy()

    #now we must remove the DOW
    cur_x_test_base_vals_for_pred= cur_x_test_base_vals_for_pred_with_DOW[:,:,:-7].copy()
    print("Shape of cur_x_test_base_vals_for_pred after removing days of the week: %s" %str(cur_x_test_base_vals_for_pred.shape))

    cur_x_test_base_vals_for_pred=cur_x_test_base_vals_for_pred.flatten()
    print("Cur base shape after flattening: %s" %str(cur_x_test_base_vals_for_pred.shape))

    # #FOR SANITY CHECKING
    # gt_cur_x_test_base_vals=cur_x_test_base_vals_for_pred.copy()

    #save all predicted sequences
    all_predicted_sequences=[]

    #get fts in a time step
    for i in range(num_sequences_to_simulate):

        #save your new cur sequence of predictions here
        cur_sequence_of_calculated_timestep_vals_not_deltas=[]

        #loop through all timesteps in a sequence
        for j in range(num_timesteps_in_a_sequence):

            #get cur timestep of predicted deltas
            cur_predicted_timestep_array_of_deltas_to_add=prediction_deltas[j]
            # print("cur_predicted_timestep_array_of_deltas_to_add shape: %s" %str(cur_predicted_timestep_array_of_deltas_to_add.shape))

            #add pred deltas
            new_pred_array=cur_x_test_base_vals_for_pred +  cur_predicted_timestep_array_of_deltas_to_add
            # print("new_pred_array: %s" %str(new_pred_array.shape))
            cur_sequence_of_calculated_timestep_vals_not_deltas.append(new_pred_array)
            cur_x_test_base_vals_for_pred=new_pred_array.copy()


        #at the end of an iteration i, we must make a sequence array out of all the timestep features we just calculated
        #then, we make a new x_test array
        #since we're suing ground truth from x_test, there's no need to normalize
        new_x_test = x_test[i+1:i+2,:,:].astype("float32")
        # new_x_test=new_x_test.reshape((new_x_test.shape[0]*new_x_test.shape[1],new_x_test.shape[2]))
        # print("Shape of new_x_test: %s" %str(new_x_test.shape))

        #Do not normalize, we want these actual count results
        all_predicted_sequences.append(np.asarray(cur_sequence_of_calculated_timestep_vals_not_deltas).copy())
        print("\nCurrent number of predicted sequences: %d out of %d" %(len(all_predicted_sequences),num_sequences_to_simulate))
        if len(all_predicted_sequences) == num_sequences_to_simulate:
            break

        #Now we have to normalize x test
        # new_x_test=scaler.transform(new_x_test).reshape((1,new_x_test.shape[0],new_x_test.shape[1] ))

        # #predict
        # prediction_deltas=model.predict(new_x_test)

        #get first sequence of predictions
        separated_pred_deltas_per_feature=[]

        for i,model in enumerate(model_list):
            cur_ft=FEATURES_OF_INTEREST[i]
            print("Predicting %s" %cur_ft)

            cur_pred_deltas=model.predict(new_x_test)
            separated_pred_deltas_per_feature.append(cur_pred_deltas)

        prediction_deltas=np.concatenate(separated_pred_deltas_per_feature, axis=2)
        print("Current shape of the prediction_deltas: %s" %str(prediction_deltas.shape))

        #make pred array 2d
        prediction_deltas=prediction_deltas.reshape((prediction_deltas.shape[0]*prediction_deltas.shape[1],prediction_deltas.shape[2]))

        # #current starting point = last day in the sequence
        # new_x_test=new_x_test.reshape((new_x_test.shape[1],new_x_test.shape[2] ))

        if scaler_flag==True:
            #we must inverse transform our initial condition so the deltas can be added to it properly
            new_x_test_orig_shape = new_x_test.shape
            new_x_test=new_x_test.reshape((new_x_test.shape[0],new_x_test.shape[1]* new_x_test.shape[2]))
            new_x_test=scaler.inverse_transform(new_x_test)
            new_x_test=new_x_test.reshape(new_x_test_orig_shape)

        cur_x_test_base_vals_for_pred_with_DOW=new_x_test[:,num_timesteps_in_a_sequence-1:,:].copy()



        #now we must remove the DOW
        cur_x_test_base_vals_for_pred= cur_x_test_base_vals_for_pred_with_DOW[:,:,:-7].copy()
        print("Shape of cur_x_test_base_vals_for_pred after removing days of the week: %s" %str(cur_x_test_base_vals_for_pred.shape))

        cur_x_test_base_vals_for_pred=cur_x_test_base_vals_for_pred.flatten()
        print("Cur base shape after flattening: %s" %str(cur_x_test_base_vals_for_pred.shape))

    #convert our new predicted list into array
    all_predicted_sequences=np.asarray(all_predicted_sequences)
    print("all_predicted_sequences shape: %s" %str(all_predicted_sequences.shape))

    print("############ Simulation complete! ################")
    return all_predicted_sequences



def predict_with_or_without_DOW_and_GT_and_separate_models_and_prop_norm(x_test, model_list, FEATURES_OF_INTEREST, scaler, num_sequences_to_simulate,DOW_FLAG,scaler_flag=True):

    if DOW_FLAG == True:
        predictions = predict_with_DOW_and_GT_and_separate_models_and_prop_norm(x_test, model_list, FEATURES_OF_INTEREST, scaler, num_sequences_to_simulate,scaler_flag=scaler_flag)
    else:
        predictions = predict_with_NO_DOW_and_GT_and_separate_models_and_prop_norm(x_test, model_list, FEATURES_OF_INTEREST, scaler, num_sequences_to_simulate,scaler_flag=scaler_flag)

    return predictions

def pseudo_predict_with_DOW_and_GT_and_separate_models(x_test, scaler, scaler_flag=True):

    ######THIS FUNCTION PREDICTS BY TAKING THE INPUT AND MAKING IT THE OUTPUT#########################

    #sanity checking tag means these are arrays we should not have access to during the challenge
    print("\n############ Now making psuedo prediction! ################")
    print("\n We will take each input and make it the output!")

    #make the fake predictions
    predictions=x_test.copy()

    #create the final shape for later
    final_shape=(predictions.shape[0],predictions.shape[1],predictions.shape[2]-7)

    #make pred array 2d
    predictions=predictions.reshape((predictions.shape[0]*predictions.shape[1],predictions.shape[2]))

    if scaler_flag==True:
        #we must inverse transform our prediction
        predictions=scaler.inverse_transform(predictions)

    #now we must remove the days of the week!
    predictions=predictions[:,:-7]
    print("Shape of predictions after removing days of the week: %s" %str(predictions.shape))

    #now reshape
    predictions=predictions.reshape(final_shape)
    print("Final shape of predictions: %s" %str(predictions.shape))

    print("\n############ Pseudo-simulation complete! ################")

    return predictions

def pseudo_predict_with_NO_DOW_and_GT_and_separate_models_with_prop_norm(x_test, scaler, scaler_flag=True):

    ######THIS FUNCTION PREDICTS BY TAKING THE INPUT AND MAKING IT THE OUTPUT#########################

    #sanity checking tag means these are arrays we should not have access to during the challenge
    print("\n############ Now making psuedo prediction! ################")
    print("\n We will take each input and make it the output!")

    #make the fake predictions
    predictions=x_test.copy()

    #create the final shape for later
    # final_shape=(predictions.shape[0],predictions.shape[1],predictions.shape[2]-7)
    final_shape=(predictions.shape[0],predictions.shape[1],predictions.shape[2])

    if scaler_flag==True:
        #we must inverse transform our prediction
        # predictions=scaler.inverse_transform(predictions)
        predictions_orig_shape = predictions.shape
        predictions=predictions.reshape((predictions.shape[0],predictions.shape[1]* predictions.shape[2]))
        predictions=scaler.inverse_transform(predictions)
        predictions=predictions.reshape(predictions_orig_shape)

    #make pred array 2d
    predictions=predictions.reshape((predictions.shape[0]*predictions.shape[1],predictions.shape[2]))

    #now we must remove the days of the week!
    # predictions=predictions[:,:-7]
    # print("Shape of predictions after removing days of the week: %s" %str(predictions.shape))

    #now reshape
    predictions=predictions.reshape(final_shape)
    print("Final shape of predictions: %s" %str(predictions.shape))

    print("\n############ Pseudo-simulation complete! ################")

    return predictions

def pseudo_predict_with_DOW_and_GT_and_separate_models_with_prop_norm(x_test, scaler, scaler_flag=True):

    ######THIS FUNCTION PREDICTS BY TAKING THE INPUT AND MAKING IT THE OUTPUT#########################

    #sanity checking tag means these are arrays we should not have access to during the challenge
    print("\n############ Now making psuedo prediction! ################")
    print("\n We will take each input and make it the output!")

    #make the fake predictions
    predictions=x_test.copy()

    #create the final shape for later
    final_shape=(predictions.shape[0],predictions.shape[1],predictions.shape[2]-7)
    # final_shape=(predictions.shape[0],predictions.shape[1],predictions.shape[2])

    if scaler_flag==True:
        #we must inverse transform our prediction
        # predictions=scaler.inverse_transform(predictions)
        predictions_orig_shape = predictions.shape
        predictions=predictions.reshape((predictions.shape[0],predictions.shape[1]* predictions.shape[2]))
        predictions=scaler.inverse_transform(predictions)
        predictions=predictions.reshape(predictions_orig_shape)

    #make pred array 2d
    predictions=predictions.reshape((predictions.shape[0]*predictions.shape[1],predictions.shape[2]))

    #now we must remove the days of the week!
    predictions=predictions[:,:-7]
    print("Shape of predictions after removing days of the week: %s" %str(predictions.shape))

    #now reshape
    predictions=predictions.reshape(final_shape)
    print("Final shape of predictions: %s" %str(predictions.shape))

    print("\n############ Pseudo-simulation complete! ################")

    return predictions

def pseudo_predict_with_or_without_DOW_and_GT_and_separate_models_with_prop_norm(x_test, scaler, DOW_FLAG,scaler_flag=True):
    if DOW_FLAG == True:
        predictions_by_shifting=pseudo_predict_with_DOW_and_GT_and_separate_models_with_prop_norm(x_test, scaler, scaler_flag=scaler_flag)
    else:
        predictions_by_shifting=pseudo_predict_with_NO_DOW_and_GT_and_separate_models_with_prop_norm(x_test, scaler, scaler_flag=scaler_flag)

    return predictions_by_shifting

def pseudo_predict_with_NO_DOW_and_GT_and_separate_models(x_test, scaler, scaler_flag=True):

    ######THIS FUNCTION PREDICTS BY TAKING THE INPUT AND MAKING IT THE OUTPUT#########################

    #sanity checking tag means these are arrays we should not have access to during the challenge
    print("\n############ Now making psuedo prediction! ################")
    print("\n We will take each input and make it the output!")

    #make the fake predictions
    predictions=x_test.copy()

    #create the final shape for later
    # final_shape=(predictions.shape[0],predictions.shape[1],predictions.shape[2]-7)
    final_shape=(predictions.shape[0],predictions.shape[1],predictions.shape[2])

    #make pred array 2d
    predictions=predictions.reshape((predictions.shape[0]*predictions.shape[1],predictions.shape[2]))

    if scaler_flag==True:
        #we must inverse transform our prediction
        predictions=scaler.inverse_transform(predictions)

    #now we must remove the days of the week!
    # predictions=predictions[:,:-7]
    # print("Shape of predictions after removing days of the week: %s" %str(predictions.shape))

    #now reshape
    predictions=predictions.reshape(final_shape)
    print("Final shape of predictions: %s" %str(predictions.shape))

    print("\n############ Pseudo-simulation complete! ################")

    return predictions

def pseudo_predict_one_target_with_no_DOW(x_test, TARGET_FT_INDEX_IN_SOURCE_ARRAY,scaler, scaler_flag=True):

    ######THIS FUNCTION PREDICTS BY TAKING THE INPUT AND MAKING IT THE OUTPUT#########################

    #sanity checking tag means these are arrays we should not have access to during the challenge
    print("\n############ Now making psuedo prediction! ################")
    print("\n We will take each input and make it the output!")

    #make the fake predictions
    predictions=x_test.copy()

    # #create the final shape for later
    final_shape=(predictions.shape[0],predictions.shape[1],1)

    #make pred array 2d
    predictions=predictions.reshape((predictions.shape[0]*predictions.shape[1],predictions.shape[2]))

    if scaler_flag==True:
        #we must inverse transform our prediction
        predictions=scaler.inverse_transform(predictions)

    #now we must remove the days of the week!
    predictions=predictions[:,TARGET_FT_INDEX_IN_SOURCE_ARRAY]
    print("Shape of predictions after getting desired target: %s" %str(predictions.shape))

    #now reshape
    predictions=predictions.reshape(final_shape)
    print("Final shape of predictions: %s" %str(predictions.shape))

    print("\n############ Pseudo-simulation complete! ################")

    return predictions


def predict_hourly_with_GT_and_separate_models(x_test, model_list, FEATURES_OF_INTEREST, scaler, num_sequences_to_simulate,scaler_flag=True):
    #sanity checking tag means these are arrays we should not have access to during the challenge
    print("\n############ Now simulating on initial condition... ################")

    num_timesteps_in_a_sequence=x_test.shape[1]

    #create init condit array
    x_init_condit=x_test[:1,:,:].copy().astype("float32")

    #get first sequence of predictions
    separated_pred_deltas_per_feature=[]

    for i,model in enumerate(model_list):
        cur_ft=FEATURES_OF_INTEREST[i]
        print("Predicting %s" %cur_ft)

        cur_pred_deltas=model.predict(x_init_condit)
        separated_pred_deltas_per_feature.append(cur_pred_deltas)

    prediction_deltas=np.concatenate(separated_pred_deltas_per_feature,axis=2)
    print("\nCurrent shape of the prediction_deltas: %s" %str(prediction_deltas.shape))

    #make x_init_condit 2d
    x_init_condit=x_init_condit.reshape((x_init_condit.shape[0]*x_init_condit.shape[1],x_init_condit.shape[2]))

    #make pred array 2d
    prediction_deltas=prediction_deltas.reshape((prediction_deltas.shape[0]*prediction_deltas.shape[1],prediction_deltas.shape[2]))
    # print("prediction_deltas shape: %s" %str(prediction_deltas.shape))

    #current starting point = last day in the sequence
    cur_x_test_base_vals_for_pred=x_init_condit[num_timesteps_in_a_sequence-1:,:].copy()
    # cur_x_test_base_vals_for_pred=x_init_condit[num_timesteps_in_a_sequence-1:,:-7 ]
    # print("Shape of cur_x_test_base_vals_for_pred after removing days of the week: %s" %str(cur_x_test_base_vals_for_pred.shape))


    if scaler_flag==True:
        #we must inverse transform our initial condition so the deltas can be added to it properly
        cur_x_test_base_vals_for_pred=scaler.inverse_transform(cur_x_test_base_vals_for_pred)

    #now we must remove the DOW
    # cur_x_test_base_vals_for_pred= cur_x_test_base_vals_for_pred[:,:-7].copy()
    # print("Shape of cur_x_test_base_vals_for_pred after removing days of the week: %s" %str(cur_x_test_base_vals_for_pred.shape))

    cur_x_test_base_vals_for_pred=cur_x_test_base_vals_for_pred.flatten()
    print("Cur base shape after flattening: %s" %str(cur_x_test_base_vals_for_pred.shape))

    # #FOR SANITY CHECKING
    # gt_cur_x_test_base_vals=cur_x_test_base_vals_for_pred.copy()

    #save all predicted sequences
    all_predicted_sequences=[]

    #get fts in a time step
    for i in range(num_sequences_to_simulate):

        #save your new cur sequence of predictions here
        cur_sequence_of_calculated_timestep_vals_not_deltas=[]

        #loop through all timesteps in a sequence
        for j in range(num_timesteps_in_a_sequence):

            #get cur timestep of predicted deltas
            cur_predicted_timestep_array_of_deltas_to_add=prediction_deltas[j]
            # print("cur_predicted_timestep_array_of_deltas_to_add shape: %s" %str(cur_predicted_timestep_array_of_deltas_to_add.shape))

            #add pred deltas
            new_pred_array=cur_x_test_base_vals_for_pred +  cur_predicted_timestep_array_of_deltas_to_add
            # print("new_pred_array: %s" %str(new_pred_array.shape))
            cur_sequence_of_calculated_timestep_vals_not_deltas.append(new_pred_array)
            cur_x_test_base_vals_for_pred=new_pred_array.copy()


        #at the end of an iteration i, we must make a sequence array out of all the timestep features we just calculated
        #then, we make a new x_test array
        #since we're suing ground truth from x_test, there's no need to normalize
        new_x_test = x_test[i+1:i+2,:,:].astype("float32")
        # new_x_test=new_x_test.reshape((new_x_test.shape[0]*new_x_test.shape[1],new_x_test.shape[2]))
        # print("Shape of new_x_test: %s" %str(new_x_test.shape))

        #Do not normalize, we want these actual count results
        all_predicted_sequences.append(np.asarray(cur_sequence_of_calculated_timestep_vals_not_deltas).copy())
        print("\nCurrent number of predicted sequences: %d out of %d" %(len(all_predicted_sequences),num_sequences_to_simulate))
        if len(all_predicted_sequences) == num_sequences_to_simulate:
            break

        #Now we have to normalize x test
        # new_x_test=scaler.transform(new_x_test).reshape((1,new_x_test.shape[0],new_x_test.shape[1] ))

        # #predict
        # prediction_deltas=model.predict(new_x_test)

        #get first sequence of predictions
        separated_pred_deltas_per_feature=[]

        for i,model in enumerate(model_list):
            cur_ft=FEATURES_OF_INTEREST[i]
            print("Predicting %s" %cur_ft)

            cur_pred_deltas=model.predict(new_x_test)
            separated_pred_deltas_per_feature.append(cur_pred_deltas)

        prediction_deltas=np.concatenate(separated_pred_deltas_per_feature, axis=2)
        print("Current shape of the prediction_deltas: %s" %str(prediction_deltas.shape))

        #make pred array 2d
        prediction_deltas=prediction_deltas.reshape((prediction_deltas.shape[0]*prediction_deltas.shape[1],prediction_deltas.shape[2]))

        #current starting point = last day in the sequence
        new_x_test=new_x_test.reshape((new_x_test.shape[1],new_x_test.shape[2] ))

        cur_x_test_base_vals_for_pred=new_x_test[num_timesteps_in_a_sequence-1:,:].copy()

        if scaler_flag==True:
            #we must inverse transform our new text x condition so the deltas can be added to it properly
            cur_x_test_base_vals_for_pred=scaler.inverse_transform(cur_x_test_base_vals_for_pred)

        #now we must remove the DOW
        # cur_x_test_base_vals_for_pred= cur_x_test_base_vals_for_pred[:,:-7].copy()
        # print("Shape of cur_x_test_base_vals_for_pred after removing days of the week: %s" %str(cur_x_test_base_vals_for_pred.shape))

        cur_x_test_base_vals_for_pred=cur_x_test_base_vals_for_pred.flatten()
        print("Cur base shape after flattening: %s" %str(cur_x_test_base_vals_for_pred.shape))

    #convert our new predicted list into array
    all_predicted_sequences=np.asarray(all_predicted_sequences)
    print("all_predicted_sequences shape: %s" %str(all_predicted_sequences.shape))

    print("############ Simulation complete! ################")
    return all_predicted_sequences
def simulate_on_initial_condition_not_a_drill_with_separate_models(x_test, model_list,scaler, num_sequences_to_simulate,FEATURES_OF_INTEREST):
    #sanity checking tag means these are arrays we should not have access to during the challenge
    print("\n############ Now simulating on initial condition... ################")


    num_timesteps_in_a_sequence=x_test.shape[1]

    #create init condit array
    x_init_condit=x_test[:1,:,:].astype("float32")

    #get first sequence of predictions
    separated_pred_deltas_per_feature=[]

    for i,model in enumerate(model_list):
        cur_ft=FEATURES_OF_INTEREST[i]
        print("Predicting %s" %cur_ft)

        cur_pred_deltas=model.predict(x_init_condit)
        separated_pred_deltas_per_feature.append(cur_pred_deltas)

    prediction_deltas=np.concatenate(separated_pred_deltas_per_feature,axis=2)
    print("\nCurrent shape of the prediction_deltas: %s" %str(prediction_deltas.shape))

    #make x_init_condit 2d
    x_init_condit=x_init_condit.reshape((x_init_condit.shape[0]*x_init_condit.shape[1],x_init_condit.shape[2]))

    #make pred array 2d
    prediction_deltas=prediction_deltas.reshape((prediction_deltas.shape[0]*prediction_deltas.shape[1],prediction_deltas.shape[2]))
    # print("prediction_deltas shape: %s" %str(prediction_deltas.shape))

    #current starting point = last day in the sequence
    cur_x_test_base_vals_for_pred=x_init_condit[num_timesteps_in_a_sequence-1:, ]



    #we must inverse transform our initial condition so the deltas can be added to it properly
    cur_x_test_base_vals_for_pred=scaler.inverse_transform(cur_x_test_base_vals_for_pred)
    cur_x_test_base_vals_for_pred=cur_x_test_base_vals_for_pred.flatten()
    # print("Cur base shape: %s" %str(cur_x_test_base_vals_for_pred.shape))

    #FOR SANITY CHECKING
    gt_cur_x_test_base_vals=cur_x_test_base_vals_for_pred.copy()

    #save all predicted sequences
    all_predicted_sequences=[]

    #get fts in a time step
    for i in range(num_sequences_to_simulate):

        #save your new cur sequence of predictions here
        cur_sequence_of_calculated_timestep_vals_not_deltas=[]

        #might have to add back 1
        for j in range(num_timesteps_in_a_sequence):

            #get cur timestep of predicted deltas
            cur_predicted_timestep_array_of_deltas_to_add=prediction_deltas[j]
            # print("cur_predicted_timestep_array_of_deltas_to_add shape: %s" %str(cur_predicted_timestep_array_of_deltas_to_add.shape))

            #add pred deltas
            new_pred_array=cur_x_test_base_vals_for_pred +  cur_predicted_timestep_array_of_deltas_to_add
            # print("new_pred_array: %s" %str(new_pred_array.shape))
            cur_sequence_of_calculated_timestep_vals_not_deltas.append(new_pred_array)
            cur_x_test_base_vals_for_pred=new_pred_array.copy()


        #at the end of an iteration i, we must make a sequence array out of all the timestep features we just calculated
        #then, we must re-normalize them to make a new x_test array
        new_x_test = np.asarray(cur_sequence_of_calculated_timestep_vals_not_deltas)
        # print("Shape of new_x_test: %s" %str(new_x_test.shape))

        #Do not normalize, we want these actual count results
        all_predicted_sequences.append(new_x_test.copy())
        print("\nCurrent number of predicted sequences: %d out of %d" %(len(all_predicted_sequences),num_sequences_to_simulate))

        #Now we have to normalize x test
        new_x_test=scaler.transform(new_x_test).reshape((1,new_x_test.shape[0],new_x_test.shape[1] ))

        # #predict
        # prediction_deltas=model.predict(new_x_test)

        #get first sequence of predictions
        separated_pred_deltas_per_feature=[]

        for i,model in enumerate(model_list):
            cur_ft=FEATURES_OF_INTEREST[i]
            print("Predicting %s" %cur_ft)

            cur_pred_deltas=model.predict(new_x_test)
            separated_pred_deltas_per_feature.append(cur_pred_deltas)

        prediction_deltas=np.concatenate(separated_pred_deltas_per_feature, axis=2)
        print("Current shape of the prediction_deltas: %s" %str(prediction_deltas.shape))

        #make pred array 2d
        prediction_deltas=prediction_deltas.reshape((prediction_deltas.shape[0]*prediction_deltas.shape[1],prediction_deltas.shape[2]))

        #current starting point = last day in the sequence
        new_x_test=new_x_test.reshape((new_x_test.shape[1],new_x_test.shape[2] ))
        cur_x_test_base_vals_for_pred=new_x_test[num_timesteps_in_a_sequence-1:, ]

        #we must inverse transform our new text x condition so the deltas can be added to it properly
        cur_x_test_base_vals_for_pred=scaler.inverse_transform(cur_x_test_base_vals_for_pred)
        cur_x_test_base_vals_for_pred=cur_x_test_base_vals_for_pred.flatten()
        # print("Cur base shape: %s" %str(cur_x_test_base_vals_for_pred.shape))

    #convert our new predicted list into array
    all_predicted_sequences=np.asarray(all_predicted_sequences)
    print("all_predicted_sequences shape: %s" %str(all_predicted_sequences.shape))

    print("############ Simulation complete! ################")
    return all_predicted_sequences

def simulate_on_initial_condition_not_a_drill_with_DOW(x_test, model,scaler, num_sequences_to_simulate,scaler_flag=True):
    #sanity checking tag means these are arrays we should not have access to during the challenge
    print("############ Now simulating on initial condition... ################")

    #setup dow dict
    dow_dict={
    0: [1,0,0,0,0,0,0],
    1: [0,1,0,0,0,0,0],
    2: [0,0,1,0,0,0,0],
    3: [0,0,0,1,0,0,0],
    4: [0,0,0,0,1,0,0],
    5: [0,0,0,0,0,1,0],
    6: [0,0,0,0,0,0,1]
    }

    x_test=x_test.astype("float32")


    #DIFFERENCE THRESHOLD
    #SANITY CHECK PARAM
    DIFFERENCE_THRESHOLD=0.1

    num_timesteps_in_a_sequence=x_test.shape[1]

    #create init condit array
    x_init_condit=x_test[:1,:,:]
    print("Shape of x_init_condit: %s" %str(x_init_condit.shape))

    #get first sequence of predictions
    #these deltas do not have the DOW
    prediction_deltas=model.predict(x_init_condit)

    #make x_init_condit 2d
    #this array has the DOW values
    x_init_condit=x_init_condit.reshape((x_init_condit.shape[0]*x_init_condit.shape[1],x_init_condit.shape[2]))

    #make pred array 2d
    #these deltas do not have the DOW
    prediction_deltas=prediction_deltas.reshape((prediction_deltas.shape[0]*prediction_deltas.shape[1],prediction_deltas.shape[2]))
    # print("prediction_deltas shape: %s" %str(prediction_deltas.shape))

    #current starting point = last day in the sequence
    #2d array with shape (1,num_features_in_timestep+7)
    #DOW is here
    cur_x_test_base_vals_for_pred=x_init_condit[num_timesteps_in_a_sequence-1:, ]
    # print("Shape of cur_x_test_base_vals_for_pred: %s" %str(cur_x_test_base_vals_for_pred.shape))

    #we must inverse transform our initial condition so the deltas can be added to it properly
    #DOW
    if scaler_flag==True:
        cur_x_test_base_vals_for_pred=scaler.inverse_transform(cur_x_test_base_vals_for_pred)
        # cur_x_test_base_vals_for_pred = properly_denormalize_2d_data(cur_x_test_base_vals_for_pred, scaler)

    #for the base vals

    cur_x_test_base_vals_for_pred=cur_x_test_base_vals_for_pred.flatten()
    # print("Shape of flattened cur_x_test_base_vals_for_pred: %s" %str(cur_x_test_base_vals_for_pred.shape))

    #remove days of the week
    cur_dow_array=cur_x_test_base_vals_for_pred[-7:]
    # print("cur_dow_array: %s" %str(cur_dow_array))
    cur_x_test_base_vals_for_pred=cur_x_test_base_vals_for_pred[:-7]
    # print("Cur base shape: %s" %str(cur_x_test_base_vals_for_pred.shape))

    #FOR SANITY CHECKING
    # gt_cur_x_test_base_vals=cur_x_test_base_vals_for_pred.copy()

    #save all predicted sequences
    all_predicted_sequences=[]

    #get fts in a time step
    for i in range(num_sequences_to_simulate):

        #save your new cur sequence of predictions here
        cur_sequence_of_calculated_timestep_vals_not_deltas=[]

        #might have to add back 1
        for j in range(num_timesteps_in_a_sequence):

            #get cur timestep of predicted deltas
            cur_predicted_timestep_array_of_deltas_to_add=prediction_deltas[j]
            # print("cur_predicted_timestep_array_of_deltas_to_add shape: %s" %str(cur_predicted_timestep_array_of_deltas_to_add.shape))

            #add pred deltas
            new_pred_array=cur_x_test_base_vals_for_pred +  cur_predicted_timestep_array_of_deltas_to_add
            # print("new_pred_array: %s" %str(new_pred_array.shape))

            #now we must get the next dow to add back
            dow=np.argmax(cur_dow_array)
            # print("DOW is %d" %dow)

            #next dow
            next_dow=(dow + 1)%7
            # print("Next dow: %d" %next_dow)

            #next dow as array
            next_dow_as_array=dow_dict[next_dow]
            # print("Next dow as array: %s" %str(next_dow_as_array))

            #update current
            cur_dow_array=next_dow_as_array

            #before we concat, make this new pred array without the dow available for the next round of delta adding
            cur_x_test_base_vals_for_pred=new_pred_array.copy()

            #concat this next dow with new pred array
            new_pred_array=np.concatenate((new_pred_array,next_dow_as_array))

            #append new pred array with dow to our current list of predicted timestep arrays
            cur_sequence_of_calculated_timestep_vals_not_deltas.append(new_pred_array)



        #at the end of an iteration i (a sequence) , we must make a sequence array out of all the timestep features we just calculated
        #then, we must re-normalize them to make a new x_test array
        new_x_test = np.asarray(cur_sequence_of_calculated_timestep_vals_not_deltas)
        # print("Shape of new_x_test: %s" %str(new_x_test.shape))

        #Do not normalize, we want these actual count results
        all_predicted_sequences.append(new_x_test.copy())
        print("Current number of predicted sequences: %d out of %d" %(len(all_predicted_sequences),num_sequences_to_simulate))

        #Now we have to normalize x test
        if scaler_flag==True:
            new_x_test=scaler.transform(new_x_test)
            # new_x_test = properly_normalize_2d_data(new_x_test, scaler)
        new_x_test=new_x_test.reshape((1,new_x_test.shape[0],new_x_test.shape[1] ))

        #predict
        prediction_deltas=model.predict(new_x_test)

        #make pred array 2d
        prediction_deltas=prediction_deltas.reshape((prediction_deltas.shape[0]*prediction_deltas.shape[1],prediction_deltas.shape[2]))

        #current starting point = last day in the sequence
        new_x_test=new_x_test.reshape((new_x_test.shape[1],new_x_test.shape[2] ))
        cur_x_test_base_vals_for_pred=new_x_test[num_timesteps_in_a_sequence-1:, ]

        #we must inverse transform our new text x condition so the deltas can be added to it properly
        if scaler_flag==True:
            cur_x_test_base_vals_for_pred=scaler.inverse_transform(cur_x_test_base_vals_for_pred)
            # cur_x_test_base_vals_for_pred = properly_denormalize_2d_data(cur_x_test_base_vals_for_pred, scaler)
        cur_x_test_base_vals_for_pred=cur_x_test_base_vals_for_pred.flatten()
        # print("Cur base shape: %s" %str(cur_x_test_base_vals_for_pred.shape))

        #remove days of the week
        saved_dow_array=cur_x_test_base_vals_for_pred[-7:]
        print("saved_dow_array: %s" %str(saved_dow_array))
        cur_x_test_base_vals_for_pred=cur_x_test_base_vals_for_pred[:-7]

    #convert our new predicted list into array
    all_predicted_sequences=np.asarray(all_predicted_sequences)
    all_predicted_sequences=all_predicted_sequences[:,:,:-7]
    print("all_predicted_sequences shape: %s" %str(all_predicted_sequences.shape))

    print("############ Simulation complete! ################")
    return all_predicted_sequences

def simulate_on_initial_condition_not_a_drill_with_numerical_month_feature(x_test, model,scaler, num_sequences_to_simulate,scaler_flag=True):
    #sanity checking tag means these are arrays we should not have access to during the challenge
    print("############ Now simulating on initial condition... ################")

    num_month_fts=1

    # #setup month dict
    # month_dict={
    # 0: [0],
    # 1: [1],
    # 2: [2],
    # 3: [3],
    # 4: [4],
    # 5: [5],
    # 6: [6],
    # 7: [7],
    # 8: [8],
    # 9: [9],
    # 10:[10],
    # 11:[11],
    # }


    #DIFFERENCE THRESHOLD
    #SANITY CHECK PARAM
    DIFFERENCE_THRESHOLD=0.1

    num_timesteps_in_a_sequence=x_test.shape[1]

    #create init condit array
    x_test=x_test.astype("float32")
    x_init_condit=x_test[:1,:,:]
    print("Shape of x_init_condit: %s" %str(x_init_condit.shape))

    #get first sequence of predictions
    #these deltas do not have the month
    prediction_deltas=model.predict(x_init_condit)

    #make x_init_condit 2d
    #this array has the month values
    x_init_condit=x_init_condit.reshape((x_init_condit.shape[0]*x_init_condit.shape[1],x_init_condit.shape[2]))

    #make pred array 2d
    #these deltas do not have the month
    prediction_deltas=prediction_deltas.reshape((prediction_deltas.shape[0]*prediction_deltas.shape[1],prediction_deltas.shape[2]))
    # print("prediction_deltas shape: %s" %str(prediction_deltas.shape))

    #current starting point = last day in the sequence
    #2d array with shape (1,num_features_in_timestep+7)
    #month is here
    cur_x_test_base_vals_for_pred=x_init_condit[num_timesteps_in_a_sequence-1:, ]
    print("Shape of cur_x_test_base_vals_for_pred: %s" %str(cur_x_test_base_vals_for_pred.shape))

    #we must inverse transform our initial condition so the deltas can be added to it properly
    #month
    if scaler_flag==True:
        cur_x_test_base_vals_for_pred=scaler.inverse_transform(cur_x_test_base_vals_for_pred)

    #for the base vals

    cur_x_test_base_vals_for_pred=cur_x_test_base_vals_for_pred.flatten()
    print("Shape of flattened cur_x_test_base_vals_for_pred: %s" %str(cur_x_test_base_vals_for_pred.shape))

    #remove days of the week
    cur_month_array=cur_x_test_base_vals_for_pred[-num_month_fts:]
    print("cur_month_array: %s" %str(cur_month_array))
    cur_x_test_base_vals_for_pred=cur_x_test_base_vals_for_pred[:-num_month_fts]
    # print("Cur base shape: %s" %str(cur_x_test_base_vals_for_pred.shape))

    #FOR SANITY CHECKING
    # gt_cur_x_test_base_vals=cur_x_test_base_vals_for_pred.copy()

    #save all predicted sequences
    all_predicted_sequences=[]

    #get fts in a time step
    for i in range(num_sequences_to_simulate):

        #save your new cur sequence of predictions here
        cur_sequence_of_calculated_timestep_vals_not_deltas=[]

        #might have to add back 1
        for j in range(num_timesteps_in_a_sequence):

            #get cur timestep of predicted deltas
            cur_predicted_timestep_array_of_deltas_to_add=prediction_deltas[j]
            # print("cur_predicted_timestep_array_of_deltas_to_add shape: %s" %str(cur_predicted_timestep_array_of_deltas_to_add.shape))

            #add pred deltas
            new_pred_array=cur_x_test_base_vals_for_pred +  cur_predicted_timestep_array_of_deltas_to_add
            # print("new_pred_array: %s" %str(new_pred_array.shape))

            #now we must get the next month to add back
            month=cur_month_array[0]
            print("month is %d" %month)

            #next month
            next_month=(month + 1)%12
            print("Next month: %d" %next_month)

            #next month as array
            next_month_as_array=np.asarray([next_month])
            print("Next month as array: %s" %str(next_month_as_array))

            #update current
            cur_month_array=next_month_as_array

            #before we concat, make this new pred array without the month available for the next round of delta adding
            cur_x_test_base_vals_for_pred=new_pred_array.copy()

            #concat this next month with new pred array
            new_pred_array=np.concatenate((new_pred_array,next_month_as_array))

            #append new pred array with month to our current list of predicted timestep arrays
            cur_sequence_of_calculated_timestep_vals_not_deltas.append(new_pred_array)



        #at the end of an iteration i (a sequence) , we must make a sequence array out of all the timestep features we just calculated
        #then, we must re-normalize them to make a new x_test array
        new_x_test = np.asarray(cur_sequence_of_calculated_timestep_vals_not_deltas)
        # print("Shape of new_x_test: %s" %str(new_x_test.shape))

        #Do not normalize, we want these actual count results
        all_predicted_sequences.append(new_x_test.copy())
        print("Current number of predicted sequences: %d out of %d" %(len(all_predicted_sequences),num_sequences_to_simulate))

        #Now we have to normalize x test
        if scaler_flag==True:
            new_x_test=scaler.transform(new_x_test)
        new_x_test=new_x_test.reshape((1,new_x_test.shape[0],new_x_test.shape[1] ))

        #predict
        prediction_deltas=model.predict(new_x_test)

        #make pred array 2d
        prediction_deltas=prediction_deltas.reshape((prediction_deltas.shape[0]*prediction_deltas.shape[1],prediction_deltas.shape[2]))

        #current starting point = last day in the sequence
        new_x_test=new_x_test.reshape((new_x_test.shape[1],new_x_test.shape[2] ))
        cur_x_test_base_vals_for_pred=new_x_test[num_timesteps_in_a_sequence-1:, ]

        #we must inverse transform our new text x condition so the deltas can be added to it properly
        if scaler_flag==True:
            cur_x_test_base_vals_for_pred=scaler.inverse_transform(cur_x_test_base_vals_for_pred)
        cur_x_test_base_vals_for_pred=cur_x_test_base_vals_for_pred.flatten()
        # print("Cur base shape: %s" %str(cur_x_test_base_vals_for_pred.shape))

        #remove days of the week
        saved_month_array=cur_x_test_base_vals_for_pred[-num_month_fts:]
        print("saved_month_array: %s" %str(saved_month_array))
        cur_x_test_base_vals_for_pred=cur_x_test_base_vals_for_pred[:-num_month_fts]

    #convert our new predicted list into array
    all_predicted_sequences=np.asarray(all_predicted_sequences)
    all_predicted_sequences=all_predicted_sequences[:,:,:-num_month_fts]
    print("all_predicted_sequences shape: %s" %str(all_predicted_sequences.shape))

    print("############ Simulation complete! ################")
    return all_predicted_sequences

def simulate_on_initial_condition_not_a_drill_with_month_features(x_test, model,scaler, num_sequences_to_simulate,scaler_flag=True):
    #sanity checking tag means these are arrays we should not have access to during the challenge
    print("############ Now simulating on initial condition... ################")

    num_months=12

    #setup month dict
    month_dict={
    0: [1,0,0,0,0,0,0,0,0,0,0,0],
    1: [0,1,0,0,0,0,0,0,0,0,0,0],
    2: [0,0,1,0,0,0,0,0,0,0,0,0],
    3: [0,0,0,1,0,0,0,0,0,0,0,0],
    4: [0,0,0,0,1,0,0,0,0,0,0,0],
    5: [0,0,0,0,0,1,0,0,0,0,0,0],
    6: [0,0,0,0,0,0,1,0,0,0,0,0],
    7: [0,0,0,0,0,0,0,1,0,0,0,0],
    8: [0,0,0,0,0,0,0,0,1,0,0,0],
    9: [0,0,0,0,0,0,0,0,0,1,0,0],
    10:[0,0,0,0,0,0,0,0,0,0,1,0],
    11:[0,0,0,0,0,0,0,0,0,0,0,1],
    }


    #DIFFERENCE THRESHOLD
    #SANITY CHECK PARAM
    DIFFERENCE_THRESHOLD=0.1

    num_timesteps_in_a_sequence=x_test.shape[1]

    #create init condit array
    x_test=x_test.astype("float32")
    x_init_condit=x_test[:1,:,:]
    print("Shape of x_init_condit: %s" %str(x_init_condit.shape))

    #get first sequence of predictions
    #these deltas do not have the month
    prediction_deltas=model.predict(x_init_condit)

    #make x_init_condit 2d
    #this array has the month values
    x_init_condit=x_init_condit.reshape((x_init_condit.shape[0]*x_init_condit.shape[1],x_init_condit.shape[2]))

    #make pred array 2d
    #these deltas do not have the month
    prediction_deltas=prediction_deltas.reshape((prediction_deltas.shape[0]*prediction_deltas.shape[1],prediction_deltas.shape[2]))
    # print("prediction_deltas shape: %s" %str(prediction_deltas.shape))

    #current starting point = last day in the sequence
    #2d array with shape (1,num_features_in_timestep+7)
    #month is here
    cur_x_test_base_vals_for_pred=x_init_condit[num_timesteps_in_a_sequence-1:, ]
    print("Shape of cur_x_test_base_vals_for_pred: %s" %str(cur_x_test_base_vals_for_pred.shape))

    #we must inverse transform our initial condition so the deltas can be added to it properly
    #month
    if scaler_flag==True:
        cur_x_test_base_vals_for_pred=scaler.inverse_transform(cur_x_test_base_vals_for_pred)

    #for the base vals

    cur_x_test_base_vals_for_pred=cur_x_test_base_vals_for_pred.flatten()
    print("Shape of flattened cur_x_test_base_vals_for_pred: %s" %str(cur_x_test_base_vals_for_pred.shape))

    #remove days of the week
    cur_month_array=cur_x_test_base_vals_for_pred[-num_months:]
    print("cur_month_array: %s" %str(cur_month_array))
    cur_x_test_base_vals_for_pred=cur_x_test_base_vals_for_pred[:-num_months]
    # print("Cur base shape: %s" %str(cur_x_test_base_vals_for_pred.shape))

    #FOR SANITY CHECKING
    # gt_cur_x_test_base_vals=cur_x_test_base_vals_for_pred.copy()

    #save all predicted sequences
    all_predicted_sequences=[]

    #get fts in a time step
    for i in range(num_sequences_to_simulate):

        #save your new cur sequence of predictions here
        cur_sequence_of_calculated_timestep_vals_not_deltas=[]

        #might have to add back 1
        for j in range(num_timesteps_in_a_sequence):

            #get cur timestep of predicted deltas
            cur_predicted_timestep_array_of_deltas_to_add=prediction_deltas[j]
            # print("cur_predicted_timestep_array_of_deltas_to_add shape: %s" %str(cur_predicted_timestep_array_of_deltas_to_add.shape))

            #add pred deltas
            new_pred_array=cur_x_test_base_vals_for_pred +  cur_predicted_timestep_array_of_deltas_to_add
            # print("new_pred_array: %s" %str(new_pred_array.shape))

            #now we must get the next month to add back
            month=np.argmax(cur_month_array)
            print("month is %d" %month)

            #next month
            next_month=(month + 1)%num_months
            print("Next month: %d" %next_month)

            #next month as array
            next_month_as_array=month_dict[next_month]
            print("Next month as array: %s" %str(next_month_as_array))

            #update current
            cur_month_array=next_month_as_array

            #before we concat, make this new pred array without the month available for the next round of delta adding
            cur_x_test_base_vals_for_pred=new_pred_array.copy()

            #concat this next month with new pred array
            new_pred_array=np.concatenate((new_pred_array,next_month_as_array))

            #append new pred array with month to our current list of predicted timestep arrays
            cur_sequence_of_calculated_timestep_vals_not_deltas.append(new_pred_array)



        #at the end of an iteration i (a sequence) , we must make a sequence array out of all the timestep features we just calculated
        #then, we must re-normalize them to make a new x_test array
        new_x_test = np.asarray(cur_sequence_of_calculated_timestep_vals_not_deltas)
        # print("Shape of new_x_test: %s" %str(new_x_test.shape))

        #Do not normalize, we want these actual count results
        all_predicted_sequences.append(new_x_test.copy())
        print("Current number of predicted sequences: %d out of %d" %(len(all_predicted_sequences),num_sequences_to_simulate))

        #Now we have to normalize x test
        if scaler_flag==True:
            new_x_test=scaler.transform(new_x_test)
        new_x_test=new_x_test.reshape((1,new_x_test.shape[0],new_x_test.shape[1] ))

        #predict
        prediction_deltas=model.predict(new_x_test)

        #make pred array 2d
        prediction_deltas=prediction_deltas.reshape((prediction_deltas.shape[0]*prediction_deltas.shape[1],prediction_deltas.shape[2]))

        #current starting point = last day in the sequence
        new_x_test=new_x_test.reshape((new_x_test.shape[1],new_x_test.shape[2] ))
        cur_x_test_base_vals_for_pred=new_x_test[num_timesteps_in_a_sequence-1:, ]

        #we must inverse transform our new text x condition so the deltas can be added to it properly
        if scaler_flag==True:
            cur_x_test_base_vals_for_pred=scaler.inverse_transform(cur_x_test_base_vals_for_pred)
        cur_x_test_base_vals_for_pred=cur_x_test_base_vals_for_pred.flatten()
        # print("Cur base shape: %s" %str(cur_x_test_base_vals_for_pred.shape))

        #remove days of the week
        saved_month_array=cur_x_test_base_vals_for_pred[-num_months:]
        print("saved_month_array: %s" %str(saved_month_array))
        cur_x_test_base_vals_for_pred=cur_x_test_base_vals_for_pred[:-num_months]

    #convert our new predicted list into array
    all_predicted_sequences=np.asarray(all_predicted_sequences)
    all_predicted_sequences=all_predicted_sequences[:,:,:-num_months]
    print("all_predicted_sequences shape: %s" %str(all_predicted_sequences.shape))

    print("############ Simulation complete! ################")
    return all_predicted_sequences



def backup_simulate_on_initial_condition_not_a_drill_with_month_features(x_test, model,scaler, num_sequences_to_simulate,scaler_flag=True):
    #sanity checking tag means these are arrays we should not have access to during the challenge
    print("############ Now simulating on initial condition... ################")

    num_months=12

    #setup month dict
    month_dict={
    0: [1,0,0,0,0,0,0,0,0,0,0,0],
    1: [0,1,0,0,0,0,0,0,0,0,0,0],
    2: [0,0,1,0,0,0,0,0,0,0,0,0],
    3: [0,0,0,1,0,0,0,0,0,0,0,0],
    4: [0,0,0,0,1,0,0,0,0,0,0,0],
    5: [0,0,0,0,0,1,0,0,0,0,0,0],
    6: [0,0,0,0,0,0,1,0,0,0,0,0],
    7: [0,0,0,0,0,0,0,1,0,0,0,0],
    8: [0,0,0,0,0,0,0,0,1,0,0,0],
    9: [0,0,0,0,0,0,0,0,0,1,0,0],
    10:[0,0,0,0,0,0,0,0,0,0,1,0],
    11:[0,0,0,0,0,0,0,0,0,0,0,1],
    }


    #DIFFERENCE THRESHOLD
    #SANITY CHECK PARAM
    DIFFERENCE_THRESHOLD=0.1

    num_timesteps_in_a_sequence=x_test.shape[1]

    #create init condit array
    x_init_condit=x_test[:1,:,:]
    print("Shape of x_init_condit: %s" %str(x_init_condit.shape))

    #get first sequence of predictions
    #these deltas do not have the month
    prediction_deltas=model.predict(x_init_condit)

    #make x_init_condit 2d
    #this array has the month values
    x_init_condit=x_init_condit.reshape((x_init_condit.shape[0]*x_init_condit.shape[1],x_init_condit.shape[2]))

    #make pred array 2d
    #these deltas do not have the month
    prediction_deltas=prediction_deltas.reshape((prediction_deltas.shape[0]*prediction_deltas.shape[1],prediction_deltas.shape[2]))
    # print("prediction_deltas shape: %s" %str(prediction_deltas.shape))

    #current starting point = last day in the sequence
    #2d array with shape (1,num_features_in_timestep+7)
    #month is here
    cur_x_test_base_vals_for_pred=x_init_condit[num_timesteps_in_a_sequence-1:, ]
    print("Shape of cur_x_test_base_vals_for_pred: %s" %str(cur_x_test_base_vals_for_pred.shape))

    #we must inverse transform our initial condition so the deltas can be added to it properly
    #month
    if scaler_flag==True:
        cur_x_test_base_vals_for_pred=scaler.inverse_transform(cur_x_test_base_vals_for_pred)

    #for the base vals

    cur_x_test_base_vals_for_pred=cur_x_test_base_vals_for_pred.flatten()
    print("Shape of flattened cur_x_test_base_vals_for_pred: %s" %str(cur_x_test_base_vals_for_pred.shape))

    #remove days of the week
    cur_month_array=cur_x_test_base_vals_for_pred[-num_months:]
    print("cur_month_array: %s" %str(cur_month_array))
    cur_x_test_base_vals_for_pred=cur_x_test_base_vals_for_pred[:-num_months]
    # print("Cur base shape: %s" %str(cur_x_test_base_vals_for_pred.shape))

    #FOR SANITY CHECKING
    # gt_cur_x_test_base_vals=cur_x_test_base_vals_for_pred.copy()

    #save all predicted sequences
    all_predicted_sequences=[]

    #get fts in a time step
    for i in range(num_sequences_to_simulate):

        #save your new cur sequence of predictions here
        cur_sequence_of_calculated_timestep_vals_not_deltas=[]

        #might have to add back 1
        for j in range(num_timesteps_in_a_sequence):

            #get cur timestep of predicted deltas
            cur_predicted_timestep_array_of_deltas_to_add=prediction_deltas[j]
            # print("cur_predicted_timestep_array_of_deltas_to_add shape: %s" %str(cur_predicted_timestep_array_of_deltas_to_add.shape))

            #add pred deltas
            new_pred_array=cur_x_test_base_vals_for_pred +  cur_predicted_timestep_array_of_deltas_to_add
            # print("new_pred_array: %s" %str(new_pred_array.shape))

            #now we must get the next month to add back
            month=np.argmax(cur_month_array)
            print("month is %d" %month)

            #next month
            next_month=(month + 1)%num_months
            print("Next month: %d" %next_month)

            #next month as array
            next_month_as_array=month_dict[next_month]
            print("Next month as array: %s" %str(next_month_as_array))

            #update current
            cur_month_array=next_month_as_array

            #before we concat, make this new pred array without the month available for the next round of delta adding
            cur_x_test_base_vals_for_pred=new_pred_array.copy()

            #concat this next month with new pred array
            new_pred_array=np.concatenate((new_pred_array,next_month_as_array))

            #append new pred array with month to our current list of predicted timestep arrays
            cur_sequence_of_calculated_timestep_vals_not_deltas.append(new_pred_array)



        #at the end of an iteration i (a sequence) , we must make a sequence array out of all the timestep features we just calculated
        #then, we must re-normalize them to make a new x_test array
        new_x_test = np.asarray(cur_sequence_of_calculated_timestep_vals_not_deltas)
        # print("Shape of new_x_test: %s" %str(new_x_test.shape))

        #Do not normalize, we want these actual count results
        all_predicted_sequences.append(new_x_test.copy())
        print("Current number of predicted sequences: %d out of %d" %(len(all_predicted_sequences),num_sequences_to_simulate))

        #Now we have to normalize x test
        if scaler_flag==True:
            new_x_test=scaler.transform(new_x_test)
        new_x_test=new_x_test.reshape((1,new_x_test.shape[0],new_x_test.shape[1] ))

        #predict
        prediction_deltas=model.predict(new_x_test)

        #make pred array 2d
        prediction_deltas=prediction_deltas.reshape((prediction_deltas.shape[0]*prediction_deltas.shape[1],prediction_deltas.shape[2]))

        #current starting point = last day in the sequence
        new_x_test=new_x_test.reshape((new_x_test.shape[1],new_x_test.shape[2] ))
        cur_x_test_base_vals_for_pred=new_x_test[num_timesteps_in_a_sequence-1:, ]

        #we must inverse transform our new text x condition so the deltas can be added to it properly
        if scaler_flag==True:
            cur_x_test_base_vals_for_pred=scaler.inverse_transform(cur_x_test_base_vals_for_pred)
        cur_x_test_base_vals_for_pred=cur_x_test_base_vals_for_pred.flatten()
        # print("Cur base shape: %s" %str(cur_x_test_base_vals_for_pred.shape))

        #remove days of the week
        saved_month_array=cur_x_test_base_vals_for_pred[-num_months:]
        print("saved_month_array: %s" %str(saved_month_array))
        cur_x_test_base_vals_for_pred=cur_x_test_base_vals_for_pred[:-num_months]

    #convert our new predicted list into array
    all_predicted_sequences=np.asarray(all_predicted_sequences)
    all_predicted_sequences=all_predicted_sequences[:,:,:-num_months]
    print("all_predicted_sequences shape: %s" %str(all_predicted_sequences.shape))

    print("############ Simulation complete! ################")
    return all_predicted_sequences
def calc_dates_for_simulation(FEATURE_DF,NUM_TIMESTEPS_IN_A_SEQUENCE,GRANULARITY):
    FEATURE_DF["created_at"]=pd.to_datetime(FEATURE_DF["created_at"],utc=True)
    init_condit_start=FEATURE_DF["created_at"].iloc[0]
    init_condit_end=FEATURE_DF["created_at"].iloc[-1]
    print("The initial condition starts on %s and ends on %s" %(str(init_condit_start),str(init_condit_end)))

    first_predicted_date=init_condit_end+ timedelta(days=1)
    n_months=14
    n_months_later = init_condit_end+relativedelta(months=12)+relativedelta(months=2)+timedelta(days=1)
    all_needed_prediction_dates=list(pd.date_range(first_predicted_date, n_months_later, freq=GRANULARITY))
    num_prediction_steps=len(all_needed_prediction_dates)
    print("There are %d days we must predict" %num_prediction_steps)
    print("First day we'll predict: %s" %str(first_predicted_date))
    print("%d months later the date will be: %s" %(n_months,n_months_later))

    #get number of times to loop the simulation
    if num_prediction_steps%NUM_TIMESTEPS_IN_A_SEQUENCE != 0:
        add_amount=1
    else:
        add_amount=0

    n_simulation_loops=int(num_prediction_steps/NUM_TIMESTEPS_IN_A_SEQUENCE) + add_amount
    print("\nWe'll need to do %d loops for the simulation" %n_simulation_loops)

    num_predicted_dates= n_simulation_loops * NUM_TIMESTEPS_IN_A_SEQUENCE
    print("Our code will predict %d dates" %num_predicted_dates)

    last_date_predicted=first_predicted_date + timedelta(days=num_predicted_dates)
    all_predicted_dates_with_extra_dates=list(pd.date_range(first_predicted_date, last_date_predicted, freq=GRANULARITY))
    print("Our prediction df will end at date %s" %str(last_date_predicted))

    return n_simulation_loops,all_needed_prediction_dates,all_predicted_dates_with_extra_dates

def calc_dates_for_simulation_monthly(FEATURE_DF,NUM_TIMESTEPS_IN_A_SEQUENCE,n_months=16):
    FEATURE_DF["created_at"]=pd.to_datetime(FEATURE_DF["created_at"],utc=True)
    init_condit_start=FEATURE_DF["created_at"].iloc[0]
    init_condit_end=FEATURE_DF["created_at"].iloc[-1]
    print("The initial condition starts on %s and ends on %s" %(str(init_condit_start),str(init_condit_end)))

    first_predicted_date=init_condit_end+ relativedelta(months=1)
    print("First predicted date: %s" %str(first_predicted_date))

    #calculate months later from predicted date
    # n_months_later = init_condit_end+timedelta(days=4)
    n_months_later = init_condit_end+relativedelta(months=1)+timedelta(days=1)
    for m in range(n_months):
        n_months_later+=relativedelta(months=1)




    # n_months_later = init_condit_end+relativedelta(months=12)+relativedelta(months=4)+timedelta(days=4)
    all_needed_prediction_dates=list(pd.date_range(first_predicted_date, n_months_later, freq="M"))
    print("Our needed prediction dates:")
    for date in all_needed_prediction_dates:
        print(date)
    num_prediction_steps=len(all_needed_prediction_dates)
    print("There are %d sequences we must predict" %num_prediction_steps)
    print("First month we'll predict: %s" %str(first_predicted_date))
    print("%d months later the date will be: %s" %(n_months,n_months_later))

    #get number of times to loop the simulation
    if num_prediction_steps%NUM_TIMESTEPS_IN_A_SEQUENCE != 0:
        add_amount=1
    else:
        add_amount=0

    n_simulation_loops=int(num_prediction_steps/NUM_TIMESTEPS_IN_A_SEQUENCE) + add_amount
    print("\nWe'll need to do %d loops for the simulation" %n_simulation_loops)

    num_predicted_dates= n_simulation_loops * NUM_TIMESTEPS_IN_A_SEQUENCE
    print("Our code will predict %d sequences" %num_predicted_dates)

    last_date_predicted=first_predicted_date
    for m in range(num_predicted_dates):
        last_date_predicted+=relativedelta(months=1)

    all_predicted_dates_with_extra_dates=list(pd.date_range(first_predicted_date, last_date_predicted, freq="M"))
    print("Our prediction df will end at date %s" %str(last_date_predicted))

    return n_simulation_loops,all_needed_prediction_dates,all_predicted_dates_with_extra_dates

def calc_dates_for_simulation_altered(FEATURE_DF,NUM_TIMESTEPS_IN_A_SEQUENCE,GRANULARITY,n_months=16):
    FEATURE_DF["created_at"]=pd.to_datetime(FEATURE_DF["created_at"],utc=True)
    init_condit_start=FEATURE_DF["created_at"].iloc[0]
    init_condit_end=FEATURE_DF["created_at"].iloc[-1]
    print("The initial condition starts on %s and ends on %s" %(str(init_condit_start),str(init_condit_end)))

    first_predicted_date=init_condit_end+ timedelta(days=1)

    #calculate months later from predicted date
    n_months_later = init_condit_end+timedelta(days=4)
    for m in range(n_months):
        n_months_later+=relativedelta(months=1)




    # n_months_later = init_condit_end+relativedelta(months=12)+relativedelta(months=4)+timedelta(days=4)
    all_needed_prediction_dates=list(pd.date_range(first_predicted_date, n_months_later, freq=GRANULARITY))
    num_prediction_steps=len(all_needed_prediction_dates)
    print("There are %d sequences we must predict" %num_prediction_steps)
    print("First day we'll predict: %s" %str(first_predicted_date))
    print("%d months later the date will be: %s" %(n_months,n_months_later))

    #get number of times to loop the simulation
    if num_prediction_steps%NUM_TIMESTEPS_IN_A_SEQUENCE != 0:
        add_amount=1
    else:
        add_amount=0

    n_simulation_loops=int(num_prediction_steps/NUM_TIMESTEPS_IN_A_SEQUENCE) + add_amount
    print("\nWe'll need to do %d loops for the simulation" %n_simulation_loops)

    num_predicted_dates= n_simulation_loops * NUM_TIMESTEPS_IN_A_SEQUENCE
    print("Our code will predict %d sequences" %num_predicted_dates)

    if GRANULARITY=="D":
        last_date_predicted=first_predicted_date + timedelta(days=num_predicted_dates)
    elif GRANULARITY=="W":
        last_date_predicted=first_predicted_date + timedelta(days=num_predicted_dates*7)
    all_predicted_dates_with_extra_dates=list(pd.date_range(first_predicted_date, last_date_predicted, freq=GRANULARITY))
    print("Our prediction df will end at date %s" %str(last_date_predicted))

    return n_simulation_loops,all_needed_prediction_dates,all_predicted_dates_with_extra_dates

def convert_arrays_to_sliding_window_format(x,y_as_values,y_as_deltas, window_size=1):
    print("Converting arrays to sliding window format...")

    #we need these later
    num_sequences=x.shape[0]
    num_timesteps_in_a_sequence=x.shape[1]
    num_fts_in_a_timestep=x.shape[2]

    #new shape
    new_2d_shape=(x.shape[0]*x.shape[1],x.shape[2])
    x=x.reshape(new_2d_shape)
    y_as_values=y_as_values.reshape(new_2d_shape)
    y_as_deltas=y_as_deltas.reshape(new_2d_shape)

    #these will be our new arrays
    new_x=[]
    new_y_as_values=[]
    new_y_as_deltas=[]

    #set number of iterations
    num_iterations=x.shape[0]-num_timesteps_in_a_sequence+window_size
    print("The formula for the number of new array elements for the sliding window configuration:")
    print("(Number of array elements) - (Number of timesteps in a sequence) + WINDOW SIZE")
    print("In this case: ")
    print("%d - %d + %d = %d" %(x.shape[0], num_timesteps_in_a_sequence, window_size, num_iterations))

    #make sliding window data!
    for i in range(num_iterations):
        cur_x_slice=x[i:i+num_timesteps_in_a_sequence,:]
        cur_y_value_slice=y_as_values[i:i+num_timesteps_in_a_sequence,:]
        cur_y_delta_slice=y_as_deltas[i:i+num_timesteps_in_a_sequence,:]

        #reshape new arrays
        # new_3d_shape=(1, cur_x_slice.shape[0],cur_x_slice.shape[1])
        # cur_x_slice=cur_x_slice.reshape(new_3d_shape)
        # cur_y_value_slice=cur_y_value_slice.reshape(new_3d_shape)
        # cur_y_delta_slice=cur_y_delta_slice.reshape(new_3d_shape)

        #append
        new_x.append(cur_x_slice)
        new_y_as_values.append(cur_y_value_slice)
        new_y_as_deltas.append(cur_y_delta_slice)

    #convert these to arrays
    new_x=np.asarray(new_x)
    new_y_as_values=np.asarray(new_y_as_values)
    new_y_as_deltas=np.asarray(new_y_as_deltas)

    print("New x array shape: %s" %str(new_x.shape))
    print("new_y_as_values shape: %s" %str( new_y_as_values.shape))
    print("new_y_as_deltas: %s" %str(new_y_as_deltas.shape))

    return new_x,new_y_as_values,new_y_as_deltas

def convert_arrays_to_sliding_window_format_with_or_without_DOW(initial_dow,x_train, x_val, x_test, y_train_as_values,y_train_as_deltas, y_val_as_values, y_val_as_deltas, y_test_as_values,y_test_as_deltas, DOW_FLAG,window_size=1):

    if DOW_FLAG==True:
        x_train, x_val, x_test, y_train_as_values,y_train_as_deltas, y_val_as_values, y_val_as_deltas, y_test_as_values, y_test_as_deltas=convert_arrays_to_sliding_window_format_with_DOW(initial_dow,x_train,
            x_val, x_test,
            y_train_as_values,
            y_train_as_deltas,
            y_val_as_values,
            y_val_as_deltas, y_test_as_values,y_test_as_deltas, window_size=1)

    else:
        x_train, x_val, x_test, y_train_as_values,y_train_as_deltas, y_val_as_values, y_val_as_deltas, y_test_as_values, y_test_as_deltas=convert_arrays_to_sliding_window_format_with_NO_DOW(initial_dow,x_train,
            x_val, x_test,
            y_train_as_values,
            y_train_as_deltas,
            y_val_as_values,
            y_val_as_deltas, y_test_as_values,y_test_as_deltas, window_size=1)


    return x_train, x_val, x_test, y_train_as_values,y_train_as_deltas, y_val_as_values, y_val_as_deltas, y_test_as_values, y_test_as_deltas

def convert_arrays_to_sliding_window_format_for_sep_x_and_y(x,y_as_values,y_as_deltas, window_size=1):
    print("Converting arrays to sliding window format...")

    #we need these later
    num_sequences=x.shape[0]
    num_timesteps_in_a_sequence=x.shape[1]
    num_fts_in_a_timestep=x.shape[2]

    #new shape
    new_x_2d_shape=(x.shape[0]*x.shape[1],x.shape[2])
    new_y_2d_shape=(y_as_values.shape[0]*y_as_values.shape[1],y_as_values.shape[2])

    #reshape arrays
    x=x.reshape(new_x_2d_shape)
    y_as_values=y_as_values.reshape(new_y_2d_shape)
    y_as_deltas=y_as_deltas.reshape(new_y_2d_shape)

    #these will be our new arrays
    new_x=[]
    new_y_as_values=[]
    new_y_as_deltas=[]

    #set number of iterations
    num_iterations=x.shape[0]-num_timesteps_in_a_sequence+window_size
    print("The formula for the number of new array elements for the sliding window configuration:")
    print("(Number of array elements) - (Number of timesteps in a sequence) + WINDOW SIZE")
    print("In this case: ")
    print("%d - %d + %d = %d" %(x.shape[0], num_timesteps_in_a_sequence, window_size, num_iterations))

    #make sliding window data!
    for i in range(num_iterations):
        cur_x_slice=x[i:i+num_timesteps_in_a_sequence,:]
        cur_y_value_slice=y_as_values[i:i+num_timesteps_in_a_sequence,:]
        cur_y_delta_slice=y_as_deltas[i:i+num_timesteps_in_a_sequence,:]

        #reshape new arrays
        # new_3d_shape=(1, cur_x_slice.shape[0],cur_x_slice.shape[1])
        # cur_x_slice=cur_x_slice.reshape(new_3d_shape)
        # cur_y_value_slice=cur_y_value_slice.reshape(new_3d_shape)
        # cur_y_delta_slice=cur_y_delta_slice.reshape(new_3d_shape)

        #append
        new_x.append(cur_x_slice)
        new_y_as_values.append(cur_y_value_slice)
        new_y_as_deltas.append(cur_y_delta_slice)

    #convert these to arrays
    new_x=np.asarray(new_x)
    new_y_as_values=np.asarray(new_y_as_values)
    new_y_as_deltas=np.asarray(new_y_as_deltas)

    print("New x array shape: %s" %str(new_x.shape))
    print("new_y_as_values shape: %s" %str( new_y_as_values.shape))
    print("new_y_as_deltas: %s" %str(new_y_as_deltas.shape))

    return new_x,new_y_as_values,new_y_as_deltas

def properly_convert_arrays_to_sliding_window_format_for_sep_x_and_y(x,y_as_values,y_as_deltas, window_size=1):
    print("Converting arrays to sliding window format...")

    #get timesteps for later
    xtimesteps_in_a_sequence=x.shape[1]
    ytimesteps_in_a_sequence=y_as_values.shape[1]

    #get total timesteps in seq
    xy_timesteps_in_a_sequence=xtimesteps_in_a_sequence + ytimesteps_in_a_sequence
    print("X timesteps in a sequence: %d" %xtimesteps_in_a_sequence)
    print("y timesteps in a sequence: %d" %ytimesteps_in_a_sequence)
    print("xy timesteps in a sequence: %d" %xy_timesteps_in_a_sequence)

    #combine x and y for now
    x_and_y_values=np.concatenate([x, y_as_values], axis=1)
    x_and_y_deltas=np.concatenate([x, y_as_deltas], axis=1)
    print("Shape of x_and_y_values: %s" %str(x_and_y_values.shape))
    print("Shape of x_and_y_deltas: %s" %str(x_and_y_deltas.shape))

    # #new shape
    new_xy_2d_shape=(x_and_y_values.shape[0]*x_and_y_values.shape[1],x_and_y_values.shape[2])
    # new_y_2d_shape=(y_as_values.shape[0]*y_as_values.shape[1],y_as_values.shape[2])

    # #reshape arrays
    x_and_y_values=x_and_y_values.reshape(new_xy_2d_shape)
    x_and_y_deltas=x_and_y_deltas.reshape(new_xy_2d_shape)

    #these will be our new arrays
    new_x=[]
    new_y_as_values=[]
    new_y_as_deltas=[]

    num_iterations=x_and_y_values.shape[0]-xy_timesteps_in_a_sequence+window_size
    print("The formula for the number of new array elements for the sliding window configuration:")
    print("(Number of array elements) - (Number of timesteps in a sequence) + WINDOW SIZE")


    #set number of iterations
    print("Since we have different array sizes for x and y we have to use the array with the lower number of elements to determine the iterations (in this case y).")
    print("In this case: ")
    print("%d - %d + %d = %d" %(x_and_y_values.shape[0], xy_timesteps_in_a_sequence, window_size, num_iterations))

    #make sliding window data!
    # for i in range(num_iterations):
    for i in range(0,num_iterations,window_size):
        cur_x_slice=x_and_y_values[i:i+xtimesteps_in_a_sequence,:]
        cur_y_value_slice=x_and_y_values[i+xtimesteps_in_a_sequence:i+xtimesteps_in_a_sequence+ytimesteps_in_a_sequence,:]
        cur_y_delta_slice=x_and_y_deltas[i+xtimesteps_in_a_sequence:i+xtimesteps_in_a_sequence+ytimesteps_in_a_sequence,:]

        #append
        new_x.append(cur_x_slice)
        new_y_as_values.append(cur_y_value_slice)
        new_y_as_deltas.append(cur_y_delta_slice)

    # # #convert these to arrays
    new_x=np.asarray(new_x)
    new_y_as_values=np.asarray(new_y_as_values)
    new_y_as_deltas=np.asarray(new_y_as_deltas)

    print("New x array shape: %s" %str(new_x.shape))
    print("new_y_as_values shape: %s" %str( new_y_as_values.shape))
    print("new_y_as_deltas: %s" %str(new_y_as_deltas.shape))

    return new_x,new_y_as_values,new_y_as_deltas

def debug_print(str_to_print,LIMIT_TEST=0,DEBUG=False):
    LIMIT=10
    if (LIMIT_TEST < LIMIT) and DEBUG==True:
        print(str_to_print)

def convert_arrays_to_sliding_window_format_with_DOW(initial_dow,x_train, x_val, x_test, y_train_as_values,y_train_as_deltas, y_val_as_values, y_val_as_deltas, y_test_as_values,y_test_as_deltas, window_size=1):

    #setup dow dict
    dow_dict={
    0: [1,0,0,0,0,0,0],
    1: [0,1,0,0,0,0,0],
    2: [0,0,1,0,0,0,0],
    3: [0,0,0,1,0,0,0],
    4: [0,0,0,0,1,0,0],
    5: [0,0,0,0,0,1,0],
    6: [0,0,0,0,0,0,1]
    }

    ############################ FIRST ADD DOW TO X TRAIN ############################
    #get orig shape
    original_shape=x_train.shape

    all_dow=[]

    #let's add dow to the training data
    #first make the training data 2d
    #new shape
    new_2d_shape=(x_train.shape[0]*x_train.shape[1],x_train.shape[2])
    x_train=x_train.reshape(new_2d_shape)

    #make new arrays for dow
    new_x_train=[]


    #make initial dow array
    cur_dow=initial_dow
    cur_dow_array=np.asarray(dow_dict[cur_dow])

    print("Adding DOW to training data...")
    for i in range(x_train.shape[0]):

        #create temp subrarry
        temp_x_train_subarray=np.concatenate((x_train[i],cur_dow_array))
        # print("\nShape of temp_x_train_subarray: %s" %str(temp_x_train_subarray.shape))
        # print(temp_x_train_subarray)
        new_x_train.append(temp_x_train_subarray)

        #get next dow array
        cur_dow=(cur_dow+1)%7

        cur_dow_array=np.asarray(dow_dict[cur_dow])
        # print(cur_dow_array)

    #turn these new lists into arrays
    x_train=np.asarray(new_x_train)
    print("Shape of new x_train: %s" %str(x_train.shape))

    #create new shape
    new_3d_shape=(original_shape[0], original_shape[1], original_shape[2]+7)
    x_train=x_train.reshape(new_3d_shape)
    ############################ DONE ADDING DOW TO X TRAIN ############################


    ############################ NOW MAKE ALL TRAINING ARRAYS SLIDING WINDOW ARRAYS ############################
    # #get sliding window format for the training data

    #get dow subarrays
    # all_x_train_dow_subarrays=x_train[:,:,-7:]
    # x_train=x_train[:,:,:-7]




    ########## wed 7:11 june 6 we gotta make new sliding window code! ##########
    x_train,y_train_as_values,y_train_as_deltas=convert_arrays_to_sliding_window_format_for_sep_x_and_y(x_train,y_train_as_values,y_train_as_deltas,window_size)

    # x_train,y_train_as_values,y_train_as_deltas=convert_arrays_to_sliding_window_format(x_train,y_train_as_values,y_train_as_deltas,window_size)
    # ############################ DONE MAKING ALL TRAINING ARRAYS SLIDING WINDOW ARRAYS ############################
    # x_train=np.concatenate((x_train))
    # # #convert arrays to original shape
    # # x_train=x_train.reshape(new_3d_shape)

    # ##############NOW FOR VALIDATION ARRAYS ###############
    # #now we must insert DOW into val and test arrays
    # new_2d_shape=(x_val.shape[0]*x_val.shape[1],x_val.shape[2])
    # x_val=x_val.reshape(new_2d_shape)


    #lists for new val arrays
    #make new arrays for dow
    new_x_val=[]

    print("Adding DOW to validation data...")
    #make initial dow array
    cur_dow=initial_dow
    cur_dow_array=np.asarray(dow_dict[cur_dow])

    #original val shape
    original_val_shape=x_val.shape

    #reshape val array
    new_2d_shape=(x_val.shape[0]*x_val.shape[1],x_val.shape[2])
    x_val=x_val.reshape(new_2d_shape)
    # print("Adding DOW to validation data")
    for i in range(x_val.shape[0]):
        #create temp subrarry
        temp_x_val_subarray=np.concatenate((x_val[i],cur_dow_array))
        # print("\nShape of temp_x_val_subarray: %s" %str(temp_x_val_subarray.shape))
        # print(temp_x_val_subarray)
        new_x_val.append(temp_x_val_subarray)

        #get next dow array
        cur_dow=(cur_dow+1)%7

        cur_dow_array=np.asarray(dow_dict[cur_dow])
        # print(cur_dow_array)

    x_val=np.asarray(new_x_val)

    #reshape new x val
    new_val_shape=(original_val_shape[0],original_val_shape[1],original_val_shape[2]+7)
    x_val=x_val.reshape(new_val_shape)
    print("New x val shape: %s" %str(x_val.shape))

    ########## wed 7:11 june 6 we gotta make new sliding window code! ##########
    x_val,y_val_as_values,y_val_as_deltas=convert_arrays_to_sliding_window_format_for_sep_x_and_y(x_val,y_val_as_values,y_val_as_deltas,window_size)

    #lists for new test array
    #make new arrays for dow
    new_x_test=[]

    #original test shape
    original_test_shape=x_test.shape

    #reshape test array
    new_2d_shape=(x_test.shape[0]*x_test.shape[1],x_test.shape[2])
    x_test=x_test.reshape(new_2d_shape)

    #make initial dow array
    cur_dow=initial_dow
    cur_dow_array=np.asarray(dow_dict[cur_dow])

    print("Adding DOW to test data...")
    for i in range(x_test.shape[0]):
        #create temp subrarry
        temp_x_test_subarray=np.concatenate((x_test[i],cur_dow_array))
        # print("\nShape of temp_x_test_subarray: %s" %str(temp_x_test_subarray.shape))
        # print(temp_x_test_subarray)
        new_x_test.append(temp_x_test_subarray)

        #get next dow array
        cur_dow=(cur_dow+1)%7

        cur_dow_array=np.asarray(dow_dict[cur_dow])
        # print(cur_dow_array)

    x_test=np.asarray(new_x_test)

    #reshape new x test
    new_test_shape=(original_test_shape[0],original_test_shape[1],original_test_shape[2]+7)
    x_test=x_test.reshape(new_test_shape)
    print("New x test shape: %s" %str(x_test.shape))


    return x_train, x_val, x_test, y_train_as_values,y_train_as_deltas, y_val_as_values, y_val_as_deltas, y_test_as_values,y_test_as_deltas

def convert_arrays_to_sliding_window_format_with_MOY(initial_dow,x_train, x_val, x_test, y_train_as_values,y_train_as_deltas, y_val_as_values, y_val_as_deltas, y_test_as_values,y_test_as_deltas, window_size=1):

    #setup dow dict
    dow_dict={
    0: [1,0,0,0,0,0,0],
    1: [0,1,0,0,0,0,0],
    2: [0,0,1,0,0,0,0],
    3: [0,0,0,1,0,0,0],
    4: [0,0,0,0,1,0,0],
    5: [0,0,0,0,0,1,0],
    6: [0,0,0,0,0,0,1]
    }

    ############################ FIRST ADD DOW TO X TRAIN ############################
    #get orig shape
    original_shape=x_train.shape

    all_dow=[]

    #let's add dow to the training data
    #first make the training data 2d
    #new shape
    new_2d_shape=(x_train.shape[0]*x_train.shape[1],x_train.shape[2])
    x_train=x_train.reshape(new_2d_shape)

    #make new arrays for dow
    new_x_train=[]


    #make initial dow array
    cur_dow=initial_dow
    cur_dow_array=np.asarray(dow_dict[cur_dow])

    print("Adding DOW to training data...")
    for i in range(x_train.shape[0]):

        #create temp subrarry
        temp_x_train_subarray=np.concatenate((x_train[i],cur_dow_array))
        # print("\nShape of temp_x_train_subarray: %s" %str(temp_x_train_subarray.shape))
        # print(temp_x_train_subarray)
        new_x_train.append(temp_x_train_subarray)

        #get next dow array
        cur_dow=(cur_dow+1)%7

        cur_dow_array=np.asarray(dow_dict[cur_dow])
        # print(cur_dow_array)

    #turn these new lists into arrays
    x_train=np.asarray(new_x_train)
    print("Shape of new x_train: %s" %str(x_train.shape))

    #create new shape
    new_3d_shape=(original_shape[0], original_shape[1], original_shape[2]+7)
    x_train=x_train.reshape(new_3d_shape)
    ############################ DONE ADDING DOW TO X TRAIN ############################


    ############################ NOW MAKE ALL TRAINING ARRAYS SLIDING WINDOW ARRAYS ############################
    # #get sliding window format for the training data

    #get dow subarrays
    # all_x_train_dow_subarrays=x_train[:,:,-7:]
    # x_train=x_train[:,:,:-7]




    ########## wed 7:11 june 6 we gotta make new sliding window code! ##########
    x_train,y_train_as_values,y_train_as_deltas=convert_arrays_to_sliding_window_format_for_sep_x_and_y(x_train,y_train_as_values,y_train_as_deltas,window_size)

    # x_train,y_train_as_values,y_train_as_deltas=convert_arrays_to_sliding_window_format(x_train,y_train_as_values,y_train_as_deltas,window_size)
    # ############################ DONE MAKING ALL TRAINING ARRAYS SLIDING WINDOW ARRAYS ############################
    # x_train=np.concatenate((x_train))
    # # #convert arrays to original shape
    # # x_train=x_train.reshape(new_3d_shape)

    # ##############NOW FOR VALIDATION ARRAYS ###############
    # #now we must insert DOW into val and test arrays
    # new_2d_shape=(x_val.shape[0]*x_val.shape[1],x_val.shape[2])
    # x_val=x_val.reshape(new_2d_shape)


    #lists for new val arrays
    #make new arrays for dow
    new_x_val=[]

    print("Adding DOW to validation data...")
    #make initial dow array
    cur_dow=initial_dow
    cur_dow_array=np.asarray(dow_dict[cur_dow])

    #original val shape
    original_val_shape=x_val.shape

    #reshape val array
    new_2d_shape=(x_val.shape[0]*x_val.shape[1],x_val.shape[2])
    x_val=x_val.reshape(new_2d_shape)
    # print("Adding DOW to validation data")
    for i in range(x_val.shape[0]):
        #create temp subrarry
        temp_x_val_subarray=np.concatenate((x_val[i],cur_dow_array))
        # print("\nShape of temp_x_val_subarray: %s" %str(temp_x_val_subarray.shape))
        # print(temp_x_val_subarray)
        new_x_val.append(temp_x_val_subarray)

        #get next dow array
        cur_dow=(cur_dow+1)%7

        cur_dow_array=np.asarray(dow_dict[cur_dow])
        # print(cur_dow_array)

    x_val=np.asarray(new_x_val)

    #reshape new x val
    new_val_shape=(original_val_shape[0],original_val_shape[1],original_val_shape[2]+7)
    x_val=x_val.reshape(new_val_shape)
    print("New x val shape: %s" %str(x_val.shape))

    ########## wed 7:11 june 6 we gotta make new sliding window code! ##########
    x_val,y_val_as_values,y_val_as_deltas=convert_arrays_to_sliding_window_format_for_sep_x_and_y(x_val,y_val_as_values,y_val_as_deltas,window_size)

    #lists for new test array
    #make new arrays for dow
    new_x_test=[]

    #original test shape
    original_test_shape=x_test.shape

    #reshape test array
    new_2d_shape=(x_test.shape[0]*x_test.shape[1],x_test.shape[2])
    x_test=x_test.reshape(new_2d_shape)

    #make initial dow array
    cur_dow=initial_dow
    cur_dow_array=np.asarray(dow_dict[cur_dow])

    print("Adding DOW to test data...")
    for i in range(x_test.shape[0]):
        #create temp subrarry
        temp_x_test_subarray=np.concatenate((x_test[i],cur_dow_array))
        # print("\nShape of temp_x_test_subarray: %s" %str(temp_x_test_subarray.shape))
        # print(temp_x_test_subarray)
        new_x_test.append(temp_x_test_subarray)

        #get next dow array
        cur_dow=(cur_dow+1)%7

        cur_dow_array=np.asarray(dow_dict[cur_dow])
        # print(cur_dow_array)

    x_test=np.asarray(new_x_test)

    #reshape new x test
    new_test_shape=(original_test_shape[0],original_test_shape[1],original_test_shape[2]+7)
    x_test=x_test.reshape(new_test_shape)
    print("New x test shape: %s" %str(x_test.shape))


    return x_train, x_val, x_test, y_train_as_values,y_train_as_deltas, y_val_as_values, y_val_as_deltas, y_test_as_values,y_test_as_deltas

def convert_arrays_to_sliding_window_format_with_NO_DOW(initial_dow,x_train, x_val, x_test, y_train_as_values,y_train_as_deltas, y_val_as_values, y_val_as_deltas, y_test_as_values,y_test_as_deltas, window_size=1):


    x_train,y_train_as_values,y_train_as_deltas=convert_arrays_to_sliding_window_format_for_sep_x_and_y(x_train,y_train_as_values,y_train_as_deltas,window_size)
    x_val,y_val_as_values,y_val_as_deltas=convert_arrays_to_sliding_window_format_for_sep_x_and_y(x_val,y_val_as_values,y_val_as_deltas,window_size)

    return x_train, x_val, x_test, y_train_as_values,y_train_as_deltas, y_val_as_values, y_val_as_deltas, y_test_as_values,y_test_as_deltas

# def run_pnnl_simulation()
def convert_arrays_to_sliding_window_format_with_month_features(initial_month,x_train, x_val, x_test, y_train_as_values,y_train_as_deltas, y_val_as_values, y_val_as_deltas, y_test_as_values,y_test_as_deltas, window_size=1):

    #setup month
    month_dict={
    0: [1,0,0,0,0,0,0,0,0,0,0,0],
    1: [0,1,0,0,0,0,0,0,0,0,0,0],
    2: [0,0,1,0,0,0,0,0,0,0,0,0],
    3: [0,0,0,1,0,0,0,0,0,0,0,0],
    4: [0,0,0,0,1,0,0,0,0,0,0,0],
    5: [0,0,0,0,0,1,0,0,0,0,0,0],
    6: [0,0,0,0,0,0,1,0,0,0,0,0],
    7: [0,0,0,0,0,0,0,1,0,0,0,0],
    8: [0,0,0,0,0,0,0,0,1,0,0,0],
    9: [0,0,0,0,0,0,0,0,0,1,0,0],
    10:[0,0,0,0,0,0,0,0,0,0,1,0],
    11:[0,0,0,0,0,0,0,0,0,0,0,1],
    }


    ############################ FIRST ADD month TO X TRAIN ############################
    #get orig shape
    original_shape=x_train.shape

    # all_month=[]

    #let's add month to the training data
    #first make the training data 2d
    #new shape
    new_2d_shape=(x_train.shape[0]*x_train.shape[1],x_train.shape[2])
    x_train=x_train.reshape(new_2d_shape)

    #make new arrays for month
    new_x_train=[]


    #make initial month array
    cur_month=initial_month
    cur_month_array=np.asarray(month_dict[cur_month])

    print("Adding month to training data...")
    for i in range(x_train.shape[0]):

        #create temp subrarry
        temp_x_train_subarray=np.concatenate((x_train[i],cur_month_array))
        # print("\nShape of temp_x_train_subarray: %s" %str(temp_x_train_subarray.shape))
        # print(temp_x_train_subarray)
        new_x_train.append(temp_x_train_subarray)

        #get next month array
        cur_month=(cur_month+1)%12

        cur_month_array=np.asarray(month_dict[cur_month])
        # print(cur_month_array)

    #turn these new lists into arrays
    x_train=np.asarray(new_x_train)
    print("Shape of new x_train: %s" %str(x_train.shape))

    #create new shape
    new_3d_shape=(original_shape[0], original_shape[1], original_shape[2]+12)
    x_train=x_train.reshape(new_3d_shape)
    ############################ DONE ADDING DOW TO X TRAIN ############################


    ############################ NOW MAKE ALL TRAINING ARRAYS SLIDING WINDOW ARRAYS ############################
    # #get sliding window format for the training data

    #get dow subarrays
    # all_x_train_dow_subarrays=x_train[:,:,-12:]
    # x_train=x_train[:,:,:-12]




    # ########## wed 12:11 june 6 we gotta make new sliding window code! ##########
    x_train,y_train_as_values,y_train_as_deltas=convert_arrays_to_sliding_window_format_for_sep_x_and_y(x_train,y_train_as_values,y_train_as_deltas,window_size)

    # x_train,y_train_as_values,y_train_as_deltas=convert_arrays_to_sliding_window_format(x_train,y_train_as_values,y_train_as_deltas,window_size)
    # ############################ DONE MAKING ALL TRAINING ARRAYS SLIDING WINDOW ARRAYS ############################
    # x_train=np.concatenate((x_train))
    # # #convert arrays to original shape
    # # x_train=x_train.reshape(new_3d_shape)

    # ##############NOW FOR VALIDATION ARRAYS ###############
    # #now we must insert DOW into val and test arrays
    # new_2d_shape=(x_val.shape[0]*x_val.shape[1],x_val.shape[2])
    # x_val=x_val.reshape(new_2d_shape)


    #lists for new val arrays
    #make new arrays for dow
    new_x_val=[]

    print("Adding month to validation data...")
    #make initial month array
    cur_month=initial_month
    cur_month_array=np.asarray(month_dict[cur_month])

    #original val shape
    original_val_shape=x_val.shape

    #reshape val array
    new_2d_shape=(x_val.shape[0]*x_val.shape[1],x_val.shape[2])
    x_val=x_val.reshape(new_2d_shape)
    # print("Adding month to validation data")
    for i in range(x_val.shape[0]):
        #create temp subrarry
        temp_x_val_subarray=np.concatenate((x_val[i],cur_month_array))
        # print("\nShape of temp_x_val_subarray: %s" %str(temp_x_val_subarray.shape))
        # print(temp_x_val_subarray)
        new_x_val.append(temp_x_val_subarray)

        #get next month array
        cur_month=(cur_month+1)%12

        cur_month_array=np.asarray(month_dict[cur_month])
        # print(cur_month_array)

    x_val=np.asarray(new_x_val)

    #reshape new x val
    new_val_shape=(original_val_shape[0],original_val_shape[1],original_val_shape[2]+12)
    x_val=x_val.reshape(new_val_shape)
    print("New x val shape: %s" %str(x_val.shape))

    #lists for new test array
    #make new arrays for month
    new_x_test=[]

    #original test shape
    original_test_shape=x_test.shape

    #reshape test array
    new_2d_shape=(x_test.shape[0]*x_test.shape[1],x_test.shape[2])
    x_test=x_test.reshape(new_2d_shape)

    #make initial month array
    cur_month=initial_month
    cur_month_array=np.asarray(month_dict[cur_month])

    #get sliding window form for val data
    x_val,y_val_as_values,y_val_as_deltas=convert_arrays_to_sliding_window_format_for_sep_x_and_y(x_val,y_val_as_values,y_val_as_deltas,window_size)

    print("Adding month to test data...")
    for i in range(x_test.shape[0]):
        #create temp subrarry
        temp_x_test_subarray=np.concatenate((x_test[i],cur_month_array))
        # print("\nShape of temp_x_test_subarray: %s" %str(temp_x_test_subarray.shape))
        # print(temp_x_test_subarray)
        new_x_test.append(temp_x_test_subarray)

        #get next dow array
        cur_month=(cur_month+1)%12

        cur_month_array=np.asarray(month_dict[cur_month])
        # print(cur_dow_array)

    x_test=np.asarray(new_x_test)

    #reshape new x test
    new_test_shape=(original_test_shape[0],original_test_shape[1],original_test_shape[2]+12)
    x_test=x_test.reshape(new_test_shape)
    print("New x test shape: %s" %str(x_test.shape))


    return x_train, x_val, x_test, y_train_as_values,y_train_as_deltas, y_val_as_values, y_val_as_deltas, y_test_as_values,y_test_as_deltas

def convert_testing_array_to_sliding_window_format_with_month_features(initial_month,x_test,  y_test_as_values,y_test_as_deltas, window_size=1):

    #setup month
    month_dict={
    0: [1,0,0,0,0,0,0,0,0,0,0,0],
    1: [0,1,0,0,0,0,0,0,0,0,0,0],
    2: [0,0,1,0,0,0,0,0,0,0,0,0],
    3: [0,0,0,1,0,0,0,0,0,0,0,0],
    4: [0,0,0,0,1,0,0,0,0,0,0,0],
    5: [0,0,0,0,0,1,0,0,0,0,0,0],
    6: [0,0,0,0,0,0,1,0,0,0,0,0],
    7: [0,0,0,0,0,0,0,1,0,0,0,0],
    8: [0,0,0,0,0,0,0,0,1,0,0,0],
    9: [0,0,0,0,0,0,0,0,0,1,0,0],
    10:[0,0,0,0,0,0,0,0,0,0,1,0],
    11:[0,0,0,0,0,0,0,0,0,0,0,1],
    }

   #lists for new test array
    #make new arrays for month
    new_x_test=[]

    #original test shape
    original_test_shape=x_test.shape

    #reshape test array
    new_2d_shape=(x_test.shape[0]*x_test.shape[1],x_test.shape[2])
    x_test=x_test.reshape(new_2d_shape)

    #make initial month array
    cur_month=initial_month
    cur_month_array=np.asarray(month_dict[cur_month])


    print("Adding month to test data...")
    for i in range(x_test.shape[0]):
        #create temp subrarry
        temp_x_test_subarray=np.concatenate((x_test[i],cur_month_array))
        # print("\nShape of temp_x_test_subarray: %s" %str(temp_x_test_subarray.shape))
        # print(temp_x_test_subarray)
        new_x_test.append(temp_x_test_subarray)

        #get next dow array
        cur_month=(cur_month+1)%12

        cur_month_array=np.asarray(month_dict[cur_month])
        # print(cur_dow_array)

    x_test=np.asarray(new_x_test)

    #reshape new x test
    new_test_shape=(original_test_shape[0],original_test_shape[1],original_test_shape[2]+12)
    x_test=x_test.reshape(new_test_shape)
    print("New x test shape: %s" %str(x_test.shape))


    return x_test, y_test_as_values,y_test_as_deltas

def convert_testing_array_to_sliding_window_format_with_DOW_features(initial_DOW,x_test,  y_test_as_values,y_test_as_deltas, window_size=1):

    #setup DOW
    DOW_dict={
    0: [1,0,0,0,0,0,0],
    1: [0,1,0,0,0,0,0],
    2: [0,0,1,0,0,0,0],
    3: [0,0,0,1,0,0,0],
    4: [0,0,0,0,1,0,0],
    5: [0,0,0,0,0,1,0],
    6: [0,0,0,0,0,0,1]
    }

   #lists for new test array
    #make new arrays for DOW
    new_x_test=[]

    #original test shape
    original_test_shape=x_test.shape

    #reshape test array
    new_2d_shape=(x_test.shape[0]*x_test.shape[1],x_test.shape[2])
    x_test=x_test.reshape(new_2d_shape)

    #make initial DOW array
    cur_DOW=initial_DOW
    cur_DOW_array=np.asarray(DOW_dict[cur_DOW])


    print("Adding DOW to test data...")
    for i in range(x_test.shape[0]):
        #create temp subrarry
        temp_x_test_subarray=np.concatenate((x_test[i],cur_DOW_array))
        # print("\nShape of temp_x_test_subarray: %s" %str(temp_x_test_subarray.shape))
        # print(temp_x_test_subarray)
        new_x_test.append(temp_x_test_subarray)

        #get next dow array
        cur_DOW=(cur_DOW+1)%7

        cur_DOW_array=np.asarray(DOW_dict[cur_DOW])
        # print(cur_dow_array)

    x_test=np.asarray(new_x_test)

    #reshape new x test
    new_test_shape=(original_test_shape[0],original_test_shape[1],original_test_shape[2]+7)
    x_test=x_test.reshape(new_test_shape)
    print("New x test shape: %s" %str(x_test.shape))


    return x_test, y_test_as_values,y_test_as_deltas

def convert_testing_array_to_sliding_window_format_with_DOW_OPTION(initial_DOW,x_test,  y_test_as_values,y_test_as_deltas,DOW_FLAG,window_size=1):
    if DOW_FLAG == True:
        return convert_testing_array_to_sliding_window_format_with_DOW_features(initial_DOW,x_test,  y_test_as_values,y_test_as_deltas,  window_size=window_size)
    else:
        return x_test,  y_test_as_values,y_test_as_deltas

def convert_training_array_to_sliding_window_format_with_month_features(initial_month,x_train,  y_train_as_values,y_train_as_deltas,  window_size=1):

    #setup month
    month_dict={
    0: [1,0,0,0,0,0,0,0,0,0,0,0],
    1: [0,1,0,0,0,0,0,0,0,0,0,0],
    2: [0,0,1,0,0,0,0,0,0,0,0,0],
    3: [0,0,0,1,0,0,0,0,0,0,0,0],
    4: [0,0,0,0,1,0,0,0,0,0,0,0],
    5: [0,0,0,0,0,1,0,0,0,0,0,0],
    6: [0,0,0,0,0,0,1,0,0,0,0,0],
    7: [0,0,0,0,0,0,0,1,0,0,0,0],
    8: [0,0,0,0,0,0,0,0,1,0,0,0],
    9: [0,0,0,0,0,0,0,0,0,1,0,0],
    10:[0,0,0,0,0,0,0,0,0,0,1,0],
    11:[0,0,0,0,0,0,0,0,0,0,0,1],
    }


    ############################ FIRST ADD month TO X TRAIN ############################
    #get orig shape
    original_shape=x_train.shape

    # all_month=[]

    #let's add month to the training data
    #first make the training data 2d
    #new shape
    new_2d_shape=(x_train.shape[0]*x_train.shape[1],x_train.shape[2])
    x_train=x_train.reshape(new_2d_shape)

    #make new arrays for month
    new_x_train=[]


    #make initial month array
    cur_month=initial_month
    cur_month_array=np.asarray(month_dict[cur_month])

    print("Adding month to training data...")
    for i in range(x_train.shape[0]):

        #create temp subrarry
        temp_x_train_subarray=np.concatenate((x_train[i],cur_month_array))
        # print("\nShape of temp_x_train_subarray: %s" %str(temp_x_train_subarray.shape))
        # print(temp_x_train_subarray)
        new_x_train.append(temp_x_train_subarray)

        #get next month array
        cur_month=(cur_month+1)%12

        cur_month_array=np.asarray(month_dict[cur_month])
        # print(cur_month_array)

    #turn these new lists into arrays
    x_train=np.asarray(new_x_train)
    print("Shape of new x_train: %s" %str(x_train.shape))

    #create new shape
    new_3d_shape=(original_shape[0], original_shape[1], original_shape[2]+12)
    x_train=x_train.reshape(new_3d_shape)
    ############################ DONE ADDING DOW TO X TRAIN ############################



    # ########## wed 12:11 june 6 we gotta make new sliding window code! ##########
    x_train,y_train_as_values,y_train_as_deltas=convert_arrays_to_sliding_window_format_for_sep_x_and_y(x_train,y_train_as_values,y_train_as_deltas,window_size)






    return x_train, y_train_as_values,y_train_as_deltas

def convert_training_array_to_sliding_window_format_with_DOW_features(initial_DOW,x_train,  y_train_as_values,y_train_as_deltas,  window_size=1):

    #setup DOW
    DOW_dict={
    0: [1,0,0,0,0,0,0],
    1: [0,1,0,0,0,0,0],
    2: [0,0,1,0,0,0,0],
    3: [0,0,0,1,0,0,0],
    4: [0,0,0,0,1,0,0],
    5: [0,0,0,0,0,1,0],
    6: [0,0,0,0,0,0,1]
    }


    ############################ FIRST ADD DOW TO X TRAIN ############################
    #get orig shape
    original_shape=x_train.shape

    # all_DOW=[]

    #let's add DOW to the training data
    #first make the training data 2d
    #new shape
    new_2d_shape=(x_train.shape[0]*x_train.shape[1],x_train.shape[2])
    x_train=x_train.reshape(new_2d_shape)

    #make new arrays for DOW
    new_x_train=[]


    #make initial DOW array
    cur_DOW=initial_DOW
    cur_DOW_array=np.asarray(DOW_dict[cur_DOW])

    print("Adding DOW to training data...")
    for i in range(x_train.shape[0]):

        #create temp subrarry
        temp_x_train_subarray=np.concatenate((x_train[i],cur_DOW_array))
        # print("\nShape of temp_x_train_subarray: %s" %str(temp_x_train_subarray.shape))
        # print(temp_x_train_subarray)
        new_x_train.append(temp_x_train_subarray)

        #get next DOW array
        cur_DOW=(cur_DOW+1)%7

        cur_DOW_array=np.asarray(DOW_dict[cur_DOW])
        # print(cur_DOW_array)

    #turn these new lists into arrays
    x_train=np.asarray(new_x_train)
    print("Shape of new x_train: %s" %str(x_train.shape))

    #create new shape
    new_3d_shape=(original_shape[0], original_shape[1], original_shape[2]+7)
    x_train=x_train.reshape(new_3d_shape)
    ############################ DONE ADDING DOW TO X TRAIN ############################



    # ########## wed 12:11 june 6 we gotta make new sliding window code! ##########
    x_train,y_train_as_values,y_train_as_deltas=convert_arrays_to_sliding_window_format_for_sep_x_and_y(x_train,y_train_as_values,y_train_as_deltas,window_size)






    return x_train, y_train_as_values,y_train_as_deltas

def convert_training_array_to_sliding_window_format_with_DOW_OPTION(initial_DOW,x_train,  y_train_as_values,y_train_as_deltas,DOW_FLAG,window_size=1):
    if DOW_FLAG == True:
        return convert_training_array_to_sliding_window_format_with_DOW_features(initial_DOW,x_train,  y_train_as_values,y_train_as_deltas,  window_size=window_size)
    else:
        return convert_arrays_to_sliding_window_format_for_sep_x_and_y(x_train,y_train_as_values,y_train_as_deltas,window_size)



def convert_arrays_to_sliding_window_format_with_numerical_month_feature(initial_month,x_train, x_val, x_test, y_train_as_values,y_train_as_deltas, y_val_as_values, y_val_as_deltas, y_test_as_values,y_test_as_deltas, window_size=1):

    #setup month
    # month_dict={
    # 0: [1,0,0,0,0,0,0,0,0,0,0,0],
    # 1: [0,1,0,0,0,0,0,0,0,0,0,0],
    # 2: [0,0,1,0,0,0,0,0,0,0,0,0],
    # 3: [0,0,0,1,0,0,0,0,0,0,0,0],
    # 4: [0,0,0,0,1,0,0,0,0,0,0,0],
    # 5: [0,0,0,0,0,1,0,0,0,0,0,0],
    # 6: [0,0,0,0,0,0,1,0,0,0,0,0],
    # 7: [0,0,0,0,0,0,0,1,0,0,0,0],
    # 8: [0,0,0,0,0,0,0,0,1,0,0,0],
    # 9: [0,0,0,0,0,0,0,0,0,1,0,0],
    # 10:[0,0,0,0,0,0,0,0,0,0,1,0],
    # 11:[0,0,0,0,0,0,0,0,0,0,0,1],
    # }

    num_month_fts=1


    ############################ FIRST ADD month TO X TRAIN ############################
    #get orig shape
    original_shape=x_train.shape

    # all_month=[]

    #let's add month to the training data
    #first make the training data 2d
    #new shape
    new_2d_shape=(x_train.shape[0]*x_train.shape[1],x_train.shape[2])
    x_train=x_train.reshape(new_2d_shape)

    #make new arrays for month
    new_x_train=[]


    #make initial month array
    cur_month=initial_month
    cur_month_array=np.asarray([cur_month])

    print("Adding month to training data...")
    for i in range(x_train.shape[0]):

        #create temp subrarry
        temp_x_train_subarray=np.concatenate((x_train[i],cur_month_array))
        # print("\nShape of temp_x_train_subarray: %s" %str(temp_x_train_subarray.shape))
        # print(temp_x_train_subarray)
        new_x_train.append(temp_x_train_subarray)

        #get next month array
        cur_month=(cur_month+1)%12

        cur_month_array=np.asarray([cur_month])
        # print(cur_month_array)

    #turn these new lists into arrays
    x_train=np.asarray(new_x_train)
    print("Shape of new x_train: %s" %str(x_train.shape))

    #create new shape
    new_3d_shape=(original_shape[0], original_shape[1], original_shape[2]+num_month_fts)
    x_train=x_train.reshape(new_3d_shape)
    ############################ DONE ADDING DOW TO X TRAIN ############################


    ############################ NOW MAKE ALL TRAINING ARRAYS SLIDING WINDOW ARRAYS ############################
    # #get sliding window format for the training data

    #get dow subarrays
    # all_x_train_dow_subarrays=x_train[:,:,-12:]
    # x_train=x_train[:,:,:-12]




    # ########## wed 12:11 june 6 we gotta make new sliding window code! ##########
    x_train,y_train_as_values,y_train_as_deltas=convert_arrays_to_sliding_window_format_for_sep_x_and_y(x_train,y_train_as_values,y_train_as_deltas,window_size)

    # x_train,y_train_as_values,y_train_as_deltas=convert_arrays_to_sliding_window_format(x_train,y_train_as_values,y_train_as_deltas,window_size)
    # ############################ DONE MAKING ALL TRAINING ARRAYS SLIDING WINDOW ARRAYS ############################
    # x_train=np.concatenate((x_train))
    # # #convert arrays to original shape
    # # x_train=x_train.reshape(new_3d_shape)

    # ##############NOW FOR VALIDATION ARRAYS ###############
    # #now we must insert DOW into val and test arrays
    # new_2d_shape=(x_val.shape[0]*x_val.shape[1],x_val.shape[2])
    # x_val=x_val.reshape(new_2d_shape)


    #lists for new val arrays
    #make new arrays for dow
    new_x_val=[]

    print("Adding month to validation data...")
    #make initial month array
    # cur_month=(cur_month+1)%12
    # cur_month_array=np.asarray([cur_month])

    #original val shape
    original_val_shape=x_val.shape

    #reshape val array
    new_2d_shape=(x_val.shape[0]*x_val.shape[1],x_val.shape[2])
    x_val=x_val.reshape(new_2d_shape)
    # print("Adding month to validation data")
    for i in range(x_val.shape[0]):
        #create temp subrarry
        temp_x_val_subarray=np.concatenate((x_val[i],cur_month_array))
        # print("\nShape of temp_x_val_subarray: %s" %str(temp_x_val_subarray.shape))
        # print(temp_x_val_subarray)
        new_x_val.append(temp_x_val_subarray)

        #get next month array
        cur_month=(cur_month+1)%num_month_fts

        cur_month_array=np.asarray([cur_month])
        # print(cur_month_array)

    x_val=np.asarray(new_x_val)

    #reshape new x val
    new_val_shape=(original_val_shape[0],original_val_shape[1],original_val_shape[2]+num_month_fts)
    x_val=x_val.reshape(new_val_shape)
    print("New x val shape: %s" %str(x_val.shape))

    x_val,y_val_as_values,y_val_as_deltas=convert_arrays_to_sliding_window_format_for_sep_x_and_y(x_val,y_val_as_values,y_val_as_deltas,window_size)

    #lists for new test array
    #make new arrays for month
    new_x_test=[]

    #original test shape
    original_test_shape=x_test.shape

    #reshape test array
    new_2d_shape=(x_test.shape[0]*x_test.shape[1],x_test.shape[2])
    x_test=x_test.reshape(new_2d_shape)

    # #make initial month array
    # cur_month=initial_month
    # cur_month_array=np.asarray(month_dict[cur_month])

    print("Adding month to test data...")
    for i in range(x_test.shape[0]):
        #create temp subrarry
        temp_x_test_subarray=np.concatenate((x_test[i],cur_month_array))
        # print("\nShape of temp_x_test_subarray: %s" %str(temp_x_test_subarray.shape))
        # print(temp_x_test_subarray)
        new_x_test.append(temp_x_test_subarray)

        #get next dow array
        cur_month=(cur_month+1)%num_month_fts

        cur_month_array=np.asarray([cur_month])
        # print(cur_dow_array)

    x_test=np.asarray(new_x_test)

    #reshape new x test
    new_test_shape=(original_test_shape[0],original_test_shape[1],original_test_shape[2]+num_month_fts)
    x_test=x_test.reshape(new_test_shape)
    print("New x test shape: %s" %str(x_test.shape))


    return x_train, x_val, x_test, y_train_as_values,y_train_as_deltas, y_val_as_values, y_val_as_deltas, y_test_as_values,y_test_as_deltas

def perform_dry_run_simulation_for_DOW_model(coin_names, GRAPH_WITH_GT, model,LOG1_FLAG,LOG2_FLAG,scaler_flag, VERSION_OUTPUT_DIR,features,
                                        ic_start_date, ic_end_date, sim_start_date,sim_end_date,twitter_pnnl_dir,telegram_pnnl_dir,num_timesteps_in_a_sequence):

    # #get features used
    # ftfp=VERSION_OUTPUT_DIR+"All-Feature-SMAPEs.csv"
    # ft_df=pd.read_csv(ftfp)
    # features=list(ft_df["feature"])
    # print(features)
    for ft in features:
        print(ft)

    #replace ubiq with <coin name>
    new_fts=[]
    print("Putting feature names in neutral format...")
    for ft in features:
        new_ft=ft.replace("ubiq", "<coin_name>")
        print(new_ft)
        new_fts.append(new_ft)
    features=new_fts

    #now simulate by coin!
    for coin in coin_names:
        print("Now simulating %s" %coin)

        #get proper coin fts
        coin_fts=configure_coin_fts(coin, features)
        print("%s fts: %s" %(coin,coin_fts))

        #get twitter fp
        twitter_pnnl_fp=twitter_pnnl_dir + "%s/%s-features.csv"%(coin,coin)

        #get telegram fp
        telegram_pnnl_fp=telegram_pnnl_dir + "%s/%s-features.csv"%(coin,coin)

        twitter_df=pd.read_csv(twitter_pnnl_fp)
        telegram_df=pd.read_csv(telegram_pnnl_fp)
        df=pd.merge(twitter_df, telegram_df, on=["created_at"], how="inner")
        print(df)

        #get relevant fts
        df=df[coin_fts + ["created_at"]]
        print(df)

        #get gt df for later
        gtdf=config_df_by_dates(df,sim_start_date,sim_end_date,time_col="created_at").copy()

        #get first dow
        initial_dow=pd.to_datetime(df["created_at"].iloc[0], utc=True).weekday()
        print("First dow for init condit array: %d" %initial_dow)

        print("Preparing init condit...")
        df=config_df_by_dates(df,ic_start_date,ic_end_date,time_col="created_at")

        #get info needed to do do simulation
        n_simulation_loops,all_needed_prediction_dates,all_predicted_dates_with_extra_dates=calc_dates_for_simulation_altered(df,num_timesteps_in_a_sequence,"D")


        df=df.drop("created_at", axis=1)
        df=df.tail(num_timesteps_in_a_sequence)
        print(df)
        initial_condit=df.values
        print("Shape of initial condit: %s" %str(initial_condit.shape))

        #log normalize
        if LOG1_FLAG==True:
            initial_condit=np.log1p(initial_condit)
        if LOG2_FLAG==True:
            initial_condit=np.log1p(initial_condit)
            print("np log1p on initial_condit")
            print(initial_condit)

        # sys.exit(0)
        initial_condit=insert_dows_into_2d_array(initial_condit, initial_dow)
        initial_condit=initial_condit.reshape((1, initial_condit.shape[0], initial_condit.shape[1]))
        print("Initial condit shape: %s" %str(initial_condit.shape))

        if scaler_flag==True:
            #load scaler
            scaler_fp=input_dir+"All-Coin-Scalers/%s-scaler.json"%coin
            scaler=load_scaler(scaler_fp)
            initial_condit = normalize_single_array_with_scaler(initial_condit,scaler)
        else:
            scaler=""

        # #load model
        # model=load_model(model_fp)
        # model.compile(loss="mae", optimizer="adam")

        predictions=simulate_on_initial_condition_not_a_drill_with_DOW(initial_condit, model,scaler, n_simulation_loops,scaler_flag=scaler_flag)

        if LOG1_FLAG==True:
            predictions=np.expm1(predictions)
        if LOG2_FLAG==True:
            predictions=np.expm1(predictions)

        #reshape prediction vector
        new_dim0=predictions.shape[0] * predictions.shape[1]
        new_dim1=len(coin_fts)
        predictions=predictions.reshape((new_dim0,new_dim1))
        predictions=np.transpose(predictions)
        print("New predictions shape: %s" %str(predictions.shape))

        #convert negs to 0
        predictions[predictions<0]=0

        #round all values in pred array
        predictions=np.round(predictions, 0)

        #CREATE PREDICTION DF
        prediction_dict={}
        for i,ft in enumerate(coin_fts):
            prediction_dict[ft]=predictions[i]
        PRED_DF=pd.DataFrame(data=prediction_dict)

        #take the preds we want
        PRED_DF=PRED_DF.head(len(all_needed_prediction_dates))
        PRED_DF["created_at"]=all_needed_prediction_dates
        PRED_DF=PRED_DF[["created_at"]+coin_fts]
        PRED_DF=config_df_by_dates(PRED_DF,sim_start_date,sim_end_date,time_col="created_at")
        print(PRED_DF)

        coin_output_dir=VERSION_OUTPUT_DIR+coin+"/"
        create_output_dir(coin_output_dir)

        #save preds to output dir
        pred_output_fp=coin_output_dir+"%s-Predictions.csv"%coin
        PRED_DF.to_csv(pred_output_fp)

        #create graph dir
        graph_output_dir=coin_output_dir+"Graphs/"

        #graph results
        if GRAPH_WITH_GT==True:

            #save gt to output dir
            gt_output_fp=coin_output_dir+"%s-Ground-Truth.csv"%coin
            gtdf.to_csv(gt_output_fp)

            #graph feature with smape
            # graph_ground_truth_vs_pred_dfs_with_smape(gtdf,PRED_DF,GRANULARITY,graph_output_dir,smape_df)
            graph_ground_truth_vs_pred_dfs(gtdf,PRED_DF,GRANULARITY,graph_output_dir)
        else:
            for i,ft in enumerate(fts):
                # print("Graphing %s" %ft)
                pred_vector=PRED_DF[ft]

                graph_simulated_feature(pred_vector, graph_output_dir,ft,GRANULARITY)

def get_pseudo_predictions_for_gt_with_or_without_DOW(x_test, DOW_FLAG):
    #NOTE: THIS FUNCTION ASSUMES NO SCALING HAS BEEN DONE YET

    if DOW_FLAG==True:
        x_test = x_test[:,:,:-7]

    predictions_by_shifting = x_test

    return predictions_by_shifting



def get_pseudo_predictions_for_nogt(x_train, x_val, x_test,remove_dow=False):



    #get pseudo predictions for later
    predictions_by_shifting=np.concatenate([x_train,x_val, x_test[:1,:,:]], axis=0)

    if remove_dow == True:
        predictions_by_shifting=predictions_by_shifting[:,:,:-7]
    print("Shape of predictions_by_shifting so far: %s" %str(predictions_by_shifting.shape))

    #get num of pred samples we need
    num_needed_pred_samples=x_test.shape[0]

    #get delete amount
    head_delete_amount = predictions_by_shifting.shape[0] - num_needed_pred_samples

    #delete this amount
    predictions_by_shifting = predictions_by_shifting[head_delete_amount:,:,:]
    print("Final shape of predictions_by_shifting: %s" %str(predictions_by_shifting.shape))

    return predictions_by_shifting

def get_pseudo_predictions_for_nogt_for_train_and_test(x_train, x_test,remove_dow=False):



    #get pseudo predictions for later
    predictions_by_shifting=np.concatenate([x_train,x_test[:1,:,:]], axis=0)

    if remove_dow == True:
        predictions_by_shifting=predictions_by_shifting[:,:,:-7]
    print("Shape of predictions_by_shifting so far: %s" %str(predictions_by_shifting.shape))

    #get num of pred samples we need
    num_needed_pred_samples=x_test.shape[0]

    #get delete amount
    head_delete_amount = predictions_by_shifting.shape[0] - num_needed_pred_samples

    #delete this amount
    predictions_by_shifting = predictions_by_shifting[head_delete_amount:,:,:]
    print("Final shape of predictions_by_shifting: %s" %str(predictions_by_shifting.shape))

    return predictions_by_shifting


def get_source_and_target_value_arrays(ORIGINAL_DATA,X_ARRAY_TIMESTEPS_IN_A_SEQUENCE,Y_ARRAY_TIMESTEPS_IN_A_SEQUENCE,interval=1):

    # TARGET_DIFF_DATA=convert_to_differences(ORIGINAL_DATA, interval)
    TARGET_VALUE_DATA=ORIGINAL_DATA[1:,:]
    SOURCE_DATA=ORIGINAL_DATA[:-1,:]
    # print("Shape of new DIFF_ARRAY: %s" %str(DIFF_ARRAY.shape))
    # print("\nShape of old TARGET_DIFF_DATA: %s" %str(TARGET_DIFF_DATA.shape))
    print("Shape of old TARGET_VALUE_DATA: %s" %str(TARGET_VALUE_DATA.shape))
    print("Shape of old SOURCE_DATA: %s" %str(SOURCE_DATA.shape))

    # NEW_TARGET_DIFF_DATA=[]
    NEW_TARGET_VALUE_DATA=[]
    NEW_SOURCE_DATA=[]

    NUM_LOOPS=TARGET_VALUE_DATA.shape[0]

    x_start_idx=0
    x_end_idx=X_ARRAY_TIMESTEPS_IN_A_SEQUENCE

    y_start_idx=x_end_idx-1
    y_end_idx =y_start_idx+ Y_ARRAY_TIMESTEPS_IN_A_SEQUENCE

    for i in range(NUM_LOOPS):
        TEMP_X=SOURCE_DATA[x_start_idx:x_end_idx,:]
        NEW_SOURCE_DATA.append(TEMP_X)

        TEMP_Y_VALUES=np.asarray(TARGET_VALUE_DATA[y_start_idx:y_end_idx,:])
        # print("\nShape of TEMP_Y_VALUES: %s" %str(TEMP_Y_VALUES.shape))
        NEW_TARGET_VALUE_DATA.append(TEMP_Y_VALUES)

        # TEMP_Y_DELTAS=np.asarray(TARGET_DIFF_DATA[y_start_idx:y_end_idx,:])
        # NEW_TARGET_DIFF_DATA.append(TEMP_Y_DELTAS)

        x_start_idx +=X_ARRAY_TIMESTEPS_IN_A_SEQUENCE
        x_end_idx+=X_ARRAY_TIMESTEPS_IN_A_SEQUENCE

        y_start_idx=x_end_idx-1
        y_end_idx =y_start_idx+ Y_ARRAY_TIMESTEPS_IN_A_SEQUENCE

        if x_start_idx >=NUM_LOOPS:
            break
        if x_end_idx >=NUM_LOOPS:
            break
        if y_start_idx >=NUM_LOOPS:
            break
        if y_end_idx >=NUM_LOOPS:
            break

    # NEW_TARGET_DIFF_DATA=np.asarray(NEW_TARGET_DIFF_DATA)
    NEW_TARGET_VALUE_DATA=np.asarray(NEW_TARGET_VALUE_DATA)
    NEW_SOURCE_DATA=np.asarray(NEW_SOURCE_DATA)

    # print(NEW_SOURCE_DATA)
    # print(NEW_TARGET_VALUE_DATA)

    # print("\nShape of new TARGET_DIFF_DATA: %s" %str(NEW_TARGET_DIFF_DATA.shape))
    print("Shape of new TARGET_VALUE_DATA: %s" %str(NEW_TARGET_VALUE_DATA.shape))
    print("Shape of new SOURCE_DATA: %s" %str(NEW_SOURCE_DATA.shape))

    x_sample_num=NEW_SOURCE_DATA.shape[0]
    y_sample_num=NEW_TARGET_VALUE_DATA.shape[0]

    MIN_SAMPLES=min([x_sample_num,y_sample_num])

    # NEW_TARGET_DIFF_DATA=NEW_TARGET_DIFF_DATA[:MIN_SAMPLES,:].astype("float32")
    NEW_TARGET_VALUE_DATA=NEW_TARGET_VALUE_DATA[:MIN_SAMPLES,:].astype("float32")
    NEW_SOURCE_DATA=NEW_SOURCE_DATA[:MIN_SAMPLES,:].astype("float32")

    # print("\nShape of final TARGET_DIFF_DATA: %s" %str(NEW_TARGET_DIFF_DATA.shape))
    print("Shape of final TARGET_VALUE_DATA: %s" %str(NEW_TARGET_VALUE_DATA.shape))
    print("Shape of final SOURCE_DATA: %s" %str(NEW_SOURCE_DATA.shape))

    return  NEW_SOURCE_DATA,NEW_TARGET_VALUE_DATA

def get_x_and_y_arrays_for_binary_classifier(x_array, y_array,VAL_PERCENTAGE,TEST_PERCENTAGE):
    NUM_SAMPLES=x_array.shape[0]
    NUM_VAL_SAMPLES=int(NUM_SAMPLES * VAL_PERCENTAGE)
    NUM_TESTING_SAMPLES=int(NUM_SAMPLES *TEST_PERCENTAGE)
    NUM_TRAINING_SAMPLES= NUM_SAMPLES - NUM_VAL_SAMPLES - NUM_TESTING_SAMPLES

    if NUM_TRAINING_SAMPLES >=3 and NUM_TESTING_SAMPLES ==0 and NUM_VAL_SAMPLES==0:
        NUM_TRAINING_SAMPLES=NUM_TRAINING_SAMPLES -2
        NUM_VAL_SAMPLES=1
        NUM_TESTING_SAMPLES=1

    print("There are %d samples" %NUM_SAMPLES)
    print("%d samples will be used training" %NUM_TRAINING_SAMPLES)
    print("%d samples will be used for validation" %NUM_VAL_SAMPLES)
    print("%d samples will be used for testing" %NUM_TESTING_SAMPLES)

    x_train=x_array[:NUM_TRAINING_SAMPLES, :, :]
    y_train=y_array[:NUM_TRAINING_SAMPLES, :, :]


    x_val= x_array[NUM_TRAINING_SAMPLES:NUM_TRAINING_SAMPLES + NUM_VAL_SAMPLES, :,:]
    y_val= y_array[NUM_TRAINING_SAMPLES:NUM_TRAINING_SAMPLES + NUM_VAL_SAMPLES, :,:]



    x_test= x_array[NUM_TRAINING_SAMPLES + NUM_VAL_SAMPLES:,:,:]
    y_test= y_array[NUM_TRAINING_SAMPLES + NUM_VAL_SAMPLES:,:,:]


    print("x_train: shape:")
    print (x_train.shape)
    print("x_val shape:")
    print (x_val.shape)
    print("x_test shape:")
    print (x_test.shape)

    print("y_train: shape:")
    print (y_train.shape)
    print("y_val shape:")
    print (y_val.shape)
    print("y_test shape:")
    print (y_test.shape)

    return x_train, x_val, x_test, y_train, y_val, y_test

def properly_convert_arrays_to_sliding_window_format_for_sep_x_and_y_without_delta_arrays(x,y, window_size=1):
    print("Converting arrays to sliding window format...")

    #get timesteps for later
    xtimesteps_in_a_sequence=x.shape[1]
    ytimesteps_in_a_sequence=y.shape[1]

    #get total timesteps in seq
    xy_timesteps_in_a_sequence=xtimesteps_in_a_sequence + ytimesteps_in_a_sequence
    print("X timesteps in a sequence: %d" %xtimesteps_in_a_sequence)
    print("y timesteps in a sequence: %d" %ytimesteps_in_a_sequence)
    print("xy timesteps in a sequence: %d" %xy_timesteps_in_a_sequence)

    #combine x and y for now
    x_and_y_values=np.concatenate([x, y], axis=1)

    print("Shape of x_and_y_values: %s" %str(x_and_y_values.shape))


    # #new shape
    new_xy_2d_shape=(x_and_y_values.shape[0]*x_and_y_values.shape[1],x_and_y_values.shape[2])
    # new_y_2d_shape=(y.shape[0]*y.shape[1],y.shape[2])

    # #reshape arrays
    x_and_y_values=x_and_y_values.reshape(new_xy_2d_shape)


    #these will be our new arrays
    new_x=[]
    new_y=[]


    num_iterations=x_and_y_values.shape[0]-xy_timesteps_in_a_sequence+window_size
    print("The formula for the number of new array elements for the sliding window configuration:")
    print("(Number of array elements) - (Number of timesteps in a sequence) + WINDOW SIZE")


    #set number of iterations
    # print("Since we have different array sizes for x and y we have to use the array with the lower number of elements to determine the iterations (in this case y).")
    print("In this case: ")
    print("%d - %d + %d = %d" %(x_and_y_values.shape[0], xy_timesteps_in_a_sequence, window_size, num_iterations))

    #make sliding window data!
    for i in range(0,num_iterations,window_size):
        cur_x_slice=x_and_y_values[i:i+xtimesteps_in_a_sequence,:]
        cur_y_value_slice=x_and_y_values[i+xtimesteps_in_a_sequence:i+xtimesteps_in_a_sequence+ytimesteps_in_a_sequence,:]


        #append
        new_x.append(cur_x_slice)
        new_y.append(cur_y_value_slice)


    # # #convert these to arrays
    new_x=np.asarray(new_x)
    new_y=np.asarray(new_y)


    print("New x array shape: %s" %str(new_x.shape))
    print("new_y shape: %s" %str( new_y.shape))


    return new_x,new_y

def convert_target_arrays_to_binary_form(x,y):
    MOD_NUM=1000

    #our i and j
    num_samples=x.shape[0]
    num_features_in_a_timestep=x.shape[2]

    #binary target array
    bt_list=[]
    for i in range(num_samples):
        bt_list.append([])

    #loop
    averaged_y_fts=np.average(y, axis=1)
    print("Shape of averaged_y_fts: %s" %str(averaged_y_fts.shape))
    # sys.exit(0)

    for i in range(num_samples):
        for j in range(num_features_in_a_timestep):

            #get last fts to compare
            last_x_ft=x[i][-1][j]


            # last_y_ft=y[i][-1][j]
            # last_y_ft = np.average(y[i], axis=2)[j]
            last_y_ft = averaged_y_fts[i][j]

            if last_x_ft >= last_y_ft:
                target_class=0
                # bt_list[i].append(0)
            else:
                target_class=1


            if i%MOD_NUM==0:
                print("The last x ft is %.4f" %last_x_ft)
                print("The last y ft is %.4f" %last_y_ft)
                print("The target class is %d" %target_class)

            bt_list[i].append(target_class)

        bt_list[i]=np.asarray(bt_list[i])

    bt_array=np.asarray(bt_list)
    print("Shape of bt_array: %s" %str(bt_array.shape))

    return bt_array

# def debug_print(str_to_print,LIMIT_TEST=0,DEBUG=False):
# 	LIMIT=10
# 	if (LIMIT_TEST < LIMIT) and DEBUG==True:
# 	    print(str_to_print)

def get_class_breakdown(y,y_name,num_classes=2,print_flag=True):
    y=y.copy().flatten()

    class_counts=[0 for i in range(num_classes)]

    total_samples=y.shape[0]

    for target in y:
        class_counts[target] += 1

    class_percentages=[]

    for cc in class_counts:
        percentage = cc/(1.0 * total_samples)
        class_percentages.append(percentage)

    if print_flag == True:
    	print("\n###### Class frequencies for %s ##########\n" %y_name)
    for i in range(num_classes):
    	if print_flag == True:
        	print("Class %d count: %d" %(i,class_counts[i]))
    if print_flag == True:
    	print("\n")


    for i in range(num_classes):
    	if print_flag == True:
        	print("Class %d freq: %.2f%%" %(i,class_percentages[i] * 100.0))

    return class_counts,class_percentages

def separate_classes(x,y,num_classes=2):
	samples_separated_by_class=[[] for i in range(num_classes)]
	labels_separated_by_class=[[] for i in range(num_classes)]

	for i in range(y.shape[0]):
		label=int(y[i])
		labels_separated_by_class[label].append(label)

		sample=x[i]
		samples_separated_by_class[label].append(sample)

	for i in range(num_classes):
		samples_separated_by_class[i]= np.asarray(samples_separated_by_class[i])

	samples_separated_by_class = np.asarray(samples_separated_by_class)

	print("Shape of samples separated by class: %s" %str(samples_separated_by_class.shape))

	return samples_separated_by_class


def unison_shuffle_x_and_y(x, y):
    assert x.shape[0] == y.shape[0]
    p = np.random.permutation(x.shape[0])
    return x[p], y[p]




def balance_binary_data(x,y,y_name):

    class_counts,class_percentages = get_class_breakdown(y,y_name)

    amount_to_remove = abs(class_counts[0] - class_counts[1])
    print("Amount to remove: %d" %amount_to_remove)

    amount_to_random_sample = min([class_counts[0], class_counts[1]])
    print("Amount to random_sample: %d" %amount_to_random_sample)

    #you have to separate the classes
    samples_separated_by_class = separate_classes(x,y)

    #put the 2d arrays in a list
    #we need to subsample
    balanced_samples_list=[]
    for class_2d_array in samples_separated_by_class:

    	class_2d_array = class_2d_array[np.random.choice(class_2d_array.shape[0], amount_to_random_sample)]

    	balanced_samples_list.append(class_2d_array)

    balance_samples = np.asarray(balanced_samples_list)
    print("Shape of our balanced samples: %s" %str(balance_samples.shape))

    #make labels
    y=[]
    for i in range(len(balanced_samples_list)):
    	for j in range(balanced_samples_list[i].shape[0]):
    		y.append(i)
    y=np.asarray(y)

    #make new x
    x = np.concatenate(balanced_samples_list)

    #unison shuffle
    x,y = unison_shuffle_x_and_y(x, y)
    print("Shape of balanced x after unison shuffle: %s" %str(x.shape))
    print("Shape of balanced y after unison shuffle: %s" %str(y.shape))

    return x,y






def get_all_x_and_y_arrays_for_binary_price_predictor(SOURCE_DATA,TARGET_VALUE_DATA, FEATURES_OF_INTEREST,binary_feature_of_interest,VAL_PERCENTAGE,TEST_PERCENTAGE,SLIDING_WINDOW=True,WINDOW_SIZE=1,TEST_SLIDING_WINDOW=False,TEST_WINDOW_SIZE=1):
    print("\n################ Getting Binary Classifier arrays ########################")

    #get index of feature we want to predict
    bc_target_ft_idx=FEATURES_OF_INTEREST.index(binary_feature_of_interest)
    print("\n%s is at index %d" %(binary_feature_of_interest,bc_target_ft_idx))

    #now slice arrays to get just this feature
    x_array = SOURCE_DATA
    y_array = TARGET_VALUE_DATA
    # y_array = TARGET_VALUE_DATA[:,:,bc_target_ft_idx:bc_target_ft_idx + 1]
    print("Shape of x_array so far: %s" %str(x_array.shape))
    print("Shape of y_array so far: %s" %str(y_array.shape))

    #get proper train,val, and test arrays
    x_train, x_val, x_test, y_train, y_val, y_test = get_x_and_y_arrays_for_binary_classifier(x_array, y_array,VAL_PERCENTAGE,TEST_PERCENTAGE)

    #Now get these arrays in sliding window form
    if SLIDING_WINDOW==True:
        x_train, y_train = properly_convert_arrays_to_sliding_window_format_for_sep_x_and_y_without_delta_arrays(x_train,y_train,WINDOW_SIZE)
        x_val, y_val = properly_convert_arrays_to_sliding_window_format_for_sep_x_and_y_without_delta_arrays(x_val,y_val,WINDOW_SIZE)

    if TEST_SLIDING_WINDOW == True:
        x_test, y_test = properly_convert_arrays_to_sliding_window_format_for_sep_x_and_y_without_delta_arrays(x_test,y_test,TEST_WINDOW_SIZE)

    print("\n###### Array shapes so far: #####")
    print("Shape of x_train so far: %s" %str(x_train.shape))
    print("Shape of y_train so far: %s" %str(y_train.shape))
    print("Shape of x_val so far: %s" %str(x_val.shape))
    print("Shape of y_val so far: %s" %str(y_val.shape))
    print("Shape of x_test so far: %s" %str(x_test.shape))
    print("Shape of y_test so far: %s" %str(y_test.shape))
    print("#################################\n")

    print("Creating binary arrays...")
    y_train = convert_target_arrays_to_binary_form(x_train,y_train)
    y_val = convert_target_arrays_to_binary_form(x_val,y_val)
    y_test = convert_target_arrays_to_binary_form(x_test,y_test)

    #slice by relevant target index
    y_train = y_train[:,bc_target_ft_idx:bc_target_ft_idx + 1]
    y_val = y_val[:,bc_target_ft_idx:bc_target_ft_idx + 1]
    y_test = y_test[:,bc_target_ft_idx:bc_target_ft_idx + 1]

    #we have to reshape the x arrays as well
    #goodbye lstm... hello fcnn
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
    x_val = x_val.reshape((x_val.shape[0], x_val.shape[1] * x_val.shape[2]))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1] * x_test.shape[2]))

    print("\n###### Final BC array shapes: #####")
    print("Shape of x_train: %s" %str(x_train.shape))
    print("Shape of y_train: %s" %str(y_train.shape))
    print("Shape of x_val: %s" %str(x_val.shape))
    print("Shape of y_val: %s" %str(y_val.shape))
    print("Shape of x_test: %s" %str(x_test.shape))
    print("Shape of y_test: %s" %str(y_test.shape))
    print("#################################\n")

    get_class_breakdown(y_train, "y_train")
    get_class_breakdown(y_val, "y_val")
    get_class_breakdown(y_test, "y_test")


    return x_train, x_val, x_test, y_train, y_val, y_test


def save_scaling_info(SCALE_INFO_DIR,LOG1_FLAG,LOG2_FLAG,scaler_flag):
    create_output_dir(SCALE_INFO_DIR)
    if scaler_flag==True:
        f=open(SCALE_INFO_DIR+"Scaler.txt", "w")
        f.close()
    if LOG1_FLAG==True:
        f=open(SCALE_INFO_DIR+"Log1.txt", "w")
        f.close()
    if LOG2_FLAG==True:
        f=open(SCALE_INFO_DIR+"Log2.txt", "w")
        f.close()






def get_feature_df_from_many_params(coin,all_dirs,desired_features,desired_feature_dir_dict,
        twitter_gen_fts, telegram_gen_fts, start_date, end_date,GRANULARITY, simple_price_fts,
        nazim_twitter_fts, nazim_price_fts, newest_nazim_fts, nazim_telegram_fts):
    #get twitter fts (pnnl)
    tagged_twitter_fts=configure_coin_fts(coin, twitter_gen_fts)

    #get telegram fts (pnnl)
    tagged_telegram_fts=configure_coin_fts(coin, telegram_gen_fts)

    #NOTE: We have nazim fts already
    #and simple price fts
    twitter_pnnl_dir=desired_feature_dir_dict["fred_pnnl_twitter_features"]
    telegram_pnnl_dir = desired_feature_dir_dict["fred_pnnl_telegram_features"]
    price_dir = desired_feature_dir_dict["simple_price_features"]
    nazim_dir = desired_feature_dir_dict["nazim_features"]
    new_nazim_dir = desired_feature_dir_dict["new_nazim_features"]

    for cur_dir in all_dirs:
        if cur_dir == twitter_pnnl_dir:
            twitter_pnnl_fp=twitter_pnnl_dir + "%s/%s-features.csv"%(coin,coin)

        if cur_dir == telegram_pnnl_dir:
            telegram_pnnl_fp=telegram_pnnl_dir + "%s/%s-features.csv"%(coin,coin)

        if cur_dir == price_dir:
            price_fp=price_dir+"%s.csv"%coin

        if cur_dir == nazim_dir:
            nazim_price_fp=nazim_dir+"%s/Price-%s-Nazim-Features-%s.csv"%(coin,coin,GRANULARITY)
            nazim_telegram_fp=nazim_dir+"%s/Telegram-%s-Nazim-Features-%s.csv"%(coin,coin,GRANULARITY)
            nazim_twitter_fp=nazim_dir + "%s/Twitter-%s-Nazim-Features-%s.csv"%(coin,coin,GRANULARITY)

        if cur_dir ==  new_nazim_dir:
            new_nazim_fp=new_nazim_dir + "%s.csv"%coin

    #setup desired features
    desired_feature_dict={
    "fred_pnnl_twitter_features":twitter_pnnl_fp,
    "fred_pnnl_telegram_features": telegram_pnnl_fp,
    "simple_price_features":price_fp,
    "nazim_price_features":nazim_price_fp,
    "nazim_telegram_features":nazim_telegram_fp,
    "nazim_twitter_features":nazim_twitter_fp,
    "new_nazim_features":new_nazim_fp
    }


    all_fps=[]
    print("\nFeatures we'll be using")
    for des_ft in desired_features:
        print(des_ft)
        fp=desired_feature_dict[des_ft]
        all_fps.append(fp)

    print("List of fps")
    for fp in all_fps:
        print(fp)


    #dict for fp tags
    fp_tag_dict={
    twitter_pnnl_fp: "",
    telegram_pnnl_fp: "",
    price_fp: "_simple_price_ft",
    nazim_price_fp: "_nazim_price_ft",
    nazim_telegram_fp: "_nazim_telegram_ft",
    nazim_twitter_fp: "_nazim_twitter_ft",
    new_nazim_fp: "_nazim_new_ft"
    }

    #dict for desired ft list
    fp_ft_list_dict={
    twitter_pnnl_fp: tagged_twitter_fts,
    telegram_pnnl_fp: tagged_telegram_fts,
    price_fp: simple_price_fts,
    nazim_price_fp: nazim_price_fts,
    nazim_telegram_fp: nazim_telegram_fts,
    nazim_twitter_fp: nazim_twitter_fts,
    new_nazim_fp: newest_nazim_fts

    }

    #this established the # of features each sample should have
    ALL_POTENTIAL_FTS=[]
    for fp in all_fps:
        ft_list=fp_ft_list_dict[fp]
        ALL_POTENTIAL_FTS+=ft_list
    NUM_FTS_TO_ENFORCE=len(ALL_POTENTIAL_FTS)

    #make feature df from list
    FEATURE_DF,start_date,end_date=get_feature_df_from_fp_list_and_fix_column_names(all_fps,fp_tag_dict,fp_ft_list_dict, start_date,end_date,keep_time_col=True)

    return FEATURE_DF,start_date,end_date,NUM_FTS_TO_ENFORCE

# def debug_print(str_to_print,LIMIT_TEST=0,DEBUG=False):
# 	LIMIT=10
# 	if (LIMIT_TEST < LIMIT) and DEBUG==True:
# 	    print(str_to_print)

def get_accuracy_by_class(y_test, predictions,num_classes=2):
	if len(y_test.shape) == 2:
		y_test = np.argmax(y_test, axis=1)
		print("Shape of y_test: %s" %str(y_test.shape))

	if len(predictions.shape) == 2:
		predictions = np.argmax(predictions, axis=1)
		print("Shape of predictions: %s" %str(predictions.shape))

	#get class counts
	#we'll use these for calculating accuracy
	class_counts,_ = get_class_breakdown(y_test,y_name="",num_classes=2,print_flag=False)

	#get correct count
	correct_predictions=[0 for i in range(num_classes)]

	#flatten y and preds if not already
	y_test= y_test.flatten().astype("int32")
	predictions=predictions.flatten().astype("int32")

	for i in range(y_test.shape[0]):
		if y_test[i] == predictions[i]:
			correct_predictions[y_test[i]]+=1

	#get accuracies
	accs=[]
	for i in range(num_classes):
		try:
			acc = 100.0 * correct_predictions[i] /(1.0 * class_counts[i])
		except ZeroDivisionError:
			acc = 100
		accs.append(acc)

	for i,acc in enumerate(accs):
		print("Class %d acc: %.2f%%" %(i,acc))

	return accs


def insert_7_dow_cols(df, GRANULARITY,start_date_offset=0):
    if GRANULARITY != "D":
        print("This function only works with daily data! Terminating!")
        sys.exit(0)

    start_date = df["created_at"].iloc[start_date_offset]
    end_date = df["created_at"].iloc[-1]
    df=config_df_by_dates(df,start_date,end_date)
    start_dow=start_date.weekday()
    initial_dow=start_dow
    print("Starting DOW: %s" %str(start_dow))

    num_col_elements=df.shape[0]

    dow_cols=[[0 for j in range(num_col_elements)] for i in range(7)]

    dow_vals=[]

    for i in range(num_col_elements):

        if GRANULARITY == "D":
            cur_dow = (start_dow+ i)%7

        print("Day %d, dow: %d" %(i,cur_dow))
        dow_cols[cur_dow][i]=1
        dow_vals.append(cur_dow)
        # sys.exit(0)

    df["isMonday"]=dow_cols[0]
    df["isTuesday"]=dow_cols[1]
    df["isWednesday"]=dow_cols[2]
    df["isThursday"]=dow_cols[3]
    df["isFriday"]=dow_cols[4]
    df["isSaturday"]=dow_cols[5]
    df["isSunday"]=dow_cols[6]

    df["DOW"]=dow_vals


    return df


def get_proper_number_of_timestamps(PND_DF,GRANULARITY):
    PND_DF["created_at"] = pd.to_datetime(PND_DF["created_at"], utc=True)
    PND_DF=PND_DF.sort_values(by=["created_at"])
    # PND_DF["created_at"]=PND_DF["created_at"].dt.floor(GRANULARITY)

    if GRANULARITY != "W" and GRANULARITY != "W-MON":
        PND_DF["created_at"]=PND_DF["created_at"].dt.floor(GRANULARITY)
    else:
        print("Adjusting PND_DF by week granularity")
        PND_DF["created_at"]=PND_DF["created_at"] - pd.to_timedelta(7, unit="d")
        PND_DF["created_at"]=PND_DF["created_at"].dt.floor("D")

    start_date=PND_DF["created_at"].iloc[0]
    end_date=PND_DF["created_at"].iloc[-1]

    ALL_DATES=[]
    if GRANULARITY=="7D":
        for cur_date in datespan(start_date, end_date,delta=timedelta(days=7)):
           ALL_DATES.append(cur_date)
    if GRANULARITY=="D":
        for cur_date in datespan(start_date, end_date,delta=timedelta(days=1)):
           ALL_DATES.append(cur_date)
    elif GRANULARITY=="H":
        for cur_date in datespan(start_date, end_date,delta=timedelta(hours=1)):
           ALL_DATES.append(cur_date)
    elif GRANULARITY=="min":
        for cur_date in datespan(start_date, end_date,delta=timedelta(minutes=1)):
           ALL_DATES.append(cur_date)
    elif GRANULARITY=="S":
        for cur_date in datespan(start_date, end_date,delta=timedelta(seconds=1)):
           ALL_DATES.append(cur_date)
    elif GRANULARITY=="W" or GRANULARITY=="W-MON":
        print(start_date)
        print(end_date)
        for cur_date in datespan(start_date, end_date,delta=timedelta(weeks=1)):
            ALL_DATES.append(cur_date)

    return len(ALL_DATES)

def verify_df_size(AGG_DF,PROPER_NUM_OF_TIMESTAMPS):
    if AGG_DF.shape[0] != PROPER_NUM_OF_TIMESTAMPS:
        print("AGG_DF.shape[0]: %d, PROPER_NUM_OF_TIMESTAMPS: %d"%(AGG_DF.shape[0],PROPER_NUM_OF_TIMESTAMPS))
        print("Error! AGG_DF.shape[0] != PROPER_NUM_OF_TIMESTAMPS")

        sys.exit(0)

def verify_df_size_by_time(AGG_DF,GRANULARITY):

    if AGG_DF.shape[0]==0:
        print("Df is empty.")
        return

    PROPER_NUM_OF_TIMESTAMPS=get_proper_number_of_timestamps(AGG_DF,GRANULARITY)
    print("Verifying that df size is accurate...There should be %d rows in the df" %PROPER_NUM_OF_TIMESTAMPS )
    if AGG_DF.shape[0] != PROPER_NUM_OF_TIMESTAMPS:
        print("AGG_DF.shape[0]: %d, PROPER_NUM_OF_TIMESTAMPS: %d"%(AGG_DF.shape[0],PROPER_NUM_OF_TIMESTAMPS))
        print("Error! AGG_DF.shape[0] != PROPER_NUM_OF_TIMESTAMPS")

        sys.exit(0)
    print("Verification was successful")

def calculate_train_and_test_dates(dates,num_training_dates, num_test_dates,main_output_dir):
    print("Calculating dates...")
    total_dates = num_training_dates + num_test_dates

    dates=dates[:total_dates]
    training_dates = dates[:num_training_dates]

    test_dates = dates[num_training_dates:num_training_dates + num_test_dates]

    if (len(test_dates) + len(training_dates)) != total_dates:
        print("Error! (len(test_dates) + len(training_dates)) != total_dates")
        sys.exit(0)

    #record this info
    date_info_fp=main_output_dir + "Date-Info.txt"
    f=open(date_info_fp, "w")

    screen_and_file_print("############ Date Info ###########", f)
    screen_and_file_print("Num training dates: %s" %str(num_training_dates), f)
    screen_and_file_print("Num test dates: %s" %str(num_test_dates), f)

    screen_and_file_print("\nTraining start: %s" %str(training_dates[0]), f)
    screen_and_file_print("Training end: %s" %str(training_dates[-1]), f)
    screen_and_file_print("Test start: %s" %str(test_dates[0]), f)
    screen_and_file_print("Test end: %s" %str(test_dates[-1]), f)

    return training_dates,test_dates

def calculate_train_val_and_test_dates(dates,num_training_dates, num_val_dates,num_test_dates,main_output_dir,array_name=""):
    # if array_name != "X" or array_name != "Y":
    #     print("Choose 'X' or 'Y' for array name")
    #     sys.exit(0)

    # if array_name == "Y":
    #     dates=

    print("Calculating dates...")
    total_dates = num_training_dates + num_val_dates + num_test_dates

    dates=dates[:total_dates]
    training_dates = dates[:num_training_dates]

    val_dates = dates[num_training_dates:num_training_dates + num_val_dates]

    test_dates = dates[num_training_dates+ num_val_dates:]

    if (len(test_dates) + len(training_dates)+ len(val_dates)) != total_dates:
        print("Error! (len(test_dates) + len(training_dates)+ len(val_dates)) != total_dates")
        sys.exit(0)

    #record this info
    date_info_fp=main_output_dir + "Date-Info.txt"
    f=open(date_info_fp, "w")

    screen_and_file_print("\n############ Date Info ###########",f)
    screen_and_file_print("Num training dates: %s" %str(num_training_dates), f)
    screen_and_file_print("Num val dates: %s" %str(num_val_dates), f)
    screen_and_file_print("Num test dates: %s" %str(num_test_dates), f)

    screen_and_file_print("\nTraining start: %s" %str(training_dates[0]), f)
    screen_and_file_print("Training end: %s" %str(training_dates[-1]), f)

    screen_and_file_print("\nVal start: %s" %str(val_dates[0]), f)
    screen_and_file_print("Val end: %s" %str(val_dates[-1]), f)

    screen_and_file_print("\nTest start: %s" %str(test_dates[0]), f)
    screen_and_file_print("Test end: %s" %str(test_dates[-1]), f)

    return training_dates, val_dates,test_dates



def get_ts_features(input_dir,main_output_dir,start_date,end_date, target_cols, X_ARRAY_TIMESTEPS_IN_A_SEQUENCE,Y_ARRAY_TIMESTEPS_IN_A_SEQUENCE,VAL_PERCENTAGE,
    TEST_PERCENTAGE,kickout_cols=[],dow_cols=[],SLIDING_WINDOW_AMOUNT=7,DEBUG=False,FT_LIMIT_FLAG=False,FT_LIMIT = 100, keep_cols=[]):



    #get all infps
    infps=os.listdir(input_dir)

    #make sure only csv files are here
    for infp in infps:
        if ".csv" not in infp:
            infps.remove(infp)

    # if DEBUG==True:
    #   infps=infps[:1]
    print(infps)

    #save arrays here
    x_train_arrays=[]
    y_train_as_values_arrays=[]
    y_train_as_deltas_arrays=[]

    x_test_arrays=[]
    y_test_as_values_arrays=[]
    y_test_as_deltas_arrays=[]

    #test data dict for each coin
    test_data_dict={}

    #get the starting day of the week
    start_date_obj=pd.to_datetime(start_date, utc=True)
    start_dow=start_date_obj.weekday()
    initial_dow=start_dow
    print("Starting DOW: %s" %str(start_dow))


    #now open each infp as df
    #then retrieve the arrays
    for infp in infps:
        df=pd.read_csv(input_dir + infp)

        #get first dow
        first_dow_from_df = int(df["DOW"].iloc[0])

        #config df by dates
        df = config_df_by_dates(df,start_date,end_date)

        #save dow df for later
        dow_df=df[["created_at"] + dow_cols]

        #get dates
        dates=list(df["created_at"])

        #kickout cols
        df=df.drop(kickout_cols, axis=1)

        #move dow cols to end
        df_cols = list(df)
        df_cols = remove_unnamed_col(df_cols)

        # if TWEET_FT_FILTER == True:
        #     # from ts_tweet_fts import tweet_fts
        #     df_cols = tweet_fts

        if FT_LIMIT_FLAG == True:
            df_cols = df_cols[:FT_LIMIT] + target_cols + keep_cols + dow_cols
            df_cols = list(set(df_cols))

        df = df[df_cols]

        df =df[[c for c in df if c not in dow_cols] + dow_cols]
        # df =df.drop(dow_cols, axis=1)

        #get feature array
        FEATURE_ARRAY= np.asarray(df).astype("float32")
        print("Shape of feature array: %s" %str(FEATURE_ARRAY.shape))

        #get source and target arrays for later
        SOURCE_DATA,TARGET_VALUE_DATA,TARGET_DIFF_DATA = get_target_value_and_difference_arrays(FEATURE_ARRAY,X_ARRAY_TIMESTEPS_IN_A_SEQUENCE,Y_ARRAY_TIMESTEPS_IN_A_SEQUENCE)

        #get x and y arrays
        x_train, x_test, y_train_as_values,y_train_as_deltas, y_test_as_values,y_test_as_deltas = get_x_and_y_arrays_for_2d_input(SOURCE_DATA,TARGET_VALUE_DATA,
            TARGET_DIFF_DATA,VAL_PERCENTAGE,TEST_PERCENTAGE)

        #needed for later
        num_orig_x_train_samples = x_train.shape[0]
        num_orig_x_test_samples = x_test.shape[0]



        #convert arrays to sliding window format
        x_train,y_train_as_values,y_train_as_deltas = properly_convert_arrays_to_sliding_window_format_for_sep_x_and_y(x_train,y_train_as_values,y_train_as_deltas, window_size=SLIDING_WINDOW_AMOUNT)

        #save arrays
        x_train_arrays.append(x_train.copy())
        y_train_as_values_arrays.append(y_train_as_values.copy())
        y_train_as_deltas_arrays.append(y_train_as_deltas.copy())

        print("Initial dow: %d, first dow from df: %d" %(initial_dow, first_dow_from_df))

        if (int(initial_dow) == int(first_dow_from_df)):
            x_test_arrays.append(x_test.copy())
            y_test_as_values_arrays.append(y_test_as_values.copy())
            y_test_as_deltas_arrays.append(y_test_as_deltas.copy())

        # #save coin test data later
        # test_data_dict[coin] = (x_test.copy(), y_test_as_values.copy(), y_test_as_deltas.copy())

    #now calulcate dates
    num_training_dates = num_orig_x_train_samples * X_ARRAY_TIMESTEPS_IN_A_SEQUENCE
    num_test_dates = num_orig_x_test_samples * X_ARRAY_TIMESTEPS_IN_A_SEQUENCE
    training_dates,test_dates = calculate_train_and_test_dates(dates,num_training_dates, num_test_dates,main_output_dir)



    #get features of interest
    FEATURES_OF_INTEREST = list(df)

    #get indices of target cols
    target_col_indices=[]
    print("\nGetting target col indices...")
    for tc in target_cols:
        idx = FEATURES_OF_INTEREST.index(tc)
        target_col_indices.append(idx)
        print("%s is at index %d" %(tc,idx))

    #now concat these arrays along 0 axis
    x_train = np.concatenate(x_train_arrays, axis=0)
    y_train_as_values = np.concatenate(y_train_as_values_arrays, axis=0)
    y_train_as_deltas = np.concatenate(y_train_as_deltas_arrays, axis=0)

    x_test = np.concatenate(x_test_arrays, axis=0)
    y_test_as_values = np.concatenate(y_test_as_values_arrays, axis=0)
    y_test_as_deltas = np.concatenate(y_test_as_deltas_arrays, axis=0)

    #Now filter y arrays to only target target cols
    y_train_as_values = y_train_as_values[:,:,target_col_indices]
    y_train_as_deltas = y_train_as_deltas[:,:,target_col_indices]
    y_test_as_values = y_test_as_values[:,:,target_col_indices]
    y_test_as_deltas = y_test_as_deltas[:,:,target_col_indices]

    print("\nx_train shape: %s" %str(x_train.shape))
    print("y_train_as_values shape: %s" %str(y_train_as_values.shape))
    print("y_train_as_deltas shape: %s" %str(y_train_as_deltas.shape))

    print("\nx_test shape: %s" %str(x_test.shape))
    print("y_test_as_values shape: %s" %str(y_test_as_values.shape))
    print("y_test_as_deltas shape: %s" %str(y_test_as_deltas.shape))

    return x_train, y_train_as_values,y_train_as_deltas, x_test, y_test_as_values,y_test_as_deltas,FEATURES_OF_INTEREST

def get_aggregated_daily_features(pseudo_pred_dict,LOG1_FLAG, LOG2_FLAG, LOG_ADD_COEFF,AGG_AMOUNT,fts_to_predict_dict,coin,dir_dict,merge_cols,input_dir_list,main_output_dir,start_date,end_date, target_cols, X_ARRAY_TIMESTEPS_IN_A_SEQUENCE,Y_ARRAY_TIMESTEPS_IN_A_SEQUENCE,VAL_PERCENTAGE,
    TEST_PERCENTAGE,kickout_cols=[],dow_cols=[],SLIDING_WINDOW_AMOUNT=1,DEBUG=False,FT_LIMIT_FLAG=False,FT_LIMIT = 100, keep_cols=[], timecol="created_at"):


    #get dir types
    platforms_of_interest = dir_dict.keys()

    #get desired features
    fts_to_predict=[]
    for platform in platforms_of_interest:
        fts_to_predict+=fts_to_predict_dict[platform]

    #save arrays here
    x_train_arrays=[]
    y_train_as_values_arrays=[]
    y_train_as_deltas_arrays=[]

    x_val_arrays=[]
    y_val_as_values_arrays=[]
    y_val_as_deltas_arrays=[]

    x_test_arrays=[]
    y_test_as_values_arrays=[]
    y_test_as_deltas_arrays=[]

    #test data dict for each coin
    test_data_dict={}

    #get the starting day of the week
    start_date_obj=pd.to_datetime(start_date, utc=True)
    start_dow=start_date_obj.weekday()
    initial_dow=start_dow
    print("Starting DOW: %s" %str(start_dow))

    # dates=pd.date_range()
    dates=list(pd.date_range(start_date, end_date, freq="%dD"%AGG_AMOUNT))


    #now open each infp as df
    #then retrieve the arrays
    # for infp in infps:

    # pseudo_pred_dict = {}

    for cur_dow in range(7):

        df_list = []

        for platform in platforms_of_interest:
            cur_dir = dir_dict[platform]
            cur_fp = cur_dir  + coin +"-Start-DOW-%d.csv"%cur_dow
            df = pd.read_csv(cur_fp)
            df_list.append(df)

        df = reduce(lambda left,right: pd.merge(left,right,on=merge_cols,how="inner"), df_list)

        # #resample cols
        # fts_to_agg = list(df)
        # non_agg_cols = kickout_cols + dow_cols + [timecol]
        # for ft in fts_to_agg:
        #     if ft in non_agg_cols:
        #         fts_to_agg.remove(ft)

        #set num rows
        num_rows = df.shape[0]

        #agg fts by agg amount
        agg_groups=[]

        agg_group = 0
        for i in range(num_rows):

            if i%AGG_AMOUNT== 0 and i != 0:
                agg_group+=1

            agg_groups.append(agg_group)

        df["agg_group"] = agg_groups
        print(df)
        # sys.exit(0)
        for ft in fts_to_predict:
            print("Adding %s" %ft)
            df[ft] = df.groupby("agg_group")[ft].transform("sum")
        # df=df.drop("agg_group", axis=1)

        subset = fts_to_predict + ["agg_group"]
        df=df.drop_duplicates(subset=subset,keep="first").reset_index(drop=True)
        df=df.drop("agg_group", axis=1)
        print(df)


        # #get dates
        # dates=list(df["created_at"])



        #get first dow
        first_dow_from_df = cur_dow

        #config df by dates
        df = config_df_by_dates(df,start_date,end_date)

        #save dow df for later
        dow_df=df[["created_at"] + dow_cols].copy()

        created_at_series = df["created_at"].copy()

        #get ft cols only
        FEATURES_ONLY_DF = df[fts_to_predict].copy()

        #log normalize
        if LOG1_FLAG==True:
            FEATURES_ONLY_DF=np.log1p(FEATURES_ONLY_DF + LOG_ADD_COEFF)
        if LOG2_FLAG==True:
            FEATURES_ONLY_DF=np.log1p(FEATURES_ONLY_DF + LOG_ADD_COEFF)
        print("np log1p on df")
        print(FEATURES_ONLY_DF)

        FEATURES_ONLY_DF["created_at"] = created_at_series
        df = FEATURES_ONLY_DF
        df = pd.merge(df, dow_df, on="created_at", how="inner")
        df_cols = fts_to_predict + dow_cols
        df=df[df_cols]
        print(df)
        # sys.exit(0)





        # #kickout cols
        # df=df.drop(kickout_cols, axis=1)

        # #move dow cols to end
        # df_cols = list(df)
        # df_cols = remove_unnamed_col(df_cols)

        # df = df[df_cols]

        # df =df[[c for c in df if c not in dow_cols] + dow_cols]
        # print(df)
        # sys.exit(0)
        # df =df.drop(dow_cols, axis=1)

        #get feature array
        FEATURE_ARRAY= np.asarray(df).astype("float32")
        print("Shape of feature array: %s" %str(FEATURE_ARRAY.shape))

        #get source and target arrays for later
        SOURCE_DATA,TARGET_VALUE_DATA,TARGET_DIFF_DATA = get_target_value_and_difference_arrays(FEATURE_ARRAY,X_ARRAY_TIMESTEPS_IN_A_SEQUENCE,Y_ARRAY_TIMESTEPS_IN_A_SEQUENCE)

        # #get x and y arrays
        # x_train, x_val, x_test, y_train_as_values,y_train_as_deltas, y_val_as_values, y_val_as_deltas, y_test_as_values,y_test_as_deltas = get_x_and_y_arrays_for_2d_input(SOURCE_DATA,TARGET_VALUE_DATA,
        #     TARGET_DIFF_DATA,VAL_PERCENTAGE,TEST_PERCENTAGE)

       #get x and y arrays
        x_train, x_val, x_test, y_train_as_values,y_train_as_deltas, y_val_as_values, y_val_as_deltas, y_test_as_values,y_test_as_deltas = get_train_val_and_test_arrays(SOURCE_DATA,TARGET_VALUE_DATA,
            TARGET_DIFF_DATA,VAL_PERCENTAGE,TEST_PERCENTAGE)

        #get pseudo predictions for later
        if (int(initial_dow) == int(first_dow_from_df)):
            print("Initial dow: %d, first dow from df: %d" %(initial_dow, first_dow_from_df))
            print("Getting %s test arrays" %coin)
            predictions_by_shifting = get_pseudo_predictions_for_nogt(x_train, x_val, x_test,True)
            pseudo_pred_dict[coin] = predictions_by_shifting
            # sys.exit(0)

        #needed for later
        num_orig_x_train_samples = x_train.shape[0]
        num_orig_x_val_samples = x_val.shape[0]
        num_orig_x_test_samples = x_test.shape[0]

        #convert arrays to sliding window format
        x_train,y_train_as_values,y_train_as_deltas = properly_convert_arrays_to_sliding_window_format_for_sep_x_and_y(x_train,y_train_as_values,y_train_as_deltas, window_size=SLIDING_WINDOW_AMOUNT)
        x_val,y_val_as_values,y_val_as_deltas = properly_convert_arrays_to_sliding_window_format_for_sep_x_and_y(x_val,y_val_as_values,y_val_as_deltas, window_size=SLIDING_WINDOW_AMOUNT)


        # #needed for later
        # num_orig_x_train_samples = x_train.shape[0]
        # num_orig_x_val_samples = x_val.shape[0]
        # num_orig_x_test_samples = x_test.shape[0]

        # #convert arrays to sliding window format
        # x_train,y_train_as_values,y_train_as_deltas = properly_convert_arrays_to_sliding_window_format_for_sep_x_and_y(x_train,y_train_as_values,y_train_as_deltas, window_size=SLIDING_WINDOW_AMOUNT)
        # x_val,y_val_as_values,y_val_as_deltas = properly_convert_arrays_to_sliding_window_format_for_sep_x_and_y(x_val,y_val_as_values,y_val_as_deltas, window_size=SLIDING_WINDOW_AMOUNT)

        #save arrays
        x_train_arrays.append(x_train.copy())
        y_train_as_values_arrays.append(y_train_as_values.copy())
        y_train_as_deltas_arrays.append(y_train_as_deltas.copy())

        x_val_arrays.append(x_val.copy())
        y_val_as_values_arrays.append(y_val_as_values.copy())
        y_val_as_deltas_arrays.append(y_val_as_deltas.copy())

        print("Initial dow: %d, first dow from df: %d" %(initial_dow, first_dow_from_df))

        if (int(initial_dow) == int(first_dow_from_df)):

            x_test_arrays.append(x_test.copy())
            y_test_as_values_arrays.append(y_test_as_values.copy())
            y_test_as_deltas_arrays.append(y_test_as_deltas.copy())
            # sys.exit(0)


    #now calulcate dates
    num_training_dates = num_orig_x_train_samples * X_ARRAY_TIMESTEPS_IN_A_SEQUENCE
    num_val_dates = num_orig_x_val_samples * X_ARRAY_TIMESTEPS_IN_A_SEQUENCE
    num_test_dates = num_orig_x_test_samples * X_ARRAY_TIMESTEPS_IN_A_SEQUENCE

    #get dates
    training_dates,val_dates,test_dates = calculate_train_val_and_test_dates(dates,num_training_dates, num_val_dates,num_test_dates,main_output_dir)

    #get features of interest
    FEATURES_OF_INTEREST = list(df)

    #get indices of target cols
    target_cols = fts_to_predict
    target_col_indices=[]
    print("\nGetting target col indices...")
    for tc in target_cols:
        idx = FEATURES_OF_INTEREST.index(tc)
        target_col_indices.append(idx)
        print("%s is at index %d" %(tc,idx))
    # sys.exit(0)

    #now concat these arrays along 0 axis
    x_train = np.concatenate(x_train_arrays, axis=0)
    y_train_as_values = np.concatenate(y_train_as_values_arrays, axis=0)
    y_train_as_deltas = np.concatenate(y_train_as_deltas_arrays, axis=0)

    x_val = np.concatenate(x_val_arrays, axis=0)
    y_val_as_values = np.concatenate(y_val_as_values_arrays, axis=0)
    y_val_as_deltas = np.concatenate(y_val_as_deltas_arrays, axis=0)

    x_test = np.concatenate(x_test_arrays, axis=0)
    y_test_as_values = np.concatenate(y_test_as_values_arrays, axis=0)
    y_test_as_deltas = np.concatenate(y_test_as_deltas_arrays, axis=0)

    #Now filter y arrays to only target target cols
    y_train_as_values = y_train_as_values[:,:,target_col_indices]
    y_train_as_deltas = y_train_as_deltas[:,:,target_col_indices]

    y_val_as_values = y_val_as_values[:,:,target_col_indices]
    y_val_as_deltas = y_val_as_deltas[:,:,target_col_indices]

    y_test_as_values = y_test_as_values[:,:,target_col_indices]
    y_test_as_deltas = y_test_as_deltas[:,:,target_col_indices]

    print("\n Training: \n")
    print(x_train)
    print(y_train_as_values)
    print(y_train_as_deltas)

    print("\n Val: \n")
    print(x_val)
    print(y_val_as_values)
    print(y_val_as_deltas)

    print("\n Testget_aggregated_daily_featuresget_aggregated_daily_features: \n")
    print(x_test)
    print(y_test_as_values)
    print(y_test_as_deltas)



    print("\nx_train shape: %s" %str(x_train.shape))
    print("y_train_as_values shape: %s" %str(y_train_as_values.shape))
    print("y_train_as_deltas shape: %s" %str(y_train_as_deltas.shape))

    print("\nx_val shape: %s" %str(x_val.shape))
    print("y_val_as_values shape: %s" %str(y_val_as_values.shape))
    print("y_val_as_deltas shape: %s" %str(y_val_as_deltas.shape))

    print("\nx_test shape: %s" %str(x_test.shape))
    print("y_test_as_values shape: %s" %str(y_test_as_values.shape))
    print("y_test_as_deltas shape: %s" %str(y_test_as_deltas.shape))

    SOURCE_FEATURES = FEATURES_OF_INTEREST
    TARGET_FEATURES = fts_to_predict

    return x_train, y_train_as_values,y_train_as_deltas,x_val, y_val_as_values,y_val_as_deltas, x_test, y_test_as_values,y_test_as_deltas,SOURCE_FEATURES,TARGET_FEATURES,pseudo_pred_dict




def normalize_train_and_test_and_save_scaler(coin,x_train, x_test,main_output_dir,feature_range=(0,1)):
    print("\nNormalizing data...")
    x_train, x_test, scaler=normalize_train_and_test_data(x_train, x_test, feature_range=feature_range)
    #SAVE THE SCALER
    SCALER_OUTPUT_DIR=main_output_dir+ "All-Coin-Scalers/"
    create_output_dir(SCALER_OUTPUT_DIR)
    SCALER_OUTPUT_FP=SCALER_OUTPUT_DIR+"%s-scaler.json"%coin
    save_scaler(scaler,SCALER_OUTPUT_FP)

    #make sure the save worked
    SCALER_INPUT_FP=SCALER_OUTPUT_FP
    scaler=load_scaler(SCALER_INPUT_FP)
    print("Done normalizing data!")

    return x_train, x_test, scaler, SCALER_OUTPUT_FP

def properly_normalize_data(x_train, x_val, x_test, feature_range=(0,1)):
    scaler=MinMaxScaler(feature_range)

    old_train_shape=x_train.shape
    old_val_shape=x_val.shape
    old_test_shape=x_test.shape

    # old_data_shape_dict={
    # "old_train_shape":old_train_shape,
    # "old_val_shape" : old_val_shape,
    # "old_test_shape":old_test_shape
    # }

    # x_train=x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2])
    # x_val=x_val.reshape(x_val.shape[0],x_val.shape[1]*x_val.shape[2])
    # x_test=x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2])

    if len(old_train_shape) == 3:
        # x_train=x_train.reshape(x_train.shape[0]*x_train.shape[1],x_train.shape[2])
        # x_val=x_val.reshape(x_val.shape[0]*x_val.shape[1],x_val.shape[2])
        # x_test=x_test.reshape(x_test.shape[0]*x_test.shape[1],x_test.shape[2])
        x_train=x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2])
        x_val=x_val.reshape(x_val.shape[0],x_val.shape[1]*x_val.shape[2])
        x_test=x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2])



    print(scaler.fit(x_train))
    print(scaler.data_max_)

    x_train=scaler.transform(x_train)
    x_val=scaler.transform(x_val)
    x_test=scaler.transform(x_test)

    x_train=x_train.reshape(old_train_shape)
    x_val=x_val.reshape(old_val_shape)
    x_test=x_test.reshape(old_test_shape)

    # print("Shape of x_train: %s" %str(x_train.shape))
    # print("Shape of x_val: %s" %str(x_val.shape))
    # print("Shape of x_test: %s" %str(x_test.shape))

    # sys.exit(0)

    return x_train, x_val, x_test, scaler

def normalize_training_data(x_train, feature_range=(0,1)):
    print("\nNormalizing training data!")

    #get scaler
    scaler=MinMaxScaler(feature_range)

    #get train shape
    old_train_shape=x_train.shape

    #reshape if shape is 3
    if len(old_train_shape) == 3:
        x_train=x_train.reshape(x_train.shape[0]*x_train.shape[1],x_train.shape[2])

    print(scaler.fit(x_train))
    print(scaler.data_max_)

    #fit scaler
    print(scaler.fit(x_train))
    print(scaler.data_max_)

    #now transform

    x_train=scaler.transform(x_train)

    #reshape x_train to orig shape
    x_train=x_train.reshape(old_train_shape)

    return x_train,scaler

def normalize_data_with_scaler(data, scaler):

    #get train shape
    old_data_shape=data.shape

    #reshape if shape is 3
    if len(old_data_shape) == 3:
        data=data.reshape(data.shape[0]*data.shape[1],data.shape[2])

    data=scaler.transform(data)

    data=data.reshape(old_data_shape)

    return data





def normalize_training_data_and_save_scaler(x_train, main_output_dir, feature_range=(0,1)):

    #norm x train
    x_train,scaler = normalize_training_data(x_train,feature_range)

    SCALER_OUTPUT_DIR=main_output_dir
    create_output_dir(SCALER_OUTPUT_DIR)
    SCALER_OUTPUT_FP=SCALER_OUTPUT_DIR+"Scaler.json"
    save_scaler(scaler,SCALER_OUTPUT_FP)

    #make sure the save worked
    SCALER_INPUT_FP=SCALER_OUTPUT_FP
    scaler=load_scaler(SCALER_INPUT_FP)
    print("Done normalizing data!")

    return x_train,scaler, SCALER_OUTPUT_FP

def properly_normalize_all_data_and_save_scaler(x_train, x_val, coin_test_dict, main_output_dir,feature_range=(0,1)):

    #norm training data and get scaler
    x_train,scaler, SCALER_OUTPUT_FP = normalize_training_data_and_save_scaler(x_train, main_output_dir,feature_range)

    #time to norm val data
    old_val_shape=x_val.shape
    if len(old_val_shape) == 3:
        # x_val=x_val.reshape(x_val.shape[0],x_val.shape[1]*x_val.shape[2])

        x_val=x_val.reshape(x_val.shape[0] * x_val.shape[1],x_val.shape[2])

    #rescale val
    x_val=scaler.transform(x_val)

    #get orig shape
    x_val = x_val.reshape(old_val_shape)

    #now reshape x test
    coins = coin_test_dict.keys()
    for coin in coins:

        #get cur test array
        x_test = coin_test_dict[coin][0]
        y_test_as_values = coin_test_dict[coin][1]
        y_test_as_deltas= coin_test_dict[coin][2]

        #save old shape
        old_test_shape=x_test.shape
        if len(old_test_shape) == 3:
            x_test=x_test.reshape(x_test.shape[0]*x_test.shape[1],x_test.shape[2])

        #rescale test data
        x_test=scaler.transform(x_test)

        #reshape
        x_test = x_test.reshape(old_test_shape)

        #now save
        coin_test_dict[coin] = (x_test, y_test_as_values, y_test_as_deltas)

    return x_train, x_val ,scaler, SCALER_OUTPUT_FP,coin_test_dict

def normalize_train_val_and_test_and_save_scaler(coin,x_train, x_val,x_test,main_output_dir,feature_range=(0,1)):
    print("\nNormalizing data...")
    x_train, x_val,x_test, scaler=normalize_data(x_train, x_val,x_test, feature_range=feature_range)
    #SAVE THE SCALER
    SCALER_OUTPUT_DIR=main_output_dir+ "All-Coin-Scalers/"
    create_output_dir(SCALER_OUTPUT_DIR)
    SCALER_OUTPUT_FP=SCALER_OUTPUT_DIR+"%s-scaler.json"%coin
    save_scaler(scaler,SCALER_OUTPUT_FP)

    #make sure the save worked
    SCALER_INPUT_FP=SCALER_OUTPUT_FP
    scaler=load_scaler(SCALER_INPUT_FP)
    print("Done normalizing data!")

    return x_train, x_val, x_test, scaler, SCALER_OUTPUT_FP

def reconfig_feature_list_for_2d_array(SOURCE_FEATURE_LIST, X_ARRAY_TIMESTEPS_IN_A_SEQUENCE):
    new_fts=[]
    orig_fts=[]
    # tag = "_timestep_"
    for i in range(X_ARRAY_TIMESTEPS_IN_A_SEQUENCE):
        for ft in SOURCE_FEATURE_LIST:
            new_ft = ft + "_timestep_%d"%i
            new_fts.append(new_ft)
            orig_fts.append(ft)

    data = {"ft_name":new_fts, "original_ft_name":orig_fts}

    df = pd.DataFrame(data = data)
    # print(df)
    return new_fts, df

def reconfig_feature_list_for_2d_array_and_save(SOURCE_FEATURE_LIST, X_ARRAY_TIMESTEPS_IN_A_SEQUENCE, output_dir):
    new_fts, df = reconfig_feature_list_for_2d_array(SOURCE_FEATURE_LIST, X_ARRAY_TIMESTEPS_IN_A_SEQUENCE)

    output_fp = output_dir +  "Features.csv"
    df.to_csv(output_fp, index=False)
    print("Saved features to %s" %output_fp)

    return new_fts, df


def remove_unnamed_col(cols):
    for col in cols:
        if "Unnamed" in col:
            cols.remove(col)
            print("Removed %s" %col)
    return cols

def get_rf_feature_importances(random_forest, SOURCE_FEATURES_BY_TIMESTEP,new_ft_name_df):

    #get feature importances:
    feature_importances=pd.DataFrame(random_forest.feature_importances_, index=SOURCE_FEATURES_BY_TIMESTEP, columns=['importance']).sort_values('importance',ascending=False )
    feature_importances["ft_name"] = SOURCE_FEATURES_BY_TIMESTEP

    #merge with new name df
    feature_importances = pd.merge(feature_importances, new_ft_name_df, on=["ft_name"], how="inner")

    feature_importances["importance"] = feature_importances.groupby("original_ft_name")["importance"].transform("mean")

    feature_importances = feature_importances.drop_duplicates("original_ft_name")
    feature_importances=feature_importances[["original_ft_name", "importance"]].sort_values('importance',ascending=False ).reset_index(drop=True)

    return feature_importances

def get_and_save_rf_feature_importances(random_forest, SOURCE_FEATURES_BY_TIMESTEP,new_ft_name_df, output_dir):

    feature_importances = get_rf_feature_importances(random_forest, SOURCE_FEATURES_BY_TIMESTEP,new_ft_name_df)

    output_fp= output_dir + "Feature-Importances.csv"
    feature_importances.to_csv(output_fp, index=False)
    return feature_importances

def get_top_n_features_from_random_forest_fp(desired_ft_fp, TOP_N_FEATURES,ft_name):

    df= pd.read_csv(desired_ft_fp)

    df=df.head(TOP_N_FEATURES)

    fts= list(df[ft_name])

    return fts

def get_ts_features_for_nn(input_dir,main_output_dir,desired_features,start_date,end_date, target_cols, X_ARRAY_TIMESTEPS_IN_A_SEQUENCE,Y_ARRAY_TIMESTEPS_IN_A_SEQUENCE,VAL_PERCENTAGE,
    TEST_PERCENTAGE,kickout_cols=[],dow_cols=[],SLIDING_WINDOW_AMOUNT=7,DEBUG=False):



    #get all infps
    infps=os.listdir(input_dir)

    #make sure only csv files are here
    for infp in infps:
        if ".csv" not in infp:
            infps.remove(infp)

    # if DEBUG==True:
    #   infps=infps[:1]
    print(infps)

    #save arrays here
    x_train_arrays=[]
    y_train_as_values_arrays=[]
    y_train_as_deltas_arrays=[]

    x_val_arrays=[]
    y_val_as_values_arrays=[]
    y_val_as_deltas_arrays=[]

    x_test_arrays=[]
    y_test_as_values_arrays=[]
    y_test_as_deltas_arrays=[]

    #test data dict for each coin
    test_data_dict={}

    #get the starting day of the week
    start_date_obj=pd.to_datetime(start_date, utc=True)
    start_dow=start_date_obj.weekday()
    initial_dow=start_dow
    print("Starting DOW: %s" %str(start_dow))


    #now open each infp as df
    #then retrieve the arrays
    for infp in infps:
        df=pd.read_csv(input_dir + infp)

        #get first dow
        first_dow_from_df = int(df["DOW"].iloc[0])

        #config df by dates
        df = config_df_by_dates(df,start_date,end_date)

        #save dow df for later
        dow_df=df[["created_at"] + dow_cols]

        #get dates
        dates=list(df["created_at"])

        ################### IMPORTANT #####################
        #keep cols
        df = df[list(set(desired_features + target_cols + dow_cols))]
        #####################################################

        # #kickout cols
        # df=df.drop(kickout_cols, axis=1)

        #move dow cols to end
        df_cols = list(df)
        # df_cols = remove_unnamed_col(df_cols)

        # if TWEET_FT_FILTER == True:
        #     # from ts_tweet_fts import tweet_fts
        #     df_cols = tweet_fts

        # if FT_LIMIT_FLAG == True:
        #     df_cols = df_cols[:FT_LIMIT] + target_cols
        #     df_cols = list(set(df_cols))

        # df = df[df_cols]

        # df =df[[c for c in df if c not in dow_cols] + dow_cols]
        # df =df.drop(dow_cols, axis=1)

        #get feature array
        FEATURE_ARRAY= np.asarray(df).astype("float32")
        print("Shape of feature array: %s" %str(FEATURE_ARRAY.shape))

        #get source and target arrays for later
        SOURCE_DATA,TARGET_VALUE_DATA,TARGET_DIFF_DATA = get_target_value_and_difference_arrays(FEATURE_ARRAY,X_ARRAY_TIMESTEPS_IN_A_SEQUENCE,Y_ARRAY_TIMESTEPS_IN_A_SEQUENCE)

        #get x and y arrays
        x_train, x_val, x_test, y_train_as_values,y_train_as_deltas, y_val_as_values, y_val_as_deltas, y_test_as_values,y_test_as_deltas = get_train_val_and_test_arrays(SOURCE_DATA,TARGET_VALUE_DATA,
            TARGET_DIFF_DATA,VAL_PERCENTAGE,TEST_PERCENTAGE)

        #needed for later
        num_orig_x_train_samples = x_train.shape[0]
        num_orig_x_val_samples = x_val.shape[0]
        num_orig_x_test_samples = x_test.shape[0]

        #convert arrays to sliding window format
        x_train,y_train_as_values,y_train_as_deltas = properly_convert_arrays_to_sliding_window_format_for_sep_x_and_y(x_train,y_train_as_values,y_train_as_deltas, window_size=SLIDING_WINDOW_AMOUNT)
        x_val,y_val_as_values,y_val_as_deltas = properly_convert_arrays_to_sliding_window_format_for_sep_x_and_y(x_val,y_val_as_values,y_val_as_deltas, window_size=SLIDING_WINDOW_AMOUNT)

        #save arrays
        x_train_arrays.append(x_train.copy())
        y_train_as_values_arrays.append(y_train_as_values.copy())
        y_train_as_deltas_arrays.append(y_train_as_deltas.copy())

        x_val_arrays.append(x_val.copy())
        y_val_as_values_arrays.append(y_val_as_values.copy())
        y_val_as_deltas_arrays.append(y_val_as_deltas.copy())

        print("Initial dow: %d, first dow from df: %d" %(initial_dow, first_dow_from_df))

        if (int(initial_dow) == int(first_dow_from_df)):
            x_test_arrays.append(x_test.copy())
            y_test_as_values_arrays.append(y_test_as_values.copy())
            y_test_as_deltas_arrays.append(y_test_as_deltas.copy())


    #now calulcate dates
    num_training_dates = num_orig_x_train_samples * X_ARRAY_TIMESTEPS_IN_A_SEQUENCE
    num_val_dates = num_orig_x_val_samples * X_ARRAY_TIMESTEPS_IN_A_SEQUENCE
    num_test_dates = num_orig_x_test_samples * X_ARRAY_TIMESTEPS_IN_A_SEQUENCE

    #get dates
    training_dates,val_dates,test_dates = calculate_train_val_and_test_dates(dates,num_training_dates, num_val_dates,num_test_dates,main_output_dir)

    #get features of interest
    FEATURES_OF_INTEREST = list(df)

    #get indices of target cols
    target_col_indices=[]
    print("\nGetting target col indices...")
    for tc in target_cols:
        idx = FEATURES_OF_INTEREST.index(tc)
        target_col_indices.append(idx)
        print("%s is at index %d" %(tc,idx))

    #now concat these arrays along 0 axis
    x_train = np.concatenate(x_train_arrays, axis=0)
    y_train_as_values = np.concatenate(y_train_as_values_arrays, axis=0)
    y_train_as_deltas = np.concatenate(y_train_as_deltas_arrays, axis=0)

    x_val = np.concatenate(x_val_arrays, axis=0)
    y_val_as_values = np.concatenate(y_val_as_values_arrays, axis=0)
    y_val_as_deltas = np.concatenate(y_val_as_deltas_arrays, axis=0)

    x_test = np.concatenate(x_test_arrays, axis=0)
    y_test_as_values = np.concatenate(y_test_as_values_arrays, axis=0)
    y_test_as_deltas = np.concatenate(y_test_as_deltas_arrays, axis=0)

    #Now filter y arrays to only target target cols
    y_train_as_values = y_train_as_values[:,:,target_col_indices]
    y_train_as_deltas = y_train_as_deltas[:,:,target_col_indices]

    y_val_as_values = y_val_as_values[:,:,target_col_indices]
    y_val_as_deltas = y_val_as_deltas[:,:,target_col_indices]

    y_test_as_values = y_test_as_values[:,:,target_col_indices]
    y_test_as_deltas = y_test_as_deltas[:,:,target_col_indices]

    print("\nx_train shape: %s" %str(x_train.shape))
    print("y_train_as_values shape: %s" %str(y_train_as_values.shape))
    print("y_train_as_deltas shape: %s" %str(y_train_as_deltas.shape))

    print("\nx_val shape: %s" %str(x_val.shape))
    print("y_val_as_values shape: %s" %str(y_val_as_values.shape))
    print("y_val_as_deltas shape: %s" %str(y_val_as_deltas.shape))

    print("\nx_test shape: %s" %str(x_test.shape))
    print("y_test_as_values shape: %s" %str(y_test_as_values.shape))
    print("y_test_as_deltas shape: %s" %str(y_test_as_deltas.shape))

    return x_train, y_train_as_values,y_train_as_deltas,x_val, y_val_as_values,y_val_as_deltas, x_test, y_test_as_values,y_test_as_deltas,FEATURES_OF_INTEREST,target_col_indices


def config_ae_sliding_window_data(x, y_as_values, y_as_deltas,initial_DOW,IS_TRAINING_OR_VAL,DOW_FLAG):

    if IS_TRAINING_OR_VAL == True:

        x, y_as_values,y_as_deltas=convert_training_array_to_sliding_window_format_with_DOW_OPTION(initial_DOW,x,  y_as_values,y_as_deltas, DOW_FLAG)
    else:
        x, y_as_values,y_as_deltas=convert_testing_array_to_sliding_window_format_with_DOW_OPTION(initial_DOW,x,  y_as_values,y_as_deltas, DOW_FLAG)

    if DOW_FLAG == False:
        y = x.copy()
    else:
        y = x[:, :, :-7].copy()
        y = y.reshape(y.shape[0], y.shape[1], 1)

    return x, y

def get_main_data_arrays(df, train_start,train_end, val_start,val_end, test_start,test_end,LOG1_FLAG,LOG2_FLAG,LSTM_UNITS,
                                        LOG_ADD_COEFF,X_ARRAY_TIMESTEPS_IN_A_SEQUENCE,Y_ARRAY_TIMESTEPS_IN_A_SEQUENCE):


    #get train, val, and test dfs
    TRAIN_FEATURE_DF = config_df_by_dates(df,train_start,train_end).copy()
    VAL_FEATURE_DF = config_df_by_dates(df,val_start,val_end).copy()
    TEST_FEATURE_DF = config_df_by_dates(df,test_start,test_end).copy()


    #get training data
    x_train, y_train_as_values,y_train_as_deltas,_,initial_DOW= cp3_get_training_array_from_feature_df_DAILY(TRAIN_FEATURE_DF,LOG1_FLAG,LOG2_FLAG,train_start,LSTM_UNITS,
        LOG_ADD_COEFF,X_ARRAY_TIMESTEPS_IN_A_SEQUENCE,Y_ARRAY_TIMESTEPS_IN_A_SEQUENCE)

    #get val data
    x_val, y_val_as_values,y_val_as_deltas,_,initial_DOW = cp3_get_training_array_from_feature_df_DAILY(VAL_FEATURE_DF,LOG1_FLAG,LOG2_FLAG,test_start,
        LSTM_UNITS,LOG_ADD_COEFF,X_ARRAY_TIMESTEPS_IN_A_SEQUENCE,Y_ARRAY_TIMESTEPS_IN_A_SEQUENCE)

    #get test data
    x_test, y_test_as_values,y_test_as_deltas,_,initial_DOW = cp3_get_training_array_from_feature_df_DAILY(TEST_FEATURE_DF,LOG1_FLAG,LOG2_FLAG,test_start,
        LSTM_UNITS,LOG_ADD_COEFF,X_ARRAY_TIMESTEPS_IN_A_SEQUENCE,Y_ARRAY_TIMESTEPS_IN_A_SEQUENCE)

    #reshape x arrays
    x_train=reconfigure_x_array_shape_for_training(x_train,y_train_as_values, "x_train")
    x_val=reconfigure_x_array_shape_for_training(x_val,y_val_as_values, "x_val")
    x_test=reconfigure_x_array_shape_for_training(x_test,y_test_as_values, "x_test")

    # #get pseudo predictions for later
    predictions_by_shifting = get_pseudo_predictions_for_nogt(x_train, x_val, x_test)

    #convert arrays to sliding window format with day of the week as a feature
    x_train, y_train_as_values,y_train_as_deltas=convert_training_array_to_sliding_window_format_with_DOW_features(initial_DOW,x_train,  y_train_as_values,y_train_as_deltas)

    #convert arrays to sliding window format with day of the week as a feature
    x_val, y_val_as_values,y_val_as_deltas=convert_training_array_to_sliding_window_format_with_DOW_features(initial_DOW,x_val,  y_val_as_values,y_val_as_deltas)

    #add day of week to test data
    x_test, y_test_as_values,y_test_as_deltas=convert_testing_array_to_sliding_window_format_with_DOW_features(initial_DOW,x_test,  y_test_as_values,y_test_as_deltas)

    #print your shapes
    print("Shape of x_train: %s" %str(x_train.shape))
    print("Shape of y_train_as_deltas: %s" %str(y_train_as_deltas.shape))
    print("Shape of y_train_as_values: %s" %str(y_train_as_values.shape))

    print("Shape of x_val: %s" %str(x_val.shape))
    print("Shape of y_val_as_deltas: %s" %str(y_val_as_deltas.shape))
    print("Shape of y_val_as_values: %s" %str(y_val_as_values.shape))

    print("\nShape of x_test: %s" %str(x_test.shape))
    print("Shape of y_test_as_deltas: %s" %str(y_test_as_deltas.shape))
    print("Shape of y_test_as_values: %s" %str(y_test_as_values.shape))

    #now we can augment the data


def augment_data(FEATURE_DF,X_ARRAY_TIMESTEPS_IN_A_SEQUENCE,Y_ARRAY_TIMESTEPS_IN_A_SEQUENCE,start_date,x_data, y_as_values,y_as_deltas,LOG1_FLAG,LOG2_FLAG,LOG_ADD_COEFF, NUM_STD_DEVS, NUM_X_AXIS_BUCKETS_FOR_PSEUDO_GAUSSIAN=100,MOD_NORM=30,time_col = "created_at"):

    #get features we want to aug
    cols = list(FEATURE_DF)
    if time_col in cols:
        cols.remove(time_col)


    #log normalize
    for col in cols:
        if LOG1_FLAG==True:
            FEATURE_DF[col]=np.log1p(FEATURE_DF[col] + LOG_ADD_COEFF)
        if LOG2_FLAG==True:
            FEATURE_DF[col]=np.log1p(FEATURE_DF[col] + LOG_ADD_COEFF)

    #need to make new time series for each col
    for col in cols:
        print("Augmenting data for %s col" %col)

        time_series = FEATURE_DF[col]

        #get mean and std for later
        #get mean and standard dev
        mean,std=norm.fit(time_series)
        print("Mean: %.4f" %mean)
        print("std: %.4f" %std)

        #fit a gaussian to our time series. We'll need it for making a synthetic version.
        plt.hist(time_series, bins=30, normed=True)
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, NUM_X_AXIS_BUCKETS_FOR_PSEUDO_GAUSSIAN)
        y = norm.pdf(x, mean, std)
        # plt.plot(x, y)
        # plt.show()

        #setup std coeff param
        STD_COEFF = NUM_STD_DEVS * std
        print("Data augmenting with %d std dev(s)."%NUM_STD_DEVS)
        print("STD is %.4f" %std)
        print("STD COEFF is: %.4f" %STD_COEFF)

        #to keep track
        series_len = len(time_series)

        #gen new time series
        #new time series
        new_series = []
        for i,val in enumerate(time_series):
            print(val)
            lower = val - STD_COEFF
            upper = val + STD_COEFF
        #     print("Val is %.2f" %val)
        #     print("Lower bound is %.2f" %lower)
        #     print("Upper bound is %.2f" %upper)

            #get sub array:
            candidates = y[np.where(np.logical_and(y>=lower, y<=upper))]
            if candidates.shape[0]==0:
                print("Couldn't augment val %.4f" %val)
                new_series.append(val)
            else:
                #randomly sample
                new_val = random.sample(list(candidates), 1)[0]
                new_series.append(new_val)

            if i%MOD_NORM== 0:
                print("Processed val %d of %d in time series" %(i, series_len))

        FEATURE_DF[col] = new_series

    #UNlog normalize
    for col in cols:
        if LOG1_FLAG==True:
           FEATURE_DF[col]=np.expm1(FEATURE_DF[col]) - LOG_ADD_COEFF
        if LOG2_FLAG==True:
            FEATURE_DF[col]=np.expm1(FEATURE_DF[col]) - LOG_ADD_COEFF

    #now convert new df to an array
    #get training data
    new_x, new_y_as_values,new_y_as_deltas,initial_DOW= cp3_get_array_from_feature_df_DAILY(FEATURE_DF,LOG1_FLAG,LOG2_FLAG,start_date,
        LOG_ADD_COEFF,X_ARRAY_TIMESTEPS_IN_A_SEQUENCE,Y_ARRAY_TIMESTEPS_IN_A_SEQUENCE)

    #convert to sliding window format
    new_x, new_y_as_values,new_y_as_deltas=convert_training_array_to_sliding_window_format_with_DOW_features(initial_DOW,new_x, new_y_as_values,new_y_as_deltas)

    #combine with old data
    print("Shape of new x: %s" %str(new_x.shape))
    print("Shape of x: %s" %str(x_data.shape))
    new_x = np.concatenate([x_data, new_x])
    new_y_as_values = np.concatenate([y_as_values, new_y_as_values])
    new_y_as_deltas = np.concatenate([y_as_deltas, new_y_as_deltas])

    return new_x, new_y_as_values,new_y_as_deltas

def augment_data_with_or_without_original(FEATURE_DF,X_ARRAY_TIMESTEPS_IN_A_SEQUENCE,Y_ARRAY_TIMESTEPS_IN_A_SEQUENCE,start_date,x_data, y_as_values,y_as_deltas,LOG1_FLAG,LOG2_FLAG,LOG_ADD_COEFF, NUM_STD_DEVS, NUM_X_AXIS_BUCKETS_FOR_PSEUDO_GAUSSIAN=100,MOD_NORM=30,time_col = "created_at",RETURN_ORIGINAL_DATA=False):

    #get features we want to aug
    cols = list(FEATURE_DF)
    if time_col in cols:
        cols.remove(time_col)


    #log normalize
    for col in cols:
        if LOG1_FLAG==True:
            FEATURE_DF[col]=np.log1p(FEATURE_DF[col] + LOG_ADD_COEFF)
        if LOG2_FLAG==True:
            FEATURE_DF[col]=np.log1p(FEATURE_DF[col] + LOG_ADD_COEFF)

    #need to make new time series for each col
    for col in cols:
        print("Augmenting data for %s col" %col)

        time_series = FEATURE_DF[col]

        #get mean and std for later
        #get mean and standard dev
        mean,std=norm.fit(time_series)
        print("Mean: %.4f" %mean)
        print("std: %.4f" %std)

        #fit a gaussian to our time series. We'll need it for making a synthetic version.
        plt.hist(time_series, bins=30, normed=True)
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, NUM_X_AXIS_BUCKETS_FOR_PSEUDO_GAUSSIAN)
        y = norm.pdf(x, mean, std)
        # plt.plot(x, y)
        # plt.show()

        #setup std coeff param
        STD_COEFF = NUM_STD_DEVS * std
        print("Data augmenting with %d std dev(s)."%NUM_STD_DEVS)
        print("STD is %.4f" %std)
        print("STD COEFF is: %.4f" %STD_COEFF)

        #to keep track
        series_len = len(time_series)

        #gen new time series
        #new time series
        new_series = []
        for i,val in enumerate(time_series):
            print(val)
            lower = val - STD_COEFF
            upper = val + STD_COEFF
        #     print("Val is %.2f" %val)
        #     print("Lower bound is %.2f" %lower)
        #     print("Upper bound is %.2f" %upper)

            #get sub array:
            candidates = y[np.where(np.logical_and(y>=lower, y<=upper))]
            if candidates.shape[0]==0:
                print("Couldn't augment val %.4f" %val)
                new_series.append(val)
            else:
                #randomly sample
                new_val = random.sample(list(candidates), 1)[0]
                new_series.append(new_val)

            if i%MOD_NORM== 0:
                print("Processed val %d of %d in time series" %(i, series_len))

        FEATURE_DF[col] = new_series

    #UNlog normalize
    for col in cols:
        if LOG1_FLAG==True:
           FEATURE_DF[col]=np.expm1(FEATURE_DF[col]) - LOG_ADD_COEFF
        if LOG2_FLAG==True:
            FEATURE_DF[col]=np.expm1(FEATURE_DF[col]) - LOG_ADD_COEFF

    #now convert new df to an array
    #get training data
    new_x, new_y_as_values,new_y_as_deltas,initial_DOW= cp3_get_array_from_feature_df_DAILY(FEATURE_DF,LOG1_FLAG,LOG2_FLAG,start_date,
        LOG_ADD_COEFF,X_ARRAY_TIMESTEPS_IN_A_SEQUENCE,Y_ARRAY_TIMESTEPS_IN_A_SEQUENCE)

    #convert to sliding window format
    new_x, new_y_as_values,new_y_as_deltas=convert_training_array_to_sliding_window_format_with_DOW_features(initial_DOW,new_x, new_y_as_values,new_y_as_deltas)


    #combine with old data
    if RETURN_ORIGINAL_DATA == True:
        new_x = np.concatenate([x_data, new_x])
        new_y_as_values = np.concatenate([y_as_values, new_y_as_values])
        new_y_as_deltas = np.concatenate([y_as_deltas, new_y_as_deltas])


    print("Shape of new x: %s" %str(new_x.shape))
    print("Shape of old x: %s" %str(x_data.shape))

    return new_x, new_y_as_values,new_y_as_deltas

def multi_augment_data(ORIGINAL_FEATURE_DF,X_ARRAY_TIMESTEPS_IN_A_SEQUENCE,Y_ARRAY_TIMESTEPS_IN_A_SEQUENCE,start_date,x_data, y_as_values,y_as_deltas,LOG1_FLAG,LOG2_FLAG,LOG_ADD_COEFF, NUM_STD_DEVS_LIST, NUM_X_AXIS_BUCKETS_FOR_PSEUDO_GAUSSIAN=100,MOD_NORM=30,time_col = "created_at",RETURN_ORIGINAL_DATA=True):

    #if no stds are specified, just return data as is
    if NUM_STD_DEVS_LIST==[]:
        print("No std devs have been specified, so returning data as is...")
        return x_data, y_as_values,y_as_deltas

    #SAVE NEW DATA HERE
    ALL_NEW_X_ARRAYS = []
    ALL_NEW_Y_VALUE_ARRAYS = []
    ALL_NEW_Y_DELTA_ARRAYS = []

    #iterate and get each std dev
    for NUM_STD_DEVS in NUM_STD_DEVS_LIST:

        #get feature df
        FEATURE_DF = ORIGINAL_FEATURE_DF.copy()

        #get features we want to aug
        cols = list(FEATURE_DF)
        if time_col in cols:
            cols.remove(time_col)

        #log normalize
        for col in cols:
            if LOG1_FLAG==True:
                FEATURE_DF[col]=np.log1p(FEATURE_DF[col] + LOG_ADD_COEFF)
            if LOG2_FLAG==True:
                FEATURE_DF[col]=np.log1p(FEATURE_DF[col] + LOG_ADD_COEFF)

        #need to make new time series for each col
        for col in cols:
            print("Augmenting data for %s col" %col)

            time_series = FEATURE_DF[col]

            #get mean and std for later
            #get mean and standard dev
            mean,std=norm.fit(time_series)
            print("Mean: %.4f" %mean)
            print("std: %.4f" %std)

            #fit a gaussian to our time series. We'll need it for making a synthetic version.
            plt.hist(time_series, bins=30, normed=True)
            xmin, xmax = plt.xlim()
            x = np.linspace(xmin, xmax, NUM_X_AXIS_BUCKETS_FOR_PSEUDO_GAUSSIAN)
            y = norm.pdf(x, mean, std)
            # plt.plot(x, y)
            # plt.show()

            #setup std coeff param
            STD_COEFF = NUM_STD_DEVS * std
            print("Data augmenting with %d std dev(s)."%NUM_STD_DEVS)
            print("STD is %.4f" %std)
            print("STD COEFF is: %.4f" %STD_COEFF)

            #to keep track
            series_len = len(time_series)

            #gen new time series
            #new time series
            new_series = []
            for i,val in enumerate(time_series):
                print(val)
                lower = val - STD_COEFF
                upper = val + STD_COEFF
            #     print("Val is %.2f" %val)
            #     print("Lower bound is %.2f" %lower)
            #     print("Upper bound is %.2f" %upper)

                #get sub array:
                candidates = y[np.where(np.logical_and(y>=lower, y<=upper))]
                if candidates.shape[0]==0:
                    print("Couldn't augment val %.4f" %val)
                    new_series.append(val)
                else:
                    #randomly sample
                    new_val = random.sample(list(candidates), 1)[0]
                    new_series.append(new_val)

                if i%MOD_NORM== 0:
                    print("Processed val %d of %d in time series" %(i, series_len))

            FEATURE_DF[col] = new_series

        #UNlog normalize
        for col in cols:
            if LOG1_FLAG==True:
               FEATURE_DF[col]=np.expm1(FEATURE_DF[col]) - LOG_ADD_COEFF
            if LOG2_FLAG==True:
                FEATURE_DF[col]=np.expm1(FEATURE_DF[col]) - LOG_ADD_COEFF

        #now convert new df to an array
        #get training data
        new_x, new_y_as_values,new_y_as_deltas,initial_DOW= cp3_get_array_from_feature_df_DAILY(FEATURE_DF,LOG1_FLAG,LOG2_FLAG,start_date,
            LOG_ADD_COEFF,X_ARRAY_TIMESTEPS_IN_A_SEQUENCE,Y_ARRAY_TIMESTEPS_IN_A_SEQUENCE)

        #convert to sliding window format
        new_x, new_y_as_values,new_y_as_deltas=convert_training_array_to_sliding_window_format_with_DOW_features(initial_DOW,new_x, new_y_as_values,new_y_as_deltas)

        #save to an array
        ALL_NEW_X_ARRAYS.append(new_x)
        ALL_NEW_Y_VALUE_ARRAYS.append(new_y_as_values)
        ALL_NEW_Y_DELTA_ARRAYS.append(new_y_as_deltas)

    #concat results
    new_x = np.concatenate(ALL_NEW_X_ARRAYS)
    new_y_as_values = np.concatenate(ALL_NEW_Y_VALUE_ARRAYS)
    new_y_as_deltas = np.concatenate(ALL_NEW_Y_DELTA_ARRAYS)

    #combine with old data
    if RETURN_ORIGINAL_DATA == True:
        new_x = np.concatenate([x_data, new_x])
        new_y_as_values = np.concatenate([y_as_values, new_y_as_values])
        new_y_as_deltas = np.concatenate([y_as_deltas, new_y_as_deltas])

    return new_x, new_y_as_values, new_y_as_deltas

def get_and_save_gdf_and_pred_df_v2_with_tags(y_test,y_pred,reg_fts, y_test_dates, output_dir,pred_tag):

    #GET DFS
    gdf, pred_df = get_gtdf_and_pred_df(y_test,y_pred,reg_fts)

    print("\ny_test_dates")
    print(y_test_dates)

    #config final dfs
    gdf["nodeTime"] = y_test_dates
    pred_df["nodeTime"] = y_test_dates

    final_cols = ["nodeTime"] + list(gdf)
    gdf = gdf[final_cols]
    pred_df = pred_df[final_cols]

    final_output_dir = output_dir + "Ground-Truth-and-Predictions/"
    create_output_dir(final_output_dir)

    output_fp = final_output_dir + "Ground-Truth.csv"
    gdf.to_csv(output_fp, index=False)

    output_fp = final_output_dir + "%s-Predictions.csv"%pred_tag
    pred_df.to_csv(output_fp, index=False)

    return gdf, pred_df

def get_and_save_gdf_and_pred_df(y_test,y_pred,reg_fts, y_test_dates, output_dir):

    #GET DFS
    gdf, pred_df = get_gtdf_and_pred_df(y_test,y_pred,reg_fts)

    #config final dfs
    gdf["nodeTime"] = y_test_dates
    pred_df["nodeTime"] = y_test_dates

    final_cols = ["nodeTime"] + list(gdf)
    gdf = gdf[final_cols]
    pred_df = pred_df[final_cols]

    graph_output_dir = output_dir + "Ground-Truth-and-Predictions/"
    create_output_dir(graph_output_dir)

    output_fp = graph_output_dir + "Ground-Truth.csv"
    gdf.to_csv(output_fp, index=False)

    output_fp = graph_output_dir + "Predictions.csv"
    pred_df.to_csv(output_fp, index=False)

    return gdf, pred_df
