import sys
# sys.path.append("/data/Fmubang/cp4-code-clean/functions")
# sys.path.append("/data/Fmubang/cp4-VAM-write-up-exps/functions")
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

from sklearn.preprocessing import MinMaxScaler,StandardScaler 



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

def config_df_by_daily_dates(df,start_date,end_date,time_col="nodeTime"):
    df[time_col]=pd.to_datetime(df[time_col], utc=True)
    df = df.sort_values(time_col)
    df["temp_dates"] = df[time_col].dt.floor("D")
    df=df.set_index("temp_dates")
    df = df[(df.index >= start_date) & (df.index <= end_date)]
    df=df.reset_index(drop=True)
    return df

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

def make_param_tag(PARAM_VALS):

    PARAM_TAG = ""
    for p in PARAM_VALS:
        PARAM_TAG = PARAM_TAG + "-" + str(p)

    PARAM_TAG=PARAM_TAG[1:]


    return PARAM_TAG

def denormalize_single_array(data, scaler):
    old_data_shape=data.shape
    data=data.reshape(data.shape[0]*data.shape[1],data.shape[2])
    data=scaler.inverse_transform(data)
    data=data.reshape(old_data_shape)
    return data

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

def get_fts_from_fp(input_fp):
    model_fts = []
    with open(input_fp) as f:
        for line in f:
            line = line.replace("\n","")
            model_fts.append(line)
            print(line)
    return model_fts

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