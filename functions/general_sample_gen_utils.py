import pandas as pd
import os,sys
import numpy as np
from functools import *
import pickle

import seaborn as sns
from textwrap  import wrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import basic_utils as bu

import multiprocessing
import multiprocessing as mp


# import cp5_pnnl_metric_funcs as pmf

def log1p_with_neg_data(array):

    array = np.asarray(array)
    abs_val_array = np.abs(array)

    ones_array = array/abs_val_array
    ones_array = np.nan_to_num(ones_array)

    print("\narray")
    print(array)

    print("\nabs_val_array")
    print(abs_val_array)

    print("\nones_array")
    print(ones_array)

    abs_val_array=np.log1p(abs_val_array)
    print("\nabs_val_array log normed")
    print(abs_val_array)

    array = abs_val_array * ones_array
    print("\nlog normed orig array with negs")
    print(array)

    return array

def expm1_with_neg_data(array):

    array = np.asarray(array)
    abs_val_array = np.abs(array)

    ones_array = array/abs_val_array
    ones_array = np.nan_to_num(ones_array)

    print("\narray")
    print(array)

    print("\nabs_val_array")
    print(abs_val_array)

    print("\nones_array")
    print(ones_array)

    abs_val_array=np.expm1(abs_val_array)
    print("\nabs_val_array expm1-ed")
    print(abs_val_array)

    array = abs_val_array * ones_array
    print("\nexpm1-ed orig array with negs")
    print(array)

    return array

def get_sample_dfs_v2_para(df, train_date_tuples ,static_fts,dynamic_fts,target_fts,DEBUG_PRINT, time_col, NUM_JOBS):

    def debug_print(print_val="\n"):
        if DEBUG_PRINT == True:
            print(str(print_val))

    #all input dfs go here
    all_input_dynamic_dfs = []
    all_target_dfs = []

    #get all arg tuples for the parallel function
    arg_tuples = []
    num_train_tuples = len(train_date_tuples)
    for idx,train_tuple in enumerate(train_date_tuples):
        arg_tuple = (df,idx, train_tuple, num_train_tuples, static_fts,dynamic_fts,target_fts,DEBUG_PRINT, time_col)
        arg_tuples.append(arg_tuple)

    print("\nRunning MP...")
    pool = mp.Pool(processes= NUM_JOBS)

    print("\nLaunching parallel func...")
    sample_data_tuples = pool.map(get_single_input_and_output_dfs, arg_tuples)
    pool.close()

    all_input_dfs = []
    all_target_dfs = []

    for stuple in sample_data_tuples:

        idx,input_cols,target_cols,input_dynamic_df,target_df = stuple
        all_input_dfs.append(input_dynamic_df)
        all_target_dfs.append(target_df)

    input_dynamic_df = pd.concat(all_input_dfs).reset_index(drop=True)
    target_df = pd.concat(all_target_dfs).reset_index(drop=True)

    print("\ninput_dynamic_df before sort")
    print(input_dynamic_df)

    print("\ntarget_df before sort")
    print(target_df)

    input_dynamic_df = input_dynamic_df.sort_values("idx", ascending=True)
    target_df = target_df.sort_values("idx", ascending=True)

    print("\ninput_dynamic_df after sort")
    print(input_dynamic_df)

    print("\ntarget_df after sort")
    print(target_df)

    input_dynamic_df = input_dynamic_df.drop("idx", axis=1)
    target_df = target_df.drop("idx", axis=1)


    return input_cols,target_cols,input_dynamic_df,target_df


def get_single_input_and_output_dfs(arg_tuple):

    df,idx, train_tuple, num_train_tuples, static_fts,dynamic_fts,target_fts,DEBUG_PRINT, time_col = arg_tuple

    def debug_print(print_val="\n"):
        if DEBUG_PRINT == True:
            print(str(print_val))

    if idx%1000 == 0:
        print("Getting sample df %d of %d"%(idx+1, num_train_tuples))

    #================================ make input df ==============================================
    #all input dfs go here
    debug_print(train_tuple)

    #get dates
    train_input_start = train_tuple[0]
    train_input_end = train_tuple[1]
    train_output_start = train_tuple[2]
    train_output_end = train_tuple[3]

    #input fts
    train_input_start=pd.to_datetime(train_input_start, utc=True)
    train_input_end=pd.to_datetime(train_input_end, utc=True)
    temp = bu.config_df_by_dates(df,train_input_start,train_input_end,time_col=time_col)

    #get fts
    static_df = temp[static_fts].drop_duplicates().reset_index(drop=True)
    dynamic_df = temp[dynamic_fts]

    #setup df
    debug_print("\nstatic and dynamic fts")
    debug_print(static_df)
    timesteps = [i for i in range(1, dynamic_df.shape[0]+1)]
    dynamic_df["timestep"]=timesteps
    debug_print(dynamic_df)

    #idk
    all_temps = [static_df]
    static_cols = list(static_df)
    ts_dynamic_fts = list(static_cols)

    #insert time info
    data={}
    for timestep in timesteps:
        cur_dyn_df = dynamic_df[dynamic_df["timestep"]==timestep].reset_index(drop=True)
        for dynamic_ft in dynamic_fts:
            new_ft = "ts_" + str(timestep) + "_" + dynamic_ft
            cur_dyn_df = cur_dyn_df.rename(columns={dynamic_ft:new_ft})
            ts_dynamic_fts.append(new_ft)
            data[new_ft] = list(cur_dyn_df[new_ft])
        debug_print(cur_dyn_df)


    #df with input info
    input_dynamic_df = pd.DataFrame(data=data)
    for s in static_cols:
        input_dynamic_df[s] = static_df[s]
    input_dynamic_df = input_dynamic_df[ts_dynamic_fts]
    debug_print()
    debug_print(input_dynamic_df)


    #================================ make output df ==============================================
    train_output_start=pd.to_datetime(train_output_start, utc=True)
    train_output_end=pd.to_datetime(train_output_end, utc=True)
    temp = bu.config_df_by_dates(df,train_output_start,train_output_end,time_col=time_col)
    temp = temp[target_fts]
    timesteps = [i for i in range(1, temp.shape[0]+1)]
    temp["timestep"]=timesteps

    #time info
    all_temps = []
    ts_dynamic_fts = []
    for timestep in timesteps:
        cur_dyn_df = temp[temp["timestep"]==timestep].reset_index(drop=True)
        for target_ft in target_fts:
            new_target_ft = "ts_" + str(timestep) + "_" + target_ft
            cur_dyn_df = cur_dyn_df.rename(columns={target_ft:new_target_ft})
            ts_dynamic_fts.append(new_target_ft)
        all_temps.append(cur_dyn_df)
    target_df = pd.concat(all_temps, axis=1)
    # print(cur_dynamic_df)
    debug_print()
    debug_print(target_df)

    #get target df
    target_df = target_df.drop("timestep", axis=1)

    #cols
    input_cols = list(input_dynamic_df)
    target_cols = list(target_df )

    #make a better name
    input_df = input_dynamic_df

    input_df["idx"] = idx
    target_df["idx"] = idx

    if idx%1000 == 0:
        print("Done getting sample df %d of %d"%(idx+1, num_train_tuples))


    return  idx,input_cols,target_cols,input_df,target_df



def get_sample_dfs(df, train_date_tuples ,static_fts,dynamic_fts,target_fts,DEBUG_PRINT, time_col):

    def debug_print(print_val="\n"):
        if DEBUG_PRINT == True:
            print(str(print_val))




    #all input dfs go here
    all_input_dynamic_dfs = []
    all_target_dfs = []

    for train_tuple in train_date_tuples:
        debug_print(train_tuple)

        #get dates
        train_input_start = train_tuple[0]
        train_input_end = train_tuple[1]
        train_output_start = train_tuple[2]
        train_output_end = train_tuple[3]

        #input fts
        train_input_start=pd.to_datetime(train_input_start, utc=True)
        train_input_end=pd.to_datetime(train_input_end, utc=True)
        temp = bu.config_df_by_dates(df,train_input_start,train_input_end,time_col=time_col)

        #get fts
        static_df = temp[static_fts].drop_duplicates().reset_index(drop=True)
        dynamic_df = temp[dynamic_fts]

        print("\nstatic and dynamic fts")
        print(static_df)
        timesteps = [i for i in range(1, dynamic_df.shape[0]+1)]
        dynamic_df["timestep"]=timesteps
        print(dynamic_df)


        all_temps = [static_df]
        static_cols = list(static_df)
        ts_dynamic_fts = list(static_cols)

        data={}
        for timestep in timesteps:
            cur_dyn_df = dynamic_df[dynamic_df["timestep"]==timestep].reset_index(drop=True)
            for dynamic_ft in dynamic_fts:
                new_ft = "ts_" + str(timestep) + "_" + dynamic_ft
                cur_dyn_df = cur_dyn_df.rename(columns={dynamic_ft:new_ft})
                ts_dynamic_fts.append(new_ft)
                data[new_ft] = list(cur_dyn_df[new_ft])
            debug_print(cur_dyn_df)



        cur_dynamic_df = pd.DataFrame(data=data)

        for s in static_cols:
            cur_dynamic_df[s] = static_df[s]

        cur_dynamic_df = cur_dynamic_df[ts_dynamic_fts]

        # print()
        # print(cur_dynamic_df)

        all_input_dynamic_dfs.append(cur_dynamic_df)


        #output fts
        train_output_start=pd.to_datetime(train_output_start, utc=True)
        train_output_end=pd.to_datetime(train_output_end, utc=True)
        temp = bu.config_df_by_dates(df,train_output_start,train_output_end,time_col=time_col)
        temp = temp[target_fts]
        timesteps = [i for i in range(1, temp.shape[0]+1)]
        temp["timestep"]=timesteps
        # print(temp)
        # sys.exit(0)

        all_temps = []
        ts_dynamic_fts = []
        for timestep in timesteps:
            cur_dyn_df = temp[temp["timestep"]==timestep].reset_index(drop=True)
            for target_ft in target_fts:
                new_target_ft = "ts_" + str(timestep) + "_" + target_ft
                cur_dyn_df = cur_dyn_df.rename(columns={target_ft:new_target_ft})
                ts_dynamic_fts.append(new_target_ft)
            all_temps.append(cur_dyn_df)
        cur_dynamic_df = pd.concat(all_temps, axis=1)
        # print(cur_dynamic_df)

        all_target_dfs.append(cur_dynamic_df)

    input_dynamic_df = pd.concat(all_input_dynamic_dfs).reset_index(drop=True)

    # infoID_to_df_dict[infoID]["xdf"]=input_dynamic_df


    target_df = pd.concat(all_target_dfs).reset_index(drop=True)
    target_df = target_df.drop("timestep", axis=1)
    # infoID_to_df_dict[infoID]["ydf"]=target_df

    # print("\ntarget_df")
    # print(target_df
    #

    input_cols = list(input_dynamic_df)
    target_cols = list(target_df )


    return  input_cols,target_cols,input_dynamic_df,target_df

def cp5_get_infoID_df(arg_tuple):

    df, infoID, infoID_idx,num_infoIDs,train_date_tuples ,static_fts,dynamic_fts,target_fts,DEBUG_PRINT = arg_tuple

    def debug_print(print_val="\n"):
        if DEBUG_PRINT == True:
            print(str(print_val))


    print("\nGetting data for infoID %d of %d"%(infoID_idx+1, num_infoIDs))

    # infoID_to_df_dict[infoID]={}
    all_input_dynamic_dfs = []
    all_target_dfs = []

    for train_tuple in train_date_tuples:
        debug_print(train_tuple)
        train_input_start = train_tuple[0]
        train_input_end = train_tuple[1]
        train_output_start = train_tuple[2]
        train_output_end = train_tuple[3]

        #input fts
        # train_input_start=pd.to_datetime(train_input_start, utc=True)
        # train_input_end=pd.to_datetime(train_input_end, utc=True)
        temp = bu.config_df_by_dates(df,train_input_start,train_input_end,time_col="nodeTime")
        # print()
        static_df = temp[static_fts].drop_duplicates().reset_index(drop=True)
        dynamic_df = temp[dynamic_fts]
        # print(static_df)
        timesteps = [i for i in range(1, dynamic_df.shape[0]+1)]
        dynamic_df["timestep"]=timesteps
        # print(dynamic_df)


        all_temps = [static_df]
        static_cols = list(static_df)
        ts_dynamic_fts = list(static_cols)

        data={}
        for timestep in timesteps:
            cur_dyn_df = dynamic_df[dynamic_df["timestep"]==timestep].reset_index(drop=True)
            for dynamic_ft in dynamic_fts:
                new_ft = "ts_" + str(timestep) + "_" + dynamic_ft
                cur_dyn_df = cur_dyn_df.rename(columns={dynamic_ft:new_ft})
                ts_dynamic_fts.append(new_ft)
                data[new_ft] = list(cur_dyn_df[new_ft])
            debug_print(cur_dyn_df)



        cur_dynamic_df = pd.DataFrame(data=data)

        for s in static_cols:
            cur_dynamic_df[s] = static_df[s]

        cur_dynamic_df = cur_dynamic_df[ts_dynamic_fts]

        # print()
        # print(cur_dynamic_df)

        all_input_dynamic_dfs.append(cur_dynamic_df)


        #output fts
        temp = bu.config_df_by_dates(df,train_output_start,train_output_end,time_col="nodeTime")
        temp = temp[target_fts]
        timesteps = [i for i in range(1, temp.shape[0]+1)]
        temp["timestep"]=timesteps
        # print(temp)
        # sys.exit(0)

        all_temps = []
        ts_dynamic_fts = []
        for timestep in timesteps:
            cur_dyn_df = temp[temp["timestep"]==timestep].reset_index(drop=True)
            for target_ft in target_fts:
                new_target_ft = "ts_" + str(timestep) + "_" + target_ft
                cur_dyn_df = cur_dyn_df.rename(columns={target_ft:new_target_ft})
                ts_dynamic_fts.append(new_target_ft)
            all_temps.append(cur_dyn_df)
        cur_dynamic_df = pd.concat(all_temps, axis=1)
        # print(cur_dynamic_df)

        all_target_dfs.append(cur_dynamic_df)

    input_dynamic_df = pd.concat(all_input_dynamic_dfs).reset_index(drop=True)

    # infoID_to_df_dict[infoID]["xdf"]=input_dynamic_df


    target_df = pd.concat(all_target_dfs).reset_index(drop=True)
    target_df = target_df.drop("timestep", axis=1)
    # infoID_to_df_dict[infoID]["ydf"]=target_df

    # print("\ntarget_df")
    # print(target_df
    #

    input_cols = list(input_dynamic_df)
    target_cols = list(target_df )

    print("\nDone getting data for infoID %d of %d"%(infoID_idx+1, num_infoIDs))


    return  infoID,input_cols,target_cols,input_dynamic_df,target_df

def get_transformed_crypto_data_df_para(df, train_date_tuples,static_fts,dynamic_fts,target_fts,DEBUG_PRINT=True):
    print("\ntransforming data...")

    fixed_train_tuples = []
    for t in train_date_tuples:
        train_input_start = t[0]
        train_input_end = t[1]
        train_output_start =t[2]
        train_output_end =t[3]
        train_input_start=pd.to_datetime(train_input_start, utc=True)
        train_input_end=pd.to_datetime(train_input_end, utc=True)
        train_output_start=pd.to_datetime(train_output_start, utc=True)
        train_output_end=pd.to_datetime(train_output_end, utc=True)
        cur_tuple = (train_input_start,train_input_end ,train_output_start,train_output_end )
        fixed_train_tuples.append(cur_tuple)
    train_date_tuples = list(fixed_train_tuples)

    infoID_output_materials=[]
    print("\nRunning MP...")
    pool = mp.Pool(processes= len(infoIDs))
    arg_tuples = []

    for infoID_idx,infoID in enumerate(infoIDs):
        df = df_dict[infoID]
        arg_tuple = (df, infoID,infoID_idx,len(infoIDs),train_date_tuples ,static_fts,dynamic_fts,target_fts,DEBUG_PRINT)
        arg_tuples.append(arg_tuple)

    print("\nLaunching parallel func...")
    infoID_output_materials = pool.map(cp5_get_infoID_df, arg_tuples)
    pool.close()

    infoID_to_df_dict = {}

    for infoID_output_material_tuple in infoID_output_materials:
        infoID,input_cols,target_cols,input_dynamic_df,target_df = infoID_output_material_tuple
        infoID_to_df_dict[infoID] = {}
        infoID_to_df_dict[infoID]["xdf"]=input_dynamic_df
        infoID_to_df_dict[infoID]["ydf"]=target_df

        # for infoID in infoIDs:
    #   infoID_to_df_dict[infoID]



    idx_to_input_ft_dict = {}
    input_ft_to_idx_dict = {}

    idx_to_target_ft_dict = {}
    target_ft_to_idx_dict={}

    for i,input_col in enumerate(input_cols):
        idx_to_input_ft_dict[i]=input_col
        input_ft_to_idx_dict[input_col]=i



    for i,target_col in enumerate(target_cols):
        idx_to_target_ft_dict[i]=target_col
        target_ft_to_idx_dict[target_col]=i



    return infoID_to_df_dict, input_cols, target_cols ,idx_to_input_ft_dict, input_ft_to_idx_dict,idx_to_target_ft_dict,target_ft_to_idx_dict


def get_train_date_sample_tuples_v3_flexible(overall_start, test_output_start,  LOOKBACK_FACTOR, OUTPUT_SIZE, GRAN ,BASIC_GRAN , GRAN_AMOUNT_INT,SLIDE_SIZE=1):

    #train_output_end = pd.to_datetime(test_output_start,utc=True )- pd.DateOffset(days=(1))

    if BASIC_GRAN== "D":
        gran_str = "days"
    elif BASIC_GRAN== "H":
        gran_str = "hours"
    elif BASIC_GRAN== "min":
        gran_str = "minutes"
    else:
        print("\nError! Bad gran string!")
        print(GRAN)
        sys.exit(0)

    # if "D" in GRAN:
    #     gran_str = "days"
    # elif "min" in GRAN:
    #     gran_str = "minutes"
    # elif "H" in GRAN:
    #     gran_str = "hours"
    # else:
    #     print("\nin get_test_date_sample_tuples in crypto_sample_gen_utils -> gran is invalid!")
    #     print(GRAN)
    #     sys.exit(0)

    train_output_end = pd.to_datetime(test_output_start,utc=True )- pd.DateOffset(**{gran_str:GRAN_AMOUNT_INT})
    print("\ntrain_output_end")
    print(train_output_end)

    train_output_end = str(train_output_end).split(" ")[0]


    return get_train_date_samples_v2_proper_slide(overall_start, train_output_end,LOOKBACK_FACTOR,OUTPUT_SIZE,SLIDE_SIZE, GRAN=GRAN)



def get_train_date_sample_tuples_v2_clean(overall_start, test_output_start,  LOOKBACK_FACTOR, OUTPUT_SIZE,  SLIDE_SIZE=1, GRAN="D"):

    #train_output_end = pd.to_datetime(test_output_start,utc=True )- pd.DateOffset(days=(1))

    # if GRAN == "D":
    #     gran_str = "days"
    # elif GRAN == "H":
    #     gran_str = "hours"
    # elif GRAN == "min":
    #     gran_str = "minutes"
    # else:
    #     print("\nError! Bad gran string!")
    #     print(GRAN)
    #     sys.exit(0)

    if "D" in GRAN:
        gran_str = "days"
    elif "min" in GRAN:
        gran_str = "minutes"
    elif "H" in GRAN:
        gran_str = "hours"
    else:
        print("\nin get_test_date_sample_tuples in crypto_sample_gen_utils -> gran is invalid!")
        print(GRAN)
        sys.exit(0)

    train_output_end = pd.to_datetime(test_output_start,utc=True )- pd.DateOffset(**{gran_str:1})
    print("\ntrain_output_end")
    print(train_output_end)

    train_output_end = str(train_output_end).split(" ")[0]


    return get_train_date_samples_v2_proper_slide(overall_start, train_output_end,LOOKBACK_FACTOR,OUTPUT_SIZE,SLIDE_SIZE, GRAN=GRAN)


def get_train_date_sample_tuples_v2_clean_BACKUP(overall_start, test_output_start,  LOOKBACK_FACTOR, OUTPUT_SIZE, SLIDE_SIZE=1):

    #train_output_end = pd.to_datetime(test_output_start,utc=True )- pd.DateOffset(days=(1))

    train_output_end = pd.to_datetime(test_output_start,utc=True )- pd.DateOffset(days=(1))
    print("\ntrain_output_end")
    print(train_output_end)

    train_output_end = str(train_output_end).split(" ")[0]


    return get_train_date_samples_v2_proper_slide(overall_start, train_output_end,LOOKBACK_FACTOR,OUTPUT_SIZE,SLIDE_SIZE)


def get_train_date_sample_tuples(test_output_start, overall_start, LOOKBACK_FACTOR, OUTPUT_SIZE, SLIDE_SIZE=1):

    train_output_end = pd.to_datetime(test_output_start,utc=True )- pd.DateOffset(days=(1))
    print("\ntrain_output_end")
    print(train_output_end)

    train_output_end = str(train_output_end).split(" ")[0]


    return get_train_date_samples_v2_proper_slide(overall_start, train_output_end,LOOKBACK_FACTOR,OUTPUT_SIZE,SLIDE_SIZE)




def get_test_date_sample_tuples_BACKUP(overall_start, test_output_start ,test_output_end, LOOKBACK_FACTOR,OUTPUT_SIZE, GRAN):

    #get num dates in test set
    test_dates = pd.date_range(test_output_start, test_output_end)
    num_test_dates = len(test_dates)


    #gran has to be daily
    if GRAN != "D":
        print(GRAN)
        print("\nError! GRAN must be daily")
        sys.exit(0)

    print("lookback and output")
    print(LOOKBACK_FACTOR)
    print(OUTPUT_SIZE)

    NUM_WEEKS = int(LOOKBACK_FACTOR/7)
    NUM_DAYS = int(LOOKBACK_FACTOR%7)
    print(NUM_WEEKS)
    print(NUM_DAYS)
    # sys.exit(0)

    dates = pd.date_range(overall_start, test_output_end, freq="D")
    num_dates = len(dates)

    #get num samples
    num_samples = int(num_test_dates/OUTPUT_SIZE)
    print("\nnum_samples")
    print(num_samples)

    sample_output_date_tuple_list = []

    cur_output_start = pd.to_datetime(test_output_start, utc=True)
    for i in range(num_samples):
        cur_output_end = cur_output_start + pd.DateOffset(days=(OUTPUT_SIZE-1))
        cur_tuple = (cur_output_start ,cur_output_end)
        print(cur_tuple)
        cur_output_start=cur_output_start + pd.DateOffset(days=(OUTPUT_SIZE))
        sample_output_date_tuple_list.append(cur_tuple)

    print()
    full_sample_tuples = []
    for i in range(num_samples):
        cur_output_tuple = sample_output_date_tuple_list[i]
        cur_input_start = cur_output_tuple[0] - pd.DateOffset(days=LOOKBACK_FACTOR)
        cur_input_end = cur_output_tuple[0]- pd.DateOffset(days=1)
        cur_tuple = (cur_input_start,cur_input_end )
        # print(cur_tuple)

        full_tuple = (cur_input_start,cur_input_end, cur_output_tuple[0],cur_output_tuple[1])
        print(full_tuple)
        full_sample_tuples.append(full_tuple)

    return full_sample_tuples




def get_test_date_sample_tuples_v2_flexible(overall_start, test_output_start ,test_output_end, LOOKBACK_FACTOR,OUTPUT_SIZE, GRAN, BASIC_GRAN, GRAN_AMOUNT_INT):

    if BASIC_GRAN== "D":
        gran_str = "days"
    elif BASIC_GRAN== "min":
        gran_str = "minutes"
    elif BASIC_GRAN== "H":
        gran_str == "hours"
    else:
        print("\nin get_test_date_sample_tuples in crypto_sample_gen_utils -> gran is invalid!")
        print(BASIC_GRAN)
        sys.exit(0)

    # if "D" in GRAN:
    #     gran_str = "days"
    # elif "min" in GRAN:
    #     gran_str = "minutes"
    # elif "H" in GRAN:
    #     gran_str = "hours"
    # else:
    #     print("\nin get_test_date_sample_tuples in crypto_sample_gen_utils -> gran is invalid!")
    #     print(GRAN)
    #     sys.exit(0)

    #get num dates in test set
    test_dates = pd.date_range(test_output_start, test_output_end, freq=GRAN)
    num_test_dates = len(test_dates)

    print("\nnum_num_test_dates")
    print(num_test_dates)

    # sys.exit(0)


    # #gran has to be daily
    # if GRAN != "D":
    #     print(GRAN)
    #     print("\nError! GRAN must be daily")
    #     sys.exit(0)

    print("lookback and output")
    print(LOOKBACK_FACTOR)
    print(OUTPUT_SIZE)

    # NUM_WEEKS = int(LOOKBACK_FACTOR/7)
    # NUM_DAYS = int(LOOKBACK_FACTOR%7)
    # print(NUM_WEEKS)
    # print(NUM_DAYS)
    # sys.exit(0)

    dates = pd.date_range(overall_start, test_output_end, freq=GRAN)
    num_dates = len(dates)

    #get num samples
    num_samples = int(num_test_dates/OUTPUT_SIZE)
    print("\nnum_samples")
    print(num_samples)

    sample_output_date_tuple_list = []

    cur_output_start = pd.to_datetime(test_output_start, utc=True)
    for i in range(num_samples):
        #cur_output_end = cur_output_start + pd.DateOffset(days=(OUTPUT_SIZE-1))
        cur_output_end = cur_output_start + pd.DateOffset(**{gran_str:((OUTPUT_SIZE-1)* GRAN_AMOUNT_INT)})
        cur_tuple = (cur_output_start ,cur_output_end)
        print(cur_tuple)
        #cur_output_start=cur_output_start + pd.DateOffset(days=(OUTPUT_SIZE))
        cur_output_start=cur_output_start + pd.DateOffset(**{gran_str:(OUTPUT_SIZE* GRAN_AMOUNT_INT)})
        sample_output_date_tuple_list.append(cur_tuple)

    print()
    full_sample_tuples = []
    for i in range(num_samples):
        cur_output_tuple = sample_output_date_tuple_list[i]
        #cur_input_start = cur_output_tuple[0] - pd.DateOffset(days=LOOKBACK_FACTOR)
        cur_input_start = cur_output_tuple[0] - pd.DateOffset(**{gran_str:(LOOKBACK_FACTOR * GRAN_AMOUNT_INT)})
        #cur_input_end = cur_output_tuple[0]- pd.DateOffset(days=1)
        cur_input_end = cur_output_tuple[0]- pd.DateOffset(**{gran_str:GRAN_AMOUNT_INT})
        cur_tuple = (cur_input_start,cur_input_end )
        # print(cur_tuple)

        full_tuple = (cur_input_start,cur_input_end, cur_output_tuple[0],cur_output_tuple[1])
        print(full_tuple)
        full_sample_tuples.append(full_tuple)

    return full_sample_tuples



def get_eval_date_sample_tuples(test_output_start ,test_output_end, LOOKBACK_FACTOR,OUTPUT_SIZE, GRAN):


    if "D" in GRAN:
        gran_str = "days"
    elif "min" in GRAN:
        gran_str = "minutes"
    elif "H" in GRAN:
        gran_str = "hours"
    else:
        print("\nin get_test_date_sample_tuples in crypto_sample_gen_utils -> gran is invalid!")
        print(GRAN)
        sys.exit(0)

    #get num dates in test set
    test_dates = pd.date_range(test_output_start, test_output_end, freq=GRAN)
    num_test_dates = len(test_dates)

    print("\nnum_num_test_dates")
    print(num_test_dates)



    print("lookback and output")
    print(LOOKBACK_FACTOR)
    print(OUTPUT_SIZE)

    # dates = pd.date_range(overall_start, test_output_end, freq=GRAN)
    # num_dates = len(dates)

    #get num samples
    num_samples = int(num_test_dates/OUTPUT_SIZE)
    print("\nnum_samples")
    print(num_samples)

    sample_output_date_tuple_list = []

    cur_output_start = pd.to_datetime(test_output_start, utc=True)
    for i in range(num_samples):
        #cur_output_end = cur_output_start + pd.DateOffset(days=(OUTPUT_SIZE-1))
        cur_output_end = cur_output_start + pd.DateOffset(**{gran_str:(OUTPUT_SIZE-1)})
        cur_tuple = (cur_output_start ,cur_output_end)
        print(cur_tuple)
        #cur_output_start=cur_output_start + pd.DateOffset(days=(OUTPUT_SIZE))
        cur_output_start=cur_output_start + pd.DateOffset(**{gran_str:OUTPUT_SIZE})
        sample_output_date_tuple_list.append(cur_tuple)

    print()
    full_sample_tuples = []
    for i in range(num_samples):
        cur_output_tuple = sample_output_date_tuple_list[i]
        #cur_input_start = cur_output_tuple[0] - pd.DateOffset(days=LOOKBACK_FACTOR)
        cur_input_start = cur_output_tuple[0] - pd.DateOffset(**{gran_str:LOOKBACK_FACTOR})
        #cur_input_end = cur_output_tuple[0]- pd.DateOffset(days=1)
        cur_input_end = cur_output_tuple[0]- pd.DateOffset(**{gran_str:1})
        cur_tuple = (cur_input_start,cur_input_end )
        # print(cur_tuple)

        full_tuple = (cur_input_start,cur_input_end, cur_output_tuple[0],cur_output_tuple[1])
        print(full_tuple)
        full_sample_tuples.append(full_tuple)

    return full_sample_tuples



def get_test_date_sample_tuples(overall_start, test_output_start ,test_output_end, LOOKBACK_FACTOR,OUTPUT_SIZE, GRAN):


    if "D" in GRAN:
        gran_str = "days"
    elif "min" in GRAN:
        gran_str = "minutes"
    elif "H" in GRAN:
        gran_str = "hours"
    else:
        print("\nin get_test_date_sample_tuples in crypto_sample_gen_utils -> gran is invalid!")
        print(GRAN)
        sys.exit(0)

    #get num dates in test set
    test_dates = pd.date_range(test_output_start, test_output_end, freq=GRAN)
    num_test_dates = len(test_dates)

    print("\nnum_num_test_dates")
    print(num_test_dates)

    # sys.exit(0)


    # #gran has to be daily
    # if GRAN != "D":
    #     print(GRAN)
    #     print("\nError! GRAN must be daily")
    #     sys.exit(0)

    print("lookback and output")
    print(LOOKBACK_FACTOR)
    print(OUTPUT_SIZE)

    # NUM_WEEKS = int(LOOKBACK_FACTOR/7)
    # NUM_DAYS = int(LOOKBACK_FACTOR%7)
    # print(NUM_WEEKS)
    # print(NUM_DAYS)
    # sys.exit(0)

    dates = pd.date_range(overall_start, test_output_end, freq=GRAN)
    num_dates = len(dates)

    #get num samples
    num_samples = int(num_test_dates/OUTPUT_SIZE)
    print("\nnum_samples")
    print(num_samples)

    sample_output_date_tuple_list = []

    cur_output_start = pd.to_datetime(test_output_start, utc=True)
    for i in range(num_samples):
        #cur_output_end = cur_output_start + pd.DateOffset(days=(OUTPUT_SIZE-1))
        cur_output_end = cur_output_start + pd.DateOffset(**{gran_str:(OUTPUT_SIZE-1)})
        cur_tuple = (cur_output_start ,cur_output_end)
        print(cur_tuple)
        #cur_output_start=cur_output_start + pd.DateOffset(days=(OUTPUT_SIZE))
        cur_output_start=cur_output_start + pd.DateOffset(**{gran_str:OUTPUT_SIZE})
        sample_output_date_tuple_list.append(cur_tuple)

    print()
    full_sample_tuples = []
    for i in range(num_samples):
        cur_output_tuple = sample_output_date_tuple_list[i]
        #cur_input_start = cur_output_tuple[0] - pd.DateOffset(days=LOOKBACK_FACTOR)
        cur_input_start = cur_output_tuple[0] - pd.DateOffset(**{gran_str:LOOKBACK_FACTOR})
        #cur_input_end = cur_output_tuple[0]- pd.DateOffset(days=1)
        cur_input_end = cur_output_tuple[0]- pd.DateOffset(**{gran_str:1})
        cur_tuple = (cur_input_start,cur_input_end )
        # print(cur_tuple)

        full_tuple = (cur_input_start,cur_input_end, cur_output_tuple[0],cur_output_tuple[1])
        print(full_tuple)
        full_sample_tuples.append(full_tuple)

    return full_sample_tuples

def create_param_df_from_dicts(sample_param_dict, model_param_dict):

    param_vals = []
    param_names = []
    param_types = []
    for s,val in sample_param_dict.items():
        param_names.append(s)
        param_vals.append(val)
        param_types.append("sample")

    for m,val in model_param_dict.items():
        param_names.append(m)
        param_vals.append(val)
        param_types.append("model")

    param_df = pd.DataFrame(data={"param":param_names, "value":param_vals, "type":param_types})
    param_df = param_df.sort_values(["type", "param"])

    return param_df

def cp5_model_simulate_v2_feed_in_EXO_GT(GRAN,test_date_tuples, exo_cats, exo_df,model,infoIDs,num_test_dates, LOOKBACK_FACTOR,OUTPUT_SIZE,test_infoID_to_array_dict, TWITTER_INPUT_LOGNORM_NUM,TWITTER_OUTPUT_LOGNORM_NUM,OTHER_INPUT_LOGNORM_NUM,
    OTHER_OUTPUT_LOGNORM_NUM,input_fts, target_fts,input_ft_to_idx_dict,target_ft_to_idx_dict, dynamic_cats, target_cats,GET_STATIC_FTS, ROUND_RESULTS=True):

    exo_df = exo_df[["nodeTime"] + exo_cats]
    exo_df["nodeTime"] = pd.to_datetime(exo_df["nodeTime"], utc=True)

    infoID_to_pred_dict = {}

    for infoID in infoIDs:
        print("\nPredicting %s..."%infoID)
        x_test = test_infoID_to_array_dict[infoID]["x"]
        x_init_condit = x_test[:1, :]
        print("\nx_test shape: %s"%str(x_test.shape))
        print("x_init_condit shape: %s"%str(x_init_condit.shape))

        #now simulate
        cur_x_test = x_init_condit.copy()

        #calculate num loops needed
        num_sim_loops = int(num_test_dates/OUTPUT_SIZE)
        print("\nnum_sim_loops: %d"%num_sim_loops)

        all_infoID_pred_dfs = []
        infoID_ts_counter = 0
        for sim_loop in range(1, num_sim_loops+1):

            print("\nOn sim loop %d of %d"%(sim_loop, num_sim_loops))

            cur_x_test = lognorm_x_data(cur_x_test, TWITTER_INPUT_LOGNORM_NUM,OTHER_INPUT_LOGNORM_NUM,input_fts, input_ft_to_idx_dict)
            print("\ncur_x_test shape: %s"%str(cur_x_test.shape))

            cur_y_pred = model.predict(cur_x_test)
            print("\ncur_y_pred shape: %s"%str(cur_y_pred.shape))


            cur_x_test =delognorm_x_data(cur_x_test, TWITTER_INPUT_LOGNORM_NUM,OTHER_INPUT_LOGNORM_NUM,input_fts, input_ft_to_idx_dict)
            cur_y_pred = delognorm_y_data(cur_y_pred,TWITTER_OUTPUT_LOGNORM_NUM,OTHER_OUTPUT_LOGNORM_NUM,target_fts,target_ft_to_idx_dict)
            cur_y_pred[cur_y_pred<0]=0

            cur_x_test=cur_x_test.flatten()
            cur_y_pred=cur_y_pred.flatten()

            cur_x_test_df = convert_single_array_to_temporal_df(cur_x_test,LOOKBACK_FACTOR,dynamic_cats, input_ft_to_idx_dict)
            cur_y_pred_df = convert_single_array_to_temporal_df(cur_y_pred,OUTPUT_SIZE,target_cats, target_ft_to_idx_dict)



            cur_timesteps = []
            for i in range(OUTPUT_SIZE):
                infoID_ts_counter+=1
                cur_timesteps.append(infoID_ts_counter)
            cur_y_pred_df["timestep"]=cur_timesteps
            all_infoID_pred_dfs.append(cur_y_pred_df.copy())

            print("\ncur_x_test_df")
            print(cur_x_test_df)

            print("\ncur_y_pred_df")
            print(cur_y_pred_df)



            cur_y_pred_df = cur_y_pred_df.drop("timestep",axis=1)
            combined_data_df = pd.concat([cur_x_test_df,cur_y_pred_df]).reset_index(drop=True)
            print("\ncombined_data_df")
            print(combined_data_df)

            cur_x_tail_df = combined_data_df.tail(LOOKBACK_FACTOR).reset_index(drop=True)
            print("\ncur_x_tail_df")
            print(cur_x_tail_df)

            print("\ntest date tuples")
            for t in test_date_tuples:
                print(t)

            cur_test_date_tuple = test_date_tuples[sim_loop-1]
            print("\nOn sim loop %d of %d, so using test date tuple:\n%s"%(sim_loop, num_sim_loops, str(cur_test_date_tuple)))
            cur_exo_start = cur_test_date_tuple[0]
            cur_exo_end = cur_test_date_tuple[3]
            print("\nselected exo start and end")
            print(cur_exo_start)
            print(cur_exo_end)


            exo_dates = pd.date_range(cur_exo_start, cur_exo_end, freq=GRAN)
            exo_dates = exo_dates[OUTPUT_SIZE:]
            cur_x_tail_df["nodeTime"] = exo_dates
            cur_x_tail_df["nodeTime"] = pd.to_datetime(cur_x_tail_df["nodeTime"], utc=True)
            print(cur_x_tail_df)

            exo_tail_start = cur_x_tail_df["nodeTime"].iloc[0]
            exo_tail_end= cur_x_tail_df["nodeTime"].iloc[-1]

            cur_exo_df = bu.config_df_by_dates(exo_df, exo_tail_start, exo_tail_end, "nodeTime")
            print("\ncur_exo_df")
            print(cur_exo_df)
            exo_cols = list(cur_exo_df)
            exo_cols.remove("nodeTime")
            for col in exo_cols:
                cur_x_tail_df[col] = cur_exo_df[col].copy()

            print("\ncur_x_tail_df with exo gt fed in")
            print(cur_x_tail_df)

            cur_x_tail_df = cur_x_tail_df[dynamic_cats]


            #convert it back to an array
            cur_x_test = convert_temporal_df_back_to_single_array(cur_x_tail_df,infoID ,infoIDs,cur_x_tail_df.shape[0],dynamic_cats, input_ft_to_idx_dict,GET_STATIC_FTS )
            print("\nUpdated cur_x_test")
            print(cur_x_test)
            cur_x_test = cur_x_test.reshape(1, cur_x_test.shape[0])

        #make y pred df
        cur_infoID_y_pred_df = pd.concat(all_infoID_pred_dfs)
        cur_infoID_y_pred_df=cur_infoID_y_pred_df.sort_values("timestep").reset_index(drop=True)

        infoID_to_pred_dict[infoID]=cur_infoID_y_pred_df
        # sys.exit(0)


    if ROUND_RESULTS == True:
        for infoID in infoIDs:
            df = infoID_to_pred_dict[infoID]
            for target_cat in target_cats:
                df[target_cat] = np.round(df[target_cat], 0)
            infoID_to_pred_dict[infoID] = df

            print("\npred df")
            print(df)

    return infoID_to_pred_dict

def get_total_counts_v2_add_baseline(platform, target_rename_dict, output_dir, full_model_test_infoID_to_pred_df_dict,baseline_test_infoID_to_pred_df_dict,
    infoIDs, target_cats,challenge_model_tag, twitter_col_order, youtube_col_order, baseline_tag="Persistence_Baseline" ):

    if platform == "twitter":
        col_order =twitter_col_order
    else:
        col_order =youtube_col_order

    target_cat_count_dict = {}
    for target_cat in target_cats:
        new_tc = target_rename_dict[target_cat]
        target_cat_count_dict[new_tc] = []

    for infoID in infoIDs:
        cur_df = full_model_test_infoID_to_pred_df_dict[infoID]
        for target_cat in target_cats:
            new_tc = target_rename_dict[target_cat]
            cur_sum = cur_df[target_cat].sum()
            target_cat_count_dict[new_tc].append(cur_sum)

    df = pd.DataFrame(data=target_cat_count_dict)
    df["infoID"] = infoIDs

    df = df[["infoID"] + col_order]


    add_record = {"infoID":["total"]}
    for col in col_order:
        add_record[col] = [df[col].sum()]

    add_df = pd.DataFrame(data=add_record)
    df = df.append(add_df)

    print("\ntotal pred df")
    print(df)

    output_fp = output_dir + "Total-Counts.csv"
    df.to_csv(output_fp)

    return

def get_total_counts(platform, target_rename_dict, output_dir, full_model_test_infoID_to_pred_df_dict,
    infoIDs, target_cats,challenge_model_tag, twitter_col_order, youtube_col_order ):

    if platform == "twitter":
        col_order =twitter_col_order
    else:
        col_order =youtube_col_order

    target_cat_count_dict = {}
    for target_cat in target_cats:
        new_tc = target_rename_dict[target_cat]
        target_cat_count_dict[new_tc] = []

    for infoID in infoIDs:
        cur_df = full_model_test_infoID_to_pred_df_dict[infoID]
        for target_cat in target_cats:
            new_tc = target_rename_dict[target_cat]
            cur_sum = cur_df[target_cat].sum()
            target_cat_count_dict[new_tc].append(cur_sum)

    df = pd.DataFrame(data=target_cat_count_dict)
    df["infoID"] = infoIDs

    df = df[["infoID"] + col_order]


    add_record = {"infoID":["total"]}
    for col in col_order:
        add_record[col] = [df[col].sum()]

    add_df = pd.DataFrame(data=add_record)
    df = df.append(add_df)

    print("\ntotal pred df")
    print(df)

    output_fp = output_dir +challenge_model_tag +  "-total-counts.csv"
    df.to_csv(output_fp)

    return df

def save_table(df, title, output_dir,COL_WIDTH_COEFF= 0.5):
    fig, ax = plt.subplots()



    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')

    # num_cols = len(list(df))
    # colWidths = [1/( COL_WIDTH_COEFF* num_cols) for i in range(num_cols)]

    #df.style.set_caption("VAM Model Sample Info")
    ax.set_title(title)
    the_table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')



    # the_table.auto_set_font_size(False)
    # the_table.set_fontsize(24)

    ax.axis("off")

    # plt.figure(figsize=(20,10))

    # ax.set_title()
    # plt.title("VAM Model Sample Info")

    fig.tight_layout()

    fig.canvas.draw()
    # plt.savefig("price.png", bbox_inches="tight")
    # plt.savefig(output_dir + "%s-table.png"%title,bbox_inches="tight")
    # plt.savefig(output_dir + "%s-table.pdf"%title,bbox_inches="tight")

    png_output_dir = output_dir + "PNG/"
    create_output_dir(png_output_dir)

    svg_output_dir = output_dir + "SVG/"
    create_output_dir(svg_output_dir)

    pdf_output_dir = output_dir + "PDF/"
    create_output_dir(pdf_output_dir)

    plt.savefig(svg_output_dir + "%s-table.svg"%title)
    plt.savefig(pdf_output_dir + "%s-table.pdf"%title)
    plt.savefig(png_output_dir + "%s-table.png"%title)

def convert_df_2_cols_to_dict(df, key_col, val_col):
    return pd.Series(df[val_col].values,index=df[key_col]).to_dict()

def create_output_dir(output_dir):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(output_dir)
    else:
        print("%s already exists."%output_dir)

def plot_model_with_baseline_and_gt_v2_darpa(target_rename_dict,model_test_infoID_to_pred_df_dict,test_infoID_to_gt_df_dict,baseline_test_infoID_to_pred_df_dict, infoIDs, target_cats,comp_results_dict,metric_tags, output_dir):

    create_output_dir(output_dir)

    hyp_dict = hyphenate_infoID_dict(infoIDs)

    new_target_cats = []
    for target_cat in target_cats:
        new_target_cat = target_rename_dict[target_cat]
        new_target_cats.append(new_target_cat)

        for metric_tag in metric_tags:
            comp_results_dict[metric_tag][new_target_cat] = comp_results_dict[metric_tag].pop(target_cat)

        for infoID in infoIDs:

            model_test_infoID_to_pred_df_dict[infoID][new_target_cat] = model_test_infoID_to_pred_df_dict[infoID].pop(target_cat)
            baseline_test_infoID_to_pred_df_dict[infoID][new_target_cat] = baseline_test_infoID_to_pred_df_dict[infoID].pop(target_cat)
            test_infoID_to_gt_df_dict[infoID][new_target_cat] = test_infoID_to_gt_df_dict[infoID].pop(target_cat)

    target_cats=list(new_target_cats)

    for infoID in infoIDs:

        # fig, ax = plt.subplots()

        hyp_infoID = hyp_dict[infoID]

        num_rows = 3
        num_cols = 2
        fig, axs = plt.subplots(3, 2,figsize=(10,10))
        coord_list = [(0,0), (0,1), (1,0), (1,1), (2,0), (2,1)]

        model_coord_list = [(0,0), (1,0), (2,0)]
        baseline_coord_list = [(0,1), (1,1), (2,1)]

        # coord_list = [0, 1, 2]
        # fig, axs = plt.subplots(3,figsize=(8,8))
        target_cat_to_axis_coordinates_dict = {}

        for target_cat, coord in zip(target_cats,model_coord_list):
            print("%s: %s"%(target_cat, str(coord)))

            # cur_df = pd.DataFrame(data={"Prediction":y_pred, "Ground_Truth":y_ground_truth})

            cur_y_pred = model_test_infoID_to_pred_df_dict[infoID][target_cat]
            cur_y_ground_truth = test_infoID_to_gt_df_dict[infoID][target_cat]

            model_error_tag = "VAM - "
            for metric_tag in metric_tags:
                cur_infoID_df = comp_results_dict[metric_tag][target_cat]
                cur_infoID_df=cur_infoID_df[cur_infoID_df["infoID"]==infoID].reset_index(drop=True)
                model_error = cur_infoID_df["VAM_%s"%metric_tag].iloc[0]
                model_error_tag += metric_tag + ": %.4f; "%model_error

            title = infoID + "\n" + target_cat + "\n" + model_error_tag
            axs[coord].plot(cur_y_pred,"-r" ,label="VAM")
            axs[coord].plot(cur_y_ground_truth, "-k" ,label="Ground Truth")
            axs[coord].set_title(title)


        for target_cat,coord in zip(target_cats,baseline_coord_list ):
            print("%s: %s"%(target_cat, str(coord)))

            cur_y_ground_truth = test_infoID_to_gt_df_dict[infoID][target_cat]
            cur_y_baseline_pred =baseline_test_infoID_to_pred_df_dict[infoID][target_cat]

            baseline_error_tag = "PB - "
            for metric_tag in metric_tags:
                cur_infoID_df = comp_results_dict[metric_tag][target_cat]
                cur_infoID_df=cur_infoID_df[cur_infoID_df["infoID"]==infoID].reset_index(drop=True)
                baseline_error = cur_infoID_df["Persistence_Baseline_%s"%metric_tag].iloc[0]
                baseline_error_tag += metric_tag + ": %.4f; "%baseline_error

            title = infoID + "\n" + target_cat + "\n" + baseline_error_tag

            axs[coord].plot(cur_y_baseline_pred,"-g" ,label="Persistence_Baseline")
            axs[coord].plot(cur_y_ground_truth, "-k" ,label="Ground Truth")
            axs[coord].set_title(title)

        for ax in axs.flat:
            ax.set(xlabel='days', ylabel='Volume')

        plt.tight_layout()
        output_fp = output_dir + "%s.png"%( hyp_infoID)
        fig.savefig(output_fp)
        plt.close()
        print(output_fp)

    # for infoID in infoIDs:

    #   pred_df =



    return

def plot_model_with_baseline_only(model_test_infoID_to_pred_df_dict,baseline_test_infoID_to_pred_df_dict, infoIDs, target_cats,output_dir,model_tag, baseline_tag="Persistence_Baseline"):

    create_output_dir(output_dir)

    hyp_dict = hyphenate_infoID_dict(infoIDs)

    for infoID in infoIDs:

        # fig, ax = plt.subplots()

        hyp_infoID = hyp_dict[infoID]

        num_rows = 3
        num_cols = 1
        fig, axs = plt.subplots(num_rows, num_cols,figsize=(10,10))
        # coord_list = [(0,0), (0,1), (1,0), (1,1), (2,0), (2,1)]
        coord_list = [0, 1, 2 ]

        model_coord_list = [0, 1, 2]
        baseline_coord_list =model_coord_list
        # baseline_coord_list = [(0,1), (1,1), (2,1)]

        # coord_list = [0, 1, 2]
        # fig, axs = plt.subplots(3,figsize=(8,8))
        target_cat_to_axis_coordinates_dict = {}

        for target_cat, coord in zip(target_cats,model_coord_list):
            print("%s: %s"%(target_cat, str(coord)))

            # cur_df = pd.DataFrame(data={"Prediction":y_pred, "Ground_Truth":y_ground_truth})

            cur_y_pred = model_test_infoID_to_pred_df_dict[infoID][target_cat]
            # cur_y_ground_truth = test_infoID_to_gt_df_dict[infoID][target_cat]

            # model_error_tag = "VAM - "
            # for metric_tag in metric_tags:
            #     cur_infoID_df = comp_results_dict[metric_tag][target_cat]
            #     cur_infoID_df=cur_infoID_df[cur_infoID_df["infoID"]==infoID].reset_index(drop=True)
            #     model_error = cur_infoID_df["VAM_%s"%metric_tag].iloc[0]
            #     model_tag += metric_tag + ": %.4f; "%model_error

            # title = infoID + "\n" + target_cat + "\n" + model_tag
            axs[coord].plot(cur_y_pred,"-r" ,label=model_tag)
            # axs[coord].plot(cur_y_ground_truth, "-k" ,label="Ground Truth")
            # axs[coord].set_title(title)


        for target_cat,coord in zip(target_cats,baseline_coord_list ):
            print("%s: %s"%(target_cat, str(coord)))

            #cur_y_ground_truth = test_infoID_to_gt_df_dict[infoID][target_cat]
            cur_y_baseline_pred =baseline_test_infoID_to_pred_df_dict[infoID][target_cat]

            # baseline_error_tag = "PB - "
            # for metric_tag in metric_tags:
            #     cur_infoID_df = comp_results_dict[metric_tag][target_cat]
            #     cur_infoID_df=cur_infoID_df[cur_infoID_df["infoID"]==infoID].reset_index(drop=True)
            #     baseline_error = cur_infoID_df["Persistence_Baseline_%s"%metric_tag].iloc[0]
            #     baseline_error_tag += metric_tag + ": %.4f; "%baseline_error

            title = infoID + "\n" + target_cat + "\n" + model_tag + " vs. " + baseline_tag

            axs[coord].plot(cur_y_baseline_pred,"-g" ,label="Persistence_Baseline")
            # axs[coord].plot(cur_y_ground_truth, "-k" ,label="Ground Truth")
            axs[coord].set_title(title)

        for ax in axs.flat:
            ax.set(xlabel='days', ylabel='Volume')

        plt.legend()
        plt.tight_layout()

        output_fp = output_dir + "%s.png"%( hyp_infoID)
        fig.savefig(output_fp)
        plt.close()
        print(output_fp)

    # for infoID in infoIDs:

    #   pred_df =



    return


def plot_model_with_baseline_and_gt_v2_log_options(model_test_infoID_to_pred_df_dict,test_infoID_to_gt_df_dict,baseline_test_infoID_to_pred_df_dict, infoIDs, target_cats,comp_results_dict,metric_tags, output_dir,LOG=False):

    output_dir = output_dir + "LOG-%s/"%LOG
    create_output_dir(output_dir)

    hyp_dict = hyphenate_infoID_dict(infoIDs)

    for infoID in infoIDs:

        # fig, ax = plt.subplots()

        hyp_infoID = hyp_dict[infoID]

        num_rows = 3
        num_cols = 2
        fig, axs = plt.subplots(3, 2,figsize=(10,10))
        coord_list = [(0,0), (0,1), (1,0), (1,1), (2,0), (2,1)]

        model_coord_list = [(0,0), (1,0), (2,0)]
        baseline_coord_list = [(0,1), (1,1), (2,1)]

        # coord_list = [0, 1, 2]
        # fig, axs = plt.subplots(3,figsize=(8,8))
        target_cat_to_axis_coordinates_dict = {}

        for target_cat, coord in zip(target_cats,model_coord_list):
            print("%s: %s"%(target_cat, str(coord)))

            # cur_df = pd.DataFrame(data={"Prediction":y_pred, "Ground_Truth":y_ground_truth})

            cur_y_pred = model_test_infoID_to_pred_df_dict[infoID][target_cat]
            cur_y_ground_truth = test_infoID_to_gt_df_dict[infoID][target_cat]

            model_error_tag = "VAM - "
            for metric_tag in metric_tags:
                cur_infoID_df = comp_results_dict[metric_tag][target_cat]
                cur_infoID_df=cur_infoID_df[cur_infoID_df["infoID"]==infoID].reset_index(drop=True)
                model_error = cur_infoID_df["VAM_%s"%metric_tag].iloc[0]
                model_error_tag += metric_tag + ": %.4f;\n"%model_error

            title = infoID + "\n" + target_cat + "\n" + model_error_tag
            axs[coord].plot(cur_y_pred,"-r" ,label="VAM")
            axs[coord].plot(cur_y_ground_truth, "-k" ,label="Ground Truth")
            axs[coord].set_title(title)

            if LOG == True:
                axs[coord].set_yscale('log')


        for target_cat,coord in zip(target_cats,baseline_coord_list ):
            print("%s: %s"%(target_cat, str(coord)))

            cur_y_ground_truth = test_infoID_to_gt_df_dict[infoID][target_cat]
            cur_y_baseline_pred =baseline_test_infoID_to_pred_df_dict[infoID][target_cat]

            baseline_error_tag = "PB - "
            for metric_tag in metric_tags:
                cur_infoID_df = comp_results_dict[metric_tag][target_cat]
                cur_infoID_df=cur_infoID_df[cur_infoID_df["infoID"]==infoID].reset_index(drop=True)
                baseline_error = cur_infoID_df["Persistence_Baseline_%s"%metric_tag].iloc[0]
                baseline_error_tag += metric_tag + ": %.4f;\n"%baseline_error

            title = infoID + "\n" + target_cat + "\n" + baseline_error_tag

            axs[coord].plot(cur_y_baseline_pred,"-g" ,label="Persistence_Baseline")
            axs[coord].plot(cur_y_ground_truth, "-k" ,label="Ground Truth")
            axs[coord].set_title(title)

            if LOG == True:
                axs[coord].set_yscale('log')

        for ax in axs.flat:
            ax.set(xlabel='days', ylabel='Volume')

        plt.tight_layout()
        output_fp = output_dir + "%s.png"%( hyp_infoID)
        fig.savefig(output_fp)
        plt.close()
        print(output_fp)

    # for infoID in infoIDs:

    #   pred_df =



    return




# def plot_model_with_baseline_and_gt(model_test_infoID_to_pred_df_dict,baseline_test_infoID_to_pred_df_dict, infoIDs, target_cats,output_dir):

#     create_output_dir(output_dir)

#     hyp_dict = hyphenate_infoID_dict(infoIDs)

#     for infoID in infoIDs:

#         # fig, ax = plt.subplots()

#         hyp_infoID = hyp_dict[infoID]

#         num_rows = 3
#         num_cols = 1
#         fig, axs = plt.subplots(3, 1,figsize=(10,10))
#         coord_list = [(0,0), (0,1), (1,0), (1,1), (2,0), (2,1)]

#         model_coord_list = [(0,0), (1,0), (2,0)]
#         baseline_coord_list = [(0,1), (1,1), (2,1)]

#         # coord_list = [0, 1, 2]
#         # fig, axs = plt.subplots(3,figsize=(8,8))
#         target_cat_to_axis_coordinates_dict = {}

#         for target_cat, coord in zip(target_cats,model_coord_list):
#             print("%s: %s"%(target_cat, str(coord)))

#             # cur_df = pd.DataFrame(data={"Prediction":y_pred, "Ground_Truth":y_ground_truth})

#             cur_y_pred = model_test_infoID_to_pred_df_dict[infoID][target_cat]
#             cur_y_ground_truth = test_infoID_to_gt_df_dict[infoID][target_cat]

#             model_error_tag = "VAM - "
#             for metric_tag in metric_tags:
#                 cur_infoID_df = comp_results_dict[metric_tag][target_cat]
#                 cur_infoID_df=cur_infoID_df[cur_infoID_df["infoID"]==infoID].reset_index(drop=True)
#                 model_error = cur_infoID_df["VAM_%s"%metric_tag].iloc[0]
#                 model_error_tag += metric_tag + ": %.4f;\n"%model_error

#             title = infoID + "\n" + target_cat + "\n" + model_error_tag
#             axs[coord].plot(cur_y_pred,"-r" ,label="VAM")
#             axs[coord].plot(cur_y_ground_truth, "-k" ,label="Ground Truth")
#             axs[coord].set_title(title)


#         for target_cat,coord in zip(target_cats,baseline_coord_list ):
#             print("%s: %s"%(target_cat, str(coord)))

#             cur_y_ground_truth = test_infoID_to_gt_df_dict[infoID][target_cat]
#             cur_y_baseline_pred =baseline_test_infoID_to_pred_df_dict[infoID][target_cat]

#             baseline_error_tag = "PB - "
#             for metric_tag in metric_tags:
#                 cur_infoID_df = comp_results_dict[metric_tag][target_cat]
#                 cur_infoID_df=cur_infoID_df[cur_infoID_df["infoID"]==infoID].reset_index(drop=True)
#                 baseline_error = cur_infoID_df["Persistence_Baseline_%s"%metric_tag].iloc[0]
#                 baseline_error_tag += metric_tag + ": %.4f;\n"%baseline_error

#             title = infoID + "\n" + target_cat + "\n" + baseline_error_tag

#             axs[coord].plot(cur_y_baseline_pred,"-g" ,label="Persistence_Baseline")
#             axs[coord].plot(cur_y_ground_truth, "-k" ,label="Ground Truth")
#             axs[coord].set_title(title)

#         for ax in axs.flat:
#             ax.set(xlabel='days', ylabel='Volume')

#         plt.tight_layout()
#         output_fp = output_dir + "%s.png"%( hyp_infoID)
#         fig.savefig(output_fp)
#         plt.close()
#         print(output_fp)

#     # for infoID in infoIDs:

#     #   pred_df =



#     return





def plot_model_with_baseline_and_gt(model_test_infoID_to_pred_df_dict,test_infoID_to_gt_df_dict,baseline_test_infoID_to_pred_df_dict, infoIDs, target_cats,comp_results_dict,metric_tags, output_dir):

    create_output_dir(output_dir)

    hyp_dict = hyphenate_infoID_dict(infoIDs)

    for infoID in infoIDs:

        # fig, ax = plt.subplots()

        hyp_infoID = hyp_dict[infoID]

        num_rows = 3
        num_cols = 2
        fig, axs = plt.subplots(3, 2,figsize=(10,10))
        coord_list = [(0,0), (0,1), (1,0), (1,1), (2,0), (2,1)]

        model_coord_list = [(0,0), (1,0), (2,0)]
        baseline_coord_list = [(0,1), (1,1), (2,1)]

        # coord_list = [0, 1, 2]
        # fig, axs = plt.subplots(3,figsize=(8,8))
        target_cat_to_axis_coordinates_dict = {}

        for target_cat, coord in zip(target_cats,model_coord_list):
            print("%s: %s"%(target_cat, str(coord)))

            # cur_df = pd.DataFrame(data={"Prediction":y_pred, "Ground_Truth":y_ground_truth})

            cur_y_pred = model_test_infoID_to_pred_df_dict[infoID][target_cat]
            cur_y_ground_truth = test_infoID_to_gt_df_dict[infoID][target_cat]

            model_error_tag = "VAM - "
            for metric_tag in metric_tags:
                cur_infoID_df = comp_results_dict[metric_tag][target_cat]
                cur_infoID_df=cur_infoID_df[cur_infoID_df["infoID"]==infoID].reset_index(drop=True)
                model_error = cur_infoID_df["VAM_%s"%metric_tag].iloc[0]
                model_error_tag += metric_tag + ": %.4f;\n"%model_error

            title = infoID + "\n" + target_cat + "\n" + model_error_tag
            axs[coord].plot(cur_y_pred,"-r" ,label="VAM")
            axs[coord].plot(cur_y_ground_truth, "-k" ,label="Ground Truth")
            axs[coord].set_title(title)


        for target_cat,coord in zip(target_cats,baseline_coord_list ):
            print("%s: %s"%(target_cat, str(coord)))

            cur_y_ground_truth = test_infoID_to_gt_df_dict[infoID][target_cat]
            cur_y_baseline_pred =baseline_test_infoID_to_pred_df_dict[infoID][target_cat]

            baseline_error_tag = "PB - "
            for metric_tag in metric_tags:
                cur_infoID_df = comp_results_dict[metric_tag][target_cat]
                cur_infoID_df=cur_infoID_df[cur_infoID_df["infoID"]==infoID].reset_index(drop=True)
                baseline_error = cur_infoID_df["Persistence_Baseline_%s"%metric_tag].iloc[0]
                baseline_error_tag += metric_tag + ": %.4f;\n"%baseline_error

            title = infoID + "\n" + target_cat + "\n" + baseline_error_tag

            axs[coord].plot(cur_y_baseline_pred,"-g" ,label="Persistence_Baseline")
            axs[coord].plot(cur_y_ground_truth, "-k" ,label="Ground Truth")
            axs[coord].set_title(title)

        for ax in axs.flat:
            ax.set(xlabel='days', ylabel='Volume')

        plt.tight_layout()
        output_fp = output_dir + "%s.png"%( hyp_infoID)
        fig.savefig(output_fp)
        plt.close()
        print(output_fp)

    # for infoID in infoIDs:

    #   pred_df =



    return

def plot_model_with_baseline_and_gt_BACKUP(model_test_infoID_to_pred_df_dict,test_infoID_to_gt_df_dict,baseline_test_infoID_to_pred_df_dict, infoIDs, target_cats,comp_results_dict,metric_tags, output_dir):

    create_output_dir(output_dir)

    hyp_dict = hyphenate_infoID_dict(infoIDs)

    for infoID in infoIDs:

        # fig, ax = plt.subplots()

        hyp_infoID = hyp_dict[infoID]

        coord_list = [0, 1, 2]
        fig, axs = plt.subplots(3,figsize=(8,8))
        target_cat_to_axis_coordinates_dict = {}



        idx=0
        for target_cat,coord in zip(target_cats,coord_list ):
            target_cat_to_axis_coordinates_dict[target_cat] = coord
            print("%s: %s"%(target_cat, str(coord)))

            # cur_df = pd.DataFrame(data={"Prediction":y_pred, "Ground_Truth":y_ground_truth})

            cur_y_pred = model_test_infoID_to_pred_df_dict[infoID][target_cat]
            cur_y_ground_truth = test_infoID_to_gt_df_dict[infoID][target_cat]
            cur_y_baseline_pred =baseline_test_infoID_to_pred_df_dict[infoID][target_cat]

            target_no_underscore = target_cat.replace("_"," ")
            # title_tag = "%s %s"%(infoID, target_no_underscore)
            # title = ("\n".join(wrap(title_tag, 20)))
            # title = '%s \n%s'%(infoID, target_cat)

            model_error_tag = "VAM - "
            baseline_error_tag = "PB - "
            for metric_tag in metric_tags:
                cur_infoID_df = comp_results_dict[metric_tag][target_cat]
                cur_infoID_df=cur_infoID_df[cur_infoID_df["infoID"]==infoID].reset_index(drop=True)
                model_error = cur_infoID_df["VAM_%s"%metric_tag].iloc[0]
                baseline_error = cur_infoID_df["Persistence_Baseline_%s"%metric_tag].iloc[0]

                model_error_tag += metric_tag + ": %.4f; "%model_error
                baseline_error_tag += metric_tag + ": %.4f; "%baseline_error

            title = infoID + "\n" + target_cat + "\n" + model_error_tag + "\n" + baseline_error_tag

            # x_coor = coord[0]
            # y_coor = coord[1]
            print("\ncur_y_pred shape")
            print(cur_y_pred.shape)
            print("\ncur_y_ground_truth shape")
            print(cur_y_ground_truth.shape)
            axs[coord].plot(cur_y_pred,"-r" ,label="VAM")
            axs[coord].plot(cur_y_baseline_pred,"-g" ,label="Persistence_Baseline")
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

    # for infoID in infoIDs:

    #   pred_df =



    return

def compare_model_and_baseline_results(model_metric_tag_to_result_dict,baseline_metric_tag_to_result_dict,infoIDs,metric_tags, target_cats):

    comp_results_dict = {}
    for metric_tag in metric_tags:
        comp_results_dict[metric_tag]={}
        for target_cat in target_cats:
            model_error_results = model_metric_tag_to_result_dict[metric_tag][target_cat]
            baseline_error_results = baseline_metric_tag_to_result_dict[metric_tag][target_cat]
            print("\nmodel_error_results")
            print(model_error_results)
            print("\nbaseline_error_results")
            print(baseline_error_results)

            model_error_results = model_error_results.rename(columns={metric_tag: "VAM_%s"%metric_tag})
            baseline_error_results = baseline_error_results.rename(columns={metric_tag: "Persistence_Baseline_%s"%metric_tag})
            combined_error_df = pd.merge(model_error_results, baseline_error_results, on="infoID", how="inner").reset_index(drop=True)

            model_errors = list(combined_error_df["VAM_%s"%metric_tag])
            baseline_errors = list(combined_error_df["Persistence_Baseline_%s"%metric_tag])

            combined_error_df["VAM_is_winner"] = [1 if model_error <= baseline_error else 0 for model_error,baseline_error in zip(model_errors,baseline_errors)]
            print("\ncombined_error_df")
            print(combined_error_df)
            comp_results_dict[metric_tag][target_cat]=combined_error_df

    return comp_results_dict

def compare_model_and_baseline_results_v2_strict_win_count(model_metric_tag_to_result_dict,baseline_metric_tag_to_result_dict,infoIDs,metric_tags, target_cats):

    comp_results_dict = {}
    for metric_tag in metric_tags:
        comp_results_dict[metric_tag]={}
        for target_cat in target_cats:
            model_error_results = model_metric_tag_to_result_dict[metric_tag][target_cat]
            baseline_error_results = baseline_metric_tag_to_result_dict[metric_tag][target_cat]
            print("\nmodel_error_results")
            print(model_error_results)
            print("\nbaseline_error_results")
            print(baseline_error_results)

            model_error_results = model_error_results.rename(columns={metric_tag: "VAM_%s"%metric_tag})
            baseline_error_results = baseline_error_results.rename(columns={metric_tag: "Persistence_Baseline_%s"%metric_tag})
            combined_error_df = pd.merge(model_error_results, baseline_error_results, on="infoID", how="inner").reset_index(drop=True)

            model_errors = list(combined_error_df["VAM_%s"%metric_tag])
            baseline_errors = list(combined_error_df["Persistence_Baseline_%s"%metric_tag])

            combined_error_df["VAM_is_winner"] = [1 if model_error < baseline_error else 1 if (model_error==0) and (baseline_error==0) else 0 for model_error,baseline_error in zip(model_errors,baseline_errors)]
            print("\ncombined_error_df")
            print(combined_error_df)
            comp_results_dict[metric_tag][target_cat]=combined_error_df

    return comp_results_dict

def get_model_error_results_with_mult_error_funcs(metric_tags, metric_str_to_function_dict,target_cats ,model_test_infoID_to_pred_df_dict, test_infoID_to_gt_df_dict, infoIDs):

    model_metric_tag_to_result_dict = {}
    for metric_tag in metric_tags:
        error_func = metric_str_to_function_dict[metric_tag]
        model_metric_tag_to_result_dict[metric_tag] = get_model_error_results(target_cats ,model_test_infoID_to_pred_df_dict, test_infoID_to_gt_df_dict, infoIDs, error_func,metric_tag)


    return model_metric_tag_to_result_dict

def get_model_error_results(target_cats ,model_test_infoID_to_pred_df_dict, test_infoID_to_gt_df_dict, infoIDs, error_func,error_func_tag):

    target_cat_to_df_dict = {}


    for target_cat in target_cats:
        # target_cat_to_df_dict[target_cat] = []
        error_results = []
        for infoID in infoIDs:
            pred = model_test_infoID_to_pred_df_dict[infoID][target_cat]
            gt = test_infoID_to_gt_df_dict[infoID][target_cat]
            # error_result = error_func(pred, gt)
            error_result = error_func(gt,pred)
            error_results.append(error_result)

        df = pd.DataFrame(data={"infoID":infoIDs, error_func_tag:error_results})
        df = df[["infoID", error_func_tag]]
        target_cat_to_df_dict[target_cat]=df
        print()
        print(df)

    return target_cat_to_df_dict

def get_baseline_pred_dict(infoIDs,test_start,test_end,num_test_dates,ft_data_dir,target_cats,GRAN):

    #baseline_pred_dict = get_data_df_dict(infoIDs ,test_start,test_end,ft_data_dir ,static_fts,dynamic_fts,target_fts)
    num_test_dates = len(pd.date_range(test_start,test_end, freq=GRAN))

    baseline_start = test_start - pd.DateOffset(days=num_test_dates)
    baseline_end = test_start - pd.DateOffset(days=1)
    print(baseline_start)
    print(baseline_end)
    baseline_pred_dict = {}

    hyp_dict = hyphenate_infoID_dict(infoIDs)

    for infoID in infoIDs:
        hyp_infoID = hyp_dict[infoID]

        fp = ft_data_dir + hyp_infoID + "-features.csv"
        df = pd.read_csv(fp)
        # fts = list(set(static_fts+dynamic_fts+target_fts))
        df = df[["nodeTime"]+target_cats]
        df = config_df_by_dates(df,baseline_start,baseline_end,time_col="nodeTime")
        print()
        print(df)
        df=df.drop("nodeTime", axis=1)
        baseline_pred_dict[infoID]=df

    return baseline_pred_dict

def get_baseline_pred_dict_v3_EXO_HACK(infoIDs,test_start,test_end,num_test_dates,ft_data_dir,target_cats,GRAN, exo_df):

    #baseline_pred_dict = get_data_df_dict(infoIDs ,test_start,test_end,ft_data_dir ,static_fts,dynamic_fts,target_fts)
    num_test_dates = len(pd.date_range(test_start,test_end, freq=GRAN))

    test_start = pd.to_datetime(test_start, utc=True)
    test_end = pd.to_datetime(test_end, utc=True)

    baseline_start = test_start - pd.DateOffset(days=num_test_dates)
    baseline_end = test_start - pd.DateOffset(days=1)
    print(baseline_start)
    print(baseline_end)
    baseline_pred_dict = {}

    hyp_dict = hyphenate_infoID_dict(infoIDs)

    for infoID in infoIDs:
        hyp_infoID = hyp_dict[infoID]

        fp = ft_data_dir + hyp_infoID + "-features.csv"
        df = pd.read_csv(fp)
        # fts = list(set(static_fts+dynamic_fts+target_fts))
        df["nodeTime"] = pd.to_datetime(df["nodeTime"], utc=True)
        exo_df["nodeTime"] = pd.to_datetime(exo_df["nodeTime"], utc=True)
        df = pd.merge(df, exo_df , on="nodeTime", how="inner")

        df = df[["nodeTime"]+target_cats]
        df = config_df_by_dates(df,baseline_start,baseline_end,time_col="nodeTime")
        print()
        print(df)
        df=df.drop("nodeTime", axis=1)
        baseline_pred_dict[infoID]=df

    return baseline_pred_dict



def get_baseline_pred_dict_v2_fixed_dates(infoIDs,test_start,test_end,num_test_dates,ft_data_dir,target_cats,GRAN):

    #baseline_pred_dict = get_data_df_dict(infoIDs ,test_start,test_end,ft_data_dir ,static_fts,dynamic_fts,target_fts)
    num_test_dates = len(pd.date_range(test_start,test_end, freq=GRAN))

    test_start = pd.to_datetime(test_start, utc=True)
    test_end = pd.to_datetime(test_end, utc=True)

    baseline_start = test_start - pd.DateOffset(days=num_test_dates)
    baseline_end = test_start - pd.DateOffset(days=1)
    print(baseline_start)
    print(baseline_end)
    baseline_pred_dict = {}

    hyp_dict = hyphenate_infoID_dict(infoIDs)

    for infoID in infoIDs:
        hyp_infoID = hyp_dict[infoID]

        fp = ft_data_dir + hyp_infoID + "-features.csv"
        df = pd.read_csv(fp)
        # fts = list(set(static_fts+dynamic_fts+target_fts))
        df = df[["nodeTime"]+target_cats]
        df = config_df_by_dates(df,baseline_start,baseline_end,time_col="nodeTime")
        print()
        print(df)
        df=df.drop("nodeTime", axis=1)
        baseline_pred_dict[infoID]=df

    return baseline_pred_dict

def cp6_get_baseline_pred_dict_with_exo(infoIDs,test_start,test_end,num_test_dates,ft_data_dir,target_cats,GRAN, exo_ft_dir, exo_fts, internal_fts):

    #baseline_pred_dict = get_data_df_dict(infoIDs ,test_start,test_end,ft_data_dir ,static_fts,dynamic_fts,target_fts)
    num_test_dates = len(pd.date_range(test_start,test_end, freq=GRAN))

    test_start = pd.to_datetime(test_start, utc=True)
    test_end = pd.to_datetime(test_end, utc=True)

    baseline_start = test_start - pd.DateOffset(days=num_test_dates)
    baseline_end = test_start - pd.DateOffset(days=1)
    print(baseline_start)
    print(baseline_end)
    baseline_pred_dict = {}

    hyp_dict = hyphenate_infoID_dict(infoIDs)

    for infoID in infoIDs:
        hyp_infoID = hyp_dict[infoID]

        fp = ft_data_dir + hyp_infoID + "-features.csv"
        df = pd.read_csv(fp)
        # fts = list(set(static_fts+dynamic_fts+target_fts))
        df = df[["nodeTime"]+internal_fts]
        df = config_df_by_dates(df,baseline_start,baseline_end,time_col="nodeTime")
        print()
        print(df)


        #get exo fts
        exo_fp = exo_ft_dir + hyp_infoID + "-exo-features.csv"
        exo_df = pd.read_csv(exo_fp)
        exo_df = exo_df[["nodeTime"] + exo_fts]
        exo_df = config_df_by_dates(exo_df,baseline_start,baseline_end,time_col="nodeTime")
        print()
        print(exo_df)

        df = pd.merge(df, exo_df, on="nodeTime", how="inner").reset_index(drop=True)
        df = df.sort_values("nodeTime").reset_index(drop=True)

        dates = pd.date_range(baseline_start, baseline_end, freq=GRAN)
        num_dates = len(dates)

        print()
        print(df)

        if df.shape[0] != num_dates:
            print("\nError! %d != %d"%(df.shape[0], num_dates))
            sys.exit(0)



        df=df.drop("nodeTime", axis=1)
        baseline_pred_dict[infoID]=df

    return baseline_pred_dict

def plot_pred_vs_gt(test_infoID_to_pred_df_dict, test_infoID_to_gt_df_dict , infoIDs, output_dir, target_cats):


    # ax.plot(y_test,"-k",label="Ground Truth")
    # ax.plot(predictions,":r",label="Prediction")
    hyp_dict = hyphenate_infoID_dict(infoIDs)

    for infoID in infoIDs:

        # fig, ax = plt.subplots()

        hyp_infoID = hyp_dict[infoID]

        coord_list = [0, 1, 2]
        fig, axs = plt.subplots(3,figsize=(8,8))
        target_cat_to_axis_coordinates_dict = {}



        idx=0
        for target_cat,coord in zip(target_cats,coord_list ):
            target_cat_to_axis_coordinates_dict[target_cat] = coord
            print("%s: %s"%(target_cat, str(coord)))

            # cur_df = pd.DataFrame(data={"Prediction":y_pred, "Ground_Truth":y_ground_truth})

            cur_y_pred = test_infoID_to_pred_df_dict[infoID][target_cat]
            cur_y_ground_truth = test_infoID_to_gt_df_dict [infoID][target_cat]

            target_no_underscore = target_cat.replace("_"," ")
            title_tag = "%s %s"%(infoID, target_no_underscore)
            title = ("\n".join(wrap(title_tag, 20)))
            # title = '%s \n%s'%(infoID, target_cat)

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

    # for infoID in infoIDs:

    #   pred_df =

    return

def convert_array_dict_to_df_dict( tag,infoID_to_array_dict, infoIDs,temporal_cats, ft_to_idx_dict,num_desired_dates, OUTPUT_SIZE):

    num_dfs_to_combine = int(num_desired_dates/OUTPUT_SIZE)

    infoID_to_df_dict = {}
    for infoID in infoIDs:

        arrays = infoID_to_array_dict[infoID][tag]
        print("\n%s arrays shape: %s"%(tag,str(arrays.shape)))

        cur_df_list = []
        for array in arrays:
            cur_df = convert_single_array_to_temporal_df(array,OUTPUT_SIZE,temporal_cats, ft_to_idx_dict)
            cur_df_list.append(cur_df)

        infoID_df = pd.concat(cur_df_list).reset_index(drop=True)
        infoID_to_df_dict[infoID]=infoID_df
        print()
        print(infoID_df)

    return infoID_to_df_dict

def convert_single_array_to_temporal_df(array,num_timesteps,temporal_cats, ft_to_idx_dict):

    array_shape = array.shape
    array_shape_size = len(array_shape)

    if array_shape_size > 1:
        print(array_shape_size)
        print("\nError! array_shape_size > 1!")
        sys.exit(0)

    temporal_cat_to_time_series_dict = {}
    for temporal_cat in temporal_cats:
        temporal_cat_to_time_series_dict[temporal_cat] = []
        for timestep in range(1, num_timesteps+1):

                cur_ft = "ts_%d_%s"%(timestep, temporal_cat)
                ft_idx = ft_to_idx_dict[cur_ft]
                ft_val = array[ft_idx]
                temporal_cat_to_time_series_dict[temporal_cat].append(ft_val)

    df = pd.DataFrame(data=temporal_cat_to_time_series_dict)
    df = df[temporal_cats]
    return df

def convert_temporal_df_back_to_single_array(df,infoID_of_interest ,infoIDs,num_timesteps,temporal_cats, ft_to_idx_dict,GET_STATIC_FTS ):

    num_input_fts = len(list(ft_to_idx_dict.keys()))

    array_ft_vals = ["blah" for i in range(num_input_fts)]

    temporal_data_dict = {}
    for t in temporal_cats:
        temporal_data_dict[t] = df[t].values
        for timestep in range(1, num_timesteps+1):

            cur_ft = "ts_%d_%s"%(timestep, t)

            ft_idx = ft_to_idx_dict[cur_ft]

            ft_val = temporal_data_dict[t][timestep-1]

            array_ft_vals[ft_idx]=ft_val

    if GET_STATIC_FTS == True:
        for infoID in infoIDs:

            ft_idx =  ft_to_idx_dict[infoID]

            if infoID == infoID_of_interest:
                array_ft_vals[ft_idx]=1
            else:
                array_ft_vals[ft_idx]=0

    #check for mistake
    for val in array_ft_vals:
        if val=="blah":
            print("\nError! Null value in array vals")
            print(array_ft_vals)
            sys.exit(0)

    array_ft_vals=np.asarray(array_ft_vals)

    return array_ft_vals

def update_x_test_with_prediction(cur_x_test, cur_y_pred, LOOKBACK_FACTOR,OUTPUT_SIZE,test_infoID_to_array_dict, TWITTER_INPUT_LOGNORM_NUM,TWITTER_OUTPUT_LOGNORM_NUM,OTHER_INPUT_LOGNORM_NUM,OTHER_OUTPUT_LOGNORM_NUM,input_fts, target_fts,input_ft_to_idx_dict,target_ft_to_idx_dict):

    #make a df with x test


    return

def cp6_model_simulate_with_exo_fts(GRAN,test_date_tuples, model,infoIDs,num_test_dates, LOOKBACK_FACTOR,OUTPUT_SIZE,test_infoID_to_array_dict, TWITTER_INPUT_LOGNORM_NUM,TWITTER_OUTPUT_LOGNORM_NUM,OTHER_INPUT_LOGNORM_NUM,
    OTHER_OUTPUT_LOGNORM_NUM,input_fts, target_fts,input_ft_to_idx_dict,target_ft_to_idx_dict, dynamic_cats, target_cats,GET_STATIC_FTS, exo_ft_dir, exo_cats, ROUND_RESULTS=True):

    # exo_df = exo_df[["nodeTime"] + exo_cats]
    # exo_df["nodeTime"] = pd.to_datetime(exo_df["nodeTime"], utc=True)

    hyp_dict = hyphenate_infoID_dict(infoIDs)

    infoID_to_pred_dict = {}

    for infoID in infoIDs:

        hyp_infoID = hyp_dict[infoID]
        print("\nPredicting %s..."%infoID)
        x_test = test_infoID_to_array_dict[infoID]["x"]
        x_init_condit = x_test[:1, :]
        print("\nx_test shape: %s"%str(x_test.shape))
        print("x_init_condit shape: %s"%str(x_init_condit.shape))

        #get exo fts
        exo_fp = exo_ft_dir + hyp_infoID + "-exo-features.csv"
        exo_df = pd.read_csv(exo_fp)
        exo_df = exo_df[["nodeTime"] + exo_cats]
        exo_df["nodeTime"] = pd.to_datetime(exo_df["nodeTime"], utc=True)
        # exo_df = bu.config_df_by_dates(exo_df)

        #now simulate
        cur_x_test = x_init_condit.copy()

        #calculate num loops needed
        num_sim_loops = int(num_test_dates/OUTPUT_SIZE)
        print("\nnum_sim_loops: %d"%num_sim_loops)

        all_infoID_pred_dfs = []
        infoID_ts_counter = 0
        for sim_loop in range(1, num_sim_loops+1):

            print("\nOn sim loop %d of %d"%(sim_loop, num_sim_loops))

            cur_x_test = lognorm_x_data(cur_x_test, TWITTER_INPUT_LOGNORM_NUM,OTHER_INPUT_LOGNORM_NUM,input_fts, input_ft_to_idx_dict)
            print("\ncur_x_test shape: %s"%str(cur_x_test.shape))

            cur_y_pred = model.predict(cur_x_test)
            print("\ncur_y_pred shape: %s"%str(cur_y_pred.shape))


            cur_x_test =delognorm_x_data(cur_x_test, TWITTER_INPUT_LOGNORM_NUM,OTHER_INPUT_LOGNORM_NUM,input_fts, input_ft_to_idx_dict)
            cur_y_pred = delognorm_y_data(cur_y_pred,TWITTER_OUTPUT_LOGNORM_NUM,OTHER_OUTPUT_LOGNORM_NUM,target_fts,target_ft_to_idx_dict)
            cur_y_pred[cur_y_pred<0]=0

            cur_x_test=cur_x_test.flatten()
            cur_y_pred=cur_y_pred.flatten()

            cur_x_test_df = convert_single_array_to_temporal_df(cur_x_test,LOOKBACK_FACTOR,dynamic_cats, input_ft_to_idx_dict)
            cur_y_pred_df = convert_single_array_to_temporal_df(cur_y_pred,OUTPUT_SIZE,target_cats, target_ft_to_idx_dict)



            cur_timesteps = []
            for i in range(OUTPUT_SIZE):
                infoID_ts_counter+=1
                cur_timesteps.append(infoID_ts_counter)
            cur_y_pred_df["timestep"]=cur_timesteps
            all_infoID_pred_dfs.append(cur_y_pred_df.copy())

            print("\ncur_x_test_df")
            print(cur_x_test_df)

            print("\ncur_y_pred_df")
            print(cur_y_pred_df)



            cur_y_pred_df = cur_y_pred_df.drop("timestep",axis=1)
            combined_data_df = pd.concat([cur_x_test_df,cur_y_pred_df]).reset_index(drop=True)
            print("\ncombined_data_df")
            print(combined_data_df)

            cur_x_tail_df = combined_data_df.tail(LOOKBACK_FACTOR).reset_index(drop=True)
            print("\ncur_x_tail_df")
            print(cur_x_tail_df)

            print("\ntest date tuples")
            for t in test_date_tuples:
                print(t)

            cur_test_date_tuple = test_date_tuples[sim_loop-1]
            print("\nOn sim loop %d of %d, so using test date tuple:\n%s"%(sim_loop, num_sim_loops, str(cur_test_date_tuple)))
            cur_exo_start = cur_test_date_tuple[0]
            cur_exo_end = cur_test_date_tuple[3]
            print("\nselected exo start and end")
            print(cur_exo_start)
            print(cur_exo_end)


            exo_dates = pd.date_range(cur_exo_start, cur_exo_end, freq=GRAN)
            exo_dates = exo_dates[OUTPUT_SIZE:]
            cur_x_tail_df["nodeTime"] = exo_dates
            cur_x_tail_df["nodeTime"] = pd.to_datetime(cur_x_tail_df["nodeTime"], utc=True)
            print(cur_x_tail_df)

            exo_tail_start = cur_x_tail_df["nodeTime"].iloc[0]
            exo_tail_end= cur_x_tail_df["nodeTime"].iloc[-1]

            cur_exo_df = bu.config_df_by_dates(exo_df, exo_tail_start, exo_tail_end, "nodeTime")
            print("\ncur_exo_df")
            print(cur_exo_df)
            exo_cols = list(cur_exo_df)
            exo_cols.remove("nodeTime")
            for col in exo_cols:
                cur_x_tail_df[col] = cur_exo_df[col].copy()

            print("\ncur_x_tail_df with exo gt fed in")
            print(cur_x_tail_df)

            cur_x_tail_df = cur_x_tail_df[dynamic_cats]


            #convert it back to an array
            cur_x_test = convert_temporal_df_back_to_single_array(cur_x_tail_df,infoID ,infoIDs,cur_x_tail_df.shape[0],dynamic_cats, input_ft_to_idx_dict,GET_STATIC_FTS )
            print("\nUpdated cur_x_test")
            print(cur_x_test)
            cur_x_test = cur_x_test.reshape(1, cur_x_test.shape[0])

        #make y pred df
        cur_infoID_y_pred_df = pd.concat(all_infoID_pred_dfs)
        cur_infoID_y_pred_df=cur_infoID_y_pred_df.sort_values("timestep").reset_index(drop=True)

        infoID_to_pred_dict[infoID]=cur_infoID_y_pred_df
        # sys.exit(0)


    if ROUND_RESULTS == True:
        for infoID in infoIDs:
            df = infoID_to_pred_dict[infoID]
            for target_cat in target_cats:
                df[target_cat] = np.round(df[target_cat], 0)
            infoID_to_pred_dict[infoID] = df

            print("\npred df")
            print(df)

    return infoID_to_pred_dict

def cp5_model_simulate(model,infoIDs,num_test_dates, LOOKBACK_FACTOR,OUTPUT_SIZE,test_infoID_to_array_dict, TWITTER_INPUT_LOGNORM_NUM,TWITTER_OUTPUT_LOGNORM_NUM,OTHER_INPUT_LOGNORM_NUM,
    OTHER_OUTPUT_LOGNORM_NUM,input_fts, target_fts,input_ft_to_idx_dict,target_ft_to_idx_dict, dynamic_cats, target_cats,GET_STATIC_FTS, ROUND_RESULTS=True):

    infoID_to_pred_dict = {}

    for infoID in infoIDs:
        print("\nPredicting %s..."%infoID)
        x_test = test_infoID_to_array_dict[infoID]["x"]
        x_init_condit = x_test[:1, :]
        print("\nx_test shape: %s"%str(x_test.shape))
        print("x_init_condit shape: %s"%str(x_init_condit.shape))

        #now simulate
        cur_x_test = x_init_condit.copy()

        #calculate num loops needed
        num_sim_loops = int(num_test_dates/OUTPUT_SIZE)
        print("\nnum_sim_loops: %d"%num_sim_loops)

        all_infoID_pred_dfs = []
        infoID_ts_counter = 0
        for sim_loop in range(1, num_sim_loops+1):

            print("\nOn sim loop %d of %d"%(sim_loop, num_sim_loops))

            cur_x_test = lognorm_x_data(cur_x_test, TWITTER_INPUT_LOGNORM_NUM,OTHER_INPUT_LOGNORM_NUM,input_fts, input_ft_to_idx_dict)
            print("\ncur_x_test shape: %s"%str(cur_x_test.shape))

            cur_y_pred = model.predict(cur_x_test)
            print("\ncur_y_pred shape: %s"%str(cur_y_pred.shape))


            cur_x_test =delognorm_x_data(cur_x_test, TWITTER_INPUT_LOGNORM_NUM,OTHER_INPUT_LOGNORM_NUM,input_fts, input_ft_to_idx_dict)
            cur_y_pred = delognorm_y_data(cur_y_pred,TWITTER_OUTPUT_LOGNORM_NUM,OTHER_OUTPUT_LOGNORM_NUM,target_fts,target_ft_to_idx_dict)
            cur_y_pred[cur_y_pred<0]=0

            cur_x_test=cur_x_test.flatten()
            cur_y_pred=cur_y_pred.flatten()

            cur_x_test_df = convert_single_array_to_temporal_df(cur_x_test,LOOKBACK_FACTOR,dynamic_cats, input_ft_to_idx_dict)
            cur_y_pred_df = convert_single_array_to_temporal_df(cur_y_pred,OUTPUT_SIZE,target_cats, target_ft_to_idx_dict)



            cur_timesteps = []
            for i in range(OUTPUT_SIZE):
                infoID_ts_counter+=1
                cur_timesteps.append(infoID_ts_counter)
            cur_y_pred_df["timestep"]=cur_timesteps
            all_infoID_pred_dfs.append(cur_y_pred_df.copy())

            print("\ncur_x_test_df")
            print(cur_x_test_df)

            print("\ncur_y_pred_df")
            print(cur_y_pred_df)



            cur_y_pred_df = cur_y_pred_df.drop("timestep",axis=1)
            combined_data_df = pd.concat([cur_x_test_df,cur_y_pred_df]).reset_index(drop=True)
            print("\ncombined_data_df")
            print(combined_data_df)

            cur_x_tail_df = combined_data_df.tail(LOOKBACK_FACTOR).reset_index(drop=True)
            print("\ncur_x_tail_df")
            print(cur_x_tail_df)

            #convert it back to an array
            cur_x_test = convert_temporal_df_back_to_single_array(cur_x_tail_df,infoID ,infoIDs,cur_x_tail_df.shape[0],dynamic_cats, input_ft_to_idx_dict,GET_STATIC_FTS )
            print("\nUpdated cur_x_test")
            print(cur_x_test)
            cur_x_test = cur_x_test.reshape(1, cur_x_test.shape[0])

        #make y pred df
        cur_infoID_y_pred_df = pd.concat(all_infoID_pred_dfs)
        cur_infoID_y_pred_df=cur_infoID_y_pred_df.sort_values("timestep").reset_index(drop=True)

        infoID_to_pred_dict[infoID]=cur_infoID_y_pred_df
        # sys.exit(0)


    if ROUND_RESULTS == True:
        for infoID in infoIDs:
            df = infoID_to_pred_dict[infoID]
            for target_cat in target_cats:
                df[target_cat] = np.round(df[target_cat], 0)
            infoID_to_pred_dict[infoID] = df

            print("\npred df")
            print(df)

    return infoID_to_pred_dict

def save_list_as_txt_file(output_fp, my_list):
    with open(output_fp, "w") as f:
        for l in my_list:
            f.write(str(l) + "\n")
    print("Saved %s"%output_fp)
    return

def delognorm_x_data(x, TWITTER_INPUT_LOGNORM_NUM,OTHER_INPUT_LOGNORM_NUM,input_fts, input_ft_to_idx_dict):

    print("\nLog normalizing data...")
    x = x.T

    for input_ft in input_fts:
        idx = input_ft_to_idx_dict[input_ft]
        if "twitter" in input_ft:
            for L in range(TWITTER_INPUT_LOGNORM_NUM):
                x[idx]=np.expm1(x[idx])
        else:
            for L in range(OTHER_INPUT_LOGNORM_NUM):
                x[idx]=np.expm1(x[idx])

    x = x.T
    print("\nDone log normalizing data...")

    return x

def delognorm_y_data(y,TWITTER_OUTPUT_LOGNORM_NUM,OTHER_OUTPUT_LOGNORM_NUM,target_fts,target_ft_to_idx_dict):

    print("\nLog normalizing data...")
    y = y.T

    for target_ft in target_fts:
        idx = target_ft_to_idx_dict[target_ft]
        if "twitter" in target_ft:
            for L in range(TWITTER_OUTPUT_LOGNORM_NUM):
                y[idx]=np.expm1(y[idx])
        else:
            for L in range(OTHER_OUTPUT_LOGNORM_NUM):
                y[idx]=np.expm1(y[idx])

    y = y.T
    print("\nDone log normalizing data...")

    return y

def lognorm_x_data(x, TWITTER_INPUT_LOGNORM_NUM,OTHER_INPUT_LOGNORM_NUM,input_fts, input_ft_to_idx_dict):

    print("\nLog normalizing data...")
    x = x.T

    for input_ft in input_fts:
        idx = input_ft_to_idx_dict[input_ft]
        if "twitter" in input_ft:
            for L in range(TWITTER_INPUT_LOGNORM_NUM):
                x[idx]=np.log1p(x[idx])
        else:
            for L in range(OTHER_INPUT_LOGNORM_NUM):
                x[idx]=np.log1p(x[idx])

    x = x.T
    print("\nDone log normalizing data...")

    return x

def lognorm_y_data(y,TWITTER_OUTPUT_LOGNORM_NUM,OTHER_OUTPUT_LOGNORM_NUM,target_fts,target_ft_to_idx_dict):

    print("\nLog normalizing data...")
    y = y.T

    for target_ft in target_fts:
        idx = target_ft_to_idx_dict[target_ft]
        if "twitter" in target_ft:
            for L in range(TWITTER_OUTPUT_LOGNORM_NUM):
                y[idx]=np.log1p(y[idx])
        else:
            for L in range(OTHER_OUTPUT_LOGNORM_NUM):
                y[idx]=np.log1p(y[idx])

    y = y.T
    print("\nDone log normalizing data...")

    return y

def lognorm_data(x, y,TWITTER_INPUT_LOGNORM_NUM,TWITTER_OUTPUT_LOGNORM_NUM,OTHER_INPUT_LOGNORM_NUM,OTHER_OUTPUT_LOGNORM_NUM,input_fts, target_fts,input_ft_to_idx_dict,target_ft_to_idx_dict):

    print("\nLog normalizing data...")
    x = x.T
    y = y.T

    for input_ft in input_fts:
        idx = input_ft_to_idx_dict[input_ft]
        if "twitter" in input_ft:
            for L in range(TWITTER_INPUT_LOGNORM_NUM):
                x[idx]=np.log1p(x[idx])
        else:
            for L in range(OTHER_INPUT_LOGNORM_NUM):
                x[idx]=np.log1p(x[idx])

    for target_ft in target_fts:
        idx = target_ft_to_idx_dict[target_ft]
        if "twitter" in target_ft:
            for L in range(TWITTER_OUTPUT_LOGNORM_NUM):
                y[idx]=np.log1p(y[idx])
        else:
            for L in range(OTHER_OUTPUT_LOGNORM_NUM):
                y[idx]=np.log1p(y[idx])

    x = x.T
    y = y.T
    print("\nDone log normalizing data...")

    return x,y

def remove_unnamed(df):
    if "Unnamed: 0" in list(df):
        df = df.drop("Unnamed: 0", axis=1)
    return df

def save_pickle(data, output_fp):
    print("\nSaving to %s..."%output_fp)
    with open(output_fp, 'wb') as handle:
        pickle.dump(data, handle)
    print("Saved pickle!")

def load_data_from_pickle(data_fp, VERBOSE=True):

    if VERBOSE == True:
        print("\nGetting data from %s..."%data_fp)
    file = open(data_fp,'rb')
    data = pickle.load(file)
    if VERBOSE == True:
        print("Got data!")
    return data

def combine_arrays(infoIDs,infoID_to_array_dict):
    x_arrays = []
    y_arrays = []

    for infoID in infoIDs:
        x = infoID_to_array_dict[infoID]["x"]
        y = infoID_to_array_dict[infoID]["y"]
        x_arrays.append(x)
        y_arrays.append(y)

    x_arrays = np.concatenate(x_arrays, axis=0)
    y_arrays = np.concatenate(y_arrays, axis=0)

    return x_arrays,y_arrays

def convert_df_dict_to_array_dict(infoID_to_df_dict, infoIDs, tag,input_fts, target_fts):

    #infoID_to_df_dict[infoID]["xdf"]
    infoID_to_array_dict = {}
    for infoID in infoIDs:
        infoID_to_array_dict[infoID]={}
        xdf = infoID_to_df_dict[infoID]["xdf"]
        print(xdf)
        # sys.exit(0)
        ydf = infoID_to_df_dict[infoID]["ydf"]
        x = xdf[input_fts].values
        y = ydf[target_fts].values
        infoID_to_array_dict[infoID]["x"]=x
        infoID_to_array_dict[infoID]["y"]=y

        print("\nx and y %s shapes"%(tag))
        print(x.shape)
        print(y.shape)


    return infoID_to_array_dict

def convert_df_dict_to_array_dict_x_only(infoID_to_df_dict, infoIDs, tag,input_fts):

    #infoID_to_df_dict[infoID]["xdf"]
    infoID_to_array_dict = {}
    for infoID in infoIDs:
        infoID_to_array_dict[infoID]={}
        xdf = infoID_to_df_dict[infoID]["xdf"]
        print(xdf)
        x = xdf[input_fts].values
        infoID_to_array_dict[infoID]["x"]=x

        print("\nx %s shape"%(tag))
        print(x.shape)


    return infoID_to_array_dict

def merge_mult_dfs(merge_list, on, how, VERBOSE=True):
    if VERBOSE == True:
        print("\nMerging multiple dfs...")
    return reduce(lambda  left,right: pd.merge(left,right,on=on, how=how), merge_list)

def get_transformed_data_df_dict_v2_input_only(df_dict, infoIDs,train_date_tuples,static_fts,dynamic_fts):
    print("\ntransforming data...")

    fixed_train_tuples = []
    for t in train_date_tuples:
        train_input_start = t[0]
        train_input_end = t[1]
        # train_output_start =t[2]
        # train_output_end =t[3]
        train_input_start=pd.to_datetime(train_input_start, utc=True)
        train_input_end=pd.to_datetime(train_input_end, utc=True)
        # train_output_start=pd.to_datetime(train_output_start, utc=True)
        # train_output_end=pd.to_datetime(train_output_end, utc=True)
        #cur_tuple = (train_input_start,train_input_end ,train_output_start,train_output_end )
        cur_tuple = (train_input_start,train_input_end )
        fixed_train_tuples.append(cur_tuple)
    train_date_tuples = list(fixed_train_tuples)

    infoID_to_df_dict = {}
    for infoID in infoIDs:
        infoID_to_df_dict[infoID]={}
        df = df_dict[infoID]
        # df = config_df_by_dates(df,start,end,time_col="nodeTime")

        all_input_dynamic_dfs = []
        all_target_dfs = []

        for train_tuple in train_date_tuples:
            print(train_tuple)
            train_input_start = train_tuple[0]
            train_input_end = train_tuple[1]
            # train_output_start = train_tuple[2]
            # train_output_end = train_tuple[3]

            #input fts
            temp = config_df_by_dates(df,train_input_start,train_input_end,time_col="nodeTime")
            # print()
            static_df = temp[static_fts].drop_duplicates().reset_index(drop=True)
            dynamic_df = temp[dynamic_fts]
            # print(static_df)
            timesteps = [i for i in range(1, dynamic_df.shape[0]+1)]
            dynamic_df["timestep"]=timesteps
            # print(dynamic_df)


            all_temps = [static_df]
            static_cols = list(static_df)
            ts_dynamic_fts = list(static_cols)

            data={}
            for timestep in timesteps:
                cur_dyn_df = dynamic_df[dynamic_df["timestep"]==timestep].reset_index(drop=True)
                for dynamic_ft in dynamic_fts:
                    new_ft = "ts_" + str(timestep) + "_" + dynamic_ft
                    cur_dyn_df = cur_dyn_df.rename(columns={dynamic_ft:new_ft})
                    ts_dynamic_fts.append(new_ft)
                    data[new_ft] = list(cur_dyn_df[new_ft])
                print(cur_dyn_df)



            cur_dynamic_df = pd.DataFrame(data=data)
            # for cur_infoID in infoIDs:
            #   if cur_infoID==infoID:
            #       cur_dynamic_df[infoID]=1
            #   else:
            #       cur_dynamic_df[infoID]=0

            # print("\ncur_dynamic_df")
            # print(cur_dynamic_df)

            for s in static_cols:
                cur_dynamic_df[s] = static_df[s]

            cur_dynamic_df = cur_dynamic_df[ts_dynamic_fts]

            # print()
            # print(cur_dynamic_df)

            all_input_dynamic_dfs.append(cur_dynamic_df)


            # #output fts
            # temp = config_df_by_dates(df,train_output_start,train_output_end,time_col="nodeTime")
            # temp = temp[target_fts]
            # timesteps = [i for i in range(1, temp.shape[0]+1)]
            # temp["timestep"]=timesteps
            # # print(temp)
            # # sys.exit(0)

            # all_temps = []
            # ts_dynamic_fts = []
            # for timestep in timesteps:
            #     cur_dyn_df = temp[temp["timestep"]==timestep].reset_index(drop=True)
            #     for target_ft in target_fts:
            #         new_target_ft = "ts_" + str(timestep) + "_" + target_ft
            #         cur_dyn_df = cur_dyn_df.rename(columns={target_ft:new_target_ft})
            #         ts_dynamic_fts.append(new_target_ft)
            #     all_temps.append(cur_dyn_df)
            # cur_dynamic_df = pd.concat(all_temps, axis=1)
            # # print(cur_dynamic_df)

            # all_target_dfs.append(cur_dynamic_df)

        input_dynamic_df = pd.concat(all_input_dynamic_dfs).reset_index(drop=True)
        # print("\ninput_dynamic_df")
        # print(input_dynamic_df)
        #
        # input_dynamic_df = input_dynamic_df.drop("timestep", axis=1)
        # print("\ninput_dynamic_df")
        # print(input_dynamic_df)

        infoID_to_df_dict[infoID]["xdf"]=input_dynamic_df


        # target_df = pd.concat(all_target_dfs).reset_index(drop=True)
        # target_df = target_df.drop("timestep", axis=1)
        # infoID_to_df_dict[infoID]["ydf"]=target_df
        # print("\ntarget_df")
        # print(target_df)

        #

        input_cols = list(input_dynamic_df)
        # target_cols = list(target_df )
        # print("\ninput_cols")
        # for col in input_cols:
        #     print(col)
        # print("\ntarget_cols")
        # for col in target_cols:
        #     print(col)



    idx_to_input_ft_dict = {}
    input_ft_to_idx_dict = {}

    idx_to_target_ft_dict = {}
    target_ft_to_idx_dict={}

    for i,input_col in enumerate(input_cols):
        idx_to_input_ft_dict[i]=input_col
        input_ft_to_idx_dict[input_col]=i



    # for i,target_col in enumerate(target_cols):
    #     idx_to_target_ft_dict[i]=target_col
    #     target_ft_to_idx_dict[target_col]=i



    return infoID_to_df_dict, input_cols, idx_to_input_ft_dict, input_ft_to_idx_dict

def get_transformed_data_df_dict(df_dict, infoIDs,train_date_tuples,static_fts,dynamic_fts,target_fts):
    print("\ntransforming data...")

    fixed_train_tuples = []
    for t in train_date_tuples:
        train_input_start = t[0]
        train_input_end = t[1]
        train_output_start =t[2]
        train_output_end =t[3]
        train_input_start=pd.to_datetime(train_input_start, utc=True)
        train_input_end=pd.to_datetime(train_input_end, utc=True)
        train_output_start=pd.to_datetime(train_output_start, utc=True)
        train_output_end=pd.to_datetime(train_output_end, utc=True)
        cur_tuple = (train_input_start,train_input_end ,train_output_start,train_output_end )
        fixed_train_tuples.append(cur_tuple)
    train_date_tuples = list(fixed_train_tuples)

    infoID_to_df_dict = {}
    for infoID in infoIDs:
        infoID_to_df_dict[infoID]={}
        df = df_dict[infoID]
        # df = config_df_by_dates(df,start,end,time_col="nodeTime")

        all_input_dynamic_dfs = []
        all_target_dfs = []

        for train_tuple in train_date_tuples:
            print(train_tuple)
            train_input_start = train_tuple[0]
            train_input_end = train_tuple[1]
            train_output_start = train_tuple[2]
            train_output_end = train_tuple[3]

            #input fts
            temp = config_df_by_dates(df,train_input_start,train_input_end,time_col="nodeTime")
            # print()
            static_df = temp[static_fts].drop_duplicates().reset_index(drop=True)
            dynamic_df = temp[dynamic_fts]
            # print(static_df)
            timesteps = [i for i in range(1, dynamic_df.shape[0]+1)]
            dynamic_df["timestep"]=timesteps
            # print(dynamic_df)


            all_temps = [static_df]
            static_cols = list(static_df)
            ts_dynamic_fts = list(static_cols)

            data={}
            for timestep in timesteps:
                cur_dyn_df = dynamic_df[dynamic_df["timestep"]==timestep].reset_index(drop=True)
                for dynamic_ft in dynamic_fts:
                    new_ft = "ts_" + str(timestep) + "_" + dynamic_ft
                    cur_dyn_df = cur_dyn_df.rename(columns={dynamic_ft:new_ft})
                    ts_dynamic_fts.append(new_ft)
                    data[new_ft] = list(cur_dyn_df[new_ft])
                print(cur_dyn_df)



            cur_dynamic_df = pd.DataFrame(data=data)
            # for cur_infoID in infoIDs:
            #   if cur_infoID==infoID:
            #       cur_dynamic_df[infoID]=1
            #   else:
            #       cur_dynamic_df[infoID]=0

            # print("\ncur_dynamic_df")
            # print(cur_dynamic_df)

            for s in static_cols:
                cur_dynamic_df[s] = static_df[s]

            cur_dynamic_df = cur_dynamic_df[ts_dynamic_fts]

            # print()
            # print(cur_dynamic_df)

            all_input_dynamic_dfs.append(cur_dynamic_df)


            #output fts
            temp = config_df_by_dates(df,train_output_start,train_output_end,time_col="nodeTime")
            temp = temp[target_fts]
            timesteps = [i for i in range(1, temp.shape[0]+1)]
            temp["timestep"]=timesteps
            # print(temp)
            # sys.exit(0)

            all_temps = []
            ts_dynamic_fts = []
            for timestep in timesteps:
                cur_dyn_df = temp[temp["timestep"]==timestep].reset_index(drop=True)
                for target_ft in target_fts:
                    new_target_ft = "ts_" + str(timestep) + "_" + target_ft
                    cur_dyn_df = cur_dyn_df.rename(columns={target_ft:new_target_ft})
                    ts_dynamic_fts.append(new_target_ft)
                all_temps.append(cur_dyn_df)
            cur_dynamic_df = pd.concat(all_temps, axis=1)
            # print(cur_dynamic_df)

            all_target_dfs.append(cur_dynamic_df)

        input_dynamic_df = pd.concat(all_input_dynamic_dfs).reset_index(drop=True)
        # print("\ninput_dynamic_df")
        # print(input_dynamic_df)
        #
        # input_dynamic_df = input_dynamic_df.drop("timestep", axis=1)
        # print("\ninput_dynamic_df")
        # print(input_dynamic_df)

        infoID_to_df_dict[infoID]["xdf"]=input_dynamic_df


        target_df = pd.concat(all_target_dfs).reset_index(drop=True)
        target_df = target_df.drop("timestep", axis=1)
        infoID_to_df_dict[infoID]["ydf"]=target_df
        # print("\ntarget_df")
        # print(target_df)

        #

        input_cols = list(input_dynamic_df)
        target_cols = list(target_df )
        # print("\ninput_cols")
        # for col in input_cols:
        #     print(col)
        # print("\ntarget_cols")
        # for col in target_cols:
        #     print(col)



    idx_to_input_ft_dict = {}
    input_ft_to_idx_dict = {}

    idx_to_target_ft_dict = {}
    target_ft_to_idx_dict={}

    for i,input_col in enumerate(input_cols):
        idx_to_input_ft_dict[i]=input_col
        input_ft_to_idx_dict[input_col]=i



    for i,target_col in enumerate(target_cols):
        idx_to_target_ft_dict[i]=target_col
        target_ft_to_idx_dict[target_col]=i



    return infoID_to_df_dict, input_cols, target_cols ,idx_to_input_ft_dict, input_ft_to_idx_dict,idx_to_target_ft_dict,target_ft_to_idx_dict


def get_data_array_dict(df_dict, infoIDs,train_date_tuples,test_date_sample_tuples,start, end,static_fts,dynamic_fts,target_fts):

    fixed_train_tuples = []
    for t in train_date_tuples:
        train_input_start = t[0]
        train_input_end = t[1]
        train_output_start =t[2]
        train_output_end =t[3]
        train_input_start=pd.to_datetime(train_input_start, utc=True)
        train_input_end=pd.to_datetime(train_input_end, utc=True)
        train_output_start=pd.to_datetime(train_output_start, utc=True)
        train_output_end=pd.to_datetime(train_output_end, utc=True)
        cur_tuple = (train_input_start,train_input_end ,train_output_start,train_output_end )
        fixed_train_tuples.append(cur_tuple)
    train_date_tuples = list(fixed_train_tuples)

    infoID_to_array_dict = {}
    for infoID in infoIDs:
        df = df_dict[infoID]
        df = config_df_by_dates(df,start,end,time_col="nodeTime")

        all_input_dynamic_dfs = []
        all_target_dfs = []

        for train_tuple in train_date_tuples:
            print(train_tuple)
            train_input_start = train_tuple[0]
            train_input_end = train_tuple[1]
            train_output_start = train_tuple[2]
            train_output_end = train_tuple[3]

            #input fts
            temp = config_df_by_dates(df,train_input_start,train_input_end,time_col="nodeTime")
            print()
            static_df = temp[static_fts].drop_duplicates().reset_index(drop=True)
            dynamic_df = temp[dynamic_fts]
            print(static_df)
            timesteps = [i for i in range(1, dynamic_df.shape[0]+1)]
            dynamic_df["timestep"]=timesteps
            print(dynamic_df)

            #transform df
            all_temps = [static_df]
            ts_dynamic_fts = []
            for timestep in timesteps:
                cur_dyn_df = dynamic_df[dynamic_df["timestep"]==timestep].reset_index(drop=True)
                for dynamic_ft in dynamic_fts:
                    new_ft = "ts_" + str(timestep) + "_" + dynamic_ft
                    cur_dyn_df = cur_dyn_df.rename(columns={dynamic_ft:new_ft})
                    ts_dynamic_fts.append(new_ft)
                all_temps.append(cur_dyn_df)
            cur_dynamic_df = pd.concat(all_temps, axis=1)
            print(cur_dynamic_df)
            all_input_dynamic_dfs.append(cur_dynamic_df)

            #output fts
            temp = config_df_by_dates(df,train_output_start,train_output_end,time_col="nodeTime")
            temp = temp[target_fts]
            timesteps = [i for i in range(1, temp.shape[0]+1)]
            temp["timestep"]=timesteps
            print(temp)

            all_temps = []
            ts_dynamic_fts = []
            for timestep in timesteps:
                cur_dyn_df = temp[temp["timestep"]==timestep].reset_index(drop=True)
                for target_ft in target_fts:
                    new_target_ft = "ts_" + str(timestep) + "_" + target_ft
                    cur_dyn_df = cur_dyn_df.rename(columns={target_ft:new_target_ft})
                    ts_dynamic_fts.append(new_target_ft)
                all_temps.append(cur_dyn_df)
            cur_dynamic_df = pd.concat(all_temps, axis=1)
            print(cur_dynamic_df)
            all_target_dfs.append(cur_dynamic_df)

        input_dynamic_df = pd.concat(all_input_dynamic_dfs).reset_index(drop=True)
        input_dynamic_df = input_dynamic_df.drop("timestep", axis=1)
        print("\ninput_dynamic_df")
        print(input_dynamic_df)


        target_df = pd.concat(all_target_dfs).reset_index(drop=True)
        target_df = target_df.drop("timestep", axis=1)
        print("\ntarget_df")
        print(target_df)

        input_cols = list(input_dynamic_df)
        target_cols = list(target_df )
        print("\ninput_cols")
        for col in input_cols:
            print(col)
        print("\ntarget_cols")
        for col in target_cols:
            print(col)


    return

def config_df_by_dates(df,start_date,end_date,time_col="nodeTime"):
    df[time_col]=pd.to_datetime(df[time_col], utc=True)
    df = df.sort_values(time_col)
    df=df.set_index(time_col)

    if (start_date != None) and (end_date != None):
        df = df[(df.index >= start_date) & (df.index <= end_date)]

    df=df.reset_index(drop=False)
    return df

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

def get_data_df_dict(infoIDs ,start, end,ft_data_dir ,static_fts,dynamic_fts,target_fts):

    df_dict = {}

    hyp_dict = hyphenate_infoID_dict(infoIDs)

    for infoID in infoIDs:
        hyp_infoID = hyp_dict[infoID]

        fp = ft_data_dir + hyp_infoID + "-features.csv"
        df = pd.read_csv(fp)
        fts = list(set(static_fts+dynamic_fts+target_fts))
        df = df[["nodeTime"] + fts]
        df = config_df_by_dates(df,start,end,time_col="nodeTime")
        print()
        print(df)
        df_dict[infoID]=df

    return df_dict

def cp6_get_data_df_dict_with_exo_fts(infoIDs ,start, end,ft_data_dir ,static_fts,dynamic_fts,target_fts, exo_ft_dir, exo_fts, internal_fts, GRAN):

    df_dict = {}

    hyp_dict = hyphenate_infoID_dict(infoIDs)

    for infoID in infoIDs:
        hyp_infoID = hyp_dict[infoID]

        fp = ft_data_dir + hyp_infoID + "-features.csv"
        df = pd.read_csv(fp)

        #might have to put "target_fts" back into this list... We'll see...
        fts = list(set(static_fts+internal_fts))
        df = df[["nodeTime"] + fts]
        df = config_df_by_dates(df,start,end,time_col="nodeTime")
        print()
        print(df)

        #get exo fts
        exo_fp = exo_ft_dir + hyp_infoID + "-exo-features.csv"
        exo_df = pd.read_csv(exo_fp)
        exo_df = exo_df[["nodeTime"] + exo_fts]
        exo_df = config_df_by_dates(exo_df,start,end,time_col="nodeTime")
        print()
        print(exo_df)

        df = pd.merge(df, exo_df, on="nodeTime", how="inner").reset_index(drop=True)
        df = df.sort_values("nodeTime").reset_index(drop=True)

        dates = pd.date_range(start, end, freq=GRAN)
        num_dates = len(dates)

        if df.shape[0] != num_dates:
            print("\nError! %d != %d"%(df.shape[0], num_dates))
            sys.exit(0)

        print()
        print(df)

        df_dict[infoID]=df

    return df_dict

def get_data_df_dict_v2_with_EXO(infoIDs ,start, end,ft_data_dir ,static_fts,dynamic_fts,target_fts, exo_df):

    df_dict = {}

    hyp_dict = hyphenate_infoID_dict(infoIDs)

    for infoID in infoIDs:
        hyp_infoID = hyp_dict[infoID]

        fp = ft_data_dir + hyp_infoID + "-features.csv"
        df = pd.read_csv(fp)
        fts = list(set(static_fts+dynamic_fts+target_fts))

        df["nodeTime"] = pd.to_datetime(df["nodeTime"], utc=True)
        exo_df["nodeTime"] = pd.to_datetime(exo_df["nodeTime"], utc=True)

        df = pd.merge(exo_df, df, on="nodeTime", how="inner")
        df = config_df_by_dates(df,start,end,time_col="nodeTime")


        df = df[["nodeTime"] + fts]
        print()
        print(df)
        df_dict[infoID]=df

    return df_dict

def cp5_infoIDs():

    return ['no_narrative',
    'leadership/khan',
    'benefits/covid',
    'controversies/pakistan/students',
    'benefits/development/energy',
    'benefits/development/maritime',
    'controversies/china/exploitation',
    'leadership/sharif',
    'opposition/kashmir',
    'controversies/china/border',
    'benefits/development/roads',
    'controversies/china/uighur',
    'controversies/pakistan/baloch',
    'benefits/connections/afghanistan',
    'leadership/bajwa',
    'controversies/pakistan/army',
    'opposition/propaganda',
    'controversies/china/debt',
    'benefits/jobs',
    'controversies/china/funding',
    'controversies/pakistan/bajwa',
    'controversies/china/naval'
    ]

def cp5_infoIDs_challenge():

    return [
    "controversies/pakistan/students",
    "leadership/sharif",
    "leadership/bajwa",
    "controversies/china/uighur",
    "controversies/china/border",
    "benefits/development/roads",
    "controversies/pakistan/baloch",
    "benefits/jobs",
    "opposition/propaganda",
    "benefits/development/energy",
    "controversies/pakistan/bajwa",
    "no_narrative"
    ]

def cp5_infoIDs_challenge_with_other():

    return [
    "controversies/pakistan/students",
    "leadership/sharif",
    "leadership/bajwa",
    "controversies/china/uighur",
    "controversies/china/border",
    "benefits/development/roads",
    "controversies/pakistan/baloch",
    "benefits/jobs",
    "opposition/propaganda",
    "benefits/development/energy",
    "controversies/pakistan/bajwa",
    "other"
    ]



def get_fts_from_fp(input_fp):
    model_fts = []
    with open(input_fp) as f:
        for line in f:
            line = line.replace("\n","")
            model_fts.append(line)
            # print(line)
    return model_fts

def config_ft_lists_v3_EXO(ft_list_dir , TARGET_PLATFORM,GET_STATIC_FTS, GET_OTHER_PLATFORM_LOCAL_FTS, GET_OTHER_PLATFORM_GLOBAL_FTS, GET_TARGET_PLATFORM_GLOBAL_FTS, GET_REDDIT, GET_ACLED):

    eval_target_fts= get_fts_from_fp(ft_list_dir  + "target_fts.txt")

    if GET_STATIC_FTS == True:
        static_fts = get_fts_from_fp(ft_list_dir  + "static_fts.txt")
    else:
        static_fts = []

    #get target fts
    keep_target_fts = []
    for ft in eval_target_fts:
        if TARGET_PLATFORM in ft:
            keep_target_fts.append(ft)
            print(ft)
    eval_target_fts = list(keep_target_fts)

    #you have to get local fts
    dynamic_fts = get_fts_from_fp(ft_list_dir  + "%s_dynamic_pair_fts.txt"%TARGET_PLATFORM)



    #opt -> get target global
    if GET_TARGET_PLATFORM_GLOBAL_FTS == True:
        dynamic_fts =dynamic_fts +  get_fts_from_fp(ft_list_dir  + "%s_dynamic_global_fts.txt"%TARGET_PLATFORM)

    #other platform fts
    if TARGET_PLATFORM == "twitter":
        OTHER_PLATFORM="youtube"
    else:
        OTHER_PLATFORM="twitter"

    if GET_OTHER_PLATFORM_LOCAL_FTS == True:
        dynamic_fts =dynamic_fts +  get_fts_from_fp(ft_list_dir  + "%s_dynamic_pair_fts.txt"%OTHER_PLATFORM)

    if GET_OTHER_PLATFORM_GLOBAL_FTS == True:
        dynamic_fts =dynamic_fts +  get_fts_from_fp(ft_list_dir  + "%s_dynamic_global_fts.txt"%OTHER_PLATFORM)

    if GET_REDDIT == True:
        dynamic_fts = dynamic_fts + ["num_reddit_actions"]

    if GET_ACLED == True:
        dynamic_fts = dynamic_fts + ["num_acled_actions"]

    target_fts = list(dynamic_fts)


    print("\nstatic_fts")
    for s in static_fts:
        print(s)

    print("\ndynamic_fts")
    for d in dynamic_fts:
        print(d)

    print("\ntarget_fts")
    for t in target_fts:
        print(t)

    print("\neval_target_fts")
    for t in eval_target_fts:
        print(t)

    # sys.exit(0)

    return static_fts,dynamic_fts,target_fts,eval_target_fts

def config_ft_lists_v3_cp6(ft_list_dir , TARGET_PLATFORM,GET_STATIC_FTS, GET_OTHER_PLATFORM_LOCAL_FTS, GET_OTHER_PLATFORM_GLOBAL_FTS, GET_TARGET_PLATFORM_GLOBAL_FTS, ALL_PLATFORMS):

    # eval_target_fts= get_fts_from_fp(ft_list_dir  + "target_fts.txt")

    if GET_STATIC_FTS == True:
        static_fts = get_fts_from_fp(ft_list_dir  + "static_fts.txt")
    else:
        static_fts = []

    #get target fts
    # keep_target_fts = []
    # for ft in eval_target_fts:
    #     if TARGET_PLATFORM in ft:
    #         keep_target_fts.append(ft)
    #         print(ft)

    # if ((TARGET_PLATFORM != "youtube") and (TARGET_PLATFORM != "twitter")):
    eval_target_fts = ["%s_platform_infoID_pair_nunique_new_users"%TARGET_PLATFORM , "%s_platform_infoID_pair_nunique_old_users"%TARGET_PLATFORM, "%s_platform_infoID_pair_num_actions"%TARGET_PLATFORM]

    #you have to get local fts
    dynamic_fts = get_fts_from_fp(ft_list_dir  + "%s_dynamic_pair_fts.txt"%TARGET_PLATFORM)



    #opt -> get target global
    if GET_TARGET_PLATFORM_GLOBAL_FTS == True:
        dynamic_fts =dynamic_fts +  get_fts_from_fp(ft_list_dir  + "%s_dynamic_global_fts.txt"%TARGET_PLATFORM)

    #other platform fts
    # if TARGET_PLATFORM == "twitter":
    #     OTHER_PLATFORM="youtube"
    # else:
    #     OTHER_PLATFORM="twitter"

    # ALL_PLATFORMS = ["twitter", "reddit", "jamii", "youtube"]

    OTHER_PLATFORMS = []
    for AP in ALL_PLATFORMS:
        if AP!= TARGET_PLATFORM:
            OTHER_PLATFORMS.append(AP)

    print("\nOTHER_PLATFORMS")
    print(OTHER_PLATFORMS)

    if GET_OTHER_PLATFORM_LOCAL_FTS == True:
        for OP in OTHER_PLATFORMS:
            dynamic_fts +=get_fts_from_fp(ft_list_dir  + "%s_dynamic_pair_fts.txt"%OP)

    if GET_OTHER_PLATFORM_GLOBAL_FTS == True:
        #dynamic_fts =dynamic_fts +  get_fts_from_fp(ft_list_dir  + "%s_dynamic_global_fts.txt"%OTHER_PLATFORM)
        for OP in OTHER_PLATFORMS:
            dynamic_fts +=get_fts_from_fp(ft_list_dir  + "%s_dynamic_global_fts.txt"%OP)

    target_fts = list(dynamic_fts)


    print("\nstatic_fts")
    for s in static_fts:
        print(s)

    print("\ndynamic_fts")
    for d in dynamic_fts:
        print(d)

    print("\ntarget_fts")
    for t in target_fts:
        print(t)

    print("\neval_target_fts")
    for t in eval_target_fts:
        print(t)

    # sys.exit(0)

    return static_fts,dynamic_fts,target_fts,eval_target_fts

def config_ft_lists_v2_more_options(ft_list_dir , TARGET_PLATFORM,GET_STATIC_FTS, GET_OTHER_PLATFORM_LOCAL_FTS, GET_OTHER_PLATFORM_GLOBAL_FTS, GET_TARGET_PLATFORM_GLOBAL_FTS):

    eval_target_fts= get_fts_from_fp(ft_list_dir  + "target_fts.txt")

    if GET_STATIC_FTS == True:
        static_fts = get_fts_from_fp(ft_list_dir  + "static_fts.txt")
    else:
        static_fts = []

    #get target fts
    keep_target_fts = []
    for ft in eval_target_fts:
        if TARGET_PLATFORM in ft:
            keep_target_fts.append(ft)
            print(ft)
    eval_target_fts = list(keep_target_fts)

    #you have to get local fts
    dynamic_fts = get_fts_from_fp(ft_list_dir  + "%s_dynamic_pair_fts.txt"%TARGET_PLATFORM)



    #opt -> get target global
    if GET_TARGET_PLATFORM_GLOBAL_FTS == True:
        dynamic_fts =dynamic_fts +  get_fts_from_fp(ft_list_dir  + "%s_dynamic_global_fts.txt"%TARGET_PLATFORM)

    #other platform fts
    if TARGET_PLATFORM == "twitter":
        OTHER_PLATFORM="youtube"
    else:
        OTHER_PLATFORM="twitter"

    if GET_OTHER_PLATFORM_LOCAL_FTS == True:
        dynamic_fts =dynamic_fts +  get_fts_from_fp(ft_list_dir  + "%s_dynamic_pair_fts.txt"%OTHER_PLATFORM)

    if GET_OTHER_PLATFORM_GLOBAL_FTS == True:
        dynamic_fts =dynamic_fts +  get_fts_from_fp(ft_list_dir  + "%s_dynamic_global_fts.txt"%OTHER_PLATFORM)

    target_fts = list(dynamic_fts)


    print("\nstatic_fts")
    for s in static_fts:
        print(s)

    print("\ndynamic_fts")
    for d in dynamic_fts:
        print(d)

    print("\ntarget_fts")
    for t in target_fts:
        print(t)

    print("\neval_target_fts")
    for t in eval_target_fts:
        print(t)

    # sys.exit(0)

    return static_fts,dynamic_fts,target_fts,eval_target_fts

def config_ft_lists(ft_list_dir , TARGET_PLATFORM,GET_STATIC_FTS,GET_ALL_PLATFORM_FTS):

    target_fts= get_fts_from_fp(ft_list_dir  + "target_fts.txt")

    if GET_STATIC_FTS == True:
        static_fts = get_fts_from_fp(ft_list_dir  + "static_fts.txt")
    else:
        static_fts = []

    #get target fts
    keep_target_fts = []
    for ft in target_fts:
        if TARGET_PLATFORM in ft:
            keep_target_fts.append(ft)
            print(ft)
    target_fts = list(keep_target_fts)

    #get fts
    global_dynamic_fts = get_fts_from_fp(ft_list_dir  + "%s_dynamic_global_fts.txt"%TARGET_PLATFORM)
    pair_dynamic_fts = get_fts_from_fp(ft_list_dir  + "%s_dynamic_pair_fts.txt"%TARGET_PLATFORM)
    dynamic_fts = list(global_dynamic_fts + pair_dynamic_fts)

    if GET_ALL_PLATFORM_FTS == True:
        if TARGET_PLATFORM == "twitter":
            OTHER_PLATFORM="youtube"
        else:
            OTHER_PLATFORM="twitter"

        other_global_dynamic_fts = get_fts_from_fp(ft_list_dir  + "%s_dynamic_global_fts.txt"%OTHER_PLATFORM)
        other_pair_dynamic_fts = get_fts_from_fp(ft_list_dir  + "%s_dynamic_pair_fts.txt"%OTHER_PLATFORM)
        dynamic_fts = list(dynamic_fts+ other_global_dynamic_fts + other_pair_dynamic_fts)

    print("\nstatic_fts")
    for s in static_fts:
        print(s)

    print("\ndynamic_fts")
    for d in dynamic_fts:
        print(d)

    print("\ntarget_fts")
    for t in target_fts:
        print(t)

    # sys.exit(0)

    return static_fts,dynamic_fts,target_fts

# def get_date_samples_v2_simple(overall_start, period_output_end, num_period_dates, period_type,LOOKBACK_FACTOR,OUTPUT_SIZE):
def get_date_samples_v2_simple(date_start, date_end, period_type,LOOKBACK_FACTOR,OUTPUT_SIZE):

    print("\nGetting %s samples"%period_type)
    print("lookback and output")
    print(LOOKBACK_FACTOR)
    print(OUTPUT_SIZE)

    NUM_WEEKS = int(LOOKBACK_FACTOR/7)
    NUM_DAYS = int(LOOKBACK_FACTOR%7)
    print(NUM_WEEKS)
    print(NUM_DAYS)
    # sys.exit(0)

    # dates = pd.date_range(overall_start, period_output_end, freq="D")
    # num_dates = len(dates)
    # overall_period_output_start = dates[-num_period_dates]
    # overall_period_output_end = dates[-1]

    dates = pd.date_range(date_start, date_end, freq="D")
    num_period_dates = len(dates)
    overall_period_output_start = dates[0]
    overall_period_output_end = dates[-1]

    print("\n%s output start and end"%period_type)
    print(overall_period_output_start)
    print(overall_period_output_end)

    #get num samples
    num_samples = int(num_period_dates/OUTPUT_SIZE)
    print("\nnum_samples")
    print(num_samples)

    sample_output_date_tuple_list = []

    cur_output_start = overall_period_output_start
    for i in range(num_samples):
        cur_output_end = cur_output_start + pd.DateOffset(days=(OUTPUT_SIZE-1))
        cur_tuple = (cur_output_start ,cur_output_end)
        print(cur_tuple)
        cur_output_start=cur_output_start + pd.DateOffset(days=(OUTPUT_SIZE))
        sample_output_date_tuple_list.append(cur_tuple)

    print()
    full_sample_tuples = []
    for i in range(num_samples):
        cur_output_tuple = sample_output_date_tuple_list[i]
        cur_input_start = cur_output_tuple[0] - pd.DateOffset(days=LOOKBACK_FACTOR)
        cur_input_end = cur_output_tuple[0]- pd.DateOffset(days=1)
        cur_tuple = (cur_input_start,cur_input_end )
        # print(cur_tuple)

        full_tuple = (cur_input_start,cur_input_end, cur_output_tuple[0],cur_output_tuple[1])
        print(full_tuple)
        full_sample_tuples.append(full_tuple)

    return full_sample_tuples

def get_date_samples(overall_start, period_output_end, num_period_dates, period_type,LOOKBACK_FACTOR,OUTPUT_SIZE):

    print("\nGetting %s samples"%period_type)
    print("lookback and output")
    print(LOOKBACK_FACTOR)
    print(OUTPUT_SIZE)

    NUM_WEEKS = int(LOOKBACK_FACTOR/7)
    NUM_DAYS = int(LOOKBACK_FACTOR%7)
    print(NUM_WEEKS)
    print(NUM_DAYS)
    # sys.exit(0)

    dates = pd.date_range(overall_start, period_output_end, freq="D")
    num_dates = len(dates)
    overall_period_output_start = dates[-num_period_dates]
    overall_period_output_end = dates[-1]

    print("\n%s output start and end"%period_type)
    print(overall_period_output_start)
    print(overall_period_output_end)

    #get num samples
    num_samples = int(num_period_dates/OUTPUT_SIZE)
    print("\nnum_samples")
    print(num_samples)

    sample_output_date_tuple_list = []

    cur_output_start = overall_period_output_start
    for i in range(num_samples):
        cur_output_end = cur_output_start + pd.DateOffset(days=(OUTPUT_SIZE-1))
        cur_tuple = (cur_output_start ,cur_output_end)
        print(cur_tuple)
        cur_output_start=cur_output_start + pd.DateOffset(days=(OUTPUT_SIZE))
        sample_output_date_tuple_list.append(cur_tuple)

    print()
    full_sample_tuples = []
    for i in range(num_samples):
        cur_output_tuple = sample_output_date_tuple_list[i]
        cur_input_start = cur_output_tuple[0] - pd.DateOffset(days=LOOKBACK_FACTOR)
        cur_input_end = cur_output_tuple[0]- pd.DateOffset(days=1)
        cur_tuple = (cur_input_start,cur_input_end )
        # print(cur_tuple)

        full_tuple = (cur_input_start,cur_input_end, cur_output_tuple[0],cur_output_tuple[1])
        print(full_tuple)
        full_sample_tuples.append(full_tuple)

    return full_sample_tuples

def get_train_date_samples_v2_proper_slide(overall_start, train_output_end,LOOKBACK_FACTOR,OUTPUT_SIZE,slide_size,GRAN="D"):

    try:
        dates = pd.date_range(overall_start, train_output_end, freq=GRAN)
    except TypeError:
        overall_start=pd.to_datetime(overall_start, utc=True)
        train_output_end=pd.to_datetime(train_output_end, utc=True)
        dates = pd.date_range(overall_start, train_output_end, freq=GRAN)
    num_dates = len(dates)

    print("\nfull num_dates")
    print(num_dates)

    date_tuples = []
    start=0
    while True:
        try:
            input_start_date = dates[start]
            input_end_date = dates[start+LOOKBACK_FACTOR-1]
            output_start_date = dates[start+LOOKBACK_FACTOR]
            output_end_date = dates[start+LOOKBACK_FACTOR+OUTPUT_SIZE-1]
        except IndexError:
            break

        cur_tuple = (input_start_date,input_end_date, output_start_date,output_end_date)
        print(cur_tuple)
        date_tuples.append(cur_tuple)

        start = start+slide_size

    return date_tuples


def get_train_date_samples_v2_proper_slide_BACKUP(overall_start, train_output_end,LOOKBACK_FACTOR,OUTPUT_SIZE,slide_size):

    try:
        dates = pd.date_range(overall_start, train_output_end, freq="D")
    except TypeError:
        overall_start=pd.to_datetime(overall_start, utc=True)
        train_output_end=pd.to_datetime(train_output_end, utc=True)
        dates = pd.date_range(overall_start, train_output_end, freq="D")
    num_dates = len(dates)

    print("\nfull num_dates")
    print(num_dates)

    date_tuples = []
    start=0
    while True:
        try:
            input_start_date = dates[start]
            input_end_date = dates[start+LOOKBACK_FACTOR-1]
            output_start_date = dates[start+LOOKBACK_FACTOR]
            output_end_date = dates[start+LOOKBACK_FACTOR+OUTPUT_SIZE-1]
        except IndexError:
            break

        cur_tuple = (input_start_date,input_end_date, output_start_date,output_end_date)
        print(cur_tuple)
        date_tuples.append(cur_tuple)

        start = start+slide_size

    return date_tuples








def plot_internal_exo_gt_and_bl_with_log_options(int_task_dict, exo_task_dict, bl_task_dict, plot_output_dir, platform, infoIDs, DESIRED_METRICS,METRIC_TAG_TO_METRICS_DICT,int_tag, exo_tag, bl_tag ,DESIRED_TASKS, LOG):

    #loop for all infoIDs
    for infoID in infoIDs:

        hyp_infoID = infoID.replace("/", "_")
        print(hyp_infoID)

        num_rows = 3
        num_cols = 3
        fig, axs = plt.subplots(num_rows, num_cols,figsize=(10,10))
        coord_list = [(0,0), (0,1), (1,0), (1,1), (2,0), (2,1)]

        int_coord_list = [(0,0), (1,0), (2,0)]
        exo_coord_list = [(0,1), (1,1), (2,1)]
        bl_coord_list = [(0,2), (1,2), (2,2)]

        #================================== internal preds ===============================================
        for task, coord in zip(DESIRED_TASKS,int_coord_list):
            print("%s: %s"%(task, str(coord)))


            cur_y_pred = int_task_dict[task][1][platform][infoID]
            cur_y_ground_truth = int_task_dict[task][0][platform][infoID]

            model_error_tag = int_tag + " - "
            for metric_tag in DESIRED_METRICS:
                model_error = METRIC_TAG_TO_METRICS_DICT[metric_tag](cur_y_ground_truth, cur_y_pred)
                model_error = np.round(model_error, 4)
                model_error = str(model_error)
                model_error_tag += metric_tag + ": %s;\n"%model_error

            title = infoID + "\n" + task + "\n" + model_error_tag

            if LOG == True:
                cur_y_pred+=1
                cur_y_ground_truth+=1

            axs[coord].plot(cur_y_pred,"-r" ,label=int_tag)
            axs[coord].plot(cur_y_ground_truth, "-k" ,label="Ground Truth")
            axs[coord].set_title(title)

            if LOG == True:
                axs[coord].set_yscale('log')

        #============================== exo preds ==============================================================
        for task, coord in zip(DESIRED_TASKS,exo_coord_list):
            print("%s: %s"%(task, str(coord)))

            cur_y_pred = exo_task_dict[task][1][platform][infoID]
            cur_y_ground_truth = exo_task_dict[task][0][platform][infoID]

            model_error_tag = exo_tag + " - "
            for metric_tag in DESIRED_METRICS:
                model_error = METRIC_TAG_TO_METRICS_DICT[metric_tag](cur_y_ground_truth, cur_y_pred)
                model_error = np.round(model_error, 4)
                model_error = str(model_error)
                model_error_tag += metric_tag + ": %s;\n"%model_error

            title = infoID + "\n" + task + "\n" + model_error_tag

            if LOG == True:
                cur_y_pred+=1
                cur_y_ground_truth+=1

            axs[coord].plot(cur_y_pred,"-r" ,label=exo_tag)
            axs[coord].plot(cur_y_ground_truth, "-k" ,label="Ground Truth")
            axs[coord].set_title(title)

            if LOG == True:
                axs[coord].set_yscale('log')

        #============================== bl preds ==============================================================
        for task, coord in zip(DESIRED_TASKS,bl_coord_list):
            print("%s: %s"%(task, str(coord)))

            cur_y_pred = bl_task_dict[task][1][platform][infoID]
            cur_y_ground_truth = bl_task_dict[task][0][platform][infoID]

            model_error_tag = bl_tag + " - "
            for metric_tag in DESIRED_METRICS:
                model_error = METRIC_TAG_TO_METRICS_DICT[metric_tag](cur_y_ground_truth, cur_y_pred)
                model_error = np.round(model_error, 4)
                model_error = str(model_error)
                model_error_tag += metric_tag + ": %s;\n"%model_error

            title = infoID + "\n" + task + "\n" + model_error_tag

            if LOG == True:
                cur_y_pred+=1
                cur_y_ground_truth+=1

            axs[coord].plot(cur_y_pred,"-g" ,label=bl_tag)
            axs[coord].plot(cur_y_ground_truth, "-k" ,label="Ground Truth")
            axs[coord].set_title(title)

            if LOG == True:
                axs[coord].set_yscale('log')

        for ax in axs.flat:
            ax.set(xlabel='days', ylabel='Volume')

        plt.tight_layout()
        output_fp = plot_output_dir + "%s.png"%( hyp_infoID)
        fig.savefig(output_fp)
        plt.close()
        print(output_fp)

    return






def plot_model_with_baseline_and_gt_v2_log_options(model_test_infoID_to_pred_df_dict,test_infoID_to_gt_df_dict,baseline_test_infoID_to_pred_df_dict, infoIDs, target_cats,comp_results_dict,metric_tags, output_dir,LOG=False):

    output_dir = output_dir + "LOG-%s/"%LOG
    create_output_dir(output_dir)

    hyp_dict = hyphenate_infoID_dict(infoIDs)

    for infoID in infoIDs:

        # fig, ax = plt.subplots()

        hyp_infoID = hyp_dict[infoID]

        num_rows = 3
        num_cols = 2
        fig, axs = plt.subplots(3, 2,figsize=(10,10))
        coord_list = [(0,0), (0,1), (1,0), (1,1), (2,0), (2,1)]

        model_coord_list = [(0,0), (1,0), (2,0)]
        baseline_coord_list = [(0,1), (1,1), (2,1)]

        # coord_list = [0, 1, 2]
        # fig, axs = plt.subplots(3,figsize=(8,8))
        target_cat_to_axis_coordinates_dict = {}

        for target_cat, coord in zip(target_cats,model_coord_list):
            print("%s: %s"%(target_cat, str(coord)))

            # cur_df = pd.DataFrame(data={"Prediction":y_pred, "Ground_Truth":y_ground_truth})

            cur_y_pred = model_test_infoID_to_pred_df_dict[infoID][target_cat]
            cur_y_ground_truth = test_infoID_to_gt_df_dict[infoID][target_cat]

            model_error_tag = "VAM - "
            for metric_tag in metric_tags:
                cur_infoID_df = comp_results_dict[metric_tag][target_cat]
                cur_infoID_df=cur_infoID_df[cur_infoID_df["infoID"]==infoID].reset_index(drop=True)
                model_error = cur_infoID_df["VAM_%s"%metric_tag].iloc[0]
                model_error_tag += metric_tag + ": %.4f;\n"%model_error

            title = infoID + "\n" + target_cat + "\n" + model_error_tag
            axs[coord].plot(cur_y_pred,"-r" ,label="VAM")
            axs[coord].plot(cur_y_ground_truth, "-k" ,label="Ground Truth")
            axs[coord].set_title(title)

            if LOG == True:
                axs[coord].set_yscale('log')


        for target_cat,coord in zip(target_cats,baseline_coord_list ):
            print("%s: %s"%(target_cat, str(coord)))

            cur_y_ground_truth = test_infoID_to_gt_df_dict[infoID][target_cat]
            cur_y_baseline_pred =baseline_test_infoID_to_pred_df_dict[infoID][target_cat]

            baseline_error_tag = "PB - "
            for metric_tag in metric_tags:
                cur_infoID_df = comp_results_dict[metric_tag][target_cat]
                cur_infoID_df=cur_infoID_df[cur_infoID_df["infoID"]==infoID].reset_index(drop=True)
                baseline_error = cur_infoID_df["Persistence_Baseline_%s"%metric_tag].iloc[0]
                baseline_error_tag += metric_tag + ": %.4f;\n"%baseline_error

            title = infoID + "\n" + target_cat + "\n" + baseline_error_tag

            axs[coord].plot(cur_y_baseline_pred,"-g" ,label="Persistence_Baseline")
            axs[coord].plot(cur_y_ground_truth, "-k" ,label="Ground Truth")
            axs[coord].set_title(title)

            if LOG == True:
                axs[coord].set_yscale('log')

        for ax in axs.flat:
            ax.set(xlabel='days', ylabel='Volume')

        plt.tight_layout()
        output_fp = output_dir + "%s.png"%( hyp_infoID)
        fig.savefig(output_fp)
        plt.close()
        print(output_fp)

    # for infoID in infoIDs:

    #   pred_df =



    return
