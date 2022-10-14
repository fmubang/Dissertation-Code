import pandas as pd 
import json
import pickle
import numpy as np 
import os,sys
from functools import *
import calendar
import shutil
import stat

def save_ft_list_as_text_file(output_dir, tag,ft_list):
    output_fp = output_dir + tag + ".txt"
    with open(output_fp, "w") as f:
        print("\n%s"%tag) 
        for line in ft_list:
            f.write(line + "\n")
            print(line)

def get_sim_period_string_from_dates(test_start, test_end):



    test_start = pd.to_datetime(test_start, utc=True)
    test_end = pd.to_datetime(test_end, utc=True)
    # print(test_start)
    # print(test_end)

    start_day = test_start.day
    start_month = test_start.month
    end_day = test_end.day
    end_month = test_end.month
    start_month = calendar.month_name[start_month].lower()
    end_month = calendar.month_name[end_month].lower()

    # print(start_day)
    # print(start_month)
    # print(end_day)
    # print(end_month)

    final_str = "%s%d-%s%d"%(start_month,start_day, end_month,end_day)
    return final_str

def create_blank_date_df(start,end,GRAN):

    dates = pd.date_range(start,end,freq=GRAN)
    blank_date_df = pd.DataFrame(data={"nodeTime":dates})
    blank_date_df["nodeTime"] = pd.to_datetime(blank_date_df["nodeTime"], utc=True)
    return blank_date_df

def get_1hot_vectors(fts):

    hot_dict = {}
    for i,ft in enumerate(fts):
        hot_dict[ft] = [0 for j in range(len(fts))]
        hot_dict[ft][i] = 1
        print("%s: %s" %(ft, str(hot_dict[ft][i])))

    return hot_dict

def convert_infoID_to_underscore_form(infoID):
    if "/" in infoID:
        return infoID.replace("/", "_")
    return infoID

# def get_46_cp4_infoIDs_and_global():
#     return get_46_cp4_infoIDs() + ["global"]

def hyphenate_infoID_dict(infoIDs):

    h_dict = {}
    for infoID in infoIDs:
        if "/" in infoID:
            h_dict[infoID] = infoID.replace("/", "_")
        else:
            h_dict[infoID] =infoID

    return h_dict

def config_df_by_dates(df,start_date,end_date,time_col="nodeTime"):
    df[time_col]=pd.to_datetime(df[time_col], utc=True)
    df = df.sort_values(time_col)
    df=df.set_index(time_col)

    if (start_date != None) and (end_date != None):
        df = df[(df.index >= start_date) & (df.index <= end_date)]

    df=df.reset_index(drop=False)
    return df

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

def debug_print(str_to_print,LIMIT_TEST=0,DEBUG=False):
    LIMIT=10
    if (LIMIT_TEST < LIMIT) and DEBUG==True:
        print(str_to_print)

def save_as_json(OBJECT,OUTPUT_FP):
    print("Saving %s..."%OUTPUT_FP)
    with open(OUTPUT_FP, "w") as f:
        json.dump(OBJECT,f)
    print("Saved %s"%OUTPUT_FP)

def create_output_dir(OUTPUT_DIR):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print("Created %s" %OUTPUT_DIR)
    else:
        print("%s already exists"%OUTPUT_DIR)

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

def cp5_save_sim_df_as_json_with_header_v2(sim_df,identifier, json_output_fp,simulation_period="january25-january31"):
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

def verify_df_size(df, size_to_enforce,tag):
    print("\nVerifying %s df size"%tag)
    print(df.shape[0])
    print(size_to_enforce)
    if df.shape[0] != size_to_enforce:
        print("\nError! %s df should be of size %d but it is %d!"%(tag ,size_to_enforce, df.shape[0]))
        sys.exit(0)
    else:
        print("Df size is ok! Continuing...")

def merge_mult_dfs(merge_list, on, how):
    print("\nMerging multiple dfs...")
    return reduce(lambda  left,right: pd.merge(left,right,on=on, how=how), merge_list)

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

def make_tuple_list_from_2_cols(df,col1,col2):
    return list(zip(df[col1], df[col2]))

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