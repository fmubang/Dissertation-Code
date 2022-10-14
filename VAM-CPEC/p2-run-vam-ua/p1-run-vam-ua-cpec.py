import sys
#sys.path.append("/data/Fmubang/CP4-ORGANIZED-V3-FIX-DROPNA/functions")
# sys.path.append("/beegfs-mnt/data/fmubang/CP5-VAM-Paper-Stuff-3-3/functions")
sys.path.append("/beegfs-mnt/data/fmubang/CP4-ORGANIZED-V3-FIX-DROPNA/functions")
import pandas as pd
import os,sys
from scipy import stats
import numpy as np
from time import time
from joblib import Parallel, delayed
import multiprocessing
import vam_ua_funcs as ua
from basic_utils import create_output_dir, convert_df_2_cols_to_dict, save_pickle,hyphenate_infoID_dict,config_df_by_dates
from infoIDs18 import get_cp4_challenge_18_infoIDs
import network_metric_funcs as nmf
# from cp5_topics import get_cp5_10_topics

DEBUG = True
# new_gaivi2_input_dir = "/beegfs-mnt/data/gaivi2/data/Fmubang/"

infoIDs = [
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
    # "controversies/pakistan/bajwa",
    # "other"
    ]



#=============== seed stuff ==============================================
MODEL_CONFIG_NUM = int(sys.argv[1])
print("\nMODEL_CONFIG_NUM: %d"%MODEL_CONFIG_NUM)

if MODEL_CONFIG_NUM == 1:
    SEEDS = [100, 200, 300]
    MULT_USER_WEIGHT_BY_AGE_LIST = [False]
    TWITTER_HISTORICAL_LOOKBACK_HOURS_LIST = [24]
    YOUTUBE_HISTORICAL_LOOKBACK_HOURS_LIST = []
if MODEL_CONFIG_NUM == 2:
    SEEDS = [400, 500]
    MULT_USER_WEIGHT_BY_AGE_LIST = [False]
    TWITTER_HISTORICAL_LOOKBACK_HOURS_LIST = [24]
    YOUTUBE_HISTORICAL_LOOKBACK_HOURS_LIST = []
if MODEL_CONFIG_NUM == 3:
    SEEDS = [100, 200, 300]
    MULT_USER_WEIGHT_BY_AGE_LIST = [False]
    TWITTER_HISTORICAL_LOOKBACK_HOURS_LIST = [12]
    YOUTUBE_HISTORICAL_LOOKBACK_HOURS_LIST = []
if MODEL_CONFIG_NUM == 4:
    SEEDS = [400, 500]
    MULT_USER_WEIGHT_BY_AGE_LIST = [False]
    TWITTER_HISTORICAL_LOOKBACK_HOURS_LIST = [12]
    YOUTUBE_HISTORICAL_LOOKBACK_HOURS_LIST = []
if MODEL_CONFIG_NUM == 5:
    SEEDS = [100, 200, 300]
    MULT_USER_WEIGHT_BY_AGE_LIST = [False]
    TWITTER_HISTORICAL_LOOKBACK_HOURS_LIST = [48]
    YOUTUBE_HISTORICAL_LOOKBACK_HOURS_LIST = []
if MODEL_CONFIG_NUM == 6:
    SEEDS = [400, 500]
    MULT_USER_WEIGHT_BY_AGE_LIST = [False]
    TWITTER_HISTORICAL_LOOKBACK_HOURS_LIST = [48]
    YOUTUBE_HISTORICAL_LOOKBACK_HOURS_LIST = []

print(SEEDS)

# sys.exit(0)
#================================ basic params ===============================

DEBUG_PRINT = True
platforms = ["twitter"]


SIM_PERIOD_SIZE = 24
NUM_SIM_PERIODS = 14
REMOVE_MISSING_PARENT_USERS = True
GET_METRICS = True
GRAN = "24H"
# infoIDs = get_cp4_challenge_18_infoIDs()
NUM_JOBS = len(infoIDs)
hyp_dict = hyphenate_infoID_dict(infoIDs)
NUM_LINK_SAMPLE_ATTEMPTS = 10

#====================================================== LIST PARAMS ============================================================
INFLUENCE_EPSILON_DIV_VAL_LIST = [10]
ADD_OLD_USERS_TO_NEW_USERS_IF_TOO_MANY_LIST = [False]
#user_action_conflict_resolution_options = ["downsample_users_to_actions", "upsample_actions_to_users"]
user_action_conflict_resolution_options = ["upsample_actions_to_users"]
# MULT_USER_WEIGHT_BY_AGE_LIST = [True, False]

#dicts for later
CR_TAG_DICT = {"downsample_users_to_actions": "DU2A", "upsample_actions_to_users": "UA2U"}
USER_WEIGHT_TAG_DICT = {True: "T_MWBA", False : "F_MWBA"}

#================ pred cats ============================
twitter_target_cats = [
"twitter_platform_infoID_pair_nunique_new_users",
"twitter_platform_infoID_pair_nunique_old_users",
"twitter_platform_infoID_pair_num_actions",
]

youtube_target_cats = [
"youtube_platform_infoID_pair_nunique_new_users",
"youtube_platform_infoID_pair_nunique_old_users",
"youtube_platform_infoID_pair_num_actions",
]

target_cat_rename_dict={
"twitter_platform_infoID_pair_nunique_new_users" : "num_new_users",
"twitter_platform_infoID_pair_nunique_old_users": "num_old_users",
"twitter_platform_infoID_pair_num_actions": "num_actions",
"youtube_platform_infoID_pair_nunique_new_users" : "num_new_users",
"youtube_platform_infoID_pair_nunique_old_users": "num_old_users",
"youtube_platform_infoID_pair_num_actions": "num_actions"
}

#====================================================== emd and rh baseline dirs ============================================================
# emd_gt_and_baseline_input_dir = new_gaivi2_input_dir + "CP4-ORGANIZED-V3-FIX-DROPNA/p6-get-emd-baseline-and-gt/P3-GLOBAL-46-EMD-BASELINE-AND-GT-MATERIALS/DEBUG-False-REMOVE_MISSING_PARENT_USERS-True/"
# rh_gt_and_baseline_input_dir = new_gaivi2_input_dir + "CP4-ORGANIZED-V3-FIX-DROPNA/p7-get-rh-baseline-and-gt/P4-46-GLOBAL-RMU-RH-BASELINE-AND-GT-DATA/DEBUG-False-REMOVE_MISSING_PARENTS-True/"

emd_gt_and_baseline_input_dir = "/storage2-mnt/home/f/fmubang/CP5-VAM-Temp-1-6/vam-cp5-paper-materials-v2-1-14-22/data/P8-EMD-STUFF/DEBUG-False-REMOVE_MISSING_PARENTS-True/"
rh_gt_and_baseline_input_dir = "/storage2-mnt/home/f/fmubang/CP5-VAM-Temp-1-6/vam-cp5-paper-materials-v2-1-14-22/data/P7-RH-STUFF/DEBUG-False-REMOVE_MISSING_PARENTS-True/"

#baselines
emd_baseline_results_input_dir=emd_gt_and_baseline_input_dir +"Baseline-Results/"
rh_baseline_results_input_dir=rh_gt_and_baseline_input_dir +"Baseline-Results/"

#====================================================== lookback params ============================================================
print("\nYOUTUBE_HISTORICAL_LOOKBACK_HOURS_LIST")
print(YOUTUBE_HISTORICAL_LOOKBACK_HOURS_LIST)

print("\nTWITTER_HISTORICAL_LOOKBACK_HOURS_LIST")
print(TWITTER_HISTORICAL_LOOKBACK_HOURS_LIST)

#========================IF STATEMENT PARAMS ===================
BASIC_GRAN = GRAN[-1:]

if DEBUG == False:
    INFO_ID_DEBUG=False
    TIMESTEP_DEBUG = False
else:
    INFO_ID_DEBUG=True
    TIMESTEP_DEBUG = True

if INFO_ID_DEBUG == True:
    infoIDs = infoIDs[:2]

if TIMESTEP_DEBUG == True:
    NUM_SIM_PERIODS = 2

NUM_TEST_TIMESTEPS = NUM_SIM_PERIODS * SIM_PERIOD_SIZE

print("\nNUM_TEST_TIMESTEPS")
print(NUM_TEST_TIMESTEPS)

if len(GRAN) == 1:
    TIMESTEPS_PER_UPDATE = 1
else:
    TIMESTEPS_PER_UPDATE = int(GRAN[:-1])

#======================== config param lists ===================
if DEBUG == True:
    INFLUENCE_EPSILON_DIV_VAL_LIST=INFLUENCE_EPSILON_DIV_VAL_LIST[:1]
    user_action_conflict_resolution_options=user_action_conflict_resolution_options[:1]
    NUM_LINK_SAMPLE_ATTEMPTS = 1
    MULT_USER_WEIGHT_BY_AGE_LIST = [True]
    TWITTER_HISTORICAL_LOOKBACK_HOURS_LIST = TWITTER_HISTORICAL_LOOKBACK_HOURS_LIST[:1]
    YOUTUBE_HISTORICAL_LOOKBACK_HOURS_LIST = YOUTUBE_HISTORICAL_LOOKBACK_HOURS_LIST[:1]

#======================== CONFLICT PARAMS ===================
#other
eval_type = "test"
model_type = "XGBoost_Regressor"


#=================================== date stuff =================================
overall_history_start = "03-30-2020"
overall_history_end = "08-17-2020 23:59:59"

overall_test_start = "08-18-2020"
overall_test_end = "08-31-2020 23:59:59"

#======================== INPUT PARAMS ===================
#input stuff
twitter_model_tag = "TR-72"
youtube_model_tag = ""

#========================================== model dirs ========================================
#model pred inputs
# twitter_input_dir = new_gaivi2_input_dir + "CP4-ORGANIZED-V3-FIX-DROPNA/p3-train-models/CP4-M1-FIXED-DROPNA-FTS-DATES-XGBoost_Regressor-DEBUG-False-INFO_ID_DEBUG-False-DAY_AGG-True-agg_timestep_amount-24/twitter/train-2018-12-28-to-2019-02-07 23:59:59-val-2019-02-08-to-2019-02-14 23:59:59-test-2019-02-15-to-2019-03-07 23:59:59/"
# youtube_input_dir = new_gaivi2_input_dir + "CP4-ORGANIZED-V3-FIX-DROPNA/p3-train-models/CP4-M1-FIXED-DROPNA-FTS-DATES-XGBoost_Regressor-DEBUG-False-INFO_ID_DEBUG-False-DAY_AGG-True-agg_timestep_amount-24/youtube/train-2018-12-28-to-2019-02-07 23:59:59-val-2019-02-08-to-2019-02-14 23:59:59-test-2019-02-15-to-2019-03-07 23:59:59/"

twitter_input_dir = "/storage2-mnt/data/fmubang/CP5-VAM-Paper-Stuff-v2-12-27-21/p4-load-and-retrain-v3-on-train-and-val/TRAIN-ON-VAL-ALSO-CP5-M1-PAPER-VOL-TRAIN-DEBUG-False/SAMPLE-WEIGHT-DECAY-1/DATES-XGBoost_Regressor-DEBUG-False-INFO_ID_DEBUG-False-DAY_AGG-True-agg_timestep_amount-24/twitter/train-04-02-2020-to-08-10-2020 23:59:59-val-08-11-2020-to-08-17-2020 23:59:59-test-08-18-2020-to-08-31-2020 23:59:59/"
youtube_input_dir = ""


#model weight dir
#main_history_input_dir = new_gaivi2_input_dir + "CP4-ORGANIZED-V3-FIX-DROPNA/p4-preprocess-ua-data-FIXED/P2-V4-STATS-FIXED-PARENT-ISSUE-LINKS-GLOBAL-46-PARENTS-UA-MODULE-USER-LINK-GT-INFO-DEBUG-False/GRAN-%s-RETWEETS_KEEP_ONLY-True-parents-start-12-24-2018-end-04-04-2019 23:59:59-links-12-24-2018-to-04-04-2019 23:59:59/"%BASIC_GRAN
main_history_input_dir = "/storage2-mnt/home/f/fmubang/CP5-VAM-Temp-1-6/vam-cp5-paper-materials-v2-1-14-22/data/P5-UA-LINK-DATA/DEBUG-False-GRAN-H-RETWEETS_KEEP_ONLY-True-parents-start-03-30-2020-end-08-31-2020 23:59:59-links-03-30-2020-to-08-31-2020 23:59:59/"

#re org ft dir, hourly
#re_org_ft_input_dir = new_gaivi2_input_dir + "CP4-ORGANIZED-V3-FIX-DROPNA/data/V2-WITH-EXO-P1-M1-VOLUME-DATA/DEBUG-False-Dates-12-24-2018-to-04-04-2019 23:59:59/features/"
re_org_ft_input_dir = "/storage2-mnt/data/fmubang/CP5-VAM-Paper-Stuff-3-3/data/P4-VOLUME-DATA/DEBUG-False-Dates-03-30-2020-to-08-31-2020 23:59:59-GRAN-H/"



for SEED in SEEDS:

    #seed
    print("\nSEED: %d"%SEED)
    np.random.seed(SEED)

    #====================================================== main out ============================================================
    seed_main_output_dir = "V2-CP5-PROPER-TOPICS-TWITTER-MODEL-CP5-UA-SIMPLE-LB-DEBUG-%s/SEED-%d/"%(DEBUG,SEED)
    create_output_dir(seed_main_output_dir)

    all_time_tuples =ua.get_test_and_history_tuples(overall_history_start, overall_history_end, overall_test_start, overall_test_end, NUM_SIM_PERIODS, BASIC_GRAN, SIM_PERIOD_SIZE=24)
    # sys.exit(0)
    at_fp = seed_main_output_dir + "all_time_tuples.txt"
    with open(at_fp, "w") as f:
        for a in all_time_tuples:
            f.write(str(a)+"\n")
        print(at_fp)

    #loop
    for INFLUENCE_EPSILON_DIV_VAL in INFLUENCE_EPSILON_DIV_VAL_LIST:
        for ADD_OLD_USERS_TO_NEW_USERS_IF_TOO_MANY in ADD_OLD_USERS_TO_NEW_USERS_IF_TOO_MANY_LIST:
            for user_action_conflict_resolution_option in user_action_conflict_resolution_options:
                for MULT_USER_WEIGHT_BY_AGE in MULT_USER_WEIGHT_BY_AGE_LIST:

                    for platform in platforms:

                        main_output_dir = seed_main_output_dir + platform + "/"
                        create_output_dir(main_output_dir)

                        if platform == "twitter":
                            target_cats = twitter_target_cats
                            LOOKBACK_HOURS_LIST = TWITTER_HISTORICAL_LOOKBACK_HOURS_LIST
                            model_tag = twitter_model_tag
                            main_input_dir = twitter_input_dir
                        else:
                            target_cats = youtube_target_cats
                            LOOKBACK_HOURS_LIST = YOUTUBE_HISTORICAL_LOOKBACK_HOURS_LIST
                            model_tag = youtube_model_tag
                            main_input_dir = youtube_input_dir

                        #setup dir
                        model_input_dir = main_input_dir + model_tag + "/"

                        #debug stuff
                        for LOOKBACK_FACTOR in LOOKBACK_HOURS_LIST:

                            #get cols
                            new_user_count_target_col = "%s_platform_infoID_pair_nunique_new_users"%platform
                            old_user_count_target_col = "%s_platform_infoID_pair_nunique_old_users"%platform
                            num_actions_target_col = "%s_platform_infoID_pair_num_actions"%platform

                            #START TIMER
                            start_time = time()

                            #======================== PARAM AND OUPUT STUFF ===================
                            PARAM_VALS = [overall_test_start, overall_test_end, overall_history_start, overall_history_end,BASIC_GRAN,TIMESTEPS_PER_UPDATE ,GRAN,DEBUG, NUM_JOBS,NUM_LINK_SAMPLE_ATTEMPTS, GRAN,INFLUENCE_EPSILON_DIV_VAL,user_action_conflict_resolution_option,ADD_OLD_USERS_TO_NEW_USERS_IF_TOO_MANY,eval_type,model_type,LOOKBACK_FACTOR, platform]
                            PARAM_NAMES = ["overall_test_start", "overall_test_end", "overall_history_start", "overall_history_test_end","BASIC_GRAN" ,"TIMESTEPS_PER_UPDATE","GRAN","DEBUG","NUM_JOBS", "NUM_LINK_SAMPLE_ATTEMPTS","GRAN","INFLUENCE_EPSILON_DIV_VAL","user_action_conflict_resolution_option","ADD_OLD_USERS_TO_NEW_USERS_IF_TOO_MANY","eval_type","model_type","LOOKBACK_FACTOR", "platform"]
                            param_df = pd.DataFrame(data={"param":PARAM_NAMES, "value":PARAM_VALS})
                            param_df =param_df[["param", "value"]]

                            #make model tag here
                            cr_tag = CR_TAG_DICT[user_action_conflict_resolution_option]
                            mult_tag = USER_WEIGHT_TAG_DICT[MULT_USER_WEIGHT_BY_AGE]
                            ua_model_tag = model_tag + "V-" + str(LOOKBACK_FACTOR) + "U-" + cr_tag + "-" + mult_tag
                            print()
                            print(ua_model_tag)

                            #ua model dir
                            ua_model_output_dir = main_output_dir + ua_model_tag + "/"
                            create_output_dir(ua_model_output_dir)

                            output_dir = ua.make_output_dir_from_params_v2(PARAM_VALS, ua_model_output_dir)

                            #save params
                            param_output_fp = output_dir + "params.csv"
                            param_df.to_csv(param_output_fp, index=False)
                            print(param_output_fp)
                            param_dict = convert_df_2_cols_to_dict(param_df, "param", "value")
                            #save dict
                            param_dict_output_fp = output_dir + "param_dict"
                            save_pickle(param_dict, param_dict_output_fp)


                            #get preds
                            day_agg_input_dir = model_input_dir + "DAY_AGG-False/"
                            pred_fp =  day_agg_input_dir + "Metrics/" + model_type + "/y_%s-%s-metrics.csv"%(eval_type, model_type)
                            pred_df = pd.read_csv(pred_fp)
                            print("\npred_df")
                            print(pred_df)

                            #cleanup pred df
                            pred_without_gt = pred_df.drop("actual", axis=1)
                            print(pred_without_gt)
                            main_cleaned_pred_df_dict = ua.resolve_count_conflicts_v2_add_nodeTime(pred_without_gt, new_user_count_target_col,old_user_count_target_col,num_actions_target_col,user_action_conflict_resolution_option,overall_test_start, overall_test_end, BASIC_GRAN)

                            #for the end
                            final_main_cleaned_pred_df_dict = {}
                            for infoID in infoIDs:
                                final_main_cleaned_pred_df_dict[infoID] = []
                            #======================== count the actions ===================
                            #we need to time each loop
                            ACTION_COUNT_LIST = []
                            TIME_PER_ITERATION_LIST = []
                            HISTORY_SIZE_LIST = []

                            #save dfs here
                            infoID_to_df_list_dict = {}
                            for infoID in infoIDs:
                                infoID_to_df_list_dict[infoID]=[]

                            #save all hack track dfs for new tables
                            all_hack_track_dfs = []

                            #======================== loop per timestep ===================
                            for SIM_PERIOD in range(NUM_SIM_PERIODS):
                                print("\nOn sim period %d of %d"%(SIM_PERIOD+1, NUM_SIM_PERIODS))

                                cur_time_tuple = all_time_tuples[SIM_PERIOD]
                                print("\ncur_time_tuple")

                                #history info
                                history_start = cur_time_tuple[0]
                                history_end = cur_time_tuple[1]
                                test_start = cur_time_tuple[2]
                                test_end = cur_time_tuple[3]

                                #======================================fix according to LB factor======================================
                                test_start = pd.to_datetime(test_start, utc=True)
                                test_end = pd.to_datetime(test_end, utc=True)
                                history_start = test_start - pd.to_timedelta(LOOKBACK_FACTOR, unit=BASIC_GRAN)

                                history_start = pd.to_datetime(history_start, utc=True)
                                history_end = pd.to_datetime(history_end, utc=True)

                                print("\nFixed dates for period %d"%SIM_PERIOD)
                                print("history:")
                                print(history_start)
                                print(history_end)

                                print("test:")
                                print(test_start)
                                print(test_end)
                                #===============================================================================================
                                # #time step materials
                                #get full history
                                full_history_record_df = ua.load_full_model_weight_df_v2_check_dupes(main_history_input_dir,platform,infoIDs,hyp_dict,history_start, history_end)
                                print("\nfull_history_record_df")
                                print(full_history_record_df)

                                #parent bdate dicts
                                user_to_bdate_dict = convert_df_2_cols_to_dict(full_history_record_df, "nodeUserID", "user_birthdate")
                                parent_to_bdate_dict = convert_df_2_cols_to_dict(full_history_record_df, "parentUserID", "parent_birthdate")

                                sim_period_materials_dir = output_dir + "Sim-Period-Materials/SIM_PERIOD-%d-of-%d/"%(SIM_PERIOD+1, NUM_SIM_PERIODS)
                                create_output_dir(sim_period_materials_dir)

                                #get cur clean pred dict
                                cleaned_pred_df_dict = ua.get_and_save_cur_clean_pred_df_dict(output_dir, SIM_PERIOD, SIM_PERIOD_SIZE, NUM_SIM_PERIODS, test_start, test_end, main_cleaned_pred_df_dict, infoIDs, hyp_dict)

                                #============================ setup data structs for saving important info ============================

                                #track the test date
                                CUR_DATE = test_start
                                CUR_DATE = pd.to_datetime(CUR_DATE, utc=True)

                                #track results
                                track_dir = output_dir + "Tracker/SIM-PERIOD-%d-of-%d/"%(SIM_PERIOD+1, NUM_SIM_PERIODS)
                                create_output_dir(track_dir)

                                #count new users
                                NEW_USER_COUNTER_DICT = {}
                                for infoID in infoIDs:
                                    NEW_USER_COUNTER_DICT[infoID] = 0

                                #============================ Start looping by timestep ============================
                                print("\nSimulating by timestep...")

                                cur_history_start = pd.to_datetime(history_start, utc=True)
                                cur_history_end = pd.to_datetime(history_end, utc=True)

                                for timestep in range(1, SIM_PERIOD_SIZE + 1):
                                    print("\nCurrent timestep: %d"%timestep)

                                    #track progress
                                    track_fp = track_dir + "Timestep-%d.txt"%timestep
                                    with open(track_fp, "w") as f:
                                        print()

                                    total_action_count_this_iter = 0
                                    #================================== get current history ===========================================

                                    if timestep > 1:

                                        #======================= bdate and age stuff =========================================

                                        print()
                                        print(cur_pred_links)
                                        cur_pred_links["user_birthdate"] = cur_pred_links["nodeUserID"].map(user_to_bdate_dict)
                                        cur_pred_links["parent_birthdate"]= cur_pred_links["parentUserID"].map(parent_to_bdate_dict)

                                        #for children
                                        nan_temp = cur_pred_links[cur_pred_links["user_birthdate"].isnull()]
                                        print()
                                        print(CUR_DATE)
                                        print(nan_temp)
                                        for idx,row in nan_temp.iterrows():
                                            nodeUserID = row["nodeUserID"]
                                            user_to_bdate_dict[nodeUserID] = CUR_DATE

                                        #for PARENTS
                                        nan_temp = cur_pred_links[cur_pred_links["parent_birthdate"].isnull()]
                                        print()
                                        print(CUR_DATE)
                                        print(nan_temp)
                                        for idx,row in nan_temp.iterrows():
                                            parentUserID = row["parentUserID"]
                                            parent_to_bdate_dict[parentUserID] = CUR_DATE

                                        print()

                                        cur_pred_links["user_birthdate"] = cur_pred_links["nodeUserID"].map(user_to_bdate_dict)
                                        cur_pred_links["parent_birthdate"]= cur_pred_links["parentUserID"].map(parent_to_bdate_dict)
                                        cur_pred_links["user_birthdate"]=pd.to_datetime(cur_pred_links["user_birthdate"], utc=True)
                                        cur_pred_links["parent_birthdate"]=pd.to_datetime(cur_pred_links["parent_birthdate"], utc=True)

                                        cur_pred_links["user_age"] = (CUR_DATE - cur_pred_links["user_birthdate"]).astype('timedelta64[h]')
                                        cur_pred_links["parent_age"] = (CUR_DATE - cur_pred_links["parent_birthdate"]).astype('timedelta64[h]')

                                        print(cur_pred_links)

                                        #================================================================================================

                                        cur_history_start = cur_history_start + pd.to_timedelta(1, unit=BASIC_GRAN)
                                        cur_history_end = cur_history_end + pd.to_timedelta(1, unit=BASIC_GRAN)
                                        print("\nTimestep %d cur history dates"%timestep)
                                        print(cur_history_start)
                                        print(cur_history_end)

                                        partial_history_record_df = config_df_by_dates(full_history_record_df, cur_history_start, cur_history_end)
                                        print("\npartial_history_record_df")
                                        print(partial_history_record_df)

                                        #update table
                                        full_history_record_df_with_preds = pd.concat([full_history_record_df_with_preds,partial_history_record_df, cur_pred_links]).reset_index(drop=True)
                                        print("\nfull_history_record_df_with_preds")
                                        print(full_history_record_df_with_preds)
                                        print("\nafter drop dupes")
                                        full_history_record_df_with_preds = full_history_record_df_with_preds.drop_duplicates()
                                        full_history_record_df_with_preds["nodeTime"] = pd.to_datetime(full_history_record_df_with_preds["nodeTime"], utc=True)
                                        full_history_record_df_with_preds = full_history_record_df_with_preds.sort_values("nodeTime").reset_index(drop=True)
                                        print(full_history_record_df_with_preds)

                                        cols = list(full_history_record_df_with_preds)
                                        for col in cols:
                                            print(col)
                                            print()
                                            print(full_history_record_df_with_preds[col])

                                        # sys.exit(0)

                                    else:
                                        full_history_record_df_with_preds =full_history_record_df

                                    history_record_df= ua.get_cur_history_record_df_v2_simple(infoIDs, full_history_record_df_with_preds, cur_history_start, cur_history_end)
                                    print("\nhistory_record_df")
                                    print(history_record_df)

                                    # if timestep > 3:
                                    #     print(history_record_df)
                                    #     sys.exit(0)

                                    #increment history size
                                    HISTORY_SIZE_LIST.append(history_record_df.shape[0])

                                    #=================================#======================== get various UA tables ===============================#=================================
                                    edge_weight_df, infoID_to_new_user_edge_weight_dist_table_dict, \
                                    infoID_to_all_users_edge_weight_dist_table_dict, \
                                    infoID_to_all_users_weight_table_dict,infoID_to_new_users_only_weight_table_dict,infoID_hack_track_df= ua.get_various_ua_tables_v3_with_age(history_record_df, infoIDs,INFLUENCE_EPSILON_DIV_VAL, timestep, SIM_PERIOD, output_dir)

                                    all_hack_track_dfs.append(infoID_hack_track_df)
                                    #infoID_to_all_users_weight_table_dict,infoID_to_new_users_only_weight_table_dict = ua.get_various_ua_tables(history_record_df, infoIDs,INFLUENCE_EPSILON_DIV_VAL)
                                    #infoID_to_all_users_weight_table_dict,infoID_to_new_users_only_weight_table_dict = ua.get_various_ua_tables_v2_NEW_USER_TABLE_HACK(history_record_df, infoIDs,INFLUENCE_EPSILON_DIV_VAL, timestep, SIM_PERIOD)
                                    #======================== get infoID materials ========================
                                    infoID_materials_list =ua.get_infoID_material_list_v2_more_params(infoIDs, NEW_USER_COUNTER_DICT,hyp_dict,cleaned_pred_df_dict,timestep,infoID_to_all_users_weight_table_dict,
                                                                infoID_to_new_users_only_weight_table_dict,infoID_to_all_users_edge_weight_dist_table_dict,infoID_to_new_user_edge_weight_dist_table_dict, MULT_USER_WEIGHT_BY_AGE)

                                    #run para func
                                    iter_start_time = time()
                                    infoID_link_pred_records_list = Parallel(n_jobs=NUM_JOBS)(delayed(ua.get_user_sim_pred_records_for_cur_infoID_v4_old_user_age_sample)(infoID_materials_tuple,NUM_LINK_SAMPLE_ATTEMPTS,ADD_OLD_USERS_TO_NEW_USERS_IF_TOO_MANY, timestep, DEBUG_PRINT) for infoID_materials_tuple in infoID_materials_list)
                                    #infoID_link_pred_records_list = Parallel(n_jobs=NUM_JOBS)(delayed(ua.get_user_sim_pred_records_for_cur_infoID_v3_update_volume_counts)(infoID_materials_tuple,NUM_LINK_SAMPLE_ATTEMPTS,ADD_OLD_USERS_TO_NEW_USERS_IF_TOO_MANY, timestep, DEBUG_PRINT) for infoID_materials_tuple in infoID_materials_list)
                                    iter_end_time = time()

                                    cur_updated_clean_pred_df_dict = {}

                                    #add results to dict
                                    for infoID_link_pred_records_tuple in infoID_link_pred_records_list:
                                        infoID_link_pred_records = infoID_link_pred_records_tuple[0]
                                        infoID = infoID_link_pred_records_tuple[1]
                                        CUR_INFO_ID_NEW_USER_COUNTER = infoID_link_pred_records_tuple[2]
                                        updated_volume_pred_df = infoID_link_pred_records_tuple[3]
                                        updated_volume_pred_df["SIM_PERIOD"] = SIM_PERIOD + 1
                                        updated_volume_pred_df["sim_period_timestep"] = timestep
                                        updated_volume_pred_df["timestep"] = ((updated_volume_pred_df["SIM_PERIOD"] -1) * SIM_PERIOD_SIZE )  +  updated_volume_pred_df["sim_period_timestep"]
                                        final_main_cleaned_pred_df_dict[infoID].append(updated_volume_pred_df)
                                        cur_updated_clean_pred_df_dict[infoID] = updated_volume_pred_df
                                        infoID_link_pred_records["nodeTime"] = CUR_DATE
                                        infoID_link_pred_records["sim_period_timestep"] = timestep

                                        infoID_link_pred_records["SIM_PERIOD"] = SIM_PERIOD + 1
                                        infoID_link_pred_records["timestep"] =((infoID_link_pred_records["SIM_PERIOD"] -1) * SIM_PERIOD_SIZE )  +  infoID_link_pred_records["sim_period_timestep"]
                                        print(infoID_link_pred_records)

                                        #update stuff
                                        infoID_to_df_list_dict[infoID].append(infoID_link_pred_records)
                                        total_action_count_this_iter+=infoID_link_pred_records.shape[0]
                                        NEW_USER_COUNTER_DICT[infoID]= CUR_INFO_ID_NEW_USER_COUNTER

                                    #for the history table
                                    cur_pred_links = ua.get_cur_pred_links_as_df(infoID_to_df_list_dict, infoIDs, timestep)

                                    #this stupid col keeps causing so much trouble
                                    if "edge_weight" in list(cur_pred_links):
                                        cur_pred_links = cur_pred_links.drop("edge_weight", axis=1)

                                    #validate preds
                                    print("\nVALIDATING CUR PRED LINKS!")
                                    # print(infoID)
                                    print("SIM PERIOD: %d, timestep: %d"%(SIM_PERIOD+1, timestep))
                                    ua.validate_cur_pred_links_v2_save_problem_file(infoIDs, cur_pred_links, timestep,cur_updated_clean_pred_df_dict,output_dir ,clean_pred_ts_col="sim_period_timestep", cur_pred_links_ts_col="sim_period_timestep")

                                    #get dates
                                    CUR_DATE = CUR_DATE + pd.to_timedelta(1, unit=BASIC_GRAN )
                                    print("\nCUR_DATE")
                                    print(CUR_DATE)

                                    #update time info

                                    ITER_TIME = iter_end_time - iter_start_time
                                    ITER_TIME_IN_MIN = ITER_TIME/60.0
                                    ACTION_COUNT_LIST.append(total_action_count_this_iter)
                                    TIME_PER_ITERATION_LIST.append(ITER_TIME_IN_MIN)

                            #create final updated clean pred df
                            final_clean_output_dir =output_dir + "Modified-Clean-Preds/"
                            create_output_dir(final_clean_output_dir)
                            print("\nCreating final clean volume pred dfs...")
                            for infoID in infoIDs:
                                hyp_infoID = hyp_dict[infoID]

                                final_clean_vol_df = pd.concat(final_main_cleaned_pred_df_dict[infoID]).sort_values("timestep").reset_index(drop=True)
                                print("\nfinal_clean_vol_df")
                                final_main_cleaned_pred_df_dict[infoID]=final_clean_vol_df

                                print(infoID)
                                print(final_clean_vol_df)

                                output_fp = final_clean_output_dir + hyp_infoID + ".csv"
                                final_clean_vol_df.to_csv(output_fp, index=False)
                                print(output_fp)



                            # sys.exit(0)
                            #combine for each infoID
                            print("\nCombining and saving each infoID df...")
                            sim_output_dir = output_dir + "Simulations/"
                            create_output_dir(sim_output_dir)
                            for infoID in infoIDs:
                                print("\nGetting %s pred table"%infoID)
                                cur_infoID_df = pd.concat(infoID_to_df_list_dict[infoID]).reset_index(drop=True)
                                print("\ncur_infoID_df")
                                print(cur_infoID_df)

                                #fix edge counts and drop dupes
                                cur_infoID_df["edge_weight_this_timestep"] = cur_infoID_df.groupby(["nodeUserID","parentUserID","informationID", "timestep"])["parentUserID"].transform("count")
                                cur_infoID_df = cur_infoID_df[["nodeTime" ,"SIM_PERIOD","sim_period_timestep","timestep" ,"informationID", "nodeUserID", "parentUserID", "edge_weight_this_timestep", "nodeUserID_is_new", "parentUserID_is_new"]]
                                cur_infoID_df = cur_infoID_df.drop_duplicates().reset_index(drop=True)

                                #validate preds
                                print("\nFINAL VALIDATION: %s"%infoID)
                                for t in range(1, NUM_TEST_TIMESTEPS + 1):
                                    #ua.validate_cur_pred_links([infoID], cur_infoID_df, t,cleaned_pred_df_dict, clean_pred_ts_col="timestep", cur_pred_links_ts_col="timestep")
                                    ua.validate_cur_pred_links_v2_save_problem_file([infoID], cur_infoID_df, t,final_main_cleaned_pred_df_dict ,output_dir ,clean_pred_ts_col="timestep", cur_pred_links_ts_col="timestep")


                                #save it
                                hyp_infoID = hyp_dict[infoID]
                                output_fp = sim_output_dir + hyp_infoID + ".csv"
                                cur_infoID_df.to_csv(output_fp, index=False)
                                print(output_fp)


                            end_time = time()
                            total_time = end_time - start_time
                            time_in_min = total_time/60.0
                            time_in_min = np.round(time_in_min, 3)
                            print("\ntime_in_min")
                            print(time_in_min)
                            time_fp = output_dir + "time_in_min.txt"
                            with open(time_fp, "w") as f:
                                f.write(str(time_in_min))

                            model_tag_input_dir = output_dir
                            # infoID_to_model_sim_dict = ua.agg_sim_results(infoID_to_model_sim_dict, infoIDs)
                            # model_tag_input_dir = output_dir
                            # infoID_to_model_sim_dict = ua.agg_sim_results(infoID_to_model_sim_dict, infoIDs)

                            #=============================== hack track stuff ============================================================
                            final_hack_df = pd.concat(all_hack_track_dfs).reset_index(drop=True)
                            num_trials = final_hack_df.shape[0]
                            final_hack_df=final_hack_df[final_hack_df["used_hack"]==1].reset_index(drop=True)
                            num_hacks = final_hack_df.shape[0]
                            hack_output_dir = output_dir + "New-Table-Hack-Info/"
                            create_output_dir(hack_output_dir)

                            output_fp = hack_output_dir + "new_table_hack_df.csv"
                            final_hack_df.to_csv(output_fp , index=False)
                            print(output_fp)

                            result_str = "num hacks: %d, num_trials: %d, freq: %.4f"%(num_hacks, num_trials, num_hacks/float(num_trials))
                            output_fp = hack_output_dir + "new-table-hack-summary.txt"
                            with open(output_fp, "w") as f:
                                f.write(result_str)
                                print(result_str)
                            print()
                            print(output_fp)

                            #=============================== time stuff ============================================================
                            ua.make_ua_time_plot_and_df(TIME_PER_ITERATION_LIST, ACTION_COUNT_LIST, output_dir,"action_count")
                            ua.make_ua_time_plot_and_df(TIME_PER_ITERATION_LIST, HISTORY_SIZE_LIST, output_dir,"history_size")

                            #==================== get emd stuff ================
                            if GET_METRICS == True:
                                #emd_output_dir = get_emd_metrics_v2_SIMPLE(infoIDs, model_tag_input_dir,hyp_dict,TIMESTEP_DEBUG,emd_gt_and_baseline_input_dir,platform, model_tag)
                                emd_output_dir = nmf.get_emd_metrics_v5_CLEAN(infoIDs, model_tag_input_dir,output_dir ,hyp_dict,emd_gt_and_baseline_input_dir,platform, model_tag,NUM_TEST_TIMESTEPS,REMOVE_MISSING_PARENT_USERS=REMOVE_MISSING_PARENT_USERS)
                                # main_emd_results_input_dir = main_emd_results_output_dir
                                emd_input_dir = emd_output_dir
                                print("\nemd_input_dir")
                                print(emd_input_dir)

                                final_emd_df  = nmf.compare_model_emd_results_to_baseline_v2_SIMPLE(emd_baseline_results_input_dir, model_tag, emd_input_dir,  platform, infoIDs,hyp_dict)
                                print("\nfinal_emd_df")
                                print(final_emd_df)


                            #====================== get rh stuff =============================
                            if GET_METRICS == True:
                                rh_output_dir = nmf.eval_model_rh_v5_CLEAN(infoIDs, model_tag_input_dir,output_dir ,hyp_dict,rh_gt_and_baseline_input_dir,platform, model_tag,NUM_TEST_TIMESTEPS,REMOVE_MISSING_PARENT_USERS=REMOVE_MISSING_PARENT_USERS)
                                #rh_output_dir = eval_model_rh_v2_SIMPLE(infoIDs, model_tag_input_dir,hyp_dict,TIMESTEP_DEBUG,rh_gt_and_baseline_input_dir,platform, model_tag,main_output_dir)
                                rh_input_dir = rh_output_dir
                                final_rh_df = nmf.compare_model_rh_results_to_baseline_v2_SIMPLE(rh_baseline_results_input_dir, model_tag, rh_input_dir,  platform, infoIDs,hyp_dict)
                                print("\nfinal_rh_df")
                                print(final_rh_df)


                            # rh_df = ua.get_and_save_rh_results_v2_timestep_info(infoID_to_model_sim_dict, platform, rh_gt_and_baseline_input_dir ,output_dir, model_tag, REMOVE_MISSING_PARENT_USERS, infoIDs, hyp_dict)
                            # if DEBUG == True:
                            #     sys.exit(0)











