from basic_utils import *
import numpy as np 
import os,sys
import pandas as pd

import os,sys
import pandas as pd 
import numpy as np 
from functools import reduce
# import sample_gen_utils as sgu
from sklearn.metrics import ndcg_score
from network_metric_funcs import *
from multiprocessing import Process
import multiprocessing
import multiprocessing as mp
from time import time
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import json
import calendar

def validate_cur_pred_links_v2_save_problem_file(infoIDs, main_cur_pred_links, timestep,cleaned_pred_df_dict, output_dir, clean_pred_ts_col="timestep", cur_pred_links_ts_col="sim_period_timestep"):

    print("\nValidating preds...")

    # main_cur_pred_links = cur_pred_links.

    problem_dir = output_dir + "Problem-Files/"
    create_output_dir(problem_dir)

    hyp_dict = hyphenate_infoID_dict(infoIDs)

    for infoID in infoIDs:

        error_found = False

        #get clean preds
        cur_pred_df = cleaned_pred_df_dict[infoID]
        cur_pred_df = cur_pred_df[cur_pred_df[clean_pred_ts_col]==timestep]
        print("\ncur_pred_df")
        print(cur_pred_df)

        target_pred_num_actions = cur_pred_df["num_actions"].iloc[-1]
        target_pred_num_new_users = cur_pred_df["num_new_users"].iloc[-1]
        target_pred_num_old_users = cur_pred_df["num_old_users"].iloc[-1]

        #get links
        cur_pred_links=main_cur_pred_links[main_cur_pred_links[cur_pred_links_ts_col]==timestep].copy()
        cur_pred_links=cur_pred_links[cur_pred_links["informationID"]==infoID].copy()
        print("\ncur_pred_links")
        print(cur_pred_links)

        # print(cur_pred_links[["SIM_PERIOD", "sim_period_timestep", "timestep", "nodeTime"]])



        ua_pred_num_actions = cur_pred_links["edge_weight_this_timestep"].sum()
        ua_users = cur_pred_links[["nodeUserID", "nodeUserID_is_new"]].drop_duplicates()
        ua_pred_num_new_users = ua_users[ua_users["nodeUserID_is_new"]==1].shape[0]
        ua_pred_num_old_users = ua_users[ua_users["nodeUserID_is_new"]==0].shape[0]

        print("\n%s target clean pred counts (actions, new, old) "%infoID)
        print(target_pred_num_actions)
        print(target_pred_num_new_users)
        print(target_pred_num_old_users)

        print("\n%s UA pred counts (actions, new, old)"%infoID)
        print(ua_pred_num_actions)
        print(ua_pred_num_new_users)
        print(ua_pred_num_old_users)

        if target_pred_num_actions != ua_pred_num_actions:
            print("\nError! target_pred_num_actions != ua_pred_num_actions")
            error_found = True

        if target_pred_num_new_users != ua_pred_num_new_users:
            print("\nError! target_pred_num_actions != ua_pred_num_actions")
            error_found = True

        if target_pred_num_old_users != ua_pred_num_old_users:
            print("\nError! target_pred_num_old_users != ua_pred_num_old_users")
            error_found = True

        if error_found == True:

            log_df = pd.DataFrame(data={"target_cat":["num_actions", "num_new_users", "num_old_users"]})
            log_df["clean_pred_size"] = [target_pred_num_actions, target_pred_num_new_users, target_pred_num_old_users]
            log_df["ua_pred_size"] =  [ua_pred_num_actions, ua_pred_num_new_users, ua_pred_num_old_users]
            log_df["diff"] = log_df["clean_pred_size"] - log_df["ua_pred_size"]

            log_df = log_df[["target_cat", "clean_pred_size", "ua_pred_size", "diff"]]

            print("\nlog_df")
            print(log_df)

            output_fp = problem_dir  + hyp_dict[infoID] + "-log.csv"
            log_df.to_csv(output_fp, index=False)
            print(output_fp)

            output_fp = problem_dir + hyp_dict[infoID] +"-cur_pred_links.csv"
            cur_pred_links.to_csv(output_fp, index=False)
            print(output_fp)

            print("\nSaved error info for debugging. Now terminating...")
            sys.exit(0)


    return

def validate_cur_pred_links(infoIDs, main_cur_pred_links, timestep,cleaned_pred_df_dict, clean_pred_ts_col="timestep", cur_pred_links_ts_col="sim_period_timestep"):

    print("\nValidating preds...")

    # main_cur_pred_links = cur_pred_links.

    for infoID in infoIDs:

        #get clean preds
        cur_pred_df = cleaned_pred_df_dict[infoID]
        cur_pred_df = cur_pred_df[cur_pred_df[clean_pred_ts_col]==timestep]
        print("\ncur_pred_df")
        print(cur_pred_df)

        target_pred_num_actions = cur_pred_df["num_actions"].iloc[-1]
        target_pred_num_new_users = cur_pred_df["num_new_users"].iloc[-1]
        target_pred_num_old_users = cur_pred_df["num_old_users"].iloc[-1]

        #get links
        cur_pred_links=main_cur_pred_links[main_cur_pred_links[cur_pred_links_ts_col]==timestep].copy()
        cur_pred_links=cur_pred_links[cur_pred_links["informationID"]==infoID].copy()
        print("\ncur_pred_links")
        print(cur_pred_links)

        # print(cur_pred_links[["SIM_PERIOD", "sim_period_timestep", "timestep", "nodeTime"]])



        ua_pred_num_actions = cur_pred_links["edge_weight_this_timestep"].sum()
        ua_users = cur_pred_links[["nodeUserID", "nodeUserID_is_new"]].drop_duplicates()
        ua_pred_num_new_users = ua_users[ua_users["nodeUserID_is_new"]==1].shape[0]
        ua_pred_num_old_users = ua_users[ua_users["nodeUserID_is_new"]==0].shape[0]

        print("\n%s target clean pred counts (actions, new, old) "%infoID)
        print(target_pred_num_actions)
        print(target_pred_num_new_users)
        print(target_pred_num_old_users)

        print("\n%s UA pred counts (actions, new, old)"%infoID)
        print(ua_pred_num_actions)
        print(ua_pred_num_new_users)
        print(ua_pred_num_old_users)

        if target_pred_num_actions != ua_pred_num_actions:
            print("\nError! target_pred_num_actions != ua_pred_num_actions")
            sys.exit(0)

        if target_pred_num_new_users != ua_pred_num_new_users:
            print("\nError! target_pred_num_actions != ua_pred_num_actions")
            sys.exit(0)

        if target_pred_num_old_users != ua_pred_num_old_users:
            print("\nError! target_pred_num_old_users != ua_pred_num_old_users")
            sys.exit(0)

    return

def get_cur_pred_links_as_df_v2_simple(infoID_to_df_list_dict, infoIDs, timestep):
    all_cur_pred_links = []
    for infoID in infoIDs:
        cur_pred_links = infoID_to_df_list_dict[infoID][-1]

        cur_pred_links["nodeUserID"] = [ user.replace("new_synthetic_user", "synthetic_user_ts_%d"%timestep) if "new_synthetic_user" in user else user for user in list(cur_pred_links["nodeUserID"])]
        cur_pred_links["parentUserID"] = [ user.replace("new_synthetic_user", "synthetic_user_ts_%d"%timestep) if "new_synthetic_user" in user else user for user in list(cur_pred_links["parentUserID"])]
        # cur_pred_links = cur_pred_links.rename(columns={"edge_weight":"edge_weight_this_timestep"})

        cur_pred_links["nodeUserID_num_actions_this_timestep"] = cur_pred_links.groupby(["nodeUserID", "informationID"])["nodeUserID"].transform("count")
        cur_pred_links["edge_weight_this_timestep"] = cur_pred_links.groupby(["nodeUserID","parentUserID" ,"informationID"])["nodeUserID"].transform("count")
        cur_pred_links = cur_pred_links.drop_duplicates().reset_index(drop=True)
        print("\ncur_pred_links")
        print(cur_pred_links)

        all_cur_pred_links.append(cur_pred_links)

    cur_pred_links = pd.concat(all_cur_pred_links)
    print("\ncur_pred_links")
    print(cur_pred_links)

    return cur_pred_links

#fix this!!!!
def get_and_save_rh_results_v2_timestep_info(infoID_to_model_sim_dict, platform, gt_and_baseline_input_dir ,output_dir, model_tag, REMOVE_MISSING_USERS, infoIDs, hyp_dict):
    
    gt_input_dir = gt_and_baseline_input_dir + platform + "/RHD-Materials/Ground-Truth-Indegrees/"

    #gt_input_dir = gt_and_baseline_input_dir +"Ground-Truth-Results/" + platform + "/In-Degrees-by-Timestep"

    rh_output_dir = output_dir + "RHD-Materials/Model-Results/"
    create_output_dir(rh_output_dir)
    all_rh_results = []
    for infoID in infoIDs:
        hyp_infoID = hyp_dict[infoID]
        sdf = infoID_to_model_sim_dict[infoID]

        gfp = gt_input_dir + hyp_infoID + ".csv"
        gdf = pd.read_csv(gfp)

        print("\ngdf")
        print(gdf)

        print("\nsdf")
        print(sdf)

        sys.exit(0)



        model_indegree_dist_df = get_indegree_dist_df(sdf)
        # gt_indegree_dist_df = get_indegree_dist_df(gdf)

        print("\nmodel_indegree_dist_df")
        print(model_indegree_dist_df)

        if REMOVE_MISSING_USERS == True:
            model_indegree_dist_df = model_indegree_dist_df[model_indegree_dist_df["nodeUserID"] != "missing_parentUserID"]
            # gt_indegree_dist_df = gt_indegree_dist_df[gt_indegree_dist_df["parentUserID"] != "missing_parentUserID"]


        output_fp = rh_output_dir + hyp_infoID + ".csv"
        model_indegree_dist_df.to_csv(output_fp , index=False)
        print(output_fp)

        cur_rh = rh_distance(gdf["nodeUserID_in_degree"],model_indegree_dist_df["nodeUserID_in_degree"])
        all_rh_results.append(cur_rh)

    print(infoIDs)
    print(all_rh_results)

    rh_df = pd.DataFrame(data={"infoID":infoIDs, "RHD":all_rh_results})
    rh_df = rh_df[["infoID", "RHD"]]

    print("\nrh_df")
    print(rh_df)

    output_fp = rh_output_dir + "RHD-Results.csv"
    rh_df.to_csv(output_fp, index=False)
    print(output_fp)

    return rh_df

def get_and_save_cur_clean_pred_df_dict(output_dir, SIM_PERIOD, SIM_PERIOD_SIZE, NUM_SIM_PERIODS, test_start, test_end, cleaned_pred_df_dict, infoIDs, hyp_dict):
    #fix clean preds
    cur_clean_pred_df_dict = {}
    for infoID in infoIDs:
        pred_df = cleaned_pred_df_dict[infoID]

        pred_df = config_df_by_dates(pred_df, test_start, test_end, "nodeTime")
        pred_df["timestep"] = [i for i in range(1, SIM_PERIOD_SIZE+1)]
        print("\ncur %s pred"%infoID)
        print(pred_df)

        # if SIM_PERIOD > 0:
        #     sys.exit(0)

        cur_clean_pred_df_dict[infoID] = pred_df

    #save clean preds
    clean_output_dir = output_dir + "Cleaned-Pred-Count-Files/SIM-PERIOD-%d-of-%d/"%(SIM_PERIOD+1, NUM_SIM_PERIODS)
    create_output_dir(clean_output_dir)
    for infoID in infoIDs:
        hyp_infoID = hyp_dict[infoID]
        clean_df = cur_clean_pred_df_dict[infoID]
        print(clean_df)
        output_fp = clean_output_dir + hyp_infoID + ".csv"
        clean_df.to_csv(output_fp, index=False)
        print(output_fp)

    return cur_clean_pred_df_dict



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

def make_output_dir_from_params_v2(PARAM_VALS, main_output_dir):

  PARAM_TAG = ""
  for p in PARAM_VALS:
      PARAM_TAG = PARAM_TAG + "-" + str(p)
  print(PARAM_TAG)
  output_dir =  main_output_dir + PARAM_TAG + "/"
  if not os.path.exists(output_dir):
      os.makedirs(output_dir)

  return output_dir


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

def load_full_model_weight_df_v2_check_dupes(model_weight_dir,platform,infoIDs,hyp_dict,weight_start, weight_end):

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

        size_before = model_weight_df.shape[0]
        model_weight_df=model_weight_df[~model_weight_df["nodeUserID"].isnull()].reset_index(drop=True)
        size_after = model_weight_df.shape[0]

        if size_before != size_after:
            print("\nError! nodeUserID size_before != size_after")
            print(size_before)
            print(size_after)
            sys.exit(0)

        size_before = model_weight_df.shape[0]
        model_weight_df=model_weight_df[~model_weight_df["parentUserID"].isnull()].reset_index(drop=True)
        size_after = model_weight_df.shape[0]

        if size_before != size_after:
            print("\nError! parentUserID size_before != size_after")
            print(size_before)
            print(size_after)
            sys.exit(0)

        all_model_weight_dfs.append(model_weight_df)

    model_weight_df = pd.concat(all_model_weight_dfs).reset_index(drop=True)
    print("\nFull model_weight_df")
    print(model_weight_df)

    cols = list(model_weight_df)
    print("\nModel weight cols")
    for col in cols:
        print(col)

    return model_weight_df

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


def get_user_and_parent_birthdates(df):
    df["nodeTime"] = pd.to_datetime(df["nodeTime"], utc=True)
    df["nodeUserID_birthdate"] = df.groupby(["nodeUserID"])["nodeTime"].transform("min")
    user_to_bday_dict = convert_df_2_cols_to_dict(df, "nodeUserID", "nodeUserID_birthdate")
    df["parentUserID_birthdate"] = df["parentUserID"].map(user_to_bday_dict)

    df["parentUserID_birthdate"] = pd.to_datetime(df["parentUserID_birthdate"], utc=True)
    df["parentUserID_birthdate"] = df["parentUserID_birthdate"].fillna(df["nodeTime"].min())
    df["nodeUserID_is_new"] = [1 if nodeTime==birthdate else 0 for nodeTime,birthdate in zip(df["nodeTime"],df["nodeUserID_birthdate"])]
    df["parentUserID_is_new"] = [1 if nodeTime==birthdate else 0 for nodeTime,birthdate in zip(df["nodeTime"],df["parentUserID_birthdate"])]

    return df

def get_ua_ndcg_v3_para_v2_temporal(infoID_to_gt_dict, infoID_to_model_sim_dict, edge_weight_col, 
    WEIGHTED_LIST,main_output_dir,platform, infoIDs, hyp_dict, degree_types, user_types, 
    GRAN,REMOVE_MISSING_USERS,child_ndcg_col="nodeUserID_ndcg", parent_ndcg_col="parentUserID_ndcg", NUM_JOBS=20):

    ndcg_keep_cols = ["nodeUserID", "parentUserID"]
    ndcg_output_dir = main_output_dir + "/NDCG-Results/"
    create_output_dir(ndcg_output_dir)
    all_weighted_dfs = []
    all_unweighted_dfs = []

    ndcg_infoID_output_dir = ndcg_output_dir + "InfoID-Results/"
    create_output_dir(ndcg_infoID_output_dir)

    all_ndcg_dfs = []
    for infoID in infoIDs:

        #lists for df
        user_type_df_list = []
        dt_df_list = []
        weight_df_list = []
        ndcg_df_list = []
        infoID_df_list = []

        hyp_infoID = hyp_dict[infoID]

        #get dfs
        sdf = infoID_to_model_sim_dict[infoID]
        gdf = infoID_to_gt_dict[infoID]


        sdf[child_ndcg_col] = [user if new_flag==0 else "new_user" for user, new_flag in zip(sdf["nodeUserID"],sdf["nodeUserID_is_new"])]
        sdf[parent_ndcg_col] = [parent if new_flag==0 else "new_user" for parent,new_flag in zip(sdf["parentUserID"],sdf["parentUserID_is_new"])]

        gdf[child_ndcg_col] = [user if new_flag==0 else "new_user" for user, new_flag in zip(gdf["nodeUserID"],gdf["nodeUserID_is_new"])]
        gdf[parent_ndcg_col] = [parent if new_flag==0 else "new_user" for parent,new_flag in zip(gdf["parentUserID"],gdf["parentUserID_is_new"])]

        if REMOVE_MISSING_USERS == True:
            gdf = gdf[gdf[child_ndcg_col] != "missing_parentUserID"]
            gdf = gdf[gdf[parent_ndcg_col] != "missing_parentUserID"]
            sdf = sdf[sdf[child_ndcg_col] != "missing_parentUserID"]
            sdf = sdf[sdf[parent_ndcg_col] != "missing_parentUserID"]


        gdf["edge_weight"] = gdf.groupby([child_ndcg_col, parent_ndcg_col, "nodeTime"])[parent_ndcg_col].transform("count")
        sdf["edge_weight"] = sdf.groupby([child_ndcg_col, parent_ndcg_col, "nodeTime"])[parent_ndcg_col].transform("count")

        gdf = gdf[["nodeTime", child_ndcg_col, parent_ndcg_col, "edge_weight"]].drop_duplicates().reset_index(drop=True)
        sdf = sdf[["nodeTime", child_ndcg_col, parent_ndcg_col, "edge_weight"]].drop_duplicates().reset_index(drop=True)

        gdf = gdf.sort_values("edge_weight", ascending=False).reset_index(drop=True)
        sdf = sdf.sort_values("edge_weight", ascending=False).reset_index(drop=True)


        print()
        print(gdf)

        print()
        print(sdf)

        query_output_dir = ndcg_output_dir + "Query-Tables/%s/"%hyp_infoID
        create_output_dir(query_output_dir)

        #ndcg
        # edge_weight_col = 
        for degree_type in degree_types:
            for user_type in user_types:
                for weighted in WEIGHTED_LIST:

                    timer_start = time()

                    if weighted == True:
                        weight_tag = "weighted"
                    else:
                        weight_tag = "unweighted"
                    cur_infoID_avg_ndcg, query_df = get_fancy_ndcg_v3_para_temporal(NUM_JOBS ,gdf,sdf ,degree_type, user_type, GRAN, edge_weight_col, weighted=weighted )

                    query_fp = query_output_dir + "%s-%s-%s-%s.csv"%(hyp_infoID, degree_type, user_type, weighted)
                    query_df.to_csv(query_fp, index=False)
                    print(query_df)
                    print(query_fp)

                    print("\n%s %s %s %s AVG SCORE: %.4f"%(infoID, degree_type, user_type, weighted, cur_infoID_avg_ndcg))

                    user_type_df_list.append(user_type)
                    dt_df_list.append(degree_type)
                    weight_df_list.append(weight_tag)
                    ndcg_df_list.append(cur_infoID_avg_ndcg)
                    infoID_df_list.append(infoID)

                    timer_end = time()

                    time_fp = query_output_dir + "%s-%s-%s-%s-TIME-INFO.txt"%(hyp_infoID, degree_type, user_type, weighted)

                    total_time = timer_end - timer_start 
                    time_in_min = total_time/60.0
                    with open(time_fp, "w") as f:
                        f.write("time in min: %.2f"%time_in_min)
                        print("time in min: %.2f"%time_in_min)

        cur_infoID_result_df = pd.DataFrame(data={"infoID":infoID_df_list, "degree_type":dt_df_list, "ndcg_score":ndcg_df_list, "weighted":weight_df_list, "user_type":user_type_df_list})
        col_order = ["infoID", "degree_type", "user_type", "weighted", "ndcg_score"]
        cur_infoID_result_df = cur_infoID_result_df[col_order]
        print("\ncur_infoID_result_df")
        print(cur_infoID_result_df)

        #save
        output_fp = ndcg_infoID_output_dir + hyp_infoID + ".csv"
        cur_infoID_result_df.to_csv(output_fp, index=False)
        print(output_fp)

        all_ndcg_dfs.append(cur_infoID_result_df)

    full_ndf = pd.concat(all_ndcg_dfs).reset_index(drop=True)

    weighted_ndf = full_ndf[full_ndf["weighted"]=="weighted"].reset_index(drop=True)
    unweighted_ndf = full_ndf[full_ndf["weighted"]=="unweighted"].reset_index(drop=True)

    print("\nweighted_ndf")
    print(weighted_ndf)

    print("\nunweighted_ndf")
    print(unweighted_ndf)

    output_fp = ndcg_output_dir + "All-Weighted-NDCG-Results.csv"
    weighted_ndf.to_csv(output_fp, index=False)
    print(output_fp)

    output_fp = ndcg_output_dir + "All-Unweighted-NDCG-Results.csv"
    unweighted_ndf.to_csv(output_fp, index=False)
    print(output_fp)

    avg_weighted_ndf = weighted_ndf.copy()
    avg_unweighted_ndf = unweighted_ndf.copy()

    avg_weighted_ndf["avg_ndcg_score"] = avg_weighted_ndf.groupby("infoID")["ndcg_score"].transform("mean")
    avg_weighted_ndf = avg_weighted_ndf[["infoID", "avg_ndcg_score"]].drop_duplicates().sort_values("avg_ndcg_score", ascending=False).reset_index(drop=True)
    print("\navg_weighted_ndf")
    print(avg_weighted_ndf)

    output_fp = ndcg_output_dir + "Summary-Weighted-NDCG-Results.csv"
    avg_weighted_ndf.to_csv(output_fp, index=False)
    print(output_fp)

    avg_unweighted_ndf["avg_ndcg_score"] = avg_unweighted_ndf.groupby("infoID")["ndcg_score"].transform("mean")
    avg_unweighted_ndf = avg_unweighted_ndf[["infoID", "avg_ndcg_score"]].drop_duplicates().sort_values("avg_ndcg_score", ascending=False).reset_index(drop=True)
    print("\navg_unweighted_ndf")
    print(avg_unweighted_ndf)

    output_fp = ndcg_output_dir + "Summary-Unweighted-NDCG-Results.csv"
    avg_unweighted_ndf.to_csv(output_fp, index=False)
    print(output_fp)

    return weighted_ndf, unweighted_ndf,avg_weighted_ndf, avg_unweighted_ndf, ndcg_output_dir

def get_infoID_to_actionType_proportions(full_history_record_df, platform, infoIDs):

    df = full_history_record_df[full_history_record_df["platform"]==platform].reset_index(drop=True)

    print("\nAction freqs...")
    infoID_to_actionType_prop_dict = {}
    for infoID in infoIDs:
        temp = df[df["informationID"]==infoID]
        temp["actionType_sum"] = temp.groupby(["actionType"])["actionType"].transform("count")
        temp = temp[["informationID", "actionType", "actionType_sum"]].drop_duplicates().reset_index(drop=True)
        temp["actionType_freq"] = temp["actionType_sum"]/temp["actionType_sum"].sum()
        temp = temp[["informationID", "actionType", "actionType_freq"]]
        infoID_to_actionType_prop_dict[infoID] = temp

        print()
        print(temp)

    return infoID_to_actionType_prop_dict

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

def get_user_sim_pred_records_for_cur_infoID_v2_add_more_info_v2_ua( infoID_materials_tuple,NUM_LINK_SAMPLE_ATTEMPTS,ADD_OLD_USERS_TO_NEW_USERS_IF_TOO_MANY, timestep, DEBUG_PRINT):

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

    # if cur_new_users_only_weight_table.shape[0] > 0:
    #     initial_new_links = assign_parents_to_new_user_placeholders_v2_debug_print(initial_new_links, cur_new_users_only_weight_table, DEBUG_PRINT)
    # else:
    #     initial_new_links = assign_parents_to_new_user_placeholders_v2_debug_print(initial_new_links, cur_all_users_weight_table, DEBUG_PRINT)


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

def get_ua_ndcg_v3_para(gt_input_dir, infoID_to_model_sim_dict, query_groupby_cols, edge_weight_col, 
    WEIGHTED_LIST,main_output_dir,platform, infoIDs, hyp_dict, degree_types, user_types, 
    cur_platform_user_df, GRAN,REMOVE_MISSING_USERS,child_ndcg_col="nodeUserID_ndcg", parent_ndcg_col="parentUserID_ndcg", NUM_JOBS=20):

    ndcg_keep_cols = ["nodeUserID", "parentUserID"]
    # query_groupby_cols = ["query_user"]
    # edge_weight_col = "edge_weight"
    # WEIGHTED_LIST = [True, False]

    ndcg_output_dir = main_output_dir + "/NDCG-Results/"
    create_output_dir(ndcg_output_dir)



    all_weighted_dfs = []
    all_unweighted_dfs = []

    ndcg_infoID_output_dir = ndcg_output_dir + "InfoID-Results/"
    create_output_dir(ndcg_infoID_output_dir)

    all_ndcg_dfs = []
    for infoID in infoIDs:

        #lists for df
        user_type_df_list = []
        dt_df_list = []
        weight_df_list = []
        ndcg_df_list = []
        infoID_df_list = []


        hyp_infoID = hyp_dict[infoID]
        input_fp = gt_input_dir + hyp_infoID + ".csv"
        gdf = pd.read_csv(input_fp)

        # input_fp = model_input_dir + hyp_infoID + ".csv"
        # sdf = pd.read_csv(input_fp)

        sdf = infoID_to_model_sim_dict[infoID]

        old_user_df = cur_platform_user_df[cur_platform_user_df["informationID"]==infoID]
        old_user_df =old_user_df[old_user_df["platform"]==platform].reset_index(drop=True)

        cur_old_user_set = set(list(old_user_df["nodeUserID"].unique()) + list(old_user_df["parentUserID"].unique()))

        sdf[child_ndcg_col] = [user if user in cur_old_user_set else "new_user" for user in list(sdf["nodeUserID"])]
        sdf[parent_ndcg_col] = [user if user in cur_old_user_set else "new_user" for user in list(sdf["parentUserID"])]


        # sdf[query_user_col] = [user if user in old_user_set else "new_user" for user in list(cur_baseline_df[query_col])]

        if REMOVE_MISSING_USERS == True:
            gdf = gdf[gdf[child_ndcg_col] != "missing_parentUserID"]
            gdf = gdf[gdf[parent_ndcg_col] != "missing_parentUserID"]
            sdf = sdf[sdf[child_ndcg_col] != "missing_parentUserID"]
            sdf = sdf[sdf[parent_ndcg_col] != "missing_parentUserID"]

        # if query_groupby_cols == ["query_user", "nodeTime"]:
        #     gdf["nodeTime"]=gdf["nodeTime"].astype(str)
        #     sdf["nodeTime"]=sdf["nodeTime"].astype(str)

        #     gdf[child_ndcg_col] = gdf[child_ndcg_col] + "_" gdf["nodeTime"]
        #     sdf[child_ndcg_col] = sdf[child_ndcg_col] + "_" sdf["nodeTime"]

        gdf["edge_weight"] = gdf.groupby([child_ndcg_col, parent_ndcg_col])[parent_ndcg_col].transform("count")
        sdf["edge_weight"] = sdf.groupby([child_ndcg_col, parent_ndcg_col])[parent_ndcg_col].transform("count")

        gdf = gdf[[child_ndcg_col, parent_ndcg_col, "edge_weight"]].drop_duplicates().reset_index(drop=True)
        sdf = sdf[[child_ndcg_col, parent_ndcg_col, "edge_weight"]].drop_duplicates().reset_index(drop=True)

        gdf = gdf.sort_values("edge_weight", ascending=False).reset_index(drop=True)
        sdf = sdf.sort_values("edge_weight", ascending=False).reset_index(drop=True)



        print()
        print(gdf)

        print()
        print(sdf)

        

        #temp old users
        temp_infoID_user_df = cur_platform_user_df[cur_platform_user_df["informationID"]==infoID]
        old_user_set = set(list(temp_infoID_user_df["nodeUserID"]) + ["missing_parentUserID"])

        query_output_dir = ndcg_output_dir + "Query-Tables/%s/"%hyp_infoID
        create_output_dir(query_output_dir)

        #ndcg
        # edge_weight_col = 
        for degree_type in degree_types:
            for user_type in user_types:
                for weighted in WEIGHTED_LIST:

                    timer_start = time()

                    if weighted == True:
                        weight_tag = "weighted"
                    else:
                        weight_tag = "unweighted"
                    cur_infoID_avg_ndcg, query_df = get_fancy_ndcg_v2_para(NUM_JOBS ,gdf,sdf ,degree_type, user_type, old_user_set, GRAN, edge_weight_col, query_groupby_cols,weighted=weighted )

                    query_fp = query_output_dir + "%s-%s-%s-%s.csv"%(hyp_infoID, degree_type, user_type, weighted)
                    query_df.to_csv(query_fp, index=False)
                    print(query_df)
                    print(query_fp)

                    print("\n%s %s %s %s AVG SCORE: %.4f"%(infoID, degree_type, user_type, weighted, cur_infoID_avg_ndcg))

                    user_type_df_list.append(user_type)
                    dt_df_list.append(degree_type)
                    weight_df_list.append(weight_tag)
                    ndcg_df_list.append(cur_infoID_avg_ndcg)
                    infoID_df_list.append(infoID)

                    timer_end = time()

                    time_fp = query_output_dir + "%s-%s-%s-%s-TIME-INFO.txt"%(hyp_infoID, degree_type, user_type, weighted)

                    total_time = timer_end - timer_start 
                    time_in_min = total_time/60.0
                    with open(time_fp, "w") as f:
                        f.write("time in min: %.2f"%time_in_min)
                        print("time in min: %.2f"%time_in_min)

        cur_infoID_result_df = pd.DataFrame(data={"infoID":infoID_df_list, "degree_type":dt_df_list, "ndcg_score":ndcg_df_list, "weighted":weight_df_list, "user_type":user_type_df_list})
        col_order = ["infoID", "degree_type", "user_type", "weighted", "ndcg_score"]
        cur_infoID_result_df = cur_infoID_result_df[col_order]
        print("\ncur_infoID_result_df")
        print(cur_infoID_result_df)

        #save
        output_fp = ndcg_infoID_output_dir + hyp_infoID + ".csv"
        cur_infoID_result_df.to_csv(output_fp, index=False)
        print(output_fp)

        all_ndcg_dfs.append(cur_infoID_result_df)

    full_ndf = pd.concat(all_ndcg_dfs).reset_index(drop=True)

    weighted_ndf = full_ndf[full_ndf["weighted"]=="weighted"].reset_index(drop=True)
    unweighted_ndf = full_ndf[full_ndf["weighted"]=="unweighted"].reset_index(drop=True)

    print("\nweighted_ndf")
    print(weighted_ndf)

    print("\nunweighted_ndf")
    print(unweighted_ndf)

    output_fp = ndcg_output_dir + "All-Weighted-NDCG-Results.csv"
    weighted_ndf.to_csv(output_fp, index=False)
    print(output_fp)

    output_fp = ndcg_output_dir + "All-Unweighted-NDCG-Results.csv"
    unweighted_ndf.to_csv(output_fp, index=False)
    print(output_fp)

    avg_weighted_ndf = weighted_ndf.copy()
    avg_unweighted_ndf = unweighted_ndf.copy()

    avg_weighted_ndf["avg_ndcg_score"] = avg_weighted_ndf.groupby("infoID")["ndcg_score"].transform("mean")
    avg_weighted_ndf = avg_weighted_ndf[["infoID", "avg_ndcg_score"]].drop_duplicates().sort_values("avg_ndcg_score", ascending=False).reset_index(drop=True)
    print("\navg_weighted_ndf")
    print(avg_weighted_ndf)

    output_fp = ndcg_output_dir + "Summary-Weighted-NDCG-Results.csv"
    avg_weighted_ndf.to_csv(output_fp, index=False)
    print(output_fp)

    avg_unweighted_ndf["avg_ndcg_score"] = avg_unweighted_ndf.groupby("infoID")["ndcg_score"].transform("mean")
    avg_unweighted_ndf = avg_unweighted_ndf[["infoID", "avg_ndcg_score"]].drop_duplicates().sort_values("avg_ndcg_score", ascending=False).reset_index(drop=True)
    print("\navg_unweighted_ndf")
    print(avg_unweighted_ndf)

    output_fp = ndcg_output_dir + "Summary-Unweighted-NDCG-Results.csv"
    avg_unweighted_ndf.to_csv(output_fp, index=False)
    print(output_fp)

    return weighted_ndf, unweighted_ndf,avg_weighted_ndf, avg_unweighted_ndf, ndcg_output_dir

def get_ua_ndcg_v2_new_user_fix(gt_input_dir, infoID_to_model_sim_dict, query_groupby_cols, edge_weight_col, 
    WEIGHTED_LIST,main_output_dir,platform, infoIDs, hyp_dict, degree_types, user_types, 
    cur_platform_user_df, GRAN,REMOVE_MISSING_USERS,child_ndcg_col="nodeUserID_ndcg", parent_ndcg_col="parentUserID_ndcg"):

    ndcg_keep_cols = ["nodeUserID", "parentUserID"]
    # query_groupby_cols = ["query_user"]
    # edge_weight_col = "edge_weight"
    # WEIGHTED_LIST = [True, False]

    ndcg_output_dir = main_output_dir + "/NDCG-Results/"
    create_output_dir(ndcg_output_dir)



    all_weighted_dfs = []
    all_unweighted_dfs = []

    ndcg_infoID_output_dir = ndcg_output_dir + "InfoID-Results/"
    create_output_dir(ndcg_infoID_output_dir)

    all_ndcg_dfs = []
    for infoID in infoIDs:

        #lists for df
        user_type_df_list = []
        dt_df_list = []
        weight_df_list = []
        ndcg_df_list = []
        infoID_df_list = []


        hyp_infoID = hyp_dict[infoID]
        input_fp = gt_input_dir + hyp_infoID + ".csv"
        gdf = pd.read_csv(input_fp)

        # input_fp = model_input_dir + hyp_infoID + ".csv"
        # sdf = pd.read_csv(input_fp)

        sdf = infoID_to_model_sim_dict[infoID]

        if REMOVE_MISSING_USERS == True:
            gdf = gdf[gdf[child_ndcg_col] != "missing_parentUserID"]
            gdf = gdf[gdf[parent_ndcg_col] != "missing_parentUserID"]
            sdf = sdf[sdf[child_ndcg_col] != "missing_parentUserID"]
            sdf = sdf[sdf[parent_ndcg_col] != "missing_parentUserID"]

        gdf["edge_weight"] = gdf.groupby([child_ndcg_col, parent_ndcg_col])[parent_ndcg_col].transform("count")
        sdf["edge_weight"] = sdf.groupby([child_ndcg_col, parent_ndcg_col])[parent_ndcg_col].transform("count")

        gdf = gdf[[child_ndcg_col, parent_ndcg_col, "edge_weight"]].drop_duplicates().reset_index(drop=True)
        sdf = sdf[[child_ndcg_col, parent_ndcg_col, "edge_weight"]].drop_duplicates().reset_index(drop=True)

        gdf = gdf.sort_values("edge_weight", ascending=False).reset_index(drop=True)
        sdf = sdf.sort_values("edge_weight", ascending=False).reset_index(drop=True)



        print()
        print(gdf)

        print()
        print(sdf)

        

        #temp old users
        temp_infoID_user_df = cur_platform_user_df[cur_platform_user_df["informationID"]==infoID]
        old_user_set = set(list(temp_infoID_user_df["nodeUserID"]) + ["missing_parentUserID"])

        query_output_dir = ndcg_output_dir + "Query-Tables/%s/"%hyp_infoID
        create_output_dir(query_output_dir)

        #ndcg
        # edge_weight_col = 
        for degree_type in degree_types:
            for user_type in user_types:
                for weighted in WEIGHTED_LIST:
                    if weighted == True:
                        weight_tag = "weighted"
                    else:
                        weight_tag = "unweighted"
                    cur_infoID_avg_ndcg, query_df = get_fancy_ndcg(gdf,sdf ,degree_type, user_type, old_user_set, GRAN, edge_weight_col, query_groupby_cols,weighted=weighted )

                    query_fp = query_output_dir + "%s-%s-%s-%s.csv"%(hyp_infoID, degree_type, user_type, weighted)
                    query_df.to_csv(query_fp, index=False)
                    print(query_df)
                    print(query_fp)

                    print("\n%s %s %s %s AVG SCORE: %.4f"%(infoID, degree_type, user_type, weighted, cur_infoID_avg_ndcg))

                    user_type_df_list.append(user_type)
                    dt_df_list.append(degree_type)
                    weight_df_list.append(weight_tag)
                    ndcg_df_list.append(cur_infoID_avg_ndcg)
                    infoID_df_list.append(infoID)

        cur_infoID_result_df = pd.DataFrame(data={"infoID":infoID_df_list, "degree_type":dt_df_list, "ndcg_score":ndcg_df_list, "weighted":weight_df_list, "user_type":user_type_df_list})
        col_order = ["infoID", "degree_type", "user_type", "weighted", "ndcg_score"]
        cur_infoID_result_df = cur_infoID_result_df[col_order]
        print("\ncur_infoID_result_df")
        print(cur_infoID_result_df)

        #save
        output_fp = ndcg_infoID_output_dir + hyp_infoID + ".csv"
        cur_infoID_result_df.to_csv(output_fp, index=False)
        print(output_fp)

        all_ndcg_dfs.append(cur_infoID_result_df)

    full_ndf = pd.concat(all_ndcg_dfs).reset_index(drop=True)

    weighted_ndf = full_ndf[full_ndf["weighted"]=="weighted"].reset_index(drop=True)
    unweighted_ndf = full_ndf[full_ndf["weighted"]=="unweighted"].reset_index(drop=True)

    print("\nweighted_ndf")
    print(weighted_ndf)

    print("\nunweighted_ndf")
    print(unweighted_ndf)

    output_fp = ndcg_output_dir + "All-Weighted-NDCG-Results.csv"
    weighted_ndf.to_csv(output_fp, index=False)
    print(output_fp)

    output_fp = ndcg_output_dir + "All-Unweighted-NDCG-Results.csv"
    unweighted_ndf.to_csv(output_fp, index=False)
    print(output_fp)

    avg_weighted_ndf = weighted_ndf.copy()
    avg_unweighted_ndf = unweighted_ndf.copy()

    avg_weighted_ndf["avg_ndcg_score"] = avg_weighted_ndf.groupby("infoID")["ndcg_score"].transform("mean")
    avg_weighted_ndf = avg_weighted_ndf[["infoID", "avg_ndcg_score"]].drop_duplicates().sort_values("avg_ndcg_score", ascending=False).reset_index(drop=True)
    print("\navg_weighted_ndf")
    print(avg_weighted_ndf)

    output_fp = ndcg_output_dir + "Summary-Weighted-NDCG-Results.csv"
    avg_weighted_ndf.to_csv(output_fp, index=False)
    print(output_fp)

    avg_unweighted_ndf["avg_ndcg_score"] = avg_unweighted_ndf.groupby("infoID")["ndcg_score"].transform("mean")
    avg_unweighted_ndf = avg_unweighted_ndf[["infoID", "avg_ndcg_score"]].drop_duplicates().sort_values("avg_ndcg_score", ascending=False).reset_index(drop=True)
    print("\navg_unweighted_ndf")
    print(avg_unweighted_ndf)

    output_fp = ndcg_output_dir + "Summary-Unweighted-NDCG-Results.csv"
    avg_unweighted_ndf.to_csv(output_fp, index=False)
    print(output_fp)

    return weighted_ndf, unweighted_ndf,avg_weighted_ndf, avg_unweighted_ndf, ndcg_output_dir

def make_ua_time_plot_and_df(TIME_PER_ITERATION_LIST, COUNT_LIST, output_dir,count_col):
    
    print("\nGetting time info...")
    time_df = pd.DataFrame(data={"time_in_min":TIME_PER_ITERATION_LIST, count_col:COUNT_LIST})
    

    #plot time df
    time_df["timestep"] = [i+1 for i in range(time_df.shape[0])]
    time_df = time_df.sort_values(count_col, ascending=True)

    print("\ntime_df")
    print(time_df)


    time_fp = output_dir + "Time-vs-%s-data.csv"%count_col
    time_df.to_csv(time_fp, index=False)
    print(time_fp)

    #plot it
    time_df[count_col] = minmax_scale_series(time_df[count_col])
    time_df["time_in_min"] = minmax_scale_series(time_df["time_in_min"])
    time_df.plot(x = count_col, y="time_in_min", kind="scatter")
    fig = plt.gcf()
    ax = plt.gca()
    ax.set_title("%s vs. time_in_min"%count_col)
    output_fp = output_dir + "time-vs-%s-plot.png"%count_col
    fig.savefig(output_fp)
    print(output_fp)
    output_fp = output_dir + "time-vs-%s-plot.svg"%count_col
    fig.savefig(output_fp)
    print(output_fp)
    plt.close()

    return time_df

def make_ua_time_plot_and_df_v2_with_lr(TIME_PER_ITERATION_LIST, COUNT_LIST, output_dir,count_col, NORMALIZE=False):

    create_output_dir(output_dir)
    
    print("\nGetting time info...")
    time_df = pd.DataFrame(data={"time_in_min":TIME_PER_ITERATION_LIST, count_col:COUNT_LIST})
    

    #plot time df
    time_df["timestep"] = [i+1 for i in range(time_df.shape[0])]
    time_df = time_df.sort_values(count_col, ascending=True)

    print("\ntime_df")
    print(time_df)


    time_fp = output_dir + "Time-vs-%s-data.csv"%count_col
    time_df.to_csv(time_fp, index=False)
    print(time_fp)

    norm_x = minmax_scale_series(time_df[count_col])
    norm_y = minmax_scale_series(time_df["time_in_min"])

    if NORMALIZE == True:
        #plot it
        time_df[count_col] = norm_x
        time_df["time_in_min"] = norm_y


    x = time_df[count_col].values
    y = time_df["time_in_min"].values
    plt.scatter(x, y)

    print("\nx and y")
    print(x)
    print(y)

    try:
        r2 = r2_score(x, y)
        r2 = np.round(r2, 4)
        print("%s plot r2 score: %.4f"%(count_col, r2))
        r2_fp = output_dir + "%s-R2-Result.txt"%count_col
        with open(r2_fp, "w") as f:
            f.write(str(r2))
    except ValueError:
        print("COULD NOT GET R2")

    x = x.reshape((x.shape[0], 1))
    y = y.reshape((y.shape[0], 1))

    try:
        plt.plot( x, LinearRegression().fit(x, y).predict(x) )
    except ValueError:
        print("\nCOULD NOT PLOT LR")
        return time_df
    # plt.plot( norm_x, LinearRegression().fit(norm_x, norm_y).predict(norm_x) )

    # try:
    #     plt.plot( x, LinearRegression().fit(x, y).predict(x) )
    # except ValueError:
    #     print("\nCannot do LR...")


    fig = plt.gcf()
    ax = plt.gca()
    # plt.axis('scaled')

    ax.set_title("%s vs. time_in_min"%(count_col))

    plt.tight_layout()
    output_fp = output_dir + "time-vs-%s-plot.png"%count_col
    fig.savefig(output_fp)
    print(output_fp)
    output_fp = output_dir + "time-vs-%s-plot.svg"%count_col
    fig.savefig(output_fp)
    print(output_fp)
    plt.close()

    return time_df

def get_infoID_to_new_user_edge_weight_dist_table_dict_v2_ua(edge_weight_df):
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

def get_all_users_edge_weight_dist_table_v2_ua(infoIDs,edge_weight_df):
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

def create_infoID_to_all_users_weight_table_dict_v2_missing_infoID_v2_ua(history_record_df,INFLUENCE_EPSILON_DIV_VAL,infoIDs):

    print("\nGetting overall action and influence weights...")
    groupby_add_cols = ["nodeUserID", "parentUserID", "informationID"]

    keep_cols = ["nodeUserID", "parentUserID", "informationID","nodeUserID_num_actions_this_timestep","parentUserID_influence_this_timestep"]
    user_weight_table = history_record_df[keep_cols]

    user_weight_table["parentUserID_overall_influence_weight"] = user_weight_table.groupby(["parentUserID", "informationID"])["parentUserID_influence_this_timestep"].transform("sum")
    user_weight_table = get_user_influence_from_parent_influence_V2_no_fillna_v2_ua(user_weight_table)

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

def create_infoID_to_new_users_only_weight_table_dict_v2_ua(history_record_df,INFLUENCE_EPSILON_DIV_VAL):

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
    new_parent_only_history_record_df = get_user_influence_from_parent_influence_V2_no_fillna_v2_ua(new_parent_only_history_record_df)
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

def get_user_influence_from_parent_influence_V2_no_fillna_v2_ua(model_weight_df):
    model_weight_df["infoID_user"] = model_weight_df["nodeUserID"] + "_" + model_weight_df["informationID"]
    model_weight_df["infoID_parent"] = model_weight_df["parentUserID"] + "_" + model_weight_df["informationID"]


    parent_to_influence_dict = convert_df_2_cols_to_dict(model_weight_df, "infoID_parent", "parentUserID_overall_influence_weight")
    model_weight_df["nodeUserID_overall_influence_weight"] = model_weight_df["infoID_user"].map(parent_to_influence_dict)
    # model_weight_df["nodeUserID_overall_influence_weight"] = model_weight_df["nodeUserID_overall_influence_weight"].fillna(INFLUENCE_EPSILON)

    print(model_weight_df["nodeUserID_overall_influence_weight"].value_counts())

    model_weight_df = model_weight_df.drop(["infoID_parent", "infoID_user"], axis=1)

    return model_weight_df

def get_various_ua_tables(history_record_df, infoIDs,INFLUENCE_EPSILON_DIV_VAL):
    #get edge weight dist tables
    edge_weight_df = history_record_df[["nodeUserID", "parentUserID", "informationID", "edge_weight_this_timestep", "nodeUserID_is_new", "parentUserID_is_new"]]
    infoID_to_new_user_edge_weight_dist_table_dict = get_infoID_to_new_user_edge_weight_dist_table_dict_v2_ua(edge_weight_df)
    infoID_to_all_users_edge_weight_dist_table_dict = get_all_users_edge_weight_dist_table_v2_ua(infoIDs,edge_weight_df)

    #weight tables
    infoID_to_all_users_weight_table_dict = create_infoID_to_all_users_weight_table_dict_v2_missing_infoID_v2_ua(history_record_df,INFLUENCE_EPSILON_DIV_VAL,infoIDs)
    infoID_to_new_users_only_weight_table_dict = create_infoID_to_new_users_only_weight_table_dict_v2_ua(history_record_df,INFLUENCE_EPSILON_DIV_VAL)

    #print current weights
    for infoID in infoIDs:
        temp = infoID_to_new_users_only_weight_table_dict[infoID]
        temp = temp[temp["nodeUserID"].str.contains("synthetic")].reset_index(drop=True)
        print(temp)
        print(temp["nodeUserID_overall_influence_weight"].value_counts())

    return  edge_weight_df, infoID_to_new_user_edge_weight_dist_table_dict,infoID_to_all_users_edge_weight_dist_table_dict,infoID_to_all_users_weight_table_dict,infoID_to_new_users_only_weight_table_dict

def get_cur_pred_links_as_df(infoID_to_df_list_dict, infoIDs, timestep):
    all_cur_pred_links = []
    for infoID in infoIDs:
        cur_pred_links = infoID_to_df_list_dict[infoID][-1]

        cur_pred_links["nodeUserID"] = [ user.replace("new_synthetic_user", "synthetic_user_ts_%d"%timestep) if "new_synthetic_user" in user else user for user in list(cur_pred_links["nodeUserID"])]
        cur_pred_links["parentUserID"] = [ user.replace("new_synthetic_user", "synthetic_user_ts_%d"%timestep) if "new_synthetic_user" in user else user for user in list(cur_pred_links["parentUserID"])]
        # cur_pred_links = cur_pred_links.rename(columns={"edge_weight":"edge_weight_this_timestep"})

        cur_pred_links["nodeUserID_num_actions_this_timestep"] = cur_pred_links.groupby(["nodeUserID", "informationID"])["nodeUserID"].transform("count")

        cur_pred_links["edge_weight_this_timestep"] = cur_pred_links.groupby(["nodeUserID","parentUserID" ,"informationID"])["nodeUserID"].transform("count")

        if "edge_weight" in list(cur_pred_links):
            cur_pred_links = cur_pred_links.drop("edge_weight", axis=1)
        cur_pred_links = cur_pred_links.drop_duplicates().reset_index(drop=True)
        print("\ncur_pred_links")
        print(cur_pred_links)

        all_cur_pred_links.append(cur_pred_links)

    cur_pred_links = pd.concat(all_cur_pred_links)
    print("\ncur_pred_links")
    print(cur_pred_links)

    return cur_pred_links

def get_cur_history_record_df(infoIDs, full_history_record_df, infoID_to_lookback_over_time_df_dict, timestep):
    infoID_history_dfs = []
    for infoID in infoIDs:
        #get temp full history
        temp_infoID_history = full_history_record_df[full_history_record_df["informationID"]==infoID].reset_index(drop=True)
        print("\ntemp_infoID_history")
        print(temp_infoID_history)

        #get relevant history info
        infoID_lb_df = infoID_to_lookback_over_time_df_dict[infoID]

        print("\ninfoID_lb_df")
        print(infoID_lb_df)

        print("\ndesired timestep")
        print(timestep)


        cur_ts_lb_df = infoID_lb_df[infoID_lb_df["timestep"]==timestep]
        print("\ncur_ts_lb_df")
        print(cur_ts_lb_df)

        cur_history_start = cur_ts_lb_df["history_start"].iloc[-1]
        cur_history_end = cur_ts_lb_df["history_end"].iloc[-1]

        temp_infoID_history = config_df_by_dates(temp_infoID_history,cur_history_start, cur_history_end, "nodeTime")
        print("\ntemp_infoID_history")
        print(temp_infoID_history)

        infoID_history_dfs.append(temp_infoID_history)

    history_record_df = pd.concat(infoID_history_dfs).reset_index(drop=True)
    print("\nhistory_record_df")
    print(history_record_df)

    return history_record_df

def get_lookback_dates_per_infoID_over_time(main_lookback_over_time_df, test_start, test_end, num_test_timesteps, MIN_LB_SCALING_FACTOR, BASIC_GRAN, infoIDs):

    infoID_to_lookback_over_time_df_dict = {}
    for infoID in infoIDs:

        lookback_over_time_df = main_lookback_over_time_df[["timestep", infoID]]
        lookback_over_time_df["infoID"] = infoID
        lookback_over_time_df= lookback_over_time_df.rename(columns={infoID: "cur_lookback"})

        print("\nPrescale")
        print(lookback_over_time_df)

        lookback_over_time_df["cur_lookback"] = lookback_over_time_df["cur_lookback"] * MIN_LB_SCALING_FACTOR

        print("\nPost-scale")
        print(lookback_over_time_df)

        lookback_over_time_df = lookback_over_time_df.sort_values("timestep").reset_index(drop=True)
        dates = pd.date_range(test_start, test_end, freq=BASIC_GRAN)
        lookback_over_time_df["nodeTime"] = dates
        lookback_over_time_df["nodeTime"] = pd.to_datetime(lookback_over_time_df["nodeTime"], utc=True)
        lookback_over_time_df = lookback_over_time_df.sort_values("nodeTime").reset_index(drop=True)


        lookback_over_time_df["history_start"] = lookback_over_time_df["nodeTime"] - pd.to_timedelta(lookback_over_time_df["cur_lookback"], unit=BASIC_GRAN)
        lookback_over_time_df["history_end"] =lookback_over_time_df["nodeTime"] - pd.to_timedelta(1, unit="ms")
        

        print("\nlookback_over_time_df with nodeTime")
        print(lookback_over_time_df)

        infoID_to_lookback_over_time_df_dict[infoID] = lookback_over_time_df

    return infoID_to_lookback_over_time_df_dict

def get_lookback_dates_per_infoID_over_time_v2_reg_lookback(main_lookback_over_time_df, test_start, test_end, num_test_timesteps, LOOKBACK_FACTOR, BASIC_GRAN, infoIDs):

    infoID_to_lookback_over_time_df_dict = {}
    for infoID in infoIDs:

        lookback_over_time_df = main_lookback_over_time_df[["timestep", infoID]]
        lookback_over_time_df["infoID"] = infoID
        lookback_over_time_df= lookback_over_time_df.rename(columns={infoID: "min_lookback"})

        print("\nPre-adjustment")
        print(lookback_over_time_df)

        min_lookbacks = list(lookback_over_time_df["min_lookback"])

        adjusted_lookbacks = []
        for m in min_lookbacks:

            if LOOKBACK_FACTOR > m:
                adjusted_lookbacks.append(LOOKBACK_FACTOR)
            else:
                adjusted_lookbacks.append(m)

        lookback_over_time_df["cur_lookback"] = adjusted_lookbacks

        print("\nPost-adjustment")
        print(lookback_over_time_df)

        lookback_over_time_df = lookback_over_time_df.sort_values("timestep").reset_index(drop=True)
        dates = pd.date_range(test_start, test_end, freq=BASIC_GRAN)
        lookback_over_time_df["nodeTime"] = dates
        lookback_over_time_df["nodeTime"] = pd.to_datetime(lookback_over_time_df["nodeTime"], utc=True)
        lookback_over_time_df = lookback_over_time_df.sort_values("nodeTime").reset_index(drop=True)


        lookback_over_time_df["history_start"] = lookback_over_time_df["nodeTime"] - pd.to_timedelta(lookback_over_time_df["cur_lookback"], unit=BASIC_GRAN)
        lookback_over_time_df["history_end"] =lookback_over_time_df["nodeTime"] - pd.to_timedelta(1, unit="ms")
        

        print("\nlookback_over_time_df with nodeTime")
        print(lookback_over_time_df)

        infoID_to_lookback_over_time_df_dict[infoID] = lookback_over_time_df

    return infoID_to_lookback_over_time_df_dict

def get_lookback_dates_per_infoID_over_time_v3_FRAME_HACK(main_lookback_over_time_df, test_start, test_end, num_test_timesteps, LOOKBACK_FACTOR, BASIC_GRAN, infoIDs, HACK_LB =200):

    infoID_to_lookback_over_time_df_dict = {}
    for infoID in infoIDs:

        lookback_over_time_df = main_lookback_over_time_df[["timestep", infoID]]
        lookback_over_time_df["infoID"] = infoID
        lookback_over_time_df= lookback_over_time_df.rename(columns={infoID: "min_lookback"})

        print("\nPre-adjustment")
        print(lookback_over_time_df)

        min_lookbacks = list(lookback_over_time_df["min_lookback"])

        adjusted_lookbacks = []
        for m in min_lookbacks:

            if infoID in  ["controversies/pakistan/students", "controversies/pakistan/bajwa"]:
                adjusted_lookbacks.append(HACK_LB)
            elif LOOKBACK_FACTOR > m:
                adjusted_lookbacks.append(LOOKBACK_FACTOR)
            else:
                adjusted_lookbacks.append(m)

        lookback_over_time_df["cur_lookback"] = adjusted_lookbacks

        print("\nPost-adjustment")
        print(lookback_over_time_df)

        lookback_over_time_df = lookback_over_time_df.sort_values("timestep").reset_index(drop=True)
        dates = pd.date_range(test_start, test_end, freq=BASIC_GRAN)
        lookback_over_time_df["nodeTime"] = dates
        lookback_over_time_df["nodeTime"] = pd.to_datetime(lookback_over_time_df["nodeTime"], utc=True)
        lookback_over_time_df = lookback_over_time_df.sort_values("nodeTime").reset_index(drop=True)


        lookback_over_time_df["history_start"] = lookback_over_time_df["nodeTime"] - pd.to_timedelta(lookback_over_time_df["cur_lookback"], unit=BASIC_GRAN)
        lookback_over_time_df["history_end"] =lookback_over_time_df["nodeTime"] - pd.to_timedelta(1, unit="ms")
        

        print("\nlookback_over_time_df with nodeTime")
        print(lookback_over_time_df)

        infoID_to_lookback_over_time_df_dict[infoID] = lookback_over_time_df

    return infoID_to_lookback_over_time_df_dict

def generate_user_assign_tag_from_param_dict_v2(param_dict):

    platform = param_dict["platform"]


    try:
        model_tag = param_dict["%s_model_tag"%platform]
    except KeyError:
        model_tag = param_dict["model_tag"]
    GRAN = param_dict["GRAN"]
    
    try:
        MIN_LB_SCALING_FACTOR = param_dict["MIN_LB_SCALING_FACTOR"]
        model_tag = model_tag + "-" + str(MIN_LB_SCALING_FACTOR) + "MLBSF"
    except KeyError:
        LOOKBACK_FACTOR = param_dict["LOOKBACK_FACTOR"]
        model_tag = model_tag + "-" + str(LOOKBACK_FACTOR) + "LB"

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

def get_comp_report_df_from_tracker_dict(comp_df_tracker_dict,comp_tag_list,win_col="VAM_is_winner"):

    model_win_list = []
    trial_count_list = []

    for comp_tag in comp_tag_list:

        cur_comp_df = comp_df_tracker_dict[comp_tag]
        num_wins = cur_comp_df[cur_comp_df[win_col] == 1].shape[0]
        num_trials = cur_comp_df.shape[0]
        model_win_list.append(num_wins)
        trial_count_list.append(num_trials)

    comp_report_df = pd.DataFrame(data={"comparison_tag":comp_tag_list, "num_wins":model_win_list, "num_trials":trial_count_list})
    comp_report_df["win_freq"] = comp_report_df["num_wins"]/comp_report_df["num_trials"]

    comp_report_df = comp_report_df[["comparison_tag", "num_wins", "num_trials", "win_freq"]]

    print("\ncomp_report_df")
    print(comp_report_df)

    return comp_report_df

def get_and_save_ndcg_sim_results(platform, gt_input_dir, infoID_to_model_sim_dict, query_groupby_cols, edge_weight_col, 
                                                                WEIGHTED_LIST,output_dir,infoIDs, hyp_dict, degree_types, user_types, 
                                                                cur_platform_user_df, GRAN,REMOVE_MISSING_USERS,gt_and_baseline_input_dir,child_ndcg_col="nodeUserID_ndcg", parent_ndcg_col="parentUserID_ndcg", NUM_JOBS=20):
    #filter df
    

    # model_weighted_ndf, model_unweighted_ndf, summ_weighted_ndf, summ_unweighted_ndf, ndcg_output_dir = get_ua_ndcg(gt_input_dir, infoID_to_model_sim_dict, query_groupby_cols, edge_weight_col, 
    #                                                             WEIGHTED_LIST,output_dir,platform, infoIDs, hyp_dict, degree_types, user_types, 
    #                                                             cur_platform_user_df, GRAN,REMOVE_MISSING_USERS)

    model_weighted_ndf, model_unweighted_ndf, summ_weighted_ndf, summ_unweighted_ndf, ndcg_output_dir = get_ua_ndcg_v3_para(gt_input_dir, infoID_to_model_sim_dict, query_groupby_cols, edge_weight_col, 
    WEIGHTED_LIST,output_dir,platform, infoIDs, hyp_dict, degree_types, user_types, 
    cur_platform_user_df, GRAN,REMOVE_MISSING_USERS,child_ndcg_col=child_ndcg_col, parent_ndcg_col=parent_ndcg_col, NUM_JOBS=NUM_JOBS)

    #ndcg results
    baseline_ndcg_input_dir = gt_and_baseline_input_dir + platform + "/NDCG-Results/"
    weighted_baseline_ndcg_fp = baseline_ndcg_input_dir + "All-Weighted-NDCG-Results.csv"
    unweighted_baseline_ndcg_fp = baseline_ndcg_input_dir + "All-Unweighted-NDCG-Results.csv"
    summ_weighted_baseline_ndcg_fp = baseline_ndcg_input_dir + "Summary-Weighted-NDCG-Results.csv"
    summ_unweighted_baseline_ndcg_fp = baseline_ndcg_input_dir + "Summary-Unweighted-NDCG-Results.csv"

    weighted_baseline_ndf = pd.read_csv(weighted_baseline_ndcg_fp)
    unweighted_baseline_ndf = pd.read_csv(unweighted_baseline_ndcg_fp)
    summ_weighted_baseline_ndf = pd.read_csv(summ_weighted_baseline_ndcg_fp)
    summ_unweighted_baseline_ndf = pd.read_csv(summ_unweighted_baseline_ndcg_fp)

    #load ndcg results
    ndcg_comp_dir =output_dir+ "NDCG-Comp-Results/"
    create_output_dir(ndcg_comp_dir)
    model_weighted_comp_ndf = compare_model_and_baseline(model_weighted_ndf, weighted_baseline_ndf,"ndcg_score","Weighted-NDCG-Comps",False , category_cols=["infoID", "degree_type", "user_type", "weighted"],  model_name="VAM", output_dir=ndcg_comp_dir) 
    model_unweighted_comp_ndf = compare_model_and_baseline(model_unweighted_ndf, unweighted_baseline_ndf,"ndcg_score","Unweighted-NDCG-Comps",False , category_cols=["infoID", "degree_type", "user_type", "weighted"],  model_name="VAM", output_dir=ndcg_comp_dir) 
    summ_model_weighted_comp_ndf = compare_model_and_baseline(summ_weighted_ndf, summ_weighted_baseline_ndf,"avg_ndcg_score","Summary-Weighted-NDCG-Comps",False , category_cols=["infoID"],  model_name="VAM", output_dir=ndcg_comp_dir) 
    summ_model_unweighted_comp_ndf = compare_model_and_baseline(summ_unweighted_ndf, summ_unweighted_baseline_ndf,"avg_ndcg_score","Summary-Unweighted-NDCG-Comps",False , category_cols=["infoID"],  model_name="VAM", output_dir=ndcg_comp_dir) 

    return model_weighted_comp_ndf,model_unweighted_comp_ndf,summ_model_weighted_comp_ndf,summ_model_unweighted_comp_ndf

def update_history_record_df_with_preds(history_record_df, infoIDs, infoID_to_df_list_dict, timestep, cur_weight_start, cur_weight_end,NEW_USER_COUNTER_DICT):
    print("\nhistory_record_df")
    print(history_record_df)
    for infoID in infoIDs:

        cur_pred_links = infoID_to_df_list_dict[infoID][-1]
        print("\ncur_pred_links")
        print(cur_pred_links)

        prev_timestep = timestep - 1
        cur_pred_links["nodeUserID"] = [ user.replace("new_synthetic_user", "synthetic_user_ts_%d"%prev_timestep) if "new_synthetic_user" in user else user for user in list(cur_pred_links["nodeUserID"])]
        cur_pred_links["parentUserID"] = [ user.replace("new_synthetic_user", "synthetic_user_ts_%d"%prev_timestep) if "new_synthetic_user" in user else user for user in list(cur_pred_links["parentUserID"])]
        # cur_pred_links = cur_pred_links.rename(columns={"edge_weight":"edge_weight_this_timestep"})

        cur_pred_links["nodeUserID_num_actions_this_timestep"] = cur_pred_links.groupby(["nodeUserID", "informationID"])["nodeUserID"].transform("count")

        cur_pred_links["edge_weight_this_timestep"] = cur_pred_links.groupby(["nodeUserID","parentUserID" ,"informationID"])["nodeUserID"].transform("count")
        cur_pred_links = cur_pred_links.drop_duplicates().reset_index(drop=True)
        print("\ncur_pred_links")
        print(cur_pred_links)


        history_record_df = pd.concat([history_record_df, cur_pred_links]).reset_index(drop=True)

    print("\nupdated history_record_df")
    print(history_record_df)

    history_record_df = config_df_by_dates(history_record_df, cur_weight_start, cur_weight_end, "nodeTime")

    print("\nupdated history_record_df after date config")
    print(history_record_df)

    print("\nNew user counts so far...")
    for infoID in infoIDs:
        print("%s : %d"%(infoID, NEW_USER_COUNTER_DICT[infoID]))

    return history_record_df

def get_infoID_material_list(infoIDs, NEW_USER_COUNTER_DICT,hyp_dict,cleaned_pred_df_dict,timestep,infoID_to_all_users_weight_table_dict,
    infoID_to_new_users_only_weight_table_dict,infoID_to_all_users_edge_weight_dist_table_dict,infoID_to_new_user_edge_weight_dist_table_dict):
    infoID_to_materials_dict = {}
    infoID_materials_list = []
    num_infoIDs = len(infoIDs)

    for infoID_idx, infoID in enumerate(infoIDs):

        CUR_INFO_ID_NEW_USER_COUNTER = NEW_USER_COUNTER_DICT[infoID]

        #get cur pred df
        hyp_infoID = hyp_dict[infoID]
        print("\nGetting new user links for %s"%infoID)
        cur_pred_df = cleaned_pred_df_dict[infoID]
        cur_pred_df = cur_pred_df[cur_pred_df["timestep"]==timestep].copy().reset_index(drop=True)
        print("\ncur_pred_df")
        print(cur_pred_df)

        #============================ get weight tables ============================
        cur_all_users_weight_table  = infoID_to_all_users_weight_table_dict[infoID]
        print("\ncur_all_users_weight_table")
        print(cur_all_users_weight_table)

        cur_new_users_only_weight_table = infoID_to_new_users_only_weight_table_dict[infoID]
        print("\ncur_new_users_only_weight_table")
        print(cur_new_users_only_weight_table)

        cur_all_users_edge_weight_dist_table = infoID_to_all_users_edge_weight_dist_table_dict[infoID]
        cur_new_user_edge_weight_dist_table = infoID_to_new_user_edge_weight_dist_table_dict[infoID]

        print("\ncur_all_users_edge_weight_dist_table")
        print(cur_all_users_edge_weight_dist_table)

        print("\ncur_new_user_edge_weight_dist_table")
        print(cur_new_user_edge_weight_dist_table)

        infoID_materials_tuple = (infoID ,hyp_infoID, cur_pred_df, cur_all_users_weight_table, cur_new_users_only_weight_table,
            cur_all_users_edge_weight_dist_table,cur_new_user_edge_weight_dist_table,num_infoIDs,infoID_idx,CUR_INFO_ID_NEW_USER_COUNTER)

        infoID_materials_list.append(infoID_materials_tuple)

    return infoID_materials_list

def get_cleaned_pred_df_dict(model_test_infoID_to_pred_df_dict, target_cats, infoIDs,output_dir, platform, num_test_timesteps
    ,new_user_count_target_col,old_user_count_target_col,num_actions_target_col,user_action_conflict_resolution_option, test_start, test_end, GRAN,hyp_dict):

    #get preds without gt
    pred_without_gt = reformat_preds_for_ua_model(model_test_infoID_to_pred_df_dict, infoIDs, target_cats, platform, num_test_timesteps)
    print("\npred_without_gt")
    print(pred_without_gt)

    #clean it up
    cleaned_pred_df_dict = resolve_count_conflicts(pred_without_gt, new_user_count_target_col,old_user_count_target_col,num_actions_target_col,user_action_conflict_resolution_option)
    test_dates = pd.date_range(test_start, test_end, freq=GRAN)
    print("\nInserting dates...")
    for infoID in infoIDs:
        clean_df = cleaned_pred_df_dict[infoID]
        cols = list(clean_df)
        clean_df["nodeTime"]=test_dates
        clean_df = clean_df[["nodeTime"]+cols]
        clean_df["nodeTime"]=pd.to_datetime(clean_df["nodeTime"], utc=True)
        print(clean_df)

    #save clean preds
    clean_output_dir = output_dir + "Cleaned-Pred-Count-Files/"
    create_output_dir(clean_output_dir)
    for infoID in infoIDs:
        hyp_infoID = hyp_dict[infoID]
        clean_df = cleaned_pred_df_dict[infoID]
        print(clean_df)
        output_fp = clean_output_dir + hyp_infoID + ".csv"
        clean_df.to_csv(output_fp, index=False)
        print(output_fp)

    return cleaned_pred_df_dict

def agg_sim_results(infoID_to_model_sim_dict, infoIDs):

    for infoID in infoIDs:
        cur_df = infoID_to_model_sim_dict[infoID]
        cur_df["edge_weight"] = cur_df.groupby(["nodeUserID"])["parentUserID"].transform("count")
        cur_df = cur_df[["nodeTime" ,"informationID", "nodeUserID", "parentUserID", "edge_weight"]]
        cur_df = cur_df.drop_duplicates().reset_index(drop=True)
        infoID_to_model_sim_dict[infoID] = cur_df

        print()
        print(infoID)
        print(cur_df)

    return infoID_to_model_sim_dict

def compare_model_and_baseline(model_df, baseline_df,metric_col, tag,lower_is_better,category_cols=["infoID"],  model_name="VAM", output_dir=""):
    model_metric_col = model_name + "_" + metric_col
    baseline_metric_col = "Persistence_Baseline_" + metric_col

    model_df = model_df.rename(columns={metric_col:model_metric_col})
    baseline_df = baseline_df.rename(columns={metric_col:baseline_metric_col})

    print("\nmodel_df")
    print(model_df)

    print("\nbaseline_df")
    print(baseline_df)

    comp_df = pd.merge(model_df, baseline_df, on=category_cols, how="inner")

    if lower_is_better == True:
        comp_df["VAM_is_winner"] = [1 if model_metric_val <= baseline_metric_val else 0 for model_metric_val,baseline_metric_val in zip(comp_df[model_metric_col],comp_df[baseline_metric_col])]
    else:
        comp_df["VAM_is_winner"] = [1 if model_metric_val >= baseline_metric_val else 0 for model_metric_val,baseline_metric_val in zip(comp_df[model_metric_col],comp_df[baseline_metric_col])]


    print("\n%s comp df"%(tag))
    print(comp_df)

    if output_dir != "":
        create_output_dir(output_dir)
        result_fp = output_dir + tag + ".csv"
        comp_df.to_csv(result_fp, index=False)
        print(result_fp)

    return comp_df

def get_ua_ndcg(gt_input_dir, infoID_to_model_sim_dict, query_groupby_cols, edge_weight_col, 
    WEIGHTED_LIST,main_output_dir,platform, infoIDs, hyp_dict, degree_types, user_types, 
    cur_platform_user_df, GRAN,REMOVE_MISSING_USERS):

    ndcg_keep_cols = ["nodeUserID", "parentUserID"]
    # query_groupby_cols = ["query_user"]
    # edge_weight_col = "edge_weight"
    # WEIGHTED_LIST = [True, False]

    ndcg_output_dir = main_output_dir + "/NDCG-Results/"
    create_output_dir(ndcg_output_dir)



    all_weighted_dfs = []
    all_unweighted_dfs = []

    ndcg_infoID_output_dir = ndcg_output_dir + "InfoID-Results/"
    create_output_dir(ndcg_infoID_output_dir)

    all_ndcg_dfs = []
    for infoID in infoIDs:

        #lists for df
        user_type_df_list = []
        dt_df_list = []
        weight_df_list = []
        ndcg_df_list = []
        infoID_df_list = []


        hyp_infoID = hyp_dict[infoID]
        input_fp = gt_input_dir + hyp_infoID + ".csv"
        gdf = pd.read_csv(input_fp)

        # input_fp = model_input_dir + hyp_infoID + ".csv"
        # sdf = pd.read_csv(input_fp)

        sdf = infoID_to_model_sim_dict[infoID]

        if REMOVE_MISSING_USERS == True:
            gdf = gdf[gdf["nodeUserID"] != "missing_parentUserID"]
            gdf = gdf[gdf["parentUserID"] != "missing_parentUserID"]
            sdf = sdf[sdf["nodeUserID"] != "missing_parentUserID"]
            sdf = sdf[sdf["parentUserID"] != "missing_parentUserID"]

        gdf["edge_weight"] = gdf.groupby(["nodeUserID", "parentUserID"])["parentUserID"].transform("count")
        sdf["edge_weight"] = sdf.groupby(["nodeUserID", "parentUserID"])["parentUserID"].transform("count")

        gdf = gdf[["nodeUserID", "parentUserID", "edge_weight"]].drop_duplicates().reset_index(drop=True)
        sdf = sdf[["nodeUserID", "parentUserID", "edge_weight"]].drop_duplicates().reset_index(drop=True)

        gdf = gdf.sort_values("edge_weight", ascending=False).reset_index(drop=True)
        sdf = sdf.sort_values("edge_weight", ascending=False).reset_index(drop=True)



        print()
        print(gdf)

        print()
        print(sdf)

        

        #temp old users
        temp_infoID_user_df = cur_platform_user_df[cur_platform_user_df["informationID"]==infoID]
        old_user_set = set(list(temp_infoID_user_df["nodeUserID"]) + ["missing_parentUserID"])

        query_output_dir = ndcg_output_dir + "Query-Tables/%s/"%hyp_infoID
        create_output_dir(query_output_dir)

        #ndcg
        # edge_weight_col = 
        for degree_type in degree_types:
            for user_type in user_types:
                for weighted in WEIGHTED_LIST:
                    if weighted == True:
                        weight_tag = "weighted"
                    else:
                        weight_tag = "unweighted"
                    cur_infoID_avg_ndcg, query_df = get_fancy_ndcg(gdf,sdf ,degree_type, user_type, old_user_set, GRAN, edge_weight_col, query_groupby_cols,weighted=weighted )

                    query_fp = query_output_dir + "%s-%s-%s-%s.csv"%(hyp_infoID, degree_type, user_type, weighted)
                    query_df.to_csv(query_fp, index=False)
                    print(query_df)
                    print(query_fp)

                    print("\n%s %s %s %s AVG SCORE: %.4f"%(infoID, degree_type, user_type, weighted, cur_infoID_avg_ndcg))

                    user_type_df_list.append(user_type)
                    dt_df_list.append(degree_type)
                    weight_df_list.append(weight_tag)
                    ndcg_df_list.append(cur_infoID_avg_ndcg)
                    infoID_df_list.append(infoID)

        cur_infoID_result_df = pd.DataFrame(data={"infoID":infoID_df_list, "degree_type":dt_df_list, "ndcg_score":ndcg_df_list, "weighted":weight_df_list, "user_type":user_type_df_list})
        col_order = ["infoID", "degree_type", "user_type", "weighted", "ndcg_score"]
        cur_infoID_result_df = cur_infoID_result_df[col_order]
        print("\ncur_infoID_result_df")
        print(cur_infoID_result_df)

        #save
        output_fp = ndcg_infoID_output_dir + hyp_infoID + ".csv"
        cur_infoID_result_df.to_csv(output_fp, index=False)
        print(output_fp)

        all_ndcg_dfs.append(cur_infoID_result_df)

    full_ndf = pd.concat(all_ndcg_dfs).reset_index(drop=True)

    weighted_ndf = full_ndf[full_ndf["weighted"]=="weighted"].reset_index(drop=True)
    unweighted_ndf = full_ndf[full_ndf["weighted"]=="unweighted"].reset_index(drop=True)

    print("\nweighted_ndf")
    print(weighted_ndf)

    print("\nunweighted_ndf")
    print(unweighted_ndf)

    output_fp = ndcg_output_dir + "All-Weighted-NDCG-Results.csv"
    weighted_ndf.to_csv(output_fp, index=False)
    print(output_fp)

    output_fp = ndcg_output_dir + "All-Unweighted-NDCG-Results.csv"
    unweighted_ndf.to_csv(output_fp, index=False)
    print(output_fp)

    avg_weighted_ndf = weighted_ndf.copy()
    avg_unweighted_ndf = unweighted_ndf.copy()

    avg_weighted_ndf["avg_ndcg_score"] = avg_weighted_ndf.groupby("infoID")["ndcg_score"].transform("mean")
    avg_weighted_ndf = avg_weighted_ndf[["infoID", "avg_ndcg_score"]].drop_duplicates().sort_values("avg_ndcg_score", ascending=False).reset_index(drop=True)
    print("\navg_weighted_ndf")
    print(avg_weighted_ndf)

    output_fp = ndcg_output_dir + "Summary-Weighted-NDCG-Results.csv"
    avg_weighted_ndf.to_csv(output_fp, index=False)
    print(output_fp)

    avg_unweighted_ndf["avg_ndcg_score"] = avg_unweighted_ndf.groupby("infoID")["ndcg_score"].transform("mean")
    avg_unweighted_ndf = avg_unweighted_ndf[["infoID", "avg_ndcg_score"]].drop_duplicates().sort_values("avg_ndcg_score", ascending=False).reset_index(drop=True)
    print("\navg_unweighted_ndf")
    print(avg_unweighted_ndf)

    output_fp = ndcg_output_dir + "Summary-Unweighted-NDCG-Results.csv"
    avg_unweighted_ndf.to_csv(output_fp, index=False)
    print(output_fp)

    return weighted_ndf, unweighted_ndf,avg_weighted_ndf, avg_unweighted_ndf, ndcg_output_dir

def get_and_save_rh_results(infoID_to_model_sim_dict, platform, gt_and_baseline_input_dir ,output_dir, model_tag, REMOVE_MISSING_USERS, infoIDs, hyp_dict):
    
    gt_input_dir = gt_and_baseline_input_dir + platform + "/RHD-Materials/Ground-Truth-Indegrees/"

    rh_output_dir = output_dir + "RHD-Materials/Model-Results/"
    create_output_dir(rh_output_dir)
    all_rh_results = []
    for infoID in infoIDs:
        hyp_infoID = hyp_dict[infoID]
        sdf = infoID_to_model_sim_dict[infoID]

        gfp = gt_input_dir + hyp_infoID + ".csv"
        gdf = pd.read_csv(gfp)


        model_indegree_dist_df = get_indegree_dist_df(sdf)
        # gt_indegree_dist_df = get_indegree_dist_df(gdf)

        print("\nmodel_indegree_dist_df")
        print(model_indegree_dist_df)

        if REMOVE_MISSING_USERS == True:
            model_indegree_dist_df = model_indegree_dist_df[model_indegree_dist_df["nodeUserID"] != "missing_parentUserID"]
            # gt_indegree_dist_df = gt_indegree_dist_df[gt_indegree_dist_df["parentUserID"] != "missing_parentUserID"]


        output_fp = rh_output_dir + hyp_infoID + ".csv"
        model_indegree_dist_df.to_csv(output_fp , index=False)
        print(output_fp)

        cur_rh = rh_distance(gdf["nodeUserID_in_degree"],model_indegree_dist_df["nodeUserID_in_degree"])
        all_rh_results.append(cur_rh)

    print(infoIDs)
    print(all_rh_results)

    rh_df = pd.DataFrame(data={"infoID":infoIDs, "RHD":all_rh_results})
    rh_df = rh_df[["infoID", "RHD"]]

    print("\nrh_df")
    print(rh_df)

    output_fp = rh_output_dir + "RHD-Results.csv"
    rh_df.to_csv(output_fp, index=False)
    print(output_fp)

    return rh_df

def get_and_save_rh_results_v2_cp4_update(infoID_to_model_sim_dict, platform, gt_and_baseline_input_dir ,output_dir, model_tag, REMOVE_MISSING_USERS, infoIDs, hyp_dict):
    
    gt_input_dir = gt_and_baseline_input_dir + platform + "/RHD-Materials/Ground-Truth-Indegrees/"

    rh_output_dir = output_dir + "RHD-Materials/Model-Results/"
    create_output_dir(rh_output_dir)
    all_rh_results = []
    for infoID in infoIDs:
        hyp_infoID = hyp_dict[infoID]
        sdf = infoID_to_model_sim_dict[infoID]

        gfp = gt_input_dir + hyp_infoID + ".csv"
        gdf = pd.read_csv(gfp)


        model_indegree_dist_df = get_indegree_dist_df(sdf)
        # gt_indegree_dist_df = get_indegree_dist_df(gdf)

        print("\nmodel_indegree_dist_df")
        print(model_indegree_dist_df)

        if REMOVE_MISSING_USERS == True:
            model_indegree_dist_df = model_indegree_dist_df[model_indegree_dist_df["nodeUserID"] != "missing_parentUserID"]
            gt_indegree_dist_df = gt_indegree_dist_df[gt_indegree_dist_df["nodeUserID"] != "missing_parentUserID"]


        output_fp = rh_output_dir + hyp_infoID + ".csv"
        model_indegree_dist_df.to_csv(output_fp , index=False)
        print(output_fp)

        cur_rh = rh_distance(gdf["nodeUserID_in_degree"],model_indegree_dist_df["nodeUserID_in_degree"])
        all_rh_results.append(cur_rh)

    print(infoIDs)
    print(all_rh_results)

    rh_df = pd.DataFrame(data={"infoID":infoIDs, "RHD":all_rh_results})
    rh_df = rh_df[["infoID", "RHD"]]

    print("\nrh_df")
    print(rh_df)

    output_fp = rh_output_dir + "RHD-Results.csv"
    rh_df.to_csv(output_fp, index=False)
    print(output_fp)

    return rh_df

def get_infoID_to_vdf_dict_v2_cp4(infoIDs, hyp_dict, re_org_ft_input_dir, GRAN, history_start, history_end, target_cats,target_cat_rename_dict):

    infoID_to_vdf_dict = {}

    for infoID in infoIDs:

        #load data
        hyp_infoID = hyp_dict[infoID]

        # try:
        #     cur_input_fp = re_org_ft_input_dir  +"GRAN-%s/features/%s-features.csv"%(GRAN, hyp_infoID)
        #     df = pd.read_csv(cur_input_fp)
        # except FileNotFoundError:
        cur_input_fp = re_org_ft_input_dir  +"%s-features.csv"%(hyp_infoID)
        df = pd.read_csv(cur_input_fp)
        
        df["nodeTime"]=pd.to_datetime(df["nodeTime"], utc=True)
        df = config_df_by_dates(df, history_start, history_end, "nodeTime")
        print(df)

        df = df[["nodeTime"] + target_cats]
        df = df.rename(columns=target_cat_rename_dict)
        new_cols = list(df)
        
        # if KEEP_NODE_TIME == False:
        new_cols.remove("nodeTime")
        
        print()
        df["timestep"] = [i for i in range(1, df.shape[0]+1)]
        df = df[["nodeTime","timestep"]+new_cols]
        print(df)

        infoID_to_vdf_dict[infoID] = df

    return infoID_to_vdf_dict

def get_infoID_to_vdf_dict(infoIDs, hyp_dict, re_org_ft_input_dir, GRAN, history_start, history_end, target_cats,target_cat_rename_dict):

    infoID_to_vdf_dict = {}

    for infoID in infoIDs:

        #load data
        hyp_infoID = hyp_dict[infoID]

        try:
            cur_input_fp = re_org_ft_input_dir  +"GRAN-%s/features/%s-features.csv"%(GRAN, hyp_infoID)
            df = pd.read_csv(cur_input_fp)
        except FileNotFoundError:
            cur_input_fp = re_org_ft_input_dir  +"features/%s-features.csv"%(hyp_infoID)
            df = pd.read_csv(cur_input_fp)
        
        df["nodeTime"]=pd.to_datetime(df["nodeTime"], utc=True)
        df = config_df_by_dates(df, history_start, history_end, "nodeTime")
        print(df)

        df = df[["nodeTime"] + target_cats]
        df = df.rename(columns=target_cat_rename_dict)
        new_cols = list(df)
        
        new_cols.remove("nodeTime")
        print()

        df["timestep"] = [i for i in range(1, df.shape[0]+1)]
        df = df[["nodeTime","timestep"]+new_cols]
        print(df)

        infoID_to_vdf_dict[infoID] = df

    return infoID_to_vdf_dict


def get_timestep_info(history_start, history_end, test_start, test_end,GRAN):

    history_dates = pd.date_range(history_start, history_end, freq=GRAN)
    test_dates = pd.date_range(test_start, test_end, freq=GRAN)

    num_history_dates = len(history_dates)
    num_test_dates = len(test_dates)
    print("\nnum_history_dates: %d; num_test_dates: %d"%(num_history_dates, num_test_dates))

    history_timestep_range = (1, num_history_dates)
    test_timestep_range = (num_history_dates+1, num_history_dates + num_test_dates)
    num_overall_timesteps = num_history_dates + num_test_dates
    print("\nnum_overall_timesteps: %d"%num_overall_timesteps)

    print("\nhistory_timestep_range: %s"%str(history_timestep_range))
    print("\ntest_timestep_range: %s"%str(test_timestep_range))


    return num_overall_timesteps,num_history_dates,num_test_dates, history_timestep_range,test_timestep_range


def get_weight_start_and_weight_end_lists(TEST_START_DATE, HISTORICAL_LOOKBACK_DAYS_LIST):

    #first lets get the latest day
    TEST_START_DATE = pd.to_datetime(TEST_START_DATE, utc=True)
    CUR_DAY = TEST_START_DATE - pd.to_timedelta(1, unit="ms")
    print("\nCUR_DAY")
    print(CUR_DAY)

    weight_start_and_weight_end_lists =[]

    for h in HISTORICAL_LOOKBACK_DAYS_LIST:

        INITIAL_HISTORY_DAY = TEST_START_DATE - pd.to_timedelta(h, unit="D")
        cur_tuple = (INITIAL_HISTORY_DAY, CUR_DAY)
        weight_start_and_weight_end_lists.append(cur_tuple)
        print(cur_tuple)

    return weight_start_and_weight_end_lists

def get_pr_dist_df(df):
    #fix df
    df["full_edge_weight"] = df.groupby(["nodeUserID", "parentUserID"])["parentUserID"].transform("count")
    df = df[["nodeUserID", "parentUserID", "full_edge_weight"]].drop_duplicates()
    df = df.sort_values("full_edge_weight", ascending=False).reset_index(drop=True)
    print("\ndf")
    print(df)

    #edge
    EDGE_LIST = zip(df["nodeUserID"],df["parentUserID"],df["full_edge_weight"])
    GRAPH =nx.DiGraph()
    GRAPH.add_weighted_edges_from(EDGE_LIST)

    #pr
    try:
        PAGE_RANK_DICT = nx.pagerank(GRAPH, weight="weight")
    except:
        PAGE_RANK_DICT = nx.pagerank_numpy(GRAPH, weight="weight")

    PAGE_RANK_DF = pd.DataFrame(list(PAGE_RANK_DICT.items()), columns=["node", "page_rank"])

    if PAGE_RANK_DF.shape[0] == 0:
        PAGE_RANK_DF = pd.DataFrame(data={"node":["n/a"], "page_rank":[0]})

    PAGE_RANK_DF = PAGE_RANK_DF.sort_values("page_rank", ascending=False).reset_index(drop=True)

    return PAGE_RANK_DF

def get_indegree_dist_df(df):
    df=df[["nodeUserID", "parentUserID"]].drop_duplicates().reset_index(drop=True)
    df["parentUserID_in_degree"] = df.groupby(["parentUserID"])["nodeUserID"].transform("count")
    user_to_degree_dict = convert_df_2_cols_to_dict(df, "parentUserID", "parentUserID_in_degree")
    all_users = list(df["parentUserID"].unique()) + list(df["nodeUserID"].unique())
    all_users =list(set(all_users))
    df = pd.DataFrame(data={"nodeUserID":all_users})
    df["nodeUserID_in_degree"] = df["nodeUserID"].map(user_to_degree_dict)
    df["nodeUserID_in_degree"] =df["nodeUserID_in_degree"].fillna(0)
    df=df.sort_values("nodeUserID_in_degree", ascending=False).reset_index(drop=True)
    df["nodeUserID_in_degree"] = df["nodeUserID_in_degree"].astype("int32")

    df = df[["nodeUserID", "nodeUserID_in_degree"]]
    return df







def overlap_similarity(s1, s2):

    i = s1.intersection(s2)
    i_len = len(i)

    s1_len = len(s1)
    s2_len = len(s2)
    min_val = np.min([s1_len, s2_len])

    if (s1_len ==0) and (s2_len==0):
        return 1

    if min_val == 0:
        return 0

    os = i_len/float(min_val)



    return os

def jaccard_similarity(s1, s2):

    i = s1.intersection(s2)
    i_len = len(i)
    u = s1.union(s2)
    u_len = len(u)

    if (u_len == 0):
        return 1

    js = i_len/float(u_len)

    return js

def reformat_preds_for_ua_model(model_test_infoID_to_pred_df_dict, infoIDs, target_cats, platform, num_timesteps):
    infoID_df_list = []
    target_df_list = []
    timestep_df_list = []
    pred_df_list = []
    platform_df_list = []

    for infoID in infoIDs:
        for target_cat in target_cats:
            for t in range(num_timesteps):
                infoID_df_list.append(infoID)
                target_df_list.append(target_cat)
                pred = list(model_test_infoID_to_pred_df_dict[infoID][target_cat])[t]
                pred_df_list.append(pred)
                timestep_df_list.append(t+1)
                platform_df_list.append(platform)

    df = pd.DataFrame(data={"infoID":infoID_df_list, "target_ft":target_df_list, "timestep":timestep_df_list, "pred":pred_df_list, "platform":platform_df_list})
    df = df[["infoID", "target_ft", "timestep", "pred", "platform"]]
    return df

def merge_mult_dfs(merge_list, on, how, VERBOSE=True):
    if VERBOSE == True:
        print("\nMerging multiple dfs...")
    return reduce(lambda  left,right: pd.merge(left,right,on=on, how=how), merge_list)

def resolve_count_conflicts_v2_add_nodeTime(pred_df, new_user_count_target_col,old_user_count_target_col,num_actions_target_col,conflict_resolution_option, test_start, test_end, BASIC_GRAN):

    infoIDs = list(pred_df["infoID"].unique())
    target_fts = list(pred_df["target_ft"].unique())

    test_dates = pd.date_range(test_start, test_end, freq=BASIC_GRAN)

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

        cur_pred_df["nodeTime"]=test_dates
        cur_pred_df["nodeTime"]=pd.to_datetime(cur_pred_df["nodeTime"], utc=True)



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
    keep_cols = ["timestep" ,"nodeTime" ,"num_new_users", "num_old_users", "num_actions"]
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

def get_old_user_weight_table_info(cleaned_pred_df_dict, infoIDs):

    infoID_to_weight_table_info_df_dict = {}

    for infoID in infoIDs:
        print()
        print(infoID)

        pred_df = cleaned_pred_df_dict[infoID].copy()

        newly_added_synthetic_user_list = list(pred_df["num_new_users"])
        newly_added_synthetic_user_list = [0] + newly_added_synthetic_user_list[:-1]
        pred_df["num_old_syn_users"]=newly_added_synthetic_user_list
        pred_df["cumsum_old_syn_users"]=pred_df["num_old_syn_users"].cumsum()

        print(pred_df)

    return

#get volume history df
def get_volume_history_df_dict(volume_history_dir,hyp_dict, platform,history_start,history_end):

    print("\nGetting volume history...")
    volume_dir = volume_history_dir + platform + "/"
    infoID_to_vol_history_df_dict = {}

    for infoID,hyp_infoID in hyp_dict.items():
        try:
            volume_fp = volume_dir + hyp_infoID + ".csv"
            df = pd.read_csv(volume_fp)
        except FileNotFoundError:
            if hyp_infoID == "other":
                volume_fp = volume_dir + "no_narrative.csv"
                df = pd.read_csv(volume_fp)
            else:
                print("Error! %s does not exist"%volume_fp)
                sys.exit(0)
        
        df = config_df_by_dates(df, history_start, history_end, "nodeTime")
        print(df)
        infoID_to_vol_history_df_dict[infoID]=df

    return infoID_to_vol_history_df_dict



def combine_volume_history_with_simulation_v2_alter_nodeTimes(infoID_to_vdf_dict, cleaned_pred_df_dict, infoIDs, test_start, test_end):

    test_start = pd.to_datetime(test_start, utc=True)
    test_end = pd.to_datetime(test_end, utc=True)

    infoID_to_history_and_pred_dict = {}
    print("\nCombining history with preds")
    col_order = ["nodeTime", "timestep", "num_actions", "num_new_users", "num_old_users", "category"]
    for infoID in infoIDs:

        pred_df = cleaned_pred_df_dict[infoID]
        pred_df["nodeTime"]=pd.to_datetime(pred_df["nodeTime"] ,utc=True)
        pred_df = config_df_by_dates(pred_df, test_start, test_end, "nodeTime")


        pred_df["category"]="pred"
        history_df = infoID_to_vdf_dict[infoID]
        history_df["category"]="history"

        df = pd.concat([history_df, pred_df]).reset_index(drop=True)
        df["nodeTime"]=pd.to_datetime(df["nodeTime"], utc=True)
        df = df.sort_values("nodeTime").reset_index(drop=True)
        df["timestep"]=[i for i in range(1, df.shape[0]+1)]
        df = df[col_order]
        infoID_to_history_and_pred_dict[infoID]=df
        print()
        print(infoID)
        print(df)

    return infoID_to_history_and_pred_dict


def combine_volume_history_with_simulation(infoID_to_vdf_dict, cleaned_pred_df_dict, infoIDs):

    infoID_to_history_and_pred_dict = {}
    print("\nCombining history with preds")
    col_order = ["nodeTime", "timestep", "num_actions", "num_new_users", "num_old_users", "category"]
    for infoID in infoIDs:

        pred_df = cleaned_pred_df_dict[infoID]
        pred_df["category"]="pred"
        history_df = infoID_to_vdf_dict[infoID]
        history_df["category"]="history"

        df = pd.concat([history_df, pred_df]).reset_index(drop=True)
        df["nodeTime"]=pd.to_datetime(df["nodeTime"], utc=True)
        df = df.sort_values("nodeTime").reset_index(drop=True)
        df["timestep"]=[i for i in range(1, df.shape[0]+1)]
        df = df[col_order]
        infoID_to_history_and_pred_dict[infoID]=df
        print()
        print(infoID)
        print(df)

    return infoID_to_history_and_pred_dict

def get_infoID_to_old_user_df_dict_v2_parents(df, infoIDs,history_start,history_end, GRAN, platform, user_col="nodeUserID"):

    print("\nGetting infoID_to_old_user_df_dict...")

    df["nodeTime"]=pd.to_datetime(df["nodeTime"], utc=True)
    df["nodeTime"]=df["nodeTime"].dt.floor(GRAN)

    dates = pd.date_range(history_start, history_end, freq=GRAN)
    dates = pd.Series(dates)
    dates = pd.to_datetime(dates, utc=True)
    df = df[df["platform"]==platform]

    if user_col == "parentUserID":
        df = df[["nodeTime", "informationID", "nodeUserID", "nodeID", "parentID", "rootID"]].drop_duplicates().reset_index(drop=True)
    else:
        df = df[["nodeTime", "informationID", "nodeUserID"]]

    df = config_df_by_dates(df, history_start, history_end, "nodeTime")

    # dates = pd.to_datetime(dates, utc=True)
    date_to_timestep_dict = {}
    for i,date in enumerate(dates):
        timestep = i+1
        date_to_timestep_dict[date]=timestep

    
    df["timestep"]=df["nodeTime"].map(date_to_timestep_dict)
    print(df)

    if "other" in infoIDs:
        df["informationID"]=df["informationID"].replace({"no_narrative":"other"})
        print(df)
        # sys.exit(0)

    infoID_to_old_user_df_dict = {}
    for infoID in infoIDs:
        cur_infoID_df = df[df["informationID"]==infoID].reset_index(drop=True)

        if user_col=="parentUserID":
            cur_infoID_df = get_parentUserID_col_v2_pnnl_version(cur_infoID_df)
            cur_infoID_df = cur_infoID_df[["nodeTime","timestep", "informationID", "parentUserID"]]

        infoID_to_old_user_df_dict[infoID]=cur_infoID_df
        print()
        print(infoID)
        print(cur_infoID_df)


    return infoID_to_old_user_df_dict

def get_infoID_to_old_user_df_dict(df, infoIDs,history_start,history_end, GRAN, platform):

    print("\nGetting infoID_to_old_user_df_dict...")

    df["nodeTime"]=pd.to_datetime(df["nodeTime"], utc=True)
    df["nodeTime"]=df["nodeTime"].dt.floor(GRAN)

    dates = pd.date_range(history_start, history_end, freq=GRAN)
    dates = pd.Series(dates)
    dates = pd.to_datetime(dates, utc=True)
    df = df[df["platform"]==platform]
    df = df[["nodeTime", "informationID", "nodeUserID"]].drop_duplicates().reset_index(drop=True)

    df = config_df_by_dates(df, history_start, history_end, "nodeTime")

    # dates = pd.to_datetime(dates, utc=True)
    date_to_timestep_dict = {}
    for i,date in enumerate(dates):
        timestep = i+1
        date_to_timestep_dict[date]=timestep

    
    df["timestep"]=df["nodeTime"].map(date_to_timestep_dict)
    print(df)

    if "other" in infoIDs:
        df["informationID"]=df["informationID"].replace({"no_narrative":"other"})
        print(df)
        # sys.exit(0)

    infoID_to_old_user_df_dict = {}
    for infoID in infoIDs:
        cur_infoID_df = df[df["informationID"]==infoID].reset_index(drop=True)
        infoID_to_old_user_df_dict[infoID]=cur_infoID_df
        print()
        print(infoID)
        print(cur_infoID_df)


    return infoID_to_old_user_df_dict

def resolve_old_user_weight_table_conflicts(infoID_to_history_and_pred_dict,infoID_to_old_user_df_dict, test_start, test_end, infoIDs,num_overall_timesteps, history_timestep_range,test_timestep_range):

    infoID_to_conflict_record_dict = {}
    infoID_to_lookback_dict = {}

    for infoID in infoIDs:

        hp_df = infoID_to_history_and_pred_dict[infoID]
        old_user_df = infoID_to_old_user_df_dict[infoID]

        print("\nhp_df")
        print(hp_df)

        print("\nold_user_df")
        print(old_user_df)

        print("\ntest_timestep_range")
        print(test_timestep_range)

        test_timestep_start = test_timestep_range[0]
        test_timestep_end = test_timestep_range[1]

        CUR_LB = 1
        s = 1
        pred_idx = 0
        temp = hp_df[hp_df["category"]=="pred"]
        old_user_preds=list(temp["num_old_users"])
        new_user_preds = list(temp["num_new_users"])
        num_old_syn_users_in_lb_period = 0

        all_records = []
        for test_timestep in range(test_timestep_start, test_timestep_end + 1):

            attempt=1
            while(True):
                T_CUR = test_timestep - 1
                num_pred_old_users_at_tcur_plus1 = old_user_preds[pred_idx]
                num_pred_new_users_at_tcur_plus1 = new_user_preds[pred_idx]

                #get nunique old users
                low_end_ts = T_CUR - CUR_LB + 1
                high_end_ts = T_CUR
                print("\nlow_end_ts: %d, high_end_ts: %d"%(low_end_ts, high_end_ts))
                temp = old_user_df[(old_user_df["timestep"]>= low_end_ts) & (old_user_df["timestep"]<= high_end_ts)]
                print(temp)
                nunique_old_users_in_lb_period = temp["nodeUserID"].nunique()
                


                if test_timestep > test_timestep_start:
                    temp2 = hp_df[(hp_df["timestep"]>= low_end_ts) & (hp_df["timestep"]<= high_end_ts)]
                    print("\nnew user temp")
                    print(temp2)
                    num_old_syn_users_in_lb_period=temp2["num_new_users"].sum()
                    print("\nnum_old_syn_users_in_lb_period: %d"%num_old_syn_users_in_lb_period)

                total_old = nunique_old_users_in_lb_period + num_old_syn_users_in_lb_period

                print("\nnunique_old_users_in_lb_period: %d"%nunique_old_users_in_lb_period)
                print("num_old_syn_users_in_lb_period: %d"%num_old_syn_users_in_lb_period)
                
                print("\ntotal_old: %d"%total_old)

                print("\nnum_pred_old_users_at_tcur_plus1: %d"%num_pred_old_users_at_tcur_plus1)

                old_user_leftovers = total_old - num_pred_old_users_at_tcur_plus1
                print("\nold_user_leftovers: %d"%old_user_leftovers)

                

                cur_record_dict = {
                    "attempt":[attempt],
                    "T_CUR":[T_CUR],
                    "s":[s],
                    "CUR_LB":[CUR_LB],
                    "num_pred_old_users (tcur+1)":[num_pred_old_users_at_tcur_plus1],
                    "num_pred_new_users (tcur+1)":[num_pred_new_users_at_tcur_plus1],
                    "nunique_old_users_in_lb_period":[nunique_old_users_in_lb_period], 
                    "num_old_syn_users_in_lb_period":[num_old_syn_users_in_lb_period],
                    "total_old":[total_old],
                    "old_user_leftovers":[old_user_leftovers]
                    }

                cur_record_df = pd.DataFrame(data=cur_record_dict)
                print("\ncur_record_df")
                print(cur_record_df)
                all_records.append(cur_record_df)

                if old_user_leftovers >=0:
                    break
                attempt+=1
                CUR_LB+=1
            pred_idx+=1
            s+=1

            # sys.exit(0)

        all_records = pd.concat(all_records).reset_index(drop=True)
        print("\nall_records")
        print(all_records)
        infoID_to_conflict_record_dict[infoID]=all_records
        infoID_to_lookback_dict[infoID]=CUR_LB

    for infoID in infoIDs:
        CUR_LB = infoID_to_lookback_dict[infoID]
        print("\n%s: %d"%(infoID, CUR_LB))




    return infoID_to_conflict_record_dict, infoID_to_lookback_dict






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




def get_fancy_ndcg(gdf,sdf ,degree_type, user_type, old_user_set, GRAN, edge_weight_col, query_groupby_cols,weighted=True, child_col="nodeUserID_ndcg" , parent_col="parentUserID_ndcg"):

    MOD_NUM = 100

    if weighted == True:
        weight_tag = "weighted"
    else:
        weight_tag = "unweighted"

    #reconfig
    if degree_type == "indegree":
        query_col = parent_col
        result_col = child_col
    else:
        query_col = child_col
        result_col = parent_col

    # #make sure gran is consistent
    # gdf["nodeTime"] = pd.to_datetime(gdf["nodeTime"], utc=True).dt.floor(GRAN)
    # sdf["nodeTime"] = pd.to_datetime(sdf["nodeTime"], utc=True).dt.floor(GRAN)

    print("\nsdf")
    print(sdf)

    print("\ngdf")
    print(gdf)

    gdf = gdf.drop_duplicates()
    sdf = sdf.drop_duplicates()


    print("\nsdf")
    print(sdf)

    print("\ngdf")
    print(gdf)

    gdf["%s_is_new"%query_col] = [0 if user in old_user_set else 1 for user in list(gdf[query_col])]
    sdf["%s_is_new"%query_col] = [0 if user in old_user_set else 1 for user in list(sdf[query_col])]

    if user_type == "new":
        gdf = gdf[gdf["%s_is_new"%query_col]==1]
        sdf = sdf[sdf["%s_is_new"%query_col]==1]
    elif user_type == "old":
        gdf = gdf[gdf["%s_is_new"%query_col]==0]
        sdf = sdf[sdf["%s_is_new"%query_col]==0]

    gdf = gdf.rename(columns={query_col:"query_user", result_col : "result", edge_weight_col:"score"})
    sdf = sdf.rename(columns={query_col:"query_user", result_col : "result", edge_weight_col:"score"})

    col_order =query_groupby_cols+ ["result", "score"]

    sdf["neg_score"]=sdf["score"] * -1
    gdf["neg_score"]=gdf["score"] * -1
    # sdf = sdf.sort_values(["nodeTime", "query_user", "neg_score"]).reset_index(drop=True)
    # gdf = gdf.sort_values(["nodeTime", "query_user", "neg_score"]).reset_index(drop=True)

    sdf = sdf.sort_values(query_groupby_cols + ["neg_score"]).reset_index(drop=True)
    gdf = gdf.sort_values(query_groupby_cols + ["neg_score"]).reset_index(drop=True)
    sdf = sdf[col_order]
    gdf = gdf[col_order]

    print("\nsdf")
    print(sdf)

    print("\ngdf")
    print(gdf)

    #combine the dfs
    gdf = gdf.rename(columns={"score":"gt_score" })
    sdf = sdf.rename(columns={"score":"pred_score"})

    gt_users = gdf["query_user"].unique()
    gt_users = set(gt_users)

    print("\nbefore gt filter")
    print(sdf)

    sdf = sdf[sdf["query_user"].isin(gt_users)].reset_index(drop=True)

    print("\nafter gt filter")
    print(sdf)

    # sys.exit(0)

    if (sdf.shape[0]==0) and (gdf.shape[0]==0):
        df = pd.DataFrame(data={"query_user":[], "result":[], "gt_score":[], "pred_score":[]})
        return 1, df

    df = pd.merge(gdf, sdf, on=query_groupby_cols + ["result"], how="outer" )

    df["result"] = df["result"].fillna("n/a")
    df = df.fillna(0)
    df = df.drop_duplicates().reset_index(drop=True)

    print("\nmerged df")
    print(df)

    print(df[df["result"]=="n/a"])

    # sys.exit(0)
    #make weights

    if weighted == True:
        df["query_group_weight"]=df.groupby(query_groupby_cols)["gt_score"].transform("sum")
        qw = df[query_groupby_cols+ ["query_group_weight"]].drop_duplicates().reset_index(drop=True)
        print("\nqw")
        print(qw)
        raw_weight_total = qw["query_group_weight"].sum()
        df["query_group_weight"]=df["query_group_weight"]/raw_weight_total

        print()
        print(df["query_group_weight"])
    else:
        df["query_group_weight"]= 1



    temp = df[(df["pred_score"]>0) & (df["gt_score"]>0)].reset_index(drop=True)
    print("\ntemp")
    print(temp)


    df["query_group"] =""
    for i,col in enumerate(query_groupby_cols):

        if i != len(query_groupby_cols) - 1:
            df["query_group"] = df["query_group"]  + df[col]   +"<with>"
        else:
            df["query_group"] = df["query_group"]  + df[col]

    # df["query_nodeTime_pair"] = df["query_user"].astype("str") +"<with>" + df["nodeTime"].astype("str")

    print("\ndf with q group")
    print(df)

    # df = df[~((df["pred_score"]==0) & (df["gt_score"]==0))].reset_index(drop=True)

    temp = df[(df["pred_score"]>0) & (df["gt_score"]>0)].reset_index(drop=True)
    print("\ntemp")
    print(temp)

    # sys.exit(0)

    #make query gbo
    gbo_list = df.groupby(df["query_group"])
    gbo_list = list(gbo_list)
    gbo_list_size = len(gbo_list)
    print("\nSize of query gbo_list")
    print(gbo_list_size)

    #finally get scores!!!

    y_test = []
    y_pred = []
    sample_weights = []

    all_ndcg_scores = []

    pair_to_n_score_dict = {}
    EPSILON=0.0000001
    for i, g in enumerate(gbo_list):

        if i%MOD_NUM == 0:
            print("\nOn query group %d of %d"%(i+1, gbo_list_size))

        cur_pair = g[0]
        cur_df = g[1]



        sample_weight = cur_df["query_group_weight"].iloc[-1]



        true_relevance =cur_df["gt_score"].values
        scores =cur_df["pred_score"].values

        if np.sum(true_relevance) > 0:

            print()
            print(cur_pair)
            print(cur_df)

            print("\ntrue scores")
            print(true_relevance)

            print("\npred scores")
            print(scores)

        if (np.sum(scores)==0) and (np.sum(true_relevance) !=0):
            n_score = 0
        elif (np.sum(scores) != 0) and (np.sum(true_relevance) ==0):
            n_score = 0
        elif ((scores.shape[0]==1) and (true_relevance.shape[0]==1)) and (np.sum(scores)>=1) and (np.sum(true_relevance) >=1):
            n_score=1
        else:
            n_score = ndcg_score(np.asarray([true_relevance]), np.asarray([scores]))

        n_score = n_score * sample_weight
        pair_to_n_score_dict[cur_pair] = n_score
        all_ndcg_scores.append(n_score)

        if np.sum(true_relevance) > 0:
            print("\nn_score: %.4f"%n_score)



    df["NDCG_Score"] = df["query_group"].map(pair_to_n_score_dict)
    print("\ndf")
    print(df)

    if weighted== True:
        avg_score = np.sum(all_ndcg_scores)
    else:
        avg_score = np.mean(all_ndcg_scores)

    print("\nAVG SCORE: %.4f"%avg_score)

    # print("\n%s %s %s %s score: %.4f"%(infoID,user_type, degree_type, weight_tag, avg_score))

    # user_type_df_list.append(user_type)
    # dt_df_list.append(degree_type)
    # weight_df_list.append(weight_tag)
    # ndcg_df_list.append(avg_score)
    # infoID_df_list.append(infoID)

    # query_output_fp = query_table_dir + hyp_infoID + ".csv"
    # df.to_csv(query_output_fp, index=False)
    # print(query_output_fp)

    return avg_score, df

def resolve_old_user_weight_table_conflicts_v3_also_account_for_new_users(infoID_to_history_and_pred_dict,infoID_to_old_user_df_dict, test_start, test_end, infoIDs,num_overall_timesteps, history_timestep_range,test_timestep_range):

    infoID_to_conflict_record_dict = {}
    infoID_to_lookback_dict = {}

    infoID_to_lookback_list_dict = {}

    for infoID in infoIDs:

        infoID_to_lookback_list_dict[infoID] = []

        hp_df = infoID_to_history_and_pred_dict[infoID]
        old_user_df = infoID_to_old_user_df_dict[infoID]

        print("\nhp_df")
        print(hp_df)

        print("\nold_user_df")
        print(old_user_df)

        print("\ntest_timestep_range")
        print(test_timestep_range)

        test_timestep_start = test_timestep_range[0]
        test_timestep_end = test_timestep_range[1]

       
        s = 1
        pred_idx = 0
        temp = hp_df[hp_df["category"]=="pred"]
        old_user_preds=list(temp["num_old_users"])
        new_user_preds = list(temp["num_new_users"])
        num_old_syn_users_in_lb_period = 0

        all_records = []
        for test_timestep in range(test_timestep_start, test_timestep_end + 1):

            CUR_LB = 1
            attempt=1
            while(True):
                T_CUR = test_timestep - 1
                num_pred_old_users_at_tcur_plus1 = old_user_preds[pred_idx]
                num_pred_new_users_at_tcur_plus1 = new_user_preds[pred_idx]

                #get nunique old users
                low_end_ts = T_CUR - CUR_LB + 1
                high_end_ts = T_CUR
                print("\nlow_end_ts: %d, high_end_ts: %d"%(low_end_ts, high_end_ts))
                temp = old_user_df[(old_user_df["timestep"]>= low_end_ts) & (old_user_df["timestep"]<= high_end_ts)]
                print(temp)
                nunique_old_users_in_lb_period = temp["nodeUserID"].nunique()
                

                #count the new syn users you made in prev time periods
                #this block only activates obviously after the 1st test timestep
                if test_timestep > test_timestep_start:
                    temp2 = hp_df[(hp_df["timestep"]>= low_end_ts) & (hp_df["timestep"]<= high_end_ts)]
                    print("\nnew user temp")
                    print(temp2)
                    num_old_syn_users_in_lb_period=temp2["num_new_users"].sum()
                    print("\nnum_old_syn_users_in_lb_period: %d"%num_old_syn_users_in_lb_period)

                total_old = nunique_old_users_in_lb_period + num_old_syn_users_in_lb_period

                print("\nnunique_old_users_in_lb_period: %d"%nunique_old_users_in_lb_period)
                print("num_old_syn_users_in_lb_period: %d"%num_old_syn_users_in_lb_period)
                
                print("\ntotal_old: %d"%total_old)

                print("\nnum_pred_old_users_at_tcur_plus1: %d"%num_pred_old_users_at_tcur_plus1)

                old_user_leftovers = total_old - num_pred_old_users_at_tcur_plus1
                print("\nold_user_leftovers: %d"%old_user_leftovers)

                new_user_leftovers = total_old- num_pred_new_users_at_tcur_plus1
                print("\nnew_user_leftovers: %d"%new_user_leftovers)

                min_leftovers = np.min([old_user_leftovers,new_user_leftovers])
                print("\nmin_leftovers: %d"%min_leftovers)

                

                cur_record_dict = {
                    "attempt":[attempt],
                    "T_CUR":[T_CUR],
                    "s":[s],
                    "CUR_LB":[CUR_LB],
                    "num_pred_old_users (tcur+1)":[num_pred_old_users_at_tcur_plus1],
                    "num_pred_new_users (tcur+1)":[num_pred_new_users_at_tcur_plus1],
                    "nunique_old_users_in_lb_period":[nunique_old_users_in_lb_period], 
                    "num_old_syn_users_in_lb_period":[num_old_syn_users_in_lb_period],
                    "total_old":[total_old],
                    "old_user_leftovers":[old_user_leftovers],
                    "new_user_leftovers" : [new_user_leftovers],
                    "min_leftovers" : [min_leftovers]
                    }

                cur_record_df = pd.DataFrame(data=cur_record_dict)
                print("\ncur_record_df")
                print(cur_record_df)
                all_records.append(cur_record_df)

                if min_leftovers >=0:
                    break
                attempt+=1
                CUR_LB+=1

            infoID_to_lookback_list_dict[infoID].append(CUR_LB)
            pred_idx+=1
            s+=1

            # sys.exit(0)

        all_records = pd.concat(all_records).reset_index(drop=True)
        print("\nall_records")
        print(all_records)
        infoID_to_conflict_record_dict[infoID]=all_records
        infoID_to_lookback_dict[infoID]=CUR_LB

    # for infoID in infoIDs:
    #     CUR_LB = infoID_to_lookback_dict[infoID]
    #     print("\n%s: %d"%(infoID, CUR_LB))

    lookback_over_time_df = pd.DataFrame(data=infoID_to_lookback_list_dict)
    lookback_over_time_df["timestep"] = [i+1 for i in range(lookback_over_time_df.shape[0])]
    lookback_over_time_df = lookback_over_time_df[["timestep"] + infoIDs]
    print("\nlookback_over_time_df")
    print(lookback_over_time_df)



    return infoID_to_conflict_record_dict, lookback_over_time_df



def resolve_old_user_weight_table_conflicts_v2_dynamic_lookbacks(infoID_to_history_and_pred_dict,infoID_to_old_user_df_dict, test_start, test_end, infoIDs,num_overall_timesteps, history_timestep_range,test_timestep_range):

    infoID_to_conflict_record_dict = {}
    infoID_to_lookback_dict = {}

    infoID_to_lookback_list_dict = {}

    for infoID in infoIDs:

        infoID_to_lookback_list_dict[infoID] = []

        hp_df = infoID_to_history_and_pred_dict[infoID]
        old_user_df = infoID_to_old_user_df_dict[infoID]

        print("\nhp_df")
        print(hp_df)

        print("\nold_user_df")
        print(old_user_df)

        print("\ntest_timestep_range")
        print(test_timestep_range)

        test_timestep_start = test_timestep_range[0]
        test_timestep_end = test_timestep_range[1]

       
        s = 1
        pred_idx = 0
        temp = hp_df[hp_df["category"]=="pred"]
        old_user_preds=list(temp["num_old_users"])
        new_user_preds = list(temp["num_new_users"])
        num_old_syn_users_in_lb_period = 0

        all_records = []
        for test_timestep in range(test_timestep_start, test_timestep_end + 1):

            CUR_LB = 1
            attempt=1
            while(True):
                T_CUR = test_timestep - 1
                num_pred_old_users_at_tcur_plus1 = old_user_preds[pred_idx]
                num_pred_new_users_at_tcur_plus1 = new_user_preds[pred_idx]

                #get nunique old users
                low_end_ts = T_CUR - CUR_LB + 1
                high_end_ts = T_CUR
                print("\nlow_end_ts: %d, high_end_ts: %d"%(low_end_ts, high_end_ts))
                temp = old_user_df[(old_user_df["timestep"]>= low_end_ts) & (old_user_df["timestep"]<= high_end_ts)]
                print(temp)
                nunique_old_users_in_lb_period = temp["nodeUserID"].nunique()
                

                #count the new syn users you made in prev time periods
                #this block only activates obviously after the 1st test timestep
                if test_timestep > test_timestep_start:
                    temp2 = hp_df[(hp_df["timestep"]>= low_end_ts) & (hp_df["timestep"]<= high_end_ts)]
                    print("\nnew user temp")
                    print(temp2)
                    num_old_syn_users_in_lb_period=temp2["num_new_users"].sum()
                    print("\nnum_old_syn_users_in_lb_period: %d"%num_old_syn_users_in_lb_period)

                total_old = nunique_old_users_in_lb_period + num_old_syn_users_in_lb_period

                print("\nnunique_old_users_in_lb_period: %d"%nunique_old_users_in_lb_period)
                print("num_old_syn_users_in_lb_period: %d"%num_old_syn_users_in_lb_period)
                
                print("\ntotal_old: %d"%total_old)

                print("\nnum_pred_old_users_at_tcur_plus1: %d"%num_pred_old_users_at_tcur_plus1)

                old_user_leftovers = total_old - num_pred_old_users_at_tcur_plus1
                print("\nold_user_leftovers: %d"%old_user_leftovers)

                

                cur_record_dict = {
                    "attempt":[attempt],
                    "T_CUR":[T_CUR],
                    "s":[s],
                    "CUR_LB":[CUR_LB],
                    "num_pred_old_users (tcur+1)":[num_pred_old_users_at_tcur_plus1],
                    "num_pred_new_users (tcur+1)":[num_pred_new_users_at_tcur_plus1],
                    "nunique_old_users_in_lb_period":[nunique_old_users_in_lb_period], 
                    "num_old_syn_users_in_lb_period":[num_old_syn_users_in_lb_period],
                    "total_old":[total_old],
                    "old_user_leftovers":[old_user_leftovers]
                    }

                cur_record_df = pd.DataFrame(data=cur_record_dict)
                print("\ncur_record_df")
                print(cur_record_df)
                all_records.append(cur_record_df)

                if old_user_leftovers >=0:
                    break
                attempt+=1
                CUR_LB+=1

            infoID_to_lookback_list_dict[infoID].append(CUR_LB)
            pred_idx+=1
            s+=1

            # sys.exit(0)

        all_records = pd.concat(all_records).reset_index(drop=True)
        print("\nall_records")
        print(all_records)
        infoID_to_conflict_record_dict[infoID]=all_records
        infoID_to_lookback_dict[infoID]=CUR_LB

    # for infoID in infoIDs:
    #     CUR_LB = infoID_to_lookback_dict[infoID]
    #     print("\n%s: %d"%(infoID, CUR_LB))

    lookback_over_time_df = pd.DataFrame(data=infoID_to_lookback_list_dict)
    lookback_over_time_df["timestep"] = [i+1 for i in range(lookback_over_time_df.shape[0])]
    lookback_over_time_df = lookback_over_time_df[["timestep"] + infoIDs]
    print("\nlookback_over_time_df")
    print(lookback_over_time_df)



    return infoID_to_conflict_record_dict, lookback_over_time_df


def get_pair_ndcg_tuple( arg_tuple):

    i,g,gbo_list_size,MOD_NUM = arg_tuple
    if i%MOD_NUM == 0:
        print("\nOn query group %d of %d"%(i+1, gbo_list_size))

    cur_pair = g[0]
    cur_df = g[1]
    sample_weight = cur_df["query_group_weight"].iloc[-1]
    true_relevance =cur_df["gt_score"].values
    scores =cur_df["pred_score"].values

    # if np.sum(true_relevance) > 0:

    print()
    print(cur_pair)
    print(cur_df)

    print("\ntrue scores")
    print(true_relevance)

    print("\npred scores")
    print(scores)

    #trim zeros
    # true_relevance = np.trim_zeros(true_relevance, "b") 
    # new_size = true_relevance.shape[0]
    # scores = scores[:new_size]

    print("\nSizes after trim")
    print("\ntrue scores")
    print(true_relevance)

    print("\npred scores")
    print(scores)

    if (np.sum(scores)==0) and (np.sum(true_relevance) !=0):
        n_score = 0
    elif (np.sum(scores) != 0) and (np.sum(true_relevance) ==0):
        n_score = 0
    elif ((scores.shape[0]==1) and (true_relevance.shape[0]==1)) and (np.sum(scores)>=1) and (np.sum(true_relevance) >=1):
        n_score=1
    else:
        n_score = ndcg_score(np.asarray([true_relevance]), np.asarray([scores]))

    n_score = n_score * sample_weight
    # pair_to_n_score_dict[cur_pair] = n_score
    # all_ndcg_scores.append(n_score)

    if np.sum(true_relevance) > 0:
        print("\nn_score: %.4f"%n_score)

    if i%MOD_NUM == 0:
        print("\nGot query group %d of %d"%(i+1, gbo_list_size))

    return ( cur_pair, n_score)





def get_fancy_ndcg_v2_para(NUM_JOBS ,gdf,sdf ,degree_type, user_type, old_user_set, GRAN, edge_weight_col, query_groupby_cols,weighted=True, child_col="nodeUserID_ndcg" , parent_col="parentUserID_ndcg"):

    

    if weighted == True:
        weight_tag = "weighted"
    else:
        weight_tag = "unweighted"

    #reconfig
    if degree_type == "indegree":
        query_col = parent_col
        result_col = child_col
    else:
        query_col = child_col
        result_col = parent_col

    # #make sure gran is consistent
    # gdf["nodeTime"] = pd.to_datetime(gdf["nodeTime"], utc=True).dt.floor(GRAN)
    # sdf["nodeTime"] = pd.to_datetime(sdf["nodeTime"], utc=True).dt.floor(GRAN)

    print("\nsdf")
    print(sdf)

    print("\ngdf")
    print(gdf)

    gdf = gdf.drop_duplicates()
    sdf = sdf.drop_duplicates()

    # sdf[query_col] = [user if user in old_user_set else "new_user" for user in list(cur_baseline_df[query_col])]

    print("\nsdf")
    print(sdf)

    print("\ngdf")
    print(gdf)

    gdf["%s_is_new"%query_col] = [0 if user in old_user_set else 1 for user in list(gdf[query_col])]
    sdf["%s_is_new"%query_col] = [0 if user in old_user_set else 1 for user in list(sdf[query_col])]

    if user_type == "new":
        gdf = gdf[gdf["%s_is_new"%query_col]==1]
        sdf = sdf[sdf["%s_is_new"%query_col]==1]
    elif user_type == "old":
        gdf = gdf[gdf["%s_is_new"%query_col]==0]
        sdf = sdf[sdf["%s_is_new"%query_col]==0]

    gdf = gdf.rename(columns={query_col:"query_user", result_col : "result", edge_weight_col:"score"})
    sdf = sdf.rename(columns={query_col:"query_user", result_col : "result", edge_weight_col:"score"})

    col_order =query_groupby_cols+ ["result", "score"]

    sdf["neg_score"]=sdf["score"] * -1
    gdf["neg_score"]=gdf["score"] * -1
    # sdf = sdf.sort_values(["nodeTime", "query_user", "neg_score"]).reset_index(drop=True)
    # gdf = gdf.sort_values(["nodeTime", "query_user", "neg_score"]).reset_index(drop=True)

    sdf = sdf.sort_values(query_groupby_cols + ["neg_score"]).reset_index(drop=True)
    gdf = gdf.sort_values(query_groupby_cols + ["neg_score"]).reset_index(drop=True)
    sdf = sdf[col_order]
    gdf = gdf[col_order]

    print("\nsdf")
    print(sdf)

    print("\ngdf")
    print(gdf)

    #combine the dfs
    gdf = gdf.rename(columns={"score":"gt_score" })
    sdf = sdf.rename(columns={"score":"pred_score"})

    gt_users = gdf["query_user"].unique()
    gt_users = set(gt_users)

    # print("\nbefore gt filter")
    # print(sdf)

    # sdf = sdf[sdf["query_user"].isin(gt_users)].reset_index(drop=True)

    # print("\nafter gt filter")
    # print(sdf)

    # sys.exit(0)

    if (sdf.shape[0]==0) and (gdf.shape[0]==0):
        df = pd.DataFrame(data={"query_user":[], "result":[], "gt_score":[], "pred_score":[]})
        return 1, df

    df = pd.merge(gdf, sdf, on=query_groupby_cols + ["result"], how="outer" )

    df["result"] = df["result"].fillna("n/a")
    df = df.fillna(0)
    df = df.drop_duplicates().reset_index(drop=True)

    print("\nmerged df")
    print(df)

    print(df[df["result"]=="n/a"])

    # sys.exit(0)
    #make weights

    if weighted == True:
        df["query_group_weight"]=df.groupby(query_groupby_cols)["gt_score"].transform("sum")
        qw = df[query_groupby_cols+ ["query_group_weight"]].drop_duplicates().reset_index(drop=True)
        print("\nqw")
        print(qw)
        raw_weight_total = qw["query_group_weight"].sum()
        df["query_group_weight"]=df["query_group_weight"]/raw_weight_total

        print()
        print(df["query_group_weight"])
    else:
        df["query_group_weight"]= 1



    # temp = df[(df["pred_score"]>0) & (df["gt_score"]>0)].reset_index(drop=True)
    # print("\ntemp")
    # print(temp)


    df["query_group"] =""
    for i,col in enumerate(query_groupby_cols):

        if i != len(query_groupby_cols) - 1:
            df["query_group"] = df["query_group"]  + df[col]   +"<with>"
        else:
            df["query_group"] = df["query_group"]  + df[col]

    # df["query_nodeTime_pair"] = df["query_user"].astype("str") +"<with>" + df["nodeTime"].astype("str")

    print("\ndf with q group")
    print(df)

    # df = df[~((df["pred_score"]==0) & (df["gt_score"]==0))].reset_index(drop=True)

    temp = df[(df["pred_score"]>0) & (df["gt_score"]>0)].reset_index(drop=True)
    print("\ntemp")
    print(temp)

    # sys.exit(0)

    #make query gbo
    gbo_list = df.groupby(df["query_group"])
    gbo_list = list(gbo_list)
    gbo_list_size = len(gbo_list)
    print("\nSize of query gbo_list")
    print(gbo_list_size)

    #finally get scores!!!

    y_test = []
    y_pred = []
    sample_weights = []

    all_ndcg_scores = []

    pair_to_n_score_dict = {}
    EPSILON=0.0000001

    MOD_NUM = 100



    # #get arg tuples
    # arg_tuples = []
    # for i, g in enumerate(gbo_list):
    #     arg_tuples.append((i,g,gbo_list_size,MOD_NUM))
    # print("\nTrying multiproc...")

    # #train model
    # pool = multiprocessing.Pool(processes = NUM_JOBS)
    # p = Process(target = get_pair_ndcg_tuple, args=(arg_tuple,))
    # jobs.append(p)
    # p.start()
    # p.join()

    #=========== multi proc #===========
    pair_nscore_tuples = []
    print("\nRunning MP...")
    pool = mp.Pool(processes= NUM_JOBS)
    arg_tuples = []
    for i, g in enumerate(gbo_list):
        arg_tuples.append((i,g,gbo_list_size,MOD_NUM))

    print("\nLaunching parallel func...")
    pair_nscore_tuples = pool.map(get_pair_ndcg_tuple, arg_tuples)
    pool.close()
    #===========#===========#===========

    #make nscore dict

    pair_to_n_score_dict = dict(pair_nscore_tuples)
    df["NDCG_Score"] = df["query_group"].map(pair_to_n_score_dict)
    print("\ndf")
    print(df)

    temp = df[["query_group", "NDCG_Score"]].drop_duplicates().reset_index(drop=True)

    all_ndcg_scores = temp["NDCG_Score"].values
    print("\nnum all_ndcg_scores: %d"%len(all_ndcg_scores))

    if weighted== True:
        avg_score = np.sum(all_ndcg_scores)
    else:
        avg_score = np.mean(all_ndcg_scores)

    print("\nAVG SCORE: %.4f"%avg_score)

    # print("\n%s %s %s %s score: %.4f"%(infoID,user_type, degree_type, weight_tag, avg_score))

    # user_type_df_list.append(user_type)
    # dt_df_list.append(degree_type)
    # weight_df_list.append(weight_tag)
    # ndcg_df_list.append(avg_score)
    # infoID_df_list.append(infoID)

    # query_output_fp = query_table_dir + hyp_infoID + ".csv"
    # df.to_csv(query_output_fp, index=False)
    # print(query_output_fp)

    return avg_score, df


def get_fancy_ndcg_v3_para_temporal(NUM_JOBS ,gdf,sdf ,degree_type, user_type,  GRAN, edge_weight_col, weighted=True, child_col="nodeUserID_ndcg" , parent_col="parentUserID_ndcg"):

    

    if weighted == True:
        weight_tag = "weighted"
    else:
        weight_tag = "unweighted"

    #reconfig
    if degree_type == "indegree":
        query_user_col = parent_col
        result_col = child_col
    else:
        query_user_col = child_col
        result_col = parent_col

    query_groupby_cols=["nodeTime","query_user"]

    # #make sure gran is consistent
    # gdf["nodeTime"] = pd.to_datetime(gdf["nodeTime"], utc=True).dt.floor(GRAN)
    # sdf["nodeTime"] = pd.to_datetime(sdf["nodeTime"], utc=True).dt.floor(GRAN)

    print("\nsdf")
    print(sdf)

    print("\ngdf")
    print(gdf)

    gdf = gdf.drop_duplicates()
    sdf = sdf.drop_duplicates()

    # sdf[query_user_col] = [user if user in old_user_set else "new_user" for user in list(cur_baseline_df[query_user_col])]

    print("\nsdf")
    print(sdf)

    print("\ngdf")
    print(gdf)

    # gdf["%s_is_new"%query_user_col] = [0 if user in old_user_set else 1 for user in list(gdf[query_user_col])]
    # sdf["%s_is_new"%query_user_col] = [0 if user in old_user_set else 1 for user in list(sdf[query_user_col])]

    # if user_type == "new":
    #     gdf = gdf[gdf["%s_is_new"%query_user_col]==1]
    #     sdf = sdf[sdf["%s_is_new"%query_user_col]==1]
    # elif user_type == "old":
    #     gdf = gdf[gdf["%s_is_new"%query_user_col]==0]
    #     sdf = sdf[sdf["%s_is_new"%query_user_col]==0]

    if user_type == "new":
        gdf = gdf[gdf[query_user_col]=="new_user" ]
        sdf = sdf[sdf[query_user_col]=="new_user" ]
    elif user_type == "old":
        gdf = gdf[gdf[query_user_col]!="new_user" ]
        sdf = sdf[sdf[query_user_col]!="new_user" ]

    gdf = gdf.rename(columns={query_user_col:"query_user", result_col : "result", edge_weight_col:"score"})
    sdf = sdf.rename(columns={query_user_col:"query_user", result_col : "result", edge_weight_col:"score"})

    col_order =["nodeTime","query_user" ,"result", "score"]

    sdf["neg_score"]=sdf["score"] * -1
    gdf["neg_score"]=gdf["score"] * -1
    # sdf = sdf.sort_values(["nodeTime", "query_user", "neg_score"]).reset_index(drop=True)
    # gdf = gdf.sort_values(["nodeTime", "query_user", "neg_score"]).reset_index(drop=True)

    sdf = sdf.sort_values(["nodeTime","query_user", "neg_score"]).reset_index(drop=True)
    gdf = gdf.sort_values(["nodeTime","query_user", "neg_score"]).reset_index(drop=True)
    sdf = sdf[col_order]
    gdf = gdf[col_order]

    print("\nsdf")
    print(sdf)

    print("\ngdf")
    print(gdf)

    #combine the dfs
    gdf = gdf.rename(columns={"score":"gt_score" })
    sdf = sdf.rename(columns={"score":"pred_score"})

    gt_users = gdf["query_user"].unique()
    gt_users = set(gt_users)

    # print("\nbefore gt filter")
    # print(sdf)

    # sdf = sdf[sdf["query_user"].isin(gt_users)].reset_index(drop=True)

    # print("\nafter gt filter")
    # print(sdf)

    # sys.exit(0)

    if (sdf.shape[0]==0) and (gdf.shape[0]==0):
        df = pd.DataFrame(data={"query_user":[], "result":[], "gt_score":[], "pred_score":[]})
        return 1, df

    df = pd.merge(gdf, sdf, on=["nodeTime","query_user" ,"result"], how="outer" )

    df["result"] = df["result"].fillna("n/a")
    df = df.fillna(0)
    df = df.drop_duplicates().reset_index(drop=True)

    print("\nmerged df")
    print(df)

    print(df[df["result"]=="n/a"])

    # sys.exit(0)
    #make weights

    if weighted == True:
        df["query_group_weight"]=df.groupby(query_groupby_cols)["gt_score"].transform("sum")
        qw = df[query_groupby_cols+ ["query_group_weight"]].drop_duplicates().reset_index(drop=True)
        print("\nqw")
        print(qw)
        raw_weight_total = qw["query_group_weight"].sum()
        df["query_group_weight"]=df["query_group_weight"]/raw_weight_total

        print()
        print(df["query_group_weight"])
    else:
        df["query_group_weight"]= 1



    # temp = df[(df["pred_score"]>0) & (df["gt_score"]>0)].reset_index(drop=True)
    # print("\ntemp")
    # print(temp)


    df["query_group"] =""
    for i,col in enumerate(query_groupby_cols):

        if i != len(query_groupby_cols) - 1:
            df["query_group"] = df["query_group"]  + df[col].astype(str)   +"<with>"
        else:
            df["query_group"] = df["query_group"]  + df[col].astype(str)

    # df["query_nodeTime_pair"] = df["query_user"].astype("str") +"<with>" + df["nodeTime"].astype("str")

    print("\ndf with q group")
    print(df)

    # df = df[~((df["pred_score"]==0) & (df["gt_score"]==0))].reset_index(drop=True)

    #just if you wanna see it
    temp = df[(df["pred_score"]>0) & (df["gt_score"]>0)].reset_index(drop=True)
    print("\ntemp")
    print(temp)

    # sys.exit(0)

    #make query gbo
    gbo_list = df.groupby(df["query_group"])
    gbo_list = list(gbo_list)
    gbo_list_size = len(gbo_list)
    print("\nSize of query gbo_list")
    print(gbo_list_size)

    #finally get scores!!!

    y_test = []
    y_pred = []
    sample_weights = []

    all_ndcg_scores = []

    pair_to_n_score_dict = {}
    EPSILON=0.0000001

    MOD_NUM = 100



    # #get arg tuples
    # arg_tuples = []
    # for i, g in enumerate(gbo_list):
    #     arg_tuples.append((i,g,gbo_list_size,MOD_NUM))
    # print("\nTrying multiproc...")

    # #train model
    # pool = multiprocessing.Pool(processes = NUM_JOBS)
    # p = Process(target = get_pair_ndcg_tuple, args=(arg_tuple,))
    # jobs.append(p)
    # p.start()
    # p.join()

    #=========== multi proc #===========
    pair_nscore_tuples = []
    print("\nRunning MP...")
    pool = mp.Pool(processes= NUM_JOBS)
    arg_tuples = []
    for i, g in enumerate(gbo_list):
        arg_tuples.append((i,g,gbo_list_size,MOD_NUM))

    print("\nLaunching parallel func...")
    pair_nscore_tuples = pool.map(get_pair_ndcg_tuple, arg_tuples)
    pool.close()
    #===========#===========#===========

    #make nscore dict

    pair_to_n_score_dict = dict(pair_nscore_tuples)
    df["NDCG_Score"] = df["query_group"].map(pair_to_n_score_dict)
    print("\ndf")
    print(df)

    temp = df[["query_group", "NDCG_Score"]].drop_duplicates().reset_index(drop=True)

    all_ndcg_scores = temp["NDCG_Score"].values
    print("\nnum all_ndcg_scores: %d"%len(all_ndcg_scores))

    if weighted== True:
        avg_score = np.sum(all_ndcg_scores)
    else:
        avg_score = np.mean(all_ndcg_scores)

    print("\nAVG SCORE: %.4f"%avg_score)

    # print("\n%s %s %s %s score: %.4f"%(infoID,user_type, degree_type, weight_tag, avg_score))

    # user_type_df_list.append(user_type)
    # dt_df_list.append(degree_type)
    # weight_df_list.append(weight_tag)
    # ndcg_df_list.append(avg_score)
    # infoID_df_list.append(infoID)

    # query_output_fp = query_table_dir + hyp_infoID + ".csv"
    # df.to_csv(query_output_fp, index=False)
    # print(query_output_fp)

    return avg_score, df










