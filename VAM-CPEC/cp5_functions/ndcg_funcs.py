import os,sys
import pandas as pd 
import numpy as np 
from functools import reduce
# import sample_gen_utils as sgu
from sklearn.metrics import ndcg_score
# from network_metric_funcs import *
from multiprocessing import Process
import multiprocessing
import multiprocessing as mp
from time import time
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score
import json
import calendar

from basic_utils import create_output_dir, hyphenate_infoID_dict, convert_df_2_cols_to_dict

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