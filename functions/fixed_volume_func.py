import sys
# sys.path.append("/data/Fmubang/cp4-VAM-write-up-exps/functions")
# from cascade_ft_funcs import *
# from nn_funcs_7_27 import *
# sys.path.append("/data/Fmubang/CP5-WEEKEND-REVISE-1-23/p0-functions")
import pandas as pd 
import os,sys
from scipy import stats
import numpy as np
from basic_utils import *

from train_data_funcs import *

def get_exo_df_and_fts(news_fp,reddit_fp,gdelt_fp ):

    reddit_df = pd.read_csv(reddit_fp)
    print(reddit_df)
    gdelt_df = pd.read_csv(gdelt_fp)
    print(gdelt_df)

    news_df = pd.read_csv(news_fp)
    news_df["nodeTime"] = pd.to_datetime(news_df["nodeTime"], utc=True)
    print(news_df)
    news_cols = list(news_df)
    news_cols.remove("nodeTime")
    for col in news_cols:
        news_df = news_df.rename(columns={col : col.replace("informationID_", "news_count_")})
    print(news_df)
    news_cols = list(news_df)
    news_cols.remove("nodeTime")

    print(reddit_df)
    reddit_df["num_reddit_activities"]= reddit_df[["reddit_post", "reddit_comment"]].sum(axis=1)
    reddit_df=reddit_df.drop(["reddit_post", "reddit_comment"],axis=1)
    reddit_df["nodeTime"] = pd.to_datetime(reddit_df["nodeTime"], utc=True)
    reddit_df = reddit_df.sort_values("nodeTime").reset_index(drop=True)

    gdelt_df["nodeTime"] = pd.to_datetime(gdelt_df["nodeTime"], utc=True)

    exo_df = merge_mult_dfs([reddit_df, gdelt_df, news_df], on="nodeTime", how="inner").reset_index(drop=True)

    dynamic_gdelt_fts = [
    "AvgTone",  "GoldsteinScale",  "NumMentions"
    ]

    dynamic_reddit_fts = ["num_reddit_activities"]

    news_dynamic_fts = list(news_cols)



    return exo_df, dynamic_gdelt_fts,dynamic_reddit_fts,news_dynamic_fts

def fixed_volume_ft_grab_v2_with_exo(infoIDs,main_output_dir, start,end,DEBUG,platforms,main_df, GRAN,news_fp,reddit_fp,gdelt_fp ):

    #info id dict
    hyp_dict = hyphenate_infoID_dict(infoIDs)
    print(hyp_dict)

    #outputdir
    create_output_dir(main_output_dir)

    #dates
    dates = pd.date_range(start, end, freq=GRAN)
    date_df = pd.DataFrame(data={"nodeTime":dates})
    date_df["nodeTime"] = pd.to_datetime(date_df["nodeTime"], utc=True)

    #get relevant ids
    main_df = main_df[main_df["informationID"].isin(infoIDs)].reset_index(drop=True)

    #get parents
    print("\nGetting parents...")
    main_df = get_parentUserID_col_v2_pnnl_version(main_df)

    #fuse users
    main_df["nodeUserID"] = main_df["nodeUserID"] +"_" + main_df["informationID"] + "_"  + main_df["platform"]
    main_df["parentUserID"] = main_df["parentUserID"] +"_" + main_df["informationID"]+ "_" + main_df["platform"]

    #config dates
    main_df = config_df_by_dates(main_df,start,end,"nodeTime")
    main_df = main_df[["nodeTime", "nodeUserID", "parentUserID", "platform", "informationID"]]
    main_df["nodeTime"] = main_df["nodeTime"].dt.floor(GRAN)
    print(main_df)

    #GET INFO ID 1HOT
    infoID_1hot_dict = get_1hot_vectors(infoIDs)


    #global counts to verify later
    global_total_new_users_over_time = 0
    global_total_old_users_over_time = 0
    global_total_actions = 0

    infoID_to_ft_df_dict = {}
    for infoID in infoIDs:
        df = main_df[main_df["informationID"]==infoID]

        #mark new users
        df["nodeTime"] = pd.to_datetime(df["nodeTime"], utc=True)
        df["user_bdate"] = df.groupby(["nodeUserID"])["nodeTime"].transform("min")
        df["is_new_user"] = [1 if user_bdate==nodeTime else 0 for user_bdate,nodeTime in zip(df["user_bdate"], df["nodeTime"])]

        #ft df loop
        all_dfs = [date_df]
        for platform in platforms:
            temp = df[df["platform"]==platform].reset_index(drop=True)
            gt_total_actions = temp.shape[0]

            #make verify df
            verify_df = temp.copy()
            verify_df["num_unique_users"] = verify_df.groupby(["nodeTime"])["nodeUserID"].transform("nunique")
            verify_df = verify_df[["nodeTime","num_unique_users"]].drop_duplicates()
            print(verify_df)
            gt_num_unique_users_over_time = verify_df["num_unique_users"].sum()

            #actions
            action_col = "%s_platform_infoID_pair_num_actions"%platform
            temp[action_col ] = temp.groupby(["nodeTime"])["platform"].transform("count")
            action_df = temp[["nodeTime", action_col ]].copy().drop_duplicates().reset_index(drop=True)
            my_total_actions = action_df[action_col].sum()
            global_total_actions+=my_total_actions

            #verify
            print("\nChecking action counts...")
            print(gt_total_actions)
            print(my_total_actions)
            if gt_total_actions != my_total_actions:
                print("\nError! gt_total_actions != my_total_actions")
                sys.exit(0)

            all_dfs.append(action_df)

            #old and new
            temp = temp[["nodeTime", "nodeUserID", "is_new_user"]]
            old_user_df = temp[temp["is_new_user"]==0].copy().reset_index(drop=True)
            new_user_df = temp[temp["is_new_user"]==1].copy().reset_index(drop=True)

            print(old_user_df)
            print(new_user_df)

            old_col = "%s_platform_infoID_pair_nunique_old_users"%platform
            new_col = "%s_platform_infoID_pair_nunique_new_users"%platform
            old_user_df[old_col] = old_user_df.groupby(["nodeTime"])["nodeUserID"].transform("nunique").reset_index(drop=True)
            new_user_df[new_col] = new_user_df.groupby(["nodeTime"])["nodeUserID"].transform("nunique").reset_index(drop=True)
            print(old_user_df)
            print(new_user_df)
            old_user_df = old_user_df[["nodeTime", old_col]].drop_duplicates()
            new_user_df = new_user_df[["nodeTime", new_col]].drop_duplicates()
            all_dfs.append(old_user_df)
            all_dfs.append(new_user_df)

            print(old_user_df)
            print(new_user_df)

            #verify
            my_num_unique_users_over_time = old_user_df[old_col].sum() + new_user_df[new_col].sum()
            print("\nChecking counts...")
            print( my_num_unique_users_over_time)
            print(gt_num_unique_users_over_time)
            if  my_num_unique_users_over_time != gt_num_unique_users_over_time:
                print("\nError! my_num_unique_users_over_time != gt_num_unique_users_over_time")
                sys.exit(0)
            else:
                print("\nCounts are ok!")

            #verify
            global_total_new_users_over_time+=new_user_df[new_col].sum()
            global_total_old_users_over_time+=old_user_df[old_col].sum()

        #combine
        cur_ft_df = merge_mult_dfs(all_dfs, on="nodeTime", how="outer").fillna(0)
        print("\ncur_ft_df")
        print(cur_ft_df)

        #1 hot fts
        for infoID_1hot in infoIDs:
            if infoID == infoID_1hot:
                cur_ft_df[infoID] = 1
            else:
                cur_ft_df[infoID_1hot] = 0

        print("\ncur_ft_df")
        print(cur_ft_df)

        infoID_to_ft_df_dict[infoID] = cur_ft_df


    #add global fts
    global_df = date_df.copy()
    global_df["num_global_twitter_actions"]=0
    global_df["num_global_new_twitter_users"]=0
    global_df["num_global_old_twitter_users"]=0
    global_df["num_global_youtube_actions"]=0
    global_df["num_global_new_youtube_users"]=0
    global_df["num_global_old_youtube_users"]=0

    for infoID in infoIDs:
        cur_df = infoID_to_ft_df_dict[infoID]
        global_df["num_global_twitter_actions"] = global_df["num_global_twitter_actions"] + cur_df["twitter_platform_infoID_pair_num_actions"]
        global_df["num_global_youtube_actions"] = global_df["num_global_youtube_actions"] + cur_df["youtube_platform_infoID_pair_num_actions"]

        global_df["num_global_new_twitter_users"] = global_df["num_global_new_twitter_users"] + cur_df["twitter_platform_infoID_pair_nunique_new_users"]
        global_df["num_global_new_youtube_users"] = global_df["num_global_new_youtube_users"] + cur_df["youtube_platform_infoID_pair_nunique_new_users"]

        global_df["num_global_old_twitter_users"] = global_df["num_global_old_twitter_users"] + cur_df["twitter_platform_infoID_pair_nunique_old_users"]
        global_df["num_global_old_youtube_users"] = global_df["num_global_old_youtube_users"] + cur_df["youtube_platform_infoID_pair_nunique_old_users"]



    my_global_new_users_over_time = global_df["num_global_new_youtube_users"].sum() + global_df["num_global_new_twitter_users"].sum()
    my_global_old_users_over_time = global_df["num_global_old_youtube_users"].sum() + global_df["num_global_old_twitter_users"].sum()
    my_global_total_actions = global_df["num_global_twitter_actions"].sum() + global_df["num_global_youtube_actions"].sum()

    print(my_global_new_users_over_time)
    print(global_total_new_users_over_time)
    if my_global_new_users_over_time != global_total_new_users_over_time:
        print("\nError! my_global_new_users_over_time != global_total_new_users_over_time")
        sys.exit(0)

    print(my_global_old_users_over_time)
    print(global_total_old_users_over_time)
    if my_global_old_users_over_time != global_total_old_users_over_time:
        print("\nError! my_global_old_users_over_time != global_total_old_users_over_time")
        sys.exit(0)

    print(my_global_total_actions)
    print(global_total_actions)
    if my_global_total_actions != global_total_actions:
        print("\nError! my_global_total_actions != global_total_actions")
        sys.exit(0)

    #exo data
    exo_df, dynamic_gdelt_fts,dynamic_reddit_fts,news_dynamic_fts = get_exo_df_and_fts(news_fp,reddit_fp,gdelt_fp )

     #add them back in 
    ft_dir = main_output_dir + "features/"
    create_output_dir(ft_dir)
    for infoID in infoIDs:
        cur_df = infoID_to_ft_df_dict[infoID]
        cur_df = cur_df.merge(global_df, on="nodeTime", how="outer")
        cur_df = cur_df.merge(exo_df, on="nodeTime", how="outer")
        cur_df = cur_df.sort_values("nodeTime").reset_index(drop=True)
        print(cur_df)

        #date check
        print(len(dates))
        print(cur_df.shape[0])
        if len(dates) != cur_df.shape[0]:
            print("\nError! len(dates) != cur_df.shape[0]")
            sys.exit(0)

        hyp_infoID = hyp_dict[infoID]

        output_fp = ft_dir + hyp_infoID + "-features.csv"
        cur_df.to_csv(output_fp)
        print(output_fp)

    #save fts
    twitter_dynamic_pair_fts = [
    "twitter_platform_infoID_pair_nunique_new_users",
    "twitter_platform_infoID_pair_nunique_old_users",
    "twitter_platform_infoID_pair_num_actions"
    ]

    youtube_dynamic_pair_fts = [
    "youtube_platform_infoID_pair_nunique_new_users",
    "youtube_platform_infoID_pair_nunique_old_users",
    "youtube_platform_infoID_pair_num_actions"
    ]

    twitter_dynamic_global_fts=[
    "num_global_twitter_actions",
    "num_global_new_twitter_users",
    "num_global_old_twitter_users"
    ]

    youtube_dynamic_global_fts=[
    "num_global_youtube_actions",
    "num_global_new_youtube_users",
    "num_global_old_youtube_users"
    ]

    static_fts = list(infoIDs)
    all_fts = static_fts+ twitter_dynamic_pair_fts + youtube_dynamic_pair_fts + twitter_dynamic_global_fts + youtube_dynamic_global_fts +dynamic_gdelt_fts+dynamic_reddit_fts+news_dynamic_fts
    all_fts=list(all_fts)
    all_ft_lists_dict = {}
    all_ft_lists_dict["twitter_dynamic_pair_fts"]=twitter_dynamic_pair_fts
    all_ft_lists_dict["youtube_dynamic_pair_fts"]=youtube_dynamic_pair_fts
    all_ft_lists_dict["twitter_dynamic_global_fts"]=twitter_dynamic_global_fts
    all_ft_lists_dict["youtube_dynamic_global_fts"]=youtube_dynamic_global_fts
    all_ft_lists_dict["all_fts"]=all_fts
    all_ft_lists_dict["static_fts"]=static_fts

    all_ft_lists_dict["dynamic_gdelt_fts"]=dynamic_gdelt_fts
    all_ft_lists_dict["dynamic_reddit_fts"]=dynamic_reddit_fts
    all_ft_lists_dict["news_dynamic_fts"] = news_dynamic_fts

    target_fts=list(twitter_dynamic_pair_fts + youtube_dynamic_pair_fts)
    all_ft_lists_dict["target_fts"]=target_fts

    cur_output_dir = main_output_dir + "feature-lists/"
    create_output_dir(cur_output_dir)
    for ft_tag,ft_list in all_ft_lists_dict.items():
        print()
        print(ft_tag)
        output_fp = cur_output_dir + ft_tag + ".txt"
        print(output_fp)
        with open(output_fp, "w") as f:
            for ft in ft_list:
                f.write(ft + "\n")
                print(ft)



def fixed_volume_ft_grab(infoIDs,main_output_dir, start,end,DEBUG,platforms,main_df, GRAN):

    #info id dict
    hyp_dict = hyphenate_infoID_dict(infoIDs)
    print(hyp_dict)

    #outputdir
    create_output_dir(main_output_dir)

    #dates
    dates = pd.date_range(start, end, freq=GRAN)
    date_df = pd.DataFrame(data={"nodeTime":dates})
    date_df["nodeTime"] = pd.to_datetime(date_df["nodeTime"], utc=True)

    #get relevant ids
    main_df = main_df[main_df["informationID"].isin(infoIDs)].reset_index(drop=True)

    #get parents
    print("\nGetting parents...")
    main_df = get_parentUserID_col_v2_pnnl_version(main_df)

    #fuse users
    main_df["nodeUserID"] = main_df["nodeUserID"] +"_" + main_df["informationID"] + "_"  + main_df["platform"]
    main_df["parentUserID"] = main_df["parentUserID"] +"_" + main_df["informationID"]+ "_" + main_df["platform"]

    #config dates
    main_df = config_df_by_dates(main_df,start,end,"nodeTime")
    main_df = main_df[["nodeTime", "nodeUserID", "parentUserID", "platform", "informationID"]]
    main_df["nodeTime"] = main_df["nodeTime"].dt.floor(GRAN)
    print(main_df)

    #GET INFO ID 1HOT
    infoID_1hot_dict = get_1hot_vectors(infoIDs)


    #global counts to verify later
    global_total_new_users_over_time = 0
    global_total_old_users_over_time = 0
    global_total_actions = 0

    infoID_to_ft_df_dict = {}
    for infoID in infoIDs:
        df = main_df[main_df["informationID"]==infoID]

        #mark new users
        df["nodeTime"] = pd.to_datetime(df["nodeTime"], utc=True)
        df["user_bdate"] = df.groupby(["nodeUserID"])["nodeTime"].transform("min")
        df["is_new_user"] = [1 if user_bdate==nodeTime else 0 for user_bdate,nodeTime in zip(df["user_bdate"], df["nodeTime"])]

        #ft df loop
        all_dfs = [date_df]
        for platform in platforms:
            temp = df[df["platform"]==platform].reset_index(drop=True)
            gt_total_actions = temp.shape[0]

            #make verify df
            verify_df = temp.copy()
            verify_df["num_unique_users"] = verify_df.groupby(["nodeTime"])["nodeUserID"].transform("nunique")
            verify_df = verify_df[["nodeTime","num_unique_users"]].drop_duplicates()
            print(verify_df)
            gt_num_unique_users_over_time = verify_df["num_unique_users"].sum()

            #actions
            action_col = "%s_platform_infoID_pair_num_actions"%platform
            temp[action_col ] = temp.groupby(["nodeTime"])["platform"].transform("count")
            action_df = temp[["nodeTime", action_col ]].copy().drop_duplicates().reset_index(drop=True)
            my_total_actions = action_df[action_col].sum()
            global_total_actions+=my_total_actions

            #verify
            print("\nChecking action counts...")
            print(gt_total_actions)
            print(my_total_actions)
            if gt_total_actions != my_total_actions:
                print("\nError! gt_total_actions != my_total_actions")
                sys.exit(0)

            all_dfs.append(action_df)

            #old and new
            temp = temp[["nodeTime", "nodeUserID", "is_new_user"]]
            old_user_df = temp[temp["is_new_user"]==0].copy().reset_index(drop=True)
            new_user_df = temp[temp["is_new_user"]==1].copy().reset_index(drop=True)

            print(old_user_df)
            print(new_user_df)

            old_col = "%s_platform_infoID_pair_nunique_old_users"%platform
            new_col = "%s_platform_infoID_pair_nunique_new_users"%platform
            old_user_df[old_col] = old_user_df.groupby(["nodeTime"])["nodeUserID"].transform("nunique").reset_index(drop=True)
            new_user_df[new_col] = new_user_df.groupby(["nodeTime"])["nodeUserID"].transform("nunique").reset_index(drop=True)
            print(old_user_df)
            print(new_user_df)
            old_user_df = old_user_df[["nodeTime", old_col]].drop_duplicates()
            new_user_df = new_user_df[["nodeTime", new_col]].drop_duplicates()
            all_dfs.append(old_user_df)
            all_dfs.append(new_user_df)

            print(old_user_df)
            print(new_user_df)

            #verify
            my_num_unique_users_over_time = old_user_df[old_col].sum() + new_user_df[new_col].sum()
            print("\nChecking counts...")
            print( my_num_unique_users_over_time)
            print(gt_num_unique_users_over_time)
            if  my_num_unique_users_over_time != gt_num_unique_users_over_time:
                print("\nError! my_num_unique_users_over_time != gt_num_unique_users_over_time")
                sys.exit(0)
            else:
                print("\nCounts are ok!")

            #verify
            global_total_new_users_over_time+=new_user_df[new_col].sum()
            global_total_old_users_over_time+=old_user_df[old_col].sum()

        #combine
        cur_ft_df = merge_mult_dfs(all_dfs, on="nodeTime", how="outer").fillna(0)
        print("\ncur_ft_df")
        print(cur_ft_df)

        #1 hot fts
        for infoID_1hot in infoIDs:
            if infoID == infoID_1hot:
                cur_ft_df[infoID] = 1
            else:
                cur_ft_df[infoID_1hot] = 0

        print("\ncur_ft_df")
        print(cur_ft_df)

        infoID_to_ft_df_dict[infoID] = cur_ft_df


    #add global fts
    global_df = date_df.copy()
    global_df["num_global_twitter_actions"]=0
    global_df["num_global_new_twitter_users"]=0
    global_df["num_global_old_twitter_users"]=0
    global_df["num_global_youtube_actions"]=0
    global_df["num_global_new_youtube_users"]=0
    global_df["num_global_old_youtube_users"]=0

    for infoID in infoIDs:
        cur_df = infoID_to_ft_df_dict[infoID]
        global_df["num_global_twitter_actions"] = global_df["num_global_twitter_actions"] + cur_df["twitter_platform_infoID_pair_num_actions"]
        global_df["num_global_youtube_actions"] = global_df["num_global_youtube_actions"] + cur_df["youtube_platform_infoID_pair_num_actions"]

        global_df["num_global_new_twitter_users"] = global_df["num_global_new_twitter_users"] + cur_df["twitter_platform_infoID_pair_nunique_new_users"]
        global_df["num_global_new_youtube_users"] = global_df["num_global_new_youtube_users"] + cur_df["youtube_platform_infoID_pair_nunique_new_users"]

        global_df["num_global_old_twitter_users"] = global_df["num_global_old_twitter_users"] + cur_df["twitter_platform_infoID_pair_nunique_old_users"]
        global_df["num_global_old_youtube_users"] = global_df["num_global_old_youtube_users"] + cur_df["youtube_platform_infoID_pair_nunique_old_users"]



    my_global_new_users_over_time = global_df["num_global_new_youtube_users"].sum() + global_df["num_global_new_twitter_users"].sum()
    my_global_old_users_over_time = global_df["num_global_old_youtube_users"].sum() + global_df["num_global_old_twitter_users"].sum()
    my_global_total_actions = global_df["num_global_twitter_actions"].sum() + global_df["num_global_youtube_actions"].sum()

    print(my_global_new_users_over_time)
    print(global_total_new_users_over_time)
    if my_global_new_users_over_time != global_total_new_users_over_time:
        print("\nError! my_global_new_users_over_time != global_total_new_users_over_time")
        sys.exit(0)

    print(my_global_old_users_over_time)
    print(global_total_old_users_over_time)
    if my_global_old_users_over_time != global_total_old_users_over_time:
        print("\nError! my_global_old_users_over_time != global_total_old_users_over_time")
        sys.exit(0)

    print(my_global_total_actions)
    print(global_total_actions)
    if my_global_total_actions != global_total_actions:
        print("\nError! my_global_total_actions != global_total_actions")
        sys.exit(0)

     #add them back in 
    ft_dir = main_output_dir + "features/"
    create_output_dir(ft_dir)
    for infoID in infoIDs:
        cur_df = infoID_to_ft_df_dict[infoID]
        cur_df = cur_df.merge(global_df, on="nodeTime", how="outer")
        cur_df = cur_df.sort_values("nodeTime").reset_index(drop=True)
        print(cur_df)

        #date check
        print(len(dates))
        print(cur_df.shape[0])
        if len(dates) != cur_df.shape[0]:
            print("\nError! len(dates) != cur_df.shape[0]")
            sys.exit(0)

        hyp_infoID = hyp_dict[infoID]

        output_fp = ft_dir + hyp_infoID + "-features.csv"
        cur_df.to_csv(output_fp)
        print(output_fp)

    #save fts
    twitter_dynamic_pair_fts = [
    "twitter_platform_infoID_pair_nunique_new_users",
    "twitter_platform_infoID_pair_nunique_old_users",
    "twitter_platform_infoID_pair_num_actions"
    ]

    youtube_dynamic_pair_fts = [
    "youtube_platform_infoID_pair_nunique_new_users",
    "youtube_platform_infoID_pair_nunique_old_users",
    "youtube_platform_infoID_pair_num_actions"
    ]

    twitter_dynamic_global_fts=[
    "num_global_twitter_actions",
    "num_global_new_twitter_users",
    "num_global_old_twitter_users"
    ]

    youtube_dynamic_global_fts=[
    "num_global_youtube_actions",
    "num_global_new_youtube_users",
    "num_global_old_youtube_users"
    ]

    static_fts = list(infoIDs)
    all_fts = static_fts+ twitter_dynamic_pair_fts + youtube_dynamic_pair_fts + twitter_dynamic_global_fts + youtube_dynamic_global_fts
    all_fts=list(all_fts)
    all_ft_lists_dict = {}
    all_ft_lists_dict["twitter_dynamic_pair_fts"]=twitter_dynamic_pair_fts
    all_ft_lists_dict["youtube_dynamic_pair_fts"]=youtube_dynamic_pair_fts
    all_ft_lists_dict["twitter_dynamic_global_fts"]=twitter_dynamic_global_fts
    all_ft_lists_dict["youtube_dynamic_global_fts"]=youtube_dynamic_global_fts
    all_ft_lists_dict["all_fts"]=all_fts
    all_ft_lists_dict["static_fts"]=static_fts
    target_fts=list(twitter_dynamic_pair_fts + youtube_dynamic_pair_fts)
    all_ft_lists_dict["target_fts"]=target_fts

    cur_output_dir = main_output_dir + "feature-lists/"
    create_output_dir(cur_output_dir)
    for ft_tag,ft_list in all_ft_lists_dict.items():
        print()
        print(ft_tag)
        output_fp = cur_output_dir + ft_tag + ".txt"
        print(output_fp)
        with open(output_fp, "w") as f:
            for ft in ft_list:
                f.write(ft + "\n")
                print(ft)








