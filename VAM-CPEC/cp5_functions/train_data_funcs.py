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

def cp5_get_fts_p2(GRAN, start, end, main_input_dir, main_output_dir, platforms, infoIDs):
	create_output_dir(main_output_dir)
 
	# #set gran
	# GRAN="D"
	# start = "08-13-2009"
	# end = "07-06-2020 23:59:59"

	# #main data
	# main_data_dir = "/data/Fmubang/CP5-VAM-CHALLENGE-V2-DAILY/data/CP5-Dry-Run2-Data/DROPNA-True-DEBUG-False/"

	#make main out
	# main_data_output_dir = "/data/Fmubang/CP5-VAM-CHALLENGE-V2-DAILY/data/CP5-Dry-Run2-Data/DROPNA-True-DEBUG-False/"
	# main_output_dir =main_data_dir+ "P2-REORGANIZED-TRAIN-DATA/GRAN-%s-dates-%s-to-%s/"%(GRAN, start, end)
	# create_output_dir(main_output_dir)

	# #main in
	# # main_input_dir = "/data/Fmubang/CP5-VAM/data/cp5-updated-12-27/V3-WITH-OTHER-12-27-UPDATE-CP5-DATA-debug-False-RETWEET_REMOVE-False/08-13-2009-to-06-29-2020 23:59:59/f4-M2-User-Fts/GRAN-%s-Dates-08-13-2009-to-06-29-2020 23:59:59-DEBUG-False/"%GRAN
	# # print(main_input_dir)

	# main_input_dir = main_data_dir + "P1-PREPROCESSED-DATA-DEBUG-False/%s-to-%s/f4-M2-User-Fts/GRAN-%s-Dates-%s-to-%s-DEBUG-False/"%(start, end, GRAN, start, end)
	# print(main_input_dir)
	#for combining fts
	desired_target_cols = ["nunique_users", "num_actions"]

	#other items
	# infoIDs = sgu.cp5_infoIDs_challenge_with_other()
	hyp_dict = hyphenate_infoID_dict(infoIDs)
	# platforms = ["twitter", "youtube"]
	user_age_status_list = ["old", "new"]

	# static_fts = list(infoIDs) + ["is_twitter"]
	static_fts = list(infoIDs) 

	# 
	pre_global_actions=0
	pre_global_old_users=0
	pre_global_new_users=0

	#get all you need first, then add fts together
	platform_infoID_pair_to_df_dict = {}
	for platform in platforms:
		platform_infoID_pair_to_df_dict[platform]={}
		for infoID in infoIDs:
			hyp_infoID = hyp_dict[infoID]

			old_ft_fp = main_input_dir + platform + "/old/" + hyp_infoID + "/child-user-data.csv" 
			new_ft_fp = main_input_dir + platform + "/new/" + hyp_infoID + "/child-user-data.csv"

			old_df = pd.read_csv(old_ft_fp)
			new_df = pd.read_csv(new_ft_fp)

			#get counts to verify in a bit
			pre_num_old_users = old_df["nunique_users"].sum()
			pre_num_new_users = new_df["nunique_users"].sum()
			pre_total_num_activities = old_df["num_actions"].sum() + new_df["num_actions"].sum()
			pre_global_actions+=pre_total_num_activities
			pre_global_old_users+=pre_num_old_users
			pre_global_new_users+=pre_num_new_users

			drop_cols = ["is_child", "is_new"]
			old_df=old_df.rename(columns={"nunique_users": platform + "_" + "platform_infoID_pair_nunique_old_users", "num_actions":platform + "_" +"platform_infoID_pair_num_old_user_actions"})
			new_df=new_df.rename(columns={"nunique_users":platform + "_" +"platform_infoID_pair_nunique_new_users", "num_actions":platform + "_" +"platform_infoID_pair_num_new_user_actions"})
			old_df = old_df.drop(drop_cols,axis=1)
			new_df = new_df.drop(drop_cols,axis=1)

			print(old_df)
			print(new_df)

			merge_cols = ["nodeTime"] + static_fts

			df = pd.merge(old_df,new_df, on=merge_cols, how="inner")
			df["nodeTime"] = pd.to_datetime(df["nodeTime"],utc=True)
			df = df.sort_values("nodeTime").reset_index(drop=True)
			df[platform + "_" + "platform_infoID_pair_num_actions"]=df[[platform + "_" +"platform_infoID_pair_num_old_user_actions",platform + "_" +"platform_infoID_pair_num_new_user_actions"]].sum(axis=1)
			print("\ncombined df")
			# print(df)



			post_num_old_users = df[platform + "_" + "platform_infoID_pair_nunique_old_users"].sum()
			post_num_new_users = df[platform + "_" + "platform_infoID_pair_nunique_new_users"].sum()
			post_total_num_activities =df[platform + "_" + "platform_infoID_pair_num_actions"].sum()
			
			print(post_num_old_users)
			print(pre_num_old_users)
			if post_num_old_users != pre_num_old_users:
				print("post_num_old_users != pre_num_old_users")
				sys.exit(0)
			else:
				print("counts are ok!")

			print(post_num_new_users)
			print(pre_num_new_users)
			if post_num_new_users != pre_num_new_users:
				print("post_num_new_users != pre_num_new_users")
				sys.exit(0)
			else:
				print("counts are ok!")

			print(post_total_num_activities)
			print(pre_total_num_activities)
			if post_total_num_activities != pre_total_num_activities:
				print("post_total_num_activities != pre_total_num_activities")
				sys.exit(0)
			else:
				print("counts are ok!")

			platform_infoID_pair_to_df_dict[platform][infoID] = df

	#get global fts
	global_df=pd.DataFrame(data={"nodeTime":df["nodeTime"].copy()})
	global_df["num_global_twitter_actions"]=[0 for i in range(global_df.shape[0])]
	global_df["num_global_youtube_actions"]=[0 for i in range(global_df.shape[0])]
	# global_df["num_global_new_user_twitter_actions"]=[0 for i in range(global_df.shape[0])]
	# global_df["num_global_old_user_twitter_actions"]=[0 for i in range(global_df.shape[0])]
	# global_df["num_global_new_user_youtube_actions"]=[0 for i in range(global_df.shape[0])]
	# global_df["num_global_old_user_youtube_actions"]=[0 for i in range(global_df.shape[0])]

	global_df["num_global_new_twitter_users"]=[0 for i in range(global_df.shape[0])]
	global_df["num_global_old_twitter_users"]=[0 for i in range(global_df.shape[0])]
	global_df["num_global_new_youtube_users"]=[0 for i in range(global_df.shape[0])]
	global_df["num_global_old_youtube_users"]=[0 for i in range(global_df.shape[0])]

	global_dynamic_fts = list(global_df)
	global_dynamic_fts.remove("nodeTime")

	#do actions first
	for platform in platforms:
		for infoID in infoIDs:
			df = platform_infoID_pair_to_df_dict[platform][infoID]
			global_df["num_global_%s_actions"%platform] = global_df["num_global_%s_actions"%platform] + df[platform + "_" +"platform_infoID_pair_num_actions"]
	print(global_df)

	#get user fts
	for platform in platforms:
		for infoID in infoIDs:
			df = platform_infoID_pair_to_df_dict[platform][infoID]
			for user_status in user_age_status_list:
				global_df["num_global_%s_%s_users"%(user_status, platform)] = global_df["num_global_%s_%s_users"%(user_status, platform)] + df[platform + "_" + "platform_infoID_pair_nunique_%s_users"%user_status]
	print(global_df)

	post_global_actions=global_df["num_global_twitter_actions"].sum() + global_df["num_global_youtube_actions"].sum()
	post_global_old_users=global_df["num_global_old_youtube_users"].sum() + global_df["num_global_old_twitter_users"].sum()
	post_global_new_users=global_df["num_global_new_youtube_users"].sum() + global_df["num_global_new_twitter_users"].sum()

	print(post_global_actions)
	print(pre_global_actions)
	if post_global_actions != pre_global_actions:
		print("post_global_actions != pre_global_actions")
		sys.exit(0)
	else:
		print("counts are ok!")

	print(post_global_old_users)
	print(pre_global_old_users)
	if post_global_old_users != pre_global_old_users:
		print("post_global_old_users != pre_global_old_users")
		sys.exit(0)
	else:
		print("counts are ok!")

	print(post_global_new_users)
	print(pre_global_new_users)
	if post_global_new_users != pre_global_new_users:
		print("post_global_new_users != pre_global_new_users")
		sys.exit(0)
	else:
		print("counts are ok!")

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

	platform_fts_dict = {}

	for platform in platforms:
		for infoID in infoIDs:
			df = platform_infoID_pair_to_df_dict[platform][infoID]
			df = pd.merge(df, global_df, on="nodeTime", how="inner").reset_index(drop=True)
			print(df)
			platform_infoID_pair_to_df_dict[platform][infoID]=df

			# hyp_infoID = hyp_dict[infoID]
			# cur_output_dir = main_output_dir + platform + "/" 
			# create_output_dir(cur_output_dir)
			# output_fp = cur_output_dir + "%s-features.csv"%hyp_infoID
			# df.to_csv(output_fp, index=False)
			# print(output_fp)


	for platform in platforms:
		if platform == "twitter":
			opp_platform="youtube"
		if platform == "youtube":
			opp_platform="twitter"

		for infoID in infoIDs:
			df = platform_infoID_pair_to_df_dict[platform][infoID]
			opp_df = platform_infoID_pair_to_df_dict[opp_platform][infoID]
			opp_cols_to_get = all_ft_lists_dict["%s_dynamic_pair_fts"%opp_platform] 
			opp_cols_to_get=list(opp_cols_to_get)
			temp_opp_df = opp_df[["nodeTime"]+opp_cols_to_get]
			df = pd.merge(temp_opp_df,df,on="nodeTime",how="inner")
			platform_infoID_pair_to_df_dict[platform][infoID]=df
			print(df)

			cols = list(df)
			for col in cols:
				print(col)
			# sys.exit(0)

	#just choose 1
	for infoID in infoIDs:
		df = platform_infoID_pair_to_df_dict["twitter"][infoID]
		# df = pd.merge(df, global_df, on="nodeTime", how="inner").reset_index(drop=True)
		# print(df)
		# platform_infoID_pair_to_df_dict[platform][infoID]=df

		hyp_infoID = hyp_dict[infoID]
		cur_output_dir = main_output_dir + "/features/"
		create_output_dir(cur_output_dir)
		output_fp = cur_output_dir + "%s-features.csv"%hyp_infoID
		df.to_csv(output_fp, index=False)
		print(output_fp)





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

	return


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


def cp5_volume_fts_part1( infoIDs,main_output_dir, start,end,DEBUG,platforms,main_df, GRAN):

    main_output_dir = main_output_dir + "f4-M2-User-Fts/"
    create_output_dir(main_output_dir)

    #get infoIDs
    # infoIDs = get_46_cp4_infoIDs()
    print(infoIDs)

    #kick out 1 infoID
    main_df = main_df[main_df["informationID"].isin(infoIDs)].reset_index(drop=True)

    #get parents
    print("\nGetting parents...")
    main_df = get_parentUserID_col_v2_pnnl_version(main_df)

    #fix user names
    # main_df["nodeUserID"] = main_df["nodeUserID"] +"_" + main_df["informationID"] + "_" + main_df["actionType"] + "_" + main_df["platform"]
    # main_df["parentUserID"] = main_df["parentUserID"] +"_" + main_df["informationID"]+ "_" + main_df["actionType"]+ "_" + main_df["platform"]

    main_df["nodeUserID"] = main_df["nodeUserID"] +"_" + main_df["informationID"] + "_"  + main_df["platform"]
    main_df["parentUserID"] = main_df["parentUserID"] +"_" + main_df["informationID"]+ "_" + main_df["platform"]

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


    for platform in platforms:  
        df = main_df.copy()
        df["nodeTime"] = df["nodeTime"].dt.floor(GRAN)
        df = df[df["platform"]==platform]

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

                cur_dir = main_output_dir + "GRAN-%s-Dates-%s-to-%s-DEBUG-%s/"%(GRAN, start,end,DEBUG)
                output_dir =cur_dir+ "%s/%s/%s/"%(platform,user_status,hyp_infoID)
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

    return cur_dir