# import sys
# sys.path.append("/data/Fmubang/cp4-VAM-write-up-exps/functions")
# from cascade_ft_funcs import *
# from nn_funcs_7_27 import *
# import pandas as pd
# import os,sys
# from scipy import stats
# import numpy as np
# from vam_config_funcs_v2 import *
# from time import time
# import networkx as nx
# from scipy.stats import wasserstein_distance
# import seaborn as sns
import sys
# sys.path.append("/data/Fmubang/CP4-VAM-ORGANIZED-V2/functions")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from time import time
import networkx as nx
from basic_utils import create_output_dir, hyphenate_infoID_dict,merge_mult_dfs,convert_df_2_cols_to_dict
from infoIDs18 import get_cp4_challenge_18_infoIDs
import pandas as pd 
import numpy as np
import seaborn as sns
from bisect import bisect
from scipy.stats import wasserstein_distance
from scipy import stats
# from basic_utils import merge_mult_dfs, create_output_dir

def rh_distance(ground_truth,simulation):
    """
    Metric: rh_distance
    Description: Relative Hausdorff distance between the ground
    truth and simulation data.
    Meant for degree sequence measurements
    Inputs:
    ground_truth - ground truth in degree distrib (iterable)
    simulation - simulation in degree distrib (iterable)
    Citation:
    Aksoy, S.G., K.E. Nowak, S.J. Young,
    "A linear time algorithm and analysis of graph Relative Hausdorff
    distance", SIAM Journal on Mathematics of Data Science 1,
    no. 4 (2019): 647-666.
    """

    #Rename for notional convience, convert to lists for speed
    G=list(ground_truth)
    F=list(simulation)

    #If no edges, RH computed on closest nonzero degree sequence of [1]
    try:
        if max(G)==0:
            G=[1]
    except ValueError:
        G=[1]

    try:
        if max(F)==0:
            F=[1]
    except ValueError:
        F=[1]

    #convert degree sequence to CCDH sequence
    G.sort()
    F.sort()
    G=[len(G) - bisect(G, z) for z in range(0,max(G))]
    F=[len(F) - bisect(F, z) for z in range(0,max(F))]

    #Convert CCDH sequence to dictionary of CCDH values
    #keyed by degrees 1 to max degree.
    mDegF = len(F)
    degF = {i : F[i-1] for i in range(1, mDegF+1)}
    mDegG = len(G)
    degG = {i : G[i-1] for i in range(1, mDegG+1)}

    # Main subroutine: Return max of epsilon, delta, where delta is
    # min size box around (d,Fd) containing smooth path through G.
    def epsilon_box(d,Fd,G,mDegG,epsilon):
        right = d*(1+epsilon)
        r = int(np.floor(right))
        r_frac = right - r
        rightF = Fd*(1+epsilon)

        left = d*(1-epsilon)
        k = int(np.ceil(left))
        k_frac = k - left
        leftF = Fd*(1-epsilon)

        if ((leftF <= mDegG)
            and (rightF >= G.get(r,0)*(1-r_frac) + G.get(r+1,0)*r_frac)
            and (leftF <= G.get(k,G[1])*(1-k_frac) + G.get(k-1,G[1])*k_frac)):
            return epsilon

        if ((mDegG < leftF) and (leftF <= mDegG+1)
            and (rightF >= G.get(r,0)*(1-r_frac) + G.get(r+1,0)*r_frac)
            and (leftF <= G[mDegG]*k_frac)):
            return epsilon

        # If here, G doesn't pass through current epsilon box,
        # must find necessary epsilon box size. If G pass below box
        if G.get(d,0) < Fd:
            while k > mDegG + 1 or G.get(k-1,G[1]) < Fd*float(k-1)/d:
                k -= 1
            if k == mDegG + 1:
                return 1 - float(G[mDegG]*(mDegG+1))/(Fd + G[mDegG]*d)
            else:
                return 1 - float((1-k)*G.get(k,G[1])
                        + k*G.get(k-1,G[1]))/(Fd + d*(G.get(k-1,G[1])
                        - G.get(k,G[1])))
        # otherwise, G pass above box
        else:
            while G.get(r+1,0) > Fd*float(r+1)/d:
                r += 1
            return float((1+r)*G.get(r,0)
                        - r*G.get(r+1,0))/(d*(G.get(r,0)
                        - G.get(r+1,0)) + Fd) - 1

    #Compute RH distance
    checkF = set([1,mDegF] + [j for j in range(2,mDegF)
                if not ((degF[j] == degF[j-1]) and (degF[j] == degF[j+1]))])
    checkG = set([1,mDegG] + [j for j in range(2,mDegG)
                if not ((degG[j] == degG[j-1]) and (degG[j] == degG[j+1]))])
    epsilon = 0.0
    for d in range(1,max(mDegG,mDegF)+1):
        if d in checkF:
            epsilon_prime = epsilon_box(d,degF[d],degG,mDegG,epsilon)
            if epsilon_prime > epsilon:
                epsilon = epsilon_prime
        if d in checkG:
            epsilon_prime = epsilon_box(d,degG[d],degF,mDegF,epsilon)
            if epsilon_prime > epsilon:
                epsilon = epsilon_prime
    return epsilon


def get_emd_metrics(infoIDs, model_tag_input_dir,hyp_dict,TIMESTEP_DEBUG,gt_and_baseline_input_dir,platform, model_tag,model_output_dir):

    keep_cols = ["timestep", "nodeUserID", "parentUserID"]

    for infoID in infoIDs:
        print("\nEvaluating %s"%infoID)

        hyp_infoID = hyp_dict[infoID]


        cur_model_fp = model_tag_input_dir + "Simulations/" +  hyp_infoID + ".csv"

        try:
            cur_model_df = pd.read_csv(cur_model_fp)
        except FileNotFoundError:
            continue

        print(cur_model_df)
        # sys.exit(0)
        cur_model_df=cur_model_df[keep_cols]
        print("\ncur_model_df")
        print(cur_model_df)

        cur_model_df["edge_weight_this_timestep"] = cur_model_df.groupby(["nodeUserID", "parentUserID", "timestep"])["nodeUserID"].transform("count")
        cur_model_df = cur_model_df.drop_duplicates()
        cur_model_df=cur_model_df.sort_values(["timestep" ,"edge_weight_this_timestep"],ascending=False).reset_index(drop=True)
        print("\ncur_model_df")
        print(cur_model_df)

        #============ convert these to graphs ==============
        num_timesteps = cur_model_df["timestep"].nunique()
        if TIMESTEP_DEBUG == True:
            num_timesteps = 1

        all_timestep_emds_for_infoID = []
        for timestep in range(1, num_timesteps+1):


            #===================== baseline graph so far =====================
            temp_model_df = cur_model_df[cur_model_df["timestep"]==timestep]
            MODEL_EDGE_LIST = zip(temp_model_df["nodeUserID"],temp_model_df["parentUserID"],temp_model_df["edge_weight_this_timestep"])
            MODEL_GRAPH =nx.DiGraph()
            MODEL_GRAPH.add_weighted_edges_from(MODEL_EDGE_LIST)

            # MODEL_GRAPH = nx.from_pandas_edgelist(cur_model_df,"nodeUserID", "parentUserID",edge_attr="edge_weight_this_timestep")
            print("\nMODEL_GRAPH")
            print(MODEL_GRAPH)
            model_edges = list(MODEL_GRAPH.edges())
            # for i in range(10):
            #   print(model_edges[i])
            num_baseline_nodes = MODEL_GRAPH.number_of_nodes()
            num_model_edges = MODEL_GRAPH.number_of_edges()
            print("\nnum_baseline_nodes: %d"%num_baseline_nodes)
            print("num_model_edges: %d"%num_model_edges)
            print("is weighted:")
            print(nx.is_weighted(MODEL_GRAPH))

            #===================== pr =====================
            print("\nGetting pr dicts...")
            if platform == "youtube":
                MODEL_PAGE_RANK_DICT = nx.pagerank_numpy(MODEL_GRAPH, weight="weight")
            else:
                MODEL_PAGE_RANK_DICT = nx.pagerank(MODEL_GRAPH, weight="weight")

            print("Got page ranks!")

            #===================== sort them =====================

            MODEL_PAGE_RANK_DF = pd.DataFrame(list(MODEL_PAGE_RANK_DICT.items()), columns=["node", "page_rank"])
            if MODEL_PAGE_RANK_DF.shape[0] == 0:
                MODEL_PAGE_RANK_DF = pd.DataFrame(data={"node":["n/a"], "page_rank":[0]})
            MODEL_PAGE_RANK_DF = MODEL_PAGE_RANK_DF.sort_values("page_rank", ascending=False).reset_index(drop=True)
            print("\nMODEL_PAGE_RANK_DF")
            print(MODEL_PAGE_RANK_DF)

            model_pr_dist = MODEL_PAGE_RANK_DF["page_rank"].values

            cur_gt_fp = gt_and_baseline_input_dir+"Ground-Truth-Results/" + platform + "/Results-by-Timestep/Timestep-" + str(timestep) + "/" + hyp_infoID + ".csv"
            gt_pr_dist = pd.read_csv(cur_gt_fp)["page_rank"].values

            print("\nGetting EMD...")
            EMD = wasserstein_distance(gt_pr_dist,model_pr_dist)
            print(EMD)
            all_timestep_emds_for_infoID.append(EMD)

            #make cur save dirs
            model_timestep_output_dir = model_tag_input_dir + model_tag + "/" +platform + "/Results-by-Timestep/Timestep-%d/"%(timestep)

            print(model_tag_input_dir)

            # sys.exit(0)
            create_output_dir(model_timestep_output_dir)

            model_timestep_output_fp = model_timestep_output_dir + hyp_infoID + ".csv"
            MODEL_PAGE_RANK_DF.to_csv(model_timestep_output_fp, index=False)
            print(model_timestep_output_fp)

        cur_infoID_emd_results = pd.DataFrame(data={"infoID":[infoID for i in range(num_timesteps)], "EMD":all_timestep_emds_for_infoID, "timestep":[(i+1) for i in range(num_timesteps)]})
        cur_infoID_emd_results = cur_infoID_emd_results[["timestep", "infoID", "EMD"]]
        print("\ncur_infoID_emd_results")
        print(cur_infoID_emd_results)

        #main_emd_results_output_dir = model_tag_input_dir + model_tag+ "/"+ platform + "/"
        main_emd_results_output_dir = model_tag_input_dir

        output_dir = main_emd_results_output_dir + "EMD-Results/"
        create_output_dir(output_dir)
        output_fp = output_dir + hyp_infoID + ".csv"
        cur_infoID_emd_results.to_csv(output_fp, index=False)
        print(output_fp)

        print(cur_infoID_emd_results)


    return main_emd_results_output_dir,model_tag_input_dir




def compare_model_emd_results_to_baseline_backup(baseline_results_input_dir, model_tag, main_emd_results_input_dir,  platform, infoIDs,hyp_dict):

    main_output_dir = main_emd_results_input_dir

    model_tag_to_dir_dict = {}
    model_tag_to_dir_dict["Persistence_Baseline"]= baseline_results_input_dir + platform + "/EMD-Results/"
    model_tag_to_dir_dict[model_tag] = main_emd_results_input_dir + "/EMD-Results/"
    model_tags = ["Persistence_Baseline",model_tag]

    #get results
    all_result_dfs = []
    for infoID in infoIDs:
        hyp_infoID = hyp_dict[infoID]
        avg_emd_list = []
        emd_model_list = []
        for model_tag in model_tags:
            cur_emd_fp = model_tag_to_dir_dict[model_tag] + hyp_infoID + ".csv"
            cur_emd_df = pd.read_csv(cur_emd_fp)
            print(cur_emd_df)

            # try:
            #     cur_emd_df = pd.read_csv(cur_emd_fp)

            # except FileNotFoundError:
            #     continue
            avg_emd = cur_emd_df["EMD"].mean()
            avg_emd_list.append(avg_emd)
            emd_model_list.append(model_tag)

        all_models_emd_df = pd.DataFrame(data={"model_tag":emd_model_list, "avg_emd":avg_emd_list})
        all_models_emd_df["infoID"]=infoID
        all_models_emd_df = all_models_emd_df[["model_tag", "avg_emd","infoID"]]
        all_models_emd_df = all_models_emd_df.sort_values("avg_emd", ascending=True).reset_index(drop=True)
        print("\nall_models_emd_df")
        print(all_models_emd_df)

        # sys.exit(0)
        all_result_dfs.append(all_models_emd_df)

        # cur_output_dir = main_output_dir + platform + "/Results-by-InfoID/"
        cur_output_dir = main_output_dir + "/Results-by-InfoID/"
        create_output_dir(cur_output_dir)
        output_fp = cur_output_dir + hyp_infoID + ".csv"
        all_models_emd_df.to_csv(output_fp)
        print(output_fp)

    final_emd_df = pd.concat(all_result_dfs).reset_index(drop=True)
    print("\nfinal_emd_df")
    print(final_emd_df)

    # sys.exit(0)

    baseline_emd_df = final_emd_df[final_emd_df["model_tag"]=="Persistence_Baseline"].reset_index(drop=True)
    baseline_emd_df = baseline_emd_df.rename(columns={"avg_emd":"Persistence_Baseline_avg_emd"})



    final_emd_df_no_baseline_df = final_emd_df[final_emd_df["model_tag"] != "Persistence_Baseline"].reset_index(drop=True)
    final_emd_df_no_baseline_df=final_emd_df_no_baseline_df.drop_duplicates(["infoID"]).reset_index(drop=True)
    final_emd_df_no_baseline_df=final_emd_df_no_baseline_df.rename(columns={"avg_emd":"best_non_pb_model_avg_emd"})

    final_emd_df = final_emd_df.drop_duplicates(["infoID"]).reset_index(drop=True)
    final_emd_df=final_emd_df.rename(columns={"avg_emd":"winner_avg_emd", "model_tag":"winner_model_tag"})

    print("\nbaseline_emd_df")
    print(baseline_emd_df)
    baseline_emd_df=baseline_emd_df.drop("model_tag", axis=1)

    print("\nfinal_emd_df")
    print(final_emd_df)

    print("\nfinal_emd_df_no_baseline_df")
    final_emd_df_no_baseline_df=final_emd_df_no_baseline_df.rename(columns={"model_tag":"best_non_pb_model_tag"})
    print(final_emd_df_no_baseline_df)
    # final_emd_df_no_baseline_df=final_emd_df_no_baseline_df.drop("model_tag", axis=1)

    merge_dfs = [baseline_emd_df,final_emd_df,final_emd_df_no_baseline_df ]
    final_emd_df = merge_mult_dfs(merge_dfs, on="infoID", how="inner")
    final_emd_df = final_emd_df.sort_values("winner_avg_emd", ascending=True).reset_index(drop=True)


    #fix bug
    non_ties_result_df = final_emd_df[final_emd_df["Persistence_Baseline_avg_emd"]!=final_emd_df["best_non_pb_model_avg_emd"]].reset_index(drop=True)
    ties_result_df = final_emd_df[final_emd_df["Persistence_Baseline_avg_emd"]==final_emd_df["best_non_pb_model_avg_emd"]].reset_index(drop=True)
    if ties_result_df.shape[0] != 0:
        ties_result_df["winner_model_tag"] = "tie"
        final_emd_df = pd.concat([non_ties_result_df,ties_result_df]).reset_index(drop=True)

    print("\nfinal_emd_df")
    print(final_emd_df)

    output_fp = main_output_dir +"/Final-Results.csv"
    final_emd_df.to_csv(output_fp)
    print(output_fp)

    return final_emd_df

def eval_model_rh(infoIDs, model_tag_input_dir,hyp_dict,TIMESTEP_DEBUG,gt_and_baseline_input_dir,platform, model_tag,model_output_dir):

    keep_cols = ["timestep", "nodeUserID", "parentUserID"]

    for infoID in infoIDs:
        print("\nEvaluating %s"%infoID)

        hyp_infoID = hyp_dict[infoID]


        cur_model_fp = model_tag_input_dir + "Simulations/" +  hyp_infoID + ".csv"
        cur_model_df = pd.read_csv(cur_model_fp)
        cur_model_df=cur_model_df[keep_cols]
        print("\ncur_model_df")
        print(cur_model_df)

        # cur_model_df["edge_weight_this_timestep"] = cur_model_df.groupby(["nodeUserID", "parentUserID", "timestep"])["nodeUserID"].transform("count")
        # cur_model_df = cur_model_df.drop_duplicates()
        # cur_model_df=cur_model_df.sort_values(["timestep" ,"edge_weight_this_timestep"],ascending=False).reset_index(drop=True)
        # print("\ncur_model_df")
        # print(cur_model_df)

        #============ convert these to graphs ==============
        num_timesteps = cur_model_df["timestep"].nunique()
        if TIMESTEP_DEBUG == True:
            num_timesteps = 1

        all_timestep_rhds_for_infoID = []
        for timestep in range(1, num_timesteps+1):

            #===================== baseline graph so far =====================
            temp_model_df = cur_model_df[cur_model_df["timestep"]==timestep]
            temp_model_df=temp_model_df[["nodeUserID", "parentUserID"]].drop_duplicates().reset_index(drop=True)
            temp_model_df["parentUserID_in_degree"] = temp_model_df.groupby(["parentUserID"])["nodeUserID"].transform("count")
            user_to_degree_dict = convert_df_2_cols_to_dict(temp_model_df, "parentUserID", "parentUserID_in_degree")
            all_users = list(temp_model_df["parentUserID"].unique()) + list(temp_model_df["nodeUserID"].unique())
            all_users =list(set(all_users))
            temp_model_df = pd.DataFrame(data={"nodeUserID":all_users})
            temp_model_df["nodeUserID_in_degree"] = temp_model_df["nodeUserID"].map(user_to_degree_dict)
            temp_model_df["nodeUserID_in_degree"] =temp_model_df["nodeUserID_in_degree"].fillna(0)
            temp_model_df=temp_model_df.sort_values("nodeUserID_in_degree", ascending=False).reset_index(drop=True)

            cur_gt_fp = gt_and_baseline_input_dir+"Ground-Truth-Results/" + platform + "/In-Degrees-by-Timestep/Timestep-" + str(timestep) + "/" + hyp_infoID + ".csv"
            temp_gt_df = pd.read_csv(cur_gt_fp)

            print("\nGetting RH...")
            cur_rh = rh_distance(temp_gt_df["nodeUserID_in_degree"].astype("int32"),temp_model_df["nodeUserID_in_degree"].astype("int32"))
            all_timestep_rhds_for_infoID.append(cur_rh)

            #make cur save dirs
            model_timestep_output_dir =  model_tag_input_dir + model_tag + "/" +platform + "/In-Degrees-by-Timestep/Timestep-%d/"%(timestep)
            create_output_dir(model_timestep_output_dir)

            model_timestep_output_fp = model_timestep_output_dir + hyp_infoID + ".csv"
            temp_model_df.to_csv(model_timestep_output_fp, index=False)
            print(model_timestep_output_fp)

        cur_infoID_rh_results = pd.DataFrame(data={"infoID":[infoID for i in range(num_timesteps)], "RH":all_timestep_rhds_for_infoID, "timestep":[(i+1) for i in range(num_timesteps)]})
        cur_infoID_rh_results = cur_infoID_rh_results[["timestep", "infoID", "RH"]]
        print("\ncur_infoID_rh_results")
        print(cur_infoID_rh_results)

        #main_rh_output_dir = model_tag_input_dir + model_tag+ "/"+ platform + "/"
        main_rh_output_dir = model_tag_input_dir
        output_dir = main_rh_output_dir + "RH-Results/"
        create_output_dir(output_dir)
        output_fp = output_dir + hyp_infoID + ".csv"
        cur_infoID_rh_results.to_csv(output_fp, index=False)
        print(output_fp)

    return main_rh_output_dir,model_tag_input_dir

def compare_model_emd_results_to_baseline(baseline_results_input_dir, model_tag, main_rh_results_input_dir,  platform, infoIDs,hyp_dict):

    main_output_dir = main_rh_results_input_dir
    simplified_main_model_tag = model_tag

    model_tag_to_dir_dict = {}
    model_tag_to_dir_dict["Persistence_Baseline"]= baseline_results_input_dir + platform + "/RH-Results/"
    model_tag_to_dir_dict[model_tag] = main_rh_results_input_dir + "/RH-Results/"
    model_tags = ["Persistence_Baseline",model_tag]

    #get results
    all_result_dfs = []
    for infoID in infoIDs:
        hyp_infoID = hyp_dict[infoID]
        avg_rh_list = []
        rh_model_list = []
        for model_tag in model_tags:
            cur_rh_fp = model_tag_to_dir_dict[model_tag] + hyp_infoID + ".csv"

            try:
                cur_rh_df = pd.read_csv(cur_rh_fp)

            except FileNotFoundError:
                continue
            avg_rh = cur_rh_df["RH"].mean()
            avg_rh_list.append(avg_rh)
            rh_model_list.append(model_tag)

        all_models_rh_df = pd.DataFrame(data={"model_tag":rh_model_list, "avg_rh":avg_rh_list})
        all_models_rh_df["infoID"]=infoID
        all_models_rh_df = all_models_rh_df[["model_tag", "avg_rh","infoID"]]
        all_models_rh_df = all_models_rh_df.sort_values("avg_rh", ascending=True).reset_index(drop=True)
        print("\nall_models_rh_df")
        print(all_models_rh_df)
        all_result_dfs.append(all_models_rh_df)

        #cur_output_dir = main_output_dir + platform + "/Results-by-InfoID/"
        cur_output_dir = main_output_dir + "/Results-by-InfoID/"
        create_output_dir(cur_output_dir)
        output_fp = cur_output_dir + hyp_infoID + ".csv"
        all_models_rh_df.to_csv(output_fp)
        print(output_fp)

    final_rh_df = pd.concat(all_result_dfs).reset_index(drop=True)
    print("\nfinal_rh_df")
    print(final_rh_df)

    baseline_rh_df = final_rh_df[final_rh_df["model_tag"]=="Persistence_Baseline"].reset_index(drop=True)
    baseline_rh_df = baseline_rh_df.rename(columns={"avg_rh":"Persistence_Baseline_avg_rh"})



    final_rh_df_no_baseline_df = final_rh_df[final_rh_df["model_tag"] != "Persistence_Baseline"].reset_index(drop=True)
    final_rh_df_no_baseline_df=final_rh_df_no_baseline_df.drop_duplicates(["infoID"]).reset_index(drop=True)
    final_rh_df_no_baseline_df=final_rh_df_no_baseline_df.rename(columns={"avg_rh":"best_non_pb_model_avg_rh"})

    final_rh_df = final_rh_df.drop_duplicates(["infoID"]).reset_index(drop=True)
    final_rh_df=final_rh_df.rename(columns={"avg_rh":"winner_avg_rh", "model_tag":"winner_model_tag"})

    print("\nbaseline_rh_df")
    print(baseline_rh_df)
    baseline_rh_df=baseline_rh_df.drop("model_tag", axis=1)

    print("\nfinal_rh_df")
    print(final_rh_df)

    print("\nfinal_rh_df_no_baseline_df")
    final_rh_df_no_baseline_df=final_rh_df_no_baseline_df.rename(columns={"model_tag":"best_non_pb_model_tag"})
    print(final_rh_df_no_baseline_df)
    # final_rh_df_no_baseline_df=final_rh_df_no_baseline_df.drop("model_tag", axis=1)

    merge_dfs = [baseline_rh_df,final_rh_df,final_rh_df_no_baseline_df ]
    final_rh_df = merge_mult_dfs(merge_dfs, on="infoID", how="inner")
    final_rh_df = final_rh_df.sort_values("winner_avg_rh", ascending=True).reset_index(drop=True)


    #fix bug
    non_ties_result_df = final_rh_df[final_rh_df["Persistence_Baseline_avg_rh"]!=final_rh_df["best_non_pb_model_avg_rh"]].reset_index(drop=True)
    ties_result_df = final_rh_df[final_rh_df["Persistence_Baseline_avg_rh"]==final_rh_df["best_non_pb_model_avg_rh"]].reset_index(drop=True)
    if ties_result_df.shape[0] != 0:
        ties_result_df["winner_model_tag"] = "tie"
        final_rh_df = pd.concat([non_ties_result_df,ties_result_df]).reset_index(drop=True)

    # final_rh_df["winner_model_tag"] = final_rh_df["winner_model_tag"].replace(model_rename_dict)
    final_rh_df=final_rh_df.drop("best_non_pb_model_tag", axis=1)
    #"winner_avg_rh" : simplified_main_model_tag + "_avg_rh",
    final_rh_df = final_rh_df.drop("winner_avg_rh", axis=1)
    final_rh_df=final_rh_df.rename(columns={ "best_non_pb_model_avg_rh":simplified_main_model_tag+ "_avg_rh"})

    model_error_tag = simplified_main_model_tag+ "_avg_rh"
    baseline_error_tag = "Persistence_Baseline_avg_rh"


    # final_rh_df["norm_" + model_error_tag]=final_rh_df[model_error_tag] /final_rh_df[[ model_error_tag,baseline_error_tag]].sum(axis=1)
    # final_rh_df["norm_" +baseline_error_tag]=final_rh_df[baseline_error_tag] /final_rh_df[[ model_error_tag,baseline_error_tag]].sum(axis=1)

    final_rh_df["VAM_percent_diff_from_baseline"] = 100.0 * (final_rh_df[baseline_error_tag] - final_rh_df[model_error_tag])/final_rh_df[baseline_error_tag]

    col_order = ["infoID", model_error_tag,baseline_error_tag,"winner_model_tag","VAM_percent_diff_from_baseline"]

    # col_order = ["infoID", model_error_tag,baseline_error_tag,"winner_model_tag"]
    final_rh_df=final_rh_df[col_order]
    for col in col_order:
        final_rh_df = final_rh_df.rename(columns={col:col.replace("_", " ")})

    print("\nfinal_rh_df")
    print(final_rh_df)

    output_fp = main_output_dir +"/Final-Results.csv"
    final_rh_df.to_csv(output_fp,index=False)
    print(output_fp)

    return final_rh_df


def get_emd_metrics_v3_unweighted_option(infoIDs, model_tag_input_dir,model_tag_output_dir,hyp_dict,TIMESTEP_DEBUG,gt_and_baseline_input_dir,platform, model_tag, WEIGHT):

    keep_cols = ["timestep", "nodeUserID", "parentUserID"]

    for infoID in infoIDs:
        print("\nEvaluating %s"%infoID)

        hyp_infoID = hyp_dict[infoID]


        cur_model_fp = model_tag_input_dir + "Simulations/" +  hyp_infoID + ".csv"
        output_dir = model_tag_output_dir + "EMD-Materials/"
        create_output_dir(output_dir)

        try:
            cur_model_df = pd.read_csv(cur_model_fp)
        except FileNotFoundError:
            continue

        print(cur_model_df)
        # sys.exit(0)
        cur_model_df=cur_model_df[keep_cols]
        print("\ncur_model_df")
        print(cur_model_df)

        cur_model_df["edge_weight_this_timestep"] = cur_model_df.groupby(["nodeUserID", "parentUserID", "timestep"])["nodeUserID"].transform("count")
        cur_model_df = cur_model_df.drop_duplicates()
        cur_model_df=cur_model_df.sort_values(["timestep" ,"edge_weight_this_timestep"],ascending=False).reset_index(drop=True)
        print("\ncur_model_df")
        print(cur_model_df)

        #============ convert these to graphs ==============
        # num_timesteps = cur_model_df["timestep"].nunique()
        # if TIMESTEP_DEBUG == True:
        #     num_timesteps = 1
        timesteps = gt_df["timestep"].unique()
        timesteps = np.sort(timesteps)
        print(timesteps)

        
        if TIMESTEP_DEBUG == True:
            timesteps = timesteps[:1]

        all_timestep_emds_for_infoID = []
        for timestep in timesteps:


            #===================== baseline graph so far =====================
            temp_model_df = cur_model_df[cur_model_df["timestep"]==timestep]
            MODEL_EDGE_LIST = zip(temp_model_df["nodeUserID"],temp_model_df["parentUserID"],temp_model_df["edge_weight_this_timestep"])
            MODEL_GRAPH =nx.DiGraph()
            MODEL_GRAPH.add_weighted_edges_from(MODEL_EDGE_LIST)

            # MODEL_GRAPH = nx.from_pandas_edgelist(cur_model_df,"nodeUserID", "parentUserID",edge_attr="edge_weight_this_timestep")
            print("\nMODEL_GRAPH")
            print(MODEL_GRAPH)
            model_edges = list(MODEL_GRAPH.edges())
            # for i in range(10):
            #   print(model_edges[i])
            num_baseline_nodes = MODEL_GRAPH.number_of_nodes()
            num_model_edges = MODEL_GRAPH.number_of_edges()
            print("\nnum_baseline_nodes: %d"%num_baseline_nodes)
            print("num_model_edges: %d"%num_model_edges)
            print("is weighted:")
            print(nx.is_weighted(MODEL_GRAPH))

            #===================== pr =====================
            print("\nGetting pr dicts...")
            if platform == "youtube":
                MODEL_PAGE_RANK_DICT = nx.pagerank_numpy(MODEL_GRAPH, weight=WEIGHT)
            else:
                MODEL_PAGE_RANK_DICT = nx.pagerank(MODEL_GRAPH, weight=WEIGHT)

            print("Got page ranks!")

            #===================== sort them =====================

            MODEL_PAGE_RANK_DF = pd.DataFrame(list(MODEL_PAGE_RANK_DICT.items()), columns=["node", "page_rank"])
            if MODEL_PAGE_RANK_DF.shape[0] == 0:
                MODEL_PAGE_RANK_DF = pd.DataFrame(data={"node":["n/a"], "page_rank":[0]})
            MODEL_PAGE_RANK_DF = MODEL_PAGE_RANK_DF.sort_values("page_rank", ascending=False).reset_index(drop=True)
            print("\nMODEL_PAGE_RANK_DF")
            print(MODEL_PAGE_RANK_DF)

            model_pr_dist = MODEL_PAGE_RANK_DF["page_rank"].values

            cur_gt_fp = gt_and_baseline_input_dir+"Ground-Truth-Results/" + platform + "/Results-by-Timestep/Timestep-" + str(timestep) + "/" + hyp_infoID + ".csv"

            try:
                gt_pr_dist = pd.read_csv(cur_gt_fp)["page_rank"].values
            except:
                gt_pr_dist = np.asarray([0])

            print("\nGetting EMD...")
            EMD = wasserstein_distance(gt_pr_dist,model_pr_dist)
            print(EMD)
            all_timestep_emds_for_infoID.append(EMD)

            #make cur save dirs
            # model_timestep_output_dir = model_tag_input_dir + model_tag + "/" +platform + "/Results-by-Timestep/Timestep-%d/"%(timestep)

            model_timestep_output_dir = output_dir + "/Page-Ranks-by-Timestep/Timestep-%d/"%(timestep)
            create_output_dir(model_timestep_output_dir)

            model_timestep_output_fp = model_timestep_output_dir + hyp_infoID + ".csv"
            MODEL_PAGE_RANK_DF.to_csv(model_timestep_output_fp, index=False)
            print(model_timestep_output_fp)

        #cur_infoID_emd_results = pd.DataFrame(data={"infoID":[infoID for i in range(num_timesteps)], "EMD":all_timestep_emds_for_infoID, "timestep":[(i+1) for i in range(num_timesteps)]})
        cur_infoID_emd_results = pd.DataFrame(data={"infoID":[infoID for i in range(len(list(timesteps)))], "EMD":all_timestep_emds_for_infoID, "timestep":timesteps})
        cur_infoID_emd_results = cur_infoID_emd_results[["timestep", "infoID", "EMD"]]
        print("\ncur_infoID_emd_results")
        print(cur_infoID_emd_results)

        #main_emd_results_output_dir = model_tag_input_dir + model_tag+ "/"+ platform + "/"
        # main_emd_results_output_dir = model_tag_input_dir

        final_output_dir = output_dir + "EMD-Results/"
        create_output_dir(final_output_dir)
        output_fp = final_output_dir + hyp_infoID + ".csv"
        cur_infoID_emd_results.to_csv(output_fp, index=False)
        print(output_fp)

        print(cur_infoID_emd_results)


    return output_dir



def get_emd_metrics_v2_SIMPLE(infoIDs, model_tag_input_dir,hyp_dict,TIMESTEP_DEBUG,gt_and_baseline_input_dir,platform, model_tag):

    keep_cols = ["timestep", "nodeUserID", "parentUserID"]

    for infoID in infoIDs:
        print("\nEvaluating %s"%infoID)

        hyp_infoID = hyp_dict[infoID]


        cur_model_fp = model_tag_input_dir + "Simulations/" +  hyp_infoID + ".csv"
        output_dir = model_tag_input_dir + "EMD-Materials/"
        create_output_dir(output_dir)

        try:
            cur_model_df = pd.read_csv(cur_model_fp)
        except FileNotFoundError:
            continue

        print(cur_model_df)
        # sys.exit(0)
        cur_model_df=cur_model_df[keep_cols]
        print("\ncur_model_df")
        print(cur_model_df)

        cur_model_df["edge_weight_this_timestep"] = cur_model_df.groupby(["nodeUserID", "parentUserID", "timestep"])["nodeUserID"].transform("count")
        cur_model_df = cur_model_df.drop_duplicates()
        cur_model_df=cur_model_df.sort_values(["timestep" ,"edge_weight_this_timestep"],ascending=False).reset_index(drop=True)
        print("\ncur_model_df")
        print(cur_model_df)

        #============ convert these to graphs ==============
        # num_timesteps = cur_model_df["timestep"].nunique()
        # if TIMESTEP_DEBUG == True:
        #     num_timesteps = 1
        timesteps = gt_df["timestep"].unique()
        timesteps = np.sort(timesteps)
        print(timesteps)

        
        if TIMESTEP_DEBUG == True:
            timesteps = timesteps[:1]

        all_timestep_emds_for_infoID = []
        for timestep in timesteps:


            #===================== baseline graph so far =====================
            temp_model_df = cur_model_df[cur_model_df["timestep"]==timestep]
            MODEL_EDGE_LIST = zip(temp_model_df["nodeUserID"],temp_model_df["parentUserID"],temp_model_df["edge_weight_this_timestep"])
            MODEL_GRAPH =nx.DiGraph()
            MODEL_GRAPH.add_weighted_edges_from(MODEL_EDGE_LIST)

            # MODEL_GRAPH = nx.from_pandas_edgelist(cur_model_df,"nodeUserID", "parentUserID",edge_attr="edge_weight_this_timestep")
            print("\nMODEL_GRAPH")
            print(MODEL_GRAPH)
            model_edges = list(MODEL_GRAPH.edges())
            # for i in range(10):
            #   print(model_edges[i])
            num_baseline_nodes = MODEL_GRAPH.number_of_nodes()
            num_model_edges = MODEL_GRAPH.number_of_edges()
            print("\nnum_baseline_nodes: %d"%num_baseline_nodes)
            print("num_model_edges: %d"%num_model_edges)
            print("is weighted:")
            print(nx.is_weighted(MODEL_GRAPH))

            #===================== pr =====================
            print("\nGetting pr dicts...")
            if platform == "youtube":
                MODEL_PAGE_RANK_DICT = nx.pagerank_numpy(MODEL_GRAPH, weight="weight")
            else:
                MODEL_PAGE_RANK_DICT = nx.pagerank(MODEL_GRAPH, weight="weight")

            print("Got page ranks!")

            #===================== sort them =====================

            MODEL_PAGE_RANK_DF = pd.DataFrame(list(MODEL_PAGE_RANK_DICT.items()), columns=["node", "page_rank"])
            if MODEL_PAGE_RANK_DF.shape[0] == 0:
                MODEL_PAGE_RANK_DF = pd.DataFrame(data={"node":["n/a"], "page_rank":[0]})
            MODEL_PAGE_RANK_DF = MODEL_PAGE_RANK_DF.sort_values("page_rank", ascending=False).reset_index(drop=True)
            print("\nMODEL_PAGE_RANK_DF")
            print(MODEL_PAGE_RANK_DF)

            model_pr_dist = MODEL_PAGE_RANK_DF["page_rank"].values

            cur_gt_fp = gt_and_baseline_input_dir+"Ground-Truth-Results/" + platform + "/Results-by-Timestep/Timestep-" + str(timestep) + "/" + hyp_infoID + ".csv"

            try:
                gt_pr_dist = pd.read_csv(cur_gt_fp)["page_rank"].values
            except:
                gt_pr_dist = np.asarray([0])

            print("\nGetting EMD...")
            EMD = wasserstein_distance(gt_pr_dist,model_pr_dist)
            print(EMD)
            all_timestep_emds_for_infoID.append(EMD)

            #make cur save dirs
            # model_timestep_output_dir = model_tag_input_dir + model_tag + "/" +platform + "/Results-by-Timestep/Timestep-%d/"%(timestep)

            model_timestep_output_dir = output_dir + "/Page-Ranks-by-Timestep/Timestep-%d/"%(timestep)
            create_output_dir(model_timestep_output_dir)

            model_timestep_output_fp = model_timestep_output_dir + hyp_infoID + ".csv"
            MODEL_PAGE_RANK_DF.to_csv(model_timestep_output_fp, index=False)
            print(model_timestep_output_fp)

        #cur_infoID_emd_results = pd.DataFrame(data={"infoID":[infoID for i in range(num_timesteps)], "EMD":all_timestep_emds_for_infoID, "timestep":[(i+1) for i in range(num_timesteps)]})
        cur_infoID_emd_results = pd.DataFrame(data={"infoID":[infoID for i in range(len(list(timesteps)))], "EMD":all_timestep_emds_for_infoID, "timestep":timesteps})
        cur_infoID_emd_results = cur_infoID_emd_results[["timestep", "infoID", "EMD"]]
        print("\ncur_infoID_emd_results")
        print(cur_infoID_emd_results)

        #main_emd_results_output_dir = model_tag_input_dir + model_tag+ "/"+ platform + "/"
        # main_emd_results_output_dir = model_tag_input_dir

        final_output_dir = output_dir + "EMD-Results/"
        create_output_dir(final_output_dir)
        output_fp = final_output_dir + hyp_infoID + ".csv"
        cur_infoID_emd_results.to_csv(output_fp, index=False)
        print(output_fp)

        print(cur_infoID_emd_results)


    return output_dir



def compare_model_emd_results_to_baseline_v3_sep_output_dir(baseline_results_input_dir, model_tag, main_emd_results_input_dir, main_output_dir, platform, infoIDs,hyp_dict):

    # main_output_dir = main_emd_results_input_dir

    model_tag_to_dir_dict = {}
    model_tag_to_dir_dict["Persistence_Baseline"]= baseline_results_input_dir + platform + "/EMD-Results/"
    model_tag_to_dir_dict[model_tag] = main_emd_results_input_dir + "/EMD-Results/"
    model_tags = ["Persistence_Baseline",model_tag]

    #get results
    all_result_dfs = []
    for infoID in infoIDs:
        hyp_infoID = hyp_dict[infoID]
        avg_emd_list = []
        emd_model_list = []
        for model_tag in model_tags:
            cur_emd_fp = model_tag_to_dir_dict[model_tag] + hyp_infoID + ".csv"
            cur_emd_df = pd.read_csv(cur_emd_fp)
            print(cur_emd_df)

            # try:
            #     cur_emd_df = pd.read_csv(cur_emd_fp)

            # except FileNotFoundError:
            #     continue
            avg_emd = cur_emd_df["EMD"].mean()
            avg_emd_list.append(avg_emd)
            emd_model_list.append(model_tag)

        all_models_emd_df = pd.DataFrame(data={"model_tag":emd_model_list, "avg_emd":avg_emd_list})
        all_models_emd_df["infoID"]=infoID
        all_models_emd_df = all_models_emd_df[["model_tag", "avg_emd","infoID"]]
        all_models_emd_df = all_models_emd_df.sort_values("avg_emd", ascending=True).reset_index(drop=True)
        print("\nall_models_emd_df")
        print(all_models_emd_df)

        # sys.exit(0)
        all_result_dfs.append(all_models_emd_df)

        # cur_output_dir = main_output_dir + platform + "/Results-by-InfoID/"
        cur_output_dir = main_output_dir + "/Results-by-InfoID/"
        create_output_dir(cur_output_dir)
        output_fp = cur_output_dir + hyp_infoID + ".csv"
        all_models_emd_df.to_csv(output_fp)
        print(output_fp)

    final_emd_df = pd.concat(all_result_dfs).reset_index(drop=True)
    print("\nfinal_emd_df")
    print(final_emd_df)

    # sys.exit(0)

    baseline_emd_df = final_emd_df[final_emd_df["model_tag"]=="Persistence_Baseline"].reset_index(drop=True)
    baseline_emd_df = baseline_emd_df.rename(columns={"avg_emd":"Persistence_Baseline_avg_emd"})



    final_emd_df_no_baseline_df = final_emd_df[final_emd_df["model_tag"] != "Persistence_Baseline"].reset_index(drop=True)
    final_emd_df_no_baseline_df=final_emd_df_no_baseline_df.drop_duplicates(["infoID"]).reset_index(drop=True)
    final_emd_df_no_baseline_df=final_emd_df_no_baseline_df.rename(columns={"avg_emd":"best_non_pb_model_avg_emd"})

    final_emd_df = final_emd_df.drop_duplicates(["infoID"]).reset_index(drop=True)
    final_emd_df=final_emd_df.rename(columns={"avg_emd":"winner_avg_emd", "model_tag":"winner_model_tag"})

    print("\nbaseline_emd_df")
    print(baseline_emd_df)
    baseline_emd_df=baseline_emd_df.drop("model_tag", axis=1)

    print("\nfinal_emd_df")
    print(final_emd_df)

    print("\nfinal_emd_df_no_baseline_df")
    final_emd_df_no_baseline_df=final_emd_df_no_baseline_df.rename(columns={"model_tag":"best_non_pb_model_tag"})
    print(final_emd_df_no_baseline_df)
    # final_emd_df_no_baseline_df=final_emd_df_no_baseline_df.drop("model_tag", axis=1)

    merge_dfs = [baseline_emd_df,final_emd_df,final_emd_df_no_baseline_df ]
    final_emd_df = merge_mult_dfs(merge_dfs, on="infoID", how="inner")
    final_emd_df = final_emd_df.sort_values("winner_avg_emd", ascending=True).reset_index(drop=True)


    #fix bug
    non_ties_result_df = final_emd_df[final_emd_df["Persistence_Baseline_avg_emd"]!=final_emd_df["best_non_pb_model_avg_emd"]].reset_index(drop=True)
    ties_result_df = final_emd_df[final_emd_df["Persistence_Baseline_avg_emd"]==final_emd_df["best_non_pb_model_avg_emd"]].reset_index(drop=True)
    if ties_result_df.shape[0] != 0:
        ties_result_df["winner_model_tag"] = "tie"
        final_emd_df = pd.concat([non_ties_result_df,ties_result_df]).reset_index(drop=True)

    print("\nfinal_emd_df")
    print(final_emd_df)

    output_fp = main_output_dir +"/Final-Results.csv"
    final_emd_df.to_csv(output_fp, index=False)
    print(output_fp)

    #plot it
    plot_cols = ["infoID","best_non_pb_model_avg_emd", "Persistence_Baseline_avg_emd"]
    plot_df = final_emd_df[plot_cols]
    error_sum_series = plot_df["best_non_pb_model_avg_emd"] + plot_df["Persistence_Baseline_avg_emd"]
    plot_df["best_non_pb_model_avg_emd"] = plot_df["best_non_pb_model_avg_emd"]/error_sum_series
    plot_df["Persistence_Baseline_avg_emd"] = plot_df["Persistence_Baseline_avg_emd"]/error_sum_series
    norm_fp = main_output_dir +"/Normalized-Final-Results.csv"
    plot_df.to_csv(norm_fp, index=False)

    plot_df = plot_df.rename(columns={"best_non_pb_model_avg_emd": "VAM"})
    plot_df = plot_df.rename(columns={"Persistence_Baseline_avg_emd": "Persistence Baseline"})
    plot_df = plot_df.rename(columns={"infoID":"Topic"})
    plot_df = plot_df.set_index("Topic")

    sns.set()
    plot_df.plot(kind="bar")
    fig = plt.gcf()
    fig.set_size_inches(5,5)
    ax = plt.gca()

    ax.set_title("EMD Results")
    ax.tick_params(labelsize=7)
    plt.tight_layout()
    fig.savefig(main_output_dir + "EMD-Bar-plots.png")
    fig.savefig(main_output_dir + "EMD-Bar-plots.svg")
    plt.close()



    return final_emd_df




def compare_model_emd_results_to_baseline_v2_SIMPLE(baseline_results_input_dir, model_tag, main_emd_results_input_dir,  platform, infoIDs,hyp_dict):

    main_output_dir = main_emd_results_input_dir

    model_tag_to_dir_dict = {}
    model_tag_to_dir_dict["Persistence_Baseline"]= baseline_results_input_dir + platform + "/EMD-Results/"
    model_tag_to_dir_dict[model_tag] = main_emd_results_input_dir + "/EMD-Results/"
    model_tags = ["Persistence_Baseline",model_tag]

    #get results
    all_result_dfs = []
    for infoID in infoIDs:
        hyp_infoID = hyp_dict[infoID]
        avg_emd_list = []
        emd_model_list = []
        for model_tag in model_tags:
            cur_emd_fp = model_tag_to_dir_dict[model_tag] + hyp_infoID + ".csv"
            cur_emd_df = pd.read_csv(cur_emd_fp)
            print(cur_emd_df)

            # try:
            #     cur_emd_df = pd.read_csv(cur_emd_fp)

            # except FileNotFoundError:
            #     continue
            avg_emd = cur_emd_df["EMD"].mean()
            avg_emd_list.append(avg_emd)
            emd_model_list.append(model_tag)

        all_models_emd_df = pd.DataFrame(data={"model_tag":emd_model_list, "avg_emd":avg_emd_list})
        all_models_emd_df["infoID"]=infoID
        all_models_emd_df = all_models_emd_df[["model_tag", "avg_emd","infoID"]]
        all_models_emd_df = all_models_emd_df.sort_values("avg_emd", ascending=True).reset_index(drop=True)
        print("\nall_models_emd_df")
        print(all_models_emd_df)

        # sys.exit(0)
        all_result_dfs.append(all_models_emd_df)

        # cur_output_dir = main_output_dir + platform + "/Results-by-InfoID/"
        cur_output_dir = main_output_dir + "/Results-by-InfoID/"
        create_output_dir(cur_output_dir)
        output_fp = cur_output_dir + hyp_infoID + ".csv"
        all_models_emd_df.to_csv(output_fp)
        print(output_fp)

    final_emd_df = pd.concat(all_result_dfs).reset_index(drop=True)
    print("\nfinal_emd_df")
    print(final_emd_df)

    # sys.exit(0)

    baseline_emd_df = final_emd_df[final_emd_df["model_tag"]=="Persistence_Baseline"].reset_index(drop=True)
    baseline_emd_df = baseline_emd_df.rename(columns={"avg_emd":"Persistence_Baseline_avg_emd"})



    final_emd_df_no_baseline_df = final_emd_df[final_emd_df["model_tag"] != "Persistence_Baseline"].reset_index(drop=True)
    final_emd_df_no_baseline_df=final_emd_df_no_baseline_df.drop_duplicates(["infoID"]).reset_index(drop=True)
    final_emd_df_no_baseline_df=final_emd_df_no_baseline_df.rename(columns={"avg_emd":"best_non_pb_model_avg_emd"})

    final_emd_df = final_emd_df.drop_duplicates(["infoID"]).reset_index(drop=True)
    final_emd_df=final_emd_df.rename(columns={"avg_emd":"winner_avg_emd", "model_tag":"winner_model_tag"})

    print("\nbaseline_emd_df")
    print(baseline_emd_df)
    baseline_emd_df=baseline_emd_df.drop("model_tag", axis=1)

    print("\nfinal_emd_df")
    print(final_emd_df)

    print("\nfinal_emd_df_no_baseline_df")
    final_emd_df_no_baseline_df=final_emd_df_no_baseline_df.rename(columns={"model_tag":"best_non_pb_model_tag"})
    print(final_emd_df_no_baseline_df)
    # final_emd_df_no_baseline_df=final_emd_df_no_baseline_df.drop("model_tag", axis=1)

    merge_dfs = [baseline_emd_df,final_emd_df,final_emd_df_no_baseline_df ]
    final_emd_df = merge_mult_dfs(merge_dfs, on="infoID", how="inner")
    final_emd_df = final_emd_df.sort_values("winner_avg_emd", ascending=True).reset_index(drop=True)


    #fix bug
    non_ties_result_df = final_emd_df[final_emd_df["Persistence_Baseline_avg_emd"]!=final_emd_df["best_non_pb_model_avg_emd"]].reset_index(drop=True)
    ties_result_df = final_emd_df[final_emd_df["Persistence_Baseline_avg_emd"]==final_emd_df["best_non_pb_model_avg_emd"]].reset_index(drop=True)
    if ties_result_df.shape[0] != 0:
        ties_result_df["winner_model_tag"] = "tie"
        final_emd_df = pd.concat([non_ties_result_df,ties_result_df]).reset_index(drop=True)

    print("\nfinal_emd_df")
    print(final_emd_df)

    output_fp = main_output_dir +"/Final-Results.csv"
    final_emd_df.to_csv(output_fp, index=False)
    print(output_fp)

    #plot it
    plot_cols = ["infoID","best_non_pb_model_avg_emd", "Persistence_Baseline_avg_emd"]
    plot_df = final_emd_df[plot_cols]
    error_sum_series = plot_df["best_non_pb_model_avg_emd"] + plot_df["Persistence_Baseline_avg_emd"]
    plot_df["best_non_pb_model_avg_emd"] = plot_df["best_non_pb_model_avg_emd"]/error_sum_series
    plot_df["Persistence_Baseline_avg_emd"] = plot_df["Persistence_Baseline_avg_emd"]/error_sum_series
    norm_fp = main_output_dir +"/Normalized-Final-Results.csv"
    plot_df.to_csv(norm_fp, index=False)

    plot_df = plot_df.rename(columns={"best_non_pb_model_avg_emd": "VAM"})
    plot_df = plot_df.rename(columns={"Persistence_Baseline_avg_emd": "Persistence Baseline"})
    plot_df = plot_df.rename(columns={"infoID":"Topic"})
    plot_df = plot_df.set_index("Topic")

    sns.set()
    plot_df.plot(kind="bar")
    fig = plt.gcf()
    fig.set_size_inches(5,5)
    ax = plt.gca()

    ax.set_title("EMD Results")
    ax.tick_params(labelsize=7)
    plt.tight_layout()
    fig.savefig(main_output_dir + "EMD-Bar-plots.png")
    fig.savefig(main_output_dir + "EMD-Bar-plots.svg")
    plt.close()



    return final_emd_df


def eval_model_rh_v2_SIMPLE(infoIDs, model_tag_input_dir,hyp_dict,TIMESTEP_DEBUG,gt_and_baseline_input_dir,platform, model_tag,model_output_dir):

    keep_cols = ["timestep", "nodeUserID", "parentUserID"]

    for infoID in infoIDs:
        print("\nEvaluating %s"%infoID)

        hyp_infoID = hyp_dict[infoID]


        cur_model_fp = model_tag_input_dir + "Simulations/" +  hyp_infoID + ".csv"
        cur_model_df = pd.read_csv(cur_model_fp)
        cur_model_df=cur_model_df[keep_cols]
        print("\ncur_model_df")
        print(cur_model_df)

        output_dir = model_tag_input_dir + "RH-Materials/"
        create_output_dir(output_dir)

        # cur_model_df["edge_weight_this_timestep"] = cur_model_df.groupby(["nodeUserID", "parentUserID", "timestep"])["nodeUserID"].transform("count")
        # cur_model_df = cur_model_df.drop_duplicates()
        # cur_model_df=cur_model_df.sort_values(["timestep" ,"edge_weight_this_timestep"],ascending=False).reset_index(drop=True)
        # print("\ncur_model_df")
        # print(cur_model_df)

        #============ convert these to graphs ==============
        timesteps = gt_df["timestep"].unique()
        timesteps = np.sort(timesteps)
        print(timesteps)

        
        if TIMESTEP_DEBUG == True:
            timesteps = timesteps[:1]

        all_timestep_rhds_for_infoID = []
        for timestep in timesteps:


            #===================== baseline graph so far =====================
            temp_model_df = cur_model_df[cur_model_df["timestep"]==timestep]
            temp_model_df=temp_model_df[["nodeUserID", "parentUserID"]].drop_duplicates().reset_index(drop=True)
            temp_model_df["parentUserID_in_degree"] = temp_model_df.groupby(["parentUserID"])["nodeUserID"].transform("count")
            user_to_degree_dict = convert_df_2_cols_to_dict(temp_model_df, "parentUserID", "parentUserID_in_degree")
            all_users = list(temp_model_df["parentUserID"].unique()) + list(temp_model_df["nodeUserID"].unique())
            all_users =list(set(all_users))
            temp_model_df = pd.DataFrame(data={"nodeUserID":all_users})
            temp_model_df["nodeUserID_in_degree"] = temp_model_df["nodeUserID"].map(user_to_degree_dict)
            temp_model_df["nodeUserID_in_degree"] =temp_model_df["nodeUserID_in_degree"].fillna(0)
            temp_model_df=temp_model_df.sort_values("nodeUserID_in_degree", ascending=False).reset_index(drop=True)

            cur_gt_fp = gt_and_baseline_input_dir+"Ground-Truth-Results/" + platform + "/In-Degrees-by-Timestep/Timestep-" + str(timestep) + "/" + hyp_infoID + ".csv"

            try:
                temp_gt_df = pd.read_csv(cur_gt_fp)
            except:
                temp_gt_df = pd.DataFrame(data={"nodeUserID_in_degree": [0]})

            if temp_model_df.shape[0] == 0:
                temp_model_df = pd.DataFrame(data={"nodeUserID":["n/a"], "nodeUserID_in_degree": [0]})

            print("\nGetting RH...")
            cur_rh = rh_distance(temp_gt_df["nodeUserID_in_degree"].astype("int32"),temp_model_df["nodeUserID_in_degree"].astype("int32"))
            all_timestep_rhds_for_infoID.append(cur_rh)

            #make cur save dirs
            #model_timestep_output_dir =  model_tag_input_dir + model_tag + "/" +platform + "/In-Degrees-by-Timestep/Timestep-%d/"%(timestep)

            model_timestep_output_dir = output_dir+ "/In-Degrees-by-Timestep/Timestep-%d/"%(timestep)
            create_output_dir(model_timestep_output_dir)

            model_timestep_output_fp = model_timestep_output_dir + hyp_infoID + ".csv"
            temp_model_df.to_csv(model_timestep_output_fp, index=False)
            print(model_timestep_output_fp)

        cur_infoID_rh_results = pd.DataFrame(data={"infoID":[infoID for i in range(len(list(timesteps)))], "RH":all_timestep_rhds_for_infoID, "timestep":timesteps})
        cur_infoID_rh_results = cur_infoID_rh_results[["timestep", "infoID", "RH"]]
        print("\ncur_infoID_rh_results")
        print(cur_infoID_rh_results)

        #main_rh_output_dir = model_tag_input_dir + model_tag+ "/"+ platform + "/"
        # main_rh_output_dir = model_tag_input_dir
        final_output_dir = output_dir + "RH-Results/"
        create_output_dir(final_output_dir)
        output_fp = final_output_dir + hyp_infoID + ".csv"
        cur_infoID_rh_results.to_csv(output_fp, index=False)
        print(output_fp)

    return output_dir



def compare_model_rh_results_to_baseline_v2_SIMPLE(baseline_results_input_dir, model_tag, main_rh_results_input_dir,  platform, infoIDs,hyp_dict):

    main_output_dir = main_rh_results_input_dir

    model_tag_to_dir_dict = {}
    model_tag_to_dir_dict["Persistence_Baseline"]= baseline_results_input_dir + platform + "/RH-Results/"
    model_tag_to_dir_dict[model_tag] = main_rh_results_input_dir + "/RH-Results/"
    model_tags = ["Persistence_Baseline",model_tag]

    #get results
    all_result_dfs = []
    for infoID in infoIDs:
        hyp_infoID = hyp_dict[infoID]
        avg_rh_list = []
        rh_model_list = []
        for model_tag in model_tags:
            cur_rh_fp = model_tag_to_dir_dict[model_tag] + hyp_infoID + ".csv"
            cur_rh_df = pd.read_csv(cur_rh_fp)
            print(cur_rh_df)

            # try:
            #     cur_rh_df = pd.read_csv(cur_rh_fp)

            # except FileNotFoundError:
            #     continue
            avg_rh = cur_rh_df["RH"].mean()
            avg_rh_list.append(avg_rh)
            rh_model_list.append(model_tag)

        all_models_rh_df = pd.DataFrame(data={"model_tag":rh_model_list, "avg_rh":avg_rh_list})
        all_models_rh_df["infoID"]=infoID
        all_models_rh_df = all_models_rh_df[["model_tag", "avg_rh","infoID"]]
        all_models_rh_df = all_models_rh_df.sort_values("avg_rh", ascending=True).reset_index(drop=True)
        print("\nall_models_rh_df")
        print(all_models_rh_df)

        # sys.exit(0)
        all_result_dfs.append(all_models_rh_df)

        # cur_output_dir = main_output_dir + platform + "/Results-by-InfoID/"
        cur_output_dir = main_output_dir + "/Results-by-InfoID/"
        create_output_dir(cur_output_dir)
        output_fp = cur_output_dir + hyp_infoID + ".csv"
        all_models_rh_df.to_csv(output_fp)
        print(output_fp)

    final_rh_df = pd.concat(all_result_dfs).reset_index(drop=True)
    print("\nfinal_rh_df")
    print(final_rh_df)

    # sys.exit(0)

    baseline_rh_df = final_rh_df[final_rh_df["model_tag"]=="Persistence_Baseline"].reset_index(drop=True)
    baseline_rh_df = baseline_rh_df.rename(columns={"avg_rh":"Persistence_Baseline_avg_rh"})



    final_rh_df_no_baseline_df = final_rh_df[final_rh_df["model_tag"] != "Persistence_Baseline"].reset_index(drop=True)
    final_rh_df_no_baseline_df=final_rh_df_no_baseline_df.drop_duplicates(["infoID"]).reset_index(drop=True)
    final_rh_df_no_baseline_df=final_rh_df_no_baseline_df.rename(columns={"avg_rh":"best_non_pb_model_avg_rh"})

    final_rh_df = final_rh_df.drop_duplicates(["infoID"]).reset_index(drop=True)
    final_rh_df=final_rh_df.rename(columns={"avg_rh":"winner_avg_rh", "model_tag":"winner_model_tag"})

    print("\nbaseline_rh_df")
    print(baseline_rh_df)
    baseline_rh_df=baseline_rh_df.drop("model_tag", axis=1)

    print("\nfinal_rh_df")
    print(final_rh_df)

    print("\nfinal_rh_df_no_baseline_df")
    final_rh_df_no_baseline_df=final_rh_df_no_baseline_df.rename(columns={"model_tag":"best_non_pb_model_tag"})
    print(final_rh_df_no_baseline_df)
    # final_rh_df_no_baseline_df=final_rh_df_no_baseline_df.drop("model_tag", axis=1)

    merge_dfs = [baseline_rh_df,final_rh_df,final_rh_df_no_baseline_df ]
    final_rh_df = merge_mult_dfs(merge_dfs, on="infoID", how="inner")
    final_rh_df = final_rh_df.sort_values("winner_avg_rh", ascending=True).reset_index(drop=True)


    #fix bug
    non_ties_result_df = final_rh_df[final_rh_df["Persistence_Baseline_avg_rh"]!=final_rh_df["best_non_pb_model_avg_rh"]].reset_index(drop=True)
    ties_result_df = final_rh_df[final_rh_df["Persistence_Baseline_avg_rh"]==final_rh_df["best_non_pb_model_avg_rh"]].reset_index(drop=True)
    if ties_result_df.shape[0] != 0:
        ties_result_df["winner_model_tag"] = "tie"
        final_rh_df = pd.concat([non_ties_result_df,ties_result_df]).reset_index(drop=True)

    print("\nfinal_rh_df")
    print(final_rh_df)

    output_fp = main_output_dir +"/Final-Results.csv"
    final_rh_df.to_csv(output_fp, index=False)
    print(output_fp)

    #plot it
    # plot_cols = ["infoID","best_non_pb_model_avg_rh", "Persistence_Baseline_avg_rh"]
    # plot_df = final_rh_df[plot_cols]
    # plot_df = plot_df.rename(columns={"best_non_pb_model_avg_rh": "VAM"})
    # plot_df = plot_df.rename(columns={"Persistence_Baseline_avg_rh": "Persistence Baseline"})
    # plot_df = plot_df.set_index("infoID")

    #plot it
    plot_cols = ["infoID","best_non_pb_model_avg_rh", "Persistence_Baseline_avg_rh"]
    plot_df = final_rh_df[plot_cols]
    error_sum_series = plot_df["best_non_pb_model_avg_rh"] + plot_df["Persistence_Baseline_avg_rh"]
    plot_df["best_non_pb_model_avg_rh"] = plot_df["best_non_pb_model_avg_rh"]/error_sum_series
    plot_df["Persistence_Baseline_avg_rh"] = plot_df["Persistence_Baseline_avg_rh"]/error_sum_series
    norm_fp = main_output_dir +"/Normalized-Final-Results.csv"
    plot_df.to_csv(norm_fp, index=False)

    plot_df = plot_df.rename(columns={"best_non_pb_model_avg_rh": "VAM"})
    plot_df = plot_df.rename(columns={"Persistence_Baseline_avg_rh": "Persistence Baseline"})
    plot_df = plot_df.rename(columns={"infoID":"Topic"})
    plot_df = plot_df.set_index("Topic")

    sns.set()
    plot_df.plot(kind="bar")
    fig = plt.gcf()
    fig.set_size_inches(5,5)
    ax = plt.gca()

    ax.set_title("RH Results")
    ax.tick_params(labelsize=7)
    plt.tight_layout()
    fig.savefig(main_output_dir + "RH-Bar-plots.png")
    fig.savefig(main_output_dir + "RH-Bar-plots.svg")
    plt.close()



    return final_rh_df

def compare_model_rh_results_to_baseline_v3_sep_output_dir(baseline_results_input_dir, model_tag, main_rh_results_input_dir,main_output_dir,  platform, infoIDs,hyp_dict):

    # main_output_dir = main_rh_results_input_dir

    model_tag_to_dir_dict = {}
    model_tag_to_dir_dict["Persistence_Baseline"]= baseline_results_input_dir + platform + "/RH-Results/"
    model_tag_to_dir_dict[model_tag] = main_rh_results_input_dir + "/RH-Results/"
    model_tags = ["Persistence_Baseline",model_tag]

    #get results
    all_result_dfs = []
    for infoID in infoIDs:
        hyp_infoID = hyp_dict[infoID]
        avg_rh_list = []
        rh_model_list = []
        for model_tag in model_tags:
            cur_rh_fp = model_tag_to_dir_dict[model_tag] + hyp_infoID + ".csv"
            cur_rh_df = pd.read_csv(cur_rh_fp)
            print(cur_rh_df)

            # try:
            #     cur_rh_df = pd.read_csv(cur_rh_fp)

            # except FileNotFoundError:
            #     continue
            avg_rh = cur_rh_df["RH"].mean()
            avg_rh_list.append(avg_rh)
            rh_model_list.append(model_tag)

        all_models_rh_df = pd.DataFrame(data={"model_tag":rh_model_list, "avg_rh":avg_rh_list})
        all_models_rh_df["infoID"]=infoID
        all_models_rh_df = all_models_rh_df[["model_tag", "avg_rh","infoID"]]
        all_models_rh_df = all_models_rh_df.sort_values("avg_rh", ascending=True).reset_index(drop=True)
        print("\nall_models_rh_df")
        print(all_models_rh_df)

        # sys.exit(0)
        all_result_dfs.append(all_models_rh_df)

        # cur_output_dir = main_output_dir + platform + "/Results-by-InfoID/"
        cur_output_dir = main_output_dir + "/Results-by-InfoID/"
        create_output_dir(cur_output_dir)
        output_fp = cur_output_dir + hyp_infoID + ".csv"
        all_models_rh_df.to_csv(output_fp)
        print(output_fp)

    final_rh_df = pd.concat(all_result_dfs).reset_index(drop=True)
    print("\nfinal_rh_df")
    print(final_rh_df)

    # sys.exit(0)

    baseline_rh_df = final_rh_df[final_rh_df["model_tag"]=="Persistence_Baseline"].reset_index(drop=True)
    baseline_rh_df = baseline_rh_df.rename(columns={"avg_rh":"Persistence_Baseline_avg_rh"})



    final_rh_df_no_baseline_df = final_rh_df[final_rh_df["model_tag"] != "Persistence_Baseline"].reset_index(drop=True)
    final_rh_df_no_baseline_df=final_rh_df_no_baseline_df.drop_duplicates(["infoID"]).reset_index(drop=True)
    final_rh_df_no_baseline_df=final_rh_df_no_baseline_df.rename(columns={"avg_rh":"best_non_pb_model_avg_rh"})

    final_rh_df = final_rh_df.drop_duplicates(["infoID"]).reset_index(drop=True)
    final_rh_df=final_rh_df.rename(columns={"avg_rh":"winner_avg_rh", "model_tag":"winner_model_tag"})

    print("\nbaseline_rh_df")
    print(baseline_rh_df)
    baseline_rh_df=baseline_rh_df.drop("model_tag", axis=1)

    print("\nfinal_rh_df")
    print(final_rh_df)

    print("\nfinal_rh_df_no_baseline_df")
    final_rh_df_no_baseline_df=final_rh_df_no_baseline_df.rename(columns={"model_tag":"best_non_pb_model_tag"})
    print(final_rh_df_no_baseline_df)
    # final_rh_df_no_baseline_df=final_rh_df_no_baseline_df.drop("model_tag", axis=1)

    merge_dfs = [baseline_rh_df,final_rh_df,final_rh_df_no_baseline_df ]
    final_rh_df = merge_mult_dfs(merge_dfs, on="infoID", how="inner")
    final_rh_df = final_rh_df.sort_values("winner_avg_rh", ascending=True).reset_index(drop=True)


    #fix bug
    non_ties_result_df = final_rh_df[final_rh_df["Persistence_Baseline_avg_rh"]!=final_rh_df["best_non_pb_model_avg_rh"]].reset_index(drop=True)
    ties_result_df = final_rh_df[final_rh_df["Persistence_Baseline_avg_rh"]==final_rh_df["best_non_pb_model_avg_rh"]].reset_index(drop=True)
    if ties_result_df.shape[0] != 0:
        ties_result_df["winner_model_tag"] = "tie"
        final_rh_df = pd.concat([non_ties_result_df,ties_result_df]).reset_index(drop=True)

    print("\nfinal_rh_df")
    print(final_rh_df)

    output_fp = main_output_dir +"/Final-Results.csv"
    final_rh_df.to_csv(output_fp, index=False)
    print(output_fp)

    #plot it
    # plot_cols = ["infoID","best_non_pb_model_avg_rh", "Persistence_Baseline_avg_rh"]
    # plot_df = final_rh_df[plot_cols]
    # plot_df = plot_df.rename(columns={"best_non_pb_model_avg_rh": "VAM"})
    # plot_df = plot_df.rename(columns={"Persistence_Baseline_avg_rh": "Persistence Baseline"})
    # plot_df = plot_df.set_index("infoID")

    #plot it
    plot_cols = ["infoID","best_non_pb_model_avg_rh", "Persistence_Baseline_avg_rh"]
    plot_df = final_rh_df[plot_cols]
    error_sum_series = plot_df["best_non_pb_model_avg_rh"] + plot_df["Persistence_Baseline_avg_rh"]
    plot_df["best_non_pb_model_avg_rh"] = plot_df["best_non_pb_model_avg_rh"]/error_sum_series
    plot_df["Persistence_Baseline_avg_rh"] = plot_df["Persistence_Baseline_avg_rh"]/error_sum_series
    norm_fp = main_output_dir +"/Normalized-Final-Results.csv"
    plot_df.to_csv(norm_fp, index=False)

    plot_df = plot_df.rename(columns={"best_non_pb_model_avg_rh": "VAM"})
    plot_df = plot_df.rename(columns={"Persistence_Baseline_avg_rh": "Persistence Baseline"})
    plot_df = plot_df.rename(columns={"infoID":"Topic"})
    plot_df = plot_df.set_index("Topic")

    sns.set()
    plot_df.plot(kind="bar")
    fig = plt.gcf()
    fig.set_size_inches(5,5)
    ax = plt.gca()

    ax.set_title("RH Results")
    ax.tick_params(labelsize=7)
    plt.tight_layout()
    fig.savefig(main_output_dir + "RH-Bar-plots.png")
    fig.savefig(main_output_dir + "RH-Bar-plots.svg")
    plt.close()



    return final_rh_df


def get_emd_metrics_v4_FIX_EDGES(infoIDs, model_tag_input_dir,hyp_dict,gt_and_baseline_input_dir,platform, model_tag,NUM_TIMESTEPS, REMOVE_MISSING_PARENT_USERS=False):

    keep_cols = ["timestep", "nodeUserID", "parentUserID","edge_weight_this_timestep"]

    for infoID in infoIDs:
        print("\nEvaluating %s"%infoID)

        hyp_infoID = hyp_dict[infoID]


        cur_model_fp = model_tag_input_dir + "Simulations/" +  hyp_infoID + ".csv"
        output_dir = model_tag_input_dir + "EMD-Materials/"
        create_output_dir(output_dir)

        try:
            cur_model_df = pd.read_csv(cur_model_fp)
        except FileNotFoundError:
            continue

        print(cur_model_df)
        # sys.exit(0)
        cur_model_df=cur_model_df[keep_cols]
        print("\ncur_model_df")
        print(cur_model_df)

        if REMOVE_MISSING_PARENT_USERS == True:
            cur_model_df = cur_model_df[cur_model_df["parentUserID"] != "missing_parentUserID"].reset_index(drop=True)

        # if "edge_weight_this_timestep" not in list(cur_model_df):
        #     cur_model_df["edge_weight_this_timestep"] = cur_model_df.groupby(["nodeUserID", "parentUserID", "timestep"])["nodeUserID"].transform("count")
        cur_model_df = cur_model_df.drop_duplicates()
        cur_model_df=cur_model_df.sort_values(["timestep" ,"edge_weight_this_timestep"],ascending=False).reset_index(drop=True)
        print("\ncur_model_df")
        print(cur_model_df)

        #============ convert these to graphs ==============
        # num_timesteps = cur_model_df["timestep"].nunique()
        # if TIMESTEP_DEBUG == True:
        #     num_timesteps = 1
        
        timesteps = [i+1 for i in range(NUM_TIMESTEPS)]
        print(timesteps)
        all_timestep_emds_for_infoID = []
        for timestep in timesteps:


            #===================== baseline graph so far =====================
            temp_model_df = cur_model_df[cur_model_df["timestep"]==timestep]
            MODEL_EDGE_LIST = zip(temp_model_df["nodeUserID"],temp_model_df["parentUserID"],temp_model_df["edge_weight_this_timestep"])
            MODEL_GRAPH =nx.DiGraph()
            MODEL_GRAPH.add_weighted_edges_from(MODEL_EDGE_LIST)

            # MODEL_GRAPH = nx.from_pandas_edgelist(cur_model_df,"nodeUserID", "parentUserID",edge_attr="edge_weight_this_timestep")
            print("\nMODEL_GRAPH")
            print(MODEL_GRAPH)
            model_edges = list(MODEL_GRAPH.edges())
            # for i in range(10):
            #   print(model_edges[i])
            num_baseline_nodes = MODEL_GRAPH.number_of_nodes()
            num_model_edges = MODEL_GRAPH.number_of_edges()
            print("\nnum_baseline_nodes: %d"%num_baseline_nodes)
            print("num_model_edges: %d"%num_model_edges)
            print("is weighted:")
            print(nx.is_weighted(MODEL_GRAPH))

            #===================== pr =====================
            print("\nGetting pr dicts...")
            if platform == "youtube":
                MODEL_PAGE_RANK_DICT = nx.pagerank_numpy(MODEL_GRAPH, weight="weight")
            else:
                MODEL_PAGE_RANK_DICT = nx.pagerank(MODEL_GRAPH, weight="weight")

            print("Got page ranks!")

            #===================== sort them =====================

            MODEL_PAGE_RANK_DF = pd.DataFrame(list(MODEL_PAGE_RANK_DICT.items()), columns=["node", "page_rank"])
            if MODEL_PAGE_RANK_DF.shape[0] == 0:
                MODEL_PAGE_RANK_DF = pd.DataFrame(data={"node":["n/a"], "page_rank":[0]})
            MODEL_PAGE_RANK_DF = MODEL_PAGE_RANK_DF.sort_values("page_rank", ascending=False).reset_index(drop=True)
            print("\nMODEL_PAGE_RANK_DF")
            print(MODEL_PAGE_RANK_DF)

            model_pr_dist = MODEL_PAGE_RANK_DF["page_rank"].values

            cur_gt_fp = gt_and_baseline_input_dir+"Ground-Truth-Results/" + platform + "/Results-by-Timestep/Timestep-" + str(timestep) + "/" + hyp_infoID + ".csv"

            try:
                gt_pr_dist = pd.read_csv(cur_gt_fp)["page_rank"].values
            except:
                gt_pr_dist = np.asarray([0])

            print("\nGetting EMD...")
            EMD = wasserstein_distance(gt_pr_dist,model_pr_dist)
            print(EMD)
            all_timestep_emds_for_infoID.append(EMD)

            #make cur save dirs
            # model_timestep_output_dir = model_tag_input_dir + model_tag + "/" +platform + "/Results-by-Timestep/Timestep-%d/"%(timestep)

            model_timestep_output_dir = output_dir + "/Page-Ranks-by-Timestep/Timestep-%d/"%(timestep)
            create_output_dir(model_timestep_output_dir)

            model_timestep_output_fp = model_timestep_output_dir + hyp_infoID + ".csv"
            MODEL_PAGE_RANK_DF.to_csv(model_timestep_output_fp, index=False)
            print(model_timestep_output_fp)

        #cur_infoID_emd_results = pd.DataFrame(data={"infoID":[infoID for i in range(num_timesteps)], "EMD":all_timestep_emds_for_infoID, "timestep":[(i+1) for i in range(num_timesteps)]})
        cur_infoID_emd_results = pd.DataFrame(data={"infoID":[infoID for i in range(len(list(timesteps)))], "EMD":all_timestep_emds_for_infoID, "timestep":timesteps})
        cur_infoID_emd_results = cur_infoID_emd_results[["timestep", "infoID", "EMD"]]
        print("\ncur_infoID_emd_results")
        print(cur_infoID_emd_results)

        #main_emd_results_output_dir = model_tag_input_dir + model_tag+ "/"+ platform + "/"
        # main_emd_results_output_dir = model_tag_input_dir

        final_output_dir = output_dir + "EMD-Results/"
        create_output_dir(final_output_dir)
        output_fp = final_output_dir + hyp_infoID + ".csv"
        cur_infoID_emd_results.to_csv(output_fp, index=False)
        print(output_fp)

        print(cur_infoID_emd_results)


    return output_dir









def get_emd_metrics_v3_TS_PARAM(infoIDs, model_tag_input_dir,hyp_dict,gt_and_baseline_input_dir,platform, model_tag,NUM_TIMESTEPS, REMOVE_MISSING_PARENT_USERS=False):

    keep_cols = ["timestep", "nodeUserID", "parentUserID"]

    for infoID in infoIDs:
        print("\nEvaluating %s"%infoID)

        hyp_infoID = hyp_dict[infoID]


        cur_model_fp = model_tag_input_dir + "Simulations/" +  hyp_infoID + ".csv"
        output_dir = model_tag_input_dir + "EMD-Materials/"
        create_output_dir(output_dir)

        try:
            cur_model_df = pd.read_csv(cur_model_fp)
        except FileNotFoundError:
            continue

        print(cur_model_df)
        # sys.exit(0)
        cur_model_df=cur_model_df[keep_cols]
        print("\ncur_model_df")
        print(cur_model_df)

        if REMOVE_MISSING_PARENT_USERS == True:
            cur_model_df = cur_model_df[cur_model_df["parentUserID"] != "missing_parentUserID"].reset_index(drop=True)

        if "edge_weight_this_timestep" not in list(cur_model_df):
            cur_model_df["edge_weight_this_timestep"] = cur_model_df.groupby(["nodeUserID", "parentUserID", "timestep"])["nodeUserID"].transform("count")

        cur_model_df = cur_model_df.drop_duplicates()
        cur_model_df=cur_model_df.sort_values(["timestep" ,"edge_weight_this_timestep"],ascending=False).reset_index(drop=True)
        print("\ncur_model_df")
        print(cur_model_df)

        #============ convert these to graphs ==============
        # num_timesteps = cur_model_df["timestep"].nunique()
        # if TIMESTEP_DEBUG == True:
        #     num_timesteps = 1
        
        timesteps = [i+1 for i in range(NUM_TIMESTEPS)]
        print(timesteps)
        all_timestep_emds_for_infoID = []
        for timestep in timesteps:


            #===================== baseline graph so far =====================
            temp_model_df = cur_model_df[cur_model_df["timestep"]==timestep]
            MODEL_EDGE_LIST = zip(temp_model_df["nodeUserID"],temp_model_df["parentUserID"],temp_model_df["edge_weight_this_timestep"])
            MODEL_GRAPH =nx.DiGraph()
            MODEL_GRAPH.add_weighted_edges_from(MODEL_EDGE_LIST)

            # MODEL_GRAPH = nx.from_pandas_edgelist(cur_model_df,"nodeUserID", "parentUserID",edge_attr="edge_weight_this_timestep")
            print("\nMODEL_GRAPH")
            print(MODEL_GRAPH)
            model_edges = list(MODEL_GRAPH.edges())
            # for i in range(10):
            #   print(model_edges[i])
            num_baseline_nodes = MODEL_GRAPH.number_of_nodes()
            num_model_edges = MODEL_GRAPH.number_of_edges()
            print("\nnum_baseline_nodes: %d"%num_baseline_nodes)
            print("num_model_edges: %d"%num_model_edges)
            print("is weighted:")
            print(nx.is_weighted(MODEL_GRAPH))

            #===================== pr =====================
            print("\nGetting pr dicts...")
            if platform == "youtube":
                MODEL_PAGE_RANK_DICT = nx.pagerank_numpy(MODEL_GRAPH, weight="weight")
            else:
                MODEL_PAGE_RANK_DICT = nx.pagerank(MODEL_GRAPH, weight="weight")

            print("Got page ranks!")

            #===================== sort them =====================

            MODEL_PAGE_RANK_DF = pd.DataFrame(list(MODEL_PAGE_RANK_DICT.items()), columns=["node", "page_rank"])
            if MODEL_PAGE_RANK_DF.shape[0] == 0:
                MODEL_PAGE_RANK_DF = pd.DataFrame(data={"node":["n/a"], "page_rank":[0]})
            MODEL_PAGE_RANK_DF = MODEL_PAGE_RANK_DF.sort_values("page_rank", ascending=False).reset_index(drop=True)
            print("\nMODEL_PAGE_RANK_DF")
            print(MODEL_PAGE_RANK_DF)

            model_pr_dist = MODEL_PAGE_RANK_DF["page_rank"].values

            cur_gt_fp = gt_and_baseline_input_dir+"Ground-Truth-Results/" + platform + "/Results-by-Timestep/Timestep-" + str(timestep) + "/" + hyp_infoID + ".csv"

            try:
                gt_pr_dist = pd.read_csv(cur_gt_fp)["page_rank"].values
            except:
                gt_pr_dist = np.asarray([0])

            print("\nGetting EMD...")
            EMD = wasserstein_distance(gt_pr_dist,model_pr_dist)
            print(EMD)
            all_timestep_emds_for_infoID.append(EMD)

            #make cur save dirs
            # model_timestep_output_dir = model_tag_input_dir + model_tag + "/" +platform + "/Results-by-Timestep/Timestep-%d/"%(timestep)

            model_timestep_output_dir = output_dir + "/Page-Ranks-by-Timestep/Timestep-%d/"%(timestep)
            create_output_dir(model_timestep_output_dir)

            model_timestep_output_fp = model_timestep_output_dir + hyp_infoID + ".csv"
            MODEL_PAGE_RANK_DF.to_csv(model_timestep_output_fp, index=False)
            print(model_timestep_output_fp)

        #cur_infoID_emd_results = pd.DataFrame(data={"infoID":[infoID for i in range(num_timesteps)], "EMD":all_timestep_emds_for_infoID, "timestep":[(i+1) for i in range(num_timesteps)]})
        cur_infoID_emd_results = pd.DataFrame(data={"infoID":[infoID for i in range(len(list(timesteps)))], "EMD":all_timestep_emds_for_infoID, "timestep":timesteps})
        cur_infoID_emd_results = cur_infoID_emd_results[["timestep", "infoID", "EMD"]]
        print("\ncur_infoID_emd_results")
        print(cur_infoID_emd_results)

        #main_emd_results_output_dir = model_tag_input_dir + model_tag+ "/"+ platform + "/"
        # main_emd_results_output_dir = model_tag_input_dir

        final_output_dir = output_dir + "EMD-Results/"
        create_output_dir(final_output_dir)
        output_fp = final_output_dir + hyp_infoID + ".csv"
        cur_infoID_emd_results.to_csv(output_fp, index=False)
        print(output_fp)

        print(cur_infoID_emd_results)


    return output_dir

def eval_model_rh_v4_sep_output_dir(infoIDs, model_tag_input_dir,main_output_dir ,hyp_dict,gt_and_baseline_input_dir,platform, model_tag, NUM_TIMESTEPS, REMOVE_MISSING_PARENT_USERS=True):

    keep_cols = ["timestep", "nodeUserID", "parentUserID"]

    for infoID in infoIDs:
        print("\nEvaluating %s"%infoID)

        hyp_infoID = hyp_dict[infoID]


        cur_model_fp = model_tag_input_dir + "Simulations/" +  hyp_infoID + ".csv"
        cur_model_df = pd.read_csv(cur_model_fp)
        cur_model_df=cur_model_df[keep_cols]
        print("\ncur_model_df")
        print(cur_model_df)

        if REMOVE_MISSING_PARENT_USERS == True:
            print("\nsize before drop missing: %d"%cur_model_df.shape[0])
            cur_model_df = cur_model_df[cur_model_df["parentUserID"] != "missing_parentUserID"].reset_index(drop=True)
            print("\nsize after drop missing: %d"%cur_model_df.shape[0])

        output_dir = main_output_dir + "RH-Materials/"
        create_output_dir(output_dir)

        # cur_model_df["edge_weight_this_timestep"] = cur_model_df.groupby(["nodeUserID", "parentUserID", "timestep"])["nodeUserID"].transform("count")
        # cur_model_df = cur_model_df.drop_duplicates()
        # cur_model_df=cur_model_df.sort_values(["timestep" ,"edge_weight_this_timestep"],ascending=False).reset_index(drop=True)
        # print("\ncur_model_df")
        # print(cur_model_df)

        #============ convert these to graphs ==============
        timesteps = [i+1 for i in range(NUM_TIMESTEPS)]
        print(timesteps)
        all_timestep_rhds_for_infoID = []
        for timestep in timesteps:


            #===================== baseline graph so far =====================
            temp_model_df = cur_model_df[cur_model_df["timestep"]==timestep]
            temp_model_df=temp_model_df[["nodeUserID", "parentUserID"]].drop_duplicates().reset_index(drop=True)
            temp_model_df["parentUserID_in_degree"] = temp_model_df.groupby(["parentUserID"])["nodeUserID"].transform("count")
            user_to_degree_dict = convert_df_2_cols_to_dict(temp_model_df, "parentUserID", "parentUserID_in_degree")
            all_users = list(temp_model_df["parentUserID"].unique()) + list(temp_model_df["nodeUserID"].unique())
            all_users =list(set(all_users))
            temp_model_df = pd.DataFrame(data={"nodeUserID":all_users})
            temp_model_df["nodeUserID_in_degree"] = temp_model_df["nodeUserID"].map(user_to_degree_dict)
            temp_model_df["nodeUserID_in_degree"] =temp_model_df["nodeUserID_in_degree"].fillna(0)
            temp_model_df=temp_model_df.sort_values("nodeUserID_in_degree", ascending=False).reset_index(drop=True)

            cur_gt_fp = gt_and_baseline_input_dir+"Ground-Truth-Results/" + platform + "/In-Degrees-by-Timestep/Timestep-" + str(timestep) + "/" + hyp_infoID + ".csv"

            try:
                temp_gt_df = pd.read_csv(cur_gt_fp)
            except:
                temp_gt_df = pd.DataFrame(data={"nodeUserID_in_degree": [0]})

            if temp_model_df.shape[0] == 0:
                temp_model_df = pd.DataFrame(data={"nodeUserID":["n/a"], "nodeUserID_in_degree": [0]})

            print("\nGetting RH...")
            cur_rh = rh_distance(temp_gt_df["nodeUserID_in_degree"].astype("int32"),temp_model_df["nodeUserID_in_degree"].astype("int32"))
            all_timestep_rhds_for_infoID.append(cur_rh)

            #make cur save dirs
            #model_timestep_output_dir =  model_tag_input_dir + model_tag + "/" +platform + "/In-Degrees-by-Timestep/Timestep-%d/"%(timestep)

            model_timestep_output_dir = output_dir+ "/In-Degrees-by-Timestep/Timestep-%d/"%(timestep)
            create_output_dir(model_timestep_output_dir)

            model_timestep_output_fp = model_timestep_output_dir + hyp_infoID + ".csv"
            temp_model_df.to_csv(model_timestep_output_fp, index=False)
            print(model_timestep_output_fp)

        cur_infoID_rh_results = pd.DataFrame(data={"infoID":[infoID for i in range(len(list(timesteps)))], "RH":all_timestep_rhds_for_infoID, "timestep":timesteps})
        cur_infoID_rh_results = cur_infoID_rh_results[["timestep", "infoID", "RH"]]
        print("\ncur_infoID_rh_results")
        print(cur_infoID_rh_results)

        #main_rh_output_dir = model_tag_input_dir + model_tag+ "/"+ platform + "/"
        # main_rh_output_dir = model_tag_input_dir
        final_output_dir = output_dir + "RH-Results/"
        create_output_dir(final_output_dir)
        output_fp = final_output_dir + hyp_infoID + ".csv"
        cur_infoID_rh_results.to_csv(output_fp, index=False)
        print(output_fp)

    return output_dir

def eval_model_rh_v3_TS_PARAM(infoIDs, model_tag_input_dir,hyp_dict,gt_and_baseline_input_dir,platform, model_tag,model_output_dir, NUM_TIMESTEPS,REMOVE_MISSING_PARENT_USERS=False):

    #directed but unweighted

    keep_cols = ["timestep", "nodeUserID", "parentUserID"]

    for infoID in infoIDs:
        print("\nEvaluating %s"%infoID)

        hyp_infoID = hyp_dict[infoID]


        cur_model_fp = model_tag_input_dir + "Simulations/" +  hyp_infoID + ".csv"
        cur_model_df = pd.read_csv(cur_model_fp)
        cur_model_df=cur_model_df[keep_cols]

        if REMOVE_MISSING_PARENT_USERS==True:
            cur_model_df = cur_model_df[cur_model_df["parentUserID"] != "missing_parentUserID"].reset_index(drop=True)

        print("\ncur_model_df")
        print(cur_model_df)

        output_dir = model_tag_input_dir + "RH-Materials/"
        create_output_dir(output_dir)

        # cur_model_df["edge_weight_this_timestep"] = cur_model_df.groupby(["nodeUserID", "parentUserID", "timestep"])["nodeUserID"].transform("count")
        # cur_model_df = cur_model_df.drop_duplicates()
        # cur_model_df=cur_model_df.sort_values(["timestep" ,"edge_weight_this_timestep"],ascending=False).reset_index(drop=True)
        # print("\ncur_model_df")
        # print(cur_model_df)

        #============ convert these to graphs ==============
        timesteps = [i+1 for i in range(NUM_TIMESTEPS)]
        print(timesteps)
        all_timestep_rhds_for_infoID = []
        for timestep in timesteps:


            #===================== baseline graph so far =====================
            temp_model_df = cur_model_df[cur_model_df["timestep"]==timestep]
            temp_model_df=temp_model_df[["nodeUserID", "parentUserID"]].drop_duplicates().reset_index(drop=True)



            temp_model_df["parentUserID_in_degree"] = temp_model_df.groupby(["parentUserID"])["nodeUserID"].transform("count")
            user_to_degree_dict = convert_df_2_cols_to_dict(temp_model_df, "parentUserID", "parentUserID_in_degree")
            all_users = list(temp_model_df["parentUserID"].unique()) + list(temp_model_df["nodeUserID"].unique())
            all_users =list(set(all_users))
            temp_model_df = pd.DataFrame(data={"nodeUserID":all_users})
            temp_model_df["nodeUserID_in_degree"] = temp_model_df["nodeUserID"].map(user_to_degree_dict)
            temp_model_df["nodeUserID_in_degree"] =temp_model_df["nodeUserID_in_degree"].fillna(0)
            temp_model_df=temp_model_df.sort_values("nodeUserID_in_degree", ascending=False).reset_index(drop=True)

            cur_gt_fp = gt_and_baseline_input_dir+"Ground-Truth-Results/" + platform + "/In-Degrees-by-Timestep/Timestep-" + str(timestep) + "/" + hyp_infoID + ".csv"

            try:
                temp_gt_df = pd.read_csv(cur_gt_fp)
            except:
                temp_gt_df = pd.DataFrame(data={"nodeUserID_in_degree": [0]})

            if temp_model_df.shape[0] == 0:
                temp_model_df = pd.DataFrame(data={"nodeUserID":["n/a"], "nodeUserID_in_degree": [0]})

            print("\nGetting RH...")
            cur_rh = rh_distance(temp_gt_df["nodeUserID_in_degree"].astype("int32"),temp_model_df["nodeUserID_in_degree"].astype("int32"))
            all_timestep_rhds_for_infoID.append(cur_rh)

            #make cur save dirs
            #model_timestep_output_dir =  model_tag_input_dir + model_tag + "/" +platform + "/In-Degrees-by-Timestep/Timestep-%d/"%(timestep)

            model_timestep_output_dir = output_dir+ "/In-Degrees-by-Timestep/Timestep-%d/"%(timestep)
            create_output_dir(model_timestep_output_dir)

            model_timestep_output_fp = model_timestep_output_dir + hyp_infoID + ".csv"
            temp_model_df.to_csv(model_timestep_output_fp, index=False)
            print(model_timestep_output_fp)

        cur_infoID_rh_results = pd.DataFrame(data={"infoID":[infoID for i in range(len(list(timesteps)))], "RH":all_timestep_rhds_for_infoID, "timestep":timesteps})
        cur_infoID_rh_results = cur_infoID_rh_results[["timestep", "infoID", "RH"]]
        print("\ncur_infoID_rh_results")
        print(cur_infoID_rh_results)

        #main_rh_output_dir = model_tag_input_dir + model_tag+ "/"+ platform + "/"
        # main_rh_output_dir = model_tag_input_dir
        final_output_dir = output_dir + "RH-Results/"
        create_output_dir(final_output_dir)
        output_fp = final_output_dir + hyp_infoID + ".csv"
        cur_infoID_rh_results.to_csv(output_fp, index=False)
        print(output_fp)

    return output_dir






def get_emd_metrics_v4_unweighted(infoIDs, model_tag_input_dir,model_tag_output_dir ,hyp_dict,gt_and_baseline_input_dir,platform, model_tag,NUM_TIMESTEPS, WEIGHT):

    keep_cols = ["timestep", "nodeUserID", "parentUserID"]

    for infoID in infoIDs:
        print("\nEvaluating %s"%infoID)

        hyp_infoID = hyp_dict[infoID]


        cur_model_fp = model_tag_input_dir + "Simulations/" +  hyp_infoID + ".csv"
        output_dir = model_tag_output_dir + "EMD-Materials/"
        create_output_dir(output_dir)

        try:
            cur_model_df = pd.read_csv(cur_model_fp)
        except FileNotFoundError:
            continue

        print(cur_model_df)
        # sys.exit(0)
        cur_model_df=cur_model_df[keep_cols]
        print("\ncur_model_df")
        print(cur_model_df)

        cur_model_df["edge_weight_this_timestep"] = cur_model_df.groupby(["nodeUserID", "parentUserID", "timestep"])["nodeUserID"].transform("count")
        cur_model_df = cur_model_df.drop_duplicates()
        cur_model_df=cur_model_df.sort_values(["timestep" ,"edge_weight_this_timestep"],ascending=False).reset_index(drop=True)
        print("\ncur_model_df")
        print(cur_model_df)

        #============ convert these to graphs ==============
        # num_timesteps = cur_model_df["timestep"].nunique()
        # if TIMESTEP_DEBUG == True:
        #     num_timesteps = 1
        
        timesteps = [i+1 for i in range(NUM_TIMESTEPS)]
        print(timesteps)
        all_timestep_emds_for_infoID = []
        for timestep in timesteps:


            #===================== baseline graph so far =====================
            temp_model_df = cur_model_df[cur_model_df["timestep"]==timestep]
            MODEL_EDGE_LIST = zip(temp_model_df["nodeUserID"],temp_model_df["parentUserID"],temp_model_df["edge_weight_this_timestep"])
            MODEL_GRAPH =nx.DiGraph()
            MODEL_GRAPH.add_weighted_edges_from(MODEL_EDGE_LIST)

            # MODEL_GRAPH = nx.from_pandas_edgelist(cur_model_df,"nodeUserID", "parentUserID",edge_attr="edge_weight_this_timestep")
            print("\nMODEL_GRAPH")
            print(MODEL_GRAPH)
            model_edges = list(MODEL_GRAPH.edges())
            # for i in range(10):
            #   print(model_edges[i])
            num_baseline_nodes = MODEL_GRAPH.number_of_nodes()
            num_model_edges = MODEL_GRAPH.number_of_edges()
            print("\nnum_baseline_nodes: %d"%num_baseline_nodes)
            print("num_model_edges: %d"%num_model_edges)
            print("is weighted:")
            print(nx.is_weighted(MODEL_GRAPH))

            #weight_dict = nx.get_edge_attributes(MODEL_GRAPH,'weight')

            edges = list(MODEL_GRAPH.edges(data=True))

            if WEIGHT==None:
                for edge in edges:
                    s=edge[0]
                    t = edge[1]
                    MODEL_GRAPH[s][t]["weight"]=1


            k=100
            for edge in edges:
                print(edge)
                if k==10:
                    break
                k+=1

            sys.exit(0)

            

            #===================== pr =====================
            print("\nGetting pr dicts...")
            if platform == "youtube":
                MODEL_PAGE_RANK_DICT = nx.pagerank_numpy(MODEL_GRAPH, weight=WEIGHT)
            else:
                MODEL_PAGE_RANK_DICT = nx.pagerank(MODEL_GRAPH, weight=WEIGHT)

            print("Got page ranks!")

            #===================== sort them =====================

            MODEL_PAGE_RANK_DF = pd.DataFrame(list(MODEL_PAGE_RANK_DICT.items()), columns=["node", "page_rank"])
            if MODEL_PAGE_RANK_DF.shape[0] == 0:
                MODEL_PAGE_RANK_DF = pd.DataFrame(data={"node":["n/a"], "page_rank":[0]})
            MODEL_PAGE_RANK_DF = MODEL_PAGE_RANK_DF.sort_values("page_rank", ascending=False).reset_index(drop=True)
            print("\nMODEL_PAGE_RANK_DF")
            print(MODEL_PAGE_RANK_DF)

            


            

            model_pr_dist = MODEL_PAGE_RANK_DF["page_rank"].values

            cur_gt_fp = gt_and_baseline_input_dir+"Ground-Truth-Results/" + platform + "/Results-by-Timestep/Timestep-" + str(timestep) + "/" + hyp_infoID + ".csv"

            try:
                gt_pr_dist = pd.read_csv(cur_gt_fp)["page_rank"].values
            except:
                gt_pr_dist = np.asarray([0])

            print("\nGetting EMD...")
            EMD = wasserstein_distance(gt_pr_dist,model_pr_dist)
            print(EMD)
            all_timestep_emds_for_infoID.append(EMD)

            #make cur save dirs
            # model_timestep_output_dir = model_tag_input_dir + model_tag + "/" +platform + "/Results-by-Timestep/Timestep-%d/"%(timestep)

            model_timestep_output_dir = output_dir + "/Page-Ranks-by-Timestep/Timestep-%d/"%(timestep)
            create_output_dir(model_timestep_output_dir)

            model_timestep_output_fp = model_timestep_output_dir + hyp_infoID + ".csv"
            MODEL_PAGE_RANK_DF.to_csv(model_timestep_output_fp, index=False)
            print(model_timestep_output_fp)

        #cur_infoID_emd_results = pd.DataFrame(data={"infoID":[infoID for i in range(num_timesteps)], "EMD":all_timestep_emds_for_infoID, "timestep":[(i+1) for i in range(num_timesteps)]})
        cur_infoID_emd_results = pd.DataFrame(data={"infoID":[infoID for i in range(len(list(timesteps)))], "EMD":all_timestep_emds_for_infoID, "timestep":timesteps})
        cur_infoID_emd_results = cur_infoID_emd_results[["timestep", "infoID", "EMD"]]
        print("\ncur_infoID_emd_results")
        print(cur_infoID_emd_results)

        #main_emd_results_output_dir = model_tag_input_dir + model_tag+ "/"+ platform + "/"
        # main_emd_results_output_dir = model_tag_input_dir

        final_output_dir = output_dir + "EMD-Results/"
        create_output_dir(final_output_dir)
        output_fp = final_output_dir + hyp_infoID + ".csv"
        cur_infoID_emd_results.to_csv(output_fp, index=False)
        print(output_fp)

        print(cur_infoID_emd_results)


    return output_dir



def get_emd_metrics_v4_sep_output_dir(infoIDs, model_tag_input_dir, main_output_dir,hyp_dict,gt_and_baseline_input_dir,platform, model_tag,NUM_TIMESTEPS,REMOVE_MISSING_PARENT_USERS=True):

    keep_cols = ["timestep", "nodeUserID", "parentUserID"]

    for infoID in infoIDs:
        print("\nEvaluating %s"%infoID)

        hyp_infoID = hyp_dict[infoID]


        cur_model_fp = model_tag_input_dir + "Simulations/" +  hyp_infoID + ".csv"
        output_dir = main_output_dir + "EMD-Materials/"
        create_output_dir(output_dir)

        try:
            cur_model_df = pd.read_csv(cur_model_fp)
            if REMOVE_MISSING_PARENT_USERS == True:
                print("\nsize before drop missing: %d"%cur_model_df.shape[0])
                cur_model_df = cur_model_df[cur_model_df["parentUserID"] != "missing_parentUserID"].reset_index(drop=True)
                print("\nsize after drop missing: %d"%cur_model_df.shape[0])
        except FileNotFoundError:
            continue

        print(cur_model_df)
        # sys.exit(0)
        cur_model_df=cur_model_df[keep_cols]
        print("\ncur_model_df")
        print(cur_model_df)

        cur_model_df["edge_weight_this_timestep"] = cur_model_df.groupby(["nodeUserID", "parentUserID", "timestep"])["nodeUserID"].transform("count")
        cur_model_df = cur_model_df.drop_duplicates()
        cur_model_df=cur_model_df.sort_values(["timestep" ,"edge_weight_this_timestep"],ascending=False).reset_index(drop=True)
        print("\ncur_model_df")
        print(cur_model_df)

        #============ convert these to graphs ==============
        # num_timesteps = cur_model_df["timestep"].nunique()
        # if TIMESTEP_DEBUG == True:
        #     num_timesteps = 1
        
        timesteps = [i+1 for i in range(NUM_TIMESTEPS)]
        print(timesteps)
        all_timestep_emds_for_infoID = []
        for timestep in timesteps:


            #===================== baseline graph so far =====================
            temp_model_df = cur_model_df[cur_model_df["timestep"]==timestep]
            MODEL_EDGE_LIST = zip(temp_model_df["nodeUserID"],temp_model_df["parentUserID"],temp_model_df["edge_weight_this_timestep"])
            MODEL_GRAPH =nx.DiGraph()
            MODEL_GRAPH.add_weighted_edges_from(MODEL_EDGE_LIST)

            # MODEL_GRAPH = nx.from_pandas_edgelist(cur_model_df,"nodeUserID", "parentUserID",edge_attr="edge_weight_this_timestep")
            print("\nMODEL_GRAPH")
            print(MODEL_GRAPH)
            model_edges = list(MODEL_GRAPH.edges())
            # for i in range(10):
            #   print(model_edges[i])
            num_baseline_nodes = MODEL_GRAPH.number_of_nodes()
            num_model_edges = MODEL_GRAPH.number_of_edges()
            print("\nnum_baseline_nodes: %d"%num_baseline_nodes)
            print("num_model_edges: %d"%num_model_edges)
            print("is weighted:")
            print(nx.is_weighted(MODEL_GRAPH))

            #===================== pr =====================
            print("\nGetting pr dicts...")
            if platform == "youtube":
                MODEL_PAGE_RANK_DICT = nx.pagerank_numpy(MODEL_GRAPH, weight="weight")
            else:
                MODEL_PAGE_RANK_DICT = nx.pagerank(MODEL_GRAPH, weight="weight")

            print("Got page ranks!")

            #===================== sort them =====================

            MODEL_PAGE_RANK_DF = pd.DataFrame(list(MODEL_PAGE_RANK_DICT.items()), columns=["node", "page_rank"])
            if MODEL_PAGE_RANK_DF.shape[0] == 0:
                MODEL_PAGE_RANK_DF = pd.DataFrame(data={"node":["n/a"], "page_rank":[0]})
            MODEL_PAGE_RANK_DF = MODEL_PAGE_RANK_DF.sort_values("page_rank", ascending=False).reset_index(drop=True)
            print("\nMODEL_PAGE_RANK_DF")
            print(MODEL_PAGE_RANK_DF)

            model_pr_dist = MODEL_PAGE_RANK_DF["page_rank"].values

            cur_gt_fp = gt_and_baseline_input_dir+"Ground-Truth-Results/" + platform + "/Results-by-Timestep/Timestep-" + str(timestep) + "/" + hyp_infoID + ".csv"

            try:
                gt_pr_dist = pd.read_csv(cur_gt_fp)["page_rank"].values
            except:
                gt_pr_dist = np.asarray([0])

            print("\nGetting EMD...")
            EMD = wasserstein_distance(gt_pr_dist,model_pr_dist)
            print(EMD)
            all_timestep_emds_for_infoID.append(EMD)

            #make cur save dirs
            # model_timestep_output_dir = model_tag_input_dir + model_tag + "/" +platform + "/Results-by-Timestep/Timestep-%d/"%(timestep)

            model_timestep_output_dir = output_dir + "/Page-Ranks-by-Timestep/Timestep-%d/"%(timestep)
            create_output_dir(model_timestep_output_dir)

            model_timestep_output_fp = model_timestep_output_dir + hyp_infoID + ".csv"
            MODEL_PAGE_RANK_DF.to_csv(model_timestep_output_fp, index=False)
            print(model_timestep_output_fp)

        #cur_infoID_emd_results = pd.DataFrame(data={"infoID":[infoID for i in range(num_timesteps)], "EMD":all_timestep_emds_for_infoID, "timestep":[(i+1) for i in range(num_timesteps)]})
        cur_infoID_emd_results = pd.DataFrame(data={"infoID":[infoID for i in range(len(list(timesteps)))], "EMD":all_timestep_emds_for_infoID, "timestep":timesteps})
        cur_infoID_emd_results = cur_infoID_emd_results[["timestep", "infoID", "EMD"]]
        print("\ncur_infoID_emd_results")
        print(cur_infoID_emd_results)

        #main_emd_results_output_dir = model_tag_input_dir + model_tag+ "/"+ platform + "/"
        # main_emd_results_output_dir = model_tag_input_dir

        final_output_dir = output_dir + "EMD-Results/"
        create_output_dir(final_output_dir)
        output_fp = final_output_dir + hyp_infoID + ".csv"
        cur_infoID_emd_results.to_csv(output_fp, index=False)
        print(output_fp)

        print(cur_infoID_emd_results)


    return output_dir

def plot_basic(df, x_col, y_col, title, output_tag, kind):

    time_df[x_col] = minmax_scale_series(time_df[x_col])
    time_df[y_col] = minmax_scale_series(time_df[y_col])
    time_df.plot(x = x_col, y=y_col, kind=kind)
    fig = plt.gcf()
    ax = plt.gca()
    ax.set_title(title)
    output_fp = output_dir + "%s.png"%output_tag
    fig.savefig(output_fp)
    print(output_fp)
    output_fp = output_dir + "%s.svg"%output_tag
    fig.savefig(output_fp)
    print(output_fp)
    plt.close()


def minmax_scale_series(series):

    smax = series.max()
    smin = series.min()
    series = (series - smin)/(smax - smin)

    return series

def eval_model_rh_v5_CLEAN(infoIDs, model_tag_input_dir,main_output_dir ,hyp_dict,gt_and_baseline_input_dir,platform, model_tag, NUM_TIMESTEPS, REMOVE_MISSING_PARENT_USERS=True):

    #keep_cols = ["timestep", "nodeUserID", "parentUserID"]
    keep_cols = ["timestep", "nodeUserID", "parentUserID", "edge_weight_this_timestep"]

    for infoID in infoIDs:
        print("\nEvaluating %s"%infoID)

        hyp_infoID = hyp_dict[infoID]


        cur_model_fp = model_tag_input_dir + "Simulations/" +  hyp_infoID + ".csv"
        cur_model_df = pd.read_csv(cur_model_fp)
        cur_model_df=cur_model_df[keep_cols]
        print("\ncur_model_df")
        print(cur_model_df)

        # if REMOVE_MISSING_PARENT_USERS == True:
        #     print("\nsize before drop missing: %d"%cur_model_df.shape[0])
        #     cur_model_df = cur_model_df[cur_model_df["parentUserID"] != "missing_parentUserID"].reset_index(drop=True)
        #     print("\nsize after drop missing: %d"%cur_model_df.shape[0])

        try:
            cur_model_df = pd.read_csv(cur_model_fp)
            if REMOVE_MISSING_PARENT_USERS == True:
                print("\nsize before drop missing: %d"%cur_model_df.shape[0])
                cur_model_df = cur_model_df[cur_model_df["parentUserID"] != "missing_parentUserID"].reset_index(drop=True)
                print("\nsize after drop missing: %d"%cur_model_df.shape[0])
        except FileNotFoundError:
            continue

        output_dir = main_output_dir + "RH-Materials/"
        create_output_dir(output_dir)

        # cur_model_df["edge_weight_this_timestep"] = cur_model_df.groupby(["nodeUserID", "parentUserID", "timestep"])["nodeUserID"].transform("count")
        # cur_model_df = cur_model_df.drop_duplicates()
        # cur_model_df=cur_model_df.sort_values(["timestep" ,"edge_weight_this_timestep"],ascending=False).reset_index(drop=True)
        # print("\ncur_model_df")
        # print(cur_model_df)

        #============ convert these to graphs ==============
        timesteps = [i+1 for i in range(NUM_TIMESTEPS)]
        print(timesteps)
        all_timestep_rhds_for_infoID = []
        for timestep in timesteps:


            #===================== baseline graph so far =====================
            temp_model_df = cur_model_df[cur_model_df["timestep"]==timestep]
            temp_model_df=temp_model_df[["nodeUserID", "parentUserID"]].drop_duplicates().reset_index(drop=True)
            temp_model_df["parentUserID_in_degree"] = temp_model_df.groupby(["parentUserID"])["nodeUserID"].transform("count")
            user_to_degree_dict = convert_df_2_cols_to_dict(temp_model_df, "parentUserID", "parentUserID_in_degree")
            all_users = list(temp_model_df["parentUserID"].unique()) + list(temp_model_df["nodeUserID"].unique())
            all_users =list(set(all_users))
            temp_model_df = pd.DataFrame(data={"nodeUserID":all_users})
            temp_model_df["nodeUserID_in_degree"] = temp_model_df["nodeUserID"].map(user_to_degree_dict)
            temp_model_df["nodeUserID_in_degree"] =temp_model_df["nodeUserID_in_degree"].fillna(0)
            temp_model_df=temp_model_df.sort_values("nodeUserID_in_degree", ascending=False).reset_index(drop=True)

            cur_gt_fp = gt_and_baseline_input_dir+"Ground-Truth-Results/" + platform + "/In-Degrees-by-Timestep/Timestep-" + str(timestep) + "/" + hyp_infoID + ".csv"

            try:
                temp_gt_df = pd.read_csv(cur_gt_fp)
            except:
                temp_gt_df = pd.DataFrame(data={"nodeUserID_in_degree": [0]})

            if temp_model_df.shape[0] == 0:
                temp_model_df = pd.DataFrame(data={"nodeUserID":["n/a"], "nodeUserID_in_degree": [0]})

            print("\nGetting RH...")
            cur_rh = rh_distance(temp_gt_df["nodeUserID_in_degree"].astype("int32"),temp_model_df["nodeUserID_in_degree"].astype("int32"))
            all_timestep_rhds_for_infoID.append(cur_rh)

            #make cur save dirs
            #model_timestep_output_dir =  model_tag_input_dir + model_tag + "/" +platform + "/In-Degrees-by-Timestep/Timestep-%d/"%(timestep)

            model_timestep_output_dir = output_dir+ "/In-Degrees-by-Timestep/Timestep-%d/"%(timestep)
            create_output_dir(model_timestep_output_dir)

            model_timestep_output_fp = model_timestep_output_dir + hyp_infoID + ".csv"
            temp_model_df.to_csv(model_timestep_output_fp, index=False)
            print(model_timestep_output_fp)

        cur_infoID_rh_results = pd.DataFrame(data={"infoID":[infoID for i in range(len(list(timesteps)))], "RH":all_timestep_rhds_for_infoID, "timestep":timesteps})
        cur_infoID_rh_results = cur_infoID_rh_results[["timestep", "infoID", "RH"]]
        print("\ncur_infoID_rh_results")
        print(cur_infoID_rh_results)

        #main_rh_output_dir = model_tag_input_dir + model_tag+ "/"+ platform + "/"
        # main_rh_output_dir = model_tag_input_dir
        final_output_dir = output_dir + "RH-Results/"
        create_output_dir(final_output_dir)
        output_fp = final_output_dir + hyp_infoID + ".csv"
        cur_infoID_rh_results.to_csv(output_fp, index=False)
        print(output_fp)

    return output_dir

def get_emd_metrics_v5_CLEAN(infoIDs, model_tag_input_dir, main_output_dir,hyp_dict,gt_and_baseline_input_dir,platform, model_tag,NUM_TIMESTEPS,REMOVE_MISSING_PARENT_USERS=True):

    #NOTE: THERE CANNOT BE ANY DUPES IN THIS DATA

    keep_cols = ["timestep", "nodeUserID", "parentUserID", "edge_weight_this_timestep"]

    for infoID in infoIDs:
        print("\nEvaluating %s"%infoID)

        hyp_infoID = hyp_dict[infoID]

        cur_model_fp = model_tag_input_dir + "Simulations/" +  hyp_infoID + ".csv"
        output_dir = main_output_dir + "EMD-Materials/"
        create_output_dir(output_dir)

        try:
            cur_model_df = pd.read_csv(cur_model_fp)
            if REMOVE_MISSING_PARENT_USERS == True:
                print("\nsize before drop missing: %d"%cur_model_df.shape[0])
                cur_model_df = cur_model_df[cur_model_df["parentUserID"] != "missing_parentUserID"].reset_index(drop=True)
                print("\nsize after drop missing: %d"%cur_model_df.shape[0])
        except FileNotFoundError:
            continue

        print(cur_model_df)
        # sys.exit(0)
        cur_model_df=cur_model_df[keep_cols]
        print("\ncur_model_df")
        print(cur_model_df)

        # cur_model_df["edge_weight_this_timestep"] = cur_model_df.groupby(["nodeUserID", "parentUserID", "timestep"])["nodeUserID"].transform("count")
        size_before = cur_model_df.shape[0]
        cur_model_df = cur_model_df.drop_duplicates()
        size_after = cur_model_df.shape[0]

        print("\nDF size before and after drop dupes")
        print(size_before)
        print(size_after)
        if size_before != size_after:
            print("\nError! size_before != size_after ")
            print("Sort out your edge weights! Terminating!")
            sys.exit(0)

        cur_model_df=cur_model_df.sort_values(["timestep" ,"edge_weight_this_timestep"],ascending=False).reset_index(drop=True)
        print("\ncur_model_df")
        print(cur_model_df)

        #============ convert these to graphs ==============
        # num_timesteps = cur_model_df["timestep"].nunique()
        # if TIMESTEP_DEBUG == True:
        #     num_timesteps = 1
        
        timesteps = [i+1 for i in range(NUM_TIMESTEPS)]
        print(timesteps)
        all_timestep_emds_for_infoID = []
        for timestep in timesteps:


            #===================== baseline graph so far =====================
            temp_model_df = cur_model_df[cur_model_df["timestep"]==timestep]
            MODEL_EDGE_LIST = zip(temp_model_df["nodeUserID"],temp_model_df["parentUserID"],temp_model_df["edge_weight_this_timestep"])
            MODEL_GRAPH =nx.DiGraph()
            MODEL_GRAPH.add_weighted_edges_from(MODEL_EDGE_LIST)

            # MODEL_GRAPH = nx.from_pandas_edgelist(cur_model_df,"nodeUserID", "parentUserID",edge_attr="edge_weight_this_timestep")
            print("\nMODEL_GRAPH")
            print(MODEL_GRAPH)
            model_edges = list(MODEL_GRAPH.edges())
            # for i in range(10):
            #   print(model_edges[i])
            num_baseline_nodes = MODEL_GRAPH.number_of_nodes()
            num_model_edges = MODEL_GRAPH.number_of_edges()
            print("\nnum_baseline_nodes: %d"%num_baseline_nodes)
            print("num_model_edges: %d"%num_model_edges)
            print("is weighted:")
            print(nx.is_weighted(MODEL_GRAPH))

            #===================== pr =====================
            print("\nGetting pr dicts...")
            if platform == "youtube":
                MODEL_PAGE_RANK_DICT = nx.pagerank_numpy(MODEL_GRAPH, weight="weight")
            else:
                MODEL_PAGE_RANK_DICT = nx.pagerank(MODEL_GRAPH, weight="weight")

            print("Got page ranks!")

            #===================== sort them =====================

            MODEL_PAGE_RANK_DF = pd.DataFrame(list(MODEL_PAGE_RANK_DICT.items()), columns=["node", "page_rank"])
            if MODEL_PAGE_RANK_DF.shape[0] == 0:
                MODEL_PAGE_RANK_DF = pd.DataFrame(data={"node":["n/a"], "page_rank":[0]})
            MODEL_PAGE_RANK_DF = MODEL_PAGE_RANK_DF.sort_values("page_rank", ascending=False).reset_index(drop=True)
            print("\nMODEL_PAGE_RANK_DF")
            print(MODEL_PAGE_RANK_DF)

            model_pr_dist = MODEL_PAGE_RANK_DF["page_rank"].values

            cur_gt_fp = gt_and_baseline_input_dir+"Ground-Truth-Results/" + platform + "/Results-by-Timestep/Timestep-" + str(timestep) + "/" + hyp_infoID + ".csv"

            try:
                gt_pr_dist = pd.read_csv(cur_gt_fp)["page_rank"].values
            except:
                gt_pr_dist = np.asarray([0])

            print("\nGetting EMD...")
            EMD = wasserstein_distance(gt_pr_dist,model_pr_dist)
            print(EMD)
            all_timestep_emds_for_infoID.append(EMD)

            #make cur save dirs
            # model_timestep_output_dir = model_tag_input_dir + model_tag + "/" +platform + "/Results-by-Timestep/Timestep-%d/"%(timestep)

            model_timestep_output_dir = output_dir + "/Page-Ranks-by-Timestep/Timestep-%d/"%(timestep)
            create_output_dir(model_timestep_output_dir)

            model_timestep_output_fp = model_timestep_output_dir + hyp_infoID + ".csv"
            MODEL_PAGE_RANK_DF.to_csv(model_timestep_output_fp, index=False)
            print(model_timestep_output_fp)

        #cur_infoID_emd_results = pd.DataFrame(data={"infoID":[infoID for i in range(num_timesteps)], "EMD":all_timestep_emds_for_infoID, "timestep":[(i+1) for i in range(num_timesteps)]})
        cur_infoID_emd_results = pd.DataFrame(data={"infoID":[infoID for i in range(len(list(timesteps)))], "EMD":all_timestep_emds_for_infoID, "timestep":timesteps})
        cur_infoID_emd_results = cur_infoID_emd_results[["timestep", "infoID", "EMD"]]
        print("\ncur_infoID_emd_results")
        print(cur_infoID_emd_results)

        #main_emd_results_output_dir = model_tag_input_dir + model_tag+ "/"+ platform + "/"
        # main_emd_results_output_dir = model_tag_input_dir

        final_output_dir = output_dir + "EMD-Results/"
        create_output_dir(final_output_dir)
        output_fp = final_output_dir + hyp_infoID + ".csv"
        cur_infoID_emd_results.to_csv(output_fp, index=False)
        print(output_fp)

        print(cur_infoID_emd_results)


    return output_dir