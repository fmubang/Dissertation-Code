
import sys
#sys.path.append("/data/Fmubang/CP4-ORGANIZED-V3-FIX-DROPNA/functions")
# sys.path.append("/beegfs-mnt/data/fmubang/CP5-VAM-Paper-Stuff-3-3/functions")
sys.path.append("/storage2-mnt/data/fmubang/CP4-ORGANIZED-V3-FIX-DROPNA/functions")
sys.path.append("/storage2-mnt/data/fmubang/CP5-VAM-SOTA-1-26-22/p2-ua-exps/p1-tNodeEmbed/p2-tNodeEmbed-main-code/tNodeEmbed/src/")
import pandas as pd
import os,sys
from scipy import stats
import numpy as np
from time import time
from joblib import Parallel, delayed
import multiprocessing
# import vam_ua_funcs as ua
from basic_utils import create_output_dir, convert_df_2_cols_to_dict, save_pickle,hyphenate_infoID_dict,config_df_by_dates,gzip_save,gzip_load
from infoIDs18 import get_cp4_challenge_18_infoIDs
# import network_metric_funcs as nmf
from time import time
import loader
import models as tne_model_utils
import networkx as nx
from node2vec import Node2Vec
from sklearn.model_selection import ParameterGrid
from time import time
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from multiprocessing import Pool, cpu_count
from numpy import linalg as LA

def open_edges_in_parallel(edge_main_input_dir, open_func,NJOBS):
    edge_fps = os.listdir(edge_main_input_dir)
    num_edges = len(edge_fps)
    with Pool(NJOBS) as p:
        edge_df_list = p.map(open_func, [(edge_idx, edge_fp, edge_main_input_dir,num_edges) for edge_idx,edge_fp in enumerate(edge_fps)])

    full_edge_df = pd.concat(edge_df_list).reset_index(drop=True)
    return full_edge_df

def open_edge_fp(args):

    edge_idx, edge_fp, edge_main_input_dir,num_edges = args
    edge_df = pd.read_csv(edge_main_input_dir + edge_fp)

    if edge_idx%1000 == 0:
        print("Opened edge fp %d of %d"%(edge_idx, num_edges))
    return edge_df

class tNE_RegressionDataHandler_LOGREG:

    def __init__(self, snapshot_input_dir,ALIGN_TAG,INSERT_TS_FTS,eval_tag,pred_type ,ts_to_desired_period_tag_dict, infoIDs,x_main_emb_input_dir, y_main_ts_target_input_dir,main_output_dir,param_config_tag ,DESIRED_EVALUATION_TIMESTEP ,
        SAMPLE_LOOKBACK, FT_LOOKBACK,MERGE_METHOD, SCALER_TAG, SAMPLE_RATE, WALK_TYPE, NUM_VAL_PERIODS=7, NUM_TEST_PERIODS=21, NJOBS=32, FT_STATS_FLAG=1):


        # #use this to track edge info
        # self.train_found_count = 0
        # self.train_missing_count = 0
        # self.eval_found_count = 0
        # self.eval_missing_count = 0

        self.snapshot_input_dir = snapshot_input_dir

        self.ts_track_dict = {"train_found":0, "train_missing":0, eval_tag+"_found":0, eval_tag+"_missing":0}
        print()
        print(self.ts_track_dict)

        self.INSERT_TS_FTS = INSERT_TS_FTS
        self.FT_STATS_FLAG = FT_STATS_FLAG
        stats_flag_options = [0, 1, 2]
        if FT_STATS_FLAG not in stats_flag_options:
            print("\nError!FT_STATS_FLAG not in stats_flag_options")
            print()
            print(FT_STATS_FLAG)
            print(stats_flag_options)
            sys.exit(0)

        #check merge methods
        MERGE_OPTS = ["avg", "concat", "none"]
        if MERGE_METHOD not in MERGE_OPTS:
            print("\nError! %s is not in %s"%(MERGE_METHOD, str(MERGE_OPTS)))
            sys.exit(0)
        self.MERGE_METHOD = MERGE_METHOD

        self.pred_type = pred_type
        self.ALIGN_TAG = ALIGN_TAG

        #set tag
        if eval_tag == "val":
            self.NUM_EVAL_PERIODS = NUM_VAL_PERIODS
            eval_period_tag = "Input-Snapshot-for-Output-TS-%d-of-%d"%(DESIRED_EVALUATION_TIMESTEP, self.NUM_EVAL_PERIODS)
        elif eval_tag == "test":
            eval_period_tag = "Input-Snapshot-for-Output-TS-%d-of-%d"%(DESIRED_EVALUATION_TIMESTEP, self.NUM_EVAL_PERIODS)
            self.NUM_EVAL_PERIODS = NUM_TEST_PERIODS
        else:
            print("\nError! %s is not an accepted eval tag"%eval_tag)
            sys.exit(0)

        self.ts_to_desired_period_tag_dict = ts_to_desired_period_tag_dict
        self.eval_period_tag = eval_period_tag
        self.eval_tag = eval_tag
        self.infoIDs = infoIDs

        self.y_main_ts_target_input_dir = y_main_ts_target_input_dir
        self.x_main_emb_input_dir = x_main_emb_input_dir

        self.main_output_dir = main_output_dir
        self.param_config_tag = param_config_tag
        self.DESIRED_EVALUATION_TIMESTEP = DESIRED_EVALUATION_TIMESTEP
        self.SAMPLE_LOOKBACK = SAMPLE_LOOKBACK
        self.FT_LOOKBACK = FT_LOOKBACK
        self.SCALER_TAG = SCALER_TAG
        self.SAMPLE_RATE = SAMPLE_RATE
        self.WALK_TYPE = WALK_TYPE
        self.NJOBS = NJOBS

        #get train period tags
        train_period_tags = []
        for cur_lookback in range(1, SAMPLE_LOOKBACK+1):
            cur_train_tag = "Input-Snapshot-for-Output-TS-%d-of-%d"%(DESIRED_EVALUATION_TIMESTEP-cur_lookback, self.NUM_EVAL_PERIODS)
            train_period_tags.append(cur_train_tag)
        self.train_period_tags = train_period_tags

        print("\neval_period_tag")
        print(eval_period_tag)

        print("\nAll sample train tags")
        for t in self.train_period_tags:
            print(t)



        # sys.exit(0)

    def load_snapshot_df(self,cur_lb_ss_input_dir,data_period_tag):

        all_dfs = []
        for infoID in self.infoIDs:
            hyp_infoID = infoID.replace("/","_")
            cur_fp = cur_lb_ss_input_dir + hyp_infoID + "/" + data_period_tag + ".csv"
            df = pd.read_csv(cur_fp)
            all_dfs.append(df)
        full_ss = pd.concat(all_dfs).reset_index(drop=True)

        return full_ss

    # #def load_bc_x_and_y_node2vec_arrays(self, n2v_model,  infoIDs, y_output_ts_main_input_dir, desired_val_period_tag, data_type,UNDER_SAMPLE_NEG_CLASS):
    # def load_n2v_model(self, data_period_tag):

    #     #just to make things easier
    #     TRUE_FT_LOOKBACK =self.FT_LOOKBACK - 1

    #     #first we need to extract the current sample timestep
    #     print("\ndata_period_tag")
    #     print(data_period_tag)

    #     cur_sample_output_ts = int(data_period_tag.split("-")[5])
    #     cur_sample_input_ts = cur_sample_output_ts - 1
    #     print("\ncur_sample_input_ts")
    #     print(cur_sample_input_ts)

    #     # sys.exit(0)
    #     # CUR_SAMPLE_TIMESTEP =

    #     #get ft loockback tags
    #     ft_lookback_tags = []
    #     for cur_lookback in range(1, TRUE_FT_LOOKBACK+1):
    #         cur_ft_lb_tag = "Input-Snapshot-for-Output-TS-%d-of-%d"%(cur_sample_input_ts-cur_lookback, self.NUM_EVAL_PERIODS)
    #         ft_lookback_tags.append(cur_ft_lb_tag)

    #     print("data_period_tag")
    #     print(data_period_tag)
    #     print("\nAll ft lookback tags for input ts: %d"%cur_sample_input_ts)
    #     for t in ft_lookback_tags:
    #         print(t)

    #     #main dir
    #     #you'll have to fix this when you start using other timesteps
    #     cur_n2v_input_dir =self.x_main_emb_input_dir + self.ALIGN_TAG + "/" + self.WALK_TYPE  + "/"+ self.eval_tag +"/"+ data_period_tag + "/"
    #     print()
    #     print(cur_n2v_input_dir)

    #     #load main dict
    #     print("\nLoading main n2v dict...")
    #     main_n2v_dict = gzip_load(cur_n2v_input_dir + "node2vec_dict")

    #     #get dims
    #     for user,emb in main_n2v_dict.items():
    #         dims = emb.shape[0]
    #         break

    #     if self.FT_STATS_FLAG > 0:
    #         cur_stats_output_dir = self.main_output_dir +  "FT-Lookback-Log/" + data_period_tag + "/"
    #         create_output_dir(cur_stats_output_dir)

    #     main_user_list = []
    #     main_ts_mag_list = []
    #     main_user_embs = []

    #     #use to track zeros later
    #     user_to_zero_tracker = {}
    #     for main_user,main_emb in main_n2v_dict.items():
    #         main_user_list.append(main_user)
    #         main_user_embs.append(main_emb)
    #         user_to_zero_tracker[main_user] = 0

    #     main_user_embs = np.asarray(main_user_embs)
    #     print("\nmain_user_embs shape")
    #     print(main_user_embs.shape)

    #     mags = LA.norm(main_user_embs, axis=1)



    #     main_user_df = pd.DataFrame(data={"main_user":main_user_list})
    #     main_user_df["magnitude"] = mags
    #     print()
    #     main_user_df = main_user_df.sort_values("magnitude", ascending=False).reset_index(drop=True)
    #     print(main_user_df)






    #     #get lookback models
    #     print("\nGetting lookback models")
    #     for idx,cur_ft_lb_tag in enumerate(ft_lookback_tags):
    #         print()
    #         print(cur_ft_lb_tag)
    #         cur_ft_lb_input_dir = self.x_main_emb_input_dir + self.ALIGN_TAG + "/" + self.WALK_TYPE  + "/"+ self.eval_tag +"/"+ cur_ft_lb_tag + "/"
    #         cur_lb_n2v_dict = gzip_load(cur_ft_lb_input_dir + "node2vec_dict")

    #                     #get associated snapshot lookback df


    #         #load
    #         main_user_list = []
    #         cur_new_combined_embs = []
    #         # new_n2v_dict = {}
    #         for main_user,main_emb in main_n2v_dict.items():


    #             try:
    #                 prev_user_emb = cur_lb_n2v_dict[main_user]
    #                 # cur_new_combined_embs.append(prev_user_emb)
    #             except KeyError:
    #                 # cur_new_combined_embs.append()
    #                 prev_user_emb = np.zeros(dims)
    #                 user_to_zero_tracker[main_user]+=1

    #             #concat
    #             new_emb = np.concatenate((main_emb, prev_user_emb))
    #             cur_new_combined_embs.append(new_emb)
    #             main_user_list.append(main_user)

    #             #update dict
    #             main_n2v_dict[main_user] = new_emb

    #         #check
    #         print("\nNew new_emb shape")
    #         print(new_emb.shape)
    #         new_emb = main_n2v_dict[main_user]
    #         correct_dim_size = dims * (idx+2)
    #         print()
    #         print(new_emb.shape[0])
    #         print(correct_dim_size)
    #         if new_emb.shape[0] != correct_dim_size:
    #             print("\nError! new_emb.shape[0] != correct_dim_size!")
    #             sys.exit(0)


    #         cur_new_combined_embs = np.asarray(cur_new_combined_embs)
    #         print("\ncur_new_combined_embs")
    #         print(cur_new_combined_embs.shape)
    #         mags = LA.norm(cur_new_combined_embs, axis=1)
    #         cur_lb_df = pd.DataFrame(data={"main_user":main_user_list, "magnitude_%s"%(cur_ft_lb_tag):mags})

    #         #update
    #         old_size = main_user_df.shape[0]
    #         main_user_df = main_user_df.merge(cur_lb_df, on="main_user",how="inner").reset_index(drop=True)
    #         new_size = main_user_df.shape[0]
    #         print()
    #         print(new_size)
    #         print(old_size)

    #         if new_size != old_size:
    #             print("\nError! new_size != old_size")
    #             sys.exit(0)

    #         print()
    #         print(main_user_df)

    #     #name columns properly
    #     cols = list(main_user_df)
    #     cols.remove("main_user")
    #     for col in cols:
    #         main_user_df = main_user_df.rename(columns={col: "concat_" + col})

    #     if self.MERGE_METHOD == "avg":
    #         print("\nAveraging embs")
    #         print("\ncur_new_combined_embs shape")
    #         print(cur_new_combined_embs.shape)

    #         #reshape
    #         cur_new_combined_embs = cur_new_combined_embs.reshape((cur_new_combined_embs.shape[0], self.FT_LOOKBACK, dims))

    #         print("\nreshaped cur_new_combined_embs shape")
    #         print(cur_new_combined_embs.shape)

    #         cur_new_combined_embs = cur_new_combined_embs.mean(axis=1)

    #         print("\naveraged cur_new_combined_embs shape")
    #         print(cur_new_combined_embs.shape)

    #         #update dict
    #         print("\nUpdating dict with avg's...")
    #         main_n2v_dict = {}
    #         for i,main_user in enumerate(main_user_list):
    #             main_n2v_dict[main_user] = cur_new_combined_embs[i]
    #         print("Done!")

    #         #get mags
    #         mags = LA.norm(cur_new_combined_embs, axis=1)
    #         cur_lb_df = pd.DataFrame(data={"main_user":main_user_list, "avg_magnitude":mags})

    #         #update
    #         old_size = main_user_df.shape[0]
    #         main_user_df = main_user_df.merge(cur_lb_df, on="main_user",how="inner").reset_index(drop=True)
    #         new_size = main_user_df.shape[0]
    #         print()
    #         print(new_size)
    #         print(old_size)

    #         if new_size != old_size:
    #             print("\nError! new_size != old_size")
    #             sys.exit(0)



    #     main_user_df["zero_count"] = main_user_df["main_user"].map(user_to_zero_tracker)

    #     print()
    #     print(main_user_df)
    #     cols = list(main_user_df)
    #     cols.remove("main_user")

    #     if self.FT_STATS_FLAG > 0:
    #         #save value counts
    #         for col in cols:
    #             main_user_df[col] = main_user_df[col].round(0).astype("int32")
    #             val_counts = main_user_df[col].value_counts()
    #             print()
    #             print(val_counts)

    #             vdf = val_counts.rename_axis('unique_values').reset_index(name='counts')

    #             print()
    #             print(vdf)

    #             output_fp = cur_stats_output_dir + col + ".csv"
    #             vdf.to_csv(output_fp, index=False)
    #             print(output_fp)


    #     return main_n2v_dict

    def fix_ydf_user_cols(self, ydf, infoID):

        #fix nodeUserIDs
        infoID_concat_tag = "-with-"
        edges = list(ydf["edge"])
        new_children = []
        new_parents = []
        for e in edges:
            edge_list = e.split("-with-")
            child = edge_list[0] + infoID_concat_tag + infoID
            parent = edge_list[1] + infoID_concat_tag + infoID
            new_children.append(child)
            new_parents.append(parent)
        #fix users
        ydf["nodeUserID"] = new_children
        ydf["parentUserID"] = new_parents

        return ydf

    def convert_ydf_to_edge2ts_dict(self, ydf):

        edge2ts_fts_dict = {}
        for idx,row in ydf.iterrows():
            child = row["nodeUserID"]
            parent = row["parentUserID"]
            ts_fts = [row["ts_%d"%i] for i in range(1,25)]
            cur_edge = (child,parent)
            edge2ts_fts_dict[cur_edge] = ts_fts
            # print()
            # print(cur_edge)
            # print(ts_fts)

            # sys.exit(0)

        return edge2ts_fts_dict



    def load_tNE_reg_data(self, data_type):

        #data type can either be "train" or "eval"
        if data_type == "train":
            data_period_tags = list(self.train_period_tags)
        else:
            data_period_tags = [self.eval_period_tag]

        print("\ndata_period_tags")
        data_period_tags.reverse()
        for dp in data_period_tags:
            print(dp)

        #============================ initial prev data ================================
        #get y tag
        initial_ts = int(data_period_tags[0].split("-")[5])
        print("\ninitial_ts")
        print(initial_ts)



        if self.INSERT_TS_FTS == True:
            #get prev timestep fts
            prev_y_tag =  "Tx-%d-Ty-%d-of-%d"%(initial_ts-2, initial_ts-1, self.NUM_EVAL_PERIODS)
            print("\nprev_y_tag")
            print(prev_y_tag)



            # sys.exit(0)

            infoID_to_prev_edge2ts_dict = {}
            print("\nGetting prev ts edge info")
            for infoID in self.infoIDs:
                hyp_infoID = infoID.replace("/", "_")
                cur_y_ts_input_dir = self.y_main_ts_target_input_dir + self.eval_tag +"/"+    hyp_infoID + "/"
                prev_ydf = open_edges_in_parallel(cur_y_ts_input_dir + prev_y_tag + "/", open_edge_fp,self.NJOBS)
                print()
                print(infoID)
                prev_ydf = self.fix_ydf_user_cols(prev_ydf, infoID)
                print(prev_ydf)
                infoID_to_prev_edge2ts_dict[infoID] = self.convert_ydf_to_edge2ts_dict(prev_ydf)

                # sys.exit(0)

            #get fts



        all_x_arrays = []
        all_y_arrays = []

        for data_period_tag in data_period_tags:


            #n2v model loaded here
            if self.WALK_TYPE != "None":
                print("\nLoading n2v model...")
                # n2v_model = self.load_n2v_model(data_period_tag)
                # print("Got n2v model!")

                #you'll have to fix this when you start using other timesteps
                cur_n2v_input_dir =self.x_main_emb_input_dir + self.ALIGN_TAG + "/" + self.WALK_TYPE  + "/"+ self.eval_tag +"/"+ data_period_tag + "/"
                print()
                print(cur_n2v_input_dir)

                #load main dict
                print("\nLoading main n2v dict...")
                n2v_model = gzip_load(cur_n2v_input_dir + "node2vec_dict")
                print("Got n2v model!")

            cur_lb_ss_input_dir = self.snapshot_input_dir + "/" + self.eval_tag +"/"
            cur_ss_df = self.load_snapshot_df(cur_lb_ss_input_dir,data_period_tag)
            print("\ncur_ss_df")
            print(cur_ss_df)


            cur_ss_df["nodeUserID"] = cur_ss_df["nodeUserID"] + "-with-" + cur_ss_df["informationID"]
            cur_ss_df["parentUserID"] = cur_ss_df["parentUserID"] + "-with-" + cur_ss_df["informationID"]

            # #get y tag
            cur_ts = int(data_period_tag.split("-")[5])
            print("\ncur_ts")
            print(cur_ts)

            #y_tag = self.ts_to_desired_period_tag_dict[cur_ts]
            y_tag = "Tx-%d-Ty-%d-of-%d"%(cur_ts-1, cur_ts, self.NUM_EVAL_PERIODS)
            print(y_tag)

            for infoID in self.infoIDs:

                print("\ncur_ss_df")
                print(cur_ss_df)

                cur_infoID_ss_df = cur_ss_df[cur_ss_df["informationID"]==infoID].reset_index(drop=True)

                print("\ncur_infoID_ss_df")
                print(cur_infoID_ss_df)

                # sys.exit(0)

                #load data
                hyp_infoID = infoID.replace("/", "_")

                cur_y_ts_input_dir = self.y_main_ts_target_input_dir + self.eval_tag +"/"+    hyp_infoID + "/"
                #ydf = pd.read_csv(cur_y_ts_input_dir + y_tag + ".csv")
                full_ydf = open_edges_in_parallel(cur_y_ts_input_dir + y_tag + "/", open_edge_fp,self.NJOBS)

                print()
                print(full_ydf)

                #fix nodeUserIDs
                infoID_concat_tag = "-with-"
                edges = list(full_ydf["edge"])
                new_children = []
                new_parents = []
                for e in edges:
                    edge_list = e.split("-with-")
                    child = edge_list[0] + infoID_concat_tag + infoID
                    parent = edge_list[1] + infoID_concat_tag + infoID
                    new_children.append(child)
                    new_parents.append(parent)
                #fix users
                full_ydf["nodeUserID"] = new_children
                full_ydf["parentUserID"] = new_parents

                print()
                print(full_ydf)

                # sys.exit(0)
                cols = list(full_ydf)
                print()
                for col in cols:
                    print(col)

                #make sure all 1s
                print()
                print(full_ydf["edge"].value_counts())
                print(full_ydf)

                #hack for now
                NUM_TIMESTEPS = 24
                allowed_ts = ["ts_" + str(i) for i in range(1, NUM_TIMESTEPS+1)]
                allowed_cols = ["edge", "nodeUserID", "parentUserID","input_daily_date","output_daily_date"]
                full_ydf = full_ydf[allowed_ts+ allowed_cols]
                full_ydf["total_edge_activities"] = full_ydf[allowed_ts].sum(axis=1)

                cols = list(full_ydf)
                for col in cols:
                    print(col)

                full_ydf["is_active"] = [1 if a>0 else 0 for a in full_ydf["total_edge_activities"]]
                print(full_ydf["is_active"].value_counts())



                #sample
                if data_type == "train":
                    if self.SAMPLE_RATE != "None":
                        try:
                            nz_samples = full_ydf[full_ydf["is_active"]==1]
                            zero_samples = full_ydf[full_ydf["is_active"]==0].reset_index(drop=True)
                            print()
                            DESIRED_NUM_ZERO_SAMPLES = int(int(self.SAMPLE_RATE) * nz_samples.shape[0])
                            print(DESIRED_NUM_ZERO_SAMPLES)
                            zero_samples = zero_samples.sample(n=DESIRED_NUM_ZERO_SAMPLES, replace=False).reset_index(drop=True)
                            full_ydf = pd.concat([nz_samples, zero_samples]).reset_index(drop=True)
                            print("\nnz_samples")
                            print(nz_samples)
                            print("\nzero_samples")
                            print(zero_samples)
                        except ValueError:
                            print()

                # sys.exit(0)


                # full_ydf = full_ydf[["nodeUserID", "parentUserID", "is_active"]]
                print()
                print(full_ydf)

                #y_array = full_ydf["is_active"].values



                # links = []


                print()
                print(full_ydf[["nodeUserID", "parentUserID"]])

                print()
                print(cur_infoID_ss_df[["nodeUserID", "parentUserID"]])




                #filter full_ydf
                print("\nsize full_ydf before ss df chop")
                print(full_ydf.shape)
                print("\nSize after")
                ydf = pd.merge(full_ydf,cur_infoID_ss_df, on=["nodeUserID", "parentUserID"], how="right").reset_index(drop=True)
                ydf = ydf.fillna(0)
                print(ydf.shape)

                #look for double tag
                double_tag= "-with-" + infoID + "-with-" + infoID
                double_tag_users = [user for user in ydf["nodeUserID"] if double_tag in user]
                double_tag_users = double_tag_users + [user for user in ydf["parentUserID"] if double_tag in user]
                dt_df = pd.DataFrame(data={"dt_user":double_tag_users})
                print()
                print(dt_df)
                double_tag_users = set(double_tag_users)
                print("\ndouble_tag_users")
                dt_num  = len(double_tag_users)
                print(dt_num)



                ko_fp = self.main_output_dir + data_type + "-kickout-user-info.txt"
                print(ko_fp)
                with open(ko_fp, "w") as f:
                    f.write(str(dt_num))

                dt_df.to_csv(self.main_output_dir + data_type + "-double_tag_users.csv", index=False)

                ydf = ydf[~ydf["nodeUserID"].isin(double_tag_users)]
                ydf = ydf[~ydf["parentUserID"].isin(double_tag_users)]

                print()
                print("\nafter kickout double tag users")
                print(ydf)

                if self.pred_type == "binary_classification":
                    y_array = ydf["is_active"].values
                elif self.pred_type == "multioutput_regression":
                    y_array = ydf[allowed_ts].values

                all_y_arrays.append(y_array)



                idx=1
                num_links = ydf.shape[0]
                MOD_NUM = 100
                for child,parent in zip(ydf["nodeUserID"], ydf["parentUserID"]):

                    child = str(child)
                    parent = str(parent)

                    # print(child,parent)

                    # sys.exit(0)
                    if self.WALK_TYPE != "None":
                        link = np.concatenate(np.asarray([n2v_model[child], n2v_model[parent]]), axis=None)
                    else:
                        link = np.asarray([])

                    if self.INSERT_TS_FTS == True:
                        #get ts fts
                        try:
                            edge_ts_fts = infoID_to_prev_edge2ts_dict[infoID][(child, parent)]
                            self.ts_track_dict[data_type + "_found"]+=1
                        except KeyError:
                            self.ts_track_dict[data_type + "_missing"]+=1
                            edge_ts_fts = np.zeros(24)
                        # print()
                        # print("\nLink only ft shape")
                        # print(link.shape)
                        link = np.concatenate(np.asarray([link,edge_ts_fts]), axis=None)
                        # print("\nLink with ts fts shape")
                        # print(link.shape)



                    # infoID_to_prev_edge2ts_dict[infoID] = self.convert_ydf_to_edge2ts_dict(prev_ydf)


                    all_x_arrays.append(link)

                    if idx%MOD_NUM == 0:
                        print("Got link %d of %d"%(idx, num_links))
                    idx+=1

                if self.INSERT_TS_FTS == True:

                    #update prev dict
                    infoID_to_prev_edge2ts_dict[infoID]=self.convert_ydf_to_edge2ts_dict(full_ydf)

            print("\nlink shape")
            print(link.shape)

        x_array = np.asarray(all_x_arrays)
        print("\nx_array shape")
        print(x_array.shape)

        y_array = np.concatenate(all_y_arrays, axis=0)
        print("\ny_array shape")
        print(y_array.shape)

        #save data info
        shape_info_fp = self.main_output_dir + data_type + "-xy-shape-info.txt"
        with open(shape_info_fp, "w") as f:
            f.write(str(x_array.shape))
            f.write(str(y_array.shape))

        print("\nself.ts_track_dict")
        print(self.ts_track_dict)

        with open(self.main_output_dir + "ts_track_dict.txt", "w") as f:
            f.write(str(self.ts_track_dict))

        return x_array,y_array

    def normalize_x_train_and_eval_arrays(self, x_train, x_eval):

        if self.SCALER_TAG == "None":
            self.scaler = None
            return x_train,x_eval

        elif self.SCALER_TAG == "MinMax":
            scaler = MinMaxScaler()
        elif self.SCALER_TAG == "Standard":
            scaler = StandardScaler()
        else:
            print("%s is not a valid scaler type!"%self.SCALER_TAG)
            sys.exit(0)

        scaler.fit(x_train)
        # print("\nscaler.data_max_")
        # print(scaler.data_max_)

        print("\nmin and max before and after")
        print(x_train.min(), x_train.max())
        print(x_eval.min(), x_eval.max())
        print(x_train.mean(), x_eval.mean())

        x_train = scaler.transform(x_train)
        x_eval = scaler.transform(x_eval)

        print(x_train.min(), x_train.max())
        print(x_eval.min(), x_eval.max())
        print(x_train.mean(), x_eval.mean())

        # sys.exit(0)

        self.scaler = scaler

        return x_train,x_eval