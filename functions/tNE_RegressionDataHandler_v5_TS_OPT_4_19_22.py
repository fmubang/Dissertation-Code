
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

import multiprocessing as mp

from sample_gen_aux_funcs import *
from basic_utils import *
from multiprocessing import Pool

def open_edges_in_parallel(edge_main_input_dir, open_func,NJOBS):
    edge_fps = os.listdir(edge_main_input_dir)
    num_edges = len(edge_fps)
    with Pool(NJOBS) as p:
        edge_df_list = p.map(open_func, [(edge_idx, edge_fp, edge_main_input_dir,num_edges) for edge_idx,edge_fp in enumerate(edge_fps)])

    # NoneType = type(None)
    # fixed_edge_df_list = [e for e in edge_df_list if type(e)!= NoneType]
    # edge_df_list = fixed_edge_df_list

    full_edge_df = pd.concat(edge_df_list).reset_index(drop=True)
    return full_edge_df

def open_edge_fp(args):

    edge_idx, edge_fp, edge_main_input_dir,num_edges = args
    #print(edge_main_input_dir +edge_fp)
    # edge_df = pd.read_csv(edge_main_input_dir + edge_fp)



    if os.stat(edge_main_input_dir+edge_fp).st_size == 0:
        print("Couldn't open: %s"%(edge_main_input_dir+edge_fp))
        edge_dir_split = edge_main_input_dir.split("/")
        hyp_infoID = edge_dir_split[10]
        dp_tag = edge_dir_split[11]
        # hyp_infoID = infoID.replace("/","_")

        edge = edge_fp.replace(".csv", "")
        output_tag = edge + "-" + hyp_infoID + "-" + dp_tag
        error_output_dir = "Could-Not-Open/"
        create_output_dir(error_output_dir)
        with open(error_output_dir + output_tag + ".txt", "w") as f:
            f.write(edge_fp)
        return
    else:
        edge_df = pd.read_csv(edge_main_input_dir + edge_fp)


    # try:
    #     edge_df = pd.read_csv(edge_main_input_dir + edge_fp)
    # except:
    #     print("\nCouldn't open:")
    #     print(edge_main_input_dir + edge_fp)
    #     return

    if edge_idx%1000 == 0:
        print("Opened edge fp %d of %d"%(edge_idx, num_edges))
    return edge_df

class tNE_RegressionDataHandler:

    def __init__(self,snapshot_input_dir,INSERT_TS_FTS,ALIGN_TAG,eval_tag,pred_type ,infoIDs,x_main_emb_input_dir, y_main_ts_target_input_dir,main_output_dir,DESIRED_EVALUATION_TIMESTEP ,
        SAMPLE_LOOKBACK, SCALER_TAG, SAMPLE_RATE, WALK_TYPE,output_types, NUM_VAL_PERIODS=7, NUM_TEST_PERIODS=21, NJOBS=32):

        #setup basic params
        self.pred_type = pred_type
        self.ALIGN_TAG = ALIGN_TAG
        self.NUM_VAL_PERIODS = NUM_VAL_PERIODS
        self.NUM_TEST_PERIODS = NUM_TEST_PERIODS

        self.snapshot_input_dir = snapshot_input_dir

        self.ts_track_dict = {"train_found":0, "train_missing":0, eval_tag+"_found":0, eval_tag+"_missing":0}
        print()
        print(self.ts_track_dict)

        self.INSERT_TS_FTS = INSERT_TS_FTS

        #set tag
        if eval_tag == "val":
            self.NUM_EVAL_PERIODS = self.NUM_VAL_PERIODS
            eval_period_tag = "Input-Snapshot-for-Output-TS-%d-of-%d"%(DESIRED_EVALUATION_TIMESTEP,self.NUM_EVAL_PERIODS)
        elif eval_tag == "test":
            self.NUM_EVAL_PERIODS = self.NUM_TEST_PERIODS
            eval_period_tag = "Input-Snapshot-for-Output-TS-%d-of-%d"%(DESIRED_EVALUATION_TIMESTEP, self.NUM_EVAL_PERIODS)
        else:
            print("\nError! %s is not an accepted eval tag"%eval_tag)
            sys.exit(0)
        print("\neval_period_tag")
        print(eval_period_tag)

        #more params
        self.output_types = output_types
        self.eval_period_tag = eval_period_tag
        self.eval_tag = eval_tag
        self.infoIDs = infoIDs
        self.y_main_ts_target_input_dir = y_main_ts_target_input_dir
        self.x_main_emb_input_dir = x_main_emb_input_dir
        self.main_output_dir = main_output_dir
        self.DESIRED_EVALUATION_TIMESTEP = DESIRED_EVALUATION_TIMESTEP
        SAMPLE_LOOKBACK = int(SAMPLE_LOOKBACK)
        self.SAMPLE_LOOKBACK = SAMPLE_LOOKBACK
        self.SCALER_TAG = SCALER_TAG
        self.SAMPLE_RATE = SAMPLE_RATE
        self.WALK_TYPE = WALK_TYPE
        self.NJOBS = NJOBS

        self.data_array_dict = {}
        self.data_info_dict= {}

        print()
        print(SAMPLE_LOOKBACK)

        #better version for later
        train_period_tuples = []
        print()
        for cur_lookback in range(1, SAMPLE_LOOKBACK+2):
            CUR_DESIRED_TS = DESIRED_EVALUATION_TIMESTEP-cur_lookback
            print(CUR_DESIRED_TS)
            if (CUR_DESIRED_TS >= 1) and (self.eval_tag=="test"):
                cur_train_tag = "Input-Snapshot-for-Output-TS-%d-of-%d"%(CUR_DESIRED_TS, self.NUM_EVAL_PERIODS)
                train_tuple = ("test", cur_train_tag)
            elif (CUR_DESIRED_TS >= 1) and (self.eval_tag=="val"):
                cur_train_tag = "Input-Snapshot-for-Output-TS-%d-of-%d"%(CUR_DESIRED_TS, self.NUM_EVAL_PERIODS)
                train_tuple = ("val", cur_train_tag)
            else:
                CUR_DESIRED_TS = self.NUM_VAL_PERIODS + CUR_DESIRED_TS
                cur_train_tag = "Input-Snapshot-for-Output-TS-%d-of-%d"%(CUR_DESIRED_TS, self.NUM_EVAL_PERIODS)
                train_tuple = ("val", cur_train_tag)
            train_period_tuples.append(train_tuple)
        self.train_period_tuples = train_period_tuples

        self.init_prev_ts_tuple = self.train_period_tuples.pop()
        # self.train_period_tuples=self.train_period_tuples[1:]
        print("\nself.init_prev_ts_tuple")
        print(self.init_prev_ts_tuple)

        print("\nAll tuples")
        for t in self.train_period_tuples:
            print(t)



        # sys.exit(0)

        # #make init prev tuple
        # init_ts_prev_tuple =

        # sys.exit(0)

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

    def load_snapshot_df(self,cur_lb_ss_input_dir,data_period_tag):

        all_dfs = []
        for infoID in self.infoIDs:
            hyp_infoID = infoID.replace("/","_")
            cur_fp = cur_lb_ss_input_dir + hyp_infoID + "/" + data_period_tag + ".csv"
            df = pd.read_csv(cur_fp)
            all_dfs.append(df)
        full_ss = pd.concat(all_dfs).reset_index(drop=True)

        return full_ss

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

    #def load_bc_x_and_y_node2vec_arrays(self, n2v_model,  infoIDs, y_output_ts_main_input_dir, desired_val_period_tag, data_type,UNDER_SAMPLE_NEG_CLASS):
    def load_tNE_reg_data(self, data_type,GET_TUPLE_INFO_DICT=False):

        #data type can either be "train" or "eval"
        if data_type == "train":
            data_period_tuples = list(self.train_period_tuples)
        else:
            data_period_tuples = [(self.eval_tag ,self.eval_period_tag)]

        print("\ndata_period_tuples")
        data_period_tuples.reverse()
        for dp in data_period_tuples:
            print(dp)

        print()
        for dp in data_period_tuples:
            print(dp)

        # sys.exit(0)

        #============================ initial prev data ================================
        #get y tag
        # initial_ts = int(data_period_tuples[1].split("-")[5])
        # print("\ninitial_ts")
        # print(initial_ts)



        if self.INSERT_TS_FTS == True:

            init_prev_ts_eval_type = self.init_prev_ts_tuple[0]
            init_prev_ts_dp = self.init_prev_ts_tuple[1]

            initial_ts = int(init_prev_ts_dp.split("-")[5])
            print("\ninitial_ts")
            print(initial_ts)



            # prev_num = initial_ts - 1
            if init_prev_ts_eval_type == "val":
                INIT_PREV_PERIODS = self.NUM_VAL_PERIODS
            else:
                INIT_PREV_PERIODS = self.NUM_TEST_PERIODS

            #get prev timestep fts
            prev_y_tag =  "Tx-%d-Ty-%d-of-%d"%(initial_ts-1, initial_ts, INIT_PREV_PERIODS)
            print("\nprev_y_tag")
            print(prev_y_tag)


            infoID_to_prev_edge2ts_dict = {}
            print("\nGetting prev ts edge info")
            for infoID in self.infoIDs:
                hyp_infoID = infoID.replace("/", "_")
                cur_y_ts_input_dir = self.y_main_ts_target_input_dir + init_prev_ts_eval_type +"/"+    hyp_infoID + "/"
                prev_ydf = open_edges_in_parallel(cur_y_ts_input_dir + prev_y_tag + "/", open_edge_fp,self.NJOBS)
                print()
                print(infoID)
                prev_ydf = self.fix_ydf_user_cols(prev_ydf, infoID)
                print(prev_ydf)
                infoID_to_prev_edge2ts_dict[infoID] = self.convert_ydf_to_edge2ts_dict(prev_ydf)

        #for arrats
        all_x_arrays = []
        all_y_arrays = []
        array_info_tuples = []

        for data_period_tuple in data_period_tuples:

            #eval info
            cur_eval_tag = data_period_tuple[0]
            data_period_tag = data_period_tuple[1]

            if self.WALK_TYPE != "None":

                #dir
                cur_n2v_input_dir = self.x_main_emb_input_dir + self.ALIGN_TAG + "/"+ self.WALK_TYPE + "/"+ cur_eval_tag + "/"+ data_period_tag + "/"
                print()
                print(cur_n2v_input_dir)

                #get model
                print("\nLoading n2v model...")
                n2v_model = gzip_load(cur_n2v_input_dir + "node2vec_dict")
                # print(n2v_model)

            #snapshot stuff
            cur_lb_ss_input_dir = self.snapshot_input_dir + "/" + cur_eval_tag +"/"
            cur_ss_df = self.load_snapshot_df(cur_lb_ss_input_dir,data_period_tag)
            print("\ncur_ss_df")
            print(cur_ss_df)
            cur_ss_df["nodeUserID"] = cur_ss_df["nodeUserID"] + "-with-" + cur_ss_df["informationID"]
            cur_ss_df["parentUserID"] = cur_ss_df["parentUserID"] + "-with-" + cur_ss_df["informationID"]

            #get y tag
            cur_ts = int(data_period_tag.split("-")[5])
            print("\ncur_ts")
            print(cur_ts)

            #good for future mistakes
            if cur_eval_tag == "val":
                y_tag = "Tx-%d-Ty-%d-of-%d"%(cur_ts-1, cur_ts, self.NUM_VAL_PERIODS)
            elif cur_eval_tag == "test":
                y_tag = "Tx-%d-Ty-%d-of-%d"%(cur_ts-1, cur_ts, self.NUM_TEST_PERIODS)
            else:
                print("\nError! cur_eval_tag must be val or test!")
                print(cur_eval_tag)
                sys.exit(0)
            print(y_tag)

            #loop per infoID
            for infoID in self.infoIDs:

                #load data
                hyp_infoID = infoID.replace("/", "_")

                print("\ncur_ss_df")
                print(cur_ss_df)

                cur_infoID_ss_df = cur_ss_df[cur_ss_df["informationID"]==infoID].reset_index(drop=True)

                print("\ncur_infoID_ss_df")
                print(cur_infoID_ss_df)

                #load data
                cur_y_ts_input_dir = self.y_main_ts_target_input_dir + cur_eval_tag + "/" +  hyp_infoID + "/"
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

                #merge
                ydf = pd.merge(full_ydf,cur_infoID_ss_df, on=["nodeUserID", "parentUserID"], how="inner").reset_index(drop=True)

                print()
                print(ydf)

                # sys.exit(0)
                cols = list(ydf)
                print()
                for col in cols:
                    print(col)

                #make sure all 1s
                print()
                print(ydf["edge"].value_counts())
                print(ydf)

                #hack for now
                NUM_TIMESTEPS = 24
                allowed_ts = ["ts_" + str(i) for i in range(1, NUM_TIMESTEPS+1)]
                allowed_cols = ["edge", "nodeUserID", "parentUserID","input_daily_date","output_daily_date"]
                ydf = ydf[allowed_ts+ allowed_cols]
                ydf["total_edge_activities"] = ydf[allowed_ts].sum(axis=1)

                cols = list(ydf)
                for col in cols:
                    print(col)

                #convert to classes
                ydf["is_active"] = [1 if a>0 else 0 for a in ydf["total_edge_activities"]]
                print(ydf["is_active"].value_counts())

                #sample
                if data_type == "train":
                    if self.SAMPLE_RATE != "None":
                        self.SAMPLE_RATE = float(self.SAMPLE_RATE)
                        try:
                            nz_samples = ydf[ydf["is_active"]==1]
                            zero_samples = ydf[ydf["is_active"]==0].reset_index(drop=True)
                            print()
                            DESIRED_NUM_ZERO_SAMPLES = int(int(self.SAMPLE_RATE) * nz_samples.shape[0])
                            print(DESIRED_NUM_ZERO_SAMPLES)
                            zero_samples = zero_samples.sample(n=DESIRED_NUM_ZERO_SAMPLES, replace=False).reset_index(drop=True)
                            ydf = pd.concat([nz_samples, zero_samples]).reset_index(drop=True)
                            print("\nnz_samples")
                            print(nz_samples)
                            print("\nzero_samples")
                            print(zero_samples)
                        except ValueError:
                            print()

                print()
                ydf = ydf.fillna(0)
                print(ydf[["nodeUserID", "parentUserID"]])

                print()
                print(cur_infoID_ss_df[["nodeUserID", "parentUserID"]])

                #double tag stuff
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


                #setup pred task
                if self.pred_type == "binary_classification":
                    y_array = ydf["is_active"].values
                elif self.pred_type == "multioutput_regression":
                    y_array = ydf[allowed_ts].values
                all_y_arrays.append(y_array)

                #get info tuple
                info_df = ydf[["nodeUserID", "parentUserID"]]
                info_df["infoID"] = infoID
                info_df["data_period_tag"] = data_period_tag
                print()
                print(info_df)

                #check info tuples
                cur_info_tuples = info_df.to_records(index=False)
                for t_idx in range(10):
                    try:
                        print(cur_info_tuples[t_idx])
                    except IndexError:
                        continue

                #config info tuples
                array_info_tuples = list(array_info_tuples) + list(cur_info_tuples)

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

        print("Size of info tuples")
        print(len(array_info_tuples))
        #make dict
        array_idx_to_info_tuple_dict = {}
        print("\nMaking tuple dict...")
        for t_idx,cur_tuple in enumerate(array_info_tuples):
            array_idx_to_info_tuple_dict[t_idx] = cur_tuple
        for t_idx in range(10):
            print()
            print(t_idx)
            print(array_idx_to_info_tuple_dict[t_idx])

        # sys.exit(0)

        self.data_array_dict["x_%s"%data_type] = np.copy(x_array)
        self.data_array_dict["y_%s"%data_type] = np.copy(y_array)
        self.data_info_dict[data_type] = dict(array_idx_to_info_tuple_dict)

        if GET_TUPLE_INFO_DICT == True:
            return x_array,y_array,array_idx_to_info_tuple_dict
        else:
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

    def save_y_eval_pred_edges(self, y_array,y_tag):

        #get info dict
        info_dict = self.data_info_dict[self.eval_tag]

        #fix pred values
        y_array[y_array<0] = 0
        y_array = np.round(y_array, 0)

        #get infoIDs and sim periods
        edge_infoIDs = []
        # sim_periods = []
        child_list = []
        parent_list = []

        for idx in range(y_array.shape[0]):

            #tuple info
            cur_tuple = info_dict[idx]
            cur_infoID = cur_tuple[2]
            edge_infoIDs.append(cur_infoID)
            # dp_tag = cur_tuple[3]

            #tag
            # filtered_dp = dp_tag.replace("Input-Snapshot-for-Output-TS-","")
            # sim_period = int(filtered_dp.split("-")[0])
            # sim_periods.append(sim_period)
            # old_user_list.append(cur_tuple[0])
            child = cur_tuple[0]
            parent = cur_tuple[1]
            child_list.append(child)
            parent_list.append(parent)

        #fix time cols
        ts_cols = ["ts_%d"%(idx+1) for idx in range(y_array.shape[1])]
        ydf = pd.DataFrame(y_array, columns =ts_cols)
        print()
        ydf["infoID"] = edge_infoIDs
        # ydf["SIM_PERIOD"] = sim_periods
        ydf["child"] = child_list
        ydf["parent"] = parent_list
        print(ydf)

        #save it
        output_fp = self.main_output_dir + y_tag + "-with-edge-info.csv"
        ydf.to_csv(output_fp, index=False)
        print()
        print(output_fp)


        return ydf

    def aggregate_edge_preds_to_volume_preds(self, preds, NEW_USER_DIR=None):

        #setup output types
        output_types = [
            "twitter_platform_infoID_pair_nunique_new_users",
            "twitter_platform_infoID_pair_nunique_old_users",
            "twitter_platform_infoID_pair_num_actions"
        ]

        #name cols
        new_user_col =  "twitter_platform_infoID_pair_nunique_new_users"
        old_user_col = "twitter_platform_infoID_pair_nunique_old_users"
        action_col = "twitter_platform_infoID_pair_num_actions"

        #make pred dict
        pred_dict = {}
        info_dict = self.data_info_dict[self.eval_tag]

        #fix pred values
        preds[preds<0] = 0
        preds = np.round(preds, 0)

        #get infoIDs and sim periods
        edge_infoIDs = []
        sim_periods = []
        old_user_list = []
        for idx in range(preds.shape[0]):

            #tuple info
            cur_tuple = info_dict[idx]
            cur_infoID = cur_tuple[2]
            edge_infoIDs.append(cur_infoID)
            dp_tag = cur_tuple[3]

            #tag
            filtered_dp = dp_tag.replace("Input-Snapshot-for-Output-TS-","")
            sim_period = int(filtered_dp.split("-")[0])
            sim_periods.append(sim_period)
            old_user_list.append(cur_tuple[0])

        #fix time cols
        ts_cols = ["ts_%d"%(idx+1) for idx in range(preds.shape[1])]
        pred_df = pd.DataFrame(preds, columns =ts_cols)
        print()
        pred_df["infoID"] = edge_infoIDs
        pred_df["SIM_PERIOD"] = sim_periods
        pred_df["user"] = old_user_list
        print(pred_df)

        #make action initial df
        action_agg_df = pred_df.copy()
        action_agg_df =action_agg_df.drop("user", axis=1)
        for ts_col in ts_cols:
            action_agg_df[ts_col ] = action_agg_df.groupby(["infoID","SIM_PERIOD"])[ts_col].transform("sum")
        action_agg_df = action_agg_df.drop_duplicates().reset_index(drop=True)
        print()
        print(action_agg_df)

        #make old user initial df
        old_user_agg_df = pred_df.copy()
        print()
        print(old_user_agg_df)

        #add up actions user did in each ts
        for ts_col in ts_cols:
            old_user_agg_df[ts_col ] = old_user_agg_df.groupby(["infoID","SIM_PERIOD","user"])[ts_col].transform("sum")

        #with total actions of each old user, but with dupes
        print("\nWith summed hourly ts with dupes")
        print(old_user_agg_df)

        print("\nNo dupe user records")
        old_user_agg_df = old_user_agg_df.drop_duplicates("user").reset_index(drop=True)
        print(old_user_agg_df)

        #we only need to know if user was active or not
        old_user_agg_df[ts_cols] = old_user_agg_df[ts_cols].clip(upper=1)
        print("\nBinary old df")
        print(old_user_agg_df)

        #now kick out user col
        old_user_agg_df = old_user_agg_df.drop("user", axis=1)

        #now just get active users per ts
        for ts_col in ts_cols:
            old_user_agg_df[ts_col ] = old_user_agg_df.groupby(["infoID","SIM_PERIOD"])[ts_col].transform("sum")
        old_user_agg_df = old_user_agg_df.drop_duplicates().reset_index(drop=True)
        print("\nold_user_agg_df")
        print(old_user_agg_df)

        print("\naction_agg_df")
        print(action_agg_df)

        #now make full dfs
        for infoID in self.infoIDs:
            pred_dict[infoID] = {}
            SIM_PERIODS = old_user_agg_df["SIM_PERIOD"].unique()
            for SIM_PERIOD in SIM_PERIODS:

                old_user_temp_sim_period_df = old_user_agg_df[(old_user_agg_df["infoID"]==infoID) & (old_user_agg_df["SIM_PERIOD"]==SIM_PERIOD)].reset_index(drop=True)
                action_temp_sim_period_df = action_agg_df[(action_agg_df["infoID"]==infoID) & (action_agg_df["SIM_PERIOD"]==SIM_PERIOD)].reset_index(drop=True)
                print()
                print(old_user_temp_sim_period_df)
                print()
                print(action_temp_sim_period_df)

                #get arrays
                actions = action_temp_sim_period_df[ts_cols].values.flatten()
                old_users = old_user_temp_sim_period_df[ts_cols].values.flatten()
                print()
                print(actions)
                print(old_users)

                if NEW_USER_DIR == None:
                    new_users = [0 for t in ts_cols]
                else:
                    hyp_infoID = infoID.replace("/","_")
                    cur_new_user_input_fp = NEW_USER_DIR + hyp_infoID + "/%s-day-%d-of-%d.csv"%(self.eval_tag, SIM_PERIOD, self.NUM_EVAL_PERIODS)
                    print(cur_new_user_input_fp)
                    new_user_df = pd.read_csv(cur_new_user_input_fp)
                    new_user_df = new_user_df.sort_values("timestep").reset_index(drop=True)
                    print()
                    print(new_user_df)
                    new_users = new_user_df["num_new_users"].values
                    new_user_actions = new_user_df["num_new_user_actions"].values
                    actions = actions + new_user_actions


                # print(new_users)

                #make df
                data={new_user_col:new_users, old_user_col:old_users, action_col:actions}
                cur_df = pd.DataFrame(data=data)
                cur_df["timestep"] = [i+1 for i in range(cur_df.shape[0])]
                cur_df = cur_df[["timestep",new_user_col, old_user_col, action_col]]
                pred_dict[infoID][SIM_PERIOD] = cur_df

                print()
                print(cur_df)

                # sys.exit(0)

        return pred_dict

    def get_full_gt_dict(self, full_gt_input_dir):

        new_user_col =  "twitter_platform_infoID_pair_nunique_new_users"
        old_user_col = "twitter_platform_infoID_pair_nunique_old_users"
        action_col = "twitter_platform_infoID_pair_num_actions"

        gt_dict = {}
        SIM_PERIODS = [self.DESIRED_EVALUATION_TIMESTEP]
        for infoID in self.infoIDs:
            hyp_infoID = infoID.replace("/","_")
            gt_dict[infoID] = {}
            for SIM_PERIOD in SIM_PERIODS:
                cur_gt_input_fp = full_gt_input_dir + hyp_infoID + "/%s-day-%d-of-%d.csv"%(self.eval_tag, SIM_PERIOD, self.NUM_EVAL_PERIODS)
                cur_gdf = pd.read_csv(cur_gt_input_fp)
                print()
                print(cur_gdf)

                cur_gdf[new_user_col] = cur_gdf["num_new_users"].copy()
                cur_gdf[old_user_col] = cur_gdf["num_old_users"].copy()
                cur_gdf[action_col] = cur_gdf["num_new_user_actions"] + cur_gdf["num_old_user_actions"]
                cur_gdf = cur_gdf[["timestep", new_user_col, old_user_col, action_col]]
                gt_dict[infoID][SIM_PERIOD] = cur_gdf


        return gt_dict

    def get_para_full_metric_df(self, y_pred_df_dict, y_true_df_dict, DESIRED_METRICS, metric_to_func_dict):

        metric_result_dict = {}

        tuple_idx = 1
        all_arg_tuples = []

        for METRIC in DESIRED_METRICS:

            # infoID_df_list = []
            # output_type_df_list = []
            # sim_period_df_list = []
            # metric_result_df_list = []

            for infoID in self.infoIDs:
                metric_result_dict[infoID] = {}

                y_pred_infoID_dict = y_pred_df_dict[infoID]
                y_true_infoID_dict = y_true_df_dict[infoID]

                num_tuples = len(DESIRED_METRICS)*len(self.infoIDs)*len(y_pred_infoID_dict) * len(self.output_types)

                for SIM_PERIOD, y_pred_df in y_pred_infoID_dict.items():

                    # print()
                    # print(SIM_PERIOD)
                    # print(y_pred_df)

                    y_true_df = y_true_infoID_dict[SIM_PERIOD]

                    for output_type in self.output_types:
                        y_pred = y_pred_df[output_type]
                        y_true = y_true_df[output_type]
                        metric_func = metric_to_func_dict[METRIC]

                        arg_tuple = (METRIC, infoID, SIM_PERIOD, output_type, y_true, y_pred, metric_func, tuple_idx, num_tuples)
                        all_arg_tuples.append(arg_tuple)
                        tuple_idx+=1

        if len(all_arg_tuples) != num_tuples:
            print("\nError! len(all_arg_tuples) != num_tuples")
            print(len(all_arg_tuples))
            print(num_tuples)
            sys.exit(0)

        #multi proc
        global pool
        pool = Pool(self.NJOBS)
        print("\nStarting multiproc...")
        results = pool.map(self.get_mini_metric_df, all_arg_tuples)

        #added 2/17/22 -> prevents memory leak
        pool.close()
        pool.join()


        full_metric_df = pd.concat(results)
        print("\nfull_metric df after multiproc")
        print(full_metric_df)

        return full_metric_df

    def get_mini_metric_df(self, arg_tuple):

        METRIC, infoID, SIM_PERIOD, output_type, y_true, y_pred, metric_func, tuple_idx, num_tuples = arg_tuple
        metric_result = metric_func(y_true, y_pred)

        metric_df = pd.DataFrame(data={"infoID":[infoID], "output_type":[output_type], "SIM_PERIOD":[SIM_PERIOD], "metric":[METRIC],
            "metric_result":[metric_result]})

        MOD = 300
        if tuple_idx%MOD == 0:
            print("Done with mini metric df %d of %d"%(tuple_idx, num_tuples))

        return metric_df

    # def make_rank_df(self,pred_full_metric_df, desired_output_types, DESIRED_METRICS,rank_groupby_cols = ["infoID", "output_type", "SIM_PERIOD"]):



    #     pred_full_metric_df["normed_metric_result"] = pred_full_metric_df.groupby(rank_groupby_cols).transform(lambda x: x/x.sum())
    #     print()
    #     print(pred_full_metric_df)



    #     return




