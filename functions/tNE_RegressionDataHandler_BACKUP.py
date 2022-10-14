
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

class tNE_RegressionDataHandler:

    def __init__(self,eval_tag,pred_type ,ts_to_desired_period_tag_dict, infoIDs,x_main_emb_input_dir, y_main_ts_target_input_dir,main_output_dir,param_config_tag ,DESIRED_EVALUATION_TIMESTEP ,
        SAMPLE_LOOKBACK, SCALER_TAG, SAMPLE_RATE, WALK_TYPE, NUM_VAL_PERIODS=7, NUM_TEST_PERIODS=21, NJOBS=32):

        self.pred_type = pred_type

        #set tag
        if eval_tag == "val":
            NUM_EVAL_PERIODS = NUM_VAL_PERIODS
            eval_period_tag = "Input-Snapshot-for-Output-TS-%d-of-%d"%(DESIRED_EVALUATION_TIMESTEP, NUM_EVAL_PERIODS)
        elif eval_tag == "test":
            eval_period_tag = "Input-Snapshot-for-Output-TS-%d-of-%d"%(DESIRED_EVALUATION_TIMESTEP, NUM_EVAL_PERIODS)
            NUM_EVAL_PERIODS = NUM_TEST_PERIODS
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
        SAMPLE_LOOKBACK = int(SAMPLE_LOOKBACK)
        self.SAMPLE_LOOKBACK = SAMPLE_LOOKBACK
        self.SCALER_TAG = SCALER_TAG
        self.SAMPLE_RATE = SAMPLE_RATE
        self.WALK_TYPE = WALK_TYPE
        self.NJOBS = NJOBS

        print()
        print(SAMPLE_LOOKBACK)

        #get train period tags
        train_period_tags = []
        for cur_lookback in range(1, SAMPLE_LOOKBACK+1):
            cur_train_tag = "Input-Snapshot-for-Output-TS-%d-of-%d"%(DESIRED_EVALUATION_TIMESTEP-cur_lookback, NUM_EVAL_PERIODS)
            train_period_tags.append(cur_train_tag)
        self.train_period_tags = train_period_tags

        print("\nAll tags")
        print(eval_period_tag)
        for t in self.train_period_tags:
            print(t)

        # sys.exit(0)

    #def load_bc_x_and_y_node2vec_arrays(self, n2v_model,  infoIDs, y_output_ts_main_input_dir, desired_val_period_tag, data_type,UNDER_SAMPLE_NEG_CLASS):
    def load_tNE_reg_data(self, data_type):

        #data type can either be "train" or "eval"
        if data_type == "train":
            data_period_tags = list(self.train_period_tags)
        else:
            data_period_tags = [self.eval_period_tag]



        all_x_arrays = []
        all_y_arrays = []

        for data_period_tag in data_period_tags:



            #load n2v model
            cur_n2v_input_dir = self.x_main_emb_input_dir + self.eval_tag + "/"+ data_period_tag + "/" + self.WALK_TYPE + "/"
            print()
            print(cur_n2v_input_dir)

            print("\nLoading n2v model...")
            n2v_model = gzip_load(cur_n2v_input_dir + "node2vec_model")
            print(n2v_model)

            #get y tag
            cur_ts = int(data_period_tag.split("-")[5])
            print("\ncur_ts")
            print(cur_ts)

            y_tag = self.ts_to_desired_period_tag_dict[cur_ts]
            print(y_tag)

            for infoID in self.infoIDs:

                #load data
                hyp_infoID = infoID.replace("/", "_")

                cur_y_ts_input_dir = self.y_main_ts_target_input_dir +    hyp_infoID + "/"
                ydf = pd.read_csv(cur_y_ts_input_dir + y_tag + ".csv")

                print()
                print(ydf)

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

                # sys.exit(0)


                # ydf = ydf[["nodeUserID", "parentUserID", "is_active"]]
                print()
                print(ydf)

                #y_array = ydf["is_active"].values

                if self.pred_type == "binary_classification":
                    y_array = ydf["is_active"].values
                elif self.pred_type == "multioutput_regression":
                    y_array = ydf[allowed_ts].values

                all_y_arrays.append(y_array)

                # links = []

                idx=1
                num_links = ydf.shape[0]
                MOD_NUM = 100
                for child,parent in zip(ydf["nodeUserID"], ydf["parentUserID"]):

                    child = str(child)
                    parent = str(parent)

                    link = np.concatenate(np.asarray([n2v_model.wv[child], n2v_model.wv[parent]]), axis=None)

                    all_x_arrays.append(link)

                    if idx%MOD_NUM == 0:
                        print("Got link %d of %d"%(idx, num_links))
                    idx+=1

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