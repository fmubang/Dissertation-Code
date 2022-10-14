import sys
# sys.path.append("/data/Fmubang/cp4-code-clean/functions")
# sys.path.append("/data/Fmubang/cp4-VAM-write-up-exps/functions")
sys.path.append("/storage2-mnt/data/fmubang/CP5-VAM-Paper-Stuff-3-3/functions")
import pandas as pd
import os,sys
from scipy import stats
import numpy as np
# from cascade_ft_funcs import *
import pickle
from random import seed
from random import random
from random import randrange
import xgboost as xgb
import joblib
# from ft_categories import *
import multiprocessing as mp

from sample_gen_aux_funcs import *
from basic_utils import *
from multiprocessing import Pool


class TemporalDataHandler:

    def __init__(self, reg_data_handler, NJOBS=32):
        self.reg_data_handler = reg_data_handler
        self.infoIDs = self.reg_data_handler.infoIDs
        self.output_types = self.reg_data_handler.target_ft_categories
        self.OUTPUT_SIZE = self.reg_data_handler.EXTRACT_OUTPUT_SIZE
        self.INPUT_SIZE = self.reg_data_handler.INPUT_SIZE
        self.NJOBS=NJOBS
        self.HAS_STATIC_FTS = reg_data_handler.GET_1HOT_INFO_ID_FTS


    #this is the general function
    def get_x_data_in_temporal_format(self, x_data_str):

        if x_data_str == "x_train":
            self.x_train =  self.get_x_data_in_temporal_format_specific(self.reg_data_handler.x_train)
            return self.x_train

        if x_data_str == "x_val":
            self.x_val = self.get_x_data_in_temporal_format_specific(self.reg_data_handler.x_val)
            return self.x_val

        if x_data_str == "x_test":
            self.x_test = self.get_x_data_in_temporal_format_specific(self.reg_data_handler.x_test)
            return self.x_test

        if x_data_str == "x_val_sliding_window":
            self.x_val_sliding_window = self.get_x_data_in_temporal_format_specific(self.reg_data_handler.x_val_sliding_window)
            return self.x_val_sliding_window

        if x_data_str == "x_test_sliding_window":
            self.x_test_sliding_window = self.get_x_data_in_temporal_format_specific(self.reg_data_handler.x_test_sliding_window)
            return self.x_test_sliding_window

    def get_x_data_in_temporal_format_specific(self, x_array):

        #data we need
        NUM_SAMPLES = x_array.shape[0]
        NUM_SEQUENCES = self.reg_data_handler.INPUT_SIZE
        static_fts = self.reg_data_handler.static_fts
        print("\nstatic_fts")
        print(static_fts)
        # sys.exit(0)
        target_ft_cats = self.reg_data_handler.target_ft_categories
        NUM_TARGET_FT_CATS = len(target_ft_cats)
        NUM_TOTAL_INPUT_FTS = x_array.shape[1]
        num_static_fts = len(static_fts)
        input_dynamic_ft_categories = self.reg_data_handler.input_dynamic_ft_categories
        num_dyn_ft_cats = len(input_dynamic_ft_categories)

        #show it
        print()
        print(NUM_SAMPLES)
        print(NUM_SEQUENCES)
        print(static_fts)
        print(target_ft_cats)
        print(NUM_TARGET_FT_CATS)
        print(NUM_TOTAL_INPUT_FTS)
        print(input_dynamic_ft_categories)
        print(num_dyn_ft_cats)

        dyn_new_temporal_samples = []
        static_new_temporal_samples = []

        for sample_idx in range(NUM_SAMPLES):
            cur_x_sample = x_array[sample_idx]
            # print(cur_x_sample.shape)

            #parse the sample
            flattened_sample = list(cur_x_sample)

            #remove static fts
            if self.HAS_STATIC_FTS == True:
                dyn_flattened_sample = flattened_sample[:-num_static_fts]
            else:
                dyn_flattened_sample =list(flattened_sample)
            dyn_flattened_sample = np.asarray(dyn_flattened_sample)



            #reshape it
            dyn_new_sample = dyn_flattened_sample.reshape((1, NUM_SEQUENCES, num_dyn_ft_cats))


            dyn_new_temporal_samples.append(dyn_new_sample)


            #static fts
            if self.HAS_STATIC_FTS == True:
                static_flattened_sample = flattened_sample[-num_static_fts:]
                static_flattened_sample = np.asarray(static_flattened_sample)
                static_new_sample = static_flattened_sample.reshape((1, num_static_fts))
                static_new_temporal_samples.append(static_new_sample)

        dyn_x_array = np.concatenate(dyn_new_temporal_samples, axis=0)
        print(dyn_x_array.shape)

        if self.HAS_STATIC_FTS == True:
            static_x_array = np.concatenate(static_new_temporal_samples, axis=0)
            print(static_x_array.shape)
        else:
            static_x_array = None

        # sys.exit(0)

        return  dyn_x_array,static_x_array

    def get_y_data_in_temporal_format_specific(self, y_array):

        #data we need
        NUM_SAMPLES = y_array.shape[0]
        NUM_SEQUENCES = self.reg_data_handler.EXTRACT_OUTPUT_SIZE
        target_ft_cats = self.reg_data_handler.target_ft_categories
        NUM_TARGET_FT_CATS = len(target_ft_cats)
        NUM_TOTAL_OUTPUT_FTS = y_array.shape[1]

        #show it
        print()
        print(NUM_SAMPLES)
        print(NUM_SEQUENCES)
        print(target_ft_cats)
        print(NUM_TARGET_FT_CATS)
        print(NUM_TOTAL_OUTPUT_FTS)

        new_target_samples = []
        for sample_idx in range(NUM_SAMPLES):
            flattened_target_sample = y_array[sample_idx].flatten()

            #reshape it
            new_flattened_target_sample = flattened_target_sample.reshape((1, NUM_SEQUENCES, NUM_TARGET_FT_CATS))

            new_target_samples.append(new_flattened_target_sample)

        y_array = np.concatenate(new_target_samples, axis=0)
        print(y_array.shape)

        # sys.exit(0)

        return  y_array

    def get_data_in_dict_form(self, tag):

        print("\nGetting x and y dict arrays...")
        data_array_dict = {}
        infoIDs = self.reg_data_handler.infoIDs
        infoID_train_and_test_array_dict = self.reg_data_handler.infoID_train_and_test_array_dict
        for infoID in infoIDs:

            data_array_dict[infoID] = infoID_train_and_test_array_dict[infoID][tag]
            # print()
            # print(tag)
            # print(infoID)
            # print(data_array_dict[infoID].shape)
        print("Done getting x and y dict arrays!")
        return data_array_dict

    def normalize_dict_data(self, x_data_array_dict, y_data_array_dict):

        print("\nNormalizing x and y arrays...")
        infoIDs = self.reg_data_handler.infoIDs
        x_norm_dict = {}
        y_norm_dict = {}
        for infoID in infoIDs:

            cur_x_array = x_data_array_dict[infoID]
            cur_y_array = y_data_array_dict[infoID]

            cur_x_array,cur_y_array = self.reg_data_handler.normalize_data(cur_x_array, cur_y_array)

            x_norm_dict[infoID] = cur_x_array
            y_norm_dict[infoID] = cur_y_array

            # print()
            # print(cur_x_array[0])
            # print(cur_y_array[0])
        print("Done normalizing x and y arrays!")
        return x_norm_dict, y_norm_dict

    def convert_x_dict_data_to_temporal_form(self, x_data_array_dict):

        print("\nConverting x dict data to temporal form...")
        infoIDs = self.reg_data_handler.infoIDs
        x_temporal_data_dict = {}
        for infoID in infoIDs:

            cur_x_array = x_data_array_dict[infoID]
            dynamic_x, static_x = self.get_x_data_in_temporal_format_specific(cur_x_array)
            print()
            print(dynamic_x.shape)
            if self.HAS_STATIC_FTS == True:
                print(static_x.shape)
                x_temporal_data_dict[infoID] = [dynamic_x, static_x]
            else:
                x_temporal_data_dict[infoID] = [dynamic_x]


        print("Done converting x dict data to temporal form!")

        return x_temporal_data_dict

    def predict_data_dict(self, model, x_data_array_dict):

        infoIDs = self.reg_data_handler.infoIDs
        y_pred_dict = {}
        print("\nPredicting x data...")
        for infoID in infoIDs:
            cur_x = x_data_array_dict[infoID]
            y_pred = model.predict(cur_x)
            print()
            print(infoID)
            print(y_pred.shape)
            y_pred_dict[infoID] = y_pred
        print("\nDone predicting!")

        return y_pred_dict

    def inverse_normalize_y_dict(self ,y_data_array_dict, ZERO_OUT_NEGS=True,ROUND=True):

        print("\nInverse normalizing y data...")
        infoIDs = self.reg_data_handler.infoIDs
        for infoID in infoIDs:

            cur_y = y_data_array_dict[infoID]
            cur_y = self.reg_data_handler.inverse_normalize_y_data(cur_y)
            if ZERO_OUT_NEGS == True:
                cur_y[cur_y<0] = 0
            if ROUND == True:
                cur_y = np.round(cur_y,0)
            print()
            print(cur_y[0])
            y_data_array_dict[infoID] = cur_y
        print("\nDone inverse normalizing y data!")

        return y_data_array_dict

    def convert_y_data_array_to_df_dict(self, y_data_array_dict):
        infoIDs = self.infoIDs
        y_df_data_dict = {}
        output_types = self.output_types
        #infoID -> sim_period -> df with output types
        print("\nConverting y array dict to df dict..")
        for infoID in infoIDs:

            cur_y = y_data_array_dict[infoID]
            SIM_PERIODS = cur_y.shape[0]
            y_df_data_dict[infoID] = {}

            for SIM_PERIOD in range(1, SIM_PERIODS+1):

                cur_sim_period_y = cur_y[SIM_PERIOD-1]
                print()
                print(cur_sim_period_y)

                cur_sim_period_y = cur_sim_period_y.reshape((self.OUTPUT_SIZE, len(output_types))).T
                print(cur_sim_period_y.shape)

                output_pred_dict = {}
                print(output_types)
                for o_idx,output_type in enumerate(output_types):
                    print(o_idx)
                    print(output_type)
                    output_pred_dict[output_type] = cur_sim_period_y[o_idx]

                df = pd.DataFrame(data=output_pred_dict)
                y_df_data_dict[infoID][SIM_PERIOD] = df
                print()
                print(df)

        return y_df_data_dict

    def get_metric_result_dict(self, y_pred_df_dict, y_true_df_dict, DESIRED_METRICS, metric_to_func_dict):

        metric_result_dict = {}
        for METRIC in DESIRED_METRICS:

            infoID_df_list = []
            output_type_df_list = []
            sim_period_df_list = []
            metric_result_df_list = []

            for infoID in self.infoIDs:
                metric_result_dict[infoID] = {}

                y_pred_infoID_dict = y_pred_df_dict[infoID]
                y_true_infoID_dict = y_true_df_dict[infoID]

                for SIM_PERIOD, y_pred_df in y_pred_infoID_dict.items():

                    print()
                    print(SIM_PERIOD)
                    print(y_pred_df)

                    y_true_df = y_true_infoID_dict[SIM_PERIOD]

                    for output_type in self.output_types:
                        y_pred = y_pred_df[output_type]
                        y_true = y_true_df[output_type]

                        print("\ny pred and true")
                        print()
                        print(y_pred)
                        print()
                        print(y_true)

                        metric_func = metric_to_func_dict[METRIC]
                        metric_result = metric_func(y_true, y_pred)

                        infoID_df_list.append(infoID)
                        output_type_df_list.append(output_type)
                        sim_period_df_list.append(SIM_PERIOD)
                        metric_result_df_list.append(metric_result)

            metric_df = pd.DataFrame(data={"infoID":infoID_df_list, "output_type":output_type_df_list, "SIM_PERIOD":sim_period_df_list,
                "metric_result":metric_result_df_list})

            print()
            print(metric_df)

            metric_result_dict[METRIC] = metric_df

        return metric_result_dict

    def get_mini_metric_df(self, arg_tuple):

        METRIC, infoID, SIM_PERIOD, output_type, y_true, y_pred, metric_func, tuple_idx, num_tuples = arg_tuple
        metric_result = metric_func(y_true, y_pred)

        metric_df = pd.DataFrame(data={"infoID":[infoID], "output_type":[output_type], "SIM_PERIOD":[SIM_PERIOD], "metric":[METRIC],
            "metric_result":[metric_result]})

        MOD = 300
        if tuple_idx%MOD == 0:
            print("Done with mini metric df %d of %d"%(tuple_idx, num_tuples))

        return metric_df

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

    # def

































#we might need this

# def convert_1_y_array_to_temporal_ft_df(self, array_tag, y):

#     y_array_tag = "y_" + array_tag

#     y = y[:1]
#     print("\n%s array shape"%y_array_tag)
#     print(y.shape)

#     #reshape
#     y = y.flatten()
#     y = y.reshape((self.INPUT_HOURS, len(self.target_ft_categories))).T
#     print("new shape: %s"%str(y.shape))

#     #make dict for input fts
#     ft_category_to_array_dict = {}
#     print("\nInitializing ft_category_to_array_dict...")

#     for i,cur_target_ft_category in enumerate(self.target_ft_categories):
#         print(cur_target_ft_category)
#         ft_category_to_array_dict[cur_target_ft_category] = y[i].flatten()


#     dynamic_target_ft_df = pd.DataFrame(data=ft_category_to_array_dict)
#     dynamic_target_ft_df=dynamic_target_ft_df[self.target_ft_categories]

#     print("\ndynamic_target_ft_df")
#     print(dynamic_target_ft_df)

#     return dynamic_target_ft_df


# self.input_dynamic_and_static_fts_to_idx_dict=dynamic_and_static_fts_to_idx_dict
# self.input_idx_to_dynamic_and_static_fts_dict=idx_to_dynamic_and_static_fts_dict
# self.input_dynamic_and_static_fts=dynamic_and_static_fts
# self.infoID_train_and_test_array_dict = infoID_train_and_test_array_dict

#might also need this
# target_idx_to_ft_dict = {}
# target_ft_to_idx_dict = {}
# for i,target_ft in enumerate(target_fts):
#     target_idx_to_ft_dict[i]=target_ft
#     target_ft_to_idx_dict[target_ft] = i

# self.target_idx_to_ft_dict =target_idx_to_ft_dict
# self.target_ft_to_idx_dict =target_ft_to_idx_dict
# print("\nself.target_ft_to_idx_dict")
# print(self.target_ft_to_idx_dict)



