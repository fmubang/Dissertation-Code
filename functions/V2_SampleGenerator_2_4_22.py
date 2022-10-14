import sys
# sys.path.append("/data/Fmubang/cp4-code-clean/functions")
# sys.path.append("/data/Fmubang/cp4-VAM-write-up-exps/functions")
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



class SampleGenerator:

    def __init__(self,train_start,train_end,val_start,val_end,test_start,test_end,GRAN,
        INPUT_HOURS,OUTPUT_HOURS,main_input_dir,n_jobs,platforms, infoIDs, user_statuses,
        GET_GDELT_FEATURES,GET_REDDIT_FEATURES,TARGET_PLATFORM,GET_1HOT_INFO_ID_FTS,main_output_dir,RESCALE,SCALER_TYPE,FEATURE_RANGE,RESCALE_TARGET,
        INPUT_TWITTER_LOGNORM,OUTPUT_TWITTER_LOGNORM,INPUT_OTHER_LOGNORM,OUTPUT_OTHER_LOGNORM,LOGNORM_IGNORE_TAGS,GET_EXTERNAL_PLATFORM_FTS,LOGNORM_DEBUG_PRINT,
        CORRELATION_FUNCS_TO_USE_LIST,CORRELATION_STR_TO_FUNC_DICT,AGG_TIME_SERIES, DARPA_INPUT_FORMAT, GET_NEWS_DATA, GET_GLOBAL_PLATFORM_FTS):

        self.GET_NEWS_DATA=GET_NEWS_DATA
        self.DARPA_INPUT_FORMAT = DARPA_INPUT_FORMAT
        self.CORRELATION_STR_TO_FUNC_DICT=CORRELATION_STR_TO_FUNC_DICT
        self.CORRELATION_FUNCS_TO_USE_LIST=CORRELATION_FUNCS_TO_USE_LIST

        self.train_start=train_start
        self.train_end=train_end
        self.val_start=val_start
        self.val_end=val_end
        self.test_start=test_start
        self.test_end=test_end

        if GRAN != "H":
            print("\nSorry! Hour gran only!")
            sys.exit(0)

        self.GRAN=GRAN
        self.INPUT_HOURS=INPUT_HOURS
        self.OUTPUT_HOURS=OUTPUT_HOURS
        self.main_input_dir=main_input_dir
        self.n_jobs=n_jobs
        self.hyp_dict=hyphenate_infoID_dict(infoIDs)
        self.IO_TUPLE = (INPUT_HOURS, 1)
        self.platforms = platforms
        self.infoIDs = infoIDs
        BASIC_GRAN = self.GRAN[-1]
        self.BASIC_GRAN=BASIC_GRAN
        # self.feature_list_dir = main_input_dir + "GRAN-" + BASIC_GRAN + "/feature-lists/"
        # self.features_dir = main_input_dir + "GRAN-" + BASIC_GRAN + "/features/"
        self.feature_list_dir = main_input_dir + "feature-lists/"
        self.features_dir = main_input_dir + "features/"
        self.GET_EXTERNAL_PLATFORM_FTS=GET_EXTERNAL_PLATFORM_FTS
        self.RESCALE=RESCALE
        self.RESCALE_TARGET=RESCALE_TARGET
        self.SCALER_TYPE=SCALER_TYPE
        self.FEATURE_RANGE=FEATURE_RANGE

        self.GET_GDELT_FEATURES=GET_GDELT_FEATURES
        # print("\nGET_GDELT_FEATURES")
        # print(GET_GDELT_FEATURES)
        # sys.exit(0)
        self.GET_REDDIT_FEATURES=GET_REDDIT_FEATURES
        self.TARGET_PLATFORM=TARGET_PLATFORM
        self.GET_1HOT_INFO_ID_FTS=GET_1HOT_INFO_ID_FTS

        self.INPUT_TWITTER_LOGNORM=INPUT_TWITTER_LOGNORM
        self.OUTPUT_TWITTER_LOGNORM=OUTPUT_TWITTER_LOGNORM
        self.INPUT_OTHER_LOGNORM=INPUT_OTHER_LOGNORM
        self.OUTPUT_OTHER_LOGNORM=OUTPUT_OTHER_LOGNORM
        self.LOGNORM_IGNORE_TAGS=LOGNORM_IGNORE_TAGS

        #get extract output size
        self.EXTRACT_OUTPUT_SIZE = OUTPUT_HOURS
        self.INPUT_SIZE=INPUT_HOURS
        # self.GET_1HOT_INFO_ID_FTS=GET_1HOT_INFO_ID_FTS
        self.LOGNORM_DEBUG_PRINT=LOGNORM_DEBUG_PRINT

        self.AGG_TIME_SERIES=AGG_TIME_SERIES
        self.GET_GLOBAL_PLATFORM_FTS=GET_GLOBAL_PLATFORM_FTS

        param_vals = [train_start, train_end, val_start, val_end, test_start, test_end, GRAN, INPUT_HOURS, OUTPUT_HOURS, GET_1HOT_INFO_ID_FTS, GET_GDELT_FEATURES, GET_REDDIT_FEATURES, TARGET_PLATFORM,GET_EXTERNAL_PLATFORM_FTS,RESCALE]
        param_names = ["train_start","train_end","val_start","val_end","test_start","test_end","GRAN"," INPUT_HOURS","OUTPUT_HOURS","GET_1HOT_INFO_ID_FTS","GET_GDELT_FEATURES","GET_REDDIT_FEATURES","TARGET_PLATFORM","GET_EXTERNAL_PLATFORM_FTS","RESCALE"]

        more_param_vals = [RESCALE_TARGET,SCALER_TYPE,FEATURE_RANGE,INPUT_TWITTER_LOGNORM,OUTPUT_TWITTER_LOGNORM,INPUT_OTHER_LOGNORM,OUTPUT_OTHER_LOGNORM,LOGNORM_IGNORE_TAGS, str(CORRELATION_FUNCS_TO_USE_LIST),AGG_TIME_SERIES,DARPA_INPUT_FORMAT,GET_GLOBAL_PLATFORM_FTS]
        more_param_names = ["RESCALE_TARGET","SCALER_TYPE","FEATURE_RANGE","INPUT_TWITTER_LOGNORM","OUTPUT_TWITTER_LOGNORM","INPUT_OTHER_LOGNORM","OUTPUT_OTHER_LOGNORM","LOGNORM_IGNORE_TAGS", "CORRELATION_FUNCS_TO_USE_LIST","AGG_TIME_SERIES","DARPA_INPUT_FORMAT","GET_GLOBAL_PLATFORM_FTS"]

        param_vals = list(param_vals) + list(more_param_vals)
        param_names = list(param_names) + list(more_param_names)

        param_df = pd.DataFrame(data={"param":param_names, "value":param_vals})
        self.param_df = param_df
        print(param_df)

        #make output dir
        # self.output_dir = make_output_dir_from_params(param_vals, main_output_dir,[])
        # print(self.output_dir)

        self.param_tag = make_param_tag(param_vals)
        print(self.param_tag)
        # sys.exit(0)

        #save params
        # param_fp = self.output_dir + "ft-preprocess-params.csv"
        # param_df.to_csv(param_fp)
        # print(param_fp)
        self.ft_preproc_param_df=param_df

        temp_train_start = pd.to_datetime(train_start, utc=True)
        # test_start = pd.to_datetime(test_start, utc=True)

        lookback_in_days = int(self.INPUT_HOURS/24)
        self.lookback_in_days = lookback_in_days
        print("\nlookback_in_days")
        print(lookback_in_days)
        abs_train_start = temp_train_start -  pd.to_timedelta(lookback_in_days, unit='d')
        abs_train_start = pd.to_datetime(abs_train_start, utc=True)





        # temp = pd.Series([abs_train_start,test_start])
        # temp = pd.to_datetime(temp, utc=True)
        # abs_train_start = temp.iloc[0]
        # test_start = temp.iloc[-1]
        # # sys.exit(0)

        abs_train_start = str(abs_train_start).split(" ")[0]

        print(abs_train_start)
        print(test_start)
        self.abs_train_start = abs_train_start

        self.full_dates = pd.date_range(abs_train_start, test_end, freq=self.GRAN[-1])
        print(self.full_dates)
        print()
        print(len(self.full_dates))

        self.full_dates_daily = pd.date_range(abs_train_start, test_end, freq="D")
        print(self.full_dates_daily)
        print(len(self.full_dates_daily))

        self.eval_dates = pd.date_range(train_start, test_end, freq=self.GRAN[-1])
        self.eval_dates_daily = pd.date_range(train_start, test_end, freq="D")
        # sys.exit(0)





    def get_feature_name_lists(self):
        #load features

        # fp = self.feature_list_dir + "feature-lists/"
        dynamic_fts = []
        target_fts= get_fts_from_fp(self.feature_list_dir + "target_fts.txt")
        # static_fts = get_fts_from_fp(self.feature_list_dir + "static_fts.txt")

        if self.GET_1HOT_INFO_ID_FTS == True:
            static_fts = list(self.infoIDs)
        else:
            static_fts = []

        # new_static_fts = []
        # for static_ft in static_fts:
        #     if static_ft in self.infoIDs:
        #         new_static_fts.append(static_ft)
        # static_fts = list(new_static_fts)



        self.sum_fts=[]
        self.avg_fts=[]

        if self.GET_NEWS_DATA == True:
            news_dynamic_fts=get_fts_from_fp(self.feature_list_dir + "news_dynamic_fts.txt")
            dynamic_fts = dynamic_fts + list(news_dynamic_fts)
            self.sum_fts=self.sum_fts + list(news_dynamic_fts)

        if self.GET_GDELT_FEATURES==True:
            dynamic_fts = dynamic_fts + get_fts_from_fp(self.feature_list_dir + "dynamic_gdelt_fts.txt")
            self.sum_fts=self.sum_fts + ["NumMentions"]
            self.avg_fts=self.avg_fts + ["GoldsteinScale","AvgTone"]

        # print("\nself.GET_REDDIT_FEATURES")
        # print(self.GET_REDDIT_FEATURES)
        # sys.exit(0)

        if self.GET_REDDIT_FEATURES==True:
            #dynamic_fts = dynamic_fts + get_fts_from_fp(self.feature_list_dir + "dynamic_reddit_fts.txt")
            dynamic_fts = dynamic_fts + ["num_reddit_activities"]
            self.sum_fts=self.sum_fts + ["num_reddit_activities"]


        if self.TARGET_PLATFORM == "all":
            dynamic_fts = dynamic_fts + get_fts_from_fp(self.feature_list_dir + "twitter_dynamic_global_fts.txt")
            dynamic_fts = dynamic_fts + get_fts_from_fp(self.feature_list_dir + "youtube_dynamic_global_fts.txt")
            dynamic_fts = dynamic_fts + get_fts_from_fp(self.feature_list_dir + "twitter_dynamic_pair_fts.txt")
            dynamic_fts = dynamic_fts + get_fts_from_fp(self.feature_list_dir + "youtube_dynamic_pair_fts.txt")

            # if self.GET_GLOBAL_PLATFORM_FTS == True:





        if self.TARGET_PLATFORM == "twitter":
            dynamic_fts = dynamic_fts + get_fts_from_fp(self.feature_list_dir + "twitter_dynamic_global_fts.txt")
            dynamic_fts = dynamic_fts + get_fts_from_fp(self.feature_list_dir + "twitter_dynamic_pair_fts.txt")

            if self.GET_EXTERNAL_PLATFORM_FTS==True:
                dynamic_fts = dynamic_fts + get_fts_from_fp(self.feature_list_dir + "youtube_dynamic_global_fts.txt")
                dynamic_fts = dynamic_fts + get_fts_from_fp(self.feature_list_dir + "youtube_dynamic_pair_fts.txt")

            new_target_fts = []
            # for ft in target_fts:
            #   if "youtube" in ft:
            #       target_fts.remove(ft)
            for ft in target_fts:
                if "twitter" in ft:
                    new_target_fts.append(ft)
            target_fts = list(new_target_fts)


        if self.TARGET_PLATFORM == "youtube":
            dynamic_fts = dynamic_fts + get_fts_from_fp(self.feature_list_dir + "youtube_dynamic_global_fts.txt")
            dynamic_fts = dynamic_fts + get_fts_from_fp(self.feature_list_dir + "youtube_dynamic_pair_fts.txt")

            if self.GET_EXTERNAL_PLATFORM_FTS==True:
                dynamic_fts = dynamic_fts + get_fts_from_fp(self.feature_list_dir + "twitter_dynamic_global_fts.txt")
                dynamic_fts = dynamic_fts + get_fts_from_fp(self.feature_list_dir + "twitter_dynamic_pair_fts.txt")

            # static_fts.remove("is_twitter")
            # for ft in target_fts:
            #   if "twitter" in ft:
            #       target_fts.remove(ft)

            # if self.GET_GLOBAL_PLATFORM_FTS == False:
            #     keep_dyn_fts = []
            #     for ft in dynamic_fts:
            #         if "global" not in ft:
            #             keep_dyn_fts.append(ft)
            # dynamic_fts = list(keep_dyn_fts)

            new_target_fts = []
            for ft in target_fts:
                if "youtube" in ft:
                    new_target_fts.append(ft)
            target_fts = list(new_target_fts)

        if self.GET_GLOBAL_PLATFORM_FTS == False:
            keep_dyn_fts = []
            for ft in dynamic_fts:
                if "global" not in ft:
                    keep_dyn_fts.append(ft)
            dynamic_fts = list(keep_dyn_fts)


        self.sum_fts = list(set(self.sum_fts + dynamic_fts))

        #combine
        input_fts = list(dynamic_fts) + list(static_fts)
        print("\ninput_fts")
        for ft in input_fts:
            print(ft)
        print("\ntarget_fts")
        for ft in target_fts:
            print(ft)

        # if self.GET_1HOT_INFO_ID_FTS == False:
        #     sys.exit(0)

        self.dynamic_fts = dynamic_fts


        self.static_fts = static_fts
        self.input_fts = input_fts
        self.target_ft_categories = target_fts

        target_idx_to_ft_dict = {}
        target_ft_to_idx_dict = {}
        for i,target_ft in enumerate(target_fts):
            target_idx_to_ft_dict[i]=target_ft
            target_ft_to_idx_dict[target_ft] = i

        self.target_idx_to_ft_dict =target_idx_to_ft_dict
        self.target_ft_to_idx_dict =target_ft_to_idx_dict
        print("\nself.target_ft_to_idx_dict")
        print(self.target_ft_to_idx_dict)
        # sys.exit(0)

        self.input_dynamic_ft_categories= list(dynamic_fts)
        self.input_ft_categories= list(input_fts)
        self.target_ft_categories= list(target_fts)


    def generate_platform_infoID_pair_samples(self):

        infoID_to_df_dict={}
        for infoID in self.infoIDs:
            infoID_to_df_dict[infoID]={}
            hyp_infoID = self.hyp_dict[infoID]
            fp = self.features_dir +hyp_infoID + "-features.csv"
            df = pd.read_csv(fp)

            #KeyError: "['num_reddit_activities'] not in index"

            if "num_reddit_actions" in list(df):
                df = df.rename(columns={"num_reddit_actions": "num_reddit_activities"})
            df["nodeTime"]=pd.to_datetime(df["nodeTime"], utc=True)
            print(df)

            df = alter_df_GRAN_v3_fix_duplicate_issue(df, self.GRAN, self.input_fts, self.target_ft_categories,self.sum_fts, self.avg_fts)

            print("\naltered df")
            print(df)

            # sys.exit(0)

            #get dfs
            train_and_val_df,test_df = split_dfs_into_train_and_test_with_nodeID_hack(df, self.test_start, self.test_end,self.IO_TUPLE,DEBUG_PRINT=False)
            train_df,val_df = split_dfs_into_train_and_test_with_nodeID_hack(df, self.val_start, self.val_end,self.IO_TUPLE,DEBUG_PRINT=False)

            print("\ntrain_df")
            print(train_df)

            print("\nval_df")
            print(val_df)

            print("\ntest_df")
            print(test_df)

            infoID_to_df_dict[infoID]["train_df"]=train_df
            infoID_to_df_dict[infoID]["val_df"]=val_df
            infoID_to_df_dict[infoID]["test_df"]=test_df

        self.infoID_to_df_dict = infoID_to_df_dict

    def generate_platform_infoID_pair_samples_v2_1_df(self):

        infoID_to_df_dict={}
        for infoID in self.infoIDs:
            infoID_to_df_dict[infoID]={}
            hyp_infoID = self.hyp_dict[infoID]
            fp = self.features_dir +hyp_infoID + "-features.csv"
            df = pd.read_csv(fp)

            if "num_reddit_actions" in list(df):
                df = df.rename(columns={"num_reddit_actions": "num_reddit_activities"})
            df = config_df_by_dates(df, self.abs_train_start, self.test_end)
            df["nodeTime"]=pd.to_datetime(df["nodeTime"], utc=True)
            print(df)

            print(len(self.full_dates))
            print(df.shape[0])
            if len(self.full_dates) != df.shape[0]:
                print("\nError! len(self.full_dates) != df.shape[0]")
                sys.exit(0)

            df = alter_df_GRAN_v3_fix_duplicate_issue(df, self.GRAN, self.input_fts, self.target_ft_categories,self.sum_fts, self.avg_fts)


            print("\naltered df")
            print(df)


            infoID_to_df_dict[infoID] = df


        self.infoID_to_df_dict = infoID_to_df_dict

    # self.infoID_to_df_dict = self.generate_platform_infoID_pair_samples(self)

    def get_mutli_ts_out_target_fts(self, target_ft_categories, y_array):

        new_ft_to_idx_dict = {}
        new_idx_to_ft_dict = {}
        flattened_fts = []
        num_dynamic_fts = len(target_ft_categories)
        print("\nflattened_fts")
        for idx in range(y_array.shape[1]):
            cur_dyn_idx = idx%num_dynamic_fts
            # print(cur_dyn_idx)
            cur_dyn_ft = target_ft_categories[cur_dyn_idx]
            new_idx_to_ft_dict[idx]= cur_dyn_ft + "_%d"%idx
            new_ft_to_idx_dict[cur_dyn_ft + "_%d"%idx] = idx
            flattened_fts.append(cur_dyn_ft + "_%d"%idx)
            print("%d: %s"%(idx, cur_dyn_ft + "_%d"%idx))

        target_ft_to_idx_dict = dict(new_ft_to_idx_dict)
        target_idx_to_ft_dict = dict(new_idx_to_ft_dict)
        target_fts = list(flattened_fts)

        return target_ft_to_idx_dict,target_idx_to_ft_dict,target_fts

    def remove_platform_fts(self,x, tag,dynamic_and_static_fts_to_idx_dict,idx_to_dynamic_and_static_fts_dict,dynamic_and_static_fts):
        print("\nGetting rid of youtube and twitter features!!!!")

        new_ft_to_idx_dict = {}
        new_idx_to_ft_dict = {}
        new_dynamic_fts = []
        new_static_and_dynamic_fts = []


        keep_indices = []
        new_ft_idx = 0
        for ft in dynamic_and_static_fts:
            if ("twitter" in ft) or ("youtube" in ft):
                print("Removing %s"%ft)
            else:
                idx = dynamic_and_static_fts_to_idx_dict[ft]
                keep_indices.append(idx)

                new_static_and_dynamic_fts.append(ft)
                new_ft_to_idx_dict[ft] = new_ft_idx
                new_idx_to_ft_dict[new_ft_idx] = ft

                if ft not in self.static_fts:
                    new_dynamic_fts.append(ft)

                new_ft_idx+=1

        num_fts_to_keep = len(keep_indices)
        num_original_fts = len(dynamic_and_static_fts)
        print("\nKeeping %d out of %d fts"%(num_fts_to_keep, num_original_fts))
        print("x_%s shape before ft removal"%tag)
        print(x.shape)
        x = np.take(x,keep_indices,axis=1 )
        print("x_%s shape after ft removal"%tag)
        print(x.shape)
        # sys.exit(0)
        return x, new_ft_to_idx_dict,new_idx_to_ft_dict,new_dynamic_fts,new_static_and_dynamic_fts

    def agg_y_array_v2_mult_output_fts(self,y):
        # print("\ny reshape")
        # y = y.reshape((y.shape[0],y.shape[1]*y.shape[2] ))
        print(y.shape)

        print("\ny agg")
        y = y.sum(axis=1)
        # y = y.reshape((y.shape[0], 1, 1))
        print(y.shape)

        return y

    def get_infoID_train_and_test_array_dict(self):
        infoID_train_and_test_array_dict = {}
        num_infoIDs = len(self.infoIDs)
        print("\nnum_infoIDs: %d"%num_infoIDs)



        i=1
        for infoID in self.infoIDs:
            #get dfs
            train_df = self.infoID_to_df_dict[infoID]["train_df"]
            val_df = self.infoID_to_df_dict[infoID]["val_df"]
            test_df = self.infoID_to_df_dict[infoID]["test_df"]


            #INSERt initial features
            x_train,y_train,ft_to_idx_dict =convert_df_to_sliding_window_form_diff_x_and_y_fts_no_print_with_idx_dict(train_df, self.INPUT_SIZE, self.EXTRACT_OUTPUT_SIZE, self.dynamic_fts, self.target_ft_categories)
            print("\nx and y train arrays")
            print(x_train.shape)
            print(y_train.shape)
            x_train,dynamic_and_static_fts_to_idx_dict,idx_to_dynamic_and_static_fts_dict,dynamic_and_static_fts = insert_static_1hot_fts_for_platform_samples(infoID, x_train, self.dynamic_fts,ft_to_idx_dict ,self.static_fts,self.infoIDs,self.GET_1HOT_INFO_ID_FTS)
            print("\nx and y train arrays")
            print(x_train.shape)
            print(y_train.shape)



            #get first sample
            y_train1 = y_train[0]
            num_timesteps = y_train.shape[1]
            num_fts = y_train.shape[2]
            for t in range(num_timesteps):
                print(y_train1[t])


            #agg y
            if self.AGG_TIME_SERIES == True:
                y_train = self.agg_y_array_v2_mult_output_fts(y_train)
                print("\nfirst y_train after agg")
                print(y_train[0])
            else:
                y_train = y_train.reshape((y_train.shape[0],y_train.shape[1] *y_train.shape[2] ))
            # print("\ny_train after agg")
            # print(y_train.shape)

            # sys.exit(0)

            #get val arrays
            x_val,y_val = convert_df_to_test_sliding_window_form_diff_x_and_y_fts_no_print(val_df, self.INPUT_SIZE, self.EXTRACT_OUTPUT_SIZE,self.dynamic_fts, self.target_ft_categories,MOD_NUM=1000)
            x_val,_,_,_ = insert_static_1hot_fts_for_platform_samples(infoID,x_val, self.dynamic_fts,ft_to_idx_dict ,self.static_fts,self.infoIDs,self.GET_1HOT_INFO_ID_FTS)
            if self.AGG_TIME_SERIES == True:
                y_val = self.agg_y_array_v2_mult_output_fts(y_val)
            else:
                y_val = y_val.reshape((y_val.shape[0],y_val.shape[1] *y_val.shape[2] ))
            print("\nx and y val arrays")
            print(x_val.shape)
            print(y_val.shape)

            # sys.exit(0)

            #sliding window train
            x_val_sliding_window,y_val_sliding_window,ft_to_idx_dict =convert_df_to_sliding_window_form_diff_x_and_y_fts_no_print_with_idx_dict(val_df, self.INPUT_SIZE, self.EXTRACT_OUTPUT_SIZE, self.dynamic_fts, self.target_ft_categories)
            x_val_sliding_window,_,_,_ = insert_static_1hot_fts_for_platform_samples(infoID,x_val_sliding_window, self.dynamic_fts,ft_to_idx_dict ,self.static_fts,self.infoIDs,self.GET_1HOT_INFO_ID_FTS)

            if self.AGG_TIME_SERIES == True:
                y_val_sliding_window = self.agg_y_array_v2_mult_output_fts(y_val_sliding_window)
            else:
                y_val_sliding_window = y_val_sliding_window.reshape((y_val_sliding_window.shape[0],y_val_sliding_window.shape[1] *y_val_sliding_window.shape[2] ))

            print("\nx and y val sliding arrays")
            print(x_val_sliding_window.shape)
            print(y_val_sliding_window.shape)

            #get test arrays
            x_test,y_test = convert_df_to_test_sliding_window_form_diff_x_and_y_fts_no_print(test_df, self.INPUT_SIZE, self.EXTRACT_OUTPUT_SIZE,self.dynamic_fts, self.target_ft_categories,MOD_NUM=1000)
            x_test,_,_,_ = insert_static_1hot_fts_for_platform_samples(infoID,x_test, self.dynamic_fts,ft_to_idx_dict ,self.static_fts,self.infoIDs,self.GET_1HOT_INFO_ID_FTS)

            if self.AGG_TIME_SERIES == True:
                y_test = self.agg_y_array_v2_mult_output_fts(y_test)
            else:
                y_test = y_test.reshape((y_test.shape[0],y_test.shape[1] *y_test.shape[2] ))
            print("\ny_test shape")
            print(y_test.shape)

            #sliding window test
            x_test_sliding_window,y_test_sliding_window,ft_to_idx_dict =convert_df_to_sliding_window_form_diff_x_and_y_fts_no_print_with_idx_dict(test_df, self.INPUT_SIZE, self.EXTRACT_OUTPUT_SIZE, self.dynamic_fts, self.target_ft_categories)
            x_test_sliding_window,_,_,_ = insert_static_1hot_fts_for_platform_samples(infoID,x_test_sliding_window, self.dynamic_fts,ft_to_idx_dict ,self.static_fts,self.infoIDs,self.GET_1HOT_INFO_ID_FTS)

            if self.AGG_TIME_SERIES == True:
                y_test_sliding_window = self.agg_y_array_v2_mult_output_fts(y_test_sliding_window)
            else:
                y_test_sliding_window = y_test_sliding_window.reshape((y_test_sliding_window.shape[0],y_test_sliding_window.shape[1] *y_test_sliding_window.shape[2] ))
            print("\nx and y test sliding arrays")
            print(x_test_sliding_window.shape)
            print(y_test_sliding_window.shape)
            # sys.exit(0)

            if self.DARPA_INPUT_FORMAT == True:
                x_train, new_ft_to_idx_dict,new_idx_to_ft_dict,new_dynamic_fts,new_static_and_dynamic_fts = self.remove_platform_fts(x_train, "train",dynamic_and_static_fts_to_idx_dict,idx_to_dynamic_and_static_fts_dict,dynamic_and_static_fts)
                x_val, new_ft_to_idx_dict,new_idx_to_ft_dict,new_dynamic_fts,new_static_and_dynamic_fts = self.remove_platform_fts(x_val, "val",dynamic_and_static_fts_to_idx_dict,idx_to_dynamic_and_static_fts_dict,dynamic_and_static_fts)
                x_test, new_ft_to_idx_dict,new_idx_to_ft_dict,new_dynamic_fts,new_static_and_dynamic_fts = self.remove_platform_fts(x_test, "test",dynamic_and_static_fts_to_idx_dict,idx_to_dynamic_and_static_fts_dict,dynamic_and_static_fts)
                x_test_sliding_window, new_ft_to_idx_dict,new_idx_to_ft_dict,new_dynamic_fts,new_static_and_dynamic_fts = self.remove_platform_fts(x_test_sliding_window, "test_sliding_window",dynamic_and_static_fts_to_idx_dict,idx_to_dynamic_and_static_fts_dict,dynamic_and_static_fts)
                x_val_sliding_window, new_ft_to_idx_dict,new_idx_to_ft_dict,new_dynamic_fts,new_static_and_dynamic_fts = self.remove_platform_fts(x_val_sliding_window, "val_sliding_window",dynamic_and_static_fts_to_idx_dict,idx_to_dynamic_and_static_fts_dict,dynamic_and_static_fts)

                dynamic_and_static_fts_to_idx_dict = dict(new_ft_to_idx_dict)
                idx_to_dynamic_and_static_fts_dict = dict(new_idx_to_ft_dict)
                dynamic_and_static_fts = list(new_static_and_dynamic_fts)

            #data info
            infoID_train_and_test_array_dict[infoID]={}
            infoID_train_and_test_array_dict[infoID]["x_train"] = x_train
            infoID_train_and_test_array_dict[infoID]["y_train"] = y_train
            infoID_train_and_test_array_dict[infoID]["x_val"] = x_val
            infoID_train_and_test_array_dict[infoID]["y_val"] = y_val
            infoID_train_and_test_array_dict[infoID]["x_test"] = x_test
            infoID_train_and_test_array_dict[infoID]["y_test"] = y_test

            infoID_train_and_test_array_dict[infoID]["x_val_sliding_window"] = x_val_sliding_window
            infoID_train_and_test_array_dict[infoID]["y_val_sliding_window"] = y_val_sliding_window

            infoID_train_and_test_array_dict[infoID]["x_test_sliding_window"] = x_test_sliding_window
            infoID_train_and_test_array_dict[infoID]["y_test_sliding_window"] = y_test_sliding_window


            print("Got infoID array %d of %d" %(i, num_infoIDs))
            i+=1


        #fix input fts
        # input_fts = list(dynamic_and_static_fts)

        print("\nx and y train sizes")
        print(x_train.shape)
        print(y_train.shape)

        print("\nx and y val sizes")
        print(x_val.shape)
        print(y_val.shape)

        print("\nx and y test sizes")
        print(x_test.shape)
        print(y_test.shape)

        print("\nx and y val sliding window sizes")
        print(x_val_sliding_window.shape)
        print(y_val_sliding_window.shape)

        print("\nx and y test sliding window sizes")
        print(x_test_sliding_window.shape)
        print(y_test_sliding_window.shape)

        # print("\ntransposed matrices")
        # print(x_test_sliding_window.T.shape)
        # print(y_test_sliding_window.T.shape)

        # self.input_dynamic_and_static_fts_to_idx_dict=dynamic_and_static_fts_to_idx_dict
        # self.input_idx_to_dynamic_and_static_fts_dict=idx_to_dynamic_and_static_fts_dict
        # self.input_dynamic_and_static_fts=dynamic_and_static_fts

        self.input_dynamic_and_static_fts_to_idx_dict=dynamic_and_static_fts_to_idx_dict
        self.input_idx_to_dynamic_and_static_fts_dict=idx_to_dynamic_and_static_fts_dict
        self.input_dynamic_and_static_fts=dynamic_and_static_fts
        self.infoID_train_and_test_array_dict = infoID_train_and_test_array_dict

        # self.target_ft_categories = list(self.target_ft_categories)

        target_ft_to_idx_dict,target_idx_to_ft_dict,target_fts = self.get_mutli_ts_out_target_fts(self.target_ft_categories, y_train)

        self.target_ft_to_idx_dict = target_ft_to_idx_dict
        self.target_idx_to_ft_dict = target_idx_to_ft_dict
        self.target_fts = target_fts
        # self.target_ft_categories = target_ft_categories


        return self.infoID_train_and_test_array_dict


    def configure_chosen_data_arrays(self,tag,num_timesteps_to_get):
        #========================== combine train arrays ==================================================
        x_arrays = []
        y_arrays = []
        for infoID in self.infoIDs:

            if num_timesteps_to_get == "all":
                x = self.infoID_train_and_test_array_dict[infoID]["x_%s"%tag]
                y = self.infoID_train_and_test_array_dict[infoID]["y_%s"%tag]
            else:
                x = self.infoID_train_and_test_array_dict[infoID]["x_%s"%tag][:num_timesteps_to_get, :]
                y = self.infoID_train_and_test_array_dict[infoID]["y_%s"%tag][:num_timesteps_to_get, :]

            x_arrays.append(x)
            y_arrays.append(y)
        x = np.concatenate(x_arrays, axis=0)
        y = np.concatenate(y_arrays, axis=0)
        print("\nx shape")
        print(x.shape)
        print("\ny shape")
        print(y.shape)

        return x,y

    def log_norm_x_arrays(self, x):

        for LOG_NORM in range(self.INPUT_TWITTER_LOGNORM):
            print("\nx shape before transpose: %s"%str(x.shape))
            x = x.T
            print("x shape after transpose: %s"%str(x.shape))

            for ft,ft_idx in self.input_dynamic_and_static_fts_to_idx_dict.items():
                if "twitter" in ft:

                    if self.LOGNORM_DEBUG_PRINT==True:
                        print("\nlog norm %s"%ft)
                        print(x[ft_idx])
                    x[ft_idx] =  np.log1p(x[ft_idx])

                    if self.LOGNORM_DEBUG_PRINT==True:
                        print(x[ft_idx])

            x = x.T
            print("x with original shape: %s"%str(x.shape))

        for LOG_NORM in range(self.INPUT_OTHER_LOGNORM):
            print("\nx shape before transpose: %s"%str(x.shape))
            x = x.T
            print("x shape after transpose: %s"%str(x.shape))

            for ft,ft_idx in self.input_dynamic_and_static_fts_to_idx_dict.items():
                if ("twitter" not in ft):
                    ft_is_valid=True
                    for ignore_tag in self.LOGNORM_IGNORE_TAGS:
                        if ignore_tag in ft:
                            ft_is_valid=False

                    if ft_is_valid== True:
                        if self.LOGNORM_DEBUG_PRINT==True:
                            print("\nlog norm %s"%ft)
                            print(x[ft_idx])
                        x[ft_idx] =  np.log1p(x[ft_idx])
                        if self.LOGNORM_DEBUG_PRINT==True:
                            print(x[ft_idx])
                    else:
                        print("\ninvalid ft: %s"%ft)
                        print(ft_idx)
                        # sys.exit(0)

            x = x.T
            print("x with original shape: %s"%str(x.shape))

        return x

    def log_norm_y_arrays(self, y):

        for LOG_NORM in range(self.OUTPUT_TWITTER_LOGNORM):
            print("\ny shape before transpose: %s"%str(y.shape))
            y = y.T
            print("y shape after transpose: %s"%str(y.shape))

            for ft,ft_idx in self.target_ft_to_idx_dict.items():
                if "twitter" in ft:
                    if self.LOGNORM_DEBUG_PRINT==True:
                        print("\nlog norm %s"%ft)
                        print(y[ft_idx])
                    y[ft_idx] =  np.log1p(y[ft_idx])
                    if self.LOGNORM_DEBUG_PRINT==True:
                        print(y[ft_idx])

            y = y.T
            print("y with original shape: %s"%str(y.shape))

        for LOG_NORM in range(self.OUTPUT_OTHER_LOGNORM):
            print("\ny shape before transpose: %s"%str(y.shape))
            y = y.T
            print("y shape after transpose: %s"%str(y.shape))

            for ft,ft_idx in self.target_ft_to_idx_dict.items():
                if ("twitter" not in ft):
                    ft_is_valid=True
                    for ignore_tag in self.LOGNORM_IGNORE_TAGS:
                        if ignore_tag in ft:
                            ft_is_valid=False

                    if ft_is_valid== True:
                        if self.LOGNORM_DEBUG_PRINT==True:
                            print("\nlog norm %s"%ft)
                            print(y[ft_idx])
                        y[ft_idx] =  np.log1p(y[ft_idx])
                        if self.LOGNORM_DEBUG_PRINT==True:
                            print(y[ft_idx])


            y = y.T
            print("y with original shape: %s"%str(y.shape))

        return y

    def reverse_log_norm_x_arrays(self, x):

        for LOG_NORM in range(self.INPUT_TWITTER_LOGNORM):
            print("\nx shape before transpose: %s"%str(x.shape))
            x = x.T
            print("x shape after transpose: %s"%str(x.shape))

            for ft,ft_idx in self.input_dynamic_and_static_fts_to_idx_dict.items():
                if "twitter" in ft:
                    if self.LOGNORM_DEBUG_PRINT==True:
                        print("\nexpm1 %s"%ft)
                        print(x[ft_idx])
                    x[ft_idx] =  np.expm1(x[ft_idx])
                    if self.LOGNORM_DEBUG_PRINT==True:
                        print(x[ft_idx])

            x = x.T
            print("x with original shape: %s"%str(x.shape))

        for LOG_NORM in range(self.INPUT_OTHER_LOGNORM):
            print("\nx shape before transpose: %s"%str(x.shape))
            x = x.T
            print("x shape after transpose: %s"%str(x.shape))

            for ft,ft_idx in self.input_dynamic_and_static_fts_to_idx_dict.items():
                if ("twitter" not in ft):
                    ft_is_valid=True
                    for ignore_tag in self.LOGNORM_IGNORE_TAGS:
                        if ignore_tag in ft:
                            ft_is_valid=False

                    if ft_is_valid== True:
                        if self.LOGNORM_DEBUG_PRINT==True:
                            print("\nexpm1 %s"%ft)
                            print(x[ft_idx])
                        x[ft_idx] =  np.expm1(x[ft_idx])
                        if self.LOGNORM_DEBUG_PRINT==True:
                            print(x[ft_idx])

            x = x.T
            print("x with original shape: %s"%str(x.shape))

        return x

    def reverse_log_norm_y_arrays(self, y):

        for LOG_NORM in range(self.OUTPUT_TWITTER_LOGNORM):
            print("\ny shape before transpose: %s"%str(y.shape))
            y = y.T
            print("y shape after transpose: %s"%str(y.shape))

            for ft,ft_idx in self.target_ft_to_idx_dict.items():
                if "twitter" in ft:
                    if self.LOGNORM_DEBUG_PRINT==True:
                        print("\nexpm1 %s"%ft)
                        print(y[ft_idx])
                    y[ft_idx] =  np.expm1(y[ft_idx])
                    if self.LOGNORM_DEBUG_PRINT==True:
                        print(y[ft_idx])

            y = y.T
            print("y with original shape: %s"%str(y.shape))

        for LOG_NORM in range(self.OUTPUT_OTHER_LOGNORM):
            print("\ny shape before transpose: %s"%str(y.shape))
            y = y.T
            print("y shape after transpose: %s"%str(y.shape))

            for ft,ft_idx in self.target_ft_to_idx_dict.items():
                if ("twitter" not in ft):
                    ft_is_valid=True
                    for ignore_tag in self.LOGNORM_IGNORE_TAGS:
                        if ignore_tag in ft:
                            ft_is_valid=False

                    if ft_is_valid== True:
                        if self.LOGNORM_DEBUG_PRINT==True:
                            print("\nexpm1 %s"%ft)
                            print(y[ft_idx])
                        y[ft_idx] =  np.expm1(y[ft_idx])
                        if self.LOGNORM_DEBUG_PRINT==True:
                            print(y[ft_idx])


            y = y.T
            print("y with original shape: %s"%str(y.shape))

        return y



    def configure_training_data(self):

        self.x_train,self.y_train = self.configure_chosen_data_arrays("train","all")
        print("\nself.x_train shape")
        print(self.x_train.shape)
        print("\nself.y_train shape")
        print(self.y_train.shape)

        return self.x_train,self.y_train

    def normalize_data_and_return_scalers(self, x, y):

        x = self.log_norm_x_arrays(x)
        y = self.log_norm_y_arrays(y)

        if self.RESCALE == True:
            x,self.x_scaler = normalize_single_train_only_data_v2_standard_option(x, self.SCALER_TYPE,feature_range=self.FEATURE_RANGE)

        if self.RESCALE_TARGET == True:
            y,self.y_scaler = normalize_single_train_only_data_v2_standard_option(y, self.SCALER_TYPE,feature_range=self.FEATURE_RANGE)

        return x,y,self.x_scaler,self.y_scaler

    def normalize_train_data_and_return_scalers(self):

        self.x_train = self.log_norm_x_arrays(self.x_train)
        self.y_train = self.log_norm_y_arrays(self.y_train)

        self.x_scaler=None
        if self.RESCALE == True:
            self.x_train,self.x_scaler = normalize_single_train_only_data_v2_standard_option(self.x_train, self.SCALER_TYPE,feature_range=self.FEATURE_RANGE)

        self.y_scaler=None
        if self.RESCALE_TARGET == True:
            self.y_train,self.y_scaler = normalize_single_train_only_data_v2_standard_option(self.y_train, self.SCALER_TYPE,feature_range=self.FEATURE_RANGE)

        return self.x_train,self.y_train,self.x_scaler,self.y_scaler



    def configure_training_data_and_normalize(self):

        self.x_train,self.y_train = self.configure_chosen_data_arrays("train","all")
        print("\nself.x_train shape")
        print(self.x_train.shape)
        print("\nself.y_train shape")
        print(self.y_train.shape)

        self.x_train = self.log_norm_x_arrays(self.x_train)
        self.y_train = self.log_norm_y_arrays(self.y_train)

        if self.RESCALE == True:
            self.x_train,self.x_scaler = normalize_single_train_only_data_v2_standard_option(self.x_train, self.SCALER_TYPE,feature_range=self.FEATURE_RANGE)
            # self.X_SCALER_OUTPUT_FP = self.output_dir +"x_scaler.save"
            # save_scaler(self.x_scaler,self.X_SCALER_OUTPUT_FP)

        if self.RESCALE_TARGET == True:
            self.y_train,self.y_scaler = normalize_single_train_only_data_v2_standard_option(self.y_train, self.SCALER_TYPE,feature_range=self.FEATURE_RANGE)
            # self.Y_SCALER_OUTPUT_FP = self.output_dir +"y_scaler.save"
            # save_scaler(self.y_scaler,self.Y_SCALER_OUTPUT_FP)

        return self.x_train,self.y_train

    def normalize_data(self,x,y):

        x = self.log_norm_x_arrays(x)
        y = self.log_norm_y_arrays(y)

        if self.RESCALE == True:
            x = normalize_single_array_with_scaler(x,self.x_scaler)

        if self.RESCALE_TARGET == True:
            y = normalize_single_array_with_scaler(y,self.y_scaler)

        return x,y

    def inverse_normalize_data(self,x,y):

        if self.RESCALE == True:
            x = denormalize_single_array(x,self.x_scaler)

        if self.RESCALE_TARGET == True:
            y = denormalize_single_array(y,self.y_scaler)

        x = self.reverse_log_norm_x_arrays(x)
        y = self.reverse_log_norm_y_arrays(y)



        return x,y

    def normalize_x_array(self, x):
        x = self.log_norm_x_arrays(x)
        if self.RESCALE == True:
            x = normalize_single_array_with_scaler(x,self.x_scaler)
        return x

    def normalize_y_array(self, y):
        y = self.log_norm_y_arrays(y)
        if self.RESCALE_TARGET == True:
            y = normalize_single_array_with_scaler(y,self.y_scaler)
        return y

    def inverse_normalize_x_data(self,x):

        if self.RESCALE == True:
            x = denormalize_single_array(x,self.x_scaler)

        x = self.reverse_log_norm_x_arrays(x)
        return x

    def inverse_normalize_y_data(self,y):
        if self.RESCALE_TARGET == True:
            y = denormalize_single_array(y,self.y_scaler)

        y = self.reverse_log_norm_y_arrays(y)
        return y



    def configure_val_or_test_data_and_normalize(self, tag, num_timesteps_to_get):
        x,y = self.configure_chosen_data_arrays(tag,num_timesteps_to_get)
        print("\nx shape")
        print(x.shape)
        print("\ny shape")
        print(y.shape)

        #normalize
        x, y = self.normalize_data(x,y)

        return x, y

    def configure_val_or_test_data(self, tag, num_timesteps_to_get):
        x,y = self.configure_chosen_data_arrays(tag,num_timesteps_to_get)
        print("\nx shape")
        print(x.shape)
        print("\ny shape")
        print(y.shape)

        return x, y



    # def reverse_normalize_data(self, x, y):



    def get_all_data_arrays_from_get_infoID_train_and_test_array_dict(self):

        #========================== combine train arrays ==================================================
        x_train_arrays = []
        y_train_arrays = []
        for infoID in self.infoIDs:
            x_train = self.infoID_train_and_test_array_dict[infoID]["x_train"]
            y_train = self.infoID_train_and_test_array_dict[infoID]["y_train"]
            x_train_arrays.append(x_train)
            y_train_arrays.append(y_train)
        x_train = np.concatenate(x_train_arrays, axis=0)
        y_train = np.concatenate(y_train_arrays, axis=0)
        print("\nx train shape")
        print(x_train.shape)
        print("\ny train shape")
        print(y_train.shape)



        #========================== combine test arrays ==================================================
        x_val_arrays = []
        y_val_arrays = []
        for infoID in self.infoIDs:
            x_val = self.infoID_train_and_test_array_dict[infoID]["x_val"]
            y_val = self.infoID_train_and_test_array_dict[infoID]["y_val"]
            x_val_arrays.append(np.copy(x_val))
            y_val_arrays.append(np.copy(y_val))
        x_val = np.concatenate(x_val_arrays, axis=0)
        y_val = np.concatenate(y_val_arrays, axis=0)
        print("\nx val shape")
        print(x_val.shape)
        print("\ny val shape")
        print(y_val.shape)

        #========================== combine test arrays ==================================================
        x_test_arrays = []
        y_test_arrays = []
        for infoID in self.infoIDs:
            x_test = self.infoID_train_and_test_array_dict[infoID]["x_test"]
            y_test = self.infoID_train_and_test_array_dict[infoID]["y_test"]
            x_test_arrays.append(np.copy(x_test))
            y_test_arrays.append(np.copy(y_test))
        x_test = np.concatenate(x_test_arrays, axis=0)
        y_test = np.concatenate(y_test_arrays, axis=0)
        print("\nx test shape")
        print(x_test.shape)
        print("\ny test shape")
        print(y_test.shape)

        #========================== combine val sliding window arrays ==================================================
        x_val_sliding_window_arrays_list = []
        y_val_sliding_window_arrays_list = []
        for infoID in self.infoIDs:
            x_val_sliding_window  = self.infoID_train_and_test_array_dict[infoID]["x_val_sliding_window"]
            y_val_sliding_window  = self.infoID_train_and_test_array_dict[infoID]["y_val_sliding_window"]
            x_val_sliding_window_arrays_list.append(np.copy(x_val_sliding_window))
            y_val_sliding_window_arrays_list.append(np.copy(y_val_sliding_window))
        x_val_sliding_window = np.concatenate(x_val_sliding_window_arrays_list, axis=0)
        y_val_sliding_window = np.concatenate(y_val_sliding_window_arrays_list, axis=0)
        print("\nx_val_sliding_window_arrays shape")
        print(x_val_sliding_window.shape)
        print("\ny_val_sliding_window_arrays shape")
        print(y_val_sliding_window.shape)

        #========================== combine test sliding window arrays ==================================================
        x_test_sliding_window_arrays_list = []
        y_test_sliding_window_arrays_list = []
        for infoID in self.infoIDs:
            x_test_sliding_window  = self.infoID_train_and_test_array_dict[infoID]["x_test_sliding_window"]
            y_test_sliding_window  = self.infoID_train_and_test_array_dict[infoID]["y_test_sliding_window"]
            x_test_sliding_window_arrays_list.append(np.copy(x_test_sliding_window))
            y_test_sliding_window_arrays_list.append(np.copy(y_test_sliding_window))
        x_test_sliding_window = np.concatenate(x_test_sliding_window_arrays_list, axis=0)
        y_test_sliding_window = np.concatenate(y_test_sliding_window_arrays_list, axis=0)
        print("\nx_test_sliding_window_arrays shape")
        print(x_test_sliding_window.shape)
        print("\ny_test_sliding_window_arrays shape")
        print(y_test_sliding_window.shape)

        self.x_train=x_train
        self.y_train=y_train
        self.x_val=x_val
        self.y_val=y_val
        self.x_test=x_test
        self.y_test=y_test
        self.x_val_sliding_window=x_val_sliding_window
        self.y_val_sliding_window=y_val_sliding_window
        self.x_test_sliding_window=x_test_sliding_window
        self.y_test_sliding_window=y_test_sliding_window

    def get_features(self):
        self.get_feature_name_lists()
        self.generate_platform_infoID_pair_samples()
        self.get_infoID_train_and_test_array_dict()
        # self.get_all_data_arrays_from_get_infoID_train_and_test_array_dict()
        self.configure_training_data()

        self.x_val,self.y_val = self.configure_val_or_test_data("val", "all")
        self.x_test,self.y_test = self.configure_val_or_test_data("test", "all")
        self.x_val_sliding_window,self.y_val_sliding_window = self.configure_val_or_test_data("val_sliding_window", "all")
        self.x_test_sliding_window,self.y_test_sliding_window = self.configure_val_or_test_data("test_sliding_window", "all")

    def configure_features_and_get_infoID_train_and_test_array_dict_v2_fixed_dates(self):
        self.get_feature_name_lists()
        self.generate_platform_infoID_pair_samples_v2_1_df()


        self.get_infoID_train_and_test_array_dict_v2_fixed_dates()
        return self.infoID_train_and_test_array_dict

    def create_samples_and_get_date_lists(self, x, y, all_dates, start, end, array_tag):

        start = pd.to_datetime(start, utc=True)

        end = pd.to_datetime(end, utc=True)
        end = pd.Series([end])
        # end = end - self.INPUT_HOURS
        end = end.dt.floor("H")
        if "sliding_window" not in array_tag:
            end = end.dt.floor("D")
        end = end[0]
        print(end)
        # sys.exit(0)
        all_dates = pd.Series(all_dates)
        all_dates = pd.to_datetime(all_dates, utc=True)

        i=0
        for date in all_dates:
            if date==start:
                start_idx = i
            if date==end:
                end_idx = i
                if "sliding_window" in array_tag:
                    end_idx = i - self.OUTPUT_HOURS + 1
                    end = all_dates[end_idx]

            i+=1

        print("\nFor %s: with start and end %s to %s; start_idx is %d, end_idx is %d"%(array_tag, str(start), str(end), start_idx , end_idx))

        new_x = x.copy()
        new_y = y.copy()
        new_x = new_x[start_idx:end_idx+1 , :]
        new_y = new_y[start_idx:end_idx+1 , :]
        print("\nNew array shapes for %s"%array_tag)
        print(new_x.shape)
        print(new_y.shape)


        if "sliding_window" in array_tag:
            dates = pd.date_range(start, end, freq="H")
        else:
            dates = pd.date_range(start, end, freq="D")

        print(len(dates))
        print(new_x.shape[0])

        if array_tag != "test_sliding_window":
            if len(dates)!= new_x.shape[0]:
                print("\nError! len(dates)!= new_x.shape[0]")
                sys.exit(0)

        if len(dates)!= new_x.shape[0]:
            print("\nError! len(dates)!= new_x.shape[0]")
            sys.exit(0)

        return new_x, new_y, (start, end)

    def get_infoID_train_and_test_array_dict_v2_fixed_dates(self):
        infoID_train_and_test_array_dict = {}
        num_infoIDs = len(self.infoIDs)
        print("\nnum_infoIDs: %d"%num_infoIDs)


        #self.infoID_to_df_dict
        #make sliding window samples and non sliding window samples


        i=1
        for infoID in self.infoIDs:
            #get dfs
            # train_df = self.infoID_to_df_dict[infoID]["train_df"]
            # val_df = self.infoID_to_df_dict[infoID]["val_df"]
            # test_df = self.infoID_to_df_dict[infoID]["test_df"]

            df = self.infoID_to_df_dict[infoID]
            NUM_PROPER_SLIDING_WINDOW_SAMPLES = df.shape[0] - self.OUTPUT_HOURS - (self.OUTPUT_HOURS * self.lookback_in_days) + 1
            print("\nNUM_PROPER_SLIDING_WINDOW_SAMPLES: %d"%NUM_PROPER_SLIDING_WINDOW_SAMPLES)


            print(df)

            #============================== get all sliding window samples ==============================

            x_sliding_window, y_sliding_window, ft_to_idx_dict=convert_df_to_sliding_window_form_diff_x_and_y_fts_no_print_with_idx_dict(df, self.INPUT_SIZE, self.EXTRACT_OUTPUT_SIZE, self.dynamic_fts, self.target_ft_categories)
            print("\nx and y sliding window arrays")
            print(x_sliding_window.shape)
            print(y_sliding_window.shape)
            x_sliding_window,dynamic_and_static_fts_to_idx_dict,idx_to_dynamic_and_static_fts_dict,dynamic_and_static_fts = insert_static_1hot_fts_for_platform_samples(infoID, x_sliding_window, self.dynamic_fts,ft_to_idx_dict ,self.static_fts,self.infoIDs,self.GET_1HOT_INFO_ID_FTS)


            if x_sliding_window.shape[0] != NUM_PROPER_SLIDING_WINDOW_SAMPLES:
                print("\nError! x_sliding_window.shape[0] != NUM_PROPER_SLIDING_WINDOW_SAMPLES")
                sys.exit(0)

            if self.AGG_TIME_SERIES == True:
                y_sliding_window = self.agg_y_array_v2_mult_output_fts(y_sliding_window)
                print("\nfirst y_sliding_window after agg")
                print(y_sliding_window[0])
            else:
                y_sliding_window = y_sliding_window.reshape((y_sliding_window.shape[0],y_sliding_window.shape[1] *y_sliding_window.shape[2] ))

            print("\nx and y train arrays")
            print(x_sliding_window.shape)
            print(y_sliding_window.shape)


            # self.eval_dates = pd.date_range(train_start, test_end, freq=self.GRAN[-1])
            # self.eval_dates_daily = pd.date_range(abs_train_start, test_end, freq="D")

            #============================== get all non sliding window samples ==============================
            NUM_PROPER_NONSLIDING_WINDOW_SAMPLES = len(self.eval_dates_daily)
            print(NUM_PROPER_NONSLIDING_WINDOW_SAMPLES)


            x_nonsliding_window,y_nonsliding_window = convert_df_to_test_sliding_window_form_diff_x_and_y_fts_no_print(df, self.INPUT_SIZE, self.EXTRACT_OUTPUT_SIZE,self.dynamic_fts, self.target_ft_categories,MOD_NUM=1000)
            x_nonsliding_window,_,_,_ = insert_static_1hot_fts_for_platform_samples(infoID,x_nonsliding_window, self.dynamic_fts,ft_to_idx_dict ,self.static_fts,self.infoIDs,self.GET_1HOT_INFO_ID_FTS)
            if self.AGG_TIME_SERIES == True:
                y_nonsliding_window = self.agg_y_array_v2_mult_output_fts(y_nonsliding_window)
            else:
                y_nonsliding_window = y_nonsliding_window.reshape((y_nonsliding_window.shape[0],y_nonsliding_window.shape[1] *y_nonsliding_window.shape[2] ))
            print("\nx and y nonsliding_window arrays")
            print(x_nonsliding_window.shape)
            print(y_nonsliding_window.shape)

            print(NUM_PROPER_NONSLIDING_WINDOW_SAMPLES)
            print(x_nonsliding_window.shape[0])
            if NUM_PROPER_NONSLIDING_WINDOW_SAMPLES != x_nonsliding_window.shape[0]:
                print("\nError! NUM_PROPER_NONSLIDING_WINDOW_SAMPLES != x_nonsliding_window.shape[0] ")
                sys.exit(0)



            x_train, y_train, train_y_target_dates = self.create_samples_and_get_date_lists(x_sliding_window, y_sliding_window, self.eval_dates, self.train_start, self.train_end, "train_sliding_window")
            x_val_sliding_window, y_val_sliding_window, val_sliding_window_y_target_dates = self.create_samples_and_get_date_lists(x_sliding_window, y_sliding_window, self.eval_dates, self.val_start, self.val_end, "val_sliding_window")
            x_test_sliding_window, y_test_sliding_window, test_sliding_window_y_target_dates = self.create_samples_and_get_date_lists(x_sliding_window, y_sliding_window, self.eval_dates, self.test_start, self.test_end, "test_sliding_window")
            x_val, y_val, val_y_target_dates = self.create_samples_and_get_date_lists(x_nonsliding_window, y_nonsliding_window, self.eval_dates_daily, self.val_start, self.val_end, "val")
            x_test, y_test, test_y_target_dates = self.create_samples_and_get_date_lists(x_nonsliding_window, y_nonsliding_window, self.eval_dates_daily, self.test_start, self.test_end, "test")

            # sys.exit(0)

            self.target_date_tuple_dict = {}
            self.target_date_tuple_dict["y_train"] = train_y_target_dates
            self.target_date_tuple_dict["y_val_sliding_window"] = val_sliding_window_y_target_dates
            self.target_date_tuple_dict["y_test_sliding_window"] = test_sliding_window_y_target_dates
            self.target_date_tuple_dict["y_val"] = val_y_target_dates
            self.target_date_tuple_dict["y_test"] = test_y_target_dates


            if self.DARPA_INPUT_FORMAT == True:
                x_train, new_ft_to_idx_dict,new_idx_to_ft_dict,new_dynamic_fts,new_static_and_dynamic_fts = self.remove_platform_fts(x_train, "train",dynamic_and_static_fts_to_idx_dict,idx_to_dynamic_and_static_fts_dict,dynamic_and_static_fts)
                x_val, new_ft_to_idx_dict,new_idx_to_ft_dict,new_dynamic_fts,new_static_and_dynamic_fts = self.remove_platform_fts(x_val, "val",dynamic_and_static_fts_to_idx_dict,idx_to_dynamic_and_static_fts_dict,dynamic_and_static_fts)
                x_test, new_ft_to_idx_dict,new_idx_to_ft_dict,new_dynamic_fts,new_static_and_dynamic_fts = self.remove_platform_fts(x_test, "test",dynamic_and_static_fts_to_idx_dict,idx_to_dynamic_and_static_fts_dict,dynamic_and_static_fts)
                x_test_sliding_window, new_ft_to_idx_dict,new_idx_to_ft_dict,new_dynamic_fts,new_static_and_dynamic_fts = self.remove_platform_fts(x_test_sliding_window, "test_sliding_window",dynamic_and_static_fts_to_idx_dict,idx_to_dynamic_and_static_fts_dict,dynamic_and_static_fts)
                x_val_sliding_window, new_ft_to_idx_dict,new_idx_to_ft_dict,new_dynamic_fts,new_static_and_dynamic_fts = self.remove_platform_fts(x_val_sliding_window, "val_sliding_window",dynamic_and_static_fts_to_idx_dict,idx_to_dynamic_and_static_fts_dict,dynamic_and_static_fts)

                dynamic_and_static_fts_to_idx_dict = dict(new_ft_to_idx_dict)
                idx_to_dynamic_and_static_fts_dict = dict(new_idx_to_ft_dict)
                dynamic_and_static_fts = list(new_static_and_dynamic_fts)

            #data info
            infoID_train_and_test_array_dict[infoID]={}
            infoID_train_and_test_array_dict[infoID]["x_train"] = x_train
            infoID_train_and_test_array_dict[infoID]["y_train"] = y_train
            infoID_train_and_test_array_dict[infoID]["x_val"] = x_val
            infoID_train_and_test_array_dict[infoID]["y_val"] = y_val
            infoID_train_and_test_array_dict[infoID]["x_test"] = x_test
            infoID_train_and_test_array_dict[infoID]["y_test"] = y_test

            infoID_train_and_test_array_dict[infoID]["x_val_sliding_window"] = x_val_sliding_window
            infoID_train_and_test_array_dict[infoID]["y_val_sliding_window"] = y_val_sliding_window

            infoID_train_and_test_array_dict[infoID]["x_test_sliding_window"] = x_test_sliding_window
            infoID_train_and_test_array_dict[infoID]["y_test_sliding_window"] = y_test_sliding_window


            print("Got infoID array %d of %d" %(i, num_infoIDs))
            i+=1


        #fix input fts
        # input_fts = list(dynamic_and_static_fts)

        print("\nx and y train sizes")
        print(x_train.shape)
        print(y_train.shape)

        print("\nx and y val sizes")
        print(x_val.shape)
        print(y_val.shape)

        print("\nx and y test sizes")
        print(x_test.shape)
        print(y_test.shape)

        print("\nx and y val sliding window sizes")
        print(x_val_sliding_window.shape)
        print(y_val_sliding_window.shape)

        print("\nx and y test sliding window sizes")
        print(x_test_sliding_window.shape)
        print(y_test_sliding_window.shape)

        # print("\ntransposed matrices")
        # print(x_test_sliding_window.T.shape)
        # print(y_test_sliding_window.T.shape)

        # self.input_dynamic_and_static_fts_to_idx_dict=dynamic_and_static_fts_to_idx_dict
        # self.input_idx_to_dynamic_and_static_fts_dict=idx_to_dynamic_and_static_fts_dict
        # self.input_dynamic_and_static_fts=dynamic_and_static_fts

        self.input_dynamic_and_static_fts_to_idx_dict=dynamic_and_static_fts_to_idx_dict
        self.input_idx_to_dynamic_and_static_fts_dict=idx_to_dynamic_and_static_fts_dict
        self.input_dynamic_and_static_fts=dynamic_and_static_fts
        self.infoID_train_and_test_array_dict = infoID_train_and_test_array_dict

        # self.target_ft_categories = list(self.target_ft_categories)

        target_ft_to_idx_dict,target_idx_to_ft_dict,target_fts = self.get_mutli_ts_out_target_fts(self.target_ft_categories, y_train)

        self.target_ft_to_idx_dict = target_ft_to_idx_dict
        self.target_idx_to_ft_dict = target_idx_to_ft_dict
        self.target_fts = target_fts
        # self.target_ft_categories = target_ft_categories


        return self.infoID_train_and_test_array_dict

    def configure_features_and_get_infoID_train_and_test_array_dict(self):
        self.get_feature_name_lists()
        self.generate_platform_infoID_pair_samples()
        self.get_infoID_train_and_test_array_dict()

        return self.infoID_train_and_test_array_dict

    def save_scalers_and_metadata(self, output_dir):

        create_output_dir(output_dir)

        if self.RESCALE==True:
            self.X_SCALER_OUTPUT_FP = output_dir +"x_scaler.save"
            save_scaler(self.x_scaler,self.X_SCALER_OUTPUT_FP)

        if self.RESCALE_TARGET==True:
            self.Y_SCALER_OUTPUT_FP = output_dir +"y_scaler.save"
            save_scaler(self.y_scaler,self.Y_SCALER_OUTPUT_FP)

        save_pickle(self.input_dynamic_and_static_fts_to_idx_dict, output_dir+"input_dynamic_and_static_fts_to_idx_dict")
        save_pickle(self.input_idx_to_dynamic_and_static_fts_dict, output_dir+"input_idx_to_dynamic_and_static_fts_dict")

        save_ft_list_as_text_file(output_dir, "input_dynamic_and_static_fts",self.input_dynamic_and_static_fts)
        save_ft_list_as_text_file(output_dir, "input_dynamic_ft_categories",self.input_dynamic_ft_categories)
        save_ft_list_as_text_file(output_dir, "static_fts",self.static_fts)
        save_ft_list_as_text_file(output_dir, "input_ft_categories",self.input_ft_categories)
        save_ft_list_as_text_file(output_dir, "target_ft_categories",self.target_ft_categories)



        # self.target_idx_to_ft_dict =target_idx_to_ft_dict
        # self.target_ft_to_idx_dict =target_ft_to_idx_dict

        save_pickle(self.target_idx_to_ft_dict, output_dir+"target_idx_to_ft_dict")
        save_pickle(self.target_ft_to_idx_dict, output_dir+"target_ft_to_idx_dict")

    def save_ft_lists(self, output_dir):

        create_output_dir(output_dir)
        save_ft_list_as_text_file(output_dir, "input_dynamic_and_static_fts",self.input_dynamic_and_static_fts)
        save_ft_list_as_text_file(output_dir, "input_dynamic_ft_categories",self.input_dynamic_ft_categories)
        save_ft_list_as_text_file(output_dir, "input_static_fts",self.static_fts)
        save_ft_list_as_text_file(output_dir, "input_ft_categories",self.input_ft_categories)
        save_ft_list_as_text_file(output_dir, "target_ft_categories",self.target_ft_categories)

    def save_params(self, output_dir):
        param_fp = output_dir + "ft-preprocess-params.csv"
        self.ft_preproc_param_df.to_csv(param_fp,index=False)

    def get_target_ft_category_to_list_of_temporal_indices(self):
        target_ft_category_to_list_of_temporal_indices_dict = {}

        for cur_target_ft_category in self.target_ft_categories:
            target_ft_category_to_list_of_temporal_indices_dict[cur_target_ft_category] = {}
            cur_target_ft_temporal_indices_list = []
            cur_target_ft_temporal_ft_name_list = []

            for i,cur_ft in enumerate(self.input_dynamic_and_static_fts):
                if cur_target_ft_category in cur_ft:
                    # print(cur_target_ft_category)
                    cur_target_ft_temporal_ft_name_list.append(cur_ft)
                    cur_target_ft_temporal_indices_list.append(i)

                    # print("ft: %s, idx: %d"%(cur_ft, i))
            target_ft_category_to_list_of_temporal_indices_dict[cur_target_ft_category]["ft_name_list"]=cur_target_ft_temporal_ft_name_list
            target_ft_category_to_list_of_temporal_indices_dict[cur_target_ft_category]["indices_list"]=cur_target_ft_temporal_indices_list

        self.target_ft_category_to_list_of_temporal_indices_dict=target_ft_category_to_list_of_temporal_indices_dict

        return self.target_ft_category_to_list_of_temporal_indices_dict

    def get_input_ft_category_to_list_of_temporal_indices(self):
        input_ft_category_to_list_of_temporal_indices_dict = {}

        for cur_input_ft_category in self.input_dynamic_ft_categories:
            input_ft_category_to_list_of_temporal_indices_dict[cur_input_ft_category] = {}
            cur_input_ft_temporal_indices_list = []
            cur_input_ft_temporal_ft_name_list = []

            for i,cur_ft in enumerate(self.input_dynamic_and_static_fts):
                if cur_input_ft_category in cur_ft:
                    # print(cur_input_ft_category)
                    cur_input_ft_temporal_ft_name_list.append(cur_ft)
                    cur_input_ft_temporal_indices_list.append(i)

                    # print("ft: %s, idx: %d"%(cur_ft, i))
            input_ft_category_to_list_of_temporal_indices_dict[cur_input_ft_category]["ft_name_list"]=cur_input_ft_temporal_ft_name_list
            input_ft_category_to_list_of_temporal_indices_dict[cur_input_ft_category]["indices_list"]=cur_input_ft_temporal_indices_list

        self.input_ft_category_to_list_of_temporal_indices_dict=input_ft_category_to_list_of_temporal_indices_dict

        # sys.exit(0)

        return self.input_ft_category_to_list_of_temporal_indices_dict

    def convert_1_x_array_to_temporal_ft_df(self, array_tag, x):

        x_array_tag = "x_" + array_tag

        x = x[:1]
        print("\n%s array shape"%x_array_tag)
        print(x.shape)

        #flatten it
        x = x.flatten()
        print("flat shape: %s"%str(x.shape))

        #make dict for input fts
        ft_category_to_array_dict = {}
        print("\nInitializing ft_category_to_array_dict...")
        for cur_input_ft_category in self.input_dynamic_ft_categories:
            print(cur_input_ft_category)
            ft_category_to_array_dict[cur_input_ft_category] = []

        #put them into dict
        for cur_input_ft_category in self.input_dynamic_ft_categories:
            cur_input_ft_indices = self.input_ft_category_to_list_of_temporal_indices_dict[cur_input_ft_category]["indices_list"]
            x_cur_input_ft_array = np.take(x, cur_input_ft_indices)
            ft_category_to_array_dict[cur_input_ft_category] = x_cur_input_ft_array
            print()
            print(cur_input_ft_category)
            print(x_cur_input_ft_array)

        #make a df
        dynamic_input_ft_df = pd.DataFrame(data=ft_category_to_array_dict)
        dynamic_input_ft_df = dynamic_input_ft_df[self.input_dynamic_ft_categories]
        print("\ndynamic_input_ft_df")
        print(dynamic_input_ft_df)

        return dynamic_input_ft_df

    def convert_1_y_array_to_temporal_ft_df(self, array_tag, y):

        y_array_tag = "y_" + array_tag

        y = y[:1]
        print("\n%s array shape"%y_array_tag)
        print(y.shape)

        #reshape
        y = y.flatten()
        y = y.reshape((self.INPUT_HOURS, len(self.target_ft_categories))).T
        print("new shape: %s"%str(y.shape))

        #make dict for input fts
        ft_category_to_array_dict = {}
        print("\nInitializing ft_category_to_array_dict...")

        for i,cur_target_ft_category in enumerate(self.target_ft_categories):
            print(cur_target_ft_category)
            ft_category_to_array_dict[cur_target_ft_category] = y[i].flatten()


        dynamic_target_ft_df = pd.DataFrame(data=ft_category_to_array_dict)
        dynamic_target_ft_df=dynamic_target_ft_df[self.target_ft_categories]

        print("\ndynamic_target_ft_df")
        print(dynamic_target_ft_df)

        return dynamic_target_ft_df

    def convert_data_arrays_to_dynamic_ft_dfs(self, array_tag):

        x_array_tag = "x_" + array_tag
        y_array_tag = "y_" + array_tag

        overall_timestep_to_dynamic_ft_df_dict = {}

        for infoID in self.infoIDs:
            overall_timestep_to_dynamic_ft_df_dict[infoID] = {}
            x = self.infoID_train_and_test_array_dict[infoID][x_array_tag]

            # cur_ft_dict = {}

            print(self.input_dynamic_ft_categories)
            # sys.exit(0)
            for cur_input_ft_category in self.input_dynamic_ft_categories:

                print("\ncur x shape")
                print(x.shape)

                cur_input_ft_indices = self.input_ft_category_to_list_of_temporal_indices_dict[cur_input_ft_category]["indices_list"]
                x_cur_target_ft_category_temp_array = np.take(x, cur_input_ft_indices,axis=1)

                print("\nx_cur_target_ft_category_temp_array shape")
                print(x_cur_target_ft_category_temp_array.shape)

                all_timesteps = x_cur_target_ft_category_temp_array.shape[0]
                for overall_timestep in range(all_timesteps):
                    if overall_timestep not in overall_timestep_to_dynamic_ft_df_dict[infoID]:
                        overall_timestep_to_dynamic_ft_df_dict[infoID][overall_timestep] = pd.DataFrame(data={cur_input_ft_category:x_cur_target_ft_category_temp_array[overall_timestep]})
                        # df = overall_timestep_to_dynamic_ft_df_dict[infoID][overall_timestep]
                        # print("\noverall_timestep")
                        # print(overall_timestep)
                        # print(df)
                        # sys.exit(0)
                    else:
                        overall_timestep_to_dynamic_ft_df_dict[infoID][overall_timestep][cur_input_ft_category] = x_cur_target_ft_category_temp_array[overall_timestep]



        self.overall_timestep_to_dynamic_ft_df_dict = overall_timestep_to_dynamic_ft_df_dict
        return overall_timestep_to_dynamic_ft_df_dict

    def convert_train_arrays_to_dynamic_ft_dfs(self):
        self.train_overall_timestep_to_dynamic_ft_df_dict = self.convert_data_arrays_to_dynamic_ft_dfs("train")
        return self.train_overall_timestep_to_dynamic_ft_df_dict

    def convert_val_arrays_to_dynamic_ft_dfs(self):
        self.val_overall_timestep_to_dynamic_ft_df_dict = self.convert_data_arrays_to_dynamic_ft_dfs("val")
        return self.val_overall_timestep_to_dynamic_ft_df_dict

    def convert_test_arrays_to_dynamic_ft_dfs(self):
        self.test_overall_timestep_to_dynamic_ft_df_dict = self.convert_data_arrays_to_dynamic_ft_dfs("test")
        return self.test_overall_timestep_to_dynamic_ft_df_dict

    def convert_val_sliding_window_arrays_to_dynamic_ft_dfs(self):
        self.val_sliding_window_overall_timestep_to_dynamic_ft_df_dict = self.convert_data_arrays_to_dynamic_ft_dfs("val_sliding_window")
        return self.val_sliding_window_overall_timestep_to_dynamic_ft_df_dict

    def convert_test_sliding_window_arrays_to_dynamic_ft_dfs(self):
        self.test_sliding_window_overall_timestep_to_dynamic_ft_df_dict = self.convert_data_arrays_to_dynamic_ft_dfs("test_sliding_window")
        return self.test_sliding_window_overall_timestep_to_dynamic_ft_df_dict

    def convert_all_arrays_to_dynamic_ft_dfs_dict(self):

        array_type_to_dynamic_ft_df_dict = {}
        self.train_overall_timestep_to_dynamic_ft_df_dict=self.convert_train_arrays_to_dynamic_ft_dfs()
        self.val_overall_timestep_to_dynamic_ft_df_dict=self.convert_val_arrays_to_dynamic_ft_dfs()
        self.test_overall_timestep_to_dynamic_ft_df_dict=self.convert_test_arrays_to_dynamic_ft_dfs()
        self.val_sliding_window_overall_timestep_to_dynamic_ft_df_dict=self.convert_val_sliding_window_arrays_to_dynamic_ft_dfs()
        self.test_sliding_window_overall_timestep_to_dynamic_ft_df_dict=self.convert_test_sliding_window_arrays_to_dynamic_ft_dfs()

        array_type_to_dynamic_ft_df_dict["train"]=self.train_overall_timestep_to_dynamic_ft_df_dict
        array_type_to_dynamic_ft_df_dict["val"]=self.val_overall_timestep_to_dynamic_ft_df_dict
        array_type_to_dynamic_ft_df_dict["val_sliding_window"]=self.val_sliding_window_overall_timestep_to_dynamic_ft_df_dict
        array_type_to_dynamic_ft_df_dict["test"]=self.test_overall_timestep_to_dynamic_ft_df_dict
        array_type_to_dynamic_ft_df_dict["test_sliding_window"]=self.test_sliding_window_overall_timestep_to_dynamic_ft_df_dict
        self.array_type_to_dynamic_ft_df_dict=array_type_to_dynamic_ft_df_dict
        return array_type_to_dynamic_ft_df_dict



    def insert_correlation_features_1_at_a_time(self, array_tag,DEBUG_PRINT=False):

        def debug_print(print_val="\n"):
            if DEBUG_PRINT == True:
                print(str(print_val))

        x_array_tag = "x_" + array_tag
        y_array_tag = "y_" + array_tag

        num_infoIDs = len(self.infoIDs)

        # self.dynamic_fts = dynamic_fts
        # self.static_fts = static_fts
        # self.input_fts = input_fts
        # self.target_ft_categories = target_fts

        # self.input_dynamic_ft_categories= list(dynamic_fts)
        # self.input_ft_categories= list(input_fts)
        # self.target_ft_categories= list(target_fts)

        #get the dicts you need

        overall_timestep_to_dynamic_ft_df_dict = self.array_type_to_dynamic_ft_df_dict[array_tag]

        #create the new fts to be added
        new_corr_fts = []
        print("\nnew corr fts: \n")
        for corr_func_str in self.CORRELATION_FUNCS_TO_USE_LIST:
            for cur_target_col in self.target_ft_categories:
                new_ft = "%s_corr_%s_with_aux_time_series"%(corr_func_str, cur_target_col)
                print(new_ft)
                new_corr_fts.append(new_ft)
        # sys.exit(0)

        #make new dicts
        idx_to_corr_ft_dict = {}
        corr_ft_to_idx_dict = {}
        for idx,corr_ft in enumerate(new_corr_fts):
            idx_to_corr_ft_dict[idx]=corr_ft
            corr_ft_to_idx_dict[corr_ft]=idx

        self.idx_to_corr_ft_dict=idx_to_corr_ft_dict
        self.corr_ft_to_idx_dict=corr_ft_to_idx_dict
        self.corr_fts = new_corr_fts



        infoID_to_overall_timestep_to_corr_ft_vector_dict = {}
        infoID_to_overall_timestep_to_corr_func_to_target_ft_dict = {}
        print("\nGetting corr vectors...")
        for infoID in self.infoIDs:
            print("\nGetting corr vectors for %s..."%infoID)
            infoID_to_overall_timestep_to_corr_func_to_target_ft_dict[infoID]={}
            infoID_to_overall_timestep_to_corr_ft_vector_dict[infoID]={}
            for overall_timestep, cur_timestep_df in overall_timestep_to_dynamic_ft_df_dict[infoID].items():
                infoID_to_overall_timestep_to_corr_ft_vector_dict[infoID][overall_timestep]={}
                infoID_to_overall_timestep_to_corr_func_to_target_ft_dict[infoID][overall_timestep]={}
                debug_print(infoID)
                debug_print(overall_timestep)
                debug_print(cur_timestep_df)


                cur_avg_corr_ft_vals = []
                for corr_func_str in self.CORRELATION_FUNCS_TO_USE_LIST:
                    infoID_to_overall_timestep_to_corr_func_to_target_ft_dict[infoID][overall_timestep][corr_func_str]={}

                    corr_func = self.CORRELATION_STR_TO_FUNC_DICT[corr_func_str]
                    debug_print("\ncorr_func: ")
                    debug_print(corr_func)

                    for cur_target_col in self.target_ft_categories:

                        cur_target_corr_ft_vals = []
                        debug_print()
                        debug_print(corr_func_str)
                        debug_print(cur_target_col)

                        cur_target_ts_of_interest = cur_timestep_df[cur_target_col]
                        cur_aux_df_without_target = cur_timestep_df.copy().drop(cur_target_col, axis=1)

                        debug_print("\ncur_target_ts_of_interest")
                        debug_print(cur_target_ts_of_interest)

                        debug_print("\ncur_aux_df_without_target")
                        debug_print(cur_aux_df_without_target)

                        aux_cols = list(cur_aux_df_without_target)
                        for aux_col in aux_cols:
                            cur_aux_series = cur_aux_df_without_target[aux_col]

                            cur_corr = corr_func(cur_target_ts_of_interest, cur_aux_series)[0]
                            debug_print("%s to %s corr: %.4f"%(aux_col , cur_target_col, cur_corr))
                            # if cur_corr != np.nan:
                            cur_target_corr_ft_vals.append(cur_corr)

                        cur_target_corr_ft_vals = pd.Series(cur_target_corr_ft_vals)
                        debug_print(cur_target_corr_ft_vals)
                        cur_target_corr_ft_vals = cur_target_corr_ft_vals.dropna().values
                        debug_print("\nInfoID: %s; Timestep %d; %s to aux %s corrs: "%( infoID,overall_timestep,cur_target_col,corr_func_str ))
                        # cur_target_corr_ft_vals = cur_target_corr_ft_vals[~np.isnan(cur_target_corr_ft_vals)]
                        cur_target_corr_ft_vals = np.abs(cur_target_corr_ft_vals)
                        debug_print(cur_target_corr_ft_vals)

                        #now take the avg
                        avg_corr = np.mean(cur_target_corr_ft_vals)
                        if  np.isnan(avg_corr):
                            avg_corr = 0
                            debug_print("\navg corr was nan, so we converted it to 0")
                            # sys.exit(0)
                        infoID_to_overall_timestep_to_corr_func_to_target_ft_dict[infoID][overall_timestep][corr_func_str][cur_target_col]=avg_corr
                        cur_avg_corr_ft_vals.append(avg_corr)
                debug_print("\ncur_avg_corr_ft_vals")
                debug_print(cur_avg_corr_ft_vals)
                debug_print("\nwith fillna")
                cur_avg_corr_ft_vals = pd.Series(cur_avg_corr_ft_vals).fillna(0).values
                debug_print(cur_avg_corr_ft_vals)

                debug_print("\ncorr scores")
                for corr_ft,corr_score in zip(new_corr_fts, cur_avg_corr_ft_vals):
                    debug_print("%s: %.4f"%(corr_ft, corr_score))

                infoID_to_overall_timestep_to_corr_ft_vector_dict[infoID][overall_timestep]=cur_avg_corr_ft_vals
            print("Got corr fts for %s"%infoID)

        #make dicts
        self.infoID_to_overall_timestep_to_corr_ft_vector_dict=infoID_to_overall_timestep_to_corr_ft_vector_dict
        self.infoID_to_overall_timestep_to_corr_func_to_target_ft_dict=infoID_to_overall_timestep_to_corr_func_to_target_ft_dict

        return infoID_to_overall_timestep_to_corr_ft_vector_dict,infoID_to_overall_timestep_to_corr_func_to_target_ft_dict,idx_to_corr_ft_dict,corr_ft_to_idx_dict,new_corr_fts

    def insert_correlation_features(self, array_tag,DEBUG_PRINT=False):

        def debug_print(print_val="\n"):
            if DEBUG_PRINT == True:
                print(str(print_val))

        x_array_tag = "x_" + array_tag
        y_array_tag = "y_" + array_tag

        num_infoIDs = len(self.infoIDs)

        # self.dynamic_fts = dynamic_fts
        # self.static_fts = static_fts
        # self.input_fts = input_fts
        # self.target_ft_categories = target_fts

        # self.input_dynamic_ft_categories= list(dynamic_fts)
        # self.input_ft_categories= list(input_fts)
        # self.target_ft_categories= list(target_fts)

        #get the dicts you need

        overall_timestep_to_dynamic_ft_df_dict = self.array_type_to_dynamic_ft_df_dict[array_tag]

        #create the new fts to be added
        new_corr_fts = []
        print("\nnew corr fts: \n")
        for corr_func_str in self.CORRELATION_FUNCS_TO_USE_LIST:
            for cur_target_col in self.target_ft_categories:
                new_ft = "%s_corr_%s_with_aux_time_series"%(corr_func_str, cur_target_col)
                print(new_ft)
                new_corr_fts.append(new_ft)
        # sys.exit(0)

        #make new dicts
        idx_to_corr_ft_dict = {}
        corr_ft_to_idx_dict = {}
        for idx,corr_ft in enumerate(new_corr_fts):
            idx_to_corr_ft_dict[idx]=corr_ft
            corr_ft_to_idx_dict[corr_ft]=idx

        self.idx_to_corr_ft_dict=idx_to_corr_ft_dict
        self.corr_ft_to_idx_dict=corr_ft_to_idx_dict
        self.corr_fts = new_corr_fts



        infoID_to_overall_timestep_to_corr_ft_vector_dict = {}
        infoID_to_overall_timestep_to_corr_func_to_target_ft_dict = {}
        print("\nGetting corr vectors...")
        for infoID in self.infoIDs:
            print("\nGetting corr vectors for %s..."%infoID)
            infoID_to_overall_timestep_to_corr_func_to_target_ft_dict[infoID]={}
            infoID_to_overall_timestep_to_corr_ft_vector_dict[infoID]={}
            for overall_timestep, cur_timestep_df in overall_timestep_to_dynamic_ft_df_dict[infoID].items():
                infoID_to_overall_timestep_to_corr_ft_vector_dict[infoID][overall_timestep]={}
                infoID_to_overall_timestep_to_corr_func_to_target_ft_dict[infoID][overall_timestep]={}
                debug_print(infoID)
                debug_print(overall_timestep)
                debug_print(cur_timestep_df)


                cur_avg_corr_ft_vals = []
                for corr_func_str in self.CORRELATION_FUNCS_TO_USE_LIST:
                    infoID_to_overall_timestep_to_corr_func_to_target_ft_dict[infoID][overall_timestep][corr_func_str]={}

                    corr_func = self.CORRELATION_STR_TO_FUNC_DICT[corr_func_str]
                    debug_print("\ncorr_func: ")
                    debug_print(corr_func)

                    for cur_target_col in self.target_ft_categories:

                        # cur_target_corr_ft_vals = []
                        debug_print()
                        debug_print(corr_func_str)
                        debug_print(cur_target_col)

                        cur_target_ts_of_interest = cur_timestep_df[cur_target_col]
                        cur_aux_df_without_target = cur_timestep_df.copy().drop(cur_target_col, axis=1)

                        debug_print("\ncur_target_ts_of_interest")
                        debug_print(cur_target_ts_of_interest)

                        debug_print("\ncur_aux_df_without_target")
                        debug_print(cur_aux_df_without_target)

                        aux_cols = list(cur_aux_df_without_target)
                        # for aux_col in aux_cols:
                        #   cur_aux_series = cur_aux_df_without_target[aux_col]

                        cur_matrix_df = pd.concat([cur_target_ts_of_interest, cur_aux_df_without_target], axis=1)
                        debug_print("\ncur_matrix_df")
                        debug_print(cur_matrix_df)
                        cur_target_corr_ft_vals_matrix= corr_func(cur_matrix_df ,axis=0)[0]
                        # cur_target_corr_ft_vals_matrix= corr_func(cur_target_ts_of_interest, cur_aux_df_without_target.T,axis=1)[0]
                        debug_print("\ncur_target_corr_ft_vals_matrix")
                        debug_print(cur_target_corr_ft_vals_matrix)

                        debug_print("\ncur_target_corr_ft_vals_matrix shape")
                        debug_print(cur_target_corr_ft_vals_matrix.shape)

                        debug_print("\ncur_target_corr_ft_vals_matrix entry 0")
                        debug_print(cur_target_corr_ft_vals_matrix[0])

                        debug_print("\ncur_target_corr_ft_vals_matrix entry 1")
                        debug_print(cur_target_corr_ft_vals_matrix[1])

                        debug_print("\ncur_target_corr_ft_vals_matrix entry 0 shape")
                        debug_print(cur_target_corr_ft_vals_matrix[0].shape)

                        debug_print("\ncur_target_corr_ft_vals_matrix entry 1 shape")
                        debug_print(cur_target_corr_ft_vals_matrix[1].shape)

                        cur_ft_matrix_cols = list(cur_matrix_df)
                        cur_aux_corr_vals=np.abs(cur_target_corr_ft_vals_matrix[0])
                        for cur_ft, cur_aux_corr in zip(cur_ft_matrix_cols,cur_aux_corr_vals):
                            debug_print("%s: %.4f"%(cur_ft, cur_aux_corr))

                        cur_aux_corr_vals = pd.Series(cur_aux_corr_vals)
                        debug_print(cur_aux_corr_vals)
                        cur_aux_corr_vals = cur_aux_corr_vals.dropna().values
                        avg_corr = np.mean(cur_aux_corr_vals)
                        # cur_avg_corr_ft_vals.append(avg_corr)
                        # debug_print("\nInfoID: %s; Timestep %d; %s to aux %s corrs: "%( infoID,overall_timestep,cur_target_col,corr_func_str ))
                        # # cur_target_corr_ft_vals = cur_target_corr_ft_vals[~np.isnan(cur_target_corr_ft_vals)]
                        # cur_target_corr_ft_vals = np.abs(cur_target_corr_ft_vals)
                        # debug_print(cur_target_corr_ft_vals)

                        #now take the avg
                        # avg_corr = np.mean(cur_target_corr_ft_vals)
                        if  np.isnan(avg_corr):
                            avg_corr = 0
                            debug_print("\navg corr was nan, so we converted it to 0")
                            # sys.exit(0)

                        debug_print("%s %d %s %s corr: %.4f"%(infoID, overall_timestep, corr_func_str, cur_target_col, avg_corr))
                        infoID_to_overall_timestep_to_corr_func_to_target_ft_dict[infoID][overall_timestep][corr_func_str][cur_target_col]=avg_corr
                        cur_avg_corr_ft_vals.append(avg_corr)
                        # sys.exit(0)
                debug_print("\ncur_avg_corr_ft_vals")
                debug_print(cur_avg_corr_ft_vals)
                debug_print("\nwith fillna")
                cur_avg_corr_ft_vals = pd.Series(cur_avg_corr_ft_vals).fillna(0).values
                debug_print(cur_avg_corr_ft_vals)

                debug_print("\ncorr scores")
                for corr_ft,corr_score in zip(new_corr_fts, cur_avg_corr_ft_vals):
                    debug_print("%s: %.4f"%(corr_ft, corr_score))

                infoID_to_overall_timestep_to_corr_ft_vector_dict[infoID][overall_timestep]=cur_avg_corr_ft_vals
            print("Got corr fts for %s"%infoID)

        # #make dicts
        # self.infoID_to_overall_timestep_to_corr_ft_vector_dict=infoID_to_overall_timestep_to_corr_ft_vector_dict
        # self.infoID_to_overall_timestep_to_corr_func_to_target_ft_dict=infoID_to_overall_timestep_to_corr_func_to_target_ft_dict
        num_overall_timesteps = overall_timestep+1

        return infoID_to_overall_timestep_to_corr_ft_vector_dict,infoID_to_overall_timestep_to_corr_func_to_target_ft_dict,idx_to_corr_ft_dict,corr_ft_to_idx_dict,new_corr_fts,num_overall_timesteps

    def get_correlation_ft_materials_dict(self):

        array_type_to_correlation_materials_dict = {}
        array_types = ["train", "val", "test", "test_sliding_window", "val_sliding_window"]
        # array_types = ["val_sliding_window"]
        for array_type in array_types:
            print("\nGetting corr materials for %s"%array_type)
            array_type_to_correlation_materials_dict[array_type] = {}
            infoID_to_overall_timestep_to_corr_ft_vector_dict,infoID_to_overall_timestep_to_corr_func_to_target_ft_dict,idx_to_corr_ft_dict,corr_ft_to_idx_dict,new_corr_fts,num_overall_timesteps = self.insert_correlation_features(array_type,DEBUG_PRINT=False)
            array_type_to_correlation_materials_dict[array_type]["infoID_to_overall_timestep_to_corr_ft_vector_dict"] = infoID_to_overall_timestep_to_corr_ft_vector_dict.copy()
            array_type_to_correlation_materials_dict[array_type]["infoID_to_overall_timestep_to_corr_func_to_target_ft_dict"] = infoID_to_overall_timestep_to_corr_func_to_target_ft_dict.copy()
            array_type_to_correlation_materials_dict[array_type]["num_overall_timesteps"]=num_overall_timesteps

        self.array_type_to_correlation_materials_dict = array_type_to_correlation_materials_dict
        print("\nDone getting corr materials!")
        return array_type_to_correlation_materials_dict

    def update_feature_sets_with_corr_fts(self, array_types=["train", "val", "test", "test_sliding_window", "val_sliding_window"]):
        print("\nUpdating ft sets...")

        # array_types = ["val_sliding_window"]


        for array_type in array_types:
            x_array_tag = "x_" + array_type
            cur_array_type_correlation_materials_dict = self.array_type_to_correlation_materials_dict[array_type]
            num_overall_timesteps = cur_array_type_correlation_materials_dict["num_overall_timesteps"]
            infoID_to_overall_timestep_to_corr_ft_vector_dict = cur_array_type_correlation_materials_dict["infoID_to_overall_timestep_to_corr_ft_vector_dict"]

            print()
            print(array_type)
            print("num_overall_timesteps: %d"%num_overall_timesteps)

            for infoID in self.infoIDs:
                cur_new_ft_vectors = []
                print()
                print(infoID)
                for overall_timestep in range(num_overall_timesteps):
                    x_ft_array = self.infoID_train_and_test_array_dict[infoID][x_array_tag]
                    cur_new_ft_vector = infoID_to_overall_timestep_to_corr_ft_vector_dict[infoID][overall_timestep]

                    print("\nx_ft_array shape")
                    print(x_ft_array.shape)

                    print("\ncur_new_ft_vector.shape")
                    print(cur_new_ft_vector.shape)
                    cur_new_ft_vectors.append(cur_new_ft_vector)
                cur_new_ft_vectors = np.asarray(cur_new_ft_vectors)
                print("%s new vector shape: %s"%(infoID, cur_new_ft_vectors.shape))

                #concat
                x_ft_array = np.concatenate([x_ft_array, cur_new_ft_vectors], axis=1)
                print("\nx_ft_array after concat")
                print(x_ft_array.shape)
                self.infoID_train_and_test_array_dict[infoID][x_array_tag] = x_ft_array


        #update ft lists
        # self.dynamic_fts = dynamic_fts
        # self.static_fts = static_fts
        # self.input_fts = input_fts
        # self.target_ft_categories = target_fts
        self.static_fts =self.static_fts + list(self.corr_fts)
        print("\nnew self.static_fts")


        # self.input_dynamic_ft_categories= list(dynamic_fts)
        # self.input_ft_categories= list(input_fts)
        # self.target_ft_categories= list(target_fts)
        num_old_total_fts = len(self.input_dynamic_and_static_fts)
        for i,corr_ft in enumerate(self.corr_fts):
            corr_ft_idx = i + num_old_total_fts
            self.input_dynamic_and_static_fts_to_idx_dict[corr_ft]=corr_ft_idx
            self.input_idx_to_dynamic_and_static_fts_dict[corr_ft_idx]=corr_ft
            print("%s: %d"%(corr_ft, corr_ft_idx))

        self.input_dynamic_and_static_fts = list(self.input_dynamic_and_static_fts)+ list(self.corr_fts)

        num_indices = len(self.input_idx_to_dynamic_and_static_fts_dict)
        max_idx = corr_ft_idx
        correct_max_idx = num_indices - 1
        print("\nmax_idx: %d"%max_idx)
        print("correct_max_idx: %d"%correct_max_idx)
        if max_idx!= correct_max_idx:
            print("max_idx!= correct_max_idx")
            sys.exit(0)

        # self.input_dynamic_and_static_fts_to_idx_dict=dynamic_and_static_fts_to_idx_dict
        # self.input_idx_to_dynamic_and_static_fts_dict=idx_to_dynamic_and_static_fts_dict
        # self.input_dynamic_and_static_fts=self.input_dynamic_and_static_fts + list(self.corr_fts)
        # self.infoID_train_and_test_array_dict = infoID_train_and_test_array_dict

        # for infoID in self.infoIDs:
        #   self.x_train = self.infoID_train_and_test_array_dict[infoID]["x_train"]
        #   self.x_val = self.infoID_train_and_test_array_dict[infoID]["x_val"]
        #   self.x_test = self.infoID_train_and_test_array_dict[infoID]["x_test"]
        #   self.x_val_sliding_window = self.infoID_train_and_test_array_dict[infoID]["x_val_sliding_window"]
        #   self.x_test_sliding_window = self.infoID_train_and_test_array_dict[infoID]["x_test_sliding_window"]

    def create_baseline_old(self, array_tag):

        x_array_tag = "x_" + array_tag
        y_array_tag = "y_" + array_tag

        num_infoIDs = len(self.infoIDs)
        baseline_pred_dict = {}

        # print("\nself.target_ft_categories")
        # print(self.target_ft_categories)
        print("\nself.target_ft_categories")
        print(self.target_ft_categories)
        # print("\nself.input_dynamic_and_static_fts")
        # print(self.input_dynamic_and_static_fts)
        for i in range(num_infoIDs):
            infoID=self.infoIDs[i]
            # baseline_pred_dict[infoID] = {}

            cur_pred_arrays = []

            x = self.infoID_train_and_test_array_dict[infoID][x_array_tag].copy()
            #y = infoID_train_and_test_array_dict[infoID][y_array_tag]

            for cur_target_ft_category in self.target_ft_categories:

                #get the full array for this ft
                print("\ncur_target_ft_category")
                print(cur_target_ft_category)

                print("\nx shape")
                print(x.shape)

                #get ft name indices
                target_ft_indices = self.target_ft_category_to_list_of_temporal_indices_dict[cur_target_ft_category]["indices_list"]

                reversed_target_ft_indices = reversed(target_ft_indices)

                num_target_ft_indices = len(target_ft_indices)
                print("\nnum_target_ft_indices")
                print(num_target_ft_indices)
                print(target_ft_indices)

                x_cur_target_ft_category_temp_array = np.take(x, target_ft_indices,axis=1)
                x_cur_target_ft_category_temp_array=x_cur_target_ft_category_temp_array[:,-self.EXTRACT_OUTPUT_SIZE:]
                print("\nx_cur_target_ft_category_temp_array shape")
                print(x_cur_target_ft_category_temp_array.shape)

                # if self.AGG_TIME_SERIES == True:
                x_cur_target_ft_category_temp_array = np.sum(x_cur_target_ft_category_temp_array, axis=1)

                print("\nx_cur_target_ft_category_temp_array shape after sum")
                # x_cur_target_ft_category_temp_array=x_cur_target_ft_category_temp_array.reshape((x_cur_target_ft_category_temp_array.shape[0], 1))
                print(x_cur_target_ft_category_temp_array.shape)

                cur_pred_arrays.append(x_cur_target_ft_category_temp_array)

            cur_baseline_pred_array = np.asarray(cur_pred_arrays)

            # if self.AGG_TIME_SERIES == True:
            cur_baseline_pred_array = cur_baseline_pred_array.T

            cur_baseline_pred_array = np.asarray(cur_pred_arrays)
            print("\ncur_baseline_pred_array shape")
            print(cur_baseline_pred_array.shape)

            baseline_pred_dict[infoID] = cur_baseline_pred_array

        return baseline_pred_dict

            # cur_baseline_pred_array = np.concatenate(cur_pred_arrays)

    def create_baseline(self, array_tag):

        x_array_tag = "x_" + array_tag
        y_array_tag = "y_" + array_tag

        num_infoIDs = len(self.infoIDs)
        baseline_pred_dict = {}

        # print("\nself.target_ft_categories")
        # print(self.target_ft_categories)
        print("\nself.target_ft_categories")
        print(self.target_ft_categories)
        # print("\nself.input_dynamic_and_static_fts")
        # print(self.input_dynamic_and_static_fts)
        for i in range(num_infoIDs):
            infoID=self.infoIDs[i]
            # baseline_pred_dict[infoID] = {}

            cur_pred_arrays = []

            x = self.infoID_train_and_test_array_dict[infoID][x_array_tag].copy()
            #y = infoID_train_and_test_array_dict[infoID][y_array_tag]

            x = remove_static_1hot_fts(x.copy(), self.dynamic_fts,self.static_fts,self.input_dynamic_and_static_fts,self.INPUT_HOURS ,RESHAPE=True)

            cur_baseline_pred_array = x[:, -self.OUTPUT_HOURS:].copy()
            print("\ncur_baseline_pred_array shape for %s"%array_tag)
            print(cur_baseline_pred_array.shape)

            DESIRED_TARGET_TAKE_INDICES = []
            for target_ft_cat_idx,target_ft_cat in enumerate(self.target_ft_categories):
                # print("target_ft_cat_idx: %d; target_ft_cat: %s"%(target_ft_cat_idx, target_ft_cat))
                for input_dynamic_ft_idx, dynamic_ft in enumerate(self.dynamic_fts):
                    if dynamic_ft==target_ft_cat:
                        DESIRED_TARGET_TAKE_INDICES.append(input_dynamic_ft_idx)
                        print("%s has idx %d in the input fts"%(dynamic_ft, input_dynamic_ft_idx))
            cur_baseline_pred_array = np.take(cur_baseline_pred_array,indices=DESIRED_TARGET_TAKE_INDICES,axis=2)
            print("\ncur_baseline_pred_array shape for %s after np.take"%array_tag)
            print(cur_baseline_pred_array.shape)


            # num_samples = x.shape[0]
            # num_timesteps = x.shape[1]
            # all_samples = []
            # all_timestep_lists = []
            # for cur_sample_idx in range(num_samples):
            #   for cur_timestep_idx in range(num_timesteps):
            #       for target_ft_cat_idx,target_ft_cat in enumerate(self.target_ft_categories):
            #           print("target_ft_cat_idx: %d; target_ft_cat: %s"%(target_ft_cat_idx, target_ft_cat))

            #           for input_dynamic_ft_idx, dynamic_ft in enumerate(self.dynamic_fts):
            #               if

            # sys.exit(0)
            cur_baseline_pred_array = cur_baseline_pred_array.reshape((cur_baseline_pred_array.shape[0], cur_baseline_pred_array.shape[1] * cur_baseline_pred_array.shape[2]))
            print("\ncur_baseline_pred_array shape for %s after reshape"%array_tag)
            print(cur_baseline_pred_array.shape)

            baseline_pred_dict[infoID] = cur_baseline_pred_array



        return baseline_pred_dict

    def get_test_data_baseline_pred_dict(self):
        self.y_pred_baseline_test_data_dict = self.create_baseline("test")
        return self.y_pred_baseline_test_data_dict

    def get_val_data_baseline_pred_dict(self):
        self.y_pred_baseline_val_data_dict = self.create_baseline("val")
        return self.y_pred_baseline_val_data_dict

    def get_test_sliding_window_data_baseline_pred_dict(self):
        self.y_pred_baseline_test_sliding_window_data_dict = self.create_baseline("test_sliding_window")
        return self.y_pred_baseline_test_sliding_window_data_dict

    def get_val_sliding_window_data_baseline_pred_dict(self):
        self.y_pred_baseline_val_sliding_window_data_dict = self.create_baseline("val_sliding_window")
        return self.y_pred_baseline_val_sliding_window_data_dict

    # def normalize_val_arrays(self):
    #   normalize_data(self,x,y)

    def configure_train_data(self,num_timesteps_to_get= "all"):
        self.x_train,self.y_train = self.configure_val_or_test_data("train",num_timesteps_to_get)
        return self.x_train,self.y_train

    def configure_val_data(self,num_timesteps_to_get= "all"):
        self.x_val,self.y_val = self.configure_val_or_test_data("val",num_timesteps_to_get)
        return self.x_val,self.y_val

    def configure_test_data(self,num_timesteps_to_get= "all"):
        self.x_test,self.y_test = self.configure_val_or_test_data("test",num_timesteps_to_get)
        return self.x_test,self.y_test

    def configure_val_sliding_window_data(self,num_timesteps_to_get= "all"):
        self.x_val_sliding_window,self.y_val_sliding_window = self.configure_val_or_test_data("val_sliding_window", num_timesteps_to_get)
        return self.x_val_sliding_window,self.y_val_sliding_window

    def configure_test_sliding_window_data(self,num_timesteps_to_get= "all"):
        self.x_test_sliding_window,self.y_test_sliding_window = self.configure_val_or_test_data("test_sliding_window", num_timesteps_to_get)
        return self.x_test_sliding_window,self.y_test_sliding_window

    #added rounding
    def pred_with_model(self,model, array_tag, num_timesteps_to_get="all"):

        x_tag = "x_%s"%array_tag

        y_pred_dict = {}
        for infoID in self.infoIDs:

            if num_timesteps_to_get == "all":
                cur_x_array = self.infoID_train_and_test_array_dict[infoID][x_tag]
            else:
                cur_x_array = self.infoID_train_and_test_array_dict[infoID][x_tag][:num_timesteps_to_get, :]

            print("\nbefore norm")
            print(cur_x_array)
            cur_x_array = self.normalize_x_array(cur_x_array)
            # sys.exit(0)
            print("\nafter norm")
            print(cur_x_array)

            print("\ncur_x_array shape")
            print(cur_x_array.shape)


            cur_y_pred = model.predict(cur_x_array)
            print("\ncur_y_pred before inverse norm")
            print(cur_y_pred.shape)

            # sys.exit(0)
            cur_y_pred = self.inverse_normalize_y_data(cur_y_pred)
            cur_y_pred = np.round(cur_y_pred, 0)
            y_pred_dict[infoID]=cur_y_pred

        print("\ny_%s pred shape"%array_tag)
        print(cur_y_pred.shape)
        return y_pred_dict

    def get_corr_fts(self):

        if len(self.CORRELATION_FUNCS_TO_USE_LIST) > 0:
            self.convert_all_arrays_to_dynamic_ft_dfs_dict()
            self.get_correlation_ft_materials_dict()
            self.update_feature_sets_with_corr_fts()
        else:
            print("No corr fts needed. Continuing.")

    def convert_specific_arrays_to_dynamic_ft_dfs_dict(self, desired_array_types=["test" , "val"]):

        array_type_to_dynamic_ft_df_dict = {}

        if "train" in desired_array_types:
            self.train_overall_timestep_to_dynamic_ft_df_dict=self.convert_train_arrays_to_dynamic_ft_dfs()
            array_type_to_dynamic_ft_df_dict["train"]=self.train_overall_timestep_to_dynamic_ft_df_dict

        if "val" in desired_array_types:
            self.val_overall_timestep_to_dynamic_ft_df_dict=self.convert_val_arrays_to_dynamic_ft_dfs()
            array_type_to_dynamic_ft_df_dict["val"]=self.val_overall_timestep_to_dynamic_ft_df_dict

        if "test" in desired_array_types:
            self.test_overall_timestep_to_dynamic_ft_df_dict=self.convert_test_arrays_to_dynamic_ft_dfs()
            array_type_to_dynamic_ft_df_dict["test"]=self.test_overall_timestep_to_dynamic_ft_df_dict

        if "test_sliding_window" in desired_array_types:
            self.test_sliding_window_overall_timestep_to_dynamic_ft_df_dict=self.convert_test_sliding_window_arrays_to_dynamic_ft_dfs()
            array_type_to_dynamic_ft_df_dict["test_sliding_window"]=self.test_sliding_window_overall_timestep_to_dynamic_ft_df_dict

        if "val_sliding_window" in desired_array_types:
            self.val_sliding_window_overall_timestep_to_dynamic_ft_df_dict=self.convert_val_sliding_window_arrays_to_dynamic_ft_dfs()
            array_type_to_dynamic_ft_df_dict["val_sliding_window"]=self.val_sliding_window_overall_timestep_to_dynamic_ft_df_dict

        self.array_type_to_dynamic_ft_df_dict=array_type_to_dynamic_ft_df_dict

        return array_type_to_dynamic_ft_df_dict

    # def get_corr_fts(self):

    #   if len(self.CORRELATION_FUNCS_TO_USE_LIST) > 0:
    #       self.convert_all_arrays_to_dynamic_ft_dfs_dict()
    #       self.get_correlation_ft_materials_dict()
    #       self.update_feature_sets_with_corr_fts()
    #   else:
    #       print("No corr fts needed. Continuing.")


    def get_main_ts_to_exo_ts_corrs(self, desired_array_types=["test" , "val"]):

        #returns self.array_type_to_dynamic_ft_df_dict
        if self.GET_EXTERNAL_PLATFORM_FTS == False:
            if self.GET_REDDIT_FEATURES == False:
                if self.GET_GDELT_FEATURES == False:
                    print("No corr fts needed. Continuing.")
                    return False

        self.convert_specific_arrays_to_dynamic_ft_dfs_dict(desired_array_types)
        self.specific_get_correlation_ft_materials_dict_for_exo_corr_exp(desired_array_types)

        if len(self.corr_fts) == 0:
            return False
        self.update_feature_sets_with_corr_fts(desired_array_types)
        return True

    def get_main_ts_to_exo_ts_corrs_para(self, desired_array_types=["test" , "val"]):

        #returns self.array_type_to_dynamic_ft_df_dict
        if self.GET_EXTERNAL_PLATFORM_FTS == False:
            if self.GET_REDDIT_FEATURES == False:
                if self.GET_GDELT_FEATURES == False:
                    print("No corr fts needed. Continuing.")
                    return False

        self.convert_specific_arrays_to_dynamic_ft_dfs_dict(desired_array_types)
        self.specific_get_correlation_ft_materials_dict_for_exo_corr_exp_para(desired_array_types)

        if len(self.corr_fts) == 0:
            return False
        self.update_feature_sets_with_corr_fts(desired_array_types)
        return True

    def get_full_corr_data_para(self, desired_array_types=["test" , "val"]):

        #returns self.array_type_to_dynamic_ft_df_dict
        # if self.GET_EXTERNAL_PLATFORM_FTS == False:
        #   if self.GET_REDDIT_FEATURES == False:
        #       if self.GET_GDELT_FEATURES == False:
        #           print("No corr fts needed. Continuing.")
        #           return False

        # print("bye")
        # sys.exit(0)

        self.convert_specific_arrays_to_dynamic_ft_dfs_dict(desired_array_types)
        self.specific_get_correlation_ft_materials_dict_for_exo_corr_exp_para_v2_full_data(desired_array_types)

        if len(self.corr_fts) == 0:
            print("len(self.corr_fts) == 0, returning")
            sys.exit(0)
            return False
        self.update_feature_sets_with_corr_fts(desired_array_types)
        return True

    def specific_get_correlation_ft_materials_dict_for_exo_corr_exp(self,desired_array_types = [ "val", "test"]):

        array_type_to_correlation_materials_dict = {}

        # array_types = ["val_sliding_window"]
        for array_type in desired_array_types:
            print("\nGetting corr materials for %s"%array_type)
            array_type_to_correlation_materials_dict[array_type] = {}
            infoID_to_overall_timestep_to_corr_ft_vector_dict,infoID_to_overall_timestep_to_corr_func_to_target_ft_dict,idx_to_corr_ft_dict,corr_ft_to_idx_dict,new_corr_fts,num_overall_timesteps = self.insert_correlation_features_for_corr_exp(array_type,DEBUG_PRINT=False)
            array_type_to_correlation_materials_dict[array_type]["infoID_to_overall_timestep_to_corr_ft_vector_dict"] = infoID_to_overall_timestep_to_corr_ft_vector_dict.copy()
            array_type_to_correlation_materials_dict[array_type]["infoID_to_overall_timestep_to_corr_func_to_target_ft_dict"] = infoID_to_overall_timestep_to_corr_func_to_target_ft_dict.copy()
            array_type_to_correlation_materials_dict[array_type]["num_overall_timesteps"]=num_overall_timesteps

        self.array_type_to_correlation_materials_dict = array_type_to_correlation_materials_dict
        print("\nDone getting corr materials!")
        return array_type_to_correlation_materials_dict

    def insert_correlation_features_for_corr_exp(self, array_tag,DEBUG_PRINT=False):

        def debug_print(print_val="\n"):
            if DEBUG_PRINT == True:
                print(str(print_val))

        x_array_tag = "x_" + array_tag
        y_array_tag = "y_" + array_tag

        num_infoIDs = len(self.infoIDs)

        overall_timestep_to_dynamic_ft_df_dict = self.array_type_to_dynamic_ft_df_dict[array_tag]

        #create the new fts to be added
        new_corr_fts = []
        print("\nnew corr fts: \n")
        for corr_func_str in self.CORRELATION_FUNCS_TO_USE_LIST:
            for cur_target_col in self.target_ft_categories:
                new_ft = "%s_corr_%s_with_aux_time_series"%(corr_func_str, cur_target_col)
                print(new_ft)
                new_corr_fts.append(new_ft)
        # sys.exit(0)

        #make new dicts
        idx_to_corr_ft_dict = {}
        corr_ft_to_idx_dict = {}
        for idx,corr_ft in enumerate(new_corr_fts):
            idx_to_corr_ft_dict[idx]=corr_ft
            corr_ft_to_idx_dict[corr_ft]=idx

        self.idx_to_corr_ft_dict=idx_to_corr_ft_dict
        self.corr_ft_to_idx_dict=corr_ft_to_idx_dict
        corr_fts = list(new_corr_fts)
        self.corr_fts = list(new_corr_fts)



        infoID_to_overall_timestep_to_corr_ft_vector_dict = {}
        infoID_to_overall_timestep_to_corr_func_to_target_ft_dict = {}
        print("\nGetting corr vectors...")
        for infoID in self.infoIDs:
            print("\nGetting corr vectors for %s..."%infoID)
            infoID_to_overall_timestep_to_corr_func_to_target_ft_dict[infoID]={}
            infoID_to_overall_timestep_to_corr_ft_vector_dict[infoID]={}
            for overall_timestep, cur_timestep_df in overall_timestep_to_dynamic_ft_df_dict[infoID].items():
                infoID_to_overall_timestep_to_corr_ft_vector_dict[infoID][overall_timestep]={}
                infoID_to_overall_timestep_to_corr_func_to_target_ft_dict[infoID][overall_timestep]={}
                debug_print(infoID)
                debug_print(overall_timestep)
                debug_print(cur_timestep_df)


                cur_avg_corr_ft_vals = []
                for corr_func_str in self.CORRELATION_FUNCS_TO_USE_LIST:
                    infoID_to_overall_timestep_to_corr_func_to_target_ft_dict[infoID][overall_timestep][corr_func_str]={}

                    corr_func = self.CORRELATION_STR_TO_FUNC_DICT[corr_func_str]
                    debug_print("\ncorr_func: ")
                    debug_print(corr_func)

                    for cur_target_col in self.target_ft_categories:

                        # cur_target_corr_ft_vals = []
                        debug_print()
                        debug_print(corr_func_str)
                        debug_print(cur_target_col)

                        cur_target_ts_of_interest = cur_timestep_df[cur_target_col]

                        #==================== get aux df with just exo ==========================
                        cur_aux_df_without_target = cur_timestep_df.copy().drop(cur_target_col, axis=1)

                        keep_cols = []

                        if self.GET_GDELT_FEATURES == True:
                            keep_cols = keep_cols + ["AvgTone", "GoldsteinScale", "NumMentions"]

                        if self.GET_REDDIT_FEATURES == True:
                            keep_cols = keep_cols + ["num_reddit_activities"]

                        if self.GET_EXTERNAL_PLATFORM_FTS == True:
                            # if self.TARGET_PLATFORM == "twitter":
                            #   keep_cols = keep_cols + get_fts_from_fp(self.feature_list_dir + "twitter_dynamic_global_fts.txt")
                            #   keep_cols = keep_cols + get_fts_from_fp(self.feature_list_dir + "twitter_dynamic_pair_fts.txt")
                            if self.TARGET_PLATFORM == "twitter":
                                keep_cols = keep_cols + get_fts_from_fp(self.feature_list_dir + "youtube_dynamic_global_fts.txt")
                                keep_cols = keep_cols + get_fts_from_fp(self.feature_list_dir + "youtube_dynamic_pair_fts.txt")

                            if self.TARGET_PLATFORM == "youtube":
                                keep_cols = keep_cols + get_fts_from_fp(self.feature_list_dir + "twitter_dynamic_global_fts.txt")
                                keep_cols = keep_cols + get_fts_from_fp(self.feature_list_dir + "twitter_dynamic_pair_fts.txt")

                        cur_aux_df_without_target = cur_aux_df_without_target[keep_cols]



                        print("\ncur_aux_df_without_target")
                        print(cur_aux_df_without_target)

                        if cur_aux_df_without_target.shape[1] == 1:
                            col_to_copy = list(cur_aux_df_without_target)[0]
                            cur_aux_df_without_target["dupe_%s"%col_to_copy] = cur_aux_df_without_target[col_to_copy].copy()

                            print("\ncur_aux_df_without_target only had 1 col")
                            print("Made a copy of the col for easier computation")
                            print(cur_aux_df_without_target)
                        #===========================================================

                        debug_print("\ncur_target_ts_of_interest")
                        debug_print(cur_target_ts_of_interest)

                        debug_print("\ncur_aux_df_without_target")
                        debug_print(cur_aux_df_without_target)

                        aux_cols = list(cur_aux_df_without_target)
                        # for aux_col in aux_cols:
                        #   cur_aux_series = cur_aux_df_without_target[aux_col]

                        cur_matrix_df = pd.concat([cur_target_ts_of_interest, cur_aux_df_without_target], axis=1)
                        debug_print("\ncur_matrix_df")
                        debug_print(cur_matrix_df)
                        cur_target_corr_ft_vals_matrix= corr_func(cur_matrix_df ,axis=0)[0]
                        # cur_target_corr_ft_vals_matrix= corr_func(cur_target_ts_of_interest, cur_aux_df_without_target.T,axis=1)[0]
                        debug_print("\ncur_target_corr_ft_vals_matrix")
                        debug_print(cur_target_corr_ft_vals_matrix)

                        debug_print("\ncur_target_corr_ft_vals_matrix shape")
                        debug_print(cur_target_corr_ft_vals_matrix.shape)

                        debug_print("\ncur_target_corr_ft_vals_matrix entry 0")
                        debug_print(cur_target_corr_ft_vals_matrix[0])

                        debug_print("\ncur_target_corr_ft_vals_matrix entry 1")
                        debug_print(cur_target_corr_ft_vals_matrix[1])

                        debug_print("\ncur_target_corr_ft_vals_matrix entry 0 shape")
                        debug_print(cur_target_corr_ft_vals_matrix[0].shape)

                        debug_print("\ncur_target_corr_ft_vals_matrix entry 1 shape")
                        debug_print(cur_target_corr_ft_vals_matrix[1].shape)

                        cur_ft_matrix_cols = list(cur_matrix_df)
                        cur_aux_corr_vals=np.abs(cur_target_corr_ft_vals_matrix[0])
                        for cur_ft, cur_aux_corr in zip(cur_ft_matrix_cols,cur_aux_corr_vals):
                            debug_print("%s: %.4f"%(cur_ft, cur_aux_corr))

                        cur_aux_corr_vals = pd.Series(cur_aux_corr_vals)
                        debug_print(cur_aux_corr_vals)
                        cur_aux_corr_vals = cur_aux_corr_vals.dropna().values
                        avg_corr = np.mean(cur_aux_corr_vals)
                        # cur_avg_corr_ft_vals.append(avg_corr)
                        # debug_print("\nInfoID: %s; Timestep %d; %s to aux %s corrs: "%( infoID,overall_timestep,cur_target_col,corr_func_str ))
                        # # cur_target_corr_ft_vals = cur_target_corr_ft_vals[~np.isnan(cur_target_corr_ft_vals)]
                        # cur_target_corr_ft_vals = np.abs(cur_target_corr_ft_vals)
                        # debug_print(cur_target_corr_ft_vals)

                        #now take the avg
                        # avg_corr = np.mean(cur_target_corr_ft_vals)
                        if  np.isnan(avg_corr):
                            avg_corr = 0
                            debug_print("\navg corr was nan, so we converted it to 0")
                            # sys.exit(0)

                        debug_print("%s %d %s %s corr: %.4f"%(infoID, overall_timestep, corr_func_str, cur_target_col, avg_corr))
                        infoID_to_overall_timestep_to_corr_func_to_target_ft_dict[infoID][overall_timestep][corr_func_str][cur_target_col]=avg_corr
                        cur_avg_corr_ft_vals.append(avg_corr)
                        # sys.exit(0)
                debug_print("\ncur_avg_corr_ft_vals")
                debug_print(cur_avg_corr_ft_vals)
                debug_print("\nwith fillna")
                cur_avg_corr_ft_vals = pd.Series(cur_avg_corr_ft_vals).fillna(0).values
                debug_print(cur_avg_corr_ft_vals)

                debug_print("\ncorr scores")
                for corr_ft,corr_score in zip(new_corr_fts, cur_avg_corr_ft_vals):
                    debug_print("%s: %.4f"%(corr_ft, corr_score))

                infoID_to_overall_timestep_to_corr_ft_vector_dict[infoID][overall_timestep]=cur_avg_corr_ft_vals
            print("Got corr fts for %s"%infoID)

        # #make dicts
        # self.infoID_to_overall_timestep_to_corr_ft_vector_dict=infoID_to_overall_timestep_to_corr_ft_vector_dict
        # self.infoID_to_overall_timestep_to_corr_func_to_target_ft_dict=infoID_to_overall_timestep_to_corr_func_to_target_ft_dict
        num_overall_timesteps = overall_timestep+1

        return infoID_to_overall_timestep_to_corr_ft_vector_dict,infoID_to_overall_timestep_to_corr_func_to_target_ft_dict,idx_to_corr_ft_dict,corr_ft_to_idx_dict,new_corr_fts,num_overall_timesteps

    def get_corr_ft_result_materials_by_infoID(self, arg_tuple ):

        def debug_print(print_val="\n"):
            if DEBUG_PRINT == True:
                print(str(print_val))

        infoID,infoID_idx,array_tag,num_infoIDs,new_corr_fts,DEBUG_PRINT,idx_to_corr_ft_dict,corr_ft_to_idx_dict,corr_fts= arg_tuple

        x_array_tag = "x_" + array_tag
        y_array_tag = "y_" + array_tag

        overall_timestep_to_dynamic_ft_df_dict = self.array_type_to_dynamic_ft_df_dict[array_tag]
        infoID_to_overall_timestep_to_corr_ft_vector_dict = {}
        infoID_to_overall_timestep_to_corr_func_to_target_ft_dict = {}

        print("\nGetting corr vectors for %s..."%infoID)
        infoID_to_overall_timestep_to_corr_func_to_target_ft_dict[infoID]={}
        infoID_to_overall_timestep_to_corr_ft_vector_dict[infoID]={}
        for overall_timestep, cur_timestep_df in overall_timestep_to_dynamic_ft_df_dict[infoID].items():
            infoID_to_overall_timestep_to_corr_ft_vector_dict[infoID][overall_timestep]={}
            infoID_to_overall_timestep_to_corr_func_to_target_ft_dict[infoID][overall_timestep]={}
            debug_print(infoID)
            debug_print(overall_timestep)
            debug_print(cur_timestep_df)


            cur_avg_corr_ft_vals = []
            for corr_func_str in self.CORRELATION_FUNCS_TO_USE_LIST:
                infoID_to_overall_timestep_to_corr_func_to_target_ft_dict[infoID][overall_timestep][corr_func_str]={}

                corr_func = self.CORRELATION_STR_TO_FUNC_DICT[corr_func_str]
                debug_print("\ncorr_func: ")
                debug_print(corr_func)

                for cur_target_col in self.target_ft_categories:

                    # cur_target_corr_ft_vals = []
                    debug_print("")
                    debug_print(corr_func_str)
                    debug_print(cur_target_col)

                    cur_target_ts_of_interest = cur_timestep_df[cur_target_col]

                    #==================== get aux df with just exo ==========================
                    cur_aux_df_without_target = cur_timestep_df.copy().drop(cur_target_col, axis=1)

                    keep_cols = []

                    if self.GET_GDELT_FEATURES == True:
                        keep_cols = keep_cols + ["AvgTone", "GoldsteinScale", "NumMentions"]

                    if self.GET_REDDIT_FEATURES == True:
                        keep_cols = keep_cols + ["num_reddit_activities"]

                    if self.GET_EXTERNAL_PLATFORM_FTS == True:
                        # if self.TARGET_PLATFORM == "twitter":
                        #   keep_cols = keep_cols + get_fts_from_fp(self.feature_list_dir + "twitter_dynamic_global_fts.txt")
                        #   keep_cols = keep_cols + get_fts_from_fp(self.feature_list_dir + "twitter_dynamic_pair_fts.txt")
                        if self.TARGET_PLATFORM == "twitter":
                            keep_cols = keep_cols + get_fts_from_fp(self.feature_list_dir + "youtube_dynamic_global_fts.txt")
                            keep_cols = keep_cols + get_fts_from_fp(self.feature_list_dir + "youtube_dynamic_pair_fts.txt")

                        if self.TARGET_PLATFORM == "youtube":
                            keep_cols = keep_cols + get_fts_from_fp(self.feature_list_dir + "twitter_dynamic_global_fts.txt")
                            keep_cols = keep_cols + get_fts_from_fp(self.feature_list_dir + "twitter_dynamic_pair_fts.txt")

                    cur_aux_df_without_target = cur_aux_df_without_target[keep_cols]



                    debug_print("\ncur_aux_df_without_target")
                    debug_print(cur_aux_df_without_target)

                    if cur_aux_df_without_target.shape[1] == 1:
                        col_to_copy = list(cur_aux_df_without_target)[0]
                        cur_aux_df_without_target["dupe_%s"%col_to_copy] = cur_aux_df_without_target[col_to_copy].copy()

                        debug_print("\ncur_aux_df_without_target only had 1 col")
                        debug_print("Made a copy of the col for easier computation")
                        debug_print(cur_aux_df_without_target)
                    #===========================================================

                    debug_print("\ncur_target_ts_of_interest")
                    debug_print(cur_target_ts_of_interest)

                    debug_print("\ncur_aux_df_without_target")
                    debug_print(cur_aux_df_without_target)

                    aux_cols = list(cur_aux_df_without_target)
                    # for aux_col in aux_cols:
                    #   cur_aux_series = cur_aux_df_without_target[aux_col]

                    cur_matrix_df = pd.concat([cur_target_ts_of_interest, cur_aux_df_without_target], axis=1)
                    debug_print("\ncur_matrix_df")
                    debug_print(cur_matrix_df)
                    cur_target_corr_ft_vals_matrix= corr_func(cur_matrix_df ,axis=0)[0]
                    # cur_target_corr_ft_vals_matrix= corr_func(cur_target_ts_of_interest, cur_aux_df_without_target.T,axis=1)[0]
                    debug_print("\ncur_target_corr_ft_vals_matrix")
                    debug_print(cur_target_corr_ft_vals_matrix)

                    debug_print("\ncur_target_corr_ft_vals_matrix shape")
                    debug_print(cur_target_corr_ft_vals_matrix.shape)

                    debug_print("\ncur_target_corr_ft_vals_matrix entry 0")
                    debug_print(cur_target_corr_ft_vals_matrix[0])

                    debug_print("\ncur_target_corr_ft_vals_matrix entry 1")
                    debug_print(cur_target_corr_ft_vals_matrix[1])

                    debug_print("\ncur_target_corr_ft_vals_matrix entry 0 shape")
                    debug_print(cur_target_corr_ft_vals_matrix[0].shape)

                    debug_print("\ncur_target_corr_ft_vals_matrix entry 1 shape")
                    debug_print(cur_target_corr_ft_vals_matrix[1].shape)

                    cur_ft_matrix_cols = list(cur_matrix_df)
                    cur_aux_corr_vals=np.abs(cur_target_corr_ft_vals_matrix[0])
                    for cur_ft, cur_aux_corr in zip(cur_ft_matrix_cols,cur_aux_corr_vals):
                        debug_print("%s: %.4f"%(cur_ft, cur_aux_corr))

                    cur_aux_corr_vals = pd.Series(cur_aux_corr_vals)
                    debug_print(cur_aux_corr_vals)
                    cur_aux_corr_vals = cur_aux_corr_vals.dropna().values
                    avg_corr = np.mean(cur_aux_corr_vals)
                    # cur_avg_corr_ft_vals.append(avg_corr)
                    # debug_print("\nInfoID: %s; Timestep %d; %s to aux %s corrs: "%( infoID,overall_timestep,cur_target_col,corr_func_str ))
                    # # cur_target_corr_ft_vals = cur_target_corr_ft_vals[~np.isnan(cur_target_corr_ft_vals)]
                    # cur_target_corr_ft_vals = np.abs(cur_target_corr_ft_vals)
                    # debug_print(cur_target_corr_ft_vals)

                    #now take the avg
                    # avg_corr = np.mean(cur_target_corr_ft_vals)
                    if  np.isnan(avg_corr):
                        avg_corr = 0
                        debug_print("\navg corr was nan, so we converted it to 0")
                        # sys.exit(0)

                    debug_print("%s %d %s %s corr: %.4f"%(infoID, overall_timestep, corr_func_str, cur_target_col, avg_corr))
                    infoID_to_overall_timestep_to_corr_func_to_target_ft_dict[infoID][overall_timestep][corr_func_str][cur_target_col]=avg_corr
                    cur_avg_corr_ft_vals.append(avg_corr)
                    # sys.exit(0)
            debug_print("\ncur_avg_corr_ft_vals")
            debug_print(cur_avg_corr_ft_vals)
            debug_print("\nwith fillna")
            cur_avg_corr_ft_vals = pd.Series(cur_avg_corr_ft_vals).fillna(0).values
            debug_print(cur_avg_corr_ft_vals)

            debug_print("\ncorr scores")
            for corr_ft,corr_score in zip(new_corr_fts, cur_avg_corr_ft_vals):
                debug_print("%s: %.4f"%(corr_ft, corr_score))

            infoID_to_overall_timestep_to_corr_ft_vector_dict[infoID][overall_timestep]=cur_avg_corr_ft_vals
        print("Got corr fts for %s"%infoID)

        # #make dicts
        # self.infoID_to_overall_timestep_to_corr_ft_vector_dict=infoID_to_overall_timestep_to_corr_ft_vector_dict
        # self.infoID_to_overall_timestep_to_corr_func_to_target_ft_dict=infoID_to_overall_timestep_to_corr_func_to_target_ft_dict
        num_overall_timesteps = overall_timestep+1

        return (infoID_to_overall_timestep_to_corr_ft_vector_dict,infoID_to_overall_timestep_to_corr_func_to_target_ft_dict,idx_to_corr_ft_dict,corr_ft_to_idx_dict,new_corr_fts,num_overall_timesteps)

    def insert_correlation_features_for_corr_exp_para(self, array_tag,DEBUG_PRINT=False):

        def debug_print(print_val="\n"):
            if DEBUG_PRINT == True:
                print(str(print_val))

        x_array_tag = "x_" + array_tag
        y_array_tag = "y_" + array_tag

        num_infoIDs = len(self.infoIDs)



        #create the new fts to be added
        new_corr_fts = []
        print("\nnew corr fts: \n")
        for corr_func_str in self.CORRELATION_FUNCS_TO_USE_LIST:
            for cur_target_col in self.target_ft_categories:
                new_ft = "%s_corr_%s_with_aux_time_series"%(corr_func_str, cur_target_col)
                print(new_ft)
                new_corr_fts.append(new_ft)
        # sys.exit(0)

        #make new dicts
        idx_to_corr_ft_dict = {}
        corr_ft_to_idx_dict = {}
        for idx,corr_ft in enumerate(new_corr_fts):
            idx_to_corr_ft_dict[idx]=corr_ft
            corr_ft_to_idx_dict[corr_ft]=idx

        self.idx_to_corr_ft_dict=idx_to_corr_ft_dict
        self.corr_ft_to_idx_dict=corr_ft_to_idx_dict
        corr_fts = list(new_corr_fts)
        self.corr_fts = list(corr_fts)




        # print("\nGetting corr vectors...")
        # corr_materials_result_list = []
        # for infoID_idx, infoID in enumerate(self.infoIDs):
        #   corr_materials_result_list = self.get_corr_ft_result_materials_by_infoID(infoID,infoID_idx,x_array_tag,y_array_tag,num_infoIDs,new_corr_fts)
        # print("\nnum corr_materials_result_list")
        # print(len(corr_materials_result_list))


        #=========== multi proc #===========
        print("\nGetting corr vectors...")
        pool = mp.Pool(processes=self.n_jobs)
        corr_materials_result_list = []
        arg_tuple_list = []
        for infoID_idx, infoID in enumerate(self.infoIDs):
            cur_arg_tuple = (infoID,infoID_idx,array_tag,num_infoIDs,new_corr_fts,False,idx_to_corr_ft_dict,corr_ft_to_idx_dict,corr_fts)
            arg_tuple_list.append(cur_arg_tuple)

        print("\nLaunching parallel func...")
        #corr_materials_result_list = [pool.apply_sync(self.get_corr_ft_result_materials_by_infoID, args=arg_tuple) for arg_tuple in  arg_tuple_list]
        corr_materials_result_list = pool.map(self.get_corr_ft_result_materials_by_infoID, arg_tuple_list)
        print("\nnum corr_materials_result_list")
        print(len(corr_materials_result_list))
        #===========#===========#===========


        # #train model
        # pool = multiprocessing.Pool(processes=self.n_)
        # p = Process(target = get_corr_ft_result_materials_by_infoID, args=(arg_tuple,))
        # jobs.append(p)
        # p.start()
        # p.join()




        infoID_to_overall_timestep_to_corr_ft_vector_dict = {}
        infoID_to_overall_timestep_to_corr_func_to_target_ft_dict = {}
        # idx_to_corr_ft_dict = {}
        # corr_ft_to_idx_dict={}

        for corr_materials_result_tuple in corr_materials_result_list:
            print("\ncorr_materials_result_tuple")
            print(corr_materials_result_tuple)
            cur_infoID_to_overall_timestep_to_corr_ft_vector_dict = corr_materials_result_tuple[0]
            cur_infoID_to_overall_timestep_to_corr_func_to_target_ft_dict = corr_materials_result_tuple[1]
            cur_idx_to_corr_ft_dict= corr_materials_result_tuple[2]
            cur_corr_ft_to_idx_dict = corr_materials_result_tuple[3]
            num_overall_timesteps = corr_materials_result_tuple[5]

            for infoID,infoID_dict in cur_infoID_to_overall_timestep_to_corr_ft_vector_dict.items():
                infoID_to_overall_timestep_to_corr_ft_vector_dict[infoID] = {}
                for overall_timestep, cur_avg_corr_ft_vals in infoID_dict.items():
                    infoID_to_overall_timestep_to_corr_ft_vector_dict[infoID][overall_timestep]=cur_avg_corr_ft_vals

            # infoID_to_overall_timestep_to_corr_func_to_target_ft_dict[infoID][overall_timestep][corr_func_str][cur_target_col]=avg_corr
            for infoID,infoID_dict in cur_infoID_to_overall_timestep_to_corr_func_to_target_ft_dict.items():
                infoID_to_overall_timestep_to_corr_func_to_target_ft_dict[infoID]={}
                for overall_timestep, overall_timestep_dict in infoID_dict.items():
                    infoID_to_overall_timestep_to_corr_func_to_target_ft_dict[infoID][overall_timestep]={}
                    for corr_func_str,corr_func_str_dict in overall_timestep_dict.items():
                        infoID_to_overall_timestep_to_corr_func_to_target_ft_dict[infoID][overall_timestep][corr_func_str]={}
                        for cur_target_col, avg_corr in corr_func_str_dict.items():
                            infoID_to_overall_timestep_to_corr_func_to_target_ft_dict[infoID][overall_timestep][corr_func_str][cur_target_col]=avg_corr




        return infoID_to_overall_timestep_to_corr_ft_vector_dict,infoID_to_overall_timestep_to_corr_func_to_target_ft_dict,idx_to_corr_ft_dict,corr_ft_to_idx_dict,new_corr_fts,num_overall_timesteps

    def specific_get_correlation_ft_materials_dict_for_exo_corr_exp_para(self,desired_array_types = [ "val", "test"]):

        array_type_to_correlation_materials_dict = {}

        # array_types = ["val_sliding_window"]
        for array_type in desired_array_types:
            print("\nGetting corr materials for %s"%array_type)
            array_type_to_correlation_materials_dict[array_type] = {}
            infoID_to_overall_timestep_to_corr_ft_vector_dict,infoID_to_overall_timestep_to_corr_func_to_target_ft_dict,idx_to_corr_ft_dict,corr_ft_to_idx_dict,new_corr_fts,num_overall_timesteps = self.insert_correlation_features_for_corr_exp_para(array_type,DEBUG_PRINT=False)
            array_type_to_correlation_materials_dict[array_type]["infoID_to_overall_timestep_to_corr_ft_vector_dict"] = infoID_to_overall_timestep_to_corr_ft_vector_dict.copy()
            array_type_to_correlation_materials_dict[array_type]["infoID_to_overall_timestep_to_corr_func_to_target_ft_dict"] = infoID_to_overall_timestep_to_corr_func_to_target_ft_dict.copy()
            array_type_to_correlation_materials_dict[array_type]["num_overall_timesteps"]=num_overall_timesteps

        self.array_type_to_correlation_materials_dict = array_type_to_correlation_materials_dict
        print("\nDone getting corr materials!")
        return array_type_to_correlation_materials_dict

    def specific_get_correlation_ft_materials_dict_for_exo_corr_exp_para_v2_full_data(self,desired_array_types = [ "val", "test"]):

        array_type_to_correlation_materials_dict = {}

        # array_types = ["val_sliding_window"]
        for array_type in desired_array_types:
            print("\nGetting corr materials for %s"%array_type)
            array_type_to_correlation_materials_dict[array_type] = {}
            infoID_to_overall_timestep_to_corr_ft_vector_dict,infoID_to_overall_timestep_to_corr_func_to_target_ft_dict,idx_to_corr_ft_dict,corr_ft_to_idx_dict,new_corr_fts,num_overall_timesteps = self.insert_correlation_features_for_corr_exp_para_v2(array_type,DEBUG_PRINT=False)
            array_type_to_correlation_materials_dict[array_type]["infoID_to_overall_timestep_to_corr_ft_vector_dict"] = infoID_to_overall_timestep_to_corr_ft_vector_dict.copy()
            array_type_to_correlation_materials_dict[array_type]["infoID_to_overall_timestep_to_corr_func_to_target_ft_dict"] = infoID_to_overall_timestep_to_corr_func_to_target_ft_dict.copy()
            array_type_to_correlation_materials_dict[array_type]["num_overall_timesteps"]=num_overall_timesteps

        self.array_type_to_correlation_materials_dict = array_type_to_correlation_materials_dict
        print("\nDone getting corr materials!")
        return array_type_to_correlation_materials_dict

    def insert_correlation_features_for_corr_exp_para_v2(self, array_tag,DEBUG_PRINT=False):

        def debug_print(print_val="\n"):
            if DEBUG_PRINT == True:
                print(str(print_val))

        x_array_tag = "x_" + array_tag
        y_array_tag = "y_" + array_tag

        num_infoIDs = len(self.infoIDs)



        #create the new fts to be added
        new_corr_fts = []
        print("\nnew corr fts: \n")
        for corr_func_str in self.CORRELATION_FUNCS_TO_USE_LIST:
            for cur_target_col in self.target_ft_categories:
                new_ft = "%s_corr_%s_with_aux_time_series"%(corr_func_str, cur_target_col)
                print(new_ft)
                new_corr_fts.append(new_ft)
        # sys.exit(0)

        #make new dicts
        idx_to_corr_ft_dict = {}
        corr_ft_to_idx_dict = {}
        for idx,corr_ft in enumerate(new_corr_fts):
            idx_to_corr_ft_dict[idx]=corr_ft
            corr_ft_to_idx_dict[corr_ft]=idx

        self.idx_to_corr_ft_dict=idx_to_corr_ft_dict
        self.corr_ft_to_idx_dict=corr_ft_to_idx_dict
        corr_fts = list(new_corr_fts)
        self.corr_fts = list(corr_fts)




        # print("\nGetting corr vectors...")
        # corr_materials_result_list = []
        # for infoID_idx, infoID in enumerate(self.infoIDs):
        #   corr_materials_result_list = self.get_corr_ft_result_materials_by_infoID(infoID,infoID_idx,x_array_tag,y_array_tag,num_infoIDs,new_corr_fts)
        # print("\nnum corr_materials_result_list")
        # print(len(corr_materials_result_list))


        #=========== multi proc #===========
        print("\nGetting corr vectors...")
        pool = mp.Pool(processes= self.n_jobs)
        corr_materials_result_list = []
        arg_tuple_list = []
        for infoID_idx, infoID in enumerate(self.infoIDs):
            cur_arg_tuple = (infoID,infoID_idx,array_tag,num_infoIDs,new_corr_fts,False,idx_to_corr_ft_dict,corr_ft_to_idx_dict,corr_fts)
            arg_tuple_list.append(cur_arg_tuple)

        print("\nLaunching parallel func...")
        #corr_materials_result_list = [pool.apply_sync(self.get_corr_ft_result_materials_by_infoID, args=arg_tuple) for arg_tuple in  arg_tuple_list]
        corr_materials_result_list = pool.map(self.get_corr_ft_result_materials_by_infoID_v2_full, arg_tuple_list)
        print("\nnum corr_materials_result_list")
        print(len(corr_materials_result_list))
        #===========#===========#===========


        # #train model
        # pool = multiprocessing.Pool(processes=self.n_)
        # p = Process(target = get_corr_ft_result_materials_by_infoID, args=(arg_tuple,))
        # jobs.append(p)
        # p.start()
        # p.join()




        infoID_to_overall_timestep_to_corr_ft_vector_dict = {}
        infoID_to_overall_timestep_to_corr_func_to_target_ft_dict = {}
        # idx_to_corr_ft_dict = {}
        # corr_ft_to_idx_dict={}

        for corr_materials_result_tuple in corr_materials_result_list:
            print("\ncorr_materials_result_tuple")
            print(corr_materials_result_tuple)
            cur_infoID_to_overall_timestep_to_corr_ft_vector_dict = corr_materials_result_tuple[0]
            cur_infoID_to_overall_timestep_to_corr_func_to_target_ft_dict = corr_materials_result_tuple[1]
            cur_idx_to_corr_ft_dict= corr_materials_result_tuple[2]
            cur_corr_ft_to_idx_dict = corr_materials_result_tuple[3]
            num_overall_timesteps = corr_materials_result_tuple[5]

            for infoID,infoID_dict in cur_infoID_to_overall_timestep_to_corr_ft_vector_dict.items():
                infoID_to_overall_timestep_to_corr_ft_vector_dict[infoID] = {}
                for overall_timestep, cur_avg_corr_ft_vals in infoID_dict.items():
                    infoID_to_overall_timestep_to_corr_ft_vector_dict[infoID][overall_timestep]=cur_avg_corr_ft_vals

            # infoID_to_overall_timestep_to_corr_func_to_target_ft_dict[infoID][overall_timestep][corr_func_str][cur_target_col]=avg_corr
            for infoID,infoID_dict in cur_infoID_to_overall_timestep_to_corr_func_to_target_ft_dict.items():
                infoID_to_overall_timestep_to_corr_func_to_target_ft_dict[infoID]={}
                for overall_timestep, overall_timestep_dict in infoID_dict.items():
                    infoID_to_overall_timestep_to_corr_func_to_target_ft_dict[infoID][overall_timestep]={}
                    for corr_func_str,corr_func_str_dict in overall_timestep_dict.items():
                        infoID_to_overall_timestep_to_corr_func_to_target_ft_dict[infoID][overall_timestep][corr_func_str]={}
                        for cur_target_col, avg_corr in corr_func_str_dict.items():
                            infoID_to_overall_timestep_to_corr_func_to_target_ft_dict[infoID][overall_timestep][corr_func_str][cur_target_col]=avg_corr




        return infoID_to_overall_timestep_to_corr_ft_vector_dict,infoID_to_overall_timestep_to_corr_func_to_target_ft_dict,idx_to_corr_ft_dict,corr_ft_to_idx_dict,new_corr_fts,num_overall_timesteps

    def get_corr_ft_result_materials_by_infoID_v2_full_backup(self, arg_tuple ):

        def debug_print(print_val="\n"):
            if DEBUG_PRINT == True:
                print(str(print_val))

        infoID,infoID_idx,array_tag,num_infoIDs,new_corr_fts,DEBUG_PRINT,idx_to_corr_ft_dict,corr_ft_to_idx_dict,corr_fts= arg_tuple

        x_array_tag = "x_" + array_tag
        y_array_tag = "y_" + array_tag

        overall_timestep_to_dynamic_ft_df_dict = self.array_type_to_dynamic_ft_df_dict[array_tag]
        infoID_to_overall_timestep_to_corr_ft_vector_dict = {}
        infoID_to_overall_timestep_to_corr_func_to_target_ft_dict = {}

        print("\nGetting corr vectors for %s..."%infoID)
        infoID_to_overall_timestep_to_corr_func_to_target_ft_dict[infoID]={}
        infoID_to_overall_timestep_to_corr_ft_vector_dict[infoID]={}
        for overall_timestep, cur_timestep_df in overall_timestep_to_dynamic_ft_df_dict[infoID].items():
            infoID_to_overall_timestep_to_corr_ft_vector_dict[infoID][overall_timestep]={}
            infoID_to_overall_timestep_to_corr_func_to_target_ft_dict[infoID][overall_timestep]={}
            debug_print(infoID)
            debug_print(overall_timestep)
            debug_print(cur_timestep_df)


            cur_avg_corr_ft_vals = []
            for corr_func_str in self.CORRELATION_FUNCS_TO_USE_LIST:
                infoID_to_overall_timestep_to_corr_func_to_target_ft_dict[infoID][overall_timestep][corr_func_str]={}

                corr_func = self.CORRELATION_STR_TO_FUNC_DICT[corr_func_str]
                debug_print("\ncorr_func: ")
                debug_print(corr_func)

                for cur_target_col in self.target_ft_categories:

                    # cur_target_corr_ft_vals = []
                    debug_print("")
                    debug_print(corr_func_str)
                    debug_print(cur_target_col)

                    cur_target_ts_of_interest = cur_timestep_df[cur_target_col]

                    #==================== get aux df with just exo ==========================
                    cur_aux_df_without_target = cur_timestep_df.copy().drop(cur_target_col, axis=1)

                    # keep_cols = []

                    # if self.GET_GDELT_FEATURES == True:
                    #   keep_cols = keep_cols + ["AvgTone", "GoldsteinScale", "NumMentions"]

                    # if self.GET_REDDIT_FEATURES == True:
                    #   keep_cols = keep_cols + ["num_reddit_activities"]

                    # if self.GET_EXTERNAL_PLATFORM_FTS == True:
                    #   # if self.TARGET_PLATFORM == "twitter":
                    #   #   keep_cols = keep_cols + get_fts_from_fp(self.feature_list_dir + "twitter_dynamic_global_fts.txt")
                    #   #   keep_cols = keep_cols + get_fts_from_fp(self.feature_list_dir + "twitter_dynamic_pair_fts.txt")
                    #   if self.TARGET_PLATFORM == "twitter":
                    #       keep_cols = keep_cols + get_fts_from_fp(self.feature_list_dir + "youtube_dynamic_global_fts.txt")
                    #       keep_cols = keep_cols + get_fts_from_fp(self.feature_list_dir + "youtube_dynamic_pair_fts.txt")

                    #   if self.TARGET_PLATFORM == "youtube":
                    #       keep_cols = keep_cols + get_fts_from_fp(self.feature_list_dir + "twitter_dynamic_global_fts.txt")
                    #       keep_cols = keep_cols + get_fts_from_fp(self.feature_list_dir + "twitter_dynamic_pair_fts.txt")

                    # cur_aux_df_without_target = cur_aux_df_without_target[keep_cols]



                    debug_print("\ncur_aux_df_without_target")
                    debug_print(cur_aux_df_without_target)

                    if cur_aux_df_without_target.shape[1] == 1:
                        col_to_copy = list(cur_aux_df_without_target)[0]
                        cur_aux_df_without_target["dupe_%s"%col_to_copy] = cur_aux_df_without_target[col_to_copy].copy()

                        debug_print("\ncur_aux_df_without_target only had 1 col")
                        debug_print("Made a copy of the col for easier computation")
                        debug_print(cur_aux_df_without_target)
                    #===========================================================

                    debug_print("\ncur_target_ts_of_interest")
                    debug_print(cur_target_ts_of_interest)

                    debug_print("\ncur_aux_df_without_target")
                    debug_print(cur_aux_df_without_target)

                    aux_cols = list(cur_aux_df_without_target)
                    # for aux_col in aux_cols:
                    #   cur_aux_series = cur_aux_df_without_target[aux_col]

                    cur_matrix_df = pd.concat([cur_target_ts_of_interest, cur_aux_df_without_target], axis=1)
                    debug_print("\ncur_matrix_df")
                    debug_print(cur_matrix_df)
                    cur_target_corr_ft_vals_matrix= corr_func(cur_matrix_df ,axis=0)[0]
                    # cur_target_corr_ft_vals_matrix= corr_func(cur_target_ts_of_interest, cur_aux_df_without_target.T,axis=1)[0]
                    debug_print("\ncur_target_corr_ft_vals_matrix")
                    debug_print(cur_target_corr_ft_vals_matrix)

                    debug_print("\ncur_target_corr_ft_vals_matrix shape")
                    debug_print(cur_target_corr_ft_vals_matrix.shape)

                    debug_print("\ncur_target_corr_ft_vals_matrix entry 0")
                    debug_print(cur_target_corr_ft_vals_matrix[0])

                    debug_print("\ncur_target_corr_ft_vals_matrix entry 1")
                    debug_print(cur_target_corr_ft_vals_matrix[1])

                    debug_print("\ncur_target_corr_ft_vals_matrix entry 0 shape")
                    debug_print(cur_target_corr_ft_vals_matrix[0].shape)

                    debug_print("\ncur_target_corr_ft_vals_matrix entry 1 shape")
                    debug_print(cur_target_corr_ft_vals_matrix[1].shape)

                    cur_ft_matrix_cols = list(cur_matrix_df)
                    cur_aux_corr_vals=np.abs(cur_target_corr_ft_vals_matrix[0])
                    for cur_ft, cur_aux_corr in zip(cur_ft_matrix_cols,cur_aux_corr_vals):
                        debug_print("%s: %.4f"%(cur_ft, cur_aux_corr))

                    cur_aux_corr_vals = pd.Series(cur_aux_corr_vals)
                    debug_print(cur_aux_corr_vals)
                    cur_aux_corr_vals = cur_aux_corr_vals.dropna().values
                    avg_corr = np.mean(cur_aux_corr_vals)
                    # cur_avg_corr_ft_vals.append(avg_corr)
                    # debug_print("\nInfoID: %s; Timestep %d; %s to aux %s corrs: "%( infoID,overall_timestep,cur_target_col,corr_func_str ))
                    # # cur_target_corr_ft_vals = cur_target_corr_ft_vals[~np.isnan(cur_target_corr_ft_vals)]
                    # cur_target_corr_ft_vals = np.abs(cur_target_corr_ft_vals)
                    # debug_print(cur_target_corr_ft_vals)

                    #now take the avg
                    # avg_corr = np.mean(cur_target_corr_ft_vals)
                    if  np.isnan(avg_corr):
                        avg_corr = 0
                        debug_print("\navg corr was nan, so we converted it to 0")
                        # sys.exit(0)

                    debug_print("%s %d %s %s corr: %.4f"%(infoID, overall_timestep, corr_func_str, cur_target_col, avg_corr))
                    infoID_to_overall_timestep_to_corr_func_to_target_ft_dict[infoID][overall_timestep][corr_func_str][cur_target_col]=avg_corr
                    cur_avg_corr_ft_vals.append(avg_corr)
                    # sys.exit(0)
            debug_print("\ncur_avg_corr_ft_vals")
            debug_print(cur_avg_corr_ft_vals)
            debug_print("\nwith fillna")
            cur_avg_corr_ft_vals = pd.Series(cur_avg_corr_ft_vals).fillna(0).values
            debug_print(cur_avg_corr_ft_vals)

            debug_print("\ncorr scores")
            for corr_ft,corr_score in zip(new_corr_fts, cur_avg_corr_ft_vals):
                debug_print("%s: %.4f"%(corr_ft, corr_score))

            infoID_to_overall_timestep_to_corr_ft_vector_dict[infoID][overall_timestep]=cur_avg_corr_ft_vals
        print("Got corr fts for %s"%infoID)

        # #make dicts
        # self.infoID_to_overall_timestep_to_corr_ft_vector_dict=infoID_to_overall_timestep_to_corr_ft_vector_dict
        # self.infoID_to_overall_timestep_to_corr_func_to_target_ft_dict=infoID_to_overall_timestep_to_corr_func_to_target_ft_dict
        num_overall_timesteps = overall_timestep+1

        return (infoID_to_overall_timestep_to_corr_ft_vector_dict,infoID_to_overall_timestep_to_corr_func_to_target_ft_dict,idx_to_corr_ft_dict,corr_ft_to_idx_dict,new_corr_fts,num_overall_timesteps)

    def get_corr_ft_result_materials_by_infoID_v2_full(self, arg_tuple ):



        def debug_print(print_val="\n"):
            if DEBUG_PRINT == True:
                print(str(print_val))

        infoID,infoID_idx,array_tag,num_infoIDs,new_corr_fts,DEBUG_PRINT,idx_to_corr_ft_dict,corr_ft_to_idx_dict,corr_fts= arg_tuple

        DEBUG_PRINT=True

        x_array_tag = "x_" + array_tag
        y_array_tag = "y_" + array_tag

        overall_timestep_to_dynamic_ft_df_dict = self.array_type_to_dynamic_ft_df_dict[array_tag]
        infoID_to_overall_timestep_to_corr_ft_vector_dict = {}
        infoID_to_overall_timestep_to_corr_func_to_target_ft_dict = {}

        print("\nGetting corr vectors for %s..."%infoID)
        infoID_to_overall_timestep_to_corr_func_to_target_ft_dict[infoID]={}
        infoID_to_overall_timestep_to_corr_ft_vector_dict[infoID]={}
        for overall_timestep, cur_timestep_df in overall_timestep_to_dynamic_ft_df_dict[infoID].items():
            infoID_to_overall_timestep_to_corr_ft_vector_dict[infoID][overall_timestep]={}
            infoID_to_overall_timestep_to_corr_func_to_target_ft_dict[infoID][overall_timestep]={}
            debug_print(infoID)
            debug_print(overall_timestep)
            debug_print(cur_timestep_df)


            cur_avg_corr_ft_vals = []
            for corr_func_str in self.CORRELATION_FUNCS_TO_USE_LIST:
                infoID_to_overall_timestep_to_corr_func_to_target_ft_dict[infoID][overall_timestep][corr_func_str]={}

                corr_func = self.CORRELATION_STR_TO_FUNC_DICT[corr_func_str]
                debug_print("\ncorr_func: ")
                debug_print(corr_func)

                for cur_target_col in self.target_ft_categories:

                    # cur_target_corr_ft_vals = []
                    debug_print("")
                    debug_print(corr_func_str)
                    debug_print(cur_target_col)

                    cur_target_ts_of_interest = cur_timestep_df[cur_target_col]

                    #==================== get aux df with just exo ==========================
                    cur_aux_df_without_target = cur_timestep_df.copy().drop(cur_target_col, axis=1)

                    # keep_cols = []

                    # if self.GET_GDELT_FEATURES == True:
                    #   keep_cols = keep_cols + ["AvgTone", "GoldsteinScale", "NumMentions"]

                    # if self.GET_REDDIT_FEATURES == True:
                    #   keep_cols = keep_cols + ["num_reddit_activities"]

                    # if self.GET_EXTERNAL_PLATFORM_FTS == True:
                    #   # if self.TARGET_PLATFORM == "twitter":
                    #   #   keep_cols = keep_cols + get_fts_from_fp(self.feature_list_dir + "twitter_dynamic_global_fts.txt")
                    #   #   keep_cols = keep_cols + get_fts_from_fp(self.feature_list_dir + "twitter_dynamic_pair_fts.txt")
                    #   if self.TARGET_PLATFORM == "twitter":
                    #       keep_cols = keep_cols + get_fts_from_fp(self.feature_list_dir + "youtube_dynamic_global_fts.txt")
                    #       keep_cols = keep_cols + get_fts_from_fp(self.feature_list_dir + "youtube_dynamic_pair_fts.txt")

                    #   if self.TARGET_PLATFORM == "youtube":
                    #       keep_cols = keep_cols + get_fts_from_fp(self.feature_list_dir + "twitter_dynamic_global_fts.txt")
                    #       keep_cols = keep_cols + get_fts_from_fp(self.feature_list_dir + "twitter_dynamic_pair_fts.txt")

                    # cur_aux_df_without_target = cur_aux_df_without_target[keep_cols]



                    debug_print("\ncur_aux_df_without_target")
                    debug_print(cur_aux_df_without_target)

                    if cur_aux_df_without_target.shape[1] == 1:
                        col_to_copy = list(cur_aux_df_without_target)[0]
                        cur_aux_df_without_target["dupe_%s"%col_to_copy] = cur_aux_df_without_target[col_to_copy].copy()

                        debug_print("\ncur_aux_df_without_target only had 1 col")
                        debug_print("Made a copy of the col for easier computation")
                        debug_print(cur_aux_df_without_target)
                    #===========================================================

                    debug_print("\ncur_target_ts_of_interest")
                    debug_print(cur_target_ts_of_interest)

                    debug_print("\ncur_aux_df_without_target")
                    debug_print(cur_aux_df_without_target)

                    aux_cols = list(cur_aux_df_without_target)
                    # for aux_col in aux_cols:
                    #   cur_aux_series = cur_aux_df_without_target[aux_col]

                    cur_matrix_df = pd.concat([cur_target_ts_of_interest, cur_aux_df_without_target], axis=1)
                    debug_print("\ncur_matrix_df")
                    debug_print(cur_matrix_df)

                    try:
                        cur_target_corr_ft_vals_matrix= corr_func(cur_matrix_df ,axis=0)[0]
                    except:
                        cur_target_corr_ft_vals_matrix= corr_func(cur_matrix_df,cur_matrix_df)

                    # sys.exit(0)
                    # cur_target_corr_ft_vals_matrix= corr_func(cur_target_ts_of_interest, cur_aux_df_without_target.T,axis=1)[0]
                    debug_print("\ncur_target_corr_ft_vals_matrix")
                    debug_print(cur_target_corr_ft_vals_matrix)

                    debug_print("\ncur_target_corr_ft_vals_matrix shape")
                    debug_print(cur_target_corr_ft_vals_matrix.shape)

                    debug_print("\ncur_target_corr_ft_vals_matrix entry 0")
                    debug_print(cur_target_corr_ft_vals_matrix[0])

                    debug_print("\ncur_target_corr_ft_vals_matrix entry 1")
                    debug_print(cur_target_corr_ft_vals_matrix[1])

                    debug_print("\ncur_target_corr_ft_vals_matrix entry 0 shape")
                    debug_print(cur_target_corr_ft_vals_matrix[0].shape)

                    debug_print("\ncur_target_corr_ft_vals_matrix entry 1 shape")
                    debug_print(cur_target_corr_ft_vals_matrix[1].shape)

                    # sys.exit(0)

                    cur_ft_matrix_cols = list(cur_matrix_df)
                    cur_aux_corr_vals=np.abs(cur_target_corr_ft_vals_matrix[0])
                    for cur_ft, cur_aux_corr in zip(cur_ft_matrix_cols,cur_aux_corr_vals):
                        debug_print("%s: %.4f"%(cur_ft, cur_aux_corr))

                    cur_aux_corr_vals = pd.Series(cur_aux_corr_vals)
                    debug_print(cur_aux_corr_vals)
                    cur_aux_corr_vals = cur_aux_corr_vals.dropna().values
                    avg_corr = np.mean(cur_aux_corr_vals)
                    # cur_avg_corr_ft_vals.append(avg_corr)
                    # debug_print("\nInfoID: %s; Timestep %d; %s to aux %s corrs: "%( infoID,overall_timestep,cur_target_col,corr_func_str ))
                    # # cur_target_corr_ft_vals = cur_target_corr_ft_vals[~np.isnan(cur_target_corr_ft_vals)]
                    # cur_target_corr_ft_vals = np.abs(cur_target_corr_ft_vals)
                    # debug_print(cur_target_corr_ft_vals)

                    #now take the avg
                    # avg_corr = np.mean(cur_target_corr_ft_vals)
                    if  np.isnan(avg_corr):
                        avg_corr = 0
                        debug_print("\navg corr was nan, so we converted it to 0")
                        # sys.exit(0)

                    debug_print("%s %d %s %s corr: %.4f"%(infoID, overall_timestep, corr_func_str, cur_target_col, avg_corr))
                    infoID_to_overall_timestep_to_corr_func_to_target_ft_dict[infoID][overall_timestep][corr_func_str][cur_target_col]=avg_corr
                    cur_avg_corr_ft_vals.append(avg_corr)
                    # sys.exit(0)
            debug_print("\ncur_avg_corr_ft_vals")
            debug_print(cur_avg_corr_ft_vals)
            debug_print("\nwith fillna")
            cur_avg_corr_ft_vals = pd.Series(cur_avg_corr_ft_vals).fillna(0).values
            debug_print(cur_avg_corr_ft_vals)

            debug_print("\ncorr scores")
            for corr_ft,corr_score in zip(new_corr_fts, cur_avg_corr_ft_vals):
                debug_print("%s: %.4f"%(corr_ft, corr_score))

            infoID_to_overall_timestep_to_corr_ft_vector_dict[infoID][overall_timestep]=cur_avg_corr_ft_vals
        print("Got corr fts for %s"%infoID)

        # #make dicts
        # self.infoID_to_overall_timestep_to_corr_ft_vector_dict=infoID_to_overall_timestep_to_corr_ft_vector_dict
        # self.infoID_to_overall_timestep_to_corr_func_to_target_ft_dict=infoID_to_overall_timestep_to_corr_func_to_target_ft_dict
        num_overall_timesteps = overall_timestep+1

        return (infoID_to_overall_timestep_to_corr_ft_vector_dict,infoID_to_overall_timestep_to_corr_func_to_target_ft_dict,idx_to_corr_ft_dict,corr_ft_to_idx_dict,new_corr_fts,num_overall_timesteps)

    def get_array_shape_df(self, tags):

        x_shapes = []
        y_shapes = []
        date_tuples = []

        # self.target_date_tuple_dict

        for tag in tags:

            if tag=="train":
                x_shape = self.x_train.shape
                y_shape = self.y_train.shape
                x_shapes.append(x_shape)
                y_shapes.append(y_shape)

                target_date_tuple = self.target_date_tuple_dict["y_train"]
                date_tuples.append(target_date_tuple)

            if tag=="val":
                x_shape = self.x_val.shape
                y_shape = self.y_val.shape
                x_shapes.append(x_shape)
                y_shapes.append(y_shape)

                target_date_tuple = self.target_date_tuple_dict["y_val"]
                date_tuples.append(target_date_tuple)

            if tag=="val_sliding_window":
                x_shape = self.x_val_sliding_window.shape
                y_shape = self.y_val_sliding_window.shape
                x_shapes.append(x_shape)
                y_shapes.append(y_shape)

                target_date_tuple = self.target_date_tuple_dict["y_val_sliding_window"]
                date_tuples.append(target_date_tuple)

            if tag=="test_sliding_window":
                x_shape = self.x_test_sliding_window.shape
                y_shape = self.y_test_sliding_window.shape
                x_shapes.append(x_shape)
                y_shapes.append(y_shape)

                target_date_tuple = self.target_date_tuple_dict["y_test_sliding_window"]
                date_tuples.append(target_date_tuple)


            if tag=="test":
                x_shape = self.x_test.shape
                y_shape = self.y_test.shape
                x_shapes.append(x_shape)
                y_shapes.append(y_shape)

                target_date_tuple = self.target_date_tuple_dict["y_test"]
                date_tuples.append(target_date_tuple)


        df = pd.DataFrame(data={"x_shape":x_shapes, "y_shape":y_shapes, "tag":tags, "target_start_date_range":date_tuples})
        df = df[["tag", "x_shape", "y_shape","target_start_date_range"]]






        return df
