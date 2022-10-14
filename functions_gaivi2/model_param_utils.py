import pandas as pd 
import os,sys
import numpy as np 
from basic_utils import *
from ast import literal_eval

def bool_str_to_bool(bool_str):

    if bool_str == "True":
        return True
    if bool_str == "False":
        return False
    print("Error! String is not bool!")
    sys.exit(0)


def get_selected_models_param_dict_v2(all_models_df,selected_models,overwrite_params_dict):

    cur_dict_idx = 0
    idx_to_data_and_model_materials_dict = {}

    selected_models_df = all_models_df[all_models_df["model_tag"].isin(selected_models)].reset_index(drop=True)
    print("\nselected_models_df")
    print(selected_models_df)

    data_param_num_combos = selected_models_df.shape[0]
    print("\ndata_param_num_combos")
    print(data_param_num_combos)

    model_tag_to_dir_dict = convert_df_2_cols_to_dict(selected_models_df, "model_tag", "model_dir")
    for model_tag,model_dir in model_tag_to_dir_dict.items():
        print("%s: %s"%(model_tag , model_dir))

        #current param fp
        param_fp = model_dir + "full-params.csv"
        param_df = pd.read_csv(param_fp)
        print(param_df)

        if "Unnamed: 0" in list(param_df):
            param_df = param_df.drop("Unnamed: 0",axis=1)

        input_param_dict = convert_df_2_cols_to_dict(param_df, "param", "value")
        print("\ninput_param_dict")
        print(input_param_dict)
        sample_generator_param_dict = {}

        try:
            sample_generator_param_dict["INPUT_HOURS"] = int(input_param_dict["INPUT_HOURS"])
        except KeyError:
            sample_generator_param_dict["INPUT_HOURS"] = int(input_param_dict[" INPUT_HOURS"])


        sample_generator_param_dict["train_start"] = input_param_dict["train_start"]
        sample_generator_param_dict["train_end"] = input_param_dict["train_end"]

        sample_generator_param_dict["val_start"] = input_param_dict["val_start"]
        sample_generator_param_dict["val_end"] = input_param_dict["val_end"]

        sample_generator_param_dict["test_start"] = input_param_dict["test_start"]
        sample_generator_param_dict["test_end"] = input_param_dict["test_end"]

        sample_generator_param_dict["GRAN"] = input_param_dict["GRAN"]
        sample_generator_param_dict["OUTPUT_HOURS"] = int(input_param_dict["OUTPUT_HOURS"])
        sample_generator_param_dict["GET_1HOT_INFO_ID_FTS"] =  bool_str_to_bool(input_param_dict["GET_1HOT_INFO_ID_FTS"])
        sample_generator_param_dict["GET_GDELT_FEATURES"] =  bool_str_to_bool(input_param_dict["GET_GDELT_FEATURES"])
        sample_generator_param_dict["GET_REDDIT_FEATURES"] =  bool_str_to_bool(input_param_dict["GET_REDDIT_FEATURES"])
        sample_generator_param_dict["TARGET_PLATFORM"] = input_param_dict["TARGET_PLATFORM"]
        sample_generator_param_dict["GET_EXTERNAL_PLATFORM_FTS"] = bool_str_to_bool(input_param_dict["GET_EXTERNAL_PLATFORM_FTS"])
        sample_generator_param_dict["RESCALE"] = bool_str_to_bool(input_param_dict["RESCALE"])
        sample_generator_param_dict["RESCALE_TARGET"] = bool_str_to_bool(input_param_dict["RESCALE_TARGET"])
        sample_generator_param_dict["SCALER_TYPE"] = input_param_dict["SCALER_TYPE"]
        sample_generator_param_dict["FEATURE_RANGE"] = literal_eval(input_param_dict["FEATURE_RANGE"])
        sample_generator_param_dict["INPUT_TWITTER_LOGNORM"] = int(input_param_dict["INPUT_TWITTER_LOGNORM"])
        sample_generator_param_dict["OUTPUT_TWITTER_LOGNORM"] = int(input_param_dict["OUTPUT_TWITTER_LOGNORM"])
        sample_generator_param_dict["INPUT_OTHER_LOGNORM"] = int(input_param_dict["INPUT_OTHER_LOGNORM"])
        sample_generator_param_dict["OUTPUT_OTHER_LOGNORM"] = int(input_param_dict["OUTPUT_OTHER_LOGNORM"])
        sample_generator_param_dict["LOGNORM_IGNORE_TAGS"] = literal_eval(input_param_dict["LOGNORM_IGNORE_TAGS"])
        sample_generator_param_dict["CORRELATION_FUNCS_TO_USE_LIST"] = literal_eval(input_param_dict["CORRELATION_FUNCS_TO_USE_LIST"])
        sample_generator_param_dict["n_jobs"] = int(input_param_dict["n_jobs"])

        sample_generator_param_dict.update(overwrite_params_dict)
        # idx_to_sample_generator_param_dict[cur_dict_idx]=sample_generator_param_dict

        # all_model_params = [
        # "objective","colsample_bytree","learning_rate","max_depth","n_estimators",
        #   "subsample","gamma","alpha","reg_lambda","random_state","n_jobs"
        #   ]

        int_params = ["max_depth","n_estimators","random_state","n_jobs"]
        float_params = ["colsample_bytree","learning_rate",
            "subsample","gamma","alpha","reg_lambda"]

        model_param_dict = {}
        model_param_dict["objective"] = input_param_dict["objective"]
        for param in int_params:
            model_param_dict[param] = int(input_param_dict[param])
        for param in float_params:
            model_param_dict[param] = float(input_param_dict[param])

        idx_to_data_and_model_materials_dict[cur_dict_idx] =( sample_generator_param_dict , model_param_dict ,model_tag)

        cur_dict_idx+=1

    return idx_to_data_and_model_materials_dict