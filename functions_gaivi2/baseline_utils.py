import os,sys
import pandas as pd 
import numpy as np


def get_pnnl_baseline_dates(sample_gen_param_dict):

    #get num days in test period
    test_start = sample_gen_param_dict["test_start"]
    test_end = sample_gen_param_dict["test_end"]
    train_start = sample_gen_param_dict["train_start"]
    train_end = sample_gen_param_dict["train_end"]
    val_start = sample_gen_param_dict["val_start"]
    val_end = sample_gen_param_dict["val_end"]

    train_dates = pd.date_range(train_start, train_end, freq="S")
    val_dates = pd.date_range(val_start, val_end, freq="S")
    test_dates = pd.date_range(test_start, test_end, freq="S")

    train_date_series = pd.Series(train_dates)
    val_date_series = pd.Series(val_dates)
    test_date_series = pd.Series(test_dates)

    num_test_dates = len(test_date_series)
    new_test_dates =test_date_series - pd.Timedelta(seconds=num_test_dates)
    new_train_dates =train_date_series - pd.Timedelta(seconds=num_test_dates)
    new_val_dates =val_date_series - pd.Timedelta(seconds=num_test_dates)

    print("\nnum_test_dates: %d"%num_test_dates)
    print("\nnew_train_dates")
    print(new_train_dates)
    print("\nnew_val_dates")
    print(new_val_dates)
    print("\nnew_test_dates")
    print(new_test_dates)

    train_end = new_train_dates.iloc[-1]
    val_start = new_val_dates.iloc[0]
    val_end = new_val_dates.iloc[-1]
    test_start = new_test_dates.iloc[0]
    test_end = new_test_dates.iloc[-1]

    sample_gen_param_dict["train_start"]=pd.to_datetime(train_start, utc=True)
    sample_gen_param_dict["train_end"]=pd.to_datetime(train_end, utc=True)
    sample_gen_param_dict["val_start"]=pd.to_datetime(val_start, utc=True)
    sample_gen_param_dict["val_end"]=pd.to_datetime(val_end, utc=True)
    sample_gen_param_dict["test_start"]=pd.to_datetime(test_start, utc=True)
    sample_gen_param_dict["test_end"]=pd.to_datetime(test_end, utc=True)

    key_prints = ["train_start", "train_end","val_start", "val_end","test_start", "test_end"]

    print("\nnew dates")
    for date_key in key_prints:
        print("%s: %s"%(date_key, str(sample_gen_param_dict[date_key])))
    # sys.exit(0)



    return sample_gen_param_dict

def get_pnnl_baseline_data(sample_gen_param_dict,SampleGenerator):

    baseline_dates_sample_gen_param_dict = dict(get_pnnl_baseline_dates(sample_gen_param_dict))

    sample_generator = SampleGenerator(**baseline_dates_sample_gen_param_dict)
    print("\nsample_generator")
    print(sample_generator)



    infoIDs = list(sample_gen_param_dict["infoIDs"])
    #==================== put corrs in here ====================

    #get array dict
    infoID_train_and_test_array_dict = sample_generator.configure_features_and_get_infoID_train_and_test_array_dict()

    print("\nMaking baselines...")
    y_pred_baseline_test_data_dict = {}
    y_pred_baseline_val_data_dict = {}
    y_pred_baseline_test_sliding_window_data_dict = {}
    y_pred_baseline_val_sliding_window_data_dict = {}
    for infoID in infoIDs:
        y_pred_baseline_test_data_dict[infoID] = infoID_train_and_test_array_dict[infoID]["y_test"]
        y_pred_baseline_val_data_dict[infoID] = infoID_train_and_test_array_dict[infoID]["y_val"]
        y_pred_baseline_test_sliding_window_data_dict[infoID] = infoID_train_and_test_array_dict[infoID]["y_test_sliding_window"]
        y_pred_baseline_val_sliding_window_data_dict[infoID] = infoID_train_and_test_array_dict[infoID]["y_val_sliding_window"]

    # # #baselines
    # print("\nMaking baselines...")
    # y_pred_baseline_test_data_dict = sample_generator.get_test_data_baseline_pred_dict().copy()
    # y_pred_baseline_val_data_dict = sample_generator.get_val_data_baseline_pred_dict().copy()
    # y_pred_baseline_test_sliding_window_data_dict = sample_generator.get_test_sliding_window_data_baseline_pred_dict().copy()
    # y_pred_baseline_val_sliding_window_data_dict = sample_generator.get_val_sliding_window_data_baseline_pred_dict().copy()

    return y_pred_baseline_test_data_dict,y_pred_baseline_val_data_dict,y_pred_baseline_test_sliding_window_data_dict,y_pred_baseline_val_sliding_window_data_dict


def get_baseline_data(sample_gen_param_dict,SampleGenerator):
    sample_generator = SampleGenerator(**sample_gen_param_dict)
    print("\nsample_generator")
    print(sample_generator)

    #==================== put corrs in here ====================

    #get array dict
    infoID_train_and_test_array_dict = sample_generator.configure_features_and_get_infoID_train_and_test_array_dict()

    # sys.exit(0)

    target_ft_category_to_list_of_temporal_indices_dict = sample_generator.get_target_ft_category_to_list_of_temporal_indices()
    for target_ft_cat,target_ft_cat_dict in target_ft_category_to_list_of_temporal_indices_dict.items():
        target_ft_idx_list = target_ft_cat_dict["indices_list"]
        print("%s: %s"%(target_ft_cat, target_ft_idx_list))


    input_ft_category_to_list_of_temporal_indices_dict = sample_generator.get_input_ft_category_to_list_of_temporal_indices()
    for input_ft_cat,input_ft_cat_dict in input_ft_category_to_list_of_temporal_indices_dict.items():
        print()
        input_ft_idx_list = input_ft_cat_dict["indices_list"]
        print("%s: %s"%(input_ft_cat, input_ft_idx_list))

    #============================================================

    #train data
    x_train,y_train = sample_generator.configure_training_data()


    # #get the other arrays
    x_val,y_val = sample_generator.configure_val_data()
    x_test,y_test = sample_generator.configure_test_data()
    x_val_sliding_window,y_val_sliding_window = sample_generator.configure_val_sliding_window_data()
    x_test_sliding_window,y_test_sliding_window = sample_generator.configure_test_sliding_window_data()

    # #baselines
    print("\nMaking baselines...")
    y_pred_baseline_test_data_dict = sample_generator.get_test_data_baseline_pred_dict().copy()
    y_pred_baseline_val_data_dict = sample_generator.get_val_data_baseline_pred_dict().copy()
    y_pred_baseline_test_sliding_window_data_dict = sample_generator.get_test_sliding_window_data_baseline_pred_dict().copy()
    y_pred_baseline_val_sliding_window_data_dict = sample_generator.get_val_sliding_window_data_baseline_pred_dict().copy()



    return y_pred_baseline_test_data_dict,y_pred_baseline_val_data_dict,y_pred_baseline_test_sliding_window_data_dict,y_pred_baseline_val_sliding_window_data_dict