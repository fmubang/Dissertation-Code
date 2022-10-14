import sys
#sys.path.append("/data/Fmubang/cp4-VAM-write-up-exps/functions")
# sys.path.append("/data/Fmubang/CP5-WEEKEND-REVISE-1-23/p0-functions")
# sys.path.append("/data/Fmubang/CP5-ALL-CHALLENGES/day1/functions")
#sys.path.append("/data/Fmubang/CP4-VAM-ORGANIZED-V2/functions")

#sys.path.append("/data/Fmubang/CP4-ORGANIZED-V3-FIX-DROPNA/functions")
sys.path.append("/storage2-mnt/data/fmubang/Defense-Github-Stuff-10-6-22/functions")



import pandas as pd 
import os,sys
from scipy import stats
import numpy as np
from pnnl_metric_funcs_v2 import *

import basic_utils as bu
import train_data_funcs as tdf
import fixed_volume_func as fvf
from infoIDs18 import get_cp4_challenge_18_infoIDs
import model_param_utils as mp
import baseline_utils as base_u
from SampleGenerator import SampleGenerator
from ModelHandler import ModelHandler,model_type_to_param_name_order_dict,xgboost_param_name_order,model_type_to_model_func_dict
from ft_importance_funcs import save_ft_info_v2_simpler
import model_eval_funcs as mef
import joblib


#================================= most important params ====================
DEBUG = True
INFO_ID_DEBUG=True
TARGET_PLATFORM="twitter"
main_output_tag = "CP4-M1-FIXED-DROPNA-FTS"

#============================= date stuff =============================
#these are desired eval periods
train_start = "2018-12-28"
train_end = "2019-02-07 23:59:59"

val_start = "2019-02-08"
val_end = "2019-02-14 23:59:59"

test_start = "2019-02-15"
test_end = "2019-03-07 23:59:59"

#================================= input dirs ====================
#main_input_dir = "/data/Fmubang/CP4-VAM-ORGANIZED-V2/data/P1-M1-VOLUME-DATA/DEBUG-False-Dates-12-24-2018-to-04-04-2019 23:59:59/"
#main_input_dir = "/data/Fmubang/CP4-VAM-ORGANIZED-V2/data/V2-WITH-EXO-P1-M1-VOLUME-DATA/DEBUG-False-Dates-12-24-2018-to-04-04-2019 23:59:59/"
# main_input_dir = "/data/Fmubang/CP4-ORGANIZED-V3-FIX-DROPNA/data/V2-WITH-EXO-P1-M1-VOLUME-DATA/DEBUG-False-Dates-12-24-2018-to-04-04-2019 23:59:59/"
# original_model_dir = "/data/Fmubang/CP4-ORGANIZED-V3-FIX-DROPNA/models/Original-VAM-CP4-Models/"

main_input_dir = "/storage2-mnt/data/fmubang/Defense-Github-Stuff-10-6-22/VAM-Vz19/data/VOLUME-DATA/DEBUG-False-Dates-12-24-2018-to-04-04-2019 23:59:59/"
original_model_dir = "/storage2-mnt/data/fmubang/Defense-Github-Stuff-10-6-22/VAM-Vz19/data/Original-VAM-Vz19-Models/"

#============================= get selected models =============================

model_platform_dir = original_model_dir + TARGET_PLATFORM + "/"
selected_models = os.listdir(model_platform_dir)
keep_models = []
for m in selected_models:
	if "VAM" in m:
		keep_models.append(m)
selected_models=list(keep_models)

#get all the models
all_model_dirs = []
for m in selected_models:
	cur_dir = model_platform_dir + m + "/"
	all_model_dirs.append(cur_dir)
all_models_df = pd.DataFrame(data={"model_tag":selected_models, "model_dir":all_model_dirs})
print("\nall_models_df")
print(all_models_df)

#============================= basic params =============================
GET_NEWS_DATA = False
if DEBUG == False:
	INFO_ID_DEBUG=False
DARPA_INPUT_FORMAT = False
remove_tags = "Ensemble"
platform = str(TARGET_PLATFORM)

#dates
start = "12-24-2018"
end = "04-04-2019 23:59:59"

#============================= debug stuff =============================
if DEBUG == True:
	if TARGET_PLATFORM == "twitter":
		selected_models = ["VAM-TR-24"]
	else:
		selected_models = ["VAM-Y-48"]

	all_models_df = all_models_df[all_models_df["model_tag"].isin(selected_models)]

#============================= more params =============================
DAY_AGG = True
if DAY_AGG == True:
	DAY_AGG_LIST = [True, False]
	agg_timestep_amount = 24
AGG_TIME_SERIES_LIST = [False]

VALIDATE_ON_SLIDING_WINDOW_DATA=True
VERBOSE=True
SAVE_MODEL=True
metrics_to_get=["rmse", "nc_rmse", "ape"]
SAVE_PREDS=True
LOGNORM_DEBUG_PRINT=False
SAVE_SCALERS_AND_METADATA=False
SLIDING_WINDOW_EVAL=False
CORRELATION_FUNCS_TO_USE_LIST_LIST = [[]]

#============================= metric stuff =============================

CORRELATION_STR_TO_FUNC_DICT={
	"pearson":stats.pearsonr,
	"spearman":stats.spearmanr
}

metric_str_to_function_dict={
	"rmse":get_regular_rmse,
	"nc_rmse":get_nc_rmse,
	"ape" : get_ape}

#============================= even more param stuff =============================
model_type = "XGBoost_Regressor"
model_tag = str(model_type)
baseline_tag = "Persistence_Baseline"

main_output_dir = "%s-DATES-%s-DEBUG-%s-INFO_ID_DEBUG-%s-DAY_AGG-%s-agg_timestep_amount-%d/%s/"%(main_output_tag,model_type,DEBUG, INFO_ID_DEBUG,DAY_AGG , agg_timestep_amount,TARGET_PLATFORM)
bu.create_output_dir(main_output_dir)

date_dir = "train-%s-to-%s-val-%s-to-%s-test-%s-to-%s/"%(train_start,train_end,val_start,val_end,test_start,test_end)
main_output_dir = main_output_dir + date_dir
bu.create_output_dir(main_output_dir)

combo_dir = main_output_dir + "Combo-Counts/"
bu.create_output_dir(combo_dir)

#get gran
GRAN="H"
test_dates = pd.date_range(test_start,test_end, freq="D")
val_dates = pd.date_range(val_start,val_end, freq="D")
# main_input_dir = "/data/Fmubang/cp4-VAM-write-up-exps/p38-m1-m2-fused/V4-CP4-FIXED-COLL-DATA-QUOTE-REPLY-KICKED-OUT-Reorganized-User-Activity-Platform-Features/"
# main_input_dir = "/data/Fmubang/cp4-VAM-write-up-exps/p38-m1-m2-fused/V5-NEWS-DATA-CP4-FIXED-COLL-DATA-QUOTE-REPLY-KICKED-OUT-Reorganized-User-Activity-Platform-Features/"

#more ft things
infoIDs = get_cp4_challenge_18_infoIDs()
platforms = ["twitter", "youtube"]
user_statuses = ["new", "old"]

if (DEBUG == True) and (INFO_ID_DEBUG==True):
	infoIDs = infoIDs[:1]

#n jobs
n_jobs=25

#============================= model params =============================
overwrite_params_dict = {}
# overwrite_params_dict["DEBUG"]=DEBUG
overwrite_params_dict["GET_NEWS_DATA"] = GET_NEWS_DATA
overwrite_params_dict["train_start"]=train_start
overwrite_params_dict["train_end"]=train_end
overwrite_params_dict["val_start"]=val_start
overwrite_params_dict["val_end"]=val_end
overwrite_params_dict["test_start"]=test_start
overwrite_params_dict["test_end"]=test_end
overwrite_params_dict["n_jobs"]=n_jobs
overwrite_params_dict["DARPA_INPUT_FORMAT"]=DARPA_INPUT_FORMAT
overwrite_params_dict["AGG_TIME_SERIES"]=False
overwrite_params_dict["main_input_dir"] = main_input_dir
overwrite_params_dict["platforms"] = platforms
overwrite_params_dict["infoIDs"] = infoIDs
overwrite_params_dict["user_statuses"] = user_statuses
overwrite_params_dict["main_output_dir"] = main_output_dir
overwrite_params_dict["LOGNORM_DEBUG_PRINT"] = LOGNORM_DEBUG_PRINT
overwrite_params_dict["CORRELATION_STR_TO_FUNC_DICT"] = CORRELATION_STR_TO_FUNC_DICT

idx_to_data_and_model_materials_dict = mp.get_selected_models_param_dict_v2(all_models_df,selected_models,overwrite_params_dict)
total_combos = len(idx_to_data_and_model_materials_dict.keys())
print("\ntotal_combos: %d"%total_combos)

cur_total_combo=0
for cur_dict_idx in range(total_combos):
	print("\nDict %d of %d"%(cur_dict_idx+1, total_combos))
	cur_tuple = idx_to_data_and_model_materials_dict[cur_dict_idx]
	sample_gen_param_dict = cur_tuple[0]
	model_param_dict = cur_tuple[1]
	specific_model_tag = cur_tuple[2]

	
	temp_dict = dict(sample_gen_param_dict)
	temp_dict["DARPA_INPUT_FORMAT"] = False

	if DARPA_INPUT_FORMAT == True:
		print("\nGetting pnnl baseline data...")
		y_pred_baseline_test_data_dict,y_pred_baseline_val_data_dict,y_pred_baseline_test_sliding_window_data_dict,y_pred_baseline_val_sliding_window_data_dict = base_u.get_pnnl_baseline_data(temp_dict,SampleGenerator)

	else:
		print("\nGetting regular baseline data...")
		y_pred_baseline_test_data_dict,y_pred_baseline_val_data_dict,y_pred_baseline_test_sliding_window_data_dict,y_pred_baseline_val_sliding_window_data_dict = base_u.get_baseline_data(temp_dict,SampleGenerator)

	print("\nGetting sample_generator...")
	sample_generator = SampleGenerator(**sample_gen_param_dict)
	print("\nsample_generator")
	print(sample_generator)

	#==================== basic data ====================
	#get array dict

	# generate_platform_infoID_pair_samples_v2_1_df(self)
	infoID_train_and_test_array_dict = sample_generator.configure_features_and_get_infoID_train_and_test_array_dict_v2_fixed_dates()
	# sys.exit(0)
	# infoID_train_and_test_array_dict = sample_generator.configure_features_and_get_infoID_train_and_test_array_dict_v2_fixed_dates()

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

	#now go back and get rid of platform fts
	for infoID in infoIDs:
		print()
		print(infoID)
		print(y_pred_baseline_val_data_dict[infoID])

	#normalize data
	x_train,y_train,x_scaler,y_scaler = sample_generator.normalize_train_data_and_return_scalers()
	x_val,y_val=sample_generator.normalize_data(x_val,y_val)
	x_test,y_test=sample_generator.normalize_data(x_test,y_test)
	x_val_sliding_window,y_val_sliding_window=sample_generator.normalize_data(x_val_sliding_window,y_val_sliding_window)
	x_test_sliding_window,y_test_sliding_window=sample_generator.normalize_data(x_test_sliding_window,y_test_sliding_window)

	# shapes = []
	# array_labels = [x_train.shape, x_val.shape]


	#make out dir
	data_proc_param_output_tag = sample_generator.param_tag
	# data_proc_output_dir = main_output_dir + data_proc_param_output_tag + "/"
	# create_output_dir(data_proc_output_dir)

	data_proc_param_df = sample_generator.ft_preproc_param_df
	print(data_proc_param_df)

	target_fts = sample_generator.target_fts
	print("\ntarget_fts")
	print(target_fts)

	target_ft_categories = sample_generator.target_ft_categories
	print("\ntarget_ft_categories")
	print(target_ft_categories)

	print("\nGetting model")
	xgboost_model_handler = ModelHandler(model_param_dict, model_type,model_type_to_param_name_order_dict)
	model = xgboost_model_handler.initialize_model()
	model_param_df,model_param_tag = xgboost_model_handler.get_model_param_tag()

	# model_exp_output_dir = data_proc_output_dir + model_param_tag + "/"
	# create_output_dir(model_exp_output_dir) 

	model_exp_output_dir = main_output_dir + specific_model_tag + "/"
	bu.create_output_dir(model_exp_output_dir)
	#get cur full param df
	cur_full_param_df = pd.concat([data_proc_param_df,model_param_df]).reset_index(drop=True)
	print("\ncur_full_param_df")
	print(cur_full_param_df)

	param_fp = model_exp_output_dir + "full-params.csv"
	cur_full_param_df.to_csv(param_fp)
	print(param_fp)

	array_shape_fp = model_exp_output_dir + "sample-shape-info.csv"

	array_shape_df = sample_generator.get_array_shape_df(["train", "val_sliding_window", "test"])
	print(array_shape_df)
	array_shape_df.to_csv(array_shape_fp, index=False)
	print(array_shape_fp)

	# sys.exit(0)

	#save metadata
	if SAVE_SCALERS_AND_METADATA==True:
		sample_generator.save_scalers_and_metadata( model_exp_output_dir)

	sample_generator.save_ft_lists(model_exp_output_dir)

	if VALIDATE_ON_SLIDING_WINDOW_DATA == True:
		validation_data=[( x_val_sliding_window,y_val_sliding_window)]
	else:
		validation_data=[( x_val,y_val)]

	print("\nx and y train before train")
	print(x_train.shape)
	print(y_train.shape)

	model.fit(X=x_train, y = y_train)

	if SAVE_MODEL == True:
		print("\nSaving model...")
		model_output_fp  = model_exp_output_dir+ model_type + "-model"
		joblib.dump(model, model_output_fp)

		print("\nLoading model to make sure it was saved right...")
		model = joblib.load(model_output_fp)

	#ft importance
	input_static_fts = sample_generator.static_fts
	input_dynamic_and_static_fts = sample_generator.input_dynamic_and_static_fts

	print("\ninput_static_fts")
	for ft in input_static_fts:
		print(ft)

	print("\ninput_dynamic_and_static_fts")
	for ft in input_dynamic_and_static_fts:
		print(ft)

	for m_idx, cur_model in enumerate(model.estimators_):
		cur_target_model_ft = target_fts[m_idx]
		save_ft_info_v2_simpler(model_exp_output_dir, cur_model,input_static_fts,input_dynamic_and_static_fts, tag="Model-%d-%s"%(m_idx+1, cur_target_model_ft))

	#test everyone
	y_val_pred_dict = sample_generator.pred_with_model(model, "val", num_timesteps_to_get="all")
	y_test_pred_dict = sample_generator.pred_with_model(model, "test", num_timesteps_to_get="all")
	# y_val_sliding_window_pred_dict = sample_generator.pred_with_model(model, "val_sliding_window", num_timesteps_to_get="all")
	# y_test_sliding_window_pred_dict = sample_generator.pred_with_model(model, "test_sliding_window", num_timesteps_to_get="all")

	groupby_cols = ["infoID", "platform", "target_ft"]

	for DAY_AGG in DAY_AGG_LIST:
		#comp stuff

		cur_model_exp_output_dir = model_exp_output_dir + "DAY_AGG-%s/"%DAY_AGG
		bu.create_output_dir(cur_model_exp_output_dir)

		baseline_comp_output_dir = cur_model_exp_output_dir + "Baseline-Comparisons/"
		merge_cols = ["infoID", "target_ft","y_tag"]

		for infoID in infoIDs:
			print()
			print(infoID)
			print(y_pred_baseline_val_data_dict[infoID])

		#================================================ val eval ================================================
		try:
			array_type_to_correlation_materials_dict = sample_generator.array_type_to_correlation_materials_dict
		except:
			array_type_to_correlation_materials_dict=None

		y_val_pred_df = mef.create_prediction_df_with_corr_data_v2_hourly_option(sample_generator.CORRELATION_FUNCS_TO_USE_LIST ,array_type_to_correlation_materials_dict,platforms,y_val_pred_dict, sample_generator.infoID_train_and_test_array_dict,infoIDs,target_ft_categories,"val",model_type, cur_model_exp_output_dir,SAVE_PREDS=SAVE_PREDS)
		y_pred_baseline_val_df = mef.create_prediction_df_with_corr_data_v2_hourly_option(sample_generator.CORRELATION_FUNCS_TO_USE_LIST,array_type_to_correlation_materials_dict,platforms,y_pred_baseline_val_data_dict ,sample_generator.infoID_train_and_test_array_dict,infoIDs,target_ft_categories,"val","Persistence_Baseline", cur_model_exp_output_dir,SAVE_PREDS=SAVE_PREDS)

		if DAY_AGG==True:
			print("\ny_val_pred_df before day agg")
			print(y_val_pred_df)
			y_val_pred_df = mef.agg_timesteps_to_higher_gran(agg_timestep_amount ,y_val_pred_df, "pred","actual",groupby_cols,"timestep" )
			print("\ny_val_pred_df after day agg")
			print(y_val_pred_df)

			print("\ny_pred_baseline_val_df before day agg")
			print(y_pred_baseline_val_df)
			y_pred_baseline_val_df = mef.agg_timesteps_to_higher_gran(agg_timestep_amount ,y_pred_baseline_val_df, "pred","actual",groupby_cols,"timestep" )
			print("\ny_pred_baseline_val_df after day agg")
			print(y_pred_baseline_val_df)


		print("\ntarget_ft_categories")
		print(target_ft_categories)

		mef.graph_pred_vs_gt_options_v2_from_df(y_pred_baseline_val_df, cur_model_exp_output_dir + "Graphs/Persistence_Baseline/")
		mef.graph_pred_vs_gt_options_v2_from_df(y_val_pred_df, cur_model_exp_output_dir + "Graphs/"+ model_type + "/")


		#insert metrics
		y_val_pred_metric_df = mef.insert_metrics_into_pred_df_v2_TEMPORAL(y_val_pred_df, infoIDs, platforms, cur_model_exp_output_dir + "Metrics/",target_ft_categories,["rmse"],metric_str_to_function_dict)
		y_pred_baseline_val_metric_df = mef.insert_metrics_into_pred_df_v2_TEMPORAL(y_pred_baseline_val_df, infoIDs, platforms, cur_model_exp_output_dir + "Metrics/",target_ft_categories,["rmse"],metric_str_to_function_dict)

		rename_cols = ["pred"] + list(metrics_to_get)
		comp_merge_cols = ["timestep", "actual", "platform", "infoID", "target_ft", "y_tag"]
		proper_baseline_dir =  cur_model_exp_output_dir + "Proper-Baseline-Comparisons/"
		bu.create_output_dir(proper_baseline_dir)

		y_val_comp_df = mef.create_proper_baseline_comparison_df(y_pred_baseline_val_metric_df, y_val_pred_metric_df, rename_cols, comp_merge_cols,proper_baseline_dir,"y_val", metric_tag="rmse")
		graph_output_dir = cur_model_exp_output_dir + "Proper-Baseline-Comparison-Graphs/y_val/"
		mef.graph_pred_vs_gt_with_baseline_from_comp_df_v2_simple(y_val_comp_df, model_tag, baseline_tag, "rmse",platform, graph_output_dir)

		#get comparisons
		val_infoID_comp_df, val_target_ft_comp_df, val_overall_summary_df = mef.get_summary_info_of_baseline_comp_df(y_val_comp_df, proper_baseline_dir, "y_val", "rmse", model_tag, baseline_tag)
		
		#================================================ TEST eval ================================================
		y_test_pred_df = mef.create_prediction_df_with_corr_data_v2_hourly_option(sample_generator.CORRELATION_FUNCS_TO_USE_LIST,array_type_to_correlation_materials_dict,platforms,y_test_pred_dict, sample_generator.infoID_train_and_test_array_dict,infoIDs,target_ft_categories,"test",model_type, cur_model_exp_output_dir,SAVE_PREDS=SAVE_PREDS)
		y_pred_baseline_test_df = mef.create_prediction_df_with_corr_data_v2_hourly_option(sample_generator.CORRELATION_FUNCS_TO_USE_LIST,array_type_to_correlation_materials_dict,platforms,y_pred_baseline_test_data_dict ,sample_generator.infoID_train_and_test_array_dict,infoIDs,target_ft_categories,"test","Persistence_Baseline", cur_model_exp_output_dir,SAVE_PREDS=SAVE_PREDS)

		if DAY_AGG==True:
			print("\ny_test_pred_df before day agg")
			print(y_test_pred_df)
			y_test_pred_df = mef.agg_timesteps_to_higher_gran(agg_timestep_amount ,y_test_pred_df, "pred","actual",groupby_cols,"timestep" )
			print("\ny_test_pred_df after day agg")
			print(y_test_pred_df)

			print("\ny_pred_baseline_test_df before day agg")
			print(y_pred_baseline_test_df)
			y_pred_baseline_test_df = mef.agg_timesteps_to_higher_gran(agg_timestep_amount ,y_pred_baseline_test_df, "pred","actual",groupby_cols,"timestep" )
			print("\ny_pred_baseline_test_df after day agg")
			print(y_pred_baseline_test_df)



		# sys.exit(0)
		print("\ntarget_ft_categories")
		print(target_ft_categories)

		mef.graph_pred_vs_gt_options_v2_from_df(y_pred_baseline_test_df, cur_model_exp_output_dir + "Graphs/Persistence_Baseline/")
		mef.graph_pred_vs_gt_options_v2_from_df(y_test_pred_df, cur_model_exp_output_dir + "Graphs/"+ model_type + "/")

		#insert metrics
		y_test_pred_metric_df = mef.insert_metrics_into_pred_df_v2_TEMPORAL(y_test_pred_df, infoIDs, platforms, cur_model_exp_output_dir + "Metrics/",target_ft_categories,["rmse"],metric_str_to_function_dict)
		y_pred_baseline_test_metric_df = mef.insert_metrics_into_pred_df_v2_TEMPORAL(y_pred_baseline_test_df, infoIDs, platforms, cur_model_exp_output_dir + "Metrics/",target_ft_categories,["rmse"],metric_str_to_function_dict)

		y_test_comp_df = mef.create_proper_baseline_comparison_df(y_pred_baseline_test_metric_df, y_test_pred_metric_df, rename_cols, comp_merge_cols,proper_baseline_dir,"y_test", metric_tag="rmse")
		graph_output_dir = cur_model_exp_output_dir + "Proper-Baseline-Comparison-Graphs/y_test/"
		mef.graph_pred_vs_gt_with_baseline_from_comp_df_v2_simple(y_test_comp_df, model_tag, baseline_tag, "rmse",platform, graph_output_dir)

		#get comparisons
		test_infoID_comp_df, test_target_ft_comp_df, test_overall_summary_df = mef.get_summary_info_of_baseline_comp_df(y_test_comp_df, proper_baseline_dir, "y_test", "rmse", model_tag, baseline_tag)

	cur_total_combo+=1
	combo_fp = combo_dir + "%d-of-%d-done.txt"%(cur_total_combo, total_combos)
	with open(combo_fp, "w") as f:
		print("\ncur_total_combo: %d of %d"%( cur_total_combo, total_combos))


