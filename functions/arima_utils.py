import pandas as pd
import numpy as np
import sys
sys.path.append("/data/Fmubang/CP5-VAM-Paper-Stuff-3-3/functions")
from statsmodels.tsa.arima.model import ARIMA
from basic_utils import *
import basic_utils as bu
from sklearn.metrics import mean_squared_error
from time import time



def proper_scikit_rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)

def train_arima_on_combo_v4_KF(platform, infoID, EVAL_TYPE, CUR_SIM_PERIOD, p, q, d, cur_train_start, cur_train_end, cur_eval_start, cur_eval_end,
     NUM_SIM_PERIODS, train_input_fp, eval_input_fp, combo_num, num_combos, main_output_dir, cp, GRAN_WORD,overall_input_dir=""):

    print("\nWorking on combo %d of %d"%(combo_num , num_combos))

    start_time = time()

    #get output dir
    hyp_infoID = infoID.replace("/", "_")
    #cur_output_dir = main_output_dir + platform + "/"+ hyp_infoID +  "/" + EVAL_TYPE + "/%s/SIM-PERIOD-%d-of-%d/p-%d-d-%d-q-%d/"%(output_type , CUR_SIM_PERIOD,NUM_SIM_PERIODS ,p,d , q)
    cur_output_dir = main_output_dir + platform + "/"+ cp + "/" + hyp_infoID +  "/" + EVAL_TYPE + "/%s/SIM-PERIOD-%d-of-%d/p-%d-d-%d-q-%d/"%(GRAN_WORD, CUR_SIM_PERIOD,NUM_SIM_PERIODS ,p,d , q)
    bu.create_output_dir(cur_output_dir)

    #check if file exists
    time_fp = cur_output_dir + "total-time-in-sec.txt"
    # rmse_output_fp = cur_output_dir + "RMSE.txt"

    if os.path.exists(time_fp):
        print("%s already exists, Done with on combo %d of %d"%(time_fp ,combo_num , num_combos))
        return

    #open the train df
    cur_train_df = pd.read_csv(overall_input_dir + train_input_fp)

    #eval df
    cur_eval_df = pd.read_csv(overall_input_dir + eval_input_fp)
    OUTPUT_SIZE = cur_eval_df.shape[0]

    print("\ntrain and eval")
    print(cur_train_df)
    print(cur_eval_df)

    #get ts
    train_ts = cur_train_df[infoID].values
    eval_ts = cur_eval_df[infoID].values

    print("\nTraining model...")

    try:
        model = ARIMA(train_ts, order=(p, d, q)).fit()
        forecast = list(model.forecast(steps=OUTPUT_SIZE))
    except:
        forecast = [0 for i in range(OUTPUT_SIZE)]
        output_fp = cur_output_dir + "error.txt"
        with open(output_fp, "w") as f:
            print(output_fp)

    forecast = np.round(forecast, 0)
    forecast = np.asarray(forecast)
    forecast[forecast<0] = 0

    print("\n%s to %s forecast"%(cur_eval_start, cur_eval_end))
    print(forecast)

    infoID_df_list =[infoID for i in range(len(forecast))]
    eval_type_df_list =[EVAL_TYPE for i in range(len(forecast))]
    sim_period_df_list = [CUR_SIM_PERIOD for i in range(len(forecast))]
    timestep_df_list = [i+1 for i in range(len(forecast))]
    # output_type_df_list = [output_type for i in range(len(forecast))]
    cp_type_df_list = [cp for i in range(len(forecast))]
    gran_word_df_list = [GRAN_WORD for i in range(len(forecast))]
    p_df_list = [p for i in range(len(forecast))]
    d_df_list = [d for i in range(len(forecast))]
    q_df_list = [q for i in range(len(forecast))]
    forecast_df_list =list(forecast)

    cur_eval_start_df_list = [cur_eval_start for i in range(len(forecast))]
    cur_eval_end_df_list = [cur_eval_end for i in range(len(forecast))]

    data={
    "infoID":infoID_df_list, "eval_type":eval_type_df_list,
    "sim_period" : sim_period_df_list, "timestep":timestep_df_list,
     "p":p_df_list,
    "d":d_df_list, "q":q_df_list, "forecast":forecast_df_list,
    "cur_eval_start" : cur_eval_start_df_list, "cur_eval_end" : cur_eval_end_df_list, "cp": cp_type_df_list, "GRAN_WORD":gran_word_df_list
    }

    col_order = ["infoID", "eval_type", "sim_period", "timestep", "p", "d", "q", "cur_eval_start","cur_eval_end", "cp","GRAN_WORD", "forecast", "actual", ]

    RMSE = proper_scikit_rmse(eval_ts, forecast)

    cur_df = pd.DataFrame(data=data)
    cur_df["actual"] = eval_ts
    cur_df=cur_df[col_order]
    print()
    print(cur_df)

    output_fp = cur_output_dir + "forecasts.csv"
    cur_df.to_csv(output_fp, index=False)

    rmse_output_fp = cur_output_dir + "RMSE.txt"
    with open(rmse_output_fp, "w") as f:
        f.write(str(RMSE))

    end_time = time()

    total_time = end_time - start_time
    total_time = np.round(total_time, 2)

    with open(time_fp, "w") as f:
        f.write(str(total_time))

    print("\nDone with on combo %d of %d. Total time in sec: %s"%(combo_num , num_combos, str(total_time)))

    return




def train_arima_on_combo_v3(platform, infoID, EVAL_TYPE, CUR_SIM_PERIOD, p, q, d, cur_train_start, cur_train_end, cur_eval_start, cur_eval_end,
    output_type, NUM_SIM_PERIODS, train_input_fp, eval_input_fp, combo_num, num_combos, main_output_dir):

    print("\nWorking on combo %d of %d"%(combo_num , num_combos))

    start_time = time()

    #get output dir
    hyp_infoID = infoID.replace("/", "_")
    cur_output_dir = main_output_dir + platform + "/"+ hyp_infoID +  "/" + EVAL_TYPE + "/%s/SIM-PERIOD-%d-of-%d/p-%d-d-%d-q-%d/"%(output_type , CUR_SIM_PERIOD,NUM_SIM_PERIODS ,p,d , q)
    bu.create_output_dir(cur_output_dir)

    #check if file exists
    time_fp = cur_output_dir + "total-time-in-sec.txt"
    # rmse_output_fp = cur_output_dir + "RMSE.txt"

    if os.path.exists(time_fp):
        print("%s already exists, Done with on combo %d of %d"%(time_fp ,combo_num , num_combos))
        return

    #open the train df
    cur_train_df = pd.read_csv(train_input_fp)

    #eval df
    cur_eval_df = pd.read_csv(eval_input_fp)
    OUTPUT_SIZE = cur_eval_df.shape[0]

    print("\ntrain and eval")
    print(cur_train_df)
    print(cur_eval_df)

    #get ts
    train_ts = cur_train_df[output_type].values
    eval_ts = cur_eval_df[output_type].values

    print("\nTraining model...")

    try:
        model = ARIMA(train_ts, order=(p, d, q)).fit()
        forecast = list(model.forecast(steps=OUTPUT_SIZE))
    except:
        forecast = [0 for i in range(OUTPUT_SIZE)]
        output_fp = cur_output_dir + "error.txt"
        with open(output_fp, "w") as f:
            print(output_fp)

    forecast = np.round(forecast, 0)
    forecast = np.asarray(forecast)
    forecast[forecast<0] = 0

    print("\n%s to %s forecast"%(cur_eval_start, cur_eval_end))
    print(forecast)

    infoID_df_list =[infoID for i in range(len(forecast))]
    eval_type_df_list =[EVAL_TYPE for i in range(len(forecast))]
    sim_period_df_list = [CUR_SIM_PERIOD for i in range(len(forecast))]
    timestep_df_list = [i+1 for i in range(len(forecast))]
    output_type_df_list = [output_type for i in range(len(forecast))]
    p_df_list = [p for i in range(len(forecast))]
    d_df_list = [d for i in range(len(forecast))]
    q_df_list = [q for i in range(len(forecast))]
    forecast_df_list =list(forecast)

    cur_eval_start_df_list = [cur_eval_start for i in range(len(forecast))]
    cur_eval_end_df_list = [cur_eval_end for i in range(len(forecast))]

    data={
    "infoID":infoID_df_list, "eval_type":eval_type_df_list,
    "sim_period" : sim_period_df_list, "timestep":timestep_df_list,
    "output_type":output_type_df_list, "p":p_df_list,
    "d":d_df_list, "q":q_df_list, "forecast":forecast_df_list,
    "cur_eval_start" : cur_eval_start_df_list, "cur_eval_end" : cur_eval_end_df_list
    }

    col_order = ["infoID", "eval_type", "sim_period", "timestep", "output_type", "p", "d", "q", "cur_eval_start","cur_eval_end","forecast", "actual"]

    RMSE = proper_scikit_rmse(eval_ts, forecast)

    cur_df = pd.DataFrame(data=data)
    cur_df["actual"] = eval_ts
    cur_df=cur_df[col_order]
    print()
    print(cur_df)

    output_fp = cur_output_dir + "forecasts.csv"
    cur_df.to_csv(output_fp, index=False)

    rmse_output_fp = cur_output_dir + "RMSE.txt"
    with open(rmse_output_fp, "w") as f:
        f.write(str(RMSE))

    end_time = time()

    total_time = end_time - start_time
    total_time = np.round(total_time, 2)

    with open(time_fp, "w") as f:
        f.write(str(total_time))

    print("\nDone with on combo %d of %d. Total time in sec: %s"%(combo_num , num_combos, str(total_time)))

    return

def train_arima_on_combo_v2(platform, main_output_dir, arima_param_dict, PARAM_IDX, NUM_PARAM_COMBOS, CUR_SIM_PERIOD,NUM_SIM_PERIODS, infoID, EVAL_TYPE, cur_train_df, cur_eval_df,  OUTPUT_SIZE, cur_eval_start, cur_eval_end):

    print("\nOn arima combo %d of %d"%(PARAM_IDX+1, NUM_PARAM_COMBOS))

    p = arima_param_dict["p"]
    d = arima_param_dict["d"]
    q = arima_param_dict["q"]
    output_type = arima_param_dict["output_type"]

    train_ts = cur_train_df[output_type].values
    eval_ts = cur_eval_df[output_type].values

    arima_params = (p,d,q)

    # p = arima_params[0]
    # d = arima_params[1]
    # q = arima_params[2]

    hyp_infoID = infoID.replace("/", "_")

    cur_output_dir = main_output_dir + platform + "/"+ hyp_infoID +  "/" + EVAL_TYPE + "/%s/SIM-PERIOD-%d-of-%d/p-%d-d-%d-q-%d/"%(output_type , CUR_SIM_PERIOD,NUM_SIM_PERIODS ,p,d , q)
    bu.create_output_dir(cur_output_dir)

    print("\nTraining model...")

    try:
        model = ARIMA(train_ts, order=tuple(arima_params)).fit()
        forecast = list(model.forecast(steps=OUTPUT_SIZE))
    except:
        forecast = [0 for i in range(OUTPUT_SIZE)]
        output_fp = cur_output_dir + "error.txt"
        with open(output_fp, "w") as f:
            print(output_fp)

    forecast = np.round(forecast, 0)
    forecast = np.asarray(forecast)
    forecast[forecast<0] = 0

    print("\n%s to %s forecast"%(cur_eval_start, cur_eval_end))
    print(forecast)

    infoID_df_list =[infoID for i in range(len(forecast))]
    eval_type_df_list =[EVAL_TYPE for i in range(len(forecast))]
    sim_period_df_list = [CUR_SIM_PERIOD for i in range(len(forecast))]
    timestep_df_list = [i+1 for i in range(len(forecast))]
    output_type_df_list = [output_type for i in range(len(forecast))]
    p_df_list = [p for i in range(len(forecast))]
    d_df_list = [d for i in range(len(forecast))]
    q_df_list = [q for i in range(len(forecast))]
    forecast_df_list =list(forecast)

    cur_eval_start_df_list = [cur_eval_start for i in range(len(forecast))]
    cur_eval_end_df_list = [cur_eval_end for i in range(len(forecast))]

    data={
    "infoID":infoID_df_list, "eval_type":eval_type_df_list,
    "sim_period" : sim_period_df_list, "timestep":timestep_df_list,
    "output_type":output_type_df_list, "p":p_df_list,
    "d":d_df_list, "q":q_df_list, "forecast":forecast_df_list,
    "cur_eval_start" : cur_eval_start_df_list, "cur_eval_end" : cur_eval_end_df_list
    }

    col_order = ["infoID", "eval_type", "sim_period", "timestep", "output_type", "p", "d", "q", "cur_eval_start","cur_eval_end","forecast", "actual"]

    RMSE = proper_scikit_rmse(eval_ts, forecast)

    cur_df = pd.DataFrame(data=data)
    cur_df["actual"] = eval_ts
    cur_df=cur_df[col_order]
    print()
    print(cur_df)

    output_fp = cur_output_dir + "forecasts.csv"
    cur_df.to_csv(output_fp, index=False)

    rmse_output_fp = cur_output_dir + "RMSE.txt"
    with open(rmse_output_fp, "w") as f:
        f.write(str(RMSE))

    print("Done with arima combo %d of %d"%(PARAM_IDX+1, NUM_PARAM_COMBOS))

    return cur_df













def train_arima_on_combo(platform, main_output_dir, arima_param_dict, PARAM_IDX, NUM_PARAM_COMBOS, output_type, CUR_SIM_PERIOD,NUM_SIM_PERIODS, infoID, EVAL_TYPE, train_ts, eval_ts, OUTPUT_SIZE, cur_eval_start, cur_eval_end):

    print("\nOn arima combo %d of %d"%(PARAM_IDX+1, NUM_PARAM_COMBOS))

    p = arima_param_dict["p"]
    d = arima_param_dict["d"]
    q = arima_param_dict["q"]

    arima_params = (p,d,q)

    # p = arima_params[0]
    # d = arima_params[1]
    # q = arima_params[2]

    hyp_infoID = infoID.replace("/", "_")

    cur_output_dir = main_output_dir + platform + "/"+ hyp_infoID +  "/" + EVAL_TYPE + "/%s/SIM-PERIOD-%d-of-%d/p-%d-d-%d-q-%d/"%(output_type , CUR_SIM_PERIOD,NUM_SIM_PERIODS ,p,d , q)
    bu.create_output_dir(cur_output_dir)

    print("\nTraining model...")
    model = ARIMA(train_ts, order=tuple(arima_params)).fit()

    forecast = list(model.forecast(steps=OUTPUT_SIZE))

    forecast = np.round(forecast, 0)

    print("\n%s to %s forecast"%(cur_eval_start, cur_eval_end))
    print(forecast)

    infoID_df_list =[infoID for i in range(len(forecast))]
    eval_type_df_list =[EVAL_TYPE for i in range(len(forecast))]
    sim_period_df_list = [CUR_SIM_PERIOD for i in range(len(forecast))]
    timestep_df_list = [i+1 for i in range(len(forecast))]
    output_type_df_list = [output_type for i in range(len(forecast))]
    p_df_list = [p for i in range(len(forecast))]
    d_df_list = [d for i in range(len(forecast))]
    q_df_list = [q for i in range(len(forecast))]
    forecast_df_list =list(forecast)

    cur_eval_start_df_list = [cur_eval_start for i in range(len(forecast))]
    cur_eval_end_df_list = [cur_eval_end for i in range(len(forecast))]

    data={
    "infoID":infoID_df_list, "eval_type":eval_type_df_list,
    "sim_period" : sim_period_df_list, "timestep":timestep_df_list,
    "output_type":output_type_df_list, "p":p_df_list,
    "d":d_df_list, "q":q_df_list, "forecast":forecast_df_list,
    "cur_eval_start" : cur_eval_start_df_list, "cur_eval_end" : cur_eval_end_df_list
    }

    col_order = ["infoID", "eval_type", "sim_period", "timestep", "output_type", "p", "d", "q", "cur_eval_start","cur_eval_end","forecast", "actual"]

    RMSE = proper_scikit_rmse(eval_ts, forecast)

    cur_df = pd.DataFrame(data=data)
    cur_df["actual"] = eval_ts
    cur_df=cur_df[col_order]
    print()
    print(cur_df)

    output_fp = cur_output_dir + "forecasts.csv"
    cur_df.to_csv(output_fp, index=False)

    rmse_output_fp = cur_output_dir + "RMSE.txt"
    with open(rmse_output_fp, "w") as f:
        f.write(str(RMSE))

    print("Done with arima combo %d of %d"%(PARAM_IDX+1, NUM_PARAM_COMBOS))

    return cur_df