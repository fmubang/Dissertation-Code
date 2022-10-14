import pandas as pd
import numpy as np
import sys
sys.path.append("/data/Fmubang/CP5-VAM-Paper-Stuff-3-3/functions")
from statsmodels.tsa.arima.model import ARIMA
from basic_utils import *
import basic_utils as bu
from sklearn.metrics import mean_squared_error

def proper_scikit_rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)

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