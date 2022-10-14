import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import pandas as pd 
import numpy as np 
import os,sys


model_type_to_model_func_dict={
    "XGBoost_Regressor":xgb.XGBRegressor,
    "KNN_Regressor":KNeighborsRegressor,
    "SVR_RBF":SVR
}

xgboost_param_name_order = [
"objective",
"colsample_bytree",
"learning_rate",
"max_depth",
"n_estimators",
"subsample",
"gamma",
"alpha",
"reg_lambda",
"random_state",
"n_jobs"]

knn_param_name_order = ["weights", "algorithm", "n_neighbors", "n_jobs"]

svr_rbf_name_order = ["kernel", "C", "epsilon", "gamma"]

model_type_to_param_name_order_dict={"XGBoost_Regressor":xgboost_param_name_order, "KNN_Regressor":knn_param_name_order, "SVR_RBF":svr_rbf_name_order}

def make_param_tag(PARAM_VALS):

    PARAM_TAG = ""
    for p in PARAM_VALS:
        PARAM_TAG = PARAM_TAG + "-" + str(p)

    PARAM_TAG=PARAM_TAG[1:]


    return PARAM_TAG

class ModelHandler:

    def __init__(self, param_dict, model_type,model_type_to_param_name_order_dict):
        self.param_dict=param_dict
        self.model_type=model_type
        self.model_type_to_model_func_dict=model_type_to_model_func_dict
        self.model_type_to_param_name_order_dict=model_type_to_param_name_order_dict

        self.indiv_model_jobs = 1
        self.wrapper_model_n_jobs = int(param_dict["n_jobs"])
        self.param_dict["n_jobs"] = self.indiv_model_jobs


    def initialize_model(self):
        model_func = self.model_type_to_model_func_dict[self.model_type]
        print(model_func)

        if (self.model_type == "XGBoost_Regressor") or ("SVR" in self.model_type):
            self.model = MultiOutputRegressor(model_func(**self.param_dict), n_jobs=self.wrapper_model_n_jobs)
        else:
            self.model = model_func(**self.param_dict)
        print(self.model)
        return self.model

    def get_model_param_tag(self):

        param_name_order = model_type_to_param_name_order_dict[self.model_type]
        param_vals = []
        for param_name in param_name_order:
            param_val = self.param_dict[param_name]
            param_vals.append(param_val)
        self.param_tag = make_param_tag(param_vals)
        print(self.param_tag)
        self.model_param_df = pd.DataFrame(data={"param":param_name_order, "value":param_vals})
        print(self.model_param_df)
        return self.model_param_df,self.param_tag