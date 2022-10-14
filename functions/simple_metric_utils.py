import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dtw

from datetime import datetime, timedelta, date
from scipy.stats import iqr

import tsfresh.feature_extraction.feature_calculators as tsf
def skewness_error(v1, v2):

    sk1 = tsf.skewness(v1)
    sk2 = tsf.skewness(v2)
    return abs(sk1 - sk2)

def normed_skewness_error(v1, v2):

    v1 = normalize_vector(v1)
    v2 = normalize_vector(v2)
    return skewness_error(v1, v2)

'''
Metrics
'''

def leidos_nrmse(ground_truth, simulation,  fill_value=0,
    relative=False, cumulative=True, normed=True):
    """
    Metric: rmse
    Description: Root mean squared error
    Inputs:
    ground_truth - ground truth measurement (data frame) with measurement in
        the "value" column
    simulation - simulation measurement (data frame) with measurement in the
        "value" column
    join - type of join to perform between ground truth and simulation
    fill_value - fill value for non-overlapping joins
    """

    # if type(ground_truth) is np.ndarray:
    #     result = ground_truth - simulation
    #     result = (result ** 2).mean()
    #     result = np.sqrt(result)
    #     return result

    # if type(ground_truth) is list:

    #     ground_truth = np.nan_to_num(ground_truth)
    #     simulation   = np.nan_to_num(simulation)

    #     result = np.asarray(ground_truth) - np.asarray(simulation)
    #     result = (result ** 2).mean()
    #     result = np.sqrt(result)

    #     return result

    # df = self.join_dfs(ground_truth, simulation, join=join,
    #     fill_value=fill_value)

    df = pd.DataFrame(data={"value_gt":ground_truth ,"value_sim":simulation})


    if len(df.index) > 0:

        if cumulative:

            if df['value_sim'].sum() == 0:
                df['value_sim']=0
            else:
                df['value_sim'] = df['value_sim'].cumsum()

            #df['value_gt'] = df['value_gt'].cumsum()
            if df['value_gt'].sum() == 0:
                df['value_gt']=0
            else:
                df['value_gt'] = df['value_gt'].cumsum()

        if normed:
            if df['value_gt'].min() > 0:
                epsilon = 0.001*df[df['value_gt'] != 0.0]['value_gt'].min()
            elif df['value_sim'].min() > 0:
                epsilon = 0.001*df[df['value_sim'] != 0.0]['value_sim'].min()
            else:
                epsilon = 1.0

            df['value_sim'] = (df['value_sim'] + epsilon)/(df['value_sim'].max() + epsilon)
            df['value_gt'] = (df['value_gt'] + epsilon)/(df['value_gt'].max() + epsilon)

        if not relative:
            return np.sqrt(((df["value_sim"]-df["value_gt"])**2).mean())
        else:
            iq_range = float(iqr(df['value_gt'].values))

            result = df["value_sim"]-df["value_gt"]
            result = (result ** 2).mean()
            result = np.sqrt(result)

            if iq_range > 0:
                result = result / iq_range
            else:
                mean_value = df['value_gt'].mean()
                if mean_value > 0:
                    result = result / mean_value
                else:
                    return None

            return result
    else:
        return None


def sAPE_100(F, A):

    F_sum = np.sum(np.asarray(F))
    A_sum = np.sum(np.asarray(A))

    return 100.0* np.abs(F_sum - A_sum)/(abs(F_sum) + abs(A_sum)+np.finfo(float).eps)


def myerrsq(x,y):
    return((x-y)**2)

### s2 predictions, s1 ground truth
def dtw_(s1, s2):
    window=2

    s1= pd.DataFrame( data={ 0: s1})
    s2= pd.DataFrame( data={ 0: s2})

    z1=(s1-s1.mean())/(s1.std(ddof=0).apply(lambda m: (m if m > 0.0 else 1.0)))
    z2=(s2-s2.mean())/(s2.std(ddof=0).apply(lambda m: (m if m > 0.0 else 1.0)))

    ### first value simulation second value GT
    dtw_metric = np.sqrt(dtw.dtw(z2[0], z1[0], dist_method=myerrsq, window_type='slantedband',
                               window_args={'window_size':window}).normalizedDistance)

    return dtw_metric

def ae(v1,v2):
    v1=np.array(v1)
    v2 = np.array(v2)
    v1=np.sum(v1)
    v2=np.sum(v2)
    return np.abs(v1 - v2)

# Scale-Free Absolute Error
def sfae(v1,v2):

    v1=np.array(v1)
    v2 = np.array(v2)

    return ae(v1, v2) / np.mean(v1)

def MAD_mean_ratio(v1, v2):
    """
    MAD/mean ratio
    """
    return np.mean(sfae(v1, v2))

def normed_rmse(v1,v2):
    v1=np.cumsum(v1)
    v2=np.cumsum(v2)
    v1=v1/(np.max(v1) +np.finfo(float).eps)
    v2=v2/(np.max(v2) + np.finfo(float).eps)

    result = v1-v2
    result = (result ** 2).mean()
    result = np.sqrt(result)
    return result

def cumsum_rmse(v1,v2):
    v1=np.cumsum(v1)
    v2=np.cumsum(v2)
    # v1=v1/(np.max(v1) +np.finfo(float).eps)
    # v2=v2/(np.max(v2) + np.finfo(float).eps)

    result = v1-v2
    result = (result ** 2).mean()
    result = np.sqrt(result)
    return result

def rmse(v1,v2):
    result = np.array(v1)-np.array(v2)
    result = (result ** 2).mean()
    result = np.sqrt(result)
    return result

def ape(v1,v2):
    v1=np.sum(v1)
    v2=np.sum(v2)
    result = np.abs(float(v1) - float(v2))
    result = 100.0 * result /( np.abs(float(v1)) + np.finfo(float).eps)
    return result

def smape(A, F):
    A=np.array(A)
    F=np.array(F)
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)) + np.finfo(float).eps)

def smape100(A, F):
    A=np.array(A)
    F=np.array(F)
    return 100.0/len(A) * np.sum(np.abs(F - A) / (np.abs(A) + np.abs(F)) + np.finfo(float).eps)

def normalize_vector(vector):
    vector = np.asarray(vector)
    if np.sum(abs(vector)) == 0:
        return vector

    max_val = vector.max()
    min_val = vector.min()
    if max_val == min_val:
        return np.asarray([0 for i in range(vector.shape[0])])
    return (vector - min_val)/( max_val- min_val)

def normalized_standard_deviation_difference(y_test, y_pred):

    y_test = normalize_vector(y_test)
    y_pred = normalize_vector(y_pred)

    if np.sum(abs(y_test)) == 0:
        y_test_std = 0
    else:
        y_test_std = np.std(y_test)

    if np.sum(abs(y_pred)) == 0:
        y_pred_std = 0
    else:
        y_pred_std = np.std(y_pred)

    return np.abs(y_test_std - y_pred_std)

def standard_deviation_difference(y_test, y_pred):

    if np.sum(abs(y_test)) == 0:
        y_test_std = 0
    else:
        y_test_std = np.std(y_test)

    if np.sum(abs(y_pred)) == 0:
        y_pred_std = 0
    else:
        y_pred_std = np.std(y_pred)

    return np.abs(y_test_std - y_pred_std)


def volatility_error(y_test, y_pred):

    if np.sum(abs(y_test)) == 0:
        y_test_std = 0
    else:
        y_test_std = np.std(y_test)

    if np.sum(abs(y_pred)) == 0:
        y_pred_std = 0
    else:
        y_pred_std = np.std(y_pred)

    return np.abs(y_test_std - y_pred_std)