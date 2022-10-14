import pandas as pd 
import numpy as np 
import os,sys


def join_dfs(ground_truth, simulation, join='inner', fill_value=0):

    """
    Join the simulation and ground truth data frames
    Inputs:
    ground_truth - Ground truth measurement data frame with measurement in the
        "value" column
    simulation - Simulation measurement data frame with measurement in the
        "value" column
    join - Join method (inner, outer, left, right)
    fill_value - Value for filling NAs or method for filling in NAs
        (e.g. "ffill" for forward fill)
    """

    on = [c for c in ground_truth.columns if c!='value']

    suffixes = ('_gt', '_sim')

    df = ground_truth.merge(simulation, on=on, suffixes=suffixes, how=join)
    df = df.sort_values([c for c in ground_truth.columns if c != 'value'])

    try:
        float(fill_value)
        df = df.fillna(fill_value)
    except ValueError:
        df = df.fillna(method=fill_value)

    return(df)

def pnnl_rmse(ground_truth, simulation, join='outer', fill_value=0,
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

    # df =join_dfs(ground_truth, simulation, join=join,
    #     fill_value=fill_value)

    df = pd.DataFrame(data={"value_sim":simulation, "value_gt":ground_truth})


    if len(df.index) > 0:

        if cumulative:
            df['value_sim'] = df['value_sim'].cumsum()
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

def absolute_percentage_error(ground_truth, simulation):
    """
    Metric: absolute_percentage_error
    Description: Absolute percentage error between ground truth simulation measurement
    Meant for scalar valued measurements
    Input:
    """

    # if ground_truth is None or ground_truth==0 or simulation is None:
    #     result =  None
    # else:
    result = absolute_difference(ground_truth, simulation)
    result = 100.0 * result / np.abs(float(ground_truth))

    return result

def absolute_difference(ground_truth, simulation):
    """
    Metric: absolute_difference
    Description: Absolute difference between ground truth simulation measurement. Meant for scalar valued measurements.
    Input:
    """

    # if not ground_truth is None and not simulation is None:
    #     result = np.abs(float(simulation) - float(ground_truth))
    # else:
    #     result = None

    result = np.abs(float(simulation) - float(ground_truth))

    return result

def get_nc_rmse(ground_truth, simulation):
    return pnnl_rmse(ground_truth, simulation, cumulative=True, normed=True)

def get_regular_rmse(ground_truth, simulation):
    return pnnl_rmse(ground_truth, simulation, cumulative=False, normed=False)

def get_ape(ground_truth, simulation):

    result = np.abs(ground_truth.sum()-simulation.sum())
    result = 100.0 * result / np.abs(float(ground_truth.sum()))

    return result

metric_str_to_function_dict={
    "rmse":get_regular_rmse,
    "nc_rmse":get_nc_rmse,
    "ape" : get_ape

}