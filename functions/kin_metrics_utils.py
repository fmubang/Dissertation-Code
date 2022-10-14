import numpy as np
import dtw
import pandas as pd
from scipy.stats import entropy

import tsfresh.feature_extraction.feature_calculators as tsf

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

def skewness_error(v1, v2):

    sk1 = tsf.skewness(v1)
    sk2 = tsf.skewness(v2)
    return abs(sk1 - sk2)

def normed_skewness_error(v1, v2):

    v1 = normalize_vector(v1)
    v2 = normalize_vector(v2)
    return skewness_error(v1, v2)

def ape(gt,sim):
    gt = np.asarray(gt)
    sim = np.asarray(sim)
    gt=np.sum(gt)
    sim=np.sum(sim)
    result = np.abs(float(gt) - float(sim))
    result = 100.0 * (result / np.abs(float(gt)))
    return result

def normed_rmse(gt,sim):
    gt = np.asarray(gt)
    sim = np.asarray(sim)
    gt=np.cumsum(gt)
    sim=np.cumsum(sim)
    gt=gt/np.max(gt)
    sim=sim/np.max(sim)

    result = gt-sim
    result = (result ** 2).mean()
    result = np.sqrt(result)
    return result

def rmse(gt,sim):
    result = np.array(gt)-np.array(sim)
    result = (result ** 2).mean()
    result = np.sqrt(result)
    return result

def smape(gt, sim):
    gt=np.array(gt)
    sim=np.array(sim)
    num = 100/len(gt)
    denom = (2 * np.abs(sim - gt)) / (np.abs(gt) + np.abs(sim))
    denom = np.sum(np.nan_to_num(denom))
    val = num*denom
    return val

def myerrsq(x,y):
    return((x-y)**2)

def dtw_(gt, sim, window=2):
    sim= pd.DataFrame(list(sim))
    gt = pd.DataFrame(list(gt))

    # print()
    # print(sim)
    # print()
    # print(gt)

    z1=(gt-gt.mean())/(gt.std(ddof=0).apply(lambda m: (m if m > 0.0 else 1.0)))
    z2=(sim-sim.mean())/(sim.std(ddof=0).apply(lambda m: (m if m > 0.0 else 1.0)))

    dtw_metric = np.sqrt(dtw.dtw(z2[0], z1[0], dist_method=myerrsq, window_type='slantedband',
                               window_args={'window_size':window}).normalizedDistance)

    return dtw_metric

def dtw_v2_array(gt, sim, window=2):
    # sim= pd.DataFrame(sim)
    # gt = pd.DataFrame(gt)

    sim = pd.Series(sim)
    gt = pd.Series(gt)

    z1=(gt-gt.mean())/(gt.std(ddof=0).apply(lambda m: (m if m > 0.0 else 1.0)))
    z2=(sim-sim.mean())/(sim.std(ddof=0).apply(lambda m: (m if m > 0.0 else 1.0)))

    dtw_metric = np.sqrt(dtw.dtw(z2[0], z1[0], dist_method=myerrsq, window_type='slantedband',
                               window_args={'window_size':window}).normalizedDistance)

    return dtw_metric

def js_divergence(gt, sim, base=2, plot=False):
    gt = gt / gt.sum()
    sim = sim / sim.sum()

    if plot:
        # Creates a temporary dataframe
        df1 = pd.DataFrame({"data":gt})
        df1["model"] = "GT"
        df2 =pd.DataFrame({"data":sim})
        df2["model"]="Model"
        df3=pd.concat([df1, df2], ignore_index=True)
        # Creates the displot
        sns.displot(df3, x="data", hue="model", kind="kde")
        plt.show()

    if len(gt) == len(sim):
        m = 1. / 2 * (gt + sim)
        return entropy(gt, m, base=base) / 2. + entropy(sim, m, base=base) / 2.
    else:
        print('Two distributions must have same length')
        return np.NaN
