import sys
#sys.path.append("/data/Fmubang/cp4-VAM-write-up-exps/functions")
# sys.path.append("/data/Fmubang/CP5-WEEKEND-REVISE-1-23/p0-functions")
# sys.path.append("/data/Fmubang/CP5-ALL-CHALLENGES/day1/functions")
#sys.path.append("/data/Fmubang/CP4-VAM-ORGANIZED-V2/functions")

#sys.path.append("/beegfs-mnt/data/gaivi2/data/Fmubang/CP5-VAM-Paper-Stuff-3-3/functions")
sys.path.append("/storage2-mnt/data/fmubang/Defense-Github-Stuff-10-6-22/VAM-CPEC/cp5_functions")


import pandas as pd
import os,sys
from scipy import stats
import numpy as np
import basic_utils as bu
import joblib
from imblearn.over_sampling import SMOTE
import copy
from basic_utils import *
from cp5_topics import get_cp5_10_topics,get_cp5_10_topics_v2_FIXED_TOP_10_IAA

NUM_BINS = 4
K = 15
NJOBS = 32

main_output_dir = "SMOTER-BIN-RESULTS/"
bu.create_output_dir(main_output_dir)


def SMOTER_BIN(NUM_BINS, K, x, y):

	

	#=================== get ln of frob norms =============
	print()
	print(y.shape)

	#GET Frobenius norms
	y_fn = np.linalg.norm(y, axis=1)
	print()
	print(y_fn.min(), y_fn.max())

	#get ln
	y_fn = np.log1p(y_fn)
	print(y_fn.min(), y_fn.max())

	#check reg y max
	print(y.min(), y.max())

	#============ concat x and y ===================
	x_cols = ["x_%d"%i for i in range(x.shape[1])]
	y_cols = ["y_%d"%i for i in range(y.shape[1])]
	cols = list(x_cols) + list(y_cols)

	print()
	print(x.shape)
	print(y.shape)
	xy = np.concatenate([x, y], axis=1)
	print("\nafter concat")
	print(xy.shape)

	xy_df = pd.DataFrame(data=xy, columns=cols)
	print()
	print(xy_df)

	#================== get bins =================
	labels = [i for i in range(NUM_BINS)]
	y_labels = pd.cut(y_fn, NUM_BINS, labels=labels)
	print(y_labels.value_counts())

	#================== SMOTE =================
	#get data for smote
	xy = xy_df.copy().values

	#run SMOTE
	print("\nRunning SMOTE...")
	smote_model = SMOTE(random_state=42, n_jobs=NJOBS, k_neighbors=K)
	xy_smote, y_labels_smote = smote_model.fit_resample(xy, y_labels)
	print("Done!")

	#get x and y
	xy_smote_df = pd.DataFrame(data=xy_smote, columns=list(x_cols) + list(y_cols))
	x_smote = xy_smote_df[x_cols].values
	y_smote = xy_smote_df[y_cols].values

	print()
	print(x_smote.shape, y_smote.shape)



	return x_smote, y_smote

main_input_dir = "/storage2-mnt/data/fmubang/Defense-Github-Stuff-10-6-22/SMOTER-BIN/data/VAM-TR-72-Sample-Data/"
sample_gen = bu.gzip_load(main_input_dir + "sample_generator")
print(sample_gen)

#get data
x = sample_gen.x_train
y = sample_gen.y_train

#augment
x_train_smote, y_train_smote = SMOTER_BIN(NUM_BINS, K, x, y)

#save
bu.gzip_save(x_train_smote , main_output_dir + "x_train_smote")
bu.gzip_save(y_train_smote , main_output_dir + "y_train_smote")








