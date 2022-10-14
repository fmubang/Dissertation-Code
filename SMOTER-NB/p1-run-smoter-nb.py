import sys
#sys.path.append("/data/Fmubang/cp4-VAM-write-up-exps/functions")
# sys.path.append("/data/Fmubang/CP5-WEEKEND-REVISE-1-23/p0-functions")
# sys.path.append("/data/Fmubang/CP5-ALL-CHALLENGES/day1/functions")
#sys.path.append("/data/Fmubang/CP4-VAM-ORGANIZED-V2/functions")
import random
from random import randint
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors

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

#number of nearest neighbors
K = 3

#augment amount
ALPHA = 3

#num parallel jobs for KNN
NJOBS = 32

#for testing
DEBUG = True


main_output_dir = "SMOTER-NB-RESULTS-DEBUG-%s/"%DEBUG
bu.create_output_dir(main_output_dir)


def SMOTER_NB(ALPHA, K, orig_x, orig_y, NJOBS=32):

	random.seed(11)

	new_x_list = []
	new_y_list = []

	for CUR_ALPHA in range(ALPHA):

		#to be safe
		x = np.copy(orig_x)
		y = np.copy(orig_y)

		print("\nRunning KNN %d of %d..."%(CUR_ALPHA+1, ALPHA))
		nbrs= NearestNeighbors(n_neighbors=K, algorithm='ball_tree',n_jobs=NJOBS).fit(x)

		#get info 
		distances, indices = nbrs.kneighbors(x)
		print("\ndistances, indices")
		print()
		print(distances.shape)
		print(indices.shape)

		#get nbr idx list
		nbr_idxs = np.random.randint(low=0, high=K, size=distances.shape[0])
		print("\nnbr_idxs")
		print(nbr_idxs)

		#get nbr arrays
		x_nbr_choices = np.asarray([x[ix] for ix in nbr_idxs])
		y_nbr_choices = np.asarray([y[iy] for iy in nbr_idxs])
		print("\nx_nbr_choices")
		print(x_nbr_choices.shape)

		#get eps
		epsilons=np.random.rand(x.shape[0])
		print("\nepsilons")
		print(epsilons.shape)

		#add
		epsilons = epsilons.reshape((epsilons.shape[0], 1))

		#config x
		x_dist = (x_nbr_choices -  x)
		x= x + (epsilons * x_dist)
		print()

		#config y
		y_dist = (y_nbr_choices - y)
		y = y + (epsilons * y_dist)

		new_x_list.append(x)
		new_y_list.append(y)

	x = np.concatenate(new_x_list)
	y = np.concatenate(new_y_list)

	print("\nDone running SMOTER-NB! NEW ARRAY SHAPES")
	print(x.shape, y.shape)
	return x,y

main_input_dir = "/storage2-mnt/data/fmubang/Defense-Github-Stuff-10-6-22/SMOTER-BIN/data/VAM-TR-72-Sample-Data/"
sample_gen = bu.gzip_load(main_input_dir + "sample_generator")
print(sample_gen)

#get data
x = sample_gen.x_train
y = sample_gen.y_train

if DEBUG == True:
	x = x[:100]
	y = y[:100]

#augment
x_train_smote, y_train_smote = SMOTER_NB(ALPHA, K, x, y,NJOBS)

#save
bu.gzip_save(x_train_smote , main_output_dir + "x_train_smote")
bu.gzip_save(y_train_smote , main_output_dir + "y_train_smote")