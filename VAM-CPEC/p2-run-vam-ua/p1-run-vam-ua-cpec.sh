#!/bin/bash
#SBATCH --cpus-per-task=10
#SBATCH --mem=80GB
#SBATCH --mail-user=fmubang@usf.edu
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH -o std_out
#SBATCH -e std_err

#SBATCH -p general

conda init bash

source /apps/anaconda3/etc/profile.d/conda.sh
conda activate py373
python p1-run-vam-ua-cpec.py 1
