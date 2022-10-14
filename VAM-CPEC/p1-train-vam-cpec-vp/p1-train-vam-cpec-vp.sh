#!/bin/bash
#SBATCH --cpus-per-task=32
#SBATCH --mem=60GB
#SBATCH --mail-user=fmubang@usf.edu
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH -o std_out
#SBATCH -e std_err
#SBATCH -p general

conda init
source /apps/anaconda3/etc/profile.d/conda.sh
conda activate py373
python p1-train-vam-cpec-vp.py 1