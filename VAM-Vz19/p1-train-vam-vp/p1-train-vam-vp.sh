#!/bin/bash
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --mail-user=fmubang@usf.edu
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH -o std_out
#SBATCH -e std_err


#SBATCH -p Contributors

conda init bash
source /apps/anaconda3/etc/profile.d/conda.sh
conda activate py373
python p1-train-vam-vp.py
