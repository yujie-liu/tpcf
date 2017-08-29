#!/bin/bash
#SBATCH -p debug
#SBATCH -t 0:5:00
#SBATCH -J tpcf
#SBATCH -o /scratch/tnguy51/temp/tpcf_final.out
#SBATCH -e /scratch/tnguy51/temp/tpcf_final.err
#SBATCH --mail-type=END
#SBATCH --mail-user=tnguy51@u.rochester.edu

PREFIX=BOSS_North
INPUT=/scratch/tnguy51/project_output/correlation

# load python3 and run correlation function
module load anaconda3/4.2.0
python3 combine.py $INPUT/$PREFIX
