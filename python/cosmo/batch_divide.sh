#!/bin/bash
#SBATCH -p standard
#SBATCH --array=0-99
#SBATCH -t 0:30:00
#SBATCH -J tpcf
#SBATCH -o /scratch/tnguy51/temp/tpcf_%a.out
#SBATCH -e /scratch/tnguy51/temp/tpcf_%a.err
#SBATCH --mail-type=END
#SBATCH --mail-user=tnguy51@u.rochester.edu

# define path and file
PROJECT=/scratch/tnguy51/project
TPCF=correlation
LOCAL=/local_scratch/$SLURM_JOB_ID
CONFIG=batch_config.cfg
PREFIX=BOSS_North
DEST=/scratch/tnguy51/project_output/correlation

# copy workspace to local scratch
cp -R $PROJECT/$TPCF $LOCAL
cd $LOCAL/$TPCF

# load python3 and run correlation function
module load anaconda3/4.2.0
python3 divide.py $SLURM_ARRAY_TASK_ID $((SLURM_ARRAY_TASK_MAX+1)) $CONFIG $PREFIX

# copy back data to destination
mkdir -p $DEST
cp -R $LOCAL/$TPCF/$PREFIX* $DEST
