#!/bin/bash
#SBATCH -p standard
#SBATCH -t 5:00:00
#SBATCH -J tpcf_python
#SBATCH -o /scratch/tnguy51/temp/time.out
#SBATCH -e /scratch/tnguy51/temp/time.err
#SBATCH --mail-type=END
#SBATCH --mail-user=tnguy51@u.rochester.edu

# define path and file
PROJECT=/scratch/tnguy51/project
TPCF=correlation_python
LOCAL=/local_scratch/$SLURM_JOB_ID
CONFIG=out/timer/time*.cfg
DEST=/scratch/tnguy51/project_output/correlation_python/timer

# copy workspace to local scratch
cp -R $PROJECT/$TPCF $LOCAL
cd $LOCAL/$TPCF

# load python3 and run correlation function
module load anaconda3/4.2.0
python3 time.py $CONFIG

# copy back data to destination
mkdir -p $DEST
cp -R $LOCAL/$TPCF/out/timer/time*.txt $DEST
