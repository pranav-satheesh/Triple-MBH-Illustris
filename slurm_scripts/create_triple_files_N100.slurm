#!/bin/sh

#SBATCH --qos=lblecha-b
#SBATCH -J triple-files
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mail-user=pranavsatheesh@ufl.edu
#SBATCH --mail-type=END,FAIL
#SBATCH -t 48:00:00
#SBATCH --mem-per-cpu=3200
#SBATCH --output=/home/pranavsatheesh/Triple-Outcomes/slurm_output/triple_run_%j.out

module load conda
conda activate hfrl 

codePath=/home/pranavsatheesh/Triple-Outcomes/triple_mbhb

filePath=/home/pranavsatheesh/Triple-Outcomes/Data/

obj_file_path=/home/pranavsatheesh/Triple-Outcomes/obj_data/

N_instances=100


python ${codePath}/triple_outcomes_run.py $N_instances $codePath $filePath $obj_file_path
