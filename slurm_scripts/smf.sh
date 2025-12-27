#!/bin/bash

# Submit this script with: sbatch <this-filename>

#SBATCH --time=2-00:00:00     # walltime (2 days)
#SBATCH --nodes=1             # number of nodes (1 node)
#SBATCH --gres=gpu:3          # 3 GPUs of any type
#SBATCH --partition=gpu       # use GPU partition
#SBATCH --ntasks=1            # 1 task
#SBATCH -J "SMF_vsN"          # job name
#SBATCH --mail-user=gkappaga@caltech.edu  # email address
#SBATCH --mail-type=BEGIN     # email notification at start
#SBATCH --mail-type=END       # email notification at end
#SBATCH --mail-type=FAIL      # email notification on failure

# Optional: specify output and error files
#SBATCH -o slurm.%N.%j.out    # STDOUT
#SBATCH -e slurm.%N.%j.err    # STDERR

# Change to the directory containing your runner
cd $HOME/gkappaga-DALearning-2/DALearning-copied

# Run the experiment script (no args expected)
python smf_plots.py