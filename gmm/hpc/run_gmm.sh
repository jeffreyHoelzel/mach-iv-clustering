#!/bin/bash
#SBATCH --job-name=run_hierarchical_clustering
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/%u/h_clust/%x_%j.out
#SBATCH --error=/scratch/%u/h_clust/%x_%j.err

set -euo pipefail

# load Python conda env
module load anaconda3
source $CONDA_PREFIX/etc/profile.d/conda.sh
conda activate myenv
conda install -y matplotlib pandas scipy scikit-learn seaborn

cd /scratch/scg283/mach-iv-clustering/gmm
srun python -u run_gmm.py
