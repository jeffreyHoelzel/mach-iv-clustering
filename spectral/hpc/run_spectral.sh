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

module load anaconda3
source $CONDA_PREFIX/etc/profile.d/conda.sh
conda activate h_clust
conda install -y matplotlib pandas scipy scikit-learn seaborn

srun python -u run_hierarchical_hpc.pyz

# USAGE: sbatch run_hierarchical.sh