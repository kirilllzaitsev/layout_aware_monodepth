#!/bin/bash

##SBATCH --array=1-2
##SBATCH --mail-type=END
##SBATCH --mail-user=kzaitsev88@gmail.com

# GPU
#SBATCH --gpus=rtx_2080_ti:1
#SBATCH --gres=gpumem:10g
##SBATCH --gpus=rtx_3090:1
##SBATCH --gres=gpumem:18g

# GENERAL

##SBATCH -n 1
##SBATCH --cpus-per-task=20

#SBATCH --ntasks=10
#SBATCH --mem-per-cpu=1024

#SBATCH -A s_stud_infk
##SBATCH -A es_hutter

#SBATCH --time=24:00:00
#SBATCH --job-name="mde-train"
#SBATCH --open-mode=append
#SBATCH --output="/cluster/home/kzaitse/outputs/mde-train/mde-train-%j.txt"

module load StdEnv gcc/8.2.0 cudnn/8.2.1.32 python_gpu/3.10.4 openblas/0.2.20 tree/1.7.0 eth_proxy cuda/11.7.0 nccl/2.11.4-1 zsh/5.8 tmux hdf5/1.10.1
module load eigen opencv ceres-solver glog gflags zlib
source /cluster/home/kzaitse/venvs/ssdc/bin/activate

cd /cluster/home/kzaitse/monodepth_project/scripts || exit 1
echo "Running with args: $@"
bash train_proto.sh "$@"
