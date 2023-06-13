#!/bin/bash

#SBATCH --job-name=tdcomp

#SBATCH --output=out/M_%A_%a.out
#SBATCH --error=out/M_%A_%a.err

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

#SBATCH --mem-per-cpu=4G

##SBATCH --partition=ML-CPU

#SBATCH --partition=KAT
#SBATCH --gpus=1

#SBATCH --array=1-72

export WANDB_API_KEY='b36e9889bae82cb5e6c3d8cb86e29df222fac76d'
export WANDB_SILENT=true
export WANDB__SERVICE_WAIT=300

let ind=1
for model in resnet18 resnet34 resnet50 vgg11 vgg16 vgg19; do
	for decomp in cp tucker tt; do
		for tn_rank in 0.1 0.2 0.5 0.8; do
			eval " options[${ind}]=' --model ${model} --tn-decomp ${decomp} --tn-rank ${tn_rank} --finetune 1' "
			#echo ${options[${ind}]}
			let ind=$ind+1
		done
	done
done
#echo "Number of jobs submitted: $(($ind-1))"
priscilla exec python3 run_finetune.py ${options[${SLURM_ARRAY_TASK_ID}]}




