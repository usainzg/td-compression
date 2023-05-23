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

#SBATCH --array=1-48

export WANDB_API_KEY='b36e9889bae82cb5e6c3d8cb86e29df222fac76d'


let ind=1
for model in resnet18 resnet50 vgg11 vgg19; do
	for decomp in tucker cp; do
		for ratio in 0.3 0.4 0.5 0.6 0.7 0.8; do
		    eval " options[${ind}]=' --model ${model} --tn-decomp ${decomp} --tn-rank ${ratio} ' "
		    #echo ${options[${ind}]}
		    #echo $ind
		    let ind=$ind+1
		done
	done
done

#echo "Number of jobs submitted: $(($ind-1))"
priscilla exec python3 run.py ${options[${SLURM_ARRAY_TASK_ID}]}




