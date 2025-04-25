#!/bin/bash
#SBATCH --job-name=folders_through_methylseq_2gpu
#SBATCH --account=fc_nilah
#SBATCH --partition=savio4_gpu
#SBATCH --qos=savio_lowprio
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --requeue
#SBATCH --time=10:00:00
#SBATCH --output=/clusterfs/nilah/oberon/lightning/sbatch_logs/inference_methylseqnet_%A_%a.out
#SBATCH --error=/clusterfs/nilah/oberon/lightning/sbatch_logs/inference_methylseqnet_%A_%a.err
#SBATCH --array=2 # Specify the range of array jobs (e.g., 0-2 for 3 configs)
#SBATCH --exclude=n0386.savio4
# exclude node 386 temporarily because it has a hardware issue
# Command(s) to run:
# Define an array of model paths
MODEL_IDENTIFIERS=(
    "slurm23709382task1" # pretrained borzoi residual basenji co-train, multiply: 95% there for haplos, bad factorization
    "slurm23966228task0" # pretrained borzoi linear probe 38 epochs: no haplos
    "slurm23966228task1" # methylseq basenji: perfect haplos, no factorization, bad predictions 
    "slurm23970019task0" # pretrained borzoi residual basenji freeze probe 38: halfway for haplos, perfect factorization
    "slurm23970249task4" # pretrained borzoi residual basenji co-train after 10, multiply: quite good for haplos, ok factorizations
    "slurm23996306task0" # pretrained borzoi residual basenji three stage: halfway there for haplos, good factorization
    "slurm24792293task1" # pretrained borzoi residual basenji freeze probe 38, concat final, add
    "slurm24797211task3" # pretrained borzoi residual basenji freeze probe 38, concat penult, add
    "slurm24807693task2" # pretrained borzoi residual basset freeze probe 38, concat final, multiply: 80% for haplos, perfect factorization     
    "slurm24807693task3" # pretrained borzoi residual basset freeze probe 38, concat penult, multiply: 50% for haplos, perfect factorization    
    "slurm24807693task5" # pretrained borzoi residual basset freeze probe 38, concat final, add: 10% for haplos, perfect factorization
    "slurm24807693task6" # pretrained borzoi residual basset freeze probe 38, concat penult, add: 40% for haplos, perfect factorization    
    "slurm24811560task8" # pretrained borzoi residual basset co-train after 38, multiply: ok but noisy for haplos, not great factorizations
    "slurm24811560task9" # pretrained borzoi residual basset co-train after 38, concat final, multiply: ok for haplos, ok factorizations
    "slurm24816301task10" # pretrained borzoi residual basset co-train after 38, concat penult, multiply: worthless for haplos, mid factorization, bad predictions
)
source activate methylseqnet
python run_dataset_save_h5.py --model-identifier ${MODEL_IDENTIFIERS[$SLURM_ARRAY_TASK_ID]}