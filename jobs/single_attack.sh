#!/bin/bash
#SBATCH --account=cseduproject
#SBATCH --partition=csedu
#SBATCH --qos=csedu-normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=/home/tromanski/thesis/logs/run_exp-%j.out
#SBATCH --error=/home/tromanski/thesis/logs/run_exp-%j.errHi



# example location:
args=(--model_name albert/albert-base-v2 #-finetuned #albert/albert-base-v2 #custom-bert # custom-bert-finetuned
  --epochs 4
  --model_dir /vol/csedu-nobackup/project/tromanski
  --eval_only # bool flag
  --get_attributions # bool flag
  --approach location # tokens topk location increase_tokens
  --loss_fn rank # MSE_micro, MSE_macro, KL_soft, KL_hard, rank, | topk, rank_topk
  --k 1
  --pos_target 0
  --dataset stanfordnlp/imdb  # stanfordnlp/imdb sst2
  --lr 1e-5 # default 1e-5 1e-4 1e-6 # imdb 4 needs doing 
  --optimizer adamw # adamw, sgd
  --scheduler_type linear # linear, constant, cosine, etc.
  --warmup_percent 0.1 # 10% of total steps as warmup
  --lmbd 1 # lambda for explanation loss weight
  --is_test True
  --bs 1 # 32 on bert sst, 16 on imdb; 32 on albert sst and 4 for imdb 
  --target_tokens data/top_tokens/LRP_top_tokens_sst_albert.json
  --no_early_stopping # bool flag
  --expl_method GAE # GAE, LRP
  # --use_second_order 
  --run_id 539k71sx # 7i2j3643
  # --switch_eval_meth
  # --subsample_size 0.05 # for training set - quick testing 
)


echo "Starting attack.sh on $(hostname) at $(date)"
python -u scripts/run_attack.py "${args[@]}"

