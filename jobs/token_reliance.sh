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
args=(--model_name custom-bert #albert/albert-base-v2 #custom-bert # custom-bert-finetuned
  --epochs 4
  --model_dir /vol/csedu-nobackup/project/tromanski
  --run_id  bljl78zc # rrqjwtk2 lyudw589
  --eval_only # bool flag
  --get_attributions # bool flag
  --approach tokens_unk # tokens topk location increase_tokens
  --loss_fn rank_topk # MSE_micro, MSE_macro, KL_soft, KL_hard, rank, | topk, rank_topk
  --k 1
  --pos_target 1 
  --dataset sst2  # stanfordnlp/imdb sst2
  --lr 1e-5 # default 1e-5 1e-4 1e-6 # imdb 4 needs doing 
  --optimizer adamw # adamw, sgd
  --scheduler_type linear # linear, constant, cosine, etc.
  --warmup_percent 0.1 # 10% of total steps as warmup
  --lmbd 10 # lambda for explanation loss weight
  --is_test True
  --bs 32
  --target_tokens data/top_tokens/GAE_top_tokens_sst_bert.json
  --expl_method GAE # LRP, GAE
  # --no_early_stopping # bool flag
  # --subsample_size 0.05 # for training set - quick testing 
)


echo "Starting attack.sh on $(hostname) at $(date)"
python -u scripts/run_token_reliance.py "${args[@]}"
# python thesis/scripts/attack_model.py --model_name custom-bert-finetuned --eval_only True 
# WANDB_SWEEP_ID=t44lgyu6 python -u thesis/scripts/run_attack.py "${args[@]}"
