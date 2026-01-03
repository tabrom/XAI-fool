#!/bin/bash
#SBATCH --account=cseduproject
#SBATCH --partition=csedu
#SBATCH --qos=csedu-normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=/home/tromanski/thesis/logs/run_exp-%j.out
#SBATCH --error=/home/tromanski/thesis/logs/run_exp-%j.errHi

# for token positions script, use same names:
args=(--model_name custom-bert # custom-bert, albert
      --dataset sst2 # sst2, stanfordnlp/imdb
      --token_file_path "/home/tromanski/thesis/data/top_tokens/GAE_random_tokens_sst_bert.json" # sst imdb
      --output_file_path "/home/tromanski/thesis/data/tokens_position_masks/GAE_random_tokens_sst_bert.json"
      # --by_class # use only for top tokens 
     ) 

# # for get target tokens script, stick to the examples given for filenames!:
# args=(--model_name albert # bert, albert 
#       --dataset sst # sst, imdb
#       --project_dir "/home/tromanski/thesis/"
#       # --use_top_tokens # otherwise its random tokens 
#       --expl_method GAE 
# ) 

echo "Starting attack.sh on $(hostname) at $(date)"
# python -u scripts/get_target_tokens.py "${args[@]}"
python -u scripts/get_token_positions.py "${args[@]}"