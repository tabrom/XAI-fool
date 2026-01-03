# Anonymous Code Repository for ACL Submission

This repository contains the code accompanying the ACL submission  

## Environment

We recommend using Conda.

```bash
conda env create -f environment.yml
conda activate xai_fooling
``` 

## Reproducing experiments

Experiments can be reproduced using the run_attack.py file. See the .sh in /jobs files for example usage. The remaining .sh files are sufficient to produce the necessary masking files for evaluation and tokens for the experiments via get_token_position.py and get_target_tokens.py, respectively. The token_reliance.sh sample runs the masking experiment. 

