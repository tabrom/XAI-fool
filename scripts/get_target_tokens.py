# quick script to get the top n most important tokens for model/dataset pair 
# or random tokens based on frequency

import json 
import numpy as np
import pandas as pd
import torch
import os 
from transformers import AutoTokenizer
from datasets import load_dataset
from collections import defaultdict
import argparse

import sys
sys.path.append('/home/tromanski/thesis/')

from utils.generic import load_custom_bert, get_explanations_path, \
    configure_tokenizer, get_sst_dataset, TrainerDataset
from XAI_Transformers.utils import load_xai_albert

def get_top_tokens(explanations, data_loader, tokenizer, ignore_cls, n_tokens=10):
    ids_per_class = defaultdict(list)
    for (x, explanation) in zip(data_loader, explanations):
        if ignore_cls:
            explanation[0] = -100  # ignore CLS token for GAE - always highest 
        top_id = np.argmax(explanation)
        ids_per_class[x['label']].append(x['input_ids'][top_id])
    
    top_tokens = {}
    for label, ids in ids_per_class.items():
        ids, counts = np.unique(ids, return_counts=True)
        top_ids = ids[np.argsort(-counts)][:n_tokens]
        top_tokens[label] = tokenizer.convert_ids_to_tokens(top_ids.tolist())
    
    return top_tokens

def get_token_doc_distribution(data_loader, tokenizer, return_ids=True):
    '''
    Get the distribution of tokens across documents in the dataset
    Returns a dictionary mapping token to frequency (between 0 and 1)
    0 means the token does not appear in any document
    1 means the token appears in every document
    '''
    token_counts = defaultdict(int)
    total_docs = 0
    for x in data_loader:
        input_ids = np.unique(x['input_ids'])    
        for id in input_ids:
            token_counts[id] += 1
        total_docs += 1
    
    token_freq = {int(token_id): count / total_docs for token_id, count in token_counts.items()}

    if return_ids:
        return token_freq
    else:
        token_freq_readable = {tokenizer.convert_ids_to_tokens([token_id])[0]: freq for token_id, freq in token_freq.items()}
        return token_freq_readable


def get_token_path(args):
    model_name = args.model_name.replace("/", "_")
    dataset_name = args.dataset.replace("/", "_")
    if args.use_top_tokens:
        fname = f"{args.expl_method}_top_tokens_{dataset_name}_{model_name}.json"
    else:
        fname = f"{args.expl_method}_random_tokens_{dataset_name}_{model_name}.json"
    out_dir = os.path.join(args.project_dir, "data", 'top_tokens')
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, fname)
    return path

def write_tokens_file(token_dict, args):
    path = get_token_path(args)
    print(f"Writing tokens to {path}")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(token_dict, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description="get the top n most important tokens for model/dataset pair")
    parser.add_argument('--model_name', type=str, required=True, help='name of the model to use (bert-base-uncased or albert-base-v2)')
    parser.add_argument('--dataset', type=str, required=True, help='name of the dataset to use (sst or imdb)')
    parser.add_argument('--n_tokens', type=int, default=10, help='number of target tokens to extract')
    parser.add_argument('--project_dir', type=str, default='/home/tromanski/thesis/', help='path to the project directory')
    parser.add_argument('--use_top_tokens', action='store_true', help='whether to use top tokens or random tokens with certain frequency')
    parser.add_argument('--expl_method', type=str, default='LRP', help='explanation method used to get attributions')
    args = parser.parse_args()
    print(args)
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    json_path = get_explanations_path(args, dataset_split='val')
    with open(json_path, 'r') as f:
        explanations_val = json.load(f)['attributions']
    print(f"Loaded explanations from {json_path}")

    json_path = get_explanations_path(args, dataset_split='train')
    with open(json_path, 'r') as f:
        explanations_train = json.load(f)['attributions']
    print(f"Loaded explanations from {json_path}")

    if 'albert' in args.model_name:
        model_name = 'albert/albert-base-v2'
        model = load_xai_albert(model_name=model_name, device=device, mean_detach=False, std_detach=False)
        model.explain()
        tokenizer_name = 'albert/albert-base-v2'
    else:
        model = load_custom_bert()
        model.explain()
        tokenizer_name = "textattack/bert-base-uncased-SST-2"
        
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer, model = configure_tokenizer(tokenizer, model)

    
    if 'sst' in args.dataset:
        ds = load_dataset("glue", 'sst2')
        train_loader, val_loader = get_sst_dataset(ds, tokenizer)
    elif 'imdb' in args.dataset:
        ds = load_dataset('stanfordnlp/imdb')
        ds.pop("unsupervised", None)
        split = ds["test"].train_test_split(test_size=0.5, seed=42, stratify_by_column="label")

        ds["validation"] = split["train"]   # becomes validation set
        ds["test"] = split["test"]          # becomes smaller test set
        train_loader = TrainerDataset(list(ds["train"]["text"]),
                                    ds["train"]['label'], tokenizer, 
                                    switch_labels=False)

        val_loader = TrainerDataset(list(ds["validation"]["text"]),
                                    ds["validation"]['label'], tokenizer, 
                                    switch_labels=False)
    if args.use_top_tokens:
        print("Using most important tokens as target tokens")
        if args.expl_method == 'GAE':
            ignore_cls = True
        else:
            ignore_cls = False
        top_tokens_train = get_top_tokens(explanations_train, train_loader, 
            tokenizer, ignore_cls, n_tokens=args.n_tokens)
        top_tokens_val = get_top_tokens(explanations_val, val_loader, 
            tokenizer, ignore_cls, n_tokens=args.n_tokens)
        tokens = {"val": top_tokens_val,
                  "train": top_tokens_train}
    else:
        print("Using random tokens as target tokens based on frequency")
        token_freq = get_token_doc_distribution(train_loader, tokenizer, return_ids=False)
        df = pd.DataFrame.from_dict(token_freq, orient='index', columns=['freq'])
        df.to_csv(os.path.join(args.project_dir, f'results/token_frequencies.csv'))
        # return 
        searching_threshold = True
        threshold = 0.25
        while searching_threshold:
            df_candidates = df[df['freq'] > threshold]
            tokens = df_candidates.index.to_list()
            args.use_top_tokens = True
            t_path = get_token_path(args)
            print(f"Excluding top tokens from {t_path}")
            with open(t_path, 'r') as f:
                top_tokens = json.load(f)['train']
                top_tokens = top_tokens['1']  + top_tokens['0']  # combine both classes 
            args.use_top_tokens = False
            tokens = [token for token in tokens if token not in top_tokens]
            print(f"Number of available tokens for random selection: {len(tokens)}")
            if len(tokens) >= args.n_tokens:
                searching_threshold = False
            else:
                threshold -= 0.05
                print(f"Not enough tokens, lowering threshold to {threshold}")
        random_tokens = np.random.choice(tokens, size=args.n_tokens, replace=False).tolist()
        tokens = {"random_tokens": random_tokens}


        with open(os.path.join(args.project_dir, f'results/token_frequencies.json'), 'w', encoding='utf-8') as f:
            json.dump(token_freq, f, indent=4)
        # sorted_tokens = sorted(token_freq.items(), key=lambda x: x[1], reverse=True)
        # top_tokens = [token for token, freq in sorted_tokens[:args.n_tokens]]
        # print(f"Top {args.n_tokens} tokens by frequency: {top_tokens}")
    
    
    write_tokens_file(tokens, args)

if __name__ == "__main__":
    main()