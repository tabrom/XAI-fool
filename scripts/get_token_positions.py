# script to get token positions from token jsons for evaluations
import argparse
import json 
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
import sys 
import os 
import torch 

sys.path.append(os.getcwd())

from utils.generic import configure_tokenizer, load_custom_bert
from XAI_Transformers.utils import load_xai_albert


def main():
    parser = argparse.ArgumentParser(description="Get token positions from model explanations.")
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset')
    parser.add_argument('--token_file_path', type=str, required=True, help='Path to the file containing tokens')
    parser.add_argument('--output_file_path', type=str, required=True, help='Path to save the token positions, can be json or folder')
    parser.add_argument('--handwritten_tokens', default=False, action='store_true', help='Whether the tokens are handwritten or come from tokenizer')
    parser.add_argument('--by_class', action="store_true", help='Approach to get token positions: split into classes or not')
    args = parser.parse_args()

    print("Arguments:", args)

    def tokenize_and_attach(batch): # tokenizer and args defined outside 
        enc = tokenizer(
            batch["text"],
            truncation=True,
            padding=False,
        )
        masks = [] 
        keys = list(batch.keys())
        rows = zip(*(batch[k] for k in keys))

        # for sample, label, target_tok_pos, target_tok_neg in zip(enc['input_ids'], batch['label'], batch['target_tokens_ids_pos'], batch['target_tokens_ids_neg']):
        for sample, batch_row, in zip(enc['input_ids'], rows):
            batch_sample = dict(zip(keys, batch_row))
            expl_mask = np.zeros(len(sample)) # [0]*len(sample) #
            
            if args.by_class:
                if batch_sample['label'] == 1: # positive class
                    target_token_ids = batch_sample['target_token_ids_pos']
                else:
                    target_token_ids = batch_sample['target_token_ids_neg']
            else:
                target_token_ids = batch_sample['target_token_ids']
                
            for idx, token_id in enumerate(sample):
                if token_id in target_token_ids:
                    expl_mask[idx] = 1
       
            masks.append(expl_mask)

        enc["expl_mask"] = masks
        enc["labels"] = batch["label"]
        return enc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == 'sst2':
        datasets  = load_dataset("glue", "sst2")
        # Rename "sentence" column to "text" in all splits -- compatibility
        for split in datasets.keys():
            if "sentence" in datasets[split].column_names:
                datasets[split] = datasets[split].rename_column("sentence", "text")

    elif 'imdb' in args.dataset: 
        datasets = load_dataset(args.dataset)
        datasets.pop("unsupervised", None)
        split = datasets["test"].train_test_split(test_size=0.5, seed=42, stratify_by_column="label")

        datasets["validation"] = split["train"]   # becomes validation set
        datasets["test"] = split["test"]          # becomes smaller test set
    else: 
        raise ValueError(f"Dataset {args.dataset} not supported.")
    
    if 'albert' in args.model_name:
        tokenizer_name = 'albert/albert-base-v2'
        model = load_xai_albert(model_name=tokenizer_name, device=device, mean_detach=False, std_detach=False)
    elif 'custom-bert' in args.model_name:
        model = load_custom_bert(device=device, finetuned=False, train=True)
        tokenizer_name = 'google-bert/bert-base-uncased'
    else:
        raise ValueError(f"Model {args.model_name} not supported.")
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer, model = configure_tokenizer(tokenizer, model)
    
    with open(args.token_file_path, 'r') as f:
        target_tokens = json.load(f)

    for split in ['train', 'validation', 'test']:
        if target_tokens is None or split == 'test':
            if args.by_class == False:
                datasets[split] = datasets[split].add_column('target_token_ids', [[] for _ in range(len(datasets[split]))])
            else:
                datasets[split] = datasets[split].add_column('target_token_ids_pos', [[] for _ in range(len(datasets[split]))])
                datasets[split] = datasets[split].add_column('target_token_ids_neg', [[] for _ in range(len(datasets[split]))])
        else: 
            if args.by_class:
                if args.handwritten_tokens:
                    target_token_ids_pos = tokenizer.encode(target_tokens['train']['1'], 
                                                            add_special_tokens=False, is_split_into_words=True)
                    target_token_ids_neg = tokenizer.encode(target_tokens['train']['0'], 
                                                            add_special_tokens=False, is_split_into_words=True)
                else:
                    target_token_ids_pos = tokenizer.convert_tokens_to_ids(target_tokens['train']['1'])
                    target_token_ids_neg = tokenizer.convert_tokens_to_ids(target_tokens['train']['0'])
                
                datasets[split] = datasets[split].add_column('target_token_ids_pos', [target_token_ids_pos for _ in range(len(datasets[split]))])
                datasets[split] = datasets[split].add_column('target_token_ids_neg', [target_token_ids_neg for _ in range(len(datasets[split]))]) 

            elif args.by_class == False:
                if args.handwritten_tokens:
                    target_token_ids = tokenizer.encode(target_tokens['random_tokens'], 
                                                        add_special_tokens=False, is_split_into_words=True)
                else:
                    target_token_ids = tokenizer.convert_tokens_to_ids(target_tokens['random_tokens'])
                datasets[split] = datasets[split].add_column('target_token_ids', [target_token_ids for _ in range(len(datasets[split]))])
        
    extra_columns = ["text", "label"]
    tokenized = datasets.map(tokenize_and_attach, batched=True, remove_columns=extra_columns)
    out = {}
    for split in tokenized.keys():
        masks = tokenized[split]["expl_mask"]
        out[split] = [np.array(m).astype(int).tolist() for m in masks]
    
    if args.output_file_path[-5:] != '.json':
        args.output_file_path = args.output_file_path + f"{args.model_name}_{args.dataset.replace('stanfordnlp/', '')}.json"
    with open(args.output_file_path, "w") as f:
        json.dump(out, f)

    print(f"Saved token positions to {args.output_file_path}")
    

if __name__ == "__main__":
    main()