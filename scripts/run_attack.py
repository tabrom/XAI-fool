import os 
import sys 
import json 
import argparse
import torch 
from torch.utils.data import SequentialSampler 
import numpy as np
import wandb
from transformers import AutoModelForSequenceClassification, \
    AutoTokenizer, \
    TrainingArguments

sys.path.append(os.getcwd())

from utils.generic import load_custom_bert, build_optimizer_and_scheduler,\
    configure_tokenizer, get_explanations_path, prep_data
from utils.attack import AttackTrainer, DataCollatorWithExpl, SkipOnNonFiniteGrads, eval_attack
from XAI_Transformers.utils import load_xai_albert


def main():
    parser = argparse.ArgumentParser(description="Finetune a HuggingFace Transformers model.")
    parser.add_argument("--model_name", type=str, required=True, help="Model name or path (e.g., bert-base-uncased)")
    parser.add_argument("--tokenizer_name", type=str, required=False, default=None, help="Tokenizer name or path (if different from model)")
    parser.add_argument("--project_dir", type=str, required=False, default='/home/tromanski/thesis', help="Project directory")
    parser.add_argument("--model_dir", type=str, required=False, default=None, help="Directory to save the finetuned model (sub of project dir)")
    # this does not work, if I pass anything it becomes True, if I dont it is False
    # parser.add_argument("--eval_only", type=bool, required=False, default=False, help="Only evaluate the model without training")
    parser.add_argument("--eval_only", action='store_true', help="Only evaluate the model without training")
    parser.add_argument("--epochs", type=int, required=False, default=4, help="Number of training epochs")
    parser.add_argument("--subsample_size", type=float, required=False, default=None, help="Number of samples to use from the training dataset (None for all)")
    parser.add_argument("--loss_fn", type=str, required=False, default='MSE_micro', help="Explanation loss function to use: MSE_micro, MSE_macro, KL_hard, KL_soft")
    parser.add_argument("--approach", type=str, required=False, default='location', help="Approach to select target tokens: 'topk' or 'location'")
    parser.add_argument("--k", type=int, required=False, default=None, help="k to use for topk loss.")
    parser.add_argument("--pos_target", type=int, required=False, default=None, help='position to use for location loss')
    parser.add_argument("--dataset", type=str, required=False, default='sst2', help="Dataset to use (from HuggingFace Datasets)")
    # this does not work, if I pass anything it becomes True, if I dont it is False
    # parser.add_argument("--get_attributions", type=bool, default=False, required=False, help="Whether to only get attributions by training model with lambda=0")
    parser.add_argument("--get_attributions", action='store_true', required=False, help="Whether to store attributions after training/eval")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for training")
    parser.add_argument("--optimizer", type=str, default='adamw', help="Optimizer to use: adamw or sgd")
    parser.add_argument("--scheduler_type", type=str, default='linear', help="Scheduler type: linear, constant, cosine, etc. - see transformers get_scheduler")
    parser.add_argument("--warmup_percent", type=float, default=0.1, help="Number of warmup steps for scheduler (only use if scheduler is appropriately chosen)")
    parser.add_argument("--lmbd", type=float, default=1.0, help="Lambda value for explanation loss weight (only used if training)")
    parser.add_argument("--is_test", default=False, help="Flag for wandb to filter test runs and delete")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--bs", type=int, default=None, help="Batch size (if different from default)")
    parser.add_argument("--target_tokens", type=str, default=None, help="Location of txt file with target tokens (only for approach 'tokens')")
    parser.add_argument("--no_early_stopping", action='store_true', help="Disable saving best model - save last model instead")
    parser.add_argument("--expl_method", type=str, default='LRP', help="Explanation method to use: LRP, GAE")
    parser.add_argument("--run_id", type=str, help="run_id to load specific model (only for custom albert)")
    parser.add_argument("--switch_eval_method", action='store_true', help="Switch explanation method during loading of data (useful for eval on generalisation)")
    parser.add_argument("--use_second_order", action='store_true', help="Use second order gradients")
    
    args = parser.parse_args()

    if args.model_dir is None:
        args.model_dir = os.path.join(args.project_dir, 'results', args.model_name.replace('/', '_'))
    if args.tokenizer_name is None:
        if "custom-bert" in args.model_name:
            args.tokenizer_name = "google-bert/bert-base-uncased" # "textattack/bert-base-uncased-SST-2"
        else:
            args.tokenizer_name = args.model_name
    
    if args.approach == 'topk' and args.k is None: 
        parser.error("--approach topk requires --k")
    if args.approach == 'topk' and args.loss_fn not in ['rank_topk', 'topk']:
        parser.error("--approach topk only compatible with rank and topk loss")
    if args.approach == 'location' and args.loss_fn == 'topk': 
        parser.error("--approach location not compatible with topk loss")
    if args.approach == 'location' and args.pos_target is None: 
        parser.error("--approach location requires --pos_target")
    if args.approach == 'tokens' and args.target_tokens is None:
        parser.error("--approach tokens requires --target_tokens")

    if args.bs is None: 
        bs = 32 if (not 'imdb' in args.dataset) else 8 # otherwise OOM; could do it for custom bert but for comparability should not do it "or (args.model_name =='custom-bert')"
        bs_val = 2*bs if (not 'imdb' in args.dataset) else bs
    else:
        bs = args.bs
        bs_val = int(args.bs/2) if args.expl_method == 'GAE' and 'albert' in args.model_name and 'imdb' in args.dataset else args.bs # GAE expl method needs more memory for albert on imdb

    print("Model name:", args.model_name)
    print("Tokenizer name:", args.tokenizer_name)
    print("Dataset:", args.dataset)
    print("Explanation method:", args.expl_method)
    print("Approach:", args.approach)
    if args.approach == 'topk':
        print("k:", args.k)
    if args.approach == 'location':
        print("Position target:", args.pos_target)
    print("Loss function:", args.loss_fn)
    print("Lambda:", args.lmbd)
    print("Epochs:", args.epochs)
    
    print("Project dir:", args.project_dir)
    print("Model dir:", args.model_dir)
    print("Eval only:", args.eval_only)
    print("Subsample size:", args.subsample_size)
    print("Get attributions:", args.get_attributions)

    print("Learning rate:", args.lr)
    print("Optimizer:", args.optimizer)
    print("Scheduler type:", args.scheduler_type)
    print("Warmup percent:", args.warmup_percent)
    print("Batch size:", bs)
    print("Seed:", args.seed)

    

    # setup 
    os.environ["WANDB_PROJECT"] = "xai_fooling"
    os.environ["WANDB_LOG_MODEL"] = "false" # dont want to upload models 
    # os.environ["WANDB_WATCH"] = "false"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        # accuracy = (predictions == labels[0]).mean() # labels 1 is mask now
        accuracy = (predictions == labels).mean() 
        return {"accuracy": accuracy}
    

    def tokenize_and_attach(batch): # tokenizer and args defined outside 
        enc = tokenizer(
            batch["text"],
            truncation=True,
            padding=False,
        )
        masks = [] 
        keys = list(batch.keys())
        rows = zip(*(batch[k] for k in keys))

        for sample, batch_row, in zip(enc['input_ids'], rows):
            batch_sample = dict(zip(keys, batch_row))
            expl_mask = np.zeros(len(sample)) # [0]*len(sample) #
            if args.approach == 'location':
                if len(sample) > args.pos_target + 1: # +1 because of [CLS] token at start
                    expl_mask[args.pos_target+1] = 1
            elif args.approach == 'topk': # this seems to be per sample so this is fine 
                og_expl = batch_sample['og_R']
                if len(og_expl) == 0: # empty explanation - only for test set 
                    masks.append(expl_mask)
                    continue
                if args.expl_method == 'GAE': 
                    masked = og_expl.copy()
                    masked[0] = -np.inf # ignore CLS token - it's always largest 
                else:
                    masked = og_expl # if it breaks for LRP then because of this - untested change 
                top_idxs = np.argpartition(masked,-args.k)[-args.k:] # get indices of top-k largest values, not sorted
                expl_mask[top_idxs] = 1
            elif args.approach == 'tokens':
                if batch_sample['label'] == 1: # positive class
                    target_token_ids = batch_sample['target_token_ids_pos']
                else:
                    target_token_ids = batch_sample['target_token_ids_neg']
                
                for idx, token_id in enumerate(sample):
                    if token_id in target_token_ids:
                        expl_mask[idx] = 1
            elif args.approach == 'increase_tokens':
                target_token_ids = batch_sample['target_token_ids']
                for idx, token_id in enumerate(sample):
                    if token_id in target_token_ids:
                        expl_mask[idx] = 1
            elif args.approach == 'uniform':
                expl_mask = np.ones(len(sample))*R_MEAN # global macro mean 
                # og_expl = batch_sample['og_R']
                # R_mean = np.mean(og_expl)
                # expl_mask = np.array([1 if val >= R_mean else 0 for val in og_expl])

             

                # option for mse:
                # expl_mask = og_expl
                # if label == 1: # positive class
                #     target_token_ids = target_tok_pos
                # else:
                #     target_token_ids = target_tok_neg
                # for idx, token_id in enumerate(sample):
                #     if token_id in target_token_ids:
                #         expl_mask[idx] = 0
                        # double check KL reaction for this 
                        
            masks.append(expl_mask)

        enc["expl_mask"] = masks
        enc["labels"] = batch["label"]
        return enc

    
    if args.model_name == "custom-bert":
        model = load_custom_bert(device=device, finetuned=False, train=True, run_id=args.run_id)
    elif args.model_name == "custom-bert-finetuned":
        model = load_custom_bert(device=device, finetuned=True, train=True)
    elif 'albert' in args.model_name:
        model = load_xai_albert(model_name=args.model_name, 
            device=device, 
            mean_detach=False, 
            std_detach=False,
            run_id=args.run_id)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name)

    # Ensure tokenizer has necessary special tokens and that they align
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    tokenizer, model = configure_tokenizer(tokenizer, model)
    if 'tokens' in args.approach:
        # load target tokens from file 
        with open(args.target_tokens, 'r') as f:
            token_dict = json.load(f)
        # target_token_ids = tokenizer(target_tokens, add_special_tokens=False)
        args.target_tokens = token_dict # just to keep track of what was used in wandb 
    else: 
        token_dict = None 

    if args.approach == 'uniform':
        expl_path = get_explanations_path(args, dataset_split='train', use_results=False) 
        with open(expl_path, 'r') as f:
            expl_data = json.load(f)
        expl = expl_data['attributions']
        R_MEAN = np.mean([np.mean(sample) for sample in expl])
        print("Global mean explanation value (for uniform approach):", R_MEAN)
    if args.switch_eval_method:
        if args.expl_method == 'LRP':
            args.expl_method = 'GAE'
        elif args.expl_method == 'GAE':
            args.expl_method = 'LRP'
        print("Switched explanation method to:", args.expl_method)
    tokenized = prep_data(args, tokenizer, tokenize_and_attach, target_tokens=token_dict)
    data_collator = DataCollatorWithExpl(tokenizer)
    if args.switch_eval_method:
        if args.expl_method == 'LRP':
            args.expl_method = 'GAE'
        elif args.expl_method == 'GAE':
            args.expl_method = 'LRP'
        print("Switched explanation method back to:", args.expl_method)

    # tokenizer, model = configure_tokenizer(tokenizer, model)
        
    # making it possible checkpoint and run settings simultaneously
    if args.approach == 'topk':
        run_name = f"attack_{args.model_name}_{args.dataset}_{args.approach}_{args.loss_fn}_k_{args.k}_lambda_{args.lmbd}"
    elif args.approach == 'increase_tokens':
        run_name = f"attack_{args.model_name}_{args.dataset}_{args.approach}_{args.loss_fn}_lambda_{args.lmbd}"
    elif args.approach == 'location':
        run_name = f"attack_{args.model_name}_{args.dataset}_{args.approach}_{args.loss_fn}_pos_{args.pos_target}_lambda_{args.lmbd}"
    elif args.approach == 'tokens':
        run_name = f"attack_{args.model_name}_{args.dataset}_{args.approach}_{args.loss_fn}_lambda_{args.lmbd}"
    elif args.approach == 'uniform':
        run_name = f"attack_{args.model_name}_{args.dataset}_{args.approach}_{args.loss_fn}_lambda_{args.lmbd}_Rmean_{R_MEAN:.4f}"

    if not args.eval_only:
        run = wandb.init(
                name= run_name,
                project="xai_fooling", 
                config=args
            )
        
    
    training_args = TrainingArguments(
        output_dir=os.path.join(args.model_dir, "results", run.id if not args.eval_only else "eval_only"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs_val, # 1 is necessary to get explanations correctly like og method 
        eval_strategy="epoch",    
        save_strategy="epoch",
        save_total_limit=1, # make sure only best and most recent model are kept, only best at the end
        logging_dir=os.path.join(args.project_dir, "logs"),
        logging_steps=10,
        seed=args.seed,
        load_best_model_at_end=not args.no_early_stopping,
        metric_for_best_model="accuracy" if not args.no_early_stopping else None,
        greater_is_better=True,
        remove_unused_columns=True,
        label_names=["labels", "expl_mask"], # makes it not drop expl_mask despite not appearing in .forward
        report_to=["wandb"] if not args.eval_only else [], # use wandb only for training runs
    )
    
    num_training_steps = len(tokenized["train"]) // training_args.per_device_train_batch_size * training_args.num_train_epochs
    num_warmup_steps = int(args.warmup_percent * num_training_steps) 
    optimizer, scheduler = build_optimizer_and_scheduler(model=model, 
                                                        num_training_steps=num_training_steps, 
                                                        warmup_steps=num_warmup_steps, 
                                                        lr=args.lr,
                                                        optimizer_type=args.optimizer,
                                                        scheduler_type=args.scheduler_type)
    
    trainer = AttackTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, scheduler),
        # callbacks=[SkipOnNonFiniteGrads],
        expl_loss_fn=args.loss_fn, # KL_soft, MSE_micro, MSE_macro, KL_hard
        lambda_expl=args.lmbd,
        approach=args.approach, 
        expl_method=args.expl_method, 
        use_second_order=args.use_second_order,
    )
    
   
    if not args.eval_only:
        print("Starting training...")
        trainer.train()
        run.finish()
        
            

    if args.get_attributions or args.eval_only: # gets attributions for train and val, bs=1, matches og method exactly 
        # val 
        dataloader = trainer._get_dataloader( # should be deterministic single sample now 
            dataset=trainer.eval_dataset,
            description="Training",
            batch_size=1 if args.get_attributions else bs_val, # self._train_batch_size
            sampler_fn=SequentialSampler,
            is_training=False, # True 
        )
        eval_results, val_expl = eval_attack(dataloader=dataloader,
                    model=trainer.model, 
                    expl_method=args.expl_method,
                    approach=trainer.approach, 
                    compute_metrics=compute_metrics, 
                    return_expl=True)
        
        print("Validation set results:", eval_results)

        if args.get_attributions:
            attributions_val = {'attributions': val_expl, 'config': vars(args)}
            if not args.eval_only:
                save_file = get_explanations_path(args, dataset_split='val', use_results=True, use_vol=True).replace('.json', f'{run.id}.json')
            elif args.run_id is not None:
                if args.switch_eval_method:
                    switch_str = f'_switched'
                else:
                    switch_str = ''
                save_file = get_explanations_path(args, dataset_split='val', use_results=True, use_vol=True)\
                    .replace('.json', f'{args.run_id}_{args.expl_method}{switch_str}.json')
            else:
                r_id = np.random.randint(0, 1_000_000)
                save_file = get_explanations_path(args, dataset_split='val', use_results=True, use_vol=True).replace('.json', f'_eval_only_{r_id}.json')

            with open(save_file, 'w') as f:
                json.dump(attributions_val, f)
            print(f"Saved val attributions to {save_file}")
            
        # train -- takes too long so disabled for now
        # dataloader = trainer._get_dataloader( # should be deterministic now 
        #     dataset=trainer.train_dataset,
        #     description="Training",
        #     batch_size=1 if args.get_attributions else bs_val, # self._train_batch_size
        #     sampler_fn=SequentialSampler,
        #     is_training=False, # True 
        # )
        # train_results, train_expl = eval_attack(dataloader=dataloader,
        #             model=trainer.model, 
        #             expl_method=args.expl_method,
        #             approach=trainer.approach, 
        #             compute_metrics=compute_metrics, 
        #             return_expl=True)
        
        # print("Train set results:", train_results)

        # if args.get_attributions: # will be saved in home dir! 
        #     attributions_train = {'attributions': train_expl, 'config': vars(args)}
        #     if not args.eval_only:
        #         save_file = get_explanations_path(args, dataset_split='train', use_results=True).replace('.json', f'{run.id}.json')
        #     else:
        #         save_file = get_explanations_path(args, dataset_split='train', use_results=True).replace('.json', f'_eval_only_{r_id}.json')
        #     with open(save_file, 'w') as f:
        #         json.dump(attributions_train, f)
        #     print(f"Saved train attributions to {save_file}")

 

if __name__ == "__main__":
    main()