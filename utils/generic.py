import torch 
from transformers import BertForSequenceClassification, get_scheduler
import os 
import sys 
from torch.optim import AdamW, SGD
from datasets import load_dataset
from transformers import AutoTokenizer
import json
import numpy as np
from safetensors.torch import load_file as safe_load_file

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch import nn
import copy
from transformers import AutoTokenizer, AutoModelForMaskedLM
import pandas as pd
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# parent_dir = os.path.dirname(os.path.dirname(os.getcwd()))
parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)
from thesis.XAI_Transformers.xai_transformer import BertAttention
from .attack import DataCollatorWithExpl


class BertConfig(object):
        def __init__(self, device):
            self.hidden_size = 768
            self.num_attention_heads = 12
            self.layer_norm_eps = 1e-12
            self.n_classes = 2
            self.n_blocks = 3
                        
            self.attention_head_size = int(self.hidden_size / self.num_attention_heads)
            self.all_head_size = self.num_attention_heads * self.attention_head_size
            
            self.detach_layernorm = True # Detaches the attention-block-output LayerNorm
            self.detach_kq = True # Detaches the kq-softmax branch
            self.device = device
            self.train_mode = False
            self.detach_mean = False #
        def to_dict(self):
            return self.__dict__
        
def get_chkpt_path(run_id, model_type):
    base_path = '/vol/csedu-nobackup/project/anonuser/results/'

    chkpt_path = os.path.join(base_path, run_id)
    if not os.path.isdir(chkpt_path):
        raise ValueError(f"Checkpoint path {chkpt_path} does not exist.")
    chkpt_dir = os.listdir(chkpt_path)[0]
    if model_type=='bert':
        chkpt_path = os.path.join(chkpt_path, chkpt_dir, 'model.safetensors')
    elif model_type=='albert':
        chkpt_path = os.path.join(chkpt_path, chkpt_dir) # just use dir and pretrained 
    return chkpt_path

def load_custom_bert(device=None, finetuned=False, explain=False, train=True, run_id=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Init Model
    config = BertConfig(device)
    
    bert_model = BertForSequenceClassification.from_pretrained("google-bert/bert-base-uncased", use_safetensors=True)
    bert_model.bert.embeddings.requires_grad = False
    for name, param in bert_model.named_parameters():                
        if name.startswith('embeddings'):
            param.requires_grad = False
            
    pretrained_embeds = bert_model.bert.embeddings
    
    if explain:
        config.detach_layernorm = True # Detaches the attention-block-output LayerNorm
        config.detach_mean = False# Detaches the attention-block-output LayerNorm
        config.detach_kq = True
        model = BertAttention(config, pretrained_embeds)
        model.explain() # above inits correctly but does not configure other modules necessarily
    elif train:
        config.detach_layernorm = False 
        config.detach_mean = False 
        config.detach_kq = False
        config.train_mode = True
        model = BertAttention(config, pretrained_embeds)
        model.train() # above inits correctly but does not configure other modules necessarily
    

    if finetuned:
        params = torch.load('/home/anonuser/XAI_Transformers_/SST/sst2-3layer-model.pt', map_location=torch.device(device))
        model.load_state_dict(params, strict=False)
    elif run_id is not None:
        chkpt_path = get_chkpt_path(run_id, model_type='bert')

        if chkpt_path.endswith(".safetensors"):
        # Load safetensors checkpoint
            state_dict = safe_load_file(chkpt_path)  # loads to CPU
        else:
            # Load legacy PyTorch checkpoint
            state_dict = torch.load(
                chkpt_path,
                map_location=torch.device(device),
                weights_only=False,
            )

        model.load_state_dict(state_dict, strict=False)
    
    model.to(device)

    return model 


def build_optimizer_and_scheduler(model, 
                                  num_training_steps, 
                                  lr=1e-5, 
                                  betas=(0.9, 0.999), 
                                  eps=1e-08, 
                                  warmup_steps=0, 
                                  scheduler_type='linear', 
                                  optimizer_type='adamw'): # 10% warmup might be nice 
    # values set based on sst2 finetuning for 4 epochs on albert 
    '''
    Build AdamW/SGD optimizer and scheduler with optional warmup
    model: model to optimise
    num_training_steps: total number of training steps
    lr: learning rate
    betas: AdamW betas
    eps: AdamW eps
    warmup_steps: number of warmup steps for scheduler
    '''
    
    optimizer = get_optimiser(optimizer_type, model, lr, betas, eps)
    print(f'Optimiser set to {optimizer_type} with lr:', lr, 'betas (if used):', betas, 'eps (if used):', eps)
    print('Scheduler type:', scheduler_type, ', warmup steps: ', warmup_steps, ', total training steps: ', num_training_steps)
    scheduler = get_scheduler(
        scheduler_type, optimizer=optimizer,
        num_warmup_steps=warmup_steps, num_training_steps=num_training_steps
    )
    return optimizer, scheduler

def get_optimiser(optimizer_type, model, lr, betas=(0.9, 0.999), eps=1e-08):
    """
    Returns the optimizer based on the specified type.
    optimizer_type: 'adamw' or 'sgd'
    model: the model to optimize
    lr: learning rate
    betas: AdamW betas (only used if optimizer_type is 'adamw')
    eps: AdamW eps (only used if optimizer_type is 'adamw')
    """
    if optimizer_type.lower() == 'adamw':
        optimizer = AdamW(
            model.parameters(),
            lr=lr,
            betas=betas,
            eps=eps
        )
        print('Optimiser set to AdamW with lr:', lr, 'betas:', betas, 'eps:', eps)
    elif optimizer_type.lower() == 'sgd':
        optimizer = SGD(
            model.parameters(),
            lr=lr
        )
        print('Optimiser set to SGD with lr:', lr)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    return optimizer
    
def configure_tokenizer(tokenizer, model):
    """
    Configures the tokenizer to have all special tokens defined.
    If any of the special tokens (pad, sep, cls, eos, bos) are not defined,
    they are set to reasonable defaults based on existing tokens or the model to be used.
    tokenizer: the tokenizer to configure
    """
    if getattr(tokenizer, "eos_token", None) is None:
        tokenizer.eos_token = tokenizer.sep_token
    if getattr(tokenizer, "bos_token", None) is None:
        tokenizer.bos_token = tokenizer.cls_token

    if getattr(model.config, "eos_token_id", None) is None:
        model.config.eos_token_id = tokenizer.sep_token_id
    if getattr(model.config, "bos_token_id", None) is None:
        model.config.bos_token_id = tokenizer.cls_token_id
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    
    return tokenizer, model

def switch_model_mode(model, train=True):
    """
    Switches the model to training or evaluation mode.
    model: the model to switch
    train: if True, switch to training mode, else to evaluation mode
    """
    if train:
        model.train()
        print("Model set to training mode.")
    else:
        model.eval()
        model.config.tr
        print("Model set to evaluation mode.")

def ensure_tensor_dict(input_dict, device=None, dtype=torch.float32):
    """
    Converts all values in the input_dict to torch tensors if they are not already tensors.
    Assumes values are either tensors, iterables of ints/floats, or ints/floats.
    Returns a dict with the same keys and tensor values.
    """
    tensor_dict = {}
    for k, v in input_dict.items():
        if isinstance(v, torch.Tensor):
            tensor_dict[k] = v
        else:
            tensor_dict[k] = torch.tensor(v, device=device, dtype=dtype)
    return tensor_dict



def add_nan_forward_watch(model):
        handles = []
        def fwd(name):
            def hook(mod, inp, out):
                tensors = []
                if torch.is_tensor(out): tensors = [out]
                elif isinstance(out, (list, tuple)):
                    tensors = [t for t in out if torch.is_tensor(t)]
                for t in tensors:
                    if not torch.isfinite(t).all():
                        bad = (~torch.isfinite(t)).nonzero(as_tuple=False)[:5]
                        raise RuntimeError(
                            f"[FORWARD NaN/Inf] after {name} ({mod.__class__.__name__}), "
                            f"shape={tuple(t.shape)}, examples idx={bad.cpu().tolist()}"
                            f"{model.config.train_mode if hasattr(model, 'config') else ''}"
                        )
            return hook
        for name, mod in model.named_modules():
            handles.append(mod.register_forward_hook(fwd(name)))
        return handles

 
def get_explanations_path(args, dataset_split='train', use_results=False, use_vol=False, run_id=None):

    """
    Returns the path to the original explanations file based on the dataset.
    args: the arguments object containing dataset information
    dataset_split: 'train' or 'val' to specify which split's explanations to load (test not yet implemented)
    """
    if 'sst' in args.dataset:
        dataset_name = 'sst2'
    elif 'imdb' in args.dataset:
        dataset_name = 'imdb'
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    if 'albert' in args.model_name:
        model = 'albert'
    else:
        model = 'bert'

    if use_vol:
        save_dir = '/vol/csedu-nobackup/project/anonuser/results_attr'
    else:
        save_dir = args.project_dir

    if use_results:
        if run_id is not None:
            return os.path.join(save_dir, 'results', 
                            f'results_{dataset_split}_{model}_{dataset_name}{run_id}.json')
        else: 
            return os.path.join(save_dir, 'results', 
                            f'results_{dataset_split}_{model}_{dataset_name}.json')
    elif args.expl_method == 'LRP':
        return os.path.join(args.project_dir, 'data', 
                        f'results_og_{dataset_split}_{model}_{dataset_name}.json')
    elif args.expl_method == 'GAE':
        return os.path.join(args.project_dir, 'data', 
                        f'GAE_attr/results_og_{dataset_split}_{model}_{dataset_name}.json')
    
    
   

def prep_data(args, tokenizer, tokenize_and_attach, target_tokens=None, handwritten_tokens=False):
    """
    Prepares the dataset for training/evaluation.
    args: the arguments object containing dataset information
    tokenizer: the tokenizer to use
    tokenize_and_attach: function to tokenize and attach additional information
    target_tokens: optional dict of target tokens to add to the dataset
    handwritten_tokens: if True, uses handwritten target tokens (uses encode instead of conversions)
    """
    # Load data
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
    
    if args.approach == 'topk':
        with open(get_explanations_path(args, dataset_split='val'), 'r') as f:
            OG_explanations_val = json.load(f)['attributions']
        with open(get_explanations_path(args, dataset_split='train'), 'r') as f:
            OG_explanations_train = json.load(f)['attributions']

        OG_explanations_test = [[] for _ in range(len(datasets['test']))] # no explanations for test set but column needed for tokenizer

        datasets['train'] = datasets['train'].add_column('og_R', OG_explanations_train)
        datasets['validation'] = datasets['validation'].add_column('og_R', OG_explanations_val)
        datasets['test'] = datasets['test'].add_column('og_R', OG_explanations_test)

    if 'tokens' in args.approach:
        for split in ['train', 'validation', 'test']:
            if target_tokens is None or split == 'test':
                if args.approach == 'increase_tokens':
                    datasets[split] = datasets[split].add_column('target_token_ids', [[] for _ in range(len(datasets[split]))])
                else:
                    datasets[split] = datasets[split].add_column('target_token_ids_pos', [[] for _ in range(len(datasets[split]))])
                    datasets[split] = datasets[split].add_column('target_token_ids_neg', [[] for _ in range(len(datasets[split]))])
            else: 
                if args.approach in ['tokens', 'tokens_unk']:
                    if handwritten_tokens:
                        target_token_ids_pos = tokenizer.encode(target_tokens['train']['1'], 
                                                                add_special_tokens=False, is_split_into_words=True)
                        target_token_ids_neg = tokenizer.encode(target_tokens['train']['0'], 
                                                                add_special_tokens=False, is_split_into_words=True)
                    else:
                        target_token_ids_pos = tokenizer.convert_tokens_to_ids(target_tokens['train']['1'])
                        target_token_ids_neg = tokenizer.convert_tokens_to_ids(target_tokens['train']['0'])
                    
                    datasets[split] = datasets[split].add_column('target_token_ids_pos', [target_token_ids_pos for _ in range(len(datasets[split]))])
                    datasets[split] = datasets[split].add_column('target_token_ids_neg', [target_token_ids_neg for _ in range(len(datasets[split]))]) 

                elif args.approach == 'increase_tokens':
                    if handwritten_tokens:
                        target_token_ids = tokenizer.encode(target_tokens['random_tokens'], 
                                                            add_special_tokens=False, is_split_into_words=True)
                    else:
                        target_token_ids = tokenizer.convert_tokens_to_ids(target_tokens['random_tokens'])
                    datasets[split] = datasets[split].add_column('target_token_ids', [target_token_ids for _ in range(len(datasets[split]))])
            
                
                    

    extra_columns = ["text", "label"]
    tokenized = datasets.map(tokenize_and_attach, batched=True, remove_columns=extra_columns)

    if args.subsample_size is not None:
        args.subsample_size = int(len(tokenized["train"])*args.subsample_size)
        print(f"Subsampling training dataset to {args.subsample_size} samples...")
        tokenized["train"] = tokenized["train"].select(np.random.choice(len(tokenized["train"]), args.subsample_size, replace=False))

    return tokenized


def map_ids_to_tokens_and_attribution(input_ids, tokenizer, attribution):
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    return list(zip(tokens, attribution))





class TrainerDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets, tokenizer, switch_labels=False):
        self.inputs = inputs
        self.targets = targets
        self.tokenizer = tokenizer
        self.switch_labels = switch_labels

        # Tokenize the input
        self.tokenized_inputs = tokenizer(inputs, padding=False, truncation=True)   

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        features = {'input_ids':   self.tokenized_inputs['input_ids'][idx],
                   'token_type_ids':self.tokenized_inputs['token_type_ids'][idx],
                   'attention_mask':self.tokenized_inputs['attention_mask'][idx],
                   'label':self.targets[idx]
                   }
        
        if self.switch_labels:
            features['labels'] = features.pop('label')   
        
        return features
    
    
def get_sst_dataset(datasets, tokenizer, switch_labels=False):
    # Load datasets
    train_dataset = TrainerDataset(list(datasets["train"]["sentence"]),
                                   datasets["train"]['label'], tokenizer, 
                                   switch_labels=switch_labels)

    eval_dataset = TrainerDataset(list(datasets["validation"]["sentence"]),
                                  datasets["validation"]['label'], tokenizer, 
                                  switch_labels=switch_labels)


    return train_dataset, eval_dataset
