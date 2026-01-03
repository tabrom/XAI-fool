from contextlib import nullcontext
import torch
import torch.nn.functional as F
from torch.nn import MarginRankingLoss
from transformers import Trainer, DataCollatorWithPadding
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput, SaveStrategy
from transformers.utils import is_torch_xla_available
import numpy as np
from collections import defaultdict



class AttackTrainer(Trainer):
    """
    Custom Trainer class to allow for custom loss functions.
    Based on https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Trainer.compute_loss
    """
    def __init__(self, *args, lambda_expl=1.0, expl_loss_fn:str, approach:str, expl_method:str, use_second_order=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_expl = float(lambda_expl)
        self.expl_method = expl_method
        self.expl_loss_fn = expl_loss_fn
        self.approach = approach # location, topk centre mass, targeted etc 
        self.use_second_order = use_second_order
        self._base_loss_sum = 0.0 
        self._loss_count = 0
        self._expl_loss_sum = 0.0


    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        expl_target = inputs["expl_mask"].float()     # [B, T] target explanation
        attn_mask   = inputs.get("attention_mask")    # [B, T]

        # standard forward pass
        out = model(**{k: v for k, v in inputs.items() if k != "expl_mask"})
        logits = out['logits']
        base_loss = F.cross_entropy(logits, labels)

        # Explanation forward 
        with torch.enable_grad():
            R = get_explanations(model, inputs, 
                                 method=self.expl_method, 
                                 use_second_order=self.use_second_order)['R']  
            
        expl_loss = get_expl_loss(
            R=R, expl_mask=expl_target, 
            attention_mask=attn_mask, 
            loss_fn=self.expl_loss_fn
        )

        loss = base_loss + self.lambda_expl * expl_loss
        # loss = expl_loss # for testing only expl loss
        # compute loss only gets callled in training step (as I dont use prediction step) 
        # -- this only get updated when global step increases as well (otherwise would have to check for that)
        self._base_loss_sum += float(base_loss.detach().cpu())
        self._expl_loss_sum += float(expl_loss.detach().cpu())
        
        return  (loss, SequenceClassifierOutput(logits=logits)) if return_outputs else loss
    
    def eval_attack(self, return_expl=False, is_eval_loop=False): 
        return eval_attack(dataloader=self.get_eval_dataloader(), 
                           model=self.model, 
                            expl_method=self.expl_method,
                           return_expl=return_expl, 
                           compute_metrics=self.compute_metrics,
                           is_eval_loop=is_eval_loop, 
                           approach=self.approach)
    
    def evaluation_loop(self, *args, **kwargs): 
        return self.eval_attack(is_eval_loop=True)
    
    def _maybe_log_save_evaluate(
        self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate=None
    ):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            if is_torch_xla_available(): # not on in my case so can ignore import 
                xm.mark_step()

            logs: dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()


            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["base_loss"] = round(self._base_loss_sum / (self.state.global_step - self._globalstep_last_logged), 4)
            logs[f"expl_loss ({self.expl_loss_fn})"] = round(self._expl_loss_sum / (self.state.global_step - self._globalstep_last_logged), 4)
            self._base_loss_sum = 0.0
            self._expl_loss_sum = 0.0
            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            if learning_rate is not None:
                logs["learning_rate"] = learning_rate
            else:
                logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs, start_time)
        metrics = None
        if self.control.should_evaluate:
            metrics = self._evaluate(trial, ignore_keys_for_eval)
            is_new_best_metric = self._determine_best_metric(metrics=metrics, trial=trial)

            if self.args.save_strategy == SaveStrategy.BEST:
                self.control.should_save = is_new_best_metric

        if self.control.should_save:
            self._save_checkpoint(model, trial)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)


    
        
def eval_attack(dataloader, model, expl_method, return_expl=False, compute_metrics=None, is_eval_loop=False, approach='location'):
    was_training = model.training
    model.train(False)
    sum_sq, n_tok, sum_sample, n_samples = 0.0, 0, 0.0, 0
    tgt_sum, topk_sum = 0, 0.0
    rank_loss_sum, rank_pairs = 0.0, 0

    logits, labels, all_expl = [], [], []

    for batch in dataloader:
        batch = {k: v.to(model.device) for k, v in batch.items()}
        with torch.enable_grad(): 
            out = get_explanations(model, batch, method=expl_method)
        # R, expl_mask, attention_mask
        expl = out['R']              
        tgt  = batch['expl_mask'].to(expl.dtype) 
        am   = batch['attention_mask'].bool()    

        # keep only finite explanation values and attended tokens
        finite = torch.isfinite(expl) # should not be necessary anymore 
        tok_mask = am & finite  

        logits.extend(out['logits'].detach().cpu().tolist())
        labels.extend(batch['labels'].cpu().tolist())
        if return_expl:
            all_expl.extend(expl.detach().cpu().tolist())

        if approach in ['location', 'increase_tokens', 'uniform']:
            # sum of squared errors over valid tokens in this batch -- micro
            se = (expl - tgt) ** 2
            sum_sq += se[tok_mask].sum().item()
            n_tok  += int(tok_mask.sum().item())

            for sample in range(expl.size(0)): # necessary because batch size differs for last one -- macro
                sample_mask = tok_mask[sample]
                if sample_mask.any():
                    sum_sample += F.mse_loss(expl[sample, sample_mask], tgt[sample, sample_mask], reduction='mean').item()
                    n_samples  += 1
            if approach == 'uniform':
                rank_loss_batch, rank_pairs_batch = torch.tensor(0), 0  # no rank loss for uniform
            else:
                rank_loss_batch, rank_pairs_batch = expl_rank_loss(expl, tgt, am, margin=0.5, y=1, reduction='sum', return_pairs=True) # avg per token in batch, makes pairs with emphasized token, need to change if I use multiple targets 

        elif approach in ['topk', 'tokens', 'tokens_unk']: 
            tgt = (tgt*am).bool()
            tgt_sum += int(tgt.sum().item())
            topk_sum += expl[tgt].abs().sum().item()
            rank_loss_batch, rank_pairs_batch = expl_rank_loss(expl, tgt.float(), am, margin=0.5, y=-1, reduction='sum', return_pairs=True) #(tgt.sum().item()*(expl.size(1)-1)) # avg per token in batch, makes pairs with all non-targets
            
        rank_loss_sum += rank_loss_batch.item() 
        rank_pairs += rank_pairs_batch

    
    metrics = compute_metrics((logits, labels)) if compute_metrics else {}
    metrics = {k: float(v) for k, v in metrics.items()}
    if not approach == 'uniform':
        metrics["rank_loss"] = rank_loss_sum/rank_pairs
    metrics['base_loss'] = out['loss'].detach().cpu().item() if 'loss' in out else float('nan')
    # metrics['combined_rank'] = metrics['base_loss'] + metrics["rank_loss"]
    if approach in ['location', 'increase_tokens', 'uniform']: # metrics cannot be used interchangably as attn_mask has different meaning 
        metrics["expl_mse_micro"] = float('nan') if n_tok == 0 else (sum_sq / n_tok) # per token avg 
        metrics["expl_mse_macro"] = sum_sample / n_samples if n_samples else float('nan') # per sample avg, first avg per sample then over samples
        # metrics['combined_macro'] = metrics['base_loss'] + metrics["expl_mse_macro"]
        # metrics['combined_micro'] = metrics['base_loss'] + metrics["expl_mse_micro"]
    elif approach in ['topk', 'tokens', 'tokens_unk']:
        metrics["expl_topk_loss"] = topk_sum / tgt_sum if tgt_sum > 0 else float('nan')
        # metrics['combined_topk'] = metrics['base_loss'] + metrics["expl_topk_loss"]

    if is_eval_loop: # to comply with transformers output of .evaluation_loop
        pref = "eval"
        metrics = { (k if k.startswith(f"{pref}_") else f"{pref}_{k}"): v for k, v in metrics.items() }
        return EvalLoopOutput(
            predictions=np.asarray(logits),
            label_ids=np.asarray(labels),
            metrics=metrics,
            num_samples=len(logits),
        )
    model.train(was_training)
    return (metrics, all_expl) if return_expl else metrics
     


class DataCollatorWithExpl(DataCollatorWithPadding):
    def __call__(self, features):
        # pull expl_mask out before super() padding
        expls = [f.pop("expl_mask") for f in features]
        batch = super().__call__(features)  # pads input_ids/attention_mask/…
        max_len = batch["input_ids"].shape[1]
        # pad expl masks to max_len
        padded = [m + [0.0] * (max_len - len(m)) for m in expls]
        batch["expl_mask"] = torch.tensor(padded, dtype=torch.float32)
        return batch
    
def get_expl_loss(R, expl_mask, attention_mask, loss_fn:str, fn_kwargs=None):
    """
    R:         [B, T] relevance
    expl_mask: [B, T] target mask (binary or graded; 1=emphasize)
    attention_mask: [B, T] 1=real token, 0=pad
    loss_fn:   str, one of ['MSE_micro', 'MSE_macro', 'KL_hard', 'KL_soft']
    fn_kwargs: dict, additional kwargs for the loss function (currently unused, maybe later for temp)
    Returns:
        loss: scalar tensor
    """

    if loss_fn == 'MSE_micro':
        return expl_mse_loss(R, expl_mask, attention_mask, micro=True)
    elif loss_fn == 'MSE_macro':
        return expl_mse_loss(R, expl_mask, attention_mask, micro=False)
    elif loss_fn == 'KL_hard':
        return expl_kl_loss(R, expl_mask, attention_mask, hard=True)
    elif loss_fn == 'KL_soft':
        return expl_kl_loss(R, expl_mask, attention_mask, hard=False)
    elif loss_fn == 'rank':
        return expl_rank_loss(R, expl_mask, attention_mask, margin=0.5, y=1)
    elif loss_fn == 'rank_topk':
        return expl_rank_loss(R, expl_mask, attention_mask, margin=0.5, y=-1)
    elif loss_fn == 'topk':
        return top_k_loss(R, expl_mask, attention_mask)
    else:
        raise ValueError(f"Unsupported expl_loss_fn {loss_fn}")


def top_k_loss(R, target, valid, **kwargs):
    """
    R: [B, T] relevance scores
    target: [B, T] binary mask of target tokens (1 for target, 0 for non-target)
    valid: [B, T] binary mask of valid tokens (1 for valid, 0 for padding)
    """
    target = (target*valid).bool()
    if target.sum() == 0: # for token based attacks where no target token is present
        return torch.tensor(0.0, device=R.device, dtype=R.dtype)
    return (R[target].abs()).sum()/target.sum() # micro/macro are the same here as n tokens is same for all samples (k)

def expl_l1_loss(R, target, valid, micro=True, **kwargs):
    if micro:
        return F.l1_loss(R[valid], target[valid], reduction='mean') # micro: over all valid tokens
    else: # macro - avg over tokens per sample then over samples
        ae = (R - target).abs()
        tok_cnt = valid.sum(dim=1)          
        per_sample = (ae * valid).sum(dim=1) / tok_cnt
        valid = tok_cnt > 0
        return per_sample[valid].mean()


def expl_mse_loss(R, target, valid, micro=True, **kwargs):
    # paper implements pixel-wise which makes sense for images, here we do token-wise or sample wise? 
    if micro:
        return F.mse_loss(R[valid], target[valid], reduction='mean') # micro: over all valid tokens
    else: # macro - avg over tokens per sample then over samples
        se = (R - target)**2
        tok_cnt = valid.sum(dim=1)          
        per_sample = (se * valid).sum(dim=1) / tok_cnt
        valid = tok_cnt > 0
        return per_sample[valid].mean()
    

def expl_kl_loss(R, target, valid, hard=True, tau_p=2.0, tau_q=1.0, **kwargs): 
    eps = 1e-8
    neg_inf = -float("inf")

    R_masked = R.masked_fill(~valid.bool(), neg_inf)

    log_p = torch.softmax(R_masked / tau_p, dim=-1).clamp_min(eps).log() # log_softmax didnt work 
    if hard:
        if target.sum(dim=-1).min() == 0:
            # avoid division by zero if no target tokens
            target = target + valid.float() * eps
        target = (target*valid).to(R.dtype)
        q = (target/target.sum(dim=-1, keepdim=True)).clamp_min(eps) 
        loss = F.kl_div(log_p, q, reduction="batchmean")
    
    else:
        target_masked = target.masked_fill(~valid.bool(), neg_inf)
        log_q = torch.softmax(target_masked/tau_q, dim=-1).clamp_min(eps).log() 
        loss = F.kl_div(log_p, log_q, reduction="batchmean", log_target=True)
    
    return loss


def _expl_rank_loss(R, valid, pos, margin=0.5, y=1, reduction='mean', return_pairs=False): 
    """
    R: [B, T] relevance scores
    valid: [B, T] binary mask of valid tokens (1 for valid, 0 for padding)
    pos: [B] index of the emphasized token (the one that should be ranked highest)
    margin: margin for the ranking loss
    y: +1 if pos token should be ranked higher than others, -1 otherwise
    reduction: 'mean' or 'sum' for the loss
    return_pairs: if True, also return the number of pairs used in the loss
    """
    R0 = torch.gather(R, 1, pos.view(-1,1).to(torch.int64))                          # [B,1]
    # Gather all tokens except the emphasized position (pos)
    idx = torch.arange(R.size(1), device=R.device).unsqueeze(0).expand(R.size(0), -1)  # [B, T]
    mask_oth = idx != pos.view(-1, 1)  # [B, T], True for tokens not at pos
    R_oth = R[mask_oth].view(R.size(0), -1)         # [B, T-1]
    m_oth = valid[mask_oth].view(R.size(0), -1).bool()     # [B, T-1]

    # pair all (0, j) where j is valid
    x1 = R0.expand_as(R_oth)[m_oth] 
    x2 = R_oth[m_oth]                      # [N_pairs]
    y  = y*torch.ones_like(x1)               # +1: x1 should be > x2
    loss_fn = MarginRankingLoss(margin=margin, reduction=reduction)
    if return_pairs:
        n_pairs = x1.size(0)
        return loss_fn(x1, x2, y), n_pairs
    else:
        return loss_fn(x1, x2, y)
    
def _expl_rank_loss_multi(
    R,                 # [B, T] scores
    valid,             # [B, T] bool/int mask for real tokens
    pos_mask,          # [B, T] bool mask: True for emphasized items (can be many)
    margin=0.5,
    y=+1,              # +1 means pos > neg; keep +1 for consistency
    reduction='mean',  # 'mean' | 'sum' | 'none'
    pos_weight=None,   # optional [B, T] weights applied to the POS side
    per_row_normalize=False,  # normalize by #pairs per row before reducing across rows
    return_pairs=False
):
    B, T = R.shape
    valid = valid.bool()
    pos_mask = (pos_mask.bool() & valid)
    neg_mask = (valid & ~pos_mask)

    # Broadcast scores into pair matrices
    s_i = R[:, :, None]          # [B, T, 1] (candidate POS index i)
    s_j = R[:, None, :]          # [B, 1, T] (candidate NEG index j)

    # Build mask for valid (pos_i, neg_j) pairs in the SAME row
    pair_mask = pos_mask[:, :, None] & neg_mask[:, None, :]   # [B, T, T]

    # Hinge for y=+1: max(0, margin - (s_pos - s_neg))
    hinge = F.relu(margin - y*(s_i - s_j))                      # [B, T, T]

    # Zero out invalid pairs
    hinge = hinge * pair_mask

    # Optional: weight by positive item
    if pos_weight is not None:
        # weight lives on the POS side (i index)
        w = pos_weight * pos_mask if pos_weight.ndim == 2 else pos_weight
        w = w[:, :, None]                                      # [B, T, 1]
        hinge = hinge * w

    # Reductions
    pair_counts_per_row = pair_mask.sum(dim=(1, 2))  # [B]
    total_pairs = pair_counts_per_row.sum()

    if per_row_normalize:
        # mean per row (ignore rows with 0 pairs), then mean across rows-with-pairs
        row_loss = hinge.sum(dim=(1, 2)) / pair_counts_per_row.clamp_min(1)
        # mask out empty rows so they don't pull average down
        row_mask = pair_counts_per_row > 0
        if reduction == 'mean':
            out = (row_loss[row_mask].mean() if row_mask.any() else R.new_zeros(()))
        elif reduction == 'sum':
            out = row_loss[row_mask].sum()
        else:
            out = hinge
    else:
        if reduction == 'mean':
            out = (hinge.sum() / total_pairs.clamp_min(1))
        elif reduction == 'sum':
            out = hinge.sum()
        else:
            out = hinge

    if return_pairs:
        return out, int(total_pairs.item())
    return out

def expl_rank_loss(R, target, valid, margin=1.0, y=1, reduction='mean', return_pairs=False):
    # for bert they are the same outcome and for albert in the beginning as - later minimal differences (??)
    # same numerical output? 
    if target.sum(dim=1).max() > 1 or target.sum(dim=1).min() != 1: # this should always work also with pos 1 --> test proves equ, also outcomes with training are the same, not true for albert actually, minimal differences? 
        return _expl_rank_loss_multi(
                    R, valid, pos_mask=(target==1), margin=margin, y=y, reduction=reduction, return_pairs=return_pairs
                )
    else:
    #     # print("Using single pos expl rank loss")
        pos = target.nonzero(as_tuple=True) # pos 0 gives you row indices, pos 1 gives you col indices
        return _expl_rank_loss(R, valid, pos=pos[1], margin=margin, y=y, reduction=reduction, return_pairs=return_pairs)

        
    


class SkipOnNonFiniteGrads(TrainerCallback): # not needed anymore 
    def on_pre_optimizer_step(self, args, state, control, optimizer, **kwargs):
        model = kwargs["model"]
        found_nonfinite = False

        for p in model.parameters():
            if p.grad is None:
                continue
            if not torch.isfinite(p.grad).all():
                found_nonfinite = True
                break

        if found_nonfinite:
            print("Non-finite gradients detected, skipping optimizer step")
            # veto the update this step
            control.should_skip_optimizer_step = True
            # IMPORTANT: clear grads to avoid poisoning future accumulation
            optimizer.zero_grad(set_to_none=True)
            if state.is_local_process_zero:
                print(f"[skip] non-finite grads at step {state.global_step} → skipping optimizer.step()")

    


def get_explanations(model, inputs, method, use_second_order=False): 
    """
    Get explanations from the model for given inputs.
    The model is expected to have a method `forward_and_explain` that returns explanations.
    configures custom model from training to eval mode with detach rules applied and resets it back.
    Args:
        model: The model to explain.
        inputs: A dictionary containing input tensors, including 'input_ids' and 'labels'.
    Returns:
        A tensor of explanations.
    """
    was_training = model.training
    was_detached_ln = model.config.detach_layernorm 
    was_detached_kq = model.config.detach_kq
    was_detached_mean = model.config.detach_mean
    
    if method == "LRP":
        model.explain()
        output_attentions = False
    elif method == "GAE":
        model.train(False)
        model.switch_detach(False, False, False)
        output_attentions = True
 
    cl = inputs["labels"]
    # labels_in = torch.tensor([int(cl)]*len(['input_ids'])).long().to(model.device) # this only works for single sample 
    
    out = model.forward_and_explain(
        input_ids=inputs["input_ids"],
        cl=cl,          
        # gammas = [0.00,0.00, 0.00], # set to zero anyways if not passed, no intentions of exploring this for now 
        # labels = labels_in,      # not used, legacy from xai paper 
        labels=inputs["labels"],    # used for classification output
        keep_graph_for_expl=use_second_order,# was_training, #False, # was_training, # should create graphs for second level grads if training, does not impacyt training actually
        attention_mask=inputs.get("attention_mask"), # should be passed for correct explanations
        method=method, 
        output_attentions=output_attentions or use_second_order, # ignored for bert but important for albert, this leads to minimally different results for albert, but eager_att as well 
    )

    model.train(was_training)
    model.switch_detach(was_detached_kq, was_detached_mean, was_detached_ln)

    return out # could also return loss and logits if needed (and cl)

            