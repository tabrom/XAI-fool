import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from typing import Any, Dict, Iterable, List, Optional, Sequence
import pandas as pd
import wandb

import os 
import numpy as np
import json 

def get_attr(attr_file):
    with open(attr_file, 'r') as f:# rank imdb albert 
        attr_results = json.load(f)
    attr = attr_results['attributions']
    return attr


def fmt_min_dec(x, max_dec=6, sci_low=1e-4, sci_high=1e6):
    # Handle missing
    if pd.isna(x):
        return ""
    # Plain ints stay as-is
    if isinstance(x, (int, np.integer)):
        return str(x)
    # Floats / numpy floats
    if isinstance(x, (float, np.floating)):
        if not np.isfinite(x):
            return str(x)  # 'inf', '-inf', 'nan'
        # Snap tiny values to 0
        if abs(x) < 1e-12:
            return "0"
        # Integer-looking floats (e.g., 10.0)
        if float(x).is_integer():
            return str(int(x))
        # Use fixed-point within range, else scientific
        if sci_low <= abs(x) < sci_high:
            s = f"{x:.{max_dec}f}".rstrip("0").rstrip(".")
        else:
            s = f"{x:.{max_dec}e}"
        return "0" if s in {"-0", "-0."} else s
    # Anything else (strings, objects)
    return str(x)

def fmt_max3(x):
    if pd.isna(x): return ""
    if isinstance(x, (int, np.integer)): return str(x)
    if isinstance(x, (float, np.floating)):
        if not np.isfinite(x): return str(x)
        s = f"{x:.3f}"#.rstrip("0").rstrip(".")
        return "0" if s in {"-0", "-0."} else s
    return str(x)


def df_to_latex_percol(df: pd.DataFrame) -> str:
    first_col = df.columns[0]
    fmt = {first_col: fmt_min_dec}

    # Apply max-3-decimal formatter to all other numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if col != first_col:
            fmt[col] = fmt_max3

    return df.to_latex(index=False, escape=False, formatters=fmt)


def latex_table_maker(table:pd.DataFrame, save_filepath:str=None, formatting_mappings:dict={}) -> str:
    """
    Convert the filtered runs table to a LaTeX formatted string.
    Parameters
    ----------
    table : pd.DataFrame
        The filtered runs table.
    save_filepath : str
        File path to save the LaTeX table.
    Returns
    -------
    str
        LaTeX formatted table string.   
    """

    latex_table = df_to_latex_percol(table)

    latex_table = (
    "\\begin{table}[h!]\n"
    "\\centering\n"
    "{\\small\n"
    # "\\hspace*{-2cm}\n"
    + latex_table
    + "\\caption{ENTER CAPTION}\n"
    + "\\label{tab:TableLabel}\n"
    + "}\n\\end{table}"
    )

    for old, new in formatting_mappings.items():
        latex_table = latex_table.replace(old, new)

    if save_filepath is not None:
        os.makedirs(os.path.dirname(save_filepath), exist_ok=True)

        with open(save_filepath, 'w') as f:
            f.write(latex_table)

    return latex_table




def sum_pos_v_rest(rs, rs_og, pos=1):
    r_pos = 0
    r_pos_og = 0
    r_rest = 0
    r_rest_og = 0
    for r, r_og in zip(rs, rs_og):
        if np.isnan(r).any() or np.isnan(r_og).any():
            print("NaN in attributions")
            continue

        r_pos += r[pos]
        r_pos_og += r_og[pos]
        r_rest += sum(r[:pos]) + sum(r[pos+1:])
        # r_rest = r_rest/(len(r_og)-1)
        r_rest_og += sum(r_og[:pos]) + sum(r_og[pos+1:])
        # r_rest_og = r_rest_og/(len(r_og)-1)
    print('Position:', pos)
    print('Pos Sum: ', round(r_pos, 2), 'Pos Sum OG:', round(r_pos_og, 2))
    print('Rest Sum: ', round(r_rest, 2), 'Rest Sum OG:', round(r_rest_og, 2))


def get_position_values(lst):
        """
        Given a 2D list of inhomogeneous shape, returns a list of lists,
        where each inner list contains all values at that position across the outer list.
        """
        # Find the maximum inner length
        max_len = max(len(inner) for inner in lst)
        # For each position, collect all values at that position (if present)
        pos_vals = []
        for i in range(max_len):
            # og had some nans
            vals = [inner[i] for inner in lst if len(inner) > i and not np.isnan(inner[i])]
            pos_vals.append(vals)
        return pos_vals


def normalize_lists(lst):
        normalized = []
        for inner in lst:
            arr = np.array(inner)
            if np.isnan(arr).any():
                normalized.append(arr)
                continue

            normalized.append(get_zscore(arr))
        return normalized

def plot_distributions(list1, list2, 
                               labels=('List 1', 'List 2'), 
                               pos_of_interest=1, 
                               plot_type='boxplot', max_len=None, exclude_cls=False):
    """
    Plots boxplot distributions for each position in two 2D lists of inhomogeneous shape.
    Each list must have the same outer length.
    """
    if exclude_cls:
        list1 = [inner[1:] for inner in list1 if len(inner) > 1]
        list2 = [inner[1:] for inner in list2 if len(inner) > 1]
        pos_of_interest -= 1

    list1 = normalize_lists(list1)
    list2 = normalize_lists(list2)
    

    pos_vals1 = get_position_values(list1)
    pos_vals2 = get_position_values(list2)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    if plot_type == 'boxplot':
        axes[0].boxplot(pos_vals1, showfliers=False)
        axes[1].boxplot(pos_vals2, showfliers=False)
    elif plot_type == 'violin':
        axes[0].violinplot(pos_vals1, showextrema=False)
        axes[1].violinplot(pos_vals2, showextrema=False)
        axes[0].set_ylim(-2, 2)
        axes[1].set_ylim(-2, 2)
    elif plot_type == 'bar_mean':
        means1 = [np.mean(vals) if len(vals) > 0 else 0 for vals in pos_vals1]
        if max_len is not None:
            means1 = means1[:max_len]
        axes[0].bar(range(1, len(means1) + 1), means1)
        means2 = [np.mean(vals) if len(vals) > 0 else 0 for vals in pos_vals2]
        if max_len is not None:
            means2 = means2[:max_len]
        axes[1].bar(range(1, len(means2) + 1), means2)
    elif plot_type == 'bar_median':
        medians1 = [np.median(vals) if len(vals) > 0 else 0 for vals in pos_vals1]
        axes[0].bar(range(1, len(medians1) + 1), medians1)
        medians2 = [np.median(vals) if len(vals) > 0 else 0 for vals in pos_vals2]
        axes[1].bar(range(1, len(medians2) + 1), medians2)
    else:
        raise ValueError("Invalid plot_type. Choose from 'boxplot', 'violin', or 'bar'.")
    axes[0].set_title(labels[0])
    axes[0].set_xlabel('Position')
    axes[0].set_ylabel('Value')

    # axes[1].boxplot(pos_vals2, showfliers=False)
    axes[1].set_title(labels[1])
    axes[1].set_xlabel('Position')

    plt.tight_layout()
    plt.show()

    # Aggregate statistics for all positions except pos_of_interest
    other_positions = [i for i in range(len(pos_vals1)) if i != pos_of_interest]

    vals1_flat = [v for i in other_positions for v in pos_vals1[i]]
    vals2_flat = [v for i in other_positions for v in pos_vals2[i]]

    
    print(f"Position of interest: {pos_of_interest}"
          f"\n{labels[0]} - Median: {np.median(pos_vals1[pos_of_interest]):.4f}, Mean: {np.mean(pos_vals1[pos_of_interest]):.4f}"
          f"\n{labels[1]} - Median: {np.median(pos_vals2[pos_of_interest]):.4f}, Mean: {np.mean(pos_vals2[pos_of_interest]):.4f}")
    
    print(f"\nAggregate statistics (excluding position {pos_of_interest}):")
    print(f"{labels[0]} - Median: {np.median(vals1_flat):.4f}, Mean: {np.mean(vals1_flat):.4f}")
    print(f"{labels[1]} - Median: {np.median(vals2_flat):.4f}, Mean: {np.mean(vals2_flat):.4f}")

    return pos_vals1, pos_vals2


def _to_iso_utc(dt_like: dt.datetime | dt.date) -> str:
    """
    Convert a datetime/date to an ISO 8601 string in UTC (the format W&B filters expect).
    If a date is provided, it’s interpreted as 00:00:00 on that date (UTC).
    """
    if isinstance(dt_like, dt.date) and not isinstance(dt_like, dt.datetime):
        dt_like = dt.datetime.combine(dt_like, dt.time(0, 0, 0))
    if dt_like.tzinfo is None:
        dt_like = dt_like.replace(tzinfo=dt.timezone.utc)
    return dt_like.astimezone(dt.timezone.utc).isoformat()


def query_wandb_runs(
    entity: str,
    project: str,
    model_name: str = None,
    dataset: str = None,
    approach: str = None,
    created_before: dt.datetime | dt.date = None,
    extra_filters: Optional[Dict[str, Any]] = None,
    sweep: Optional[str] = None,
) -> List[wandb.apis.public.Run]:
    """
    Query W&B for runs that match:
      - in sweep (if provided, also looks for runs that have sweep ID as tag) and attributes below
      - config.approach == approach
      - config.model_name == model_name
      - config.dataset in dataset (or == dataset if a single string)
      - created_at < created_before (UTC)
      - plus any additional filters in `extra_filters`

    Notes on `extra_filters`:
      - Keys without a prefix are assumed to be config fields and will be
        automatically prefixed with 'config.' (e.g., 'seed' -> 'config.seed').
      - You can use Mongo-style operators, e.g.:
            extra_filters={'lmbd': {'$gte': 0.0}, 'trainer.max_steps': {'$lt': 5_000}}
      - Non-config fields you can filter on include: 'state', 'tags', 'jobType', 'group'.

    Returns a list of wandb Run objects.
    """
    api = wandb.Api()
    path = f"{entity}/{project}"

    if sweep:
        # Resolve sweep object and get canonical sweep id + (optional) sweep name
        sweeps = api.project(name=project, entity=entity).sweeps()
        sweep_obj = None
        for s in sweeps:
            if s.id == sweep or s.name == sweep:
                sweep_obj = s
                break
        if sweep_obj is None:
            raise ValueError(f"Sweep '{sweep}' not found in {path}")

        sweep_id = sweep_obj.id
        sweep_name = sweep_obj.name

    # Base filters 
    and_filters = []
    if model_name is not None:
        and_filters.append({"config.model_name": model_name})
    if approach is not None:
        and_filters.append({"config.approach": approach})
    if dataset is not None:
        and_filters.append({"config.dataset": {"$in": list(dataset)} if isinstance(dataset, (list, tuple, set)) else dataset})
    if created_before is not None:
        and_filters.append({"created_at": {"$lt": _to_iso_utc(created_before)}})

    # Extra filters (same prefixing logic)
    if extra_filters:
        ef = {}
        for k, v in extra_filters.items():
            key = k
            if not (k.startswith("config.") or k in {"state", "tags", "jobType", "group"}):
                key = f"config.{k}"
            ef[key] = v
        and_filters.append(ef)

    # OR: either truly in sweep, OR one of your “prior run” markers
    if sweep:
        or_filters = [
            {"sweep": sweep_id},                    # true sweep membership
            {"tags": {"$in": [f"{sweep_id}", f"{sweep_name}"]}},  # if you tagged them
        ]
        filters = {"$and": (and_filters + [{"$or": or_filters}])} if and_filters else {"$or": or_filters}
    else:
        # or_filters = []
        filters = {"$and": (and_filters)}

    return list(api.runs(path=path, filters=filters))


def _deep_get(d: Dict[str, Any], dotted_key: str) -> Any:
    """
    Safely get nested dict values with dot notation, e.g., 'trainer.args.lmbd'.
    Returns None if any segment is missing.
    """
    cur: Any = d
    for part in dotted_key.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def make_runs_table(
    runs: Sequence[wandb.apis.public.Run],
    include_config_cols: Optional[Sequence[str]] = None,
    metrics: Optional[Sequence[str]] = None,
    sort_by: Optional[str] = None,
    ascending: bool = False,
    include_metadata: bool = False,
) -> pd.DataFrame:
    """
    Convert a sequence of W&B Run objects into a tidy DataFrame.

    Parameters
    ----------
    include_config_cols : config keys (dot-notation allowed) to include as columns.
        Defaults to ['lmbd'].
    metrics : summary metric keys to include.
        Defaults to ['eval/accuracy', 'eval/mse_macro', 'eval/mse_micro'].
        (Slash-delimited W&B summary keys are supported.)
    sort_by : optional column name to sort by (can be a metric or config column).
    ascending : sort order if sort_by is provided.

    Returns
    -------
    pd.DataFrame with columns:
      - your requested config/metric columns
      - _run_id, _run_name, _created_at (helpers for traceability)
    """
    if include_config_cols is None:
        include_config_cols = ["lmbd", "loss_fn"]
    if metrics is None:
        metrics = ["eval/accuracy", "eval/expl_mse_macro", "eval/expl_mse_micro", "eval/rank_loss"]

    rows: List[Dict[str, Any]] = []
    for r in runs:
        row: Dict[str, Any] = {}

        # config columns (supports nested dicts via dot-notation)
        for c in include_config_cols:
            row[c] = _deep_get(r.config, c)

        # summary metrics
        for m in metrics:
            if m == 'eval/rank_loss':
                try:
                    rank_loss = r.summary.get('eval/rank_loss')
                except KeyError:
                    rank_loss = None
                row[m] = rank_loss
            else:
                row[m] = r.summary.get(m)

        # helpful run metadata
        if include_metadata:
            row["_run_id"] = r.id
            row["_run_name"] = r.name
            row["_created_at"] = r.created_at

        rows.append(row)

    df = pd.DataFrame(rows)

    # Optional sorting if the column exists
    if sort_by and sort_by in df.columns:
        df = df.sort_values(by=sort_by, ascending=ascending, kind="mergesort").reset_index(drop=True)

    return df

def filter_table(table: pd.DataFrame) -> pd.DataFrame:
    """
    Filter the runs table to:
      - Exclude runs with lmbd 0.5 and 2.0
      - If multiple runs with lmbd 0 exist, keep only one and set its loss_fn to '-'
      - remove duplicates if any
      - sort by lmbd ascending
    Returns the filtered DataFrame.
    """
    table = table[~table['lmbd'].isin([0.5, 2.0])]
    zero_idx = table.index[table['lmbd'] == 0].tolist()
    if zero_idx:
        keep = zero_idx[0]
        table.loc[keep, 'loss_fn'] = '-'
        to_drop = [i for i in zero_idx if i != keep]
        if to_drop:
            table = table.drop(to_drop)
        table = table.reset_index(drop=True)
    table = table.drop_duplicates().reset_index(drop=True)
    table = table.sort_values(by=['lmbd', 'loss_fn'], ascending=True, kind='mergesort').reset_index(drop=True)

    return table

def plot_attr_samples(attacked_attr, ref_attr, 
                      n_samples=3, seed=42, normalise=True, max_len=None, 
                      highlight_pos=None, mask_positions=None, exclude_cls=False, include_diff=True, figsize=(18,10)):
    '''
    Plot n_samples of attribution comparisons between attacked and reference attributions.
    attacked_attr: list of attacked attributions (list of lists)
    ref_attr: list of reference attributions (list of lists)
    n_samples: number of samples to plot
    seed: random seed for sampling
    normalise: whether to normalise attributions (z-score)
    max_len: maximum length of attributions to plot
    highlight_pos: position to highlight (int) or 'top' to highlight top attribution in reference
    '''
    # attacked_attr = [a[:len(og_a)] for a, og_a in zip(attacked_attr, ref_attr)]
    if seed is not None:
        np.random.seed(seed)
    sample_idx = list(np.random.choice(len(attacked_attr), size=n_samples, replace=False))
    if include_diff:
        lines = 2
    else:
        lines = 1
    fig, axes = plt.subplots(lines, n_samples, figsize=figsize, sharey=True)
    
    axes = np.array(axes).reshape(lines, n_samples)
    print(axes.shape)

    for i, idx in enumerate(sample_idx):
        og_sample = np.array(ref_attr[idx])
        attacked_sample = np.array(attacked_attr[idx][:len(og_sample)])
        if exclude_cls:
            og_sample = og_sample[1:]
            attacked_sample = attacked_sample[1:]
        if max_len is not None:
            og_sample = og_sample[:max_len]
            attacked_sample = attacked_sample[:max_len]
        if normalise:
            og_sample = get_zscore(og_sample)
            attacked_sample = get_zscore(attacked_sample)

        diff = og_sample - attacked_sample
        L = min(len(og_sample), len(attacked_sample))
        x = np.arange(L)
        width = 0.35
        
        ax = axes[0, i] #if n_samples > 1 else axes[0]
        ax.bar(x - width/2, og_sample[:L], width=width, label='Ref. Attr', alpha=0.8)
        ax.bar(x + width/2, attacked_sample[:L], width=width, label='Attacked Attr', alpha=0.6)
        ax.set_title(f"Sample {i+1}: Attributions")
        ax.set_xlabel("token index")
        if include_diff:
            ax = axes[1, i] #if n_samples > 1 else axes[1]
            ax.bar(x, diff[:L], width=width, alpha=0.7)
            ax.set_title(f"Sample {i+1}: Attribution Difference (Ref. - Attacked)")
            
        if i == 0:
            ax.set_ylabel("attribution value")
        ax.legend()

        def do_highlight_pos(pos, include_diff):
            axes[0, i].axvline(pos - 0.5, color='black', alpha=0.3)
            axes[0, i].axvline(pos + 0.5, color='black', alpha=0.3)
            if include_diff:
                axes[1, i].axvline(pos - 0.5, color='black', alpha=0.3)
                axes[1, i].axvline(pos + 0.5, color='black', alpha=0.3)
            
        if highlight_pos is not None:
            if highlight_pos == 'top':
                top_idx = np.argmax(og_sample)
                pos = top_idx
            elif highlight_pos == 'mask':
                assert mask_positions is not None, "mask_positions must be provided when highlight_pos is 'mask'"
                positions, = np.nonzero(np.array(mask_positions[idx]))
                for pos in positions:
                    do_highlight_pos(pos, include_diff)
                continue

            else:
                pos = highlight_pos
            if pos < L:
                do_highlight_pos(pos, include_diff)
                # axes[0, i].axvline(pos - 0.5, color='black', alpha=0.3)
                # axes[0, i].axvline(pos + 0.5, color='black', alpha=0.3)
                
                # axes[1, i].axvline(pos - 0.5, color='black', alpha=0.3)
                # axes[1, i].axvline(pos + 0.5, color='black', alpha=0.3)

    plt.tight_layout()
    plt.show()

def get_zscore(arr):
    '''
    Get z-score of value in arr
    arr: list or numpy array
    value: float
    returns: z-score of value in arr
    '''
    arr = np.array(arr)
    mean = np.mean(arr)
    std = np.std(arr)
    zscore = (arr - mean) / std
    return zscore

def get_percentile_ranks(arr, value):
    '''
    Get percentile ranks of value in arr
    arr: list or numpy array
    value: float
    returns: percentile rank of value in arr
    '''
    arr = np.array(arr)
    percentile_rank = (np.sum(arr <= value) / len(arr)) * 100
    return percentile_rank

def get_positions(file_path, split='validation'):
    with open(file_path, 'r') as f:
        token_data = json.load(f)
    token_positions = token_data[split]
    return token_positions


def pos_v_rest(attr, ref_attr, positions):
    '''
    Compare mean attribution of highlited tokens vs rest
    attr: list of attributions (list of lists)
    ref_attr: list of reference attributions (list of lists)
    positions: positions of highlighted tokens (list of binary masks)
    returns: dict with results
        - highlighted_mean: mean of highlighted means for attributions
        - mean_rest: mean of rest means for attributions
        - diff_mean_rest: difference between highlighted mean and rest mean for attributions
        - percentile_rank: mean percentile rank of highlighted tokens in attributions
    '''
    results = {}
    diffs = []
    highlighted_means = []
    rest_means = []
    percentile_ranks = []
    for i in range(len(attr)):
        # print(len(attr[i]), len(ref_attr[i]), len(positions[i]))
        a = np.array(attr[i])
        a = a[:len(ref_attr[i])]  # ensure same length
        if np.isnan(a).any() or sum(positions[i]) == 0:
            continue
        a = get_zscore(a)
        # print('a:', a)
        # print('positions:', positions[i])
        top_mean = a[np.array(positions[i]).astype(bool)].mean()
        rest_mean = a[~np.array(positions[i]).astype(bool)].mean()

        diffs.append(top_mean - rest_mean)
        highlighted_means.append(top_mean)
        rest_means.append(rest_mean)

        if sum(positions[i])>0:
            for idx, pos in enumerate(positions[i]):
                if pos == 1: 
                    percentile_rank = get_percentile_ranks(a, a[idx])
                    percentile_ranks.append(percentile_rank)
        else: 
            percentile_rank = get_percentile_ranks(a, a[positions[i]][0])
            percentile_ranks.append(percentile_rank)

    results['highlighted_mean'] = np.mean(highlighted_means)
    results['mean_rest'] = np.mean(rest_means)
    results['diff_mean_rest'] = np.mean(diffs)
    results['percentile_rank'] = np.mean(percentile_ranks)

    return results


def get_wandb_id(entity, project, model_name, dataset, approach, extra_filters):
    '''
    Get the wandb run ID for a specific run based on the provided parameters.
    Returns the run ID if found, else None.

    Sample: 
    get_wandb_id(entity='anonuser',
             project='xai_fooling',
             model_name='custom-bert',
             dataset='sst2',
             approach='increase_tokens',
             extra_filters={'lr': {'$eq': 1e-5}, 
                            'optimizer': {'$eq': 'adamw'}, 
                            'lmbd': {'$eq': 10}, 
                            'scheduler_type': {'$eq':'linear'}, 
                            'get_attributions': {'$eq': True}, 
                            'loss_fn': {'$eq':'MSE_macro'}})
    '''
    runs = query_wandb_runs(entity=entity,
                     project=project,
                     model_name=model_name,
                     dataset=dataset,
                     approach=approach,
                     extra_filters=extra_filters)
    if runs:
        return runs[0].id
    else:
        return None
    

def make_constant_positions(og_file, position, save_file): 
    '''
    Create a positions file where only the specified position is highlighted for all samples.
    og_file: path to original attributions file
    position: position to highlight (int)
    save_file: path to save the new positions file
    returns: list of positions (list of binary masks)
    '''
    with open(og_file, 'r') as f:
        og_results = json.load(f)
    og_attr = og_results['attributions']
    positions = []
    for i, attr in enumerate(og_attr):
        pos = np.zeros_like(attr)
        pos[position] = 1
        positions.append(pos.tolist())
        
    result = {'validation': positions}
    with open(save_file, 'w') as f:
        json.dump(result, f)
    return positions

def get_table_highlighted_tokens(attr_file_macro, 
        attr_file_rank, 
        attr_file_ref, 
        positions_file, 
        other_attr:dict=None):
    '''
    Get a table comparing attribution methods based on highlighted tokens.
    attr_file_macro: path to macro attributions file
    attr_file_rank: path to rank attributions file
    attr_file_ref: path to reference attributions file
    positions_file: path to positions file
    returns: pd.DataFrame with results
    '''
    ref_attr = get_attr(attr_file_ref)
    attr_rank = get_attr(attr_file_rank)
    attr_macro = get_attr(attr_file_macro)
    positions = get_positions(positions_file, split='validation')
    results_macro = pos_v_rest(attr_macro, ref_attr, positions)
    results_rank = pos_v_rest(attr_rank, ref_attr, positions)
    results_ref = pos_v_rest(ref_attr, ref_attr, positions)
    if other_attr is not None:
        for name, file in other_attr.items():
            attr_other = get_attr(file)
            results_other = pos_v_rest(attr_other, ref_attr, positions)
            results  = {
                'macro': results_macro,
                'rank': results_rank,
                'ref' : results_ref,
                name: results_other
            }
    else:
        results= {
            'macro': results_macro,
            'rank': results_rank,
            'ref' : results_ref
            
        }
    results_df = pd.DataFrame(results)
    return results_df.T

def top_v_rest(attr, ref_attr, topk=1, ignore_cls=False, positions=None):
    '''
    Compare mean attribution of topk important tokens vs rest
    attr: list of attributions (list of lists)
    ref_attr: list of reference attributions (list of lists)
    topk: number of top tokens to consider
    returns: dict with results
        - diff_attacked: difference between topk mean and rest mean for attacked attributions
        - diff_ref: difference between topk mean and rest mean for reference attributions
        - top_mean_attacked: mean of topk means for attacked attributions
        - top_mean_ref: mean of topk means for reference attributions
    '''
    results = {}
    diffs = []
    top_means = []
    rest_means = []
    percentile_ranks = []
    for i in range(len(attr)):
        a = np.array(attr[i])
        a = a[:len(ref_attr[i])]  # ensure same length
        r = np.array(ref_attr[i])
        if ignore_cls:
            a = a[1:]
            r = r[1:]
        a = get_zscore(a)
        r = get_zscore(r)
        if positions is None: 
            top_indices = np.argsort(r)[-topk:]
            rest_indices = np.argsort(r)[:-topk]
        else: 
            if np.isnan(a).any() or sum(positions[i]) == 0:
                continue
            top_indices = np.array(positions[i]).astype(bool)
            rest_indices = ~np.array(positions[i]).astype(bool)

        top_mean = a[top_indices].mean()
        rest_mean = a[rest_indices].mean()

        diffs.append(top_mean - rest_mean)
        top_means.append(top_mean)
        rest_means.append(rest_mean)

        if positions is not None:
            for idx, pos in enumerate(positions[i]):
                if pos == 1: 
                    percentile_rank = get_percentile_ranks(a, a[idx])
                    percentile_ranks.append(percentile_rank)
        elif topk==1:
            percentile_rank = get_percentile_ranks(a, a[top_indices][0])
            percentile_ranks.append(percentile_rank)
        

    results['top_mean'] = np.mean(top_means)
    results['mean_rest'] = np.mean(rest_means)
    results['diff_mean_rest'] = np.mean(diffs)
    if topk==1:
        results['percentile_rank'] = np.mean(percentile_ranks)
    return results


def get_table_topk(attr_file_rank, attr_file_topk, attr_file_ref, ignore_cls=False, positions_file=None):
    ref_attr = get_attr(attr_file_ref)
    attr_rank = get_attr(attr_file_rank)
    attr_topk = get_attr(attr_file_topk)

    if positions_file is not None: 
        positions = get_positions(positions_file, split='validation')
    else: 
        positions = None

    results_topk = top_v_rest(attr_topk, ref_attr, topk=1, ignore_cls=ignore_cls, positions=positions)
    results_rank = top_v_rest(attr_rank, ref_attr, topk=1, ignore_cls=ignore_cls, positions=positions)
    results_ref = top_v_rest(ref_attr, ref_attr, topk=1, ignore_cls=ignore_cls, positions=positions)
    results= {
        'topk': results_topk,
        'rank': results_rank,
        'ref' : results_ref
        
    }
    results_df = pd.DataFrame(results)
    return results_df.T


def check_checkpoint(run_id): 
    print(os.listdir(f'/vol/csedu-nobackup/project/anonuser/results/{run_id}/'))