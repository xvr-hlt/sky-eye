import torch
from collections import defaultdict
from tqdm import tqdm
from apex import amp

def train(model, optim, data, loss_fn, pre_weight=None, post_weight=None):
    model = model.train()
    loss_sum, loss_pre_sum, loss_post_sum = 0., 0., 0.
    
    for batch in tqdm(iter(data)):
        optim.zero_grad()        
        loss = 0.
        
        if pre_weight:
            pre_out = model(batch['images']['image'].to('cuda'))
            pre_targets = batch['masks']['buildings'].to('cuda')
            loss_pre = pre_weight*loss_fn(pre_out, batch['masks']['buildings'].to('cuda'))
            loss += loss_pre
            loss_pre_sum += loss_pre

        if post_weight:
            post_out = model(batch['images']['post'].to('cuda'), downscale=True)
            post_targets = batch['masks']['damage'].to('cuda')
            loss_post = post_weight*sum((loss_fn(mask_out, mask) for mask_out, mask in zip(post_out, batch['masks']['damage'].to('cuda'))))
            loss_post /= post_out.shape[1]
            loss += loss_post
            loss_post_sum += loss_post

        if pre_weight and post_weight:
            loss /= (pre_weight+post_weight)
        
        with amp.scale_loss(loss, optim) as scaled_loss:
            scaled_loss.backward()

        optim.step()
        
        loss_sum += loss
        
    return {
        'loss':loss_sum/len(data), 
        'loss_pre': loss_pre_sum/len(data) if pre_weight else None,
        'loss_post': loss_post_sum/len(data) if post_weight else None
    }

def batch_metrics(outputs, targets, threshold=0.5):
    metrics = defaultdict(float)
    pr_sum, re_sum, f_sum, iou_sum = 0., 0., 0., 0.
    for output, target in zip(outputs, targets):
        target_bool = target.to(torch.bool)
        output_bool = output.sigmoid() > threshold
        
        if target.sum() == 0:
            if output_bool.int().sum() == 0:
                precision, recall, iou = 1., 1., 1.
            else:
                precision, recall, iou = 0., 0., 0.
        else:
            precision = target_bool[output_bool].float().mean() if output_bool.int().sum() > 0 else 0.
            recall = output_bool[target_bool].float().mean()        
            intersection = output_bool[target_bool].float().sum()
            iou = intersection/(target_bool.float().sum() + output_bool.float().sum() - intersection)
        
        metrics['recall'] += recall
        metrics['precision'] += precision
        metrics['f1'] += 2*precision*recall/(precision + recall) if (precision + recall) > 0. else 0.
        metrics['iou'] += iou

    return {k:v/len(outputs) for k,v in metrics.items()}

def get_tp_fp_fn(outputs, targets, threshold=0.5):
    outputs_bool = outputs.sigmoid() > threshold
    targets_bool = targets.to(torch.uint8)
    tp = outputs_bool[targets_bool].sum().float() if targets_bool.sum() > 0 else 0.
    fn = targets_bool[~outputs_bool].sum().float() if (~outputs_bool).sum() > 0 else 0.
    fp = (~targets_bool[outputs_bool]).sum().float() if outputs_bool.sum() > 0 else 0.
    return tp, fp, fn 

def get_metrics_for_counts(tp, fp, fn):
    prec = tp/(tp+fp) if (tp+fp).sum() > 0. else 0.
    rec = tp/(tp+fn) if (tp+fn).sum() > 0. else 0.
    return {
        'precision': prec,
        'recall': rec,
        'f1': 2*prec*rec/(prec+rec) if (prec+rec).sum() > 0. else 0.
    }

def evaluate(model, data, loss_fn, threshold=0.5, pre_weight=None, post_weight=None, damage_classes=None):
    model = model.eval()
    metrics = defaultdict(float)
    
    counts_tp, counts_fp, counts_fn = defaultdict(float), defaultdict(float), defaultdict(float)
    
    with torch.no_grad():
        metric_sums = defaultdict(float)
        for batch in tqdm(iter(data)):
            loss = 0.
            
            if pre_weight:
                pre_out = model(batch['images']['image'].to('cuda'))
                pre_targets = batch['masks']['buildings'].to('cuda')
                loss_pre = pre_weight*loss_fn(pre_out, pre_targets)
                metrics['loss_pre'] += loss_pre
                loss += loss_pre                
                tp, fp, fn = get_tp_fp_fn(pre_out, pre_targets, threshold)
                counts_tp['building'] += tp
                counts_fp['building'] += fp
                counts_fn['building'] += fn

            if post_weight:
                post_out = model(batch['images']['post'].to('cuda'), downscale=True)
                post_targets = batch['masks']['damage'].to('cuda')
                loss_post = post_weight*sum((loss_fn(mask_out, mask) for mask_out, mask in zip(post_out, post_targets)))
                metrics['loss_post'] += loss_post
                loss += loss_post
                macro_metrics = defaultdict(float)
                for dmg_type, ix in train_dataset.DAMAGE_CLASSES.items():
                    tp, fp, fn = get_tp_fp_fn(post_out[:,ix], post_targets[:,ix], threshold)
                    counts_tp[f'dmg_{dmg_type}'] += tp
                    counts_fp[f'dmg_{dmg_type}'] += fp
                    counts_fn[f'dmg_{dmg_type}'] += fn

            if pre_weight and post_weight:
                loss /= pre_weight+conf.post_weight

            metrics['loss'] += loss
        metrics = {k:v/len(data) for k, v in metrics.items()}
        
        for typ in counts_tp:
            metrics.update({f'{typ}:{k}':v for k,v in get_metrics_for_counts(counts_tp[typ], counts_fp[typ], counts_fn[typ]).items()})

    return metrics