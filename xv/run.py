import torch
from collections import defaultdict
from tqdm import tqdm_notebook as tqdm
from apex import amp

def train_segment(model, optim, data, loss_fn):
    model = model.train()
    loss_sum = 0.

    for image, mask in tqdm(iter(data)):
        optim.zero_grad()
        loss = loss_fn(model(image.to('cuda')), mask.to('cuda'))
        
        with amp.scale_loss(loss, optim) as scaled_loss:
            scaled_loss.backward()

        optim.step()
        loss_sum += loss
        
    return {
        'train:loss':loss_sum/len(data)
    }

def evaluate_segment(model, data, loss_fn, threshold=0.5):
    model = model.eval()
    tps, fps, fns, loss = 0., 0., 0., 0.
    with torch.no_grad():
        for image, mask in tqdm(iter(data)):
            out = model(image.to('cuda'))
            loss += loss_fn(out, mask.cuda())
            tp, fp, fn = get_tp_fp_fn(out, mask, threshold)
            tps += tp
            fps += fp
            fns += fn
    metrics = {'loss': loss/len(data)}
    metrics.update({f'building:{k}':v for k,v in get_metrics_for_counts(tps, fps, fns).items()})
    return metrics

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
