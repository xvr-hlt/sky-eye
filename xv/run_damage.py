import scipy
from collections import defaultdict
from xv.run import get_metrics_for_counts
import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from apex import amp
import logging


def run(model, optim, data, train_resize, loss_fn):
    model = model.train()
    total_loss = 0.
    for im, boxes, clss in tqdm(iter(data)):
        if im.shape[0] == 0:
            logging.warning("Empty batch.")
            continue
        im, boxes = train_resize(im, boxes)
        optim.zero_grad()
        out = model(im.cuda(), [b.half().cuda() for b in boxes]) # :|
        loss = loss_fn(out, torch.cat(clss).long().cuda())
        
        with amp.scale_loss(loss, optim) as scaled_loss:
            scaled_loss.backward()
        
        #loss.backward()
        total_loss += loss
        optim.step()
    return {'train_loss': total_loss/len(data)}

def weighted_tp_fp_fn(pred, targ, weights, c):
    tp = (np.logical_and(pred == c, targ == c) * weights).sum()
    fp = (np.logical_and(pred != c, targ == c) * weights).sum()
    fn = (np.logical_and(pred == c, targ != c) * weights).sum()
    return tp, fp, fn

@torch.no_grad()
def evaluate(model, data, nclasses, loss_fn):
    model.eval()
    loss_sum = 0.
    tps, fps, fns = [0. for _ in range(nclasses)], [0. for _ in range(nclasses)], [0. for _ in range(nclasses)]
    
    for im, boxes, clss in tqdm(data):
        res = im.shape[-1]
        out = model(im.cuda(), [b.cuda().half() for b in boxes])
        clss = torch.cat(clss).long()
        loss_sum += loss_fn(out, clss.cuda())

        out_ix = np.array(out.argmax(1).cpu())
        clss = clss.cpu().numpy()
        boxes_flat = torch.cat(boxes)
        areas = (boxes_flat[:,2] - boxes_flat[:,0]) * (boxes_flat[:,3] - boxes_flat[:,1])
        areas = areas.cpu().numpy()

        for cl in range(nclasses):
            tp, fp, fn = weighted_tp_fp_fn(out_ix, clss, areas, cl)
            tps[cl] += tp
            fps[cl] += fp
            fns[cl] += fn

    metrics = {}
    metrics['loss'] = loss_sum / len(data)
    
    aggregate = defaultdict(list)
    for ix in range(nclasses):
        categorical_ix_metrics =  get_metrics_for_counts(tps[ix], fps[ix], fns[ix])
        for k,v in categorical_ix_metrics.items():
            metrics[f'damage:categorical:{ix}:{k}'] = v
            aggregate[f'damage:categorical:{k}'].append(v)

    hmean = {f'hmean:{k}': scipy.stats.hmean(v) if all(v) else 0. for k,v in aggregate.items()}
    metrics.update(hmean)

    mean = {f'mean:{k}':scipy.mean(v) for k,v in aggregate.items()}
    metrics.update(mean)
    
    return metrics
