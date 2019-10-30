import torch
from collections import defaultdict
from tqdm import tqdm_notebook as tqdm
from apex import amp
from . import util
from PIL import Image
import random
import numpy as np
from random import choice

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
    targets_bool = targets.to(torch.bool)
    
    tp = outputs_bool[targets_bool].float().sum() if targets_bool.float().sum() > 0 else 0.
    fn = targets_bool[~outputs_bool].float().sum() if (~outputs_bool).float().sum() > 0 else 0.
    fp = (~targets_bool[outputs_bool]).float().sum() if outputs_bool.float().sum() > 0 else 0.
    return tp, fp, fn

def get_metrics_for_counts(tp, fp, fn):
    prec = tp/(tp+fp) if (tp+fp).sum() > 0. else 0.
    rec = tp/(tp+fn) if (tp+fn).sum() > 0. else 0.
    return {
        'precision': prec,
        'recall': rec,
        'f1': 2*prec*rec/(prec+rec) if (prec+rec).sum() > 0. else 0.
    }

def sample_masks(model, instances, preprocess_fn, sz=512, n=5):
    model.eval()
    ims = []
    for i in random.sample(instances, n):
        with torch.no_grad():
            img = np.array(Image.open(i['file_name']))
            model_in = preprocess_fn(img).transpose(2,0,1)
            model_in = torch.tensor(model_in)
            model_in = model_in.reshape(1, *model_in.shape)
            mask = model(model_in.cuda())
            mask = np.array((mask > 0).cpu())
            im = util.vis_im_mask(img, mask[0], opacity=.3)
            im = im.resize((sz, sz))
            ims.append(im)
    return ims