import torch
from torch import nn
from collections import defaultdict
from tqdm import tqdm
from apex import amp
from . import util
from PIL import Image
import random
import numpy as np
from random import choice
from xv.submission_metrics import RowPairCalculator
import scipy

class MultiScaleResize(nn.Module):
    def __init__(self, mode="categorical", scales=(0.5, 0.75, 1.)):
        super().__init__()
        self.mode = mode
        self.scales = scales

    @torch.no_grad()
    def forward(self, batch):
        scale = random.choice(self.scales)
        if scale == 1.:
            return batch

        if self.mode is None or self.mode == "dual":
            im, mask = batch
            mask_dtype = mask.dtype
            im = torch.nn.functional.interpolate(im, scale_factor=scale, mode='bilinear', align_corners=False)
            mask = misc_nn_ops.interpolate(mask.float(), scale_factor=scale).to(mask_dtype)
            return im, mask

        if self.mode == "categorical":
            im, (damage_mask, damage) = batch
            dmg_msk_dtype = damage_mask.dtype()
            dmg_dtype = damage.dtype()
            im = torch.nn.functional.interpolate(im, scale_factor=scale, mode='bilinear', align_corners=False)
            damage_mask = misc_nn_ops.interpolate(damage_mask[None].float(), scale_factor=scale)[0].to(dmg_msk_dtype)
            damage_one_hot = torch.nn.functional.one_hot(damage).permute(0, 3, 1, 2)
            damage = misc_nn_ops.interpolate(damage_one_hot.float(), scale_factor=scale).argmax(1)
            return im, (damage_mask, damage)

def train_segment(model, optim, data, loss_fn, train_resize=None, mode=None):
    model = model.train()
    loss_sum = 0.

    for image, mask in tqdm(iter(data)):
        if train_resize:
            image, mask = train_resize((image, mask))
        optim.zero_grad()
        outputs = model(image.to('cuda'))
        targets = mask.to('cuda')
        
        loss = loss_fn(outputs, targets)
        
        with amp.scale_loss(loss, optim) as scaled_loss:
            scaled_loss.backward()
        
        #loss.backward()
        optim.step()
        loss_sum += loss
        
    return {
        'train:loss':loss_sum/len(data)
    }

@torch.no_grad()
def evaluate_segment(model, data, loss_fn, threshold=0.5, mode=None):
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

def train_damage(model, optim, data, loss_fn, train_resize=None, mode=None):
    model = model.train()
    loss_sum = 0.

    for image, mask in tqdm(iter(data)):
        if train_resize:
            image, mask = train_resize((image, mask))
        optim.zero_grad()
        outputs = model(image.to('cuda'))
        if mode == 'categorical':
            _, nclasses, _, _ = outputs.shape
            mask_bool, mask = mask
            if mask_bool.sum() == 0:
                continue
            loss = loss_fn(outputs.permute(0,2,3,1)[mask_bool], mask[mask_bool].cuda())
        else:
            loss = loss_fn(outputs, mask)
        
        with amp.scale_loss(loss, optim) as scaled_loss:
            scaled_loss.backward()

        optim.step()
        loss_sum += loss
        
    return {
        'train:loss':loss_sum/len(data)
    }

@torch.no_grad()
def evaluate_damage(model, data, loss_fn, threshold=0.5, mode=None):
    model = model.eval()
    metrics = {}
    loss=0.
    tps, fps, fns = defaultdict(float), defaultdict(float), defaultdict(float)
    tps_c, fps_c, fns_c = defaultdict(float), defaultdict(float), defaultdict(float)
    for image, mask in tqdm(iter(data)):
        outputs = model(image.cuda())
        _, nclasses, _, _ = outputs.shape
        
        if mode == "categorical":
            mask_bool, mask = mask
            if mask_bool.sum() == 0:
                continue
            loss += loss_fn(outputs.permute(0,2,3,1)[mask_bool], mask[mask_bool].cuda())
            
        mask = np.array(mask.cpu())
        outputs = np.array(outputs.float().argmax(1).cpu())
        
        flat_output, flat_target = outputs[mask_bool], mask[mask_bool]
        
        for ix in range(nclasses):                
            tp, fn, fp = RowPairCalculator.compute_tp_fn_fp(flat_output, flat_target, ix)
            tps[ix] += tp
            fps[ix] += fp
            fns[ix] += fn

    metrics['loss'] = loss / len(data)
    
    aggregate = defaultdict(list)
    for ix in range(nclasses):
        categorical_ix_metrics =  get_metrics_for_counts(tps[ix], fps[ix], fns[ix])
        for k,v in categorical_ix_metrics.items():
            metrics[f'damage:categorical:{ix}:{k}'] = v
            aggregate[f'damage:categorical:{k}'].append(v)
    hmean = {f'hmean:{k}': scipy.stats.hmean(v) for k,v in aggregate.items()}
    metrics.update(hmean)
    
    mean = {f'mean:{k}':scipy.mean(v) for k,v in aggregate.items()}
    metrics.update(mean)

    return metrics

def get_tp_fp_fn(outputs, targets, threshold=0.5):
    outputs_bool = outputs.sigmoid() > threshold
    if torch.__version__.startswith("1.2") or torch.__version__.startswith("1.3"):
        targets_bool = targets.to(torch.bool)
    else:
        targets_bool = targets.to(torch.uint8)
    
    tp = outputs_bool[targets_bool].float().sum() if targets_bool.float().sum() > 0 else 0.
    fn = targets_bool[~outputs_bool].float().sum() if (~outputs_bool).float().sum() > 0 else 0.
    fp = (~targets_bool[outputs_bool]).float().sum() if outputs_bool.float().sum() > 0 else 0.
    return tp, fp, fn

def get_metrics_for_counts(tp, fp, fn):
    prec = tp/(tp+fp) if tp+fp > 0. else 0.
    rec = tp/(tp+fn) if tp+fn > 0. else 0.
    return {
        'precision': prec,
        'recall': rec,
        'f1': 2*prec*rec/(prec+rec) if prec+rec > 0. else 0.
    }

def sample_masks(model, instances, preprocess_fn, sz=512, n=5, opacity=.3):
    model.eval()
    ims = []
    for i in random.sample(instances, n):
        with torch.no_grad():
            img = np.array(Image.open(i['file_name']).resize((sz, sz)))
            model_in = preprocess_fn(img).transpose(2,0,1).astype(np.float32)
            model_in = torch.tensor(model_in)
            model_in = model_in.reshape(1, *model_in.shape)
            mask = model(model_in.cuda())
            mask = np.array((mask > 0).cpu())
            ims.append([util.vis_im_mask(img, m, opacity=opacity, size=(sz,sz)) for m in mask[0]])
    return ims
