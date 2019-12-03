import os
import yaml
import wandb
from glob import glob
from tqdm import tqdm_notebook as tqdm
import numpy as np
import torch
from torch import nn
from xv.util import vis_im_mask
from pprint import pprint
import random
from xv import dataset, io, run_damage
import pandas as pd
import random
from pytorch_toolbelt import losses
from torch.nn.modules.loss import CrossEntropyLoss
from xv.nn.losses import loss_dict, WeightedLoss
from torch.nn.modules.loss import CrossEntropyLoss
from tqdm import tqdm as tqdm
import pdb
import logging
from apex import amp

#os.environ['WANDB_MODE'] = 'dryrun'

run_type = 'building-damage'
conf_file = 'config/config-damage-od.yaml'
#conf_file = 'config/config-damage-od-finetune.yaml'

with open(conf_file) as f:
    conf_init = yaml.load(f)

wandb.init(project=run_type, config=conf_init, name=conf_init['name'])
conf = wandb.config
pprint(dict(conf))

train_loader, dev_loader = io.get_damage_loaders(conf)

class MultiScaleResize(nn.Module):
    def __init__(self, scales = (0.5, 0.75, 1.)):
        super().__init__()
        self.scales = scales
    
    @torch.no_grad()
    def forward(self, x, boxes):
        scale = random.choice(self.scales)
        if scale != 1.:
            x = torch.nn.functional.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=False)
            boxes = [b*scale for b in boxes]
        return x, boxes

train_resize = MultiScaleResize(conf.training_scales)

model = io.load_damage_model(conf)
model = model.cuda()

optims = {'adam': torch.optim.Adam}
optim = optims[conf.optim](model.parameters(), lr=conf.lr)

model, optim = amp.initialize(model, optim, opt_level=conf.amp_opt_level)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optim, factor=conf.scheduler_factor, patience=conf.scheduler_patience
)

#loss_fn = WeightedLoss({loss_dict[l]():w for l, w in conf.loss_weights.items()})
#loss_fn = losses.JaccardLoss('multiclass')
#loss_fn = CrossEntropyLoss(weights)

if 'class_weight' in dict(conf):
    weights = torch.Tensor(conf.class_weight).float().cuda()
    loss_fn = CrossEntropyLoss(weights, reduction=conf.loss_reduce_mode)

#loss_fn = losses.JointLoss(loss_fn, losses.FocalLoss(), 0.5, 0.5)


epoch, best_score = 0, 0


for epoch in range(epoch, conf.epochs):
    metrics = {'epoch': epoch}
    train_metrics = run_damage.run(model, optim, train_loader, train_resize, loss_fn)
    metrics.update(train_metrics)
    
    dev_metrics = run_damage.evaluate(model, dev_loader, conf.nclasses, loss_fn)
    metrics.update(dev_metrics)
    
    wandb.log(metrics)
    score = metrics[conf.metric]
    scheduler.step(-score)
    
    
    if score > best_score:
        torch.save(model.state_dict(), os.path.join(wandb.run.dir, "state_dict.pth"))
        torch.save(optim.state_dict(), os.path.join(wandb.run.dir, "optim.pth"))
        torch.save(scheduler.state_dict(), os.path.join(wandb.run.dir, "scheduler.pth"))
        best_score = score
