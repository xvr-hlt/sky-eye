from xv import run
from torchvision.ops import misc as misc_nn_ops
from apex import amp
from torch.nn.modules.loss import CrossEntropyLoss
from xv.nn.losses import loss_dict, WeightedLoss
from pytorch_toolbelt import losses
import pandas as pd
from xv import dataset
import random
from xv.nn.layers import FrozenBatchNorm2d
from xv.util import vis_im_mask
from torch import nn
import torch
import numpy as np
from tqdm import tqdm
from glob import glob
from pprint import pprint
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import os
import wandb
import yaml
from xv import io
from pprint import pprint
import cv2

cv2.setNumThreads(0)

conf_file = "config/config-seg.yaml"

with open(conf_file) as f:
    conf_init = yaml.load(f)

#os.environ['WANDB_MODE'] = 'dryrun'
wandb.init(project=conf_init['project'], config=conf_init, name=conf_init['name'])
conf = wandb.config

pprint(dict(conf))

model, preprocess_fn = io.load_segmentation_model(conf)
model.to('cuda')

train_dataset, train_loader = io.load_training_data(conf, preprocess_fn)
dev_dataset, dev_loader = io.load_dev_data(conf, preprocess_fn)

print(f"n_train: {len(train_dataset)}")
print(f"n_dev: {len(dev_dataset)}")

loss = WeightedLoss({loss_dict[l](): w for l, w in conf.loss_weights.items()})

optims = {
    'adam': torch.optim.Adam,
    'sgd': torch.optim.SGD
}

optim = optims[conf.optim](model.parameters(), lr=conf.lr)


model, optim = amp.initialize(model, optim, opt_level=conf.amp_opt_level)

if torch.cuda.device_count() > 1:
    #if conf.sync_bn:
    #    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = nn.DataParallel(model)
    #torch.distributed.init_process_group(backend="nccl")
    #model = nn.DistributedDataParallel(model)


scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optim, factor=conf.scheduler_factor, patience=conf.scheduler_patience
)

train_resize = run.MultiScaleResize(conf.mode, conf.training_scales)

best_score = 0
epoch = 0


train_fn = run.train_segment if conf.nclasses == 1 else run.train_damage
eval_fn = run.evaluate_segment if conf.nclasses == 1 else run.evaluate_damage

for epoch in range(epoch, conf.epochs):
    print(f"epoch {epoch}/{conf.epochs}.")
    torch.save(optim.state_dict(), os.path.join(wandb.run.dir, "optim.pth"))
    torch.save(scheduler.state_dict(), os.path.join(wandb.run.dir, "scheduler.pth"))
    metrics = {'epoch': epoch}
    train_metrics = train_fn(model, optim, train_loader, loss, train_resize=train_resize, mode=conf.mode)
    metrics.update(train_metrics)

    dev_metrics = eval_fn(model, dev_loader, loss, mode=conf.mode)
    metrics.update(dev_metrics)
    wandb.log(metrics)
    score = metrics[conf.metric]
    scheduler.step(-score)
    pprint(metrics)
    if score > best_score:
        torch.save(model.state_dict(), os.path.join(wandb.run.dir, "state_dict.pth"))
        best_score = score
