from xv import run
from torchvision.ops import misc as misc_nn_ops
#from apex import amp
from torch.nn.modules.loss import CrossEntropyLoss
from xv.nn.losses import loss_dict, WeightedLoss
from pytorch_toolbelt import losses
import pandas as pd
from xv import dataset
import random
from xv.nn.layers import FrozenBatchNorm2d
import albumentations as al
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


conf_file = "config/config-seg.yaml"
# conf_file = "config/config-seg-finetune.yaml"
# conf_file = "config/config-seg-joint.yaml"

with open(conf_file) as f:
    conf_init = yaml.load(f)


os.environ['WANDB_MODE'] = 'dryrun'
wandb.init(project=conf_init['project'], config=conf_init, name=conf_init['name'])
conf = wandb.config

pprint(dict(conf))

train_dir = '../../datasets/xview/train'
test_dir = '../../datasets/xview/test'
suppl_dir = '../../datasets/xview/tier3'
tertiary_dir = '../../datasets/spacenet'

augment = al.Compose([
    al.HorizontalFlip(p=conf.aug_prob),
    al.VerticalFlip(p=conf.aug_prob),
    al.RandomRotate90(p=conf.aug_prob),
    al.Transpose(p=conf.aug_prob),
    al.GridDistortion(p=conf.aug_prob, distort_limit=.2),
    al.ShiftScaleRotate(p=conf.aug_prob),
    al.RandomBrightnessContrast(p=conf.aug_prob),
])


segmentation_types = {
    'PSPNet': smp.PSPNet,
    'FPN': smp.FPN,
    'Linknet': smp.Linknet,
    'Unet': smp.Unet
}

model_classes = conf.nclasses


model = segmentation_types[conf.segmentation_arch](
    conf.encoder,
    classes=model_classes,
    attention_type=conf.attention,
)

if conf.mode == "dual":
    model = DualWrapper(model)

if conf.load_weights:
    state_dict = torch.load(conf.load_weights)
    print(model.load_state_dict(state_dict))

preprocess_fn = get_preprocessing_fn(conf.encoder)

if conf.freeze_encoder_norm:
    model.encoder = FrozenBatchNorm2d.convert_frozen_batchnorm(model.encoder)

if conf.freeze_decoder_norm:
    model.decoder = FrozenBatchNorm2d.convert_frozen_batchnorm(model.decoder)

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

model.to('cuda')


train_stems = pd.read_csv('config/train_stems.csv', header=None)[0]
dev_stems = pd.read_csv('config/dev_stems.csv', header=None)[0]

train_files = [f'{train_dir}/labels/{stem}_{conf.data_prefix}_disaster.json' for stem in train_stems]
dev_files = [f'{train_dir}/labels/{stem}_{conf.data_prefix}_disaster.json' for stem in dev_stems]

train_instances = dataset.get_instances(train_files, filter_none=conf.filter_none)
dev_instances = dataset.get_instances(dev_files, filter_none=conf.filter_none)

len(train_instances), len(dev_instances)


if conf.add_suppl:
    train_instances *= conf.train_repeat
    suppl_files = glob(f'{suppl_dir}/labels/*{conf.data_prefix}_disaster.json')
    suppl_instances = dataset.get_instances(suppl_files, filter_none=conf.filter_none)
    train_instances += suppl_instances
    print(len(train_instances))

if conf.add_tertiary:
    import pandas as pd
    from PIL import Image
    import shapely
    tertiary_instances = []

    for subdir in tqdm(os.listdir(tertiary_dir)):
        csv = glob(f'{tertiary_dir}/{subdir}/summaryData/*.csv')[0]
        df = pd.read_csv(csv)
        for img, group in tqdm(df.groupby('ImageId')):
            filepath = f'{tertiary_dir}/{subdir}/rgb/rgb_{img}.jpg'
            _im = Image.open(filepath)
            h, w = _im.size
            polys = [shapely.wkt.loads(p) for p in group.PolygonWKT_Pix]
            points_list = [list(poly.exterior.coords) for poly in polys if poly.exterior is not None]
            points_list = [[(x/w, y/h) for x, y, _ in points] for points in points_list]
            annotations = []
            for points in points_list:
                x, y = zip(*points)
                annotations.append({
                    'bbox': [(min(x), min(y)), (max(x), max(y))],
                    'bbox_mode': None,
                    'segmentation': points,
                    'category_id': 0,
                    'iscrowd': 0
                })

            tertiary_instances.append({
                'height': h,
                'width': w,
                'file_name': filepath,
                'annotations': annotations,
            })
    train_instances += tertiary_instances
    print(len(train_instances))

loss = WeightedLoss({loss_dict[l](): w for l, w in conf.loss_weights.items()})

optims = {
    'adam': torch.optim.Adam,
    'sgd': torch.optim.SGD
}

optim = optims[conf.optim](model.parameters(), lr=conf.lr)


train_dataset = dataset.BuildingSegmentationDataset(
    instances=train_instances,
    nclasses=conf.nclasses,
    resolution=conf.training_resolution,
    augment=augment,
    preprocess_fn=preprocess_fn,
    mode=conf.mode,
)

dev_dataset = dataset.BuildingSegmentationDataset(
    instances=dev_instances,
    nclasses=conf.nclasses,
    resolution=conf.training_resolution,
    augment=None,
    preprocess_fn=preprocess_fn,
    mode=conf.mode,
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=conf.batch_size,
    shuffle=True,
    num_workers=10,
)

dev_loader = torch.utils.data.DataLoader(
    dev_dataset,
    batch_size=conf.batch_size,
    shuffle=False,
    num_workers=10,
)


#model, optim = amp.initialize(model, optim, opt_level=conf.amp_opt_level)


scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optim, factor=conf.scheduler_factor, patience=conf.scheduler_patience
)


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


train_resize = MultiScaleResize(conf.mode, conf.training_scales)


best_score = 0
epoch = 0


train_fn = run.train_segment if conf.nclasses == 1 else run.train_damage
eval_fn = run.evaluate_segment if conf.nclasses == 1 else run.evaluate_damage

for epoch in range(epoch, conf.epochs):
    metrics = {'epoch': epoch}
    train_metrics = train_fn(model, optim, train_loader, loss, train_resize=train_resize, mode=conf.mode)
    metrics.update(train_metrics)

    dev_metrics = eval_fn(model, dev_loader, loss, mode=conf.mode)
    metrics.update(dev_metrics)

    if conf.mode != "dual":
        examples = run.sample_masks(model, dev_instances, preprocess_fn, n=3)
        metrics['examples'] = [wandb.Image(im, caption=f'mask:{ix}') for e in examples for ix, im in enumerate(e)]

    wandb.log(metrics)
    scheduler.step(metrics['loss'])
    score = metrics[conf.metric]

    if score > best_score:
        torch.save(model.state_dict(), os.path.join(wandb.run.dir, "state_dict.pth"))
        best_score = score
