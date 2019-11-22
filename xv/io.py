import yaml
import json
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from xv.nn.layers import FrozenBatchNorm2d
import torch
from xv.nn.nets import BoxClassifier
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from PIL import Image
import numpy as np
import pandas as pd
from PIL import Image
import albumentations as al
import shapely
from xv import dataset
from glob import glob
import pdb

TRAIN_DIR = '../../datasets/xview/train'
TEST_DIR = '../../datasets/xview/test'
SUPPL_DIR = '../../datasets/xview/tier3'
TERTIARY_DIR = '../../datasets/spacenet'

class Config:
    def __init__(self, fp):
        with open(fp) as file_path:
            if file_path.name.endswith('.json'):
                _conf = json.load(file_path)
            elif file_path.name.endswith('.yaml'):
                _conf = yaml.load(file_path)
                if 'wandb_version' in _conf:
                    _conf = {k:v['value'] for k,v in _conf.items() if isinstance(v,dict) and 'value' in v}
            for k,v in _conf.items():
                setattr(self, k, v)

def load_segmentation_model(conf, state_file=None):
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
        attention_type=conf.attention
    )
    
    if conf.load_weights:
        state_dict = torch.load(conf.load_weights)
        print(model.load_state_dict(state_dict))
    
    if conf.freeze_encoder_norm:
        model.encoder = FrozenBatchNorm2d.convert_frozen_batchnorm(model.encoder)

    if conf.freeze_decoder_norm:
        model.decoder = FrozenBatchNorm2d.convert_frozen_batchnorm(model.decoder)

    if state_file is not None:
        state_dict = torch.load(state_file)
        model.load_state_dict(state_dict)
        
    if torch.cuda.device_count() > 1:
        if conf.sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.DataParallel(model)

    preprocess_fn = get_preprocessing_fn(conf.encoder)
        
    return model, preprocess_fn


def load_damage_model(conf, state_file):
    backbone = resnet_fpn_backbone(conf.backbone, True)
    model = BoxClassifier(backbone, conf.nclasses)
    state_dict = torch.load(state_file)
    model.load_state_dict(state_dict)
    model = model.eval().cuda()
    return model

def load_img(img_path, preprocess_fn):
    image = np.array(Image.open(img_path))
    image = preprocess_fn(image)
    image = image.transpose(2,0,1)
    image = image.astype(np.float32)
    return torch.Tensor(image[None])

def load_dmg_img(img_path, image_mean = (0.485, 0.456, 0.406), image_std = (0.229, 0.224, 0.225)):
    image = np.array(Image.open(img_path))
    image = image.astype(np.float32)
    image /= 255.
    image = (image-image_mean)/image_std
    image = image.transpose(2,0,1)
    return torch.Tensor(image[None])


def _get_augment(conf):
    return al.Compose([
        al.HorizontalFlip(p=conf.aug_prob),
        al.VerticalFlip(p=conf.aug_prob),
        al.RandomRotate90(p=conf.aug_prob),
        al.Transpose(p=conf.aug_prob),
        al.GridDistortion(p=conf.aug_prob, distort_limit=.2),
        al.ShiftScaleRotate(p=conf.aug_prob),
        al.RandomBrightnessContrast(p=conf.aug_prob),
    ])

def _load_training_patch_data(conf, preprocess_fn=None):
    N_PATCHES = 4
    train_stems = pd.read_csv('config/train_stems.csv', header=None)[0]
    train_images = []
    train_masks = []
    for stem in train_stems:
        for patch in range(N_PATCHES):
            train_images.append(f'{TRAIN_DIR}/images_split/{stem}_{patch}_{conf.data_prefix}_disaster.png')
            train_masks.append(f'{TRAIN_DIR}/masks_split/{stem}_{patch}_{conf.data_prefix}_disaster.npy')
    
    train_dataset = dataset.ImageMaskDataset(
        image_paths=train_images,
        mask_paths=train_masks,
        nclasses=conf.nclasses,
        resolution=conf.training_resolution,
        augment=_get_augment(conf),
        preprocess_fn=preprocess_fn,
        mode=conf.mode,
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=conf.batch_size,
        shuffle=True,
        num_workers=conf.n_cpus,
    )
    
    return train_dataset, train_loader
    

def load_training_data(conf, preprocess_fn=None):
    if conf.train_patch:
        return _load_training_patch_data(conf, preprocess_fn)
    
    train_stems = pd.read_csv('config/train_stems.csv', header=None)[0]
    train_files = [f'{TRAIN_DIR}/labels/{stem}_{conf.data_prefix}_disaster.json' for stem in train_stems]
    train_instances = dataset.get_instances(train_files, filter_none=conf.filter_none)
    
    if conf.add_suppl:
        train_instances *= conf.train_repeat
        suppl_files = glob(f'{SUPPL_DIR}/labels/*{conf.data_prefix}_disaster.json')
        suppl_instances = dataset.get_instances(suppl_files, filter_none=conf.filter_none)
        train_instances += suppl_instances

    if conf.add_tertiary:
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

    train_dataset = dataset.BuildingSegmentationDataset(
        instances=train_instances,
        nclasses=conf.nclasses,
        resolution=conf.training_resolution,
        augment=_get_augment(conf),
        preprocess_fn=preprocess_fn,
        mode=conf.mode,
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=conf.batch_size,
        shuffle=True,
        num_workers=conf.n_cpus,
    )
    
    return train_dataset, train_loader


def load_dev_data(conf, preprocess_fn=None):
    dev_stems = pd.read_csv('config/dev_stems.csv', header=None)[0]
    dev_files = [f'{TRAIN_DIR}/labels/{stem}_{conf.data_prefix}_disaster.json' for stem in dev_stems]
    dev_instances = dataset.get_instances(dev_files, filter_none=conf.filter_none)
    
    dev_dataset = dataset.BuildingSegmentationDataset(
        instances=dev_instances,
        nclasses=conf.nclasses,
        resolution=conf.eval_resolution,
        augment=None,
        preprocess_fn=preprocess_fn,
        mode=conf.mode,
    )

    dev_loader = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size=conf.batch_size,
        shuffle=False,
        num_workers=conf.n_cpus,
    )
    
    return dev_dataset, dev_loader
