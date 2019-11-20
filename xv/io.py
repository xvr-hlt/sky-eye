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

def load_segmentation_model(conf_file, state_file=None):
    conf = Config(conf_file)

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

    if conf.freeze_encoder_norm:
        model.encoder = FrozenBatchNorm2d.convert_frozen_batchnorm(model.encoder)

    if conf.freeze_decoder_norm:
        model.decoder = FrozenBatchNorm2d.convert_frozen_batchnorm(model.decoder)

    preprocess_fn = get_preprocessing_fn(conf.encoder)
    
    if state_file is not None:
        state_dict = torch.load(state_file)
        model.load_state_dict(state_dict)

    return model, preprocess_fn


def load_img(img_path, preprocess_fn):
    image = np.array(Image.open(img_path))
    image = preprocess_fn(image)
    image = image.transpose(2,0,1)
    image = image.astype(np.float32)
    return torch.Tensor(image[None])

def load_damage_model(conf_file, state_file):
    conf = Config(conf_file)
    backbone = resnet_fpn_backbone(conf.backbone, True)
    model = BoxClassifier(backbone, conf.nclasses)
    state_dict = torch.load(state_file)
    model.load_state_dict(state_dict)
    model = model.eval().cuda()
    return model

def load_dmg_img(img_path, image_mean = (0.485, 0.456, 0.406), image_std = (0.229, 0.224, 0.225)):
    image = np.array(Image.open(img_path))
    image = image.astype(np.float32)
    image /= 255.
    image = (image-image_mean)/image_std
    image = image.transpose(2,0,1)
    return torch.Tensor(image[None])