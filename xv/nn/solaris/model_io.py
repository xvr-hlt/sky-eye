import os
import torch
from warnings import warn
import requests
import numpy as np
from tqdm import tqdm
from .zoo import model_dict
from ..solaris import weights_dir

LAYER_BLACKLIST = {'final.0.weight', 'final.0.bias', 'encoder_stages.0.0.weight'}


def get_model(model_name, framework, model_path=None, pretrained=False,
              custom_model_dict=None):
    """Load a model from a file based on its name."""
    if custom_model_dict is not None:
        md = custom_model_dict
    else:
        md = model_dict.get(model_name, None)
        if md is None:  # if the model's not provided by solaris
            raise ValueError(f"{model_name} can't be found in solaris.")
    if model_path is None or custom_model_dict is not None:
        model_path = os.path.join(weights_dir, md.get('filename'))
    model = md.get('arch')()
    if model is not None and pretrained:
        try:
            model = _load_model_weights(model, model_path, framework)
        except (OSError, FileNotFoundError):
            warn(f'The model weights file {model_path} was not found.'
                 ' Attempting to download from the SpaceNet repository.')
            weight_path = _download_weights(md)
            model = _load_model_weights(model, weight_path, framework)

    return model


def _load_model_weights(model, path, framework):
    """Backend for loading the model."""

    if framework.lower() in ['torch', 'pytorch']:
        # pytorch already throws the right error on failed load, so no need
        # to fix exception
        if torch.cuda.is_available():
            try:
                loaded = torch.load(path)
            except FileNotFoundError:
                # first, check to see if the weights are in the default sol dir
                default_path = os.path.join(weights_dir,
                                            os.path.split(path)[1])
                loaded = torch.load(path)
        else:
            try:
                loaded = torch.load(path, map_location='cpu')
            except FileNotFoundError:
                default_path = os.path.join(weights_dir,
                                            os.path.split(path)[1])
                loaded = torch.load(path, map_location='cpu')

        loaded = {k:v for k,v in loaded.items() if k not in LAYER_BLACKLIST}
        model.load_state_dict(loaded, strict=False)
        return model


def _download_weights(model_dict):
    """Download pretrained weights for a model."""
    weight_url = model_dict.get('weight_url', None)
    filename = model_dict.get('filename', weight_url.split('/')[-1])
    weight_dest_path = os.path.join(weights_dir, filename)
    if weight_url is None:
        raise KeyError("Can't find the weights file.")
    else:
        r = requests.get(weight_url, stream=True)
        if r.status_code != 200:
            raise ValueError('The file could not be downloaded. Check the URL'
                             ' and network connections.')
        total_size = int(r.headers.get('content-length', 0))
        block_size = 1024
        with open(weight_dest_path, 'wb') as f:
            for chunk in tqdm(r.iter_content(block_size),
                              total=np.ceil(total_size//block_size),
                              unit='KB', unit_scale=False):
                if chunk:
                    f.write(chunk)

    return weight_dest_path
