from .oc import *
from PIL import Image
from xv import manager

def get_model_config(*ims):
    return [
        {
            'localization_kwargs': {
                'run_ids': ['8bh5rurv', 'cnqyhmsb', 'ri1dvvam', 'ilhz9vkq'],
                'thresh': 0.2
            },
            'damage_kwargs': {
                'damseg_ids': ['tyi7q17m','ngmxec6s', 'vrgbzaup', 'xei17o3b', '10gtu54s'],#'443t9312', 'pxcuoulb'],#
                'damod_ids': ['kx9yzcb9', 'y7n4yzya', 'c1lit0gf'],
                'seg_mean_ag': False
            }
        },
        {
            'localization_kwargs': {
                'run_ids': ['qoijsx0h'],
                'thresh': 0.5
            },
            'damage_kwargs': {
                'damseg_ids': ['hsxmom00'],# ['0gvvydkt'],
                'damod_ids': [],
                'seg_mean_ag': True
            }
        },
    ][categorise_image(*[Image.open(i) for i in ims])]
