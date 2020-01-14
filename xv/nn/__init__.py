from .oc import *
from PIL import Image
from xv import manager

def get_model_config(*ims):
     return [
        {
            "localization_kwargs": {"run_ids": ["8bh5rurv"], "thresh": 0.2},
            "damage_kwargs": {
                "damseg_ids": ["tyi7q17m"],
                "damod_ids": [],
                "seg_mean_ag": False,
            },
        },
        {
            "localization_kwargs": {"run_ids": ["qoijsx0h"], "thresh": 0.5},
            "damage_kwargs": {
                "damseg_ids": ["hsxmom00"],
                "damod_ids": [],
                "seg_mean_ag": True,
            },
        },
    ][categorise_image(*[Image.open(i) for i in ims])]
