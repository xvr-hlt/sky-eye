from xv import manager
from PIL import Image
import fire
from xv.nn import *

def classify(pre_in, post_in, pre_out, post_out, device='cpu'):
    mod_man = manager.ModelManager(device=device)
    conf = get_model_config(pre_in, post_in)
    localization = mod_man.predict_localization(pre_in, **conf['localization_kwargs'])
    damage = mod_man.predict_damage(post_in, localization, **conf['damage_kwargs'])
    Image.fromarray(localization).save(pre_out)
    Image.fromarray(damage).save(post_out)    

def main():
    fire.Fire(classify)

if __name__ == "__main__":
    main()
