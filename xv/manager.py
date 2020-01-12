from xv import io
import ttach as tta
from imantics import Polygons
from xv.tta import BoxClassifierTTA
import torch
import numpy as np
from PIL import Image
import xv
import os

class ModelManager:
    _DEFAULT_BASE_DIR = f"{os.path.split(xv.__file__)[0]}/models"
    
    def __init__(self, base_dir=None, device='cpu', size=(1024,1024)):
        self.base_dir = base_dir or self._DEFAULT_BASE_DIR
        self.device = device
        self.size = size

    @staticmethod
    def load_img(img_path, image_mean = (0.485, 0.456, 0.406), image_std = (0.229, 0.224, 0.225)):
        image = np.array(Image.open(img_path))
        image = image.astype(np.float32)
        image /= 255.
        image = (image-image_mean)/image_std
        image = image.transpose(2,0,1)
        return torch.Tensor(image[None])

    @torch.no_grad()
    def predict_seg(self, im, model_subdir, run_id):
        path = f'{self.base_dir}/{model_subdir}/{run_id}'
        conf = io.Config(f'{path}/config.yaml')
        model, _ = io.load_segmentation_model(conf, f'{path}/state_dict.pth')
        model = model.eval().to(self.device)
        model = tta.SegmentationTTAWrapper(model, tta.aliases.d4_transform(), merge_mode='mean')
        return model(im.to(self.device)).sigmoid().cpu().numpy()[0]

    def predict_seg_dam(self, im, localization, run_id, mean_ag=False):
        probs = self.predict_seg(im, 'dam_seg', run_id)
        probs = probs / probs.sum(0)
        if mean_ag:
            _probs = np.zeros(probs.shape)
            for poly in Polygons.from_mask(localization):
                poly_mask = Polygons.create([poly]).mask(*self.size).array
                poly_prob = probs[:,poly_mask].mean(1, keepdims=True)
                _probs[:,poly_mask] = poly_prob
                probs = _probs
        return probs

    @torch.no_grad()
    def predict_od_dam(self, im, localization, run_id):
        probs = np.zeros((4, *self.size))
        polys = Polygons.from_mask(localization)
        polypoints = polys.points
        
        if not polypoints:
            return probs
        
        path = f'{self.base_dir}/dam_od/{run_id}'
        conf = io.Config(f'{path}/config.yaml')
        model = io.load_damage_model(conf, f'{path}/state_dict.pth')
        model = BoxClassifierTTA(model)
        model = model.eval().to(self.device)

        boxes = torch.Tensor([[min(p[:,0]), min(p[:,1]), max(p[:,0]), max(p[:,1])] for p in polypoints])
        box_probs = model(im.to(self.device), [boxes.to(self.device)]).sigmoid().cpu().numpy()
        box_probs /= box_probs.sum(1, keepdims=True)
        
        for poly, box_prob in zip(polypoints, box_probs):
            poly_mask = Polygons([poly]).mask(*self.size).array
            probs[:, poly_mask] = box_prob[:,None]

        return probs

    def predict_localization(self, pre_im_path, run_ids, thresh=0.2):
        pre_im = self.load_img(pre_im_path)
        out = self.predict_seg(pre_im, 'loc', run_ids[0])
        for run_id in run_ids[1:]:
            out += self.predict_seg(pre_im, 'loc', run_id)
        out /= len(run_ids)
        return (out > thresh).astype(np.uint8)[0]
    
    def predict_damage(self, post_im_path, localization, damseg_ids, damod_ids, seg_mean_ag=False):
        post_im = self.load_img(post_im_path)
        localization = localization.astype(np.bool)
        probs = np.zeros((4, *self.size))
        
        for run_id in damseg_ids:
            probs += self.predict_seg_dam(post_im, localization, run_id, seg_mean_ag)
        
        for run_id in damod_ids:
            probs += self.predict_od_dam(post_im, localization, run_id)
        
        probs = probs.argmax(0).astype(np.uint8) + 1
        probs *= localization
        
        return probs