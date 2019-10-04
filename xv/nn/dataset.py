import torch
import torchvision
import cv2
from collections import defaultdict
import copy
from glob import glob
from tqdm import tqdm_notebook as tqdm
from PIL import Image, ImageDraw
import json

import numpy as np

def get_mask(polygons, w, h):
    img = Image.new('L', (w, h), 0)
    draw = ImageDraw.Draw(img)
    for vertices in polygons:
        draw.polygon(tuple((x*w, y*h) for x,y in vertices), outline=1, fill=1)
    return np.array(img)

class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, path, resolution, augment=None):
        self.instances = self._get_instances(path)
        self.resolution = resolution
        self.augment = augment
    
    def _get_images(self, instance):
        raise NotImplementedError
    
    def _get_instances(self, path):
        raise NotImplementedError
    
    def _get_masks(self, instance):
        raise NotImplementedError
        
    def _augment(self, masks, images):
        return masks, images
        
    def transform_image(self, image):
        image = cv2.resize(image, self.resolution)
        image = image.astype(np.float32) / 255.
        image = image.transpose(2,0,1) # C x W x H
        return image
    
    def inverse_transform_image(self, image):
        image = (image*255).transpose(1,2,0).astype('uint8')
        return Image.fromarray(image)
    
    def __getitem__(self, ix):
        instance = self.instances[ix]
        images, masks = self._get_images(instance), self._get_masks(instance)
        images, masks = self._augment(images, masks)
        images = {k: self.transform_image(v) for k, v in images.items()}
        return {'images': images, 'masks': masks}
    
    def __len__(self):
        return len(self.instances)

class XViewSegmentationDataset(SegmentationDataset):
    DAMAGE_CLASSES = {'no-damage':0, 'minor-damage':1, 'major-damage':2, 'destroyed':3}

    def __init__(self, dmg_downscale_ratio=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        w,h = self.resolution
        self.dmg_resolution = w//dmg_downscale_ratio, h//dmg_downscale_ratio
        if self.augment:
            self.augment = copy.copy(self.augment)
            self.augment.add_targets({
                'post': 'image',
            })

    def _get_images(self, instance):
        return {
            'image':np.array(Image.open(instance['pre']['file_path'])), 
            'post': np.array(Image.open(instance['post']['file_path']))
        }
        
    def _get_masks(self, instance):        
        buildings_mask = get_mask((v['poly'] for v in instance['pre']['features'].values()), *self.resolution).astype(np.float32)
        
        damage_hm = np.zeros((len(self.DAMAGE_CLASSES), *reversed(self.dmg_resolution)), dtype=np.float32)
        
        damage = defaultdict(list)
        for f in instance['post']['features'].values():
            if f['subtype'] != 'un-classified':
                damage[f['subtype']].append(f['poly'])

        for damage_type, polys in damage.items():
            damage_hm[self.DAMAGE_CLASSES[damage_type]] = get_mask(polys, *self.dmg_resolution).astype(np.float32)

        return {'buildings': buildings_mask, 'damage': damage_hm}
    
    def _augment(self, images, masks):
        if self.augment is None:
            return images, masks
        mask_list = [masks['buildings']]
        mask_list += [m for m in masks['damage']]
        aug = self.augment(**images, masks=mask_list)
        images = {k:aug[k] for k in images}
        
        bmask, *dmasks = aug['masks']
        masks = {
            'buildings': bmask.reshape(1,*bmask.shape),
            'damage': np.stack(dmasks)
        }
        
        return images, masks
    
    def _get_instances(self, path):
        instances = []
        labels = sorted(glob(f'{path}/labels/*.json'))
        for post, pre in tqdm(zip(labels[::2], labels[1::2]), total=len(labels)//2):
            instance = {}
            with open(pre) as f:
                instance['pre'] = process_label(json.load(f), base_dir=path)
            with open(post) as f:
                instance['post'] = process_label(json.load(f), base_dir=path)
            instances.append(instance)
        return instances
    
def process_poly(s, scale_w, scale_h):
    pts = []
    for xy in s[s.find('((')+2:s.rfind('))')].split(', '):
        x, y = xy.split(' ')
        pts.append((float(x)*scale_w, float(y)*scale_h,))
    return tuple(pts)

def process_label(label, base_dir, feature_type='xy', include_metadata=True, relative_pts=True):
    if relative_pts:
        scale_w, scale_h = 1/label['metadata']['width'], 1/label['metadata']['height']
    else:
        scale_w, scale_h= 1., 1.

    features = {}
    for f in label['features'][feature_type]:
        features[f['properties']['uid']] = {
             'feature_type': f['properties']['feature_type'],
             'subtype': f['properties'].get('subtype'),
             'poly':process_poly(f['wkt'], scale_h=scale_h, scale_w=scale_w)
         }
        
    return {
        'metadata': label['metadata'] if include_metadata else None,
        'features': features,
        'file_path': f"{base_dir}/images/{label['metadata']['img_name']}"
    }
    return label