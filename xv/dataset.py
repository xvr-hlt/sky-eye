import torch
import torchvision
import cv2
from collections import defaultdict
import copy
from glob import glob
from tqdm import tqdm_notebook as tqdm
from PIL import Image, ImageDraw
import json
import shapely.wkt
import itertools

import numpy as np

DAMAGE_CLASSES = ('no-damage', 'minor-damage', 'major-damage', 'destroyed')
DAMAGE_CAT_IDS = {i:ix for ix, i in enumerate(DAMAGE_CLASSES)}


def get_instances(files, bbox_mode=None, filter_none=False):
    dataset_dicts = []
    for file in tqdm(files):
        with open(file) as f:
            i = json.load(f)

        h,w = i['metadata']['height'], i['metadata']['width']
        objs = []
        
        for feat in i['features']['xy']:
            prop = feat['properties']
            if prop.get('subtype') == 'un-classified':
                continue
            poly = shapely.wkt.loads(feat['wkt'])
            points = list(poly.exterior.coords)
            points = [(x/w, y/h) for x,y in points]
            px, py = zip(*points)
            objs.append({
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": None,
                "segmentation": points,
                "category_id": DAMAGE_CAT_IDS[prop['subtype']] if 'subtype' in prop else 0,
                "iscrowd": 0
            })
            
        if filter_none and not objs:
            continue
            
        dataset_dicts.append({
            'height': h,
            'width': w,
            'file_name': file.replace('/labels/', '/images/').replace('json', 'png'),
            'annotations': objs
        })
    return dataset_dicts

def get_mask(polygons, w, h):
    img = Image.new('L', (w, h), 0)    
    draw = ImageDraw.Draw(img)
    for polygon in polygons:
        draw.polygon(tuple((x*w, y*h) for x,y in polygon), outline=1, fill=1)
    return np.array(img).astype(np.float32)

class BuildingSegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, instances, nclasses, augment=None, resolution=1024, preprocess_fn=None, mode=None):
        self.instances = instances
        self.nclasses = nclasses
        self.augment = augment
        self.preprocess_fn = preprocess_fn
        self.resolution = resolution
        self.mode = mode

    def __getitem__(self, ix):
        instance = self.instances[ix]
        image = np.array(Image.open(instance['file_name']))
        image = cv2.resize(image, (self.resolution, self.resolution))
        
        polygons_by_class = [[] for _ in range(self.nclasses)]
        for a in instance['annotations']:
            polygons_by_class[a['category_id']].append(a['segmentation'])
        
        
        w, h, _ = image.shape
        mask = [get_mask(polygons, w, h) for polygons in polygons_by_class]
        
        if self.augment:
            aug = self.augment(image=image, masks=mask)
            image, mask = aug['image'], aug['masks']
        
        mask = np.stack(mask)
        
        if self.mode == 'ordinal':
            for i in reversed(range(mask.shape[0])):
                mask[:i] = np.logical_or(mask[:i], mask[i])

        if self.mode == 'categorical':
            mask_bool = mask.sum(0) > 0
            mask = mask.argmax(0)
            mask = mask_bool, mask
            
        image = image.astype(np.float32)
        image = self.transform_image(image)
        
        return image, mask

    def transform_image(self, image):
        if self.preprocess_fn:
            image = self.preprocess_fn(image)
        else:
            image /= 255.
        image = image.transpose(2,0,1) # C x W x H
        return image.astype(np.float32)

    def inverse_transform_image(self, image):
        image = (image*255).transpose(1,2,0).astype('uint8')
        return Image.fromarray(image)

    def __len__(self):
        return len(self.instances)