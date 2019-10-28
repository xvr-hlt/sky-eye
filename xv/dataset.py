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

def get_instances(files, bbox_mode=None):
    dataset_dicts = []
    for file in tqdm(files):
        with open(file) as f:
            i = json.load(f)

        objs = []
        for feat in i['features']['xy']:
            poly = shapely.wkt.loads(feat['wkt'])
            xy = list(poly.exterior.coords)
            px, py = zip(*xy)
            mask = list(itertools.chain.from_iterable(xy))
            objs.append({
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": None,
                "segmentation": [mask],
                "category_id": 0,
                "iscrowd": 0
            })

        dataset_dicts.append({
            'height': i['metadata']['height'],
            'width': i['metadata']['width'],
            'file_name': file.replace('/labels/', '/images/').replace('json', 'png'),
            'annotations': objs
        })
    return dataset_dicts

def get_mask(polygons, w, h):
    img = Image.new('L', (w, h), 0)    
    draw = ImageDraw.Draw(img)
    for polygon in polygons:
        xs, ys = polygon[::2], polygon[1::2]
        draw.polygon(tuple((x, y) for x,y in zip(xs, ys)), outline=1, fill=1)
    return np.array(img).astype(np.float32)

class BuildingSegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, instances, augment=None, preprocess_fn=None):
        self.instances = instances
        self.augment = augment
        self.preprocess_fn = preprocess_fn

    def __getitem__(self, ix):
        instance = self.instances[ix]
        image = np.array(Image.open(instance['file_name']))
        
        polygons = [a['segmentation'][0] for a in instance['annotations']]
        w, h, _ = image.shape
        mask = get_mask(polygons, w, h)
        
        if self.augment:
            aug = self.augment(image=image, mask=mask)
            image, mask = aug['image'], aug['mask']
            
        mask = mask.reshape(1, *mask.shape)
        
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