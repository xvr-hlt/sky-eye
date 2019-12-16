import torch
import torchvision
import cv2
from collections import defaultdict
import copy
from glob import glob
from tqdm import tqdm
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
    def __init__(self, instances, nclasses, augment=None, resolution=1024, preprocess_fn=None, dual_input=False, mode=None):
        super().__init__()
        self.instances = instances
        self.nclasses = nclasses
        self.augment = augment
        self.preprocess_fn = preprocess_fn
        self.resolution = resolution
        self.dual_input = dual_input
        self.mode = mode
        if self.dual_input and self.augment:
            self.augment.add_targets({'image_post': 'image'})


    def get_image(self, ix):
        def load_im(f): return  cv2.resize(np.array(Image.open(fn)), (self.resolution, self.resolution))
        fn = self.instances[ix]['file_name']
        if not self.dual_input:
            return load_im(fn)
        pre_fn = fn.replace('post_disaster', 'pre_disaster')
        post_fn = fn.replace('pre_disaster', 'post_disaster')
        return load_im(pre_fn), load_im(post_fn)
        
    def get_mask(self, ix):
        polygons_by_class = [[] for _ in range(self.nclasses)]
        for a in self.instances[ix]['annotations']:
            polygons_by_class[a['category_id']].append(a['segmentation'])
        return [get_mask(polygons, self.resolution, self.resolution) for polygons in polygons_by_class]
        
    def __getitem__(self, ix):
        image = self.get_image(ix)
        
        mask = self.get_mask(ix)
        
        if self.augment and not self.dual_input:
            aug = self.augment(image=image, masks=mask)
            image, mask = aug['image'], aug['masks']
            image = image.astype(np.float32)
            image = self.transform_image(image)
        
        if self.augment and self.dual_input:
            image_pre, image_post = image
            aug = self.augment(image=image_pre, image_post=image_post, masks=mask)
            image_pre, image_post, mask = aug['image'], aug['image_post'], aug['masks']
            image = image_pre, image_post
        
        if self.dual_input:
            image_pre, image_post = image
            image_pre = self.transform_image(image_pre.astype(np.float32))
            image_post = self.transform_image(image_post.astype(np.float32))            
            image = np.concatenate([image_pre, image_post])
        
        mask = np.stack(mask)
        
        if self.mode == 'categorical':
            mask_bool = mask.sum(0) > 0
            mask = mask.argmax(0)
            mask = mask_bool, mask
        
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
    
    
class DamageClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, instances, nclasses,
                 resolution=1024, augment=None, image_mean = (0.485, 0.456, 0.406), image_std = (0.229, 0.224, 0.225)):
        super().__init__()
        self.instances = instances
        self.resolution = resolution
        self.nclasses = nclasses
        self.augment = augment
        self.image_mean = image_mean
        self.image_std = image_std

    def __getitem__(self, ix):
        instance = self.instances[ix]
        image = np.array(Image.open(instance['file_name']))
        image = cv2.resize(image, (self.resolution, self.resolution))
        
        boxes, labels = [], []
        
        for a in instance['annotations']:
            boxes.append(a['bbox'])
            labels.append(a['category_id'])
        
        h,w,_ = image.shape
        boxes = np.array(boxes)
        boxes = np.clip(boxes, 0, 1)
        
        boxes[:,0] *= w
        boxes[:,1] *= h
        boxes[:,2] *= w
        boxes[:,3] *= h
        boxes = [b for b in boxes]
        
        if self.augment:
            aug = self.augment(image=image, bboxes=boxes, labels=labels)
            image, boxes, labels = aug['image'], aug['bboxes'], aug['labels']
        
        image = self.transform_image(image)
        
        return image, boxes, labels

    def transform_image(self, image):
        image = image.astype(np.float32)
        image /= 255.
        image = (image-self.image_mean)/self.image_std
        image = image.transpose(2,0,1)
        return image

    def inverse_transform_image(self, image):
        image = image.transpose(1,2,0)
        image = (image*self.image_std) + self.image_mean
        image = (image*255).astype('uint8')
        return Image.fromarray(image)

    def __len__(self):
        return len(self.instances)


class ImageMaskDataset(BuildingSegmentationDataset):
    def __init__(self, image_paths, mask_paths, *super_args, **super_kwargs):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        super().__init__(instances=None, *super_args, **super_kwargs)
        
    def __len__(self):
        return len(self.image_paths)
    
    def get_image(self, ix):
        return np.array(Image.open(self.image_paths[ix]))
    
    def get_mask(self, ix):
        return [np.load(self.mask_paths[ix])]