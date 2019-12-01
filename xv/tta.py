import torch
from ttach import functional as tF
from torch import nn

class BoxClassifierTTA(nn.Module):
    def __init__(self, model):
        super().__init__()
        self._model = model
        
    @staticmethod
    def rot90_boxes(boxes, h, w, k=1):
        w, h = float(w), float(h)
        x_min, y_min, x_max, y_max = boxes[:,0]/w, boxes[:,1]/h, boxes[:,2]/w, boxes[:,3]/h
        for _ in range(k):
            x_min, y_min, x_max, y_max = y_min, 1 - x_max, y_max, 1 - x_min
        return torch.stack([x_min*w, y_min*h, x_max*w, y_max*h], 1)
    
    @staticmethod
    def vflip_boxes(boxes, h, w):
        boxes_flipped = boxes.clone()
        boxes_flipped[:,1] = h - boxes[:,3]
        boxes_flipped[:,3] = h - boxes[:,1]
        return boxes_flipped
    
    def forward(self, x, boxes):
        out = []
        i, _, h, w = x.shape
        
        for k in range(4):
            x_rotate = tF.rot90(x, k=k)
            boxes_rotate = [self.rot90_boxes(boxs, h, w, k=k) for boxs in boxes]
            out.append(self._model(x_rotate, boxes_rotate))
            
            x_rotate_flip = tF.vflip(x_rotate)
            boxes_rotate_flip = [self.vflip_boxes(boxs, h, w) for boxs in boxes_rotate]
            out.append(self._model(x_rotate, boxes_rotate))

        return torch.stack(out).mean(0)