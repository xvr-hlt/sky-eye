import torch
from torch import nn
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.faster_rcnn import TwoMLPHead

class BoxClassifier(nn.Module):
    def __init__(self, backbone, nclasses, featmap_names=[0, 1, 2, 3]):
        super().__init__()
        self.backbone = backbone
        
        self.roi_pool = MultiScaleRoIAlign(
            featmap_names=featmap_names,
            output_size=7,
            sampling_ratio=2
        )
        
        out_channels = self.backbone.out_channels
        resolution = self.roi_pool.output_size[0]
        representation_size = 1024
        self.box_head = TwoMLPHead(out_channels * resolution ** 2, representation_size)
        self.head = nn.Linear(representation_size, nclasses)
        
    
    def forward(self, x, boxes):
        i, _, w, h = x.shape
        sizes = [(w,h) for _ in range(i)]
        x = self.backbone(x)
        x = self.roi_pool(x, boxes, sizes)
        x = self.box_head(x)
        x = self.head(x)
        return x
