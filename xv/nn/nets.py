from xv.nn.oc import oc
import torch
from torch import nn
from torchvision.ops import MultiScaleRoIAlign
import segmentation_models_pytorch as smp
from torchvision.models.detection.faster_rcnn import TwoMLPHead
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool
from collections import OrderedDict
from xv.nn.bifpn import BIFPN


class BoxClassifier(nn.Module):
    def __init__(self, encoder_name,
                 nclasses=4,
                 use_bifpn=False,
                 features=(4,3,2,1),
                 out_channels=256, 
                 dropout=.1,
                 representation_size=1024):
        super().__init__()
        self.encoder = smp.encoders.get_encoder(encoder_name, 'imagenet')
        self.features = features
        
        self.use_bifpn = use_bifpn
        
        out_shapes = smp.encoders.encoders[encoder_name]['out_shapes']
        out_shapes = [out_shapes[f] for f in features]
        
        if not self.use_bifpn:
            self.fpn = FeaturePyramidNetwork(out_shapes,
                                             out_channels,
                                             extra_blocks=LastLevelMaxPool())
        else:
            self.fpn = BIFPN(out_shapes, out_channels, num_outs=len(features))
        
        self.roi_pool = MultiScaleRoIAlign(
            featmap_names=features,
            output_size=7,
            sampling_ratio=2
        )
        
        self.dropout = nn.Dropout2d(p=dropout)
        
        resolution = self.roi_pool.output_size[0]
        
        self.box_head = TwoMLPHead(out_channels * resolution ** 2, representation_size)
        self.head = nn.Linear(representation_size, nclasses)

    def forward(self, x, boxes):
        i, _, w, h = x.shape
        sizes = [(w, h) for _ in range(i)]
        x = self.encoder(x)
        x = [x[i] for i in self.features] if self.use_bifpn else OrderedDict((i, x[i]) for i in self.features)
        x = self.fpn(x)
        if self.use_bifpn: x = OrderedDict((ix, xi) for ix, xi in enumerate(x))
        x = self.roi_pool(x, boxes, sizes)
        x = self.dropout(x)
        x = self.box_head(x)
        x = self.head(x)
        return x


class OCSegment(nn.Module):
    def __init__(self, model, decoder_channels=128):
        super().__init__()
        self._model = model
        self.context = oc.InterlacedSparseSelfAttention(decoder_channels)

    def forward(self, x):
        features = self._model.encoder(x)
        out = self._model.decoder(*features)
        out = self.context(out)
        return self._model.segmentation_head(out)


class DualWrapper(nn.Module):
    INNER_CHANNELS_DEFAULT = 128
    
    def __init__(self, m, inner_channels=None):
        super().__init__()
        
        inner_channels = inner_channels or self.INNER_CHANNELS_DEFAULT
        
        channels_in = m.decoder.final_conv.in_channels
        channels_out = m.decoder.final_conv.out_channels
        
        m.decoder.final_conv = nn.Identity()
        
        self._model = m

        self.pre_mixin = nn.Sequential(
            nn.Conv2d(channels_in, inner_channels, kernel_size=1),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU()
        )

        self.post_mixin = nn.Sequential(
            nn.Conv2d(channels_in, inner_channels, kernel_size=1),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU()
        )
        
        self.head = nn.Conv2d(inner_channels, channels_out, kernel_size=1)

    def forward(self, x):
        pre, post = x[:, :3], x[:, 3:]
        
        pre_out = self.pre_mixin(self._model.decoder(self._model.encoder(pre)))
        post_out = self.post_mixin(self._model.decoder(self._model.encoder(post)))
        
        return self.head(post_out - pre_out)
