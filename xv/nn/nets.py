from torchvision.models.resnet import BasicBlock, Bottleneck
from torch import nn

class DownscaleLayer(nn.Module):
    def __init__(self, inplanes, blocks, strides, block, nclasses, growth_rate=2):
        super().__init__()
        self.block = block
        features = []
        planes = inplanes
        for stride, nblock in zip(strides, blocks):
            planes *= growth_rate
            features.append(self._make_layer(inplanes, planes, stride, nblock))
            inplanes = planes * block.expansion
            
        self.features = nn.ModuleList(features)
        self.head = nn.Conv2d(inplanes, nclasses, kernel_size=1, padding=0)
        self._init_weights()

        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.modules():
            if isinstance(m, Bottleneck):
                nn.init.constant_(m.bn3.weight, 0)
            elif isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)


    def _make_layer(self, inplanes, planes, stride, nblocks):
        if stride != 1 or inplanes != planes * self.block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * self.block.expansion, kernel_size=1, stride=stride, bias=False), 
                nn.BatchNorm2d(planes * self.block.expansion)
            )
        layers = []
        layers.append(self.block(inplanes, planes, stride, downsample))
        inplanes = planes * self.block.expansion
        for _ in range(1, nblocks):
            layers.append(self.block(inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        for feature in self.features:
            x = feature(x)
        return self.head(x)
    
    
class XVNet(nn.Module):
    def __init__(self, building_seg, dmg_heatmap):
        super().__init__()
        self.building_seg = building_seg
        self.dmg_heatmap = dmg_heatmap
    
    def forward(self, x, downscale=False):
        if downscale:
            x = self.building_seg(x, apply_head=False)
            return self.dmg_heatmap(x)
        return self.building_seg(x)