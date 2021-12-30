from torch import nn
from mmcv.cnn import ResNet as MMCV_ResNet


class ResNet(MMCV_ResNet):
    def __init__(self, inplanes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(inplanes, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
