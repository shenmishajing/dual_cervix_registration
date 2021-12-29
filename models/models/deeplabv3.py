from torch import nn
from torchvision.models.segmentation.deeplabv3 import DeepLabV3 as _DeepLabV3


class DeepLabV3(_DeepLabV3):
    def __init__(self, backbone: nn.Module, classifier: nn.Module, aux_classifier: nn.Module = None):
        super().__init__(backbone, classifier, None)

    def forward(self, x):
        result = super().forward(x)

        return result['out']
