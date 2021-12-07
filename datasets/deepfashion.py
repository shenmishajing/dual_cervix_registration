# Copyright (c) OpenMMLab. All rights reserved.
from .coco import CocoDataset


class DeepFashionDataset(CocoDataset):

    CLASSES = ('top', 'skirt', 'leggings', 'dress', 'outer', 'pants', 'bag',
               'neckwear', 'headwear', 'eyeglass', 'belt', 'footwear', 'hair',
               'skin', 'face')
