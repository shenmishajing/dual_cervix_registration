# Copyright (c) OpenMMLab. All rights reserved.
import platform
from mmcv.utils import Registry

if platform.system() != 'Windows':
    # https://github.com/pytorch/pytorch/issues/973
    import resource

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    base_soft_limit = rlimit[0]
    hard_limit = rlimit[1]
    soft_limit = min(max(4096, base_soft_limit), hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

PIPELINES = Registry('pipeline')
