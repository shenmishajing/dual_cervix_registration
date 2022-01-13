import os
import copy

from mmcv.cnn.utils.weight_init import _initialize


def get_log_dir(trainer):
    if trainer.checkpoint_callback is not None and trainer.checkpoint_callback.dirpath is not None:
        log_dir = os.path.dirname(trainer.checkpoint_callback.dirpath)
    else:
        if trainer.logger is not None:
            if trainer.weights_save_path != trainer.default_root_dir:
                # the user has changed weights_save_path, it overrides anything
                save_dir = trainer.weights_save_path
            else:
                save_dir = trainer.logger.save_dir or trainer.default_root_dir

            version = (
                trainer.logger.version
                if isinstance(trainer.logger.version, str)
                else f"version_{trainer.logger.version}"
            )
            log_dir = os.path.join(save_dir, str(trainer.logger.name), version)
        else:
            log_dir = trainer.weights_save_path

    return log_dir


def initialize(module, initialize_config):
    r"""Initialize a module.

    Args:
        module (``torch.nn.Module``): the module will be initialized.
        initialize_config (Mapping[str, Any] | Sequence[Mapping[str, Any]]):
            initialization configuration dict to define initializer.
            OpenMMLab has implemented 6 initializers including
            ``Constant``, ``Xavier``, ``Normal``, ``Uniform``,
            ``Kaiming``, and ``Pretrained``.
    """
    if not isinstance(initialize_config, (dict, list)):
        raise TypeError(f'initialize_config must be a dict or a list of dict, \
                but got {type(initialize_config)}')

    if isinstance(initialize_config, dict):
        initialize_config = [initialize_config]

    for cfg in initialize_config:
        # should deeply copy the original config because cfg may be used by
        # other modules, e.g., one initialize_config shared by multiple bottleneck
        # blocks, the expected cfg will be changed after pop and will change
        # the initialization behavior of other modules
        cp_cfg = copy.deepcopy(cfg)
        model_path = cp_cfg.pop('model_path', '')
        if 'pretrained' in cp_cfg:
            cp_cfg = {'type': 'Pretrained', 'checkpoint': cp_cfg['pretrained']}
        m = module
        for p in model_path.split('.'):
            if p:
                m = getattr(m, p)
        _initialize(m, cp_cfg)
