import torch
from typing import Optional

from pytorch_lightning import LightningModule, Trainer, Callback
from pytorch_lightning.utilities import rank_zero_warn


class SetSharingStrategyCallback(Callback):
    """Set sharing strategy when training starts."""

    def __init__(self,
                 strategy: Optional[str] = 'file_descriptor',
                 use_file_system: Optional[bool] = False):
        if use_file_system:
            strategy = 'file_system'
        self.strategy = strategy

    def on_before_accelerator_backend_setup(self, trainer: Trainer, pl_module: LightningModule, stage: Optional[str] = None) -> None:
        all_strategies = torch.multiprocessing.get_all_sharing_strategies()
        if self.strategy in all_strategies:
            torch.multiprocessing.set_sharing_strategy(self.strategy)
        else:
            rank_zero_warn(f'Sharing strategy {self.strategy} is not supported.'
                           f'All supported strategy are: {all_strategies}')
