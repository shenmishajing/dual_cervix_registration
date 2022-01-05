import os
from typing import Dict, Optional

from weakref import proxy
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import _METRIC
from pytorch_lightning.callbacks import ModelCheckpoint as _ModelCheckpoint


class ModelCheckpoint(_ModelCheckpoint):
    CHECKPOINT_NAME_BEST = "best"

    def __init__(
            self,
            save_best: Optional[bool] = None,
            *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.save_best = save_best

    def save_checkpoint(self, trainer: "pl.Trainer") -> None:
        """Performs the main logic around saving a checkpoint.

        This method runs on all ranks. It is the responsibility of `trainer.save_checkpoint` to correctly handle the
        behaviour in distributed training, i.e., saving only on rank 0 for data parallel use cases.
        """
        epoch = trainer.current_epoch
        global_step = trainer.global_step

        self._validate_monitor_key(trainer)

        # track epoch when ckpt was last checked
        self._last_global_step_saved = global_step

        # what can be monitored
        monitor_candidates = self._monitor_candidates(trainer, epoch = epoch, step = global_step)

        # callback supports multiple simultaneous modes
        # here we call each mode sequentially
        # Mode 1: save the top k checkpoints
        self._save_top_k_checkpoint(trainer, monitor_candidates)
        # Mode 2: save monitor=None checkpoints
        self._save_none_monitor_checkpoint(trainer, monitor_candidates)
        # Mode 3: save last checkpoints
        self._save_last_checkpoint(trainer, monitor_candidates)
        # Mode 4: save best checkpoints
        self._save_best_checkpoint(trainer, monitor_candidates)

        # notify loggers
        if trainer.is_global_zero and trainer.logger:
            trainer.logger.after_save_checkpoint(proxy(self))

    def _save_best_checkpoint(self, trainer: "pl.Trainer", monitor_candidates: Dict[str, _METRIC]) -> None:
        if not self.save_best:
            return

        filepath = self.format_checkpoint_name(monitor_candidates, self.CHECKPOINT_NAME_BEST)

        if trainer.is_global_zero:
            if self._fs.lexists(filepath):
                self._fs.rm_file(filepath)
            if self._fs.protocol == 'file':
                os.symlink(os.path.basename(self.best_model_path), filepath)
            else:
                self._fs.cp_file(self.best_model_path, filepath)
