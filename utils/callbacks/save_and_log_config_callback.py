import os
from typing import Optional

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.utilities.cloud_io import get_filesystem
from pytorch_lightning.utilities.cli import SaveConfigCallback

from ..utils import get_log_dir


class SaveAndLogConfigCallback(SaveConfigCallback):
    """Saves and logs a LightningCLI config to the log_dir when training starts."""

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: Optional[str] = None) -> None:
        # save the config in `setup` because (1) we want it to save regardless of the trainer function run
        # and we want to save before processes are spawned
        if trainer.logger is not None:
            if 'subcommand' in self.config:
                trainer.logger.log_hyperparams(self.config[self.config['subcommand']])

            log_dir = get_log_dir(trainer)
            assert log_dir is not None
            config_path = os.path.join(log_dir, self.config_filename)
            if (self.overwrite or not os.path.isfile(config_path)) and trainer.is_global_zero:
                # save only on rank zero to avoid race conditions on DDP.
                # the `log_dir` needs to be created as we rely on the logger to do it usually
                # but it hasn't logged anything at this point
                get_filesystem(log_dir).makedirs(log_dir, exist_ok = True)
                self.parser.save(
                    self.config, config_path, skip_none = False, overwrite = self.overwrite, multifile = self.multifile
                )
