import os
from typing import Optional

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.utilities.cloud_io import get_filesystem
from pytorch_lightning.utilities.cli import SaveConfigCallback


class SaveAndLogConfigCallback(SaveConfigCallback):
    """Saves and logs a LightningCLI config to the log_dir when training starts."""

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: Optional[str] = None) -> None:
        # save the config in `setup` because (1) we want it to save regardless of the trainer function run
        # and we want to save before processes are spawned
        if trainer.logger is not None:
            trainer.logger.log_hyperparams(self.config[self.config['subcommand']])

        if trainer.checkpoint_callback.dirpath is not None:
            log_dir = trainer.checkpoint_callback.dirpath
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

        assert log_dir is not None
        config_path = os.path.join(log_dir, self.config_filename)
        if not self.overwrite and os.path.isfile(config_path):
            raise RuntimeError(
                f"{self.__class__.__name__} expected {config_path} to NOT exist. Aborting to avoid overwriting"
                " results of a previous run. You can delete the previous config file,"
                " set `LightningCLI(save_config_callback=None)` to disable config saving,"
                " or set `LightningCLI(save_config_overwrite=True)` to overwrite the config file."
            )
        if trainer.is_global_zero:
            # save only on rank zero to avoid race conditions on DDP.
            # the `log_dir` needs to be created as we rely on the logger to do it usually
            # but it hasn't logged anything at this point
            get_filesystem(log_dir).makedirs(log_dir, exist_ok = True)
            self.parser.save(
                self.config, config_path, skip_none = False, overwrite = self.overwrite, multifile = self.multifile
            )
