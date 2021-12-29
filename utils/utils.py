import os


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
