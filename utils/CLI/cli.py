from pytorch_lightning import Trainer
from pytorch_lightning.utilities import _JSONARGPARSE_AVAILABLE
from pytorch_lightning.utilities.cli import LightningCLI, LightningArgumentParser, SaveConfigCallback, DATAMODULE_REGISTRY
from typing import Any, Dict, List, Optional, Tuple, Type, Union, Callable
from jsonargparse import ActionConfigFile, get_config_read_mode, Path
from jsonargparse.actions import _ActionSubCommands
from jsonargparse.loaders_dumpers import load_value, get_loader_exceptions

from utils.callbacks.save_and_log_config_callback import SaveAndLogConfigCallback
from .trainer import Trainer as _Trainer
from .parser import parse_path, parse_string

DATAMODULE_REGISTRY(object)


class LightningActionConfigFile(ActionConfigFile):
    @staticmethod
    def apply_config(parser, cfg, dest, value) -> None:
        with _ActionSubCommands.not_single_subcommand():
            if dest not in cfg:
                cfg[dest] = []
            kwargs = {'env': False, 'defaults': False, '_skip_check': True, '_fail_no_subcommand': False}
            try:
                cfg_path: Optional[Path] = Path(value, mode = get_config_read_mode())
            except TypeError as ex_path:
                try:
                    if isinstance(load_value(value), str):
                        raise ex_path
                    cfg_path = None
                    cfg_file = parse_string(parser, value, **kwargs)
                except (TypeError,) + get_loader_exceptions() as ex_str:
                    raise TypeError(f'Parser key "{dest}": {ex_str}') from ex_str
            else:
                cfg_file = parse_path(parser, value, **kwargs)
            cfg[dest].append(cfg_path)
            cfg.update(cfg_file)


class ArgumentParser(LightningArgumentParser):
    def __init__(self, *args: Any, parse_as_dict: bool = True, **kwargs: Any) -> None:
        """Initialize argument parser that supports configuration file input.

        For full details of accepted arguments see `ArgumentParser.__init__
        <https://jsonargparse.readthedocs.io/en/stable/#jsonargparse.core.ArgumentParser.__init__>`_.
        """
        if not _JSONARGPARSE_AVAILABLE:
            raise ModuleNotFoundError(
                "`jsonargparse` is not installed but it is required for the CLI."
                " Install it with `pip install jsonargparse[signatures]`."
            )
        super(LightningArgumentParser, self).__init__(*args, parse_as_dict = parse_as_dict, **kwargs)
        self.add_argument(
            "--config", action = LightningActionConfigFile, help = "Path to a configuration file in json or yaml format."
        )
        self.callback_keys: List[str] = []
        # separate optimizers and lr schedulers to know which were added
        self._optimizers: Dict[str, Tuple[Union[Type, Tuple[Type, ...]], str]] = {}
        self._lr_schedulers: Dict[str, Tuple[Union[Type, Tuple[Type, ...]], str]] = {}


class CLI(LightningCLI):
    def __init__(
            self,
            save_config_callback: Optional[Type[SaveConfigCallback]] = SaveAndLogConfigCallback,
            trainer_class: Union[Type[Trainer], Callable[..., Trainer]] = _Trainer,
            *args, **kwargs
    ) -> None:
        super().__init__(save_config_callback = save_config_callback, trainer_class = trainer_class, *args, **kwargs)

    def init_parser(self, **kwargs: Any) -> LightningArgumentParser:
        """Method that instantiates the argument parser."""
        return ArgumentParser(**kwargs)
