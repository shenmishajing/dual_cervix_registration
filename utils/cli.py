import os
from pytorch_lightning.utilities import _JSONARGPARSE_AVAILABLE
from pytorch_lightning.utilities.cli import LightningCLI, LightningArgumentParser, SaveConfigCallback, DATAMODULE_REGISTRY
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from jsonargparse import ActionConfigFile as _ActionConfigFile
from jsonargparse import get_config_read_mode, Path, Namespace
from jsonargparse.actions import _ActionSubCommands
from jsonargparse.loaders_dumpers import load_value, get_loader_exceptions

from utils.callbacks.save_and_log_config_callback import SaveAndLogConfigCallback

DATAMODULE_REGISTRY(object)


def parse_config(parser, cfg_path, seen_cfg = None, **kwargs):
    abs_cfg_path = os.path.abspath(cfg_path)
    if seen_cfg is None:
        seen_cfg = {}
    elif abs_cfg_path in seen_cfg:
        if seen_cfg[abs_cfg_path] is None:
            raise RuntimeError('Circular reference detected in config file')
        else:
            return seen_cfg[abs_cfg_path]

    cfg_file = parser.parse_path(cfg_path, **kwargs)
    seen_cfg[abs_cfg_path] = None

    if '__base__' in cfg_file:
        sub_cfg_paths = cfg_file.pop('__base__')
        if sub_cfg_paths is not None:
            if not isinstance(sub_cfg_paths, list):
                sub_cfg_paths = [sub_cfg_paths]
            sub_cfg_paths = [os.path.join(os.path.dirname(cfg_path), sub_cfg_path) for sub_cfg_path in sub_cfg_paths]
            sub_cfg_file = Namespace()
            for sub_cfg_path in sub_cfg_paths:
                sub_cfg_file.update(parse_config(parser, sub_cfg_path, seen_cfg = seen_cfg, **kwargs))
            cfg_file.update(sub_cfg_file)

    if '__import__' in cfg_file:
        cfg_file.pop('__import__')

    seen_cfg[abs_cfg_path] = cfg_file
    return cfg_file


class ActionConfigFile(_ActionConfigFile):
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
                    cfg_file = parser.parse_string(value, **kwargs)
                except (TypeError,) + get_loader_exceptions() as ex_str:
                    raise TypeError(f'Parser key "{dest}": {ex_str}') from ex_str
            else:
                cfg_file = parse_config(parser, value, **kwargs)
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
            "--config", action = ActionConfigFile, help = "Path to a configuration file in json or yaml format."
        )
        self.callback_keys: List[str] = []
        # separate optimizers and lr schedulers to know which were added
        self._optimizers: Dict[str, Tuple[Union[Type, Tuple[Type, ...]], str]] = {}
        self._lr_schedulers: Dict[str, Tuple[Union[Type, Tuple[Type, ...]], str]] = {}


class CLI(LightningCLI):
    def __init__(
            self,
            save_config_callback: Optional[Type[SaveConfigCallback]] = SaveAndLogConfigCallback,
            *args, **kwargs
    ) -> None:
        super().__init__(save_config_callback = save_config_callback, *args, **kwargs)

    def init_parser(self, **kwargs: Any) -> LightningArgumentParser:
        """Method that instantiates the argument parser."""
        return ArgumentParser(**kwargs)
