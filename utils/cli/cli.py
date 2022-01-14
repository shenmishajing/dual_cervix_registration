from pytorch_lightning import Trainer
from pytorch_lightning.utilities import _JSONARGPARSE_AVAILABLE
from pytorch_lightning.utilities.cli import LightningCLI, LightningArgumentParser, SaveConfigCallback, DATAMODULE_REGISTRY
from typing import Any, Dict, List, Optional, Tuple, Type, Union, Callable

from utils.callbacks.save_and_log_config_callback import SaveAndLogConfigCallback
from .trainer import Trainer as _Trainer
from .actions import LightningActionConfigFile

DATAMODULE_REGISTRY(object)


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
