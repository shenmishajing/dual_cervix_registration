from typing import Dict, Union

import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBar, RichProgressBarTheme


class RichDefaultThemeProgressBar(RichProgressBar):

    def __init__(
            self,
            refresh_rate_per_second: int = 10,
            leave: bool = False
    ) -> None:
        super().__init__(refresh_rate_per_second = refresh_rate_per_second, leave = leave, theme = RichProgressBarTheme())

    def get_metrics(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> Dict[str, Union[int, str]]:
        # don't show the version number
        items = super().get_metrics(trainer, pl_module)
        items.pop("v_num", None)
        return items
