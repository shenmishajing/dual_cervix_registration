from typing import Dict, Union, Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBar, RichProgressBarTheme


class RichDefaultThemeProgressBar(RichProgressBar):

    def __init__(
            self,
            refresh_rate_per_second: Optional[int] = 10,
            leave: Optional[bool] = False,
            show_version: Optional[bool] = True,
    ) -> None:
        super().__init__(refresh_rate_per_second = refresh_rate_per_second, leave = leave, theme = RichProgressBarTheme())
        self.show_version = show_version

    def get_metrics(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> Dict[str, Union[int, str]]:
        items = super().get_metrics(trainer, pl_module)
        if not self.show_version:
            # don't show the version number
            items.pop("v_num", None)
        return items
