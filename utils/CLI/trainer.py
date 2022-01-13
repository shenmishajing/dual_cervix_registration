from typing import Any, Dict, List, Optional

from pytorch_lightning import Trainer as _Trainer


class Trainer(_Trainer):
    def _configure_schedulers(
            self, schedulers: list, monitor: Optional[str], is_manual_optimization: bool
    ) -> List[Dict[str, Any]]:
        """Convert each scheduler into dict structure with relevant information."""
        return super()._configure_schedulers(schedulers, monitor, False)
