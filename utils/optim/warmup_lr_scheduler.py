import warnings
from torch.optim.lr_scheduler import _LRScheduler


class WarmupScheduler(_LRScheduler):
    """ Warm-up(increasing) learning rate in optimizer.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_iters: target warm up epoch.
    """

    def __init__(self, optimizer, warmup_iters, last_epoch = -1, verbose = False):
        self.warmup_iters = warmup_iters
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch < self.warmup_iters:
            return [base_lr * (self.last_epoch + 1) / self.warmup_iters for base_lr in self.base_lrs]
        return [group['lr'] for group in self.optimizer.param_groups]
