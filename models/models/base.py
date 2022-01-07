import copy
from typing import Any, Sequence, Mapping, Optional, Union

import torch
from pytorch_lightning import LightningModule as _LightningModule
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from utils import optim


class LightningModule(_LightningModule):

    def __init__(self,
                 normalize_config: Mapping[str, Any] = None,
                 loss_config: Mapping[str, Union[torch.nn.Module, Mapping[str, Union[torch.nn.Module, int, float]]]] = None,
                 optimizer_config: Optional[Mapping[str, Any]] = None,
                 *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.normalize_config = normalize_config

        if loss_config is not None:
            self._parse_loss_config(loss_config)

        if optimizer_config is not None:
            self.lr = self._parse_optimizer_config(optimizer_config)

    def _parse_loss_config(self, loss_config):
        for key, value in loss_config.items():
            if not isinstance(value, Mapping):
                loss_config[key] = {'module': value, 'weight': 1}
            setattr(self, 'loss_' + key, loss_config[key]['module'])
        self.loss_weight = {k: v.get('weight', 1) for k, v in loss_config.items()}

    def _parse_optimizer_config(self, optimizer_config):
        assert isinstance(optimizer_config, dict), 'optimizer_config should be a dict'
        if 'optimizer' not in optimizer_config:
            optimizer_config = {'optimizer': optimizer_config}
        if not isinstance(optimizer_config['optimizer'], Sequence):
            optimizer_config['optimizer'] = [optimizer_config['optimizer']]
        self.optimizer_config = optimizer_config
        return optimizer_config['optimizer'][0]['lr']

    def _construct_optimizer(self, optimizer, set_lr = False, params = None):
        """
        Constructs the optimizer.

        Args:
            optimizer: dictionary containing optimizer configuration.
        """
        optimizer_type = optimizer.pop('type')
        if hasattr(self, 'lr') and self.lr is not None and set_lr:
            optimizer['lr'] = self.lr
        optimizer = optim.__dict__[optimizer_type](self.parameters() if params is None else params, **optimizer)

        return optimizer

    def _construct_optimizers(self, optimizers):
        """
        Constructs all optimizers.

        Args:
            optimizers: list of dictionary containing optimizer configuration.
        """
        for i in range(len(optimizers)):
            optimizers[i] = self._construct_optimizer(optimizers[i], set_lr = i == 0)

        return optimizers

    @staticmethod
    def _construct_lr_scheduler(optimizer, lr_scheduler):
        """
        Constructs the lr_scheduler.

        Args:
            optimizer: the optimizer used to construct the lr_scheduler.
            lr_scheduler: dictionary containing lr_scheduler configuration.
        """
        lr_scheduler_type = lr_scheduler.pop('type')
        lr_scheduler = optim.__dict__[lr_scheduler_type](optimizer, **lr_scheduler)

        return lr_scheduler

    def configure_optimizers(self):
        """
        Configure optimizers for model.
        """
        optimizer_config = self.optimizer_config.copy()

        # construct optimizer
        optimizer_config['optimizer'] = self._construct_optimizers(optimizer_config['optimizer'])

        # construct lr_scheduler
        if 'lr_scheduler' in optimizer_config:
            # parse lr_scheduler
            if not isinstance(optimizer_config['lr_scheduler'], Sequence):
                optimizer_config['lr_scheduler'] = [copy.deepcopy(optimizer_config['lr_scheduler']) for _ in
                                                    range(len(optimizer_config['optimizer']))]

            warmup_lr_schedulers = []
            # construct lr_scheduler
            for i in range(len(optimizer_config['lr_scheduler'])):
                # select optimizer
                if len(optimizer_config['optimizer']) == 1:
                    opt_idx = 0
                else:
                    if 'opt_idx' in optimizer_config['lr_scheduler'][i]:
                        opt_idx = optimizer_config['lr_scheduler'][i]['opt_idx']
                    else:
                        opt_idx = i
                optimizer = optimizer_config['optimizer'][opt_idx]

                # construct lr_scheduler
                if 'scheduler' not in optimizer_config['lr_scheduler'][i]:
                    optimizer_config['lr_scheduler'][i] = {'scheduler': optimizer_config['lr_scheduler'][i]}
                optimizer_config['lr_scheduler'][i]['scheduler'] = \
                    self._construct_lr_scheduler(optimizer, optimizer_config['lr_scheduler'][i]['scheduler'])
                optimizer_config['lr_scheduler'][i]['opt_idx'] = opt_idx

                # construct warmup_lr_scheduler
                if 'warmup_config' in optimizer_config['lr_scheduler'][i]:
                    warmup_config = optimizer_config['lr_scheduler'][i].pop('warmup_config')
                    if 'scheduler' not in warmup_config:
                        warmup_config = {'scheduler': warmup_config}
                    warmup_config['scheduler']['type'] = 'WarmupScheduler'
                    warmup_config['scheduler'] = self._construct_lr_scheduler(optimizer, warmup_config['scheduler'])
                    warmup_config.update({'interval': 'step', 'opt_idx': opt_idx})
                    warmup_lr_schedulers.append(warmup_config)
            # add warmup lr schedulers
            optimizer_config['lr_scheduler'].extend(warmup_lr_schedulers)

        if 'lr_scheduler' in optimizer_config:
            return optimizer_config['optimizer'], optimizer_config['lr_scheduler']
        else:
            return optimizer_config['optimizer']

    def _accumulated_batches_reached(self,
                                     batch_idx: int) -> bool:
        """Determine if accumulation will be finished by the end of the current batch."""
        return (batch_idx + 1) % self.trainer.accumulate_grad_batches == 0

    def _num_ready_batches_reached(self,
                                   batch_idx: int) -> bool:
        """Checks if we are in the last batch or if there are more batches to follow."""
        epoch_finished_on_ready = (batch_idx + 1) == self.trainer.num_training_batches
        return epoch_finished_on_ready

    def _should_accumulate(self,
                           batch_idx: int) -> bool:
        """Checks if the optimizer step should be performed or gradients should be accumulated for the current
        step."""
        accumulation_done = self._accumulated_batches_reached(batch_idx)
        # Lightning steps on the final batch
        is_final_batch = self._num_ready_batches_reached(batch_idx)
        # but the TTP might not
        ttp_accumulates_on_final_batch = (
                self.trainer.training_type_plugin.handles_gradient_accumulation or not is_final_batch
        )
        return not accumulation_done and ttp_accumulates_on_final_batch

    def manual_lr_schedulers_step(self,
                                  interval: str,
                                  batch_idx: int = None,
                                  update_plateau_schedulers: Union[bool, None] = None) -> None:
        """updates the lr schedulers based on the given interval."""
        if interval == "step" and self._should_accumulate(batch_idx):
            return
        self._manual_lr_schedulers_step(
            interval = interval,
            batch_idx = batch_idx,
            update_plateau_schedulers = update_plateau_schedulers
        )

    def _manual_lr_schedulers_step(
            self,
            interval: str,
            batch_idx: int,
            update_plateau_schedulers: Union[bool, None] = None
    ) -> None:
        """Update learning rates.

        Args:
            interval: either 'epoch' or 'step'.
            update_plateau_schedulers: control whether ``ReduceLROnPlateau`` or non-plateau schedulers get updated.
                This is used so non-plateau schedulers can be updated before running validation. Checkpoints are
                commonly saved during validation, however, on-plateau schedulers might monitor a validation metric
                so they have to be updated separately.
        """
        if not self.trainer.lr_schedulers:
            return

        for lr_scheduler in self.trainer.lr_schedulers:
            if update_plateau_schedulers is not None and update_plateau_schedulers ^ lr_scheduler["reduce_on_plateau"]:
                continue

            current_idx = batch_idx if interval == "step" else self.trainer.current_epoch
            current_idx += 1  # account for both batch and epoch starts from 0
            # Take step if call to update_learning_rates matches the interval key and
            # the current step modulo the schedulers frequency is zero
            if lr_scheduler["interval"] == interval and current_idx % lr_scheduler["frequency"] == 0:
                if lr_scheduler["reduce_on_plateau"]:
                    # If instance of ReduceLROnPlateau, we need a monitor
                    monitor_key = lr_scheduler["monitor"]
                    monitor_val = self.trainer.callback_metrics.get(monitor_key)
                    if monitor_val is None:
                        if lr_scheduler.get("strict", True):
                            avail_metrics = list(self.trainer.callback_metrics)
                            raise MisconfigurationException(
                                f"ReduceLROnPlateau conditioned on metric {monitor_key}"
                                f" which is not available. Available metrics are: {avail_metrics}."
                                " Condition can be set using `monitor` key in lr scheduler dict"
                            )
                        rank_zero_warn(
                            f"ReduceLROnPlateau conditioned on metric {monitor_key}"
                            " which is not available but strict is set to `False`."
                            " Skipping learning rate update.",
                            RuntimeWarning,
                        )
                        continue

                    # update LR
                    lr_scheduler["scheduler"].step(monitor_val)
                else:
                    lr_scheduler["scheduler"].step()

    def _loss_step(self, batch, res, prefix = 'train'):
        raise NotImplementedError

    def loss_step(self, batch, res, prefix = 'train', use_loss_weight = True, loss_use_loss_weight = True, detach = None):
        loss = self._loss_step(batch, res, prefix)
        # multi loss weights
        if use_loss_weight:
            loss = {k: v * (1 if k not in self.loss_weight else self.loss_weight[k]) for k, v in loss.items()}
        # calculate loss
        if not use_loss_weight and loss_use_loss_weight:
            total_loss = [v * (1 if k not in self.loss_weight else self.loss_weight[k]) for k, v in loss.items()]
        else:
            total_loss = [v for k, v in loss.items()]
        loss['loss'] = torch.sum(torch.stack(total_loss))
        # add prefix
        if detach is None:
            detach = prefix != 'train'
        loss = {(f'{prefix}/' if prefix is not None else '') + ('loss_' if 'loss' not in k else '') + k: (v.detach() if detach else v) for
                k, v in loss.items()}
        return loss

    def training_step(self, batch, *args, **kwargs):
        res = self(batch)
        loss = self.loss_step(batch, res, 'train')
        self.log_dict(loss)
        return loss['train/loss']

    def training_epoch_end(self, outputs):
        if not self.automatic_optimization:
            self.manual_lr_schedulers_step('epoch')

    def validation_step(self, batch, *args, **kwargs):
        res = self(batch)
        loss = self.loss_step(batch, res, 'val')
        self.log_dict(loss)
        return loss

    def test_step(self, batch, *args, **kwargs):
        res = self(batch)
        loss = self.loss_step(batch, res, 'test')
        self.log_dict(loss)
        return loss
