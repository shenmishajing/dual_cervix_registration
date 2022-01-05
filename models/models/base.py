from typing import Any, Sequence, Mapping, Optional, Union

import torch
from pytorch_lightning import LightningModule as _LightningModule
from utils import optim


class LightningModule(_LightningModule):

    def __init__(self,
                 loss_config: Mapping[str, Union[torch.nn.Module, Mapping[str, Union[torch.nn.Module, int, float]]]] = None,
                 optimizer_config: Optional[Mapping[str, Any]] = None,
                 *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

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
                optimizer_config['lr_scheduler'] = [optimizer_config['lr_scheduler']]

            warmup_lr_schedulers = []
            # construct lr_scheduler
            for i in range(len(optimizer_config['lr_scheduler'])):
                # select optimizer
                if len(optimizer_config['optimizer']) == 1:
                    optimizer = optimizer_config['optimizer'][0]
                    if 'optimizer' in optimizer_config['lr_scheduler'][i]:
                        del optimizer_config['lr_scheduler'][i]['optimizer']
                else:
                    if 'optimizer' in optimizer_config['lr_scheduler'][i]:
                        optimizer = optimizer_config['optimizer'][optimizer_config['lr_scheduler'][i]['optimizer']]
                    else:
                        optimizer = optimizer_config['optimizer'][i]

                # construct lr_scheduler
                if 'scheduler' not in optimizer_config['lr_scheduler'][i]:
                    optimizer_config['lr_scheduler'][i] = {'scheduler': optimizer_config['lr_scheduler'][i]}
                optimizer_config['lr_scheduler'][i]['scheduler'] = self._construct_lr_scheduler(optimizer,
                                                                                                optimizer_config['lr_scheduler'][i][
                                                                                                    'scheduler'])
                # construct warmup_lr_scheduler
                if 'warmup_config' in optimizer_config['lr_scheduler'][i]:
                    warmup_config = optimizer_config['lr_scheduler'][i].pop('warmup_config')
                    if 'scheduler' in warmup_config:
                        warmup_config['scheduler']['type'] = 'WarmupScheduler'
                        warmup_config['scheduler'] = self._construct_lr_scheduler(optimizer, warmup_config['scheduler'])
                    else:
                        warmup_config['type'] = 'WarmupScheduler'
                        warmup_config = {
                            'scheduler': self._construct_lr_scheduler(optimizer, warmup_config),
                            'interval': 'step'
                        }
                    warmup_lr_schedulers.append(warmup_config)
            # add warmup lr schedulers
            optimizer_config['lr_scheduler'].extend(warmup_lr_schedulers)

        if 'lr_scheduler' in optimizer_config:
            return optimizer_config['optimizer'], optimizer_config['lr_scheduler']
        else:
            return optimizer_config['optimizer']

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
