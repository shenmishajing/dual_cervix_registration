from torch.utils.data import DataLoader, IterableDataset

from pytorch_lightning.core.datamodule import LightningDataModule as _LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from typing import Optional, Mapping, Sequence, Any, Callable


class LightningDataModule(_LightningDataModule):
    def __init__(self, data_loader_config: Optional[Mapping[str, Any]] = None):
        super().__init__()

        self.collate_fns = self.collate_fn

        if data_loader_config is None:
            self.data_loader_config = {}
        else:
            self.data_loader_config = data_loader_config

        self.train_dataset = self.val_dataset = self.test_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.trainer.overfit_batches > 0:
            self.train_dataset = self._build_data_set('train')
            return
        if stage in [None, 'fit']:
            self.train_dataset = self._build_data_set('train')
        if stage in [None, 'fit', 'validate']:
            self.val_dataset = self._build_data_set('val')
        if stage in [None, 'test', 'predict']:
            self.test_dataset = self._build_data_set('test')

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self._build_data_loader(self.train_dataset, shuffle = True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self._build_data_loader(self.val_dataset)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self._build_data_loader(self.test_dataset)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return self._build_data_loader(self.test_dataset)

    def _build_data_set(self, split):
        raise NotImplementedError

    def _build_data_loader(self, dataset, shuffle: Optional[bool] = False, collate_fn: Optional[Callable] = None) -> TRAIN_DATALOADERS:
        def dataloader(ds, cl_fn) -> DataLoader:
            return DataLoader(ds, shuffle = shuffle and not isinstance(ds, IterableDataset), collate_fn = cl_fn, **self.data_loader_config)

        if collate_fn is None:
            collate_fn = self.collate_fns
        if isinstance(dataset, Mapping):
            return {key: dataloader(ds, cl_fn = collate_fn[key] if isinstance(collate_fn, Mapping) else collate_fn) for key, ds in
                    dataset.items()}
        if isinstance(dataset, Sequence):
            return [dataloader(dataset[i], cl_fn = collate_fn[i] if isinstance(collate_fn, Sequence) else collate_fn) for i in
                    range(len(dataset))]
        return dataloader(dataset, cl_fn = collate_fn)

    @staticmethod
    def collate_fn(batch):
        raise NotImplementedError
