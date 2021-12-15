import torch
from pytorch_lightning.utilities.cli import LightningCLI, DATAMODULE_REGISTRY

from utils.callbacks.save_and_log_config_callback import SaveAndLogConfigCallback

torch.multiprocessing.set_sharing_strategy('file_system')

DATAMODULE_REGISTRY(object)


class CLI(LightningCLI):
    pass


def main():
    CLI(save_config_callback = SaveAndLogConfigCallback)


if __name__ == '__main__':
    main()