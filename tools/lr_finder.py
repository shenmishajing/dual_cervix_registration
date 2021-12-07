from pytorch_lightning.utilities.cli import LightningCLI, DATAMODULE_REGISTRY

from utils.callbacks.save_and_log_config_callback import SaveAndLogConfigCallback

DATAMODULE_REGISTRY(object)


class CLI(LightningCLI):
    pass


def main():
    cli = CLI(save_config_callback = SaveAndLogConfigCallback, run = False)
    # Run learning rate finder
    lr_finder = cli.trainer.tuner.lr_find(model = cli.model, datamodule = cli.datamodule)

    # Plot with
    fig = lr_finder.plot(suggest = True)
    fig.savefig('lr_finder.png')

    # Pick point based on plot, or get suggestion
    print(f'suggest lr: {lr_finder.suggestion()}')


if __name__ == '__main__':
    main()
