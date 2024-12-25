from pytorch_lightning.cli import LightningCLI
import pytorch_lightning as pl
from data import DataModule
from model import FCN4Flare


def cli_main():
    cli = LightningCLI(FCN4Flare, DataModule)


if __name__ == '__main__':
    cli_main()