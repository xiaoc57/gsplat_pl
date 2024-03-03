from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.demos.boring_classes import DemoModel, BoringDataModule
from lightning.pytorch.callbacks import TQDMProgressBar

from model import *
from data import *


def cli_main():
    cli = LightningCLI(
        # parser_kwargs={"fit": {"default_config_files": [".\\config\\test.yaml"]}}
        )
    # note: don't call fit!!


if __name__ == "__main__":
    cli_main()