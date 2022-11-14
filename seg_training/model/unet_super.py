import abc
from argparse import ArgumentParser

import pytorch_lightning as pl


class UnetSuper(pl.LightningModule):
    def __init__(self):
        super(UnetSuper, self.__init__()).__init__()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser
        ### Arguments
        return parser

    @abc.abstractmethod
    def forward(self, x):
        pass

    def loss(self):
        return None

    def training_step(self):
        return None

    def training_epoch_end(self, training_step_outputs):
        return None

    def validation_step(self):
        return None

    def validation_epoch_end(self, validation_step_outputs):
        return None

    def test_step(self):
        return None

    def test_epoch_end(self, test_step_output):
        return None

    def prepare_data(self):
        return None

    def predict(self):
        return None

    def configure_optimizers(self):
        return None

    def log_tb_images(self):
        return None
