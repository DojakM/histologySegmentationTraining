import abc
from argparse import ArgumentParser

import cv2
import torch
import pytorch_lightning as pl
import torchmetrics
from torch_optimizer import AdaBelief

from seg_training.utils import label2rgb, unnormalize


class UnetSuper(pl.LightningModule):
    def __init__(self, num_classes, len_test_set: int, hparams: dict, input_channels=1, min_filter=32, **kwargs):
        super(UnetSuper, self).__init__()
        self.num_classes = num_classes
        self.save_hyperparameters(hparams)
        self.args = kwargs
        self.metric = torchmetrics.classification.MulticlassJaccardIndex(self.num_classes)  # Changed because numpy conflict
        self.len_test_set = len_test_set
        self.weights = kwargs['class_weights']
        self.criterion = torch.hub.load(
            'adeelh/pytorch-multi-class-focal-loss',
            model='FocalLoss',
            alpha=torch.tensor(self.weights),
            gamma=kwargs["gamma_factor"],
            reduction='mean',
            force_reload=False
        )
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser
        ### Arguments
        return parser

    @abc.abstractmethod
    def forward(self, x):
        pass

    def loss(self, logits, labels):
        labels = labels.long()
        return self.criterion(logits, labels)

    def training_step(self, train_batch, batch_idx):
        x, y, logits, y_hat = self.predict(train_batch, batch_idx)
        loss = self.loss(logits, y)
        # Next two lines might change
        self.train_iou = self.metric.train()
        self.log('train_IoU', self.train_iou[0].mean(), on_step=False, on_epoch=True)
        for i in range(self.num_classes):
            self.log(f'train_IoU_{i}', self.train_iou[0][i], on_step=False, on_epoch=True)
        return {'loss': loss, 'iou': self.train_iou[0].mean()}


    def training_epoch_end(self, training_step_outputs):
        train_avg_loss = torch.stack([train_output['loss'] for train_output in training_step_outputs]).mean()
        self.log('train_avg_loss', train_avg_loss, sync_dist=True)
        train_avg_iou = torch.stack([train_output['iou'] for train_output in training_step_outputs]).mean()
        self.log('train_avg_iou', train_avg_iou, sync_dist=True)

    def validation_step(self, val_batch, batch_idx):
        data, target, output, prediction = self.predict(val_batch, batch_idx)
        self.val_iou = self.metric(prediction, target)
        loss = self.loss(output, target)
        self.log('val_IoU', self.val_iou[0].mean(), on_step=True, on_epoch=True, sync_dist=True)
        self.log_tb_images(batch_idx, data, target, prediction)
        for i in range(self.num_classes):
            self.log(f'val_IoU_{i}', self.val_iou[0][i], on_step=True, on_epoch=True, sync_dist=True)
        return {'loss': loss,
                'iou': torch.mean(torch.stack([self.val_iou[0][2], self.val_iou[0][3], self.val_iou[0][4]]))}
    def validation_epoch_end(self, validation_step_outputs):
        val_avg_loss = torch.stack([val_output['loss'] for val_output in validation_step_outputs]).mean()
        self.log('val_avg_loss', val_avg_loss, sync_dist=True)
        val_avg_iou = torch.stack([val_output['iou'] for val_output in validation_step_outputs]).mean()
        self.log('val_avg_iou', val_avg_iou, sync_dist=True)

    def test_step(self, test_batch, batch_idx):
        data, target, output, prediction = self.predict(test_batch, batch_idx)
        self.test_iou = self.metric(prediction, target)
        self.log('test_IoU', self.test_iou[0].mean(), on_step=False, on_epoch=True, sync_dist=True)
        for i in range(self.num_classes):
            self.log(f'test_IoU_{i}', self.test_iou[0][i], on_step=False, on_epoch=True, sync_dist=True)
        # sum up batch loss
        test_loss = self.loss(output, target)
        # get the index of the max log-probability
        correct = prediction.eq(target.data).sum()
        return {'test_loss': test_loss, 'correct': correct}

    def test_epoch_end(self, outputs):
        avg_test_loss = sum([test_output['test_loss'] for test_output in outputs]) / self.len_test_set
        test_correct = float(sum([test_output['correct'] for test_output in outputs]))
        self.log('avg_test_loss', avg_test_loss, sync_dist=True)
        self.log('test_correct', test_correct, sync_dist=True)

    def prepare_data(self):
        return {}

    def predict(self, batch, batch_idx: int, dataloader_idx = None):    # Not used but may be useful in future
        data, target = batch
        output = self.forward(data)
        _, prediction = torch.max(output, dim=1)    # Drop value and just keep ID, I think
        return data, target, output, prediction


    def configure_optimizers(self):
        self.optimizer = AdaBelief(self.parameters(), lr=self.args['lr'], eps=self.args['epsilon'],
                                   betas=(0.9, 0.999),
                                   weight_decay=self.args['weight_decay'],
                                   weight_decouple=True)
        self.scheduler = {
            'scheduler': torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.args['lr'],
                                                             total_steps=self.args["max_epochs"],
                                                             pct_start=0.45, three_phase=True),
            'monitor': 'train_avg_loss',
        }
        return [self.optimizer], [self.scheduler]

    def log_tb_images(self, batch_idx, x: torch.Tensor, y: torch.Tensor, y_hat: torch.Tensor, index=0):
        img = cv2.cvtColor(unnormalize(x[index].cpu().detach().numpy().squeeze()), cv2.COLOR_GRAY2RGB).astype(int)
        pred = y_hat[index].cpu().detach().numpy()
        mask = y[index].cpu().detach().numpy()
        alpha = 0.7
        gt = label2rgb(alpha, img, mask)
        prediction = label2rgb(alpha, img, pred)
        log = torch.stack([gt, prediction], dim=0)
        self.logger.experiment.add_images(f'Images and Masks Batch: {batch_idx}', log, self.current_epoch)
