

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from neuralop.models import FNO


class LitFNO2D(pl.LightningModule):
    def __init__(self, 
                 in_channels=3,
                 out_channels=1,
                 width=8, # 原本是32，第二次是16
                 modes1=8,  # 原本是24，第二次是16
                 modes2=8,  # 原本是24，第二次是16
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        print(self.hparams)
        self.net = FNO(n_modes=(modes1, modes2), hidden_channels=width, in_channels=in_channels, out_channels=out_channels)
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        self.num_params = sum([np.prod(p.size()) for p in model_parameters])
        print(f'\nThe model has {self.num_params} params \n')

    def forward(self, x):
        delta = self.net(x)
        return delta

    def training_step(self, batch, batch_idx):
        input, gt = batch
        # gt = gt.squeeze()
        background = input[:,2:3,:,:]
        pred_values = background + self(input)
        loss = F.mse_loss(pred_values, gt, reduction='mean')
        self.log("train_loss", loss, on_step=True, on_epoch=True,prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input, gt = batch
        # gt = gt.squeeze()
        background = input[:,2:3,:,:]
        pred_values = background + self(input)
        loss = F.mse_loss(pred_values, gt, reduction='mean')
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


