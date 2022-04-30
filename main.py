import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from dataloader import Dataloader
from model import ViT, Custom_LossFunction
import torch.nn as nn
import argparse
from pytorch_lightning.profiler import SimpleProfiler
from pytorch_lightning.loggers import CSVLogger
from tools import CosineAnnealingLRWarmup
import os
import numpy as np
import matplotlib.pyplot as plt
from ori_model import mynet


class Controller(pl.LightningModule):
    # 1. This would become the LightningModule `__init__` function.
    def __init__(self, args):
        super().__init__()
        self.args = args
        # self.model = ViT(num_classes=2500, dim=1024, depth=6, heads=16, mlp_dim=2048)
        self.model = mynet()
        self.lr = 0
        self.loss = Custom_LossFunction()
        self.lr_scheduler = None

    def forward(self, texts):
        logits = self.model(texts)
        return logits

    def training_step(self, batch, batch_idx):
        texts, targets = batch
        logits = self.model(texts)
        loss = self.loss(logits, targets)
        self.log("train_loss", loss)
        self.lr_scheduler.iter_step()
        return loss

    def validation_step(self, batch, batch_idx):
        texts, targets = batch
        logits = self.model(texts)
        logits = logits.reshape(logits.shape[0], 50, 50)

        if batch_idx == 0:
            os.makedirs(f'{self.args.wkdir}/visual/{batch_idx}/', exist_ok=True)
            for idx, l in enumerate(logits):
                x = np.arange(0, 50, 1)
                y = np.arange(0, 50, 1)
                fig, ax = plt.subplots()
                ax.pcolormesh(x, y, l.cpu().numpy())
                plt.savefig(f'{self.args.wkdir}/visual/{batch_idx}/{idx}_pred.jpg')

                ax.pcolormesh(x, y, targets[idx].cpu().numpy())
                plt.savefig(f'{self.args.wkdir}/visual/{batch_idx}/{idx}_gt.jpg')
                plt.cla()
                plt.clf()
                plt.close('all')

        loss = self.loss(logits, targets)
        return loss

    def validation_epoch_end(self, outs):
        loss_sum = sum(outs)
        self.log("val_acc", loss_sum)
        return loss_sum

    def configure_optimizers(self):
        self.lr = self.args.lr
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr,
                                      weight_decay=1e-05, eps=1e-08)
        self.lr_scheduler = CosineAnnealingLRWarmup(optimizer, verbose=False,
                                               warmup_iter=500,
                                               warmup_ratio=0.001,
                                               T_max=self.args.epoch - 1)
        return [optimizer], [self.lr_scheduler]

def main(config):
    pl.seed_everything(233, workers=True)
    model = Controller(config)
    dataModule = Dataloader(config)
    wk_dir = config.wkdir
    os.makedirs(f'{wk_dir}/visual/', exist_ok=True)
    checkpoint_callback = ModelCheckpoint(dirpath=f"{wk_dir}/models", save_top_k=1, monitor="val_acc")

    logger = CSVLogger(wk_dir, name='training_log')
    trainer = pl.Trainer(callbacks=[checkpoint_callback], accelerator='auto', check_val_every_n_epoch=1,
                         auto_lr_find=True, strategy="ddp", devices=1, limit_train_batches=1,
                         gradient_clip_val=1.0, log_every_n_steps=2, logger=logger,
                         max_epochs=config.epoch, profiler=SimpleProfiler(), default_root_dir=wk_dir)
    # trainer.tune(model, train_dataloader=dataModule.train_dataloader())
    trainer.fit(model=model, train_dataloaders=dataModule.train_dataloader(),
                val_dataloaders=dataModule.val_dataloader())


def get_parser():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--wkdir", required=False, default='./wkdir', type=str)
    parser.add_argument("--lr", required=False, default=1e-04, type=float)
    parser.add_argument("--data", required=False, default='./data', type=str)
    parser.add_argument("--bs", required=False, default=2, type=int)
    parser.add_argument("--epoch", required=False, default=500, type=int)
    parser.add_argument("--num_worker", required=False, default=0, type=int)

    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    main(args)
