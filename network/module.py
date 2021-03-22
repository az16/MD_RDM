import torch
import pytorch_lightning as pl
from metrics import MetricLogger
from network.RDM_Net import DepthEstimationNet
from network.computations import recombination
from dataloaders.nyu_dataloader import NYUDataset

class RelativeDephModule(pl.LightningModule):
    def __init__(self, path, batch_size, learning_rate, worker, metrics, *args, **kwargs):
        super().__init__()    
        self.save_hyperparameters()
        self.metric_logger = MetricLogger(metrics=metrics, module=self)
        self.train_loader = torch.utils.data.DataLoader(NYUDataset(path, dataset_type='labeled', split="train", output_size=(226, 226)),
                                                    batch_size=batch_size, 
                                                    shuffle=True, 
                                                    num_workers=worker, 
                                                    pin_memory=True)
        self.val_loader = torch.utils.data.DataLoader(NYUDataset(path, dataset_type='labeled', split="val", output_size=(226, 226)),
                                                    batch_size=1, 
                                                    shuffle=False, 
                                                    num_workers=1, 
                                                    pin_memory=True) 
        self.criterion = torch.nn.MSELoss()
        self.model = DepthEstimationNet()

    def configure_optimizers(self):
        train_param = self.model.parameters()
        # Training parameters
        optimizer = torch.optim.AdamW(train_param, lr=self.hparams.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
        scheduler = {
            'scheduler': lr_scheduler,
            'monitor': 'val_delta1'
        }
        return [optimizer], [scheduler]

    def forward(self, x):
        d1,d6,d7,d8,d9 = self.model(x)
        return d1,d6,d7,d8,d9

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader                                            

    def training_step(self, batch, batch_idx):
        if batch_idx == 0: self.metric_logger.reset()
        x, y = batch
        d1,d6,d7,d8,d9 = self(x)
        # TODO recombine
        # compute final depth image
        final_depth = compute_final_depth()
        loss = self.criterion(final_depth, y)
        return self.metric_logger.log_train(final_depth, y, loss)

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0: self.metric_logger.reset()
        x, y = batch
        y_hat = self(x)
        return self.metric_logger.log_val(y_hat, y)