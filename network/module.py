from numpy.core.fromnumeric import resize
import torch
import pytorch_lightning as pl
from metrics import MetricLogger
from network.RDM_Net import DepthEstimationNet
from network import computations as cp
import utils as u
import loss as l
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
        fine_details, ord_pred = self.model(x)
        return fine_details, ord_pred

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader                                            

    def training_step(self, batch, batch_idx):
        if batch_idx == 0: self.metric_logger.reset()
        x, y = batch
        y = cp.resize(y,128)
        fine_details, ord_pred = self(x)

        final_depth, fine_detail_loss = self.compute_final_depth(fine_details, y)
        ord_y = self.compute_ordinal_target(ord_pred, y)
        ord_loss = l.Ordinal_Loss().calc(ord_pred, ord_y)

        loss = self.criterion(final_depth, y) + ord_loss + fine_detail_loss

        return self.metric_logger.log_train(final_depth, y, loss)

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0: self.metric_logger.reset()
        x, y = batch
        y = cp.resize(y,128)

        fine_details, _ = self(x)

        y_hat, _ = self.compute_final_depth(fine_details, y)
        #ord_y = self.compute_ordinal_target(y_hat_ord, y)

        return self.metric_logger.log_val(y_hat, y)
    
    def compute_final_depth(self, fine_detail_list, target):
        #decompose target map
        component_target = cp.decompose_depth_map([], target, 7)[::-1]
        #optimize weight layer
        components, loss = cp.optimize_components(self.model.weight_layer, fine_detail_list, component_target)
        #returned optimal candidates are recombined to final depth map
        final = cp.recombination(components)
        return final,loss
    
    def compute_ordinal_target(self, ord_pred, target):
        #resize target to correct size
        target = cp.resize(target, ord_pred.shape[2])
        #transform with ordinal regression so it can be compared
        ord_target = u.get_depth_sid("nyu", target)
        return ord_target