import torch
import pytorch_lightning as pl
from torch import cuda
from metrics import MetricLogger
from network.RDM_Net import DepthEstimationNet
from network import computations as cp
import utils as u
import loss as l
from dataloaders.nyu_dataloader import NYUDataset

is_cuda=True
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
        print("Use cuda: {0}".format(is_cuda))
        if is_cuda:
            self.model = DepthEstimationNet().cuda()
        else:
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

        if is_cuda:
            x=x.cuda()

        fine_details, d_pred, l_pred = self.model(x)

        return fine_details, d_pred, l_pred

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader                                            

    def training_step(self, batch, batch_idx):
        if batch_idx == 0: self.metric_logger.reset()
        x, y = batch
        #print(x.dtype, y.dtype)
        #print(torch.min(y), torch.max(y))
        y = cp.resize(y,128)
        #print(torch.min(y), torch.max(y))

        if is_cuda:
            y = y.cuda() 
            #x = x.cuda()
            
        fine_details, ord_depth_pred, ord_label_pred = self(x)

        final_depth, fine_detail_loss = self.compute_final_depth(fine_details, y)
        #print(torch.isnan(final_depth).any())
        ord_y = self.compute_ordinal_target(ord_depth_pred, y)
        ord_loss = l.Ordinal_Loss().calc(ord_label_pred, ord_y, cuda=is_cuda)

        mse = self.criterion(final_depth, y)
        loss_all = mse + ord_loss + fine_detail_loss
       
        self.log("MSE", mse, prog_bar=True)
        self.log("Ord_Loss", ord_loss, prog_bar=True)
        self.log("Fine_Detail", fine_detail_loss, prog_bar=True)             
        return self.metric_logger.log_train(final_depth, self.normalize(y), loss_all)

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0: self.metric_logger.reset()
        x, y = batch
        y = cp.resize(y,128)

        if is_cuda:
            y = y.cuda() 
            #x = x.cuda()
            
        fine_details, _, _ = self(x)

        y_hat, _ = self.compute_final_depth(fine_details, y)

        return self.metric_logger.log_val(y_hat, self.normalize(y))
    
    def compute_final_depth(self, fine_detail_list, target):
        #decompose target map
        B,C,H,W = target.size()
        #target = self.normalize(target)
        #print(cp.quick_gm(y.view(B,H*W,1)).shape)
        #target = torch.div(target,cp.quick_gm(target.view(B,H*W,1)).expand(B,H*W).view(B,1,H,W))
        component_target = cp.decompose_depth_map([], self.normalize(target), 7)[::-1]
        ord_components = cp.decompose_depth_map([], self.normalize(u.depth2label_sid(cp.resize(target,8), cuda=is_cuda)), 3)[::-1]
        component_target[0] = ord_components[0]
       
        #optimize weight layer
        components, loss = cp.optimize_components(fine_detail_list, component_target, is_cuda)
        #returns optimal candidates are recombined to final depth map
        final = cp.recombination(components)
        return final,loss
    
    def compute_ordinal_target(self, ord_pred, target):
        #resize target to correct size
        target = cp.resize(target, ord_pred.shape[2])
        if is_cuda:
            target = target.cuda()
        #print(target.shape)
        #transform with ordinal regression so it can be compared
        ord_target = u.depth2label_sid(target, cuda=is_cuda)
        return ord_target
    
    def normalize(self, batch):
        B,C,H,W = batch.size()
        if is_cuda:
            return torch.div(batch,cp.quick_gm(batch.view(B,H*W,1), H).expand(B,H*W).view(B,1,H,W)).cuda()
        return torch.div(batch,cp.quick_gm(batch.view(B,H*W,1), H).expand(B,H*W).view(B,1,H,W))
        #return torch.div(batch,cp.quick_gm(batch.view(B,H*W,1), H).expand(B,H*W).view(B,1,H,W)) 
