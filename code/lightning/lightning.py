import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, Metric
from net import SimpleCNN


val_num = 0
# 1. 自定义空操作Metric：仅存储数值，不做任何聚合
class SimpleRecorder(Metric):
    def __init__(self):
        super().__init__()
        # 注册一个状态变量，用于存储所有step的数值（列表形式）
        self.add_state("values", default=[], dist_reduce_fx=None)  # dist_reduce_fx=None：不跨进程聚合

    def update(self, value: torch.Tensor):
        # 每次调用时，将当前step的数值添加到列表（注意：需转移到CPU并转为标量，避免设备问题）
        self.values.append(value.detach().cpu())  # 若为张量，可根据需要保留维度（如批量预测结果）

    def compute(self):
        # 计算时返回所有存储的数值（不做任何聚合）
        return self.values

# 定义PyTorch Lightning模块
class ImageClassifier(pl.LightningModule):
    def __init__(self, num_classes=10, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = SimpleCNN(num_classes)
        self.learning_rate = learning_rate
        
        # 定义评估指标
        self.train_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_accuracy = Accuracy(task='multiclass', num_classes=num_classes)

        self.test_loss_record = SimpleRecorder()
        self.test_preds_record = SimpleRecorder()
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        
        # preds = torch.argmax(logits, dim=1)
        # acc = self.train_accuracy(preds, y)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        # self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        # global val_nums
        # val_nums += 1
        x, y = batch # 128,1,28,28/[128]
        logits = self(x) # [128, 10]

        loss = F.nll_loss(logits, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True) # val_loss
        
        preds = torch.argmax(logits, dim=1) # 128
        self.val_accuracy.update(preds, y)
        self.log('val_acc', self.val_accuracy, on_step=True, on_epoch=True, prog_bar=True) # val_acc
        # acc = self.val_accuracy(preds, y)
        # self.log('val_acc', acc, on_epoch=True, prog_bar=True) # val_acc
        
        return loss
    
    def on_validation_epoch_end(self):
        global val_num
        val_num += 1
        self.trainer._logger.info(f'epoch:{val_num} **********************on validation epoch end*************************')
    
    def test_step(self, batch, batch_idx):
        # global test_nums
        # test_nums += 1

        x, y = batch # 128,1,28,28
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log('test_loss', loss, prog_bar=True, logger=True)
        
        preds = torch.argmax(logits, dim=1)
        # self.log('test_pred', preds)
        self.test_accuracy.update(preds, y)
        self.test_loss_record.update(loss)
        self.test_preds_record.update(preds)
        # self.log('test_acc', self.test_accuracy, prog_bar=True)
        # acc = self.test_accuracy(preds, y)
        # self.log('test_acc', acc, prog_bar=True)
    
    def on_test_epoch_end(self):
        all_preds = self.test_preds_record.compute()
        all_loss = self.test_loss_record.compute()
        # TODO 画图
        print(f'test has done, the lengths of preds is {len(all_preds)}==========================')
        # print(all_preds)
        self.log(f'test_nums', len(all_preds), logger=True)
        self.test_preds_record.reset()
        self.test_loss_record.reset()
        print(f'==============logger type is {type(self.logger)}=============')
        self.trainer._logger.info('=================in the Lightning module use logger function!======================')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [scheduler]