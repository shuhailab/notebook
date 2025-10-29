import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics import Accuracy
import os


val_nums = 0
test_nums = 0

# 定义数据变换
def get_transforms():
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    val_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    return train_transform, val_test_transform

# 定义简单的CNN模型
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# 定义数据模块
class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir=r'D:\\data', batch_size=64):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_transform, self.val_test_transform = get_transforms()
        
    def prepare_data(self):
        # 下载数据
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)
    
    def setup(self, stage=None):
        # 设置数据
        if stage == 'fit' or stage is None:
            mnist_full = datasets.MNIST(self.data_dir, train=True, transform=self.train_transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])
        
        if stage == 'test' or stage is None:
            self.mnist_test = datasets.MNIST(self.data_dir, train=False, transform=self.val_test_transform)
    
    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, shuffle=True, num_workers=4)
    
    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=4)
    
    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=4)
    

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
        global val_nums
        val_nums += 1
        x, y = batch # 128,1,28,28/[128]
        logits = self(x) # [128, 10]

        loss = F.nll_loss(logits, y)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True) # val_loss
        
        preds = torch.argmax(logits, dim=1) # 128
        self.val_accuracy.update(preds, y)
        self.log('val_acc', self.val_accuracy, on_epoch=True, prog_bar=True) # val_acc
        # acc = self.val_accuracy(preds, y)
        # self.log('val_acc', acc, on_epoch=True, prog_bar=True) # val_acc
        
        return loss
    
    def test_step(self, batch, batch_idx):
        global test_nums
        test_nums += 1

        x, y = batch # 128,1,28,28
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log('test_loss', loss, prog_bar=True)
        
        preds = torch.argmax(logits, dim=1)
        self.test_accuracy.update(preds, y)
        self.log('test_acc', self.test_accuracy, prog_bar=True)
        # acc = self.test_accuracy(preds, y)
        # self.log('test_acc', acc, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [scheduler]


def main():
    # 设置随机种子
    pl.seed_everything(42)
    
    # 初始化数据模块
    data_module = MNISTDataModule(batch_size=128)
    
    # 初始化模型
    model = ImageClassifier(num_classes=10, learning_rate=1e-3)
    
    # 设置模型检查点回调
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',           # 监控验证集准确率
        dirpath='checkpoints/',      # 保存路径
        filename='mnist-{epoch:02d}-{val_acc:.2f}',  # 文件名格式
        save_top_k=3,                # 保存最好的3个模型
        mode='max',                  # 监控指标越大越好
        save_last=True,              # 同时保存最后一个epoch的模型
        every_n_epochs=1             # 每1个epoch保存一次
    )
    
    # 设置早停回调（可选）
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_acc',
        patience=5,
        mode='min'
    )
    
    # 设置TensorBoard日志记录器
    logger = TensorBoardLogger('lightning_logs/', name='mnist_classifier')
    
    # 初始化训练器
    trainer = pl.Trainer(
        max_epochs=1,
        callbacks=[checkpoint_callback],
        logger=logger,
        accelerator='auto',  # 自动选择GPU或CPU
        devices='auto',
        log_every_n_steps=50,
        check_val_every_n_epoch=1
    )
    
    # 训练模型
    print("开始训练...")
    trainer.fit(model, data_module)

    print(f"最好的模型... val_nums = {val_nums}")
    print(f'best model path is {checkpoint_callback.best_model_path}')
    print(f'best model score is {checkpoint_callback.best_model_score}')
    
    # # 测试模型
    # print("开始测试...")
    # trainer.test(model, data_module)
    
    # 加载最佳模型进行最终测试
    print("使用最佳模型进行测试...")
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        best_model = ImageClassifier.load_from_checkpoint(best_model_path)
        trainer.test(best_model, data_module)
        print(f'test_nums is {test_nums}')

if __name__ == '__main__':
    main()