from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import pytorch_lightning as pl


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
            self.mnist_train, _ = random_split(mnist_full, [55000, 5000])
        
        if stage == "fit" or stage == "validate" or stage is None:
            mnist_full = datasets.MNIST(self.data_dir, train=True, transform=self.train_transform)
            _, self.mnist_val = random_split(mnist_full, [55000, 5000])
            self.trainer._logger.info(f'stage:{stage}, the val dataset has loaded!------------------------------')


        if stage == 'test' or stage is None:
            self.mnist_test = datasets.MNIST(self.data_dir, train=False, transform=self.val_test_transform)
    
    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, shuffle=True, num_workers=4)
    
    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=4)
    
    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=1, num_workers=4)