import os
from datetime import datetime
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from lightning import ImageClassifier
from dataset import MNISTDataModule
import logging
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

val_nums = 0
test_nums = 0

def main(config: DictConfig):
    # 设置随机种子
    pl.seed_everything(42)
    logger = logging.getLogger(__name__)
    logger.info(f"\nConfig:\n{OmegaConf.to_yaml(config)}") 
    
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
    # logger = TensorBoardLogger('lightning_logs/', name='mnist_classifier_1109')
    
    # 初始化训练器
    trainer = pl.Trainer(
        max_epochs=3,
        callbacks=[checkpoint_callback],
        # logger=logger,
        accelerator='auto',  # 自动选择GPU或CPU
        devices='auto',
        log_every_n_steps=50,
        check_val_every_n_epoch=1,
        limit_train_batches=0.5,
        limit_test_batches=0.1,
        logger = True,
        # default_root_dir='lightning_logs/'
    )
    
    # 供Lightning模块使用
    trainer._logger = logger

    # 训练模型
    print("开始训练...")
    trainer.fit(model, data_module)

    print(f"最好的模型... val_nums = {val_nums}")
    logger.info(f'best model path is {checkpoint_callback.best_model_path}')
    logger.info(f'best model score is {checkpoint_callback.best_model_score}')
    
    # # 测试模型
    # print("开始测试...")
    # trainer.test(model, data_module)
    
    # 加载最佳模型进行最终测试
    logger.info("use the best ckpt to infer...")
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        best_model_path = r'C:\Users\Hope\Desktop\Hope\python\checkpoints\mnist-epoch=00-val_acc=0.95-v15.ckpt'
        best_model = ImageClassifier.load_from_checkpoint(best_model_path)
        trainer.test(best_model, data_module)
        current_time = datetime.now()
        logger.info(f'{current_time.strftime("%Y-%m-%d %H:%M:%S")}, the test has done!')

@hydra.main(config_path="../config", config_name="wrist", version_base="1.1")
def cli(config: DictConfig = None) -> None:
    main(config)

if __name__ == '__main__':
    # 1. on_test_epoch_end如何接收test_step的返回值 --done
    # 2. 测试将信息打印到Logger文件 self.log关联tensorboard --done
    # 3. 属性信息的传递 --done
    cli()