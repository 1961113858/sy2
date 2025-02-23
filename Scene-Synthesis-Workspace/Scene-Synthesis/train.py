import os
import hydra
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from models.lego import LEGO
from data.dataset import RoomDataset

class SceneTrainer:
    def __init__(self, config):
        """
        场景生成模型训练器
        Args:
            config: 配置对象
        """
        self.config = config
        
        # 创建数据加载器
        self.train_loader = self._create_dataloader("train")
        self.val_loader = self._create_dataloader("val")
        
        # 创建模型
        self.model = LEGO(config)
        
        # 创建检查点回调
        self.checkpoint_callback = ModelCheckpoint(
            dirpath="checkpoints",  # 使用固定路径
            filename="lego-{epoch:02d}-{val_total_loss:.4f}",
            monitor="val_total_loss",
            mode="min",
            save_top_k=3
        )
        
        # 创建日志记录器
        if config.logging.wandb:
            self.logger = WandbLogger(
                project=config.logging.project,
                name=f"lego_{config.dataset.room_types[0]}"
            )
        else:
            self.logger = None
            
    def _create_dataloader(self, split):
        """创建数据加载器"""
        dataset = RoomDataset(
            data_root=self.config.dataset.root,
            room_type=self.config.dataset.room_types[0],
            split=split,
            max_parts=self.config.vae.max_parts
        )
        
        return DataLoader(
            dataset,
            batch_size=self.config.dataset.batch_size,
            shuffle=(split == "train"),
            num_workers=self.config.dataset.num_workers,
            pin_memory=True
        )
        
    def train(self):
        """训练模型"""
        # 创建训练器
        trainer = pl.Trainer(
            max_epochs=100,  # 使用固定值
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1,
            callbacks=[self.checkpoint_callback],
            logger=self.logger,
            log_every_n_steps=100,  # 使用固定值
            val_check_interval=0.25  # 每1/4个epoch验证一次
        )
        
        # 开始训练
        trainer.fit(
            self.model,
            train_dataloaders=self.train_loader,
            val_dataloaders=self.val_loader
        )
        
        # 保存最终模型
        trainer.save_checkpoint("checkpoints/lego_final.ckpt")

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(config):
    # 创建输出目录
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # 创建训练器
    trainer = SceneTrainer(config)
    
    # 开始训练
    trainer.train()

if __name__ == "__main__":
    main()
