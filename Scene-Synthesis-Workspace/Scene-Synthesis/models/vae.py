import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class VAE(pl.LightningModule):
    def __init__(self, config):
        """
        变分自编码器
        Args:
            config: 配置对象
        """
        super().__init__()
        self.save_hyperparameters()
        
        # 配置参数
        self.latent_dim = config.vae.latent_dim
        self.abs_dim = config.vae.abs_dim
        self.max_parts = config.vae.max_parts
        self.num_class = config.vae.num_class
        self.weight_kld = config.vae.weight_kld
        
        # 编码器
        self.encoder = nn.Sequential(
            # 输入层
            nn.Linear(self.abs_dim * self.max_parts, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            
            # 隐藏层
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )
        
        # 均值和方差预测
        self.fc_mu = nn.Linear(128, self.latent_dim)
        self.fc_var = nn.Linear(128, self.latent_dim)
        
        # 解码器
        self.decoder = nn.Sequential(
            # 输入层
            nn.Linear(self.latent_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            
            # 隐藏层
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            
            # 输出层
            nn.Linear(512, self.abs_dim * self.max_parts)
        )
        
    def encode(self, x):
        """编码"""
        # 展平输入
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        # 编码
        h = self.encoder(x)
        
        # 预测均值和方差
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        
        return mu, log_var
        
    def decode(self, z):
        """解码"""
        # 解码
        h = self.decoder(z)
        
        # 重塑输出
        batch_size = z.size(0)
        x = h.view(batch_size, self.max_parts, self.abs_dim)
        
        return x
        
    def reparameterize(self, mu, log_var):
        """重参数化采样"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, x):
        """前向传播"""
        # 编码
        mu, log_var = self.encode(x)
        
        # 采样
        z = self.reparameterize(mu, log_var)
        
        # 解码
        recon = self.decode(z)
        
        return {
            "recon": recon,
            "mu": mu,
            "log_var": log_var
        }
        
    def compute_loss(self, pred, target):
        """计算损失"""
        # 重建损失
        recon_loss = F.mse_loss(pred["recon"], target)
        
        # KL散度
        kld = -0.5 * torch.sum(
            1 + pred["log_var"] - pred["mu"].pow(2) - pred["log_var"].exp()
        )
        kld = kld * self.weight_kld
        
        # 总损失
        total_loss = recon_loss + kld
        
        return {
            "total": total_loss,
            "recon": recon_loss,
            "kld": kld
        }
        
    def training_step(self, batch, batch_idx):
        """训练步骤"""
        # 前向传播
        pred = self(batch["abs_attrs"])
        
        # 计算损失
        loss_dict = self.compute_loss(pred, batch["abs_attrs"])
        
        # 记录损失
        for name, value in loss_dict.items():
            self.log(f"train_{name}_loss", value)
            
        return loss_dict["total"]
        
    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        # 前向传播
        pred = self(batch["abs_attrs"])
        
        # 计算损失
        loss_dict = self.compute_loss(pred, batch["abs_attrs"])
        
        # 记录损失
        for name, value in loss_dict.items():
            self.log(f"val_{name}_loss", value)
            
        return loss_dict["total"]
        
    def configure_optimizers(self):
        """配置优化器"""
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.config.optimizer.learning_rate,
            weight_decay=self.hparams.config.optimizer.weight_decay
        )

if __name__ == "__main__":
    # 测试模型
    import hydra
    import torch
    
    @hydra.main(config_path="../configs", config_name="config")
    def main(config):
        # 创建模型
        model = VAE(config)
        
        # 测试前向传播
        x = torch.randn(16, 80, 9)
        pred = model(x)
        
        print("Input shape:", x.shape)
        for key, value in pred.items():
            if isinstance(value, torch.Tensor):
                print(f"{key} shape:", value.shape)
                
        # 测试损失计算
        loss_dict = model.compute_loss(pred, x)
        for key, value in loss_dict.items():
            print(f"{key} loss:", value.item())
            
    main()
