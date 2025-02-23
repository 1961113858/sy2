import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from .encoder import SceneEncoder
from .decoder import SceneDecoder

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, h, adj=None):
        Wh = self.W(h)  # [B, N, out_features]
        
        # 计算注意力系数
        B, N = h.size(0), h.size(1)
        a_input = torch.cat([
            Wh.repeat_interleave(N, dim=1),  # 源节点
            Wh.repeat(1, N, 1)  # 目标节点
        ], dim=2)  # [B, N*N, 2*out_features]
        
        e = self.leakyrelu(self.a(a_input))  # [B, N*N, 1]
        e = e.view(B, N, N)  # [B, N, N]
        
        if adj is not None:
            e = e.masked_fill(adj == 0, float('-inf'))
            
        attention = F.softmax(e, dim=2)
        attention = self.dropout(attention)
        
        h_prime = torch.bmm(attention, Wh)  # [B, N, out_features]
        
        return h_prime, attention

class RelativeAttrPredictor(nn.Module):
    def __init__(self, abs_dim, hidden_dims=[128, 64]):
        """
        相对属性预测器
        Args:
            abs_dim: 绝对属性维度
            hidden_dims: 隐藏层维度列表
        """
        super().__init__()
        self.abs_dim = abs_dim
        
        # 构建MLP层
        layers = []
        input_dim = abs_dim * 2  # 两个物体的特征拼接
        
        # 第一层
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        
        # 第二层
        layers.append(nn.Linear(hidden_dims[0], hidden_dims[0]))
        layers.append(nn.ReLU())
        
        # 第三层
        layers.append(nn.Linear(hidden_dims[0], hidden_dims[1]))
        layers.append(nn.ReLU())
        
        # 最终输出层
        layers.append(nn.Linear(hidden_dims[1], 10))  # 相对属性维度为10
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, abs_attrs):
        """
        预测相对属性
        Args:
            abs_attrs: 绝对属性 [B, N, D]
        Returns:
            rel_attrs: 相对属性 [B, N, N, 10]
            attentions: None (为了保持接口一致)
        """
        batch_size, N = abs_attrs.size(0), abs_attrs.size(1)
        
        # 获取所有物体对
        i, j = torch.meshgrid(torch.arange(N), torch.arange(N))
        i, j = i.flatten(), j.flatten()
        
        # 提取物体对特征
        obj_i = abs_attrs[:, i]  # [B, N*N, D]
        obj_j = abs_attrs[:, j]  # [B, N*N, D]
        pair_feats = torch.cat([obj_i, obj_j], dim=-1)  # [B, N*N, 2D]
        
        # 预测相对属性
        rel_attrs = self.mlp(pair_feats)  # [B, N*N, 10]
        rel_attrs = rel_attrs.view(batch_size, N, N, 10)  # [B, N, N, 10]
        
        return rel_attrs, None

class LEGO(pl.LightningModule):
    def __init__(self, config):
        """
        LEGO场景生成模型
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
            nn.Linear(self.abs_dim * self.max_parts, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256)
        )
        
        # 均值和方差预测
        self.fc_mu = nn.Linear(256, 256)
        self.fc_var = nn.Linear(256, 256)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, self.abs_dim * self.max_parts)
        )
        
        # 相对属性预测器
        self.rel_predictor = RelativeAttrPredictor(
            abs_dim=self.abs_dim,
            hidden_dims=[128, 64]  # 修改隐藏层维度
        )
        
        # 图注意力层
        self.gat = GraphAttentionLayer(
            in_features=self.abs_dim,
            out_features=self.abs_dim
        )
        
    def encode(self, x):
        """编码"""
        batch_size = x.size(0)
        # 展平输入
        x = x.view(batch_size, -1)
        # 编码
        h = self.encoder(x)
        # 预测均值和方差
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var
        
    def decode(self, z):
        """解码"""
        batch_size = z.size(0)
        # 解码
        x = self.decoder(z)
        # 重塑为原始形状
        x = x.view(batch_size, self.max_parts, self.abs_dim)
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
        
        # 解码绝对属性
        abs_attrs = self.decode(z)
        
        # 预测相对属性
        rel_attrs, attentions = self.rel_predictor(abs_attrs)
        
        return {
            "abs_attrs": abs_attrs,
            "rel_attrs": rel_attrs,
            "mu": mu,
            "log_var": log_var,
            "attentions": attentions
        }
        
    def compute_overlap_loss(self, abs_attrs):
        """
        计算家具之间的重叠损失
        Args:
            abs_attrs: 绝对属性 [B, N, D]
        Returns:
            overlap_loss: 重叠损失
        """
        batch_size, num_objects = abs_attrs.size(0), abs_attrs.size(1)
        
        # 提取位置和尺寸信息
        positions = abs_attrs[:, :, :2]  # 只考虑x-y平面 [B, N, 2]
        sizes = abs_attrs[:, :, 6:8]     # 只考虑x-y平面的尺寸 [B, N, 2]
        
        # 计算每个物体的边界框
        half_sizes = sizes / 2
        min_coords = positions - half_sizes  # [B, N, 2]
        max_coords = positions + half_sizes  # [B, N, 2]
        
        # 计算所有物体对之间的重叠
        total_overlap = 0
        for i in range(num_objects):
            for j in range(i + 1, num_objects):
                # 计算重叠面积
                overlap_min = torch.maximum(min_coords[:, i], min_coords[:, j])  # [B, 2]
                overlap_max = torch.minimum(max_coords[:, i], max_coords[:, j])  # [B, 2]
                
                # 如果有重叠，计算重叠面积
                overlap_sizes = torch.clamp(overlap_max - overlap_min, min=0)  # [B, 2]
                overlap_area = overlap_sizes[:, 0] * overlap_sizes[:, 1]  # [B]
                
                total_overlap = total_overlap + overlap_area
        
        return total_overlap.mean()
        
    def compute_loss(self, pred, batch):
        """计算损失"""
        # 重建损失
        abs_recon_loss = F.mse_loss(
            pred["abs_attrs"],
            batch["abs_attrs"]
        )
        
        rel_recon_loss = F.mse_loss(
            pred["rel_attrs"],
            batch["rel_attrs"]
        )
        
        # KL散度
        kld = -0.5 * torch.sum(
            1 + pred["log_var"] - pred["mu"].pow(2) - pred["log_var"].exp()
        )
        kld = kld * self.weight_kld
        
        # 重叠损失
        overlap_loss = self.compute_overlap_loss(pred["abs_attrs"]) * 10.0  # 增加重叠惩罚的权重
        
        # 总损失
        total_loss = abs_recon_loss + rel_recon_loss + kld + overlap_loss
        
        return {
            "total": total_loss,
            "abs_recon": abs_recon_loss,
            "rel_recon": rel_recon_loss,
            "kld": kld,
            "overlap": overlap_loss
        }
        
    def training_step(self, batch, batch_idx):
        """训练步骤"""
        # 前向传播
        pred = self(batch["abs_attrs"])
        
        # 计算损失
        loss_dict = self.compute_loss(pred, batch)
        
        # 记录损失
        for name, value in loss_dict.items():
            self.log(f"train_{name}_loss", value)
            
        return loss_dict["total"]
        
    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        # 前向传播
        pred = self(batch["abs_attrs"])
        
        # 计算损失
        loss_dict = self.compute_loss(pred, batch)
        
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
        model = LEGO(config)
        
        # 测试前向传播
        batch = {
            "abs_attrs": torch.randn(16, 80, 9),
            "rel_attrs": torch.randn(16, 80, 80, 10)
        }
        pred = model(batch["abs_attrs"])
        
        print("Input shapes:")
        for key, value in batch.items():
            print(f"{key}:", value.shape)
            
        print("\nOutput shapes:")
        for key, value in pred.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}:", value.shape)
                
        # 测试损失计算
        loss_dict = model.compute_loss(pred, batch)
        print("\nLosses:")
        for key, value in loss_dict.items():
            print(f"{key}:", value.item())
            
    main()
