import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import hydra

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        
        # 定义可学习参数
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
    
    def forward(self, h, adj=None):
        Wh = torch.mm(h, self.W)
        
        # 计算注意力系数
        a_input = torch.cat([Wh.repeat(1, h.size(0)).view(h.size(0) * h.size(0), -1),
                           Wh.repeat(h.size(0), 1)], dim=1).view(h.size(0), -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        
        if adj is not None:
            e = e.masked_fill(adj == 0, float('-inf'))
            
        attention = F.softmax(e, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        h_prime = torch.matmul(attention, Wh)
        
        return h_prime

class RelativeAttrPredictor(nn.Module):
    def __init__(self, abs_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(abs_dim * 2, 128),  # mlp.0
            nn.ReLU(),                    # mlp.1
            nn.BatchNorm1d(128, affine=False),  # mlp.2
            nn.Linear(128, 64),           # mlp.3
            nn.ReLU(),                    # mlp.4
            nn.BatchNorm1d(64, affine=False),   # mlp.5
            nn.Linear(64, 10)             # mlp.6
        )

    def forward(self, abs_attrs):
        """
        前向传播
        Args:
            abs_attrs: 绝对属性 [B, N, D]
        Returns:
            rel_attrs: 相对属性 [B, N, N, 10]
        """
        batch_size, num_objects, feat_dim = abs_attrs.shape
        
        # 创建所有物体对的组合
        obj_i = abs_attrs.unsqueeze(2).expand(batch_size, num_objects, num_objects, feat_dim)
        obj_j = abs_attrs.unsqueeze(1).expand(batch_size, num_objects, num_objects, feat_dim)
        
        # 连接物体对的特征
        pairs = torch.cat([obj_i, obj_j], dim=-1)  # [B, N, N, 2D]
        
        # 重塑以便于处理
        pairs = pairs.view(-1, feat_dim * 2)  # [B*N*N, 2D]
        
        # MLP处理
        x = self.mlp(pairs)
        
        # 重塑回原始维度
        rel_attrs = x.view(batch_size, num_objects, num_objects, -1)
        
        return rel_attrs, None  # 返回None是为了保持与原始接口兼容

class LEGO(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.latent_dim = config.vae.latent_dim  # 添加 latent_dim 属性
        
        # 编码器
        encoder_layers = []
        input_dim = config.vae.abs_dim * config.vae.max_parts
        hidden_dims = [512, 256, 128]
        
        for i, h_dim in enumerate(hidden_dims):
            encoder_layers.extend([
                nn.Linear(input_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim)
            ])
            input_dim = h_dim
        
        self.encoder = nn.ModuleDict({
            'mlp': nn.Sequential(*encoder_layers)
        })
        
        # 均值和方差预测
        self.encoder.update({
            'fc_mu': nn.Linear(hidden_dims[-1], config.vae.latent_dim),
            'fc_var': nn.Linear(hidden_dims[-1], config.vae.latent_dim)
        })
        
        # 解码器
        decoder_layers = []
        input_dim = config.vae.latent_dim
        hidden_dims = [128, 256, 512]
        
        for i, h_dim in enumerate(hidden_dims):
            decoder_layers.extend([
                nn.Linear(input_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim)
            ])
            input_dim = h_dim
        
        decoder_layers.append(nn.Linear(hidden_dims[-1], config.vae.abs_dim * config.vae.max_parts))
        
        self.decoder = nn.ModuleDict({
            'mlp': nn.Sequential(*decoder_layers)
        })
        
        # 图注意力层
        self.gat = GraphAttentionLayer(config.vae.abs_dim, config.vae.abs_dim)
        
        # 相对属性预测器
        self.rel_predictor = RelativeAttrPredictor(config.vae.abs_dim)
    
    def encode(self, x):
        """编码"""
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # 展平
        h = self.encoder['mlp'](x)
        return self.encoder['fc_mu'](h), self.encoder['fc_var'](h)
        
    def decode(self, z):
        """解码"""
        h = self.decoder['mlp'](z)
        return h.view(z.size(0), -1, self.config.vae.abs_dim)
        
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
        abs_attrs = self.decode(z)
        
        # 应用图注意力
        batch_size = abs_attrs.size(0)
        refined_attrs = []
        for i in range(batch_size):
            refined_attrs.append(self.gat(abs_attrs[i]))
        refined_attrs = torch.stack(refined_attrs)
        
        # 预测相对属性
        rel_attrs, _ = self.rel_predictor(refined_attrs)
        
        return {
            'abs_attrs': refined_attrs,
            'rel_attrs': rel_attrs,
            'mu': mu,
            'log_var': log_var
        }
        
    def compute_overlap_loss(self, abs_attrs):
        """计算重叠损失"""
        positions = abs_attrs[:, :, :2]  # 只考虑x-y平面
        sizes = abs_attrs[:, :, 6:8]     # 只考虑x-y平面的尺寸
        
        batch_size, num_objects = positions.size(0), positions.size(1)
        total_overlap = torch.tensor(0.0, device=self.device)
        
        for i in range(num_objects):
            for j in range(i + 1, num_objects):
                # 计算两个物体的边界框
                min1 = positions[:, i] - sizes[:, i] / 2
                max1 = positions[:, i] + sizes[:, i] / 2
                min2 = positions[:, j] - sizes[:, j] / 2
                max2 = positions[:, j] + sizes[:, j] / 2
                
                # 计算重叠面积
                overlap = torch.clamp(
                    torch.min(max1, max2) - torch.max(min1, min2),
                    min=0
                )
                overlap_area = overlap[:, 0] * overlap[:, 1]
                total_overlap = total_overlap + overlap_area.mean()
        
        return total_overlap
        
    def compute_loss(self, pred, batch):
        # 确保维度匹配
        pred_abs_attrs = pred['abs_attrs']  # [B, N, D]
        target_abs_attrs = batch['abs_attrs'].squeeze(1)  # 移除多余的维度 [B, 1, N, D] -> [B, N, D]
        
        # 重建损失
        recon_loss = F.mse_loss(pred_abs_attrs, target_abs_attrs)
        
        # KL散度
        kl_loss = -0.5 * torch.mean(1 + pred['log_var'] - pred['mu'].pow(2) - pred['log_var'].exp())
        
        # 相对属性损失
        pred_rel_attrs = pred['rel_attrs']  # [B, N, N, D]
        target_rel_attrs = batch['rel_attrs'].squeeze(1)  # [B, 1, N, N, D] -> [B, N, N, D]
        rel_loss = F.mse_loss(pred_rel_attrs, target_rel_attrs)
        
        # 重叠损失
        overlap_loss = self.compute_overlap_loss(pred_abs_attrs)
        
        # 总损失
        total_loss = recon_loss + \
                    self.config.vae.weight_kld * kl_loss + \
                    self.config.optimization.rel_weight * rel_loss + \
                    self.config.optimization.overlap_weight * overlap_loss
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'rel_loss': rel_loss,
            'overlap_loss': overlap_loss
        }
        
    def training_step(self, batch, batch_idx):
        pred = self(batch['abs_attrs'])
        losses = self.compute_loss(pred, batch)
        
        # 记录损失
        for name, value in losses.items():
            self.log(f'train_{name}', value)
        
        return losses['total_loss']
        
    def validation_step(self, batch, batch_idx):
        pred = self(batch['abs_attrs'])
        losses = self.compute_loss(pred, batch)
        
        # 记录损失
        for name, value in losses.items():
            self.log(f'val_{name}', value)
        
        return losses['total_loss']
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config.optimizer.learning_rate,
            betas=(self.config.optimizer.beta1, self.config.optimizer.beta2),
            weight_decay=self.config.optimizer.weight_decay
        )
        
        return optimizer

@hydra.main(config_path="../configs", config_name="config")
def main(config):
    # 创建模型
    model = LEGO(config)
    print(model)

if __name__ == "__main__":
    main() 