import torch
import torch.nn as nn

class SceneEncoder(nn.Module):
    def __init__(self, abs_dim, max_parts, latent_dim, hidden_dims=[512, 256, 128]):
        """
        场景编码器
        Args:
            abs_dim: 绝对属性维度
            max_parts: 最大物体数量
            latent_dim: 潜在空间维度
            hidden_dims: 隐藏层维度列表
        """
        super().__init__()
        
        # 保存参数
        self.abs_dim = abs_dim
        self.max_parts = max_parts
        self.latent_dim = latent_dim
        
        # 构建MLP层
        layers = []
        input_dim = abs_dim * max_parts  # 10 * 80 = 800
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h_dim))
            input_dim = h_dim
            
        self.mlp = nn.Sequential(*layers)
        
        # 均值和方差预测层
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)
        
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入场景 [B, N, D]
        Returns:
            mu: 均值 [B, latent_dim]
            log_var: 对数方差 [B, latent_dim]
        """
        batch_size = x.size(0)
        
        # 确保输入维度正确
        if x.size(1) != self.max_parts or x.size(2) != self.abs_dim:
            raise ValueError(
                f"输入维度错误：期望 [B, {self.max_parts}, {self.abs_dim}]，"
                f"实际得到 {list(x.size())}"
            )
        
        # 展平输入 [B, N, D] -> [B, N*D]
        x = x.reshape(batch_size, -1)  # 将[B, 80, 10]转换为[B, 800]
        
        # MLP编码
        h = self.mlp(x)
        
        # 预测均值和方差
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        
        return mu, log_var
        
    def encode(self, x):
        """编码接口"""
        return self.forward(x)

if __name__ == "__main__":
    # 测试编码器
    encoder = SceneEncoder(
        abs_dim=10,
        max_parts=80,
        latent_dim=256
    )
    
    # 测试前向传播
    x = torch.randn(16, 80, 10)
    mu, log_var = encoder(x)
    
    print("Input shape:", x.shape)
    print("mu shape:", mu.shape)
    print("log_var shape:", log_var.shape)
