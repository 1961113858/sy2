import torch
import torch.nn as nn

class SceneDecoder(nn.Module):
    def __init__(self, abs_dim, max_parts, latent_dim, hidden_dims=[128, 256, 512]):
        """
        场景解码器
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
        input_dim = latent_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h_dim))
            input_dim = h_dim
            
        # 最终输出层
        layers.append(nn.Linear(hidden_dims[-1], abs_dim * max_parts))
        layers.append(nn.Tanh())  # 使用Tanh限制输出范围
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, z):
        """
        前向传播
        Args:
            z: 潜在向量 [B, latent_dim]
        Returns:
            x: 重建场景 [B, N, D]
        """
        batch_size = z.size(0)
        
        # MLP解码
        x = self.mlp(z)
        
        # 重塑为目标形状
        x = x.view(batch_size, self.max_parts, self.abs_dim)
        
        return x
        
    def decode(self, z):
        """解码接口"""
        return self.forward(z)

if __name__ == "__main__":
    # 测试解码器
    decoder = SceneDecoder(
        abs_dim=10,
        max_parts=80,
        latent_dim=256
    )
    
    # 测试前向传播
    z = torch.randn(16, 256)
    x = decoder(z)
    
    print("Input shape:", z.shape)
    print("Output shape:", x.shape)
    print("Expected output shape: [16, 80, 10]")
