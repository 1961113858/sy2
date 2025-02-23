import torch
import torch.nn as nn

class LEGOModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(config.model.input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        
        # 均值和方差预测
        self.fc_mu = nn.Linear(128, config.model.latent_dim)
        self.fc_var = nn.Linear(128, config.model.latent_dim)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(config.model.latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, config.model.output_dim),
        )
        
        # 相对属性预测器
        self.rel_predictor = nn.Sequential(
            nn.Linear(config.model.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, config.model.rel_attr_dim),
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar 