class RelativeAttrPredictor(nn.Module):
    def __init__(self, abs_dim):
        super(RelativeAttrPredictor, self).__init__()
        input_dim = abs_dim * 2
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),  # mlp.0
            nn.ReLU(),                  # mlp.1
            nn.BatchNorm1d(128, affine=False),  # mlp.2 - 不使用仿射变换
            nn.Linear(128, 128),        # mlp.3
            nn.ReLU(),                  # mlp.4
            nn.BatchNorm1d(128, affine=False),  # mlp.5 - 不使用仿射变换
            nn.Linear(128, 10)          # mlp.6
        )
        
        # 初始化BatchNorm层的权重
        for m in self.mlp.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        return self.mlp(x) 