class RelativeAttrPredictor(nn.Module):
    def __init__(self, abs_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(abs_dim * 2, 128),  # mlp.0: [128, 20]
            nn.ReLU(),
            nn.BatchNorm1d(128, affine=False),  # mlp.2: [128]
            nn.Linear(128, 64),  # mlp.3: [64, 128]
            nn.ReLU(),
            nn.BatchNorm1d(64, affine=False),  # mlp.5: [64]
            nn.Linear(64, 10)  # mlp.6: [10, 64]
        )

    def forward(self, abs_attrs):
        batch_size, num_objects, _ = abs_attrs.shape
        
        # Create all possible pairs of objects
        obj_i = abs_attrs.unsqueeze(2).repeat(1, 1, num_objects, 1)
        obj_j = abs_attrs.unsqueeze(1).repeat(1, num_objects, 1, 1)
        
        # Concatenate features for each pair
        pairs = torch.cat([obj_i, obj_j], dim=-1)
        
        # Reshape to (batch_size * num_objects * num_objects, feature_dim)
        pairs = pairs.view(-1, pairs.size(-1))
        
        # Process through MLP
        rel_attrs = self.mlp(pairs)
        
        # Reshape back to (batch_size, num_objects, num_objects, output_dim)
        rel_attrs = rel_attrs.view(batch_size, num_objects, num_objects, -1)
        
        return rel_attrs 