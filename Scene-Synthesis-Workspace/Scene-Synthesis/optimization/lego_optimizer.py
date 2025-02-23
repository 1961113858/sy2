import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

class LEGOOptimizer:
    def __init__(self, lego_model, config):
        """
        LEGO场景优化器
        Args:
            lego_model: LEGO模型
            config: 配置对象
        """
        self.model = lego_model
        self.config = config
        self.device = config.train.device
        
        # 优化器参数
        self.lr = config.optimizer.learning_rate
        self.beta1 = config.optimizer.beta1
        self.beta2 = config.optimizer.beta2
        self.weight_decay = config.optimizer.weight_decay
        
        # 优化权重
        self.overlap_weight = config.optimization.overlap_weight
        self.rel_weight = config.optimization.rel_weight
        
        # 噪声参数
        self.pos_noise_level = config.optimization.pos_noise_level
        self.ang_noise_level = config.optimization.ang_noise_level
        
    def add_noise(self, scene):
        """添加噪声到场景"""
        noisy_scene = scene.clone()
        
        # 添加位置噪声
        pos_noise = torch.randn_like(noisy_scene[:, :, :3]) * self.pos_noise_level
        if self.config.optimization.denoise_weigh_by_class:
            # 根据物体体积加权噪声
            volumes = torch.prod(noisy_scene[:, :, 6:9], dim=2, keepdim=True)
            pos_noise = pos_noise * (1.0 / (volumes + 1e-6))
            
        noisy_scene[:, :, :3] = noisy_scene[:, :, :3] + pos_noise
        
        # 添加旋转噪声
        ang_noise = torch.randn_like(noisy_scene[:, :, 3:6]) * self.ang_noise_level
        noisy_scene[:, :, 3:6] = noisy_scene[:, :, 3:6] + ang_noise
        
        return noisy_scene
        
    def optimize_scene(self, init_scene, num_steps=100):
        """优化场景布局"""
        # 添加噪声
        noisy_scene = self.add_noise(init_scene)
        
        # 将场景编码到潜在空间
        with torch.no_grad():
            mu, log_var = self.model.encode(noisy_scene)
            z = self.model.reparameterize(mu, log_var)
            
        # 优化潜在变量
        z = z.detach().clone().requires_grad_(True)
        optimizer = Adam([z], lr=self.lr, betas=(self.beta1, self.beta2))
        
        best_scene = None
        best_loss = float('inf')
        
        for step in range(num_steps):
            optimizer.zero_grad()
            
            # 解码生成场景
            scene = self.model.decode(z)
            
            # 应用场景约束
            scene = self.apply_constraints(scene)
            
            # 计算损失
            loss = self.compute_loss(scene, init_scene)
            
            # 保存最佳结果
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_scene = scene.detach().clone()
                
                # 如果损失足够小，提前停止
                if best_loss < 0.01:
                    print(f"达到目标损失，在步骤 {step+1} 停止优化")
                    break
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            if (step + 1) % 10 == 0:
                print(f"步骤 {step+1}/{num_steps}, 损失: {loss.item():.4f}")
            
        return best_scene if best_scene is not None else scene.detach()
        
    def compute_loss(self, scene, target):
        """计算优化损失"""
        # 重建损失
        recon_loss = F.mse_loss(scene, target)
        
        # 重叠损失
        overlap_loss = self.compute_overlap_loss(scene) * self.overlap_weight
        
        # 相对属性损失
        with torch.no_grad():
            target_rel_attrs, _ = self.model.rel_predictor(target)
        rel_attrs, _ = self.model.rel_predictor(scene)
        rel_loss = F.mse_loss(rel_attrs, target_rel_attrs) * self.rel_weight
        
        # 距离约束损失
        dist_loss = self.compute_distance_constraint(scene)
        
        # 地板平面约束损失
        if self.config.optimization.denoise_within_floorplan:
            floorplan_loss = self.compute_floorplan_constraint(scene)
        else:
            floorplan_loss = torch.tensor(0.0, device=scene.device)
        
        # 总损失
        total_loss = recon_loss + overlap_loss + rel_loss + dist_loss + floorplan_loss
        
        return total_loss
        
    def compute_overlap_loss(self, scene):
        """计算家具之间的重叠损失"""
        if not self.config.optimization.denoise_no_penetration:
            return torch.tensor(0.0, device=scene.device)
            
        positions = scene[:, :, :2].detach()  # 只考虑x-y平面
        sizes = scene[:, :, 6:8].detach()     # 只考虑x-y平面的尺寸
        
        # 计算每个物体的边界框
        half_sizes = sizes / 2
        min_coords = positions - half_sizes  # [B, N, 2]
        max_coords = positions + half_sizes  # [B, N, 2]
        
        # 计算所有物体对之间的重叠
        total_overlap = torch.tensor(0.0, device=scene.device)
        num_objects = scene.size(1)
        
        for i in range(num_objects):
            for j in range(i + 1, num_objects):
                # 计算重叠面积
                overlap_min = torch.max(min_coords[:, i], min_coords[:, j])  # [B, 2]
                overlap_max = torch.min(max_coords[:, i], max_coords[:, j])  # [B, 2]
                
                # 如果有重叠，计算重叠面积
                overlap_sizes = torch.clamp(overlap_max - overlap_min, min=0)  # [B, 2]
                overlap_area = overlap_sizes[:, 0] * overlap_sizes[:, 1]  # [B]
                
                total_overlap = total_overlap + overlap_area.mean()
        
        return total_overlap
        
    def compute_distance_constraint(self, scene):
        """计算家具之间的最小距离约束"""
        positions = scene[:, :, :2].detach()  # 只考虑x-y平面
        batch_size, num_objects = positions.size(0), positions.size(1)
        
        min_distance = self.config.optimization.min_distance  # 最小距离阈值（米）
        dist_loss = torch.tensor(0.0, device=scene.device)
        
        for i in range(num_objects):
            for j in range(i + 1, num_objects):
                # 计算两个物体之间的距离
                dist = torch.norm(positions[:, i] - positions[:, j], dim=1)
                # 如果距离小于阈值，添加惩罚
                dist_loss = dist_loss + torch.relu(min_distance - dist).mean()
        
        return dist_loss * 5.0  # 权重为5.0
        
    def compute_floorplan_constraint(self, scene):
        """计算地板平面约束损失"""
        positions = scene[:, :, :2]  # 只考虑x-y平面
        range_limit = self.config.optimization.position_range
        
        # 计算超出范围的惩罚
        out_of_range = torch.relu(torch.abs(positions) - range_limit)
        floorplan_loss = out_of_range.mean() * 10.0  # 权重为10.0
        
        return floorplan_loss
        
    def apply_constraints(self, scene):
        """应用场景约束"""
        # 创建新的张量而不是原地修改
        constrained_scene = scene.clone()
        
        # 确保所有家具都在地面上
        constrained_scene = torch.cat([
            constrained_scene[:, :, :2],
            torch.abs(constrained_scene[:, :, 2:3]),
            constrained_scene[:, :, 3:]
        ], dim=2)
        
        # 确保尺寸合理
        constrained_scene = torch.cat([
            constrained_scene[:, :, :6],
            torch.clamp(
                constrained_scene[:, :, 6:9],
                min=self.config.optimization.size_min,
                max=self.config.optimization.size_max
            ),
            constrained_scene[:, :, 9:]
        ], dim=2)
        
        # 确保位置在合理范围内
        if self.config.optimization.denoise_within_floorplan:
            constrained_scene = torch.cat([
                torch.clamp(
                    constrained_scene[:, :, :2],
                    min=-self.config.optimization.position_range,
                    max=self.config.optimization.position_range
                ),
                constrained_scene[:, :, 2:]
            ], dim=2)
        
        # 确保旋转角度在合理范围内
        constrained_scene = torch.cat([
            constrained_scene[:, :, :3],
            torch.clamp(
                constrained_scene[:, :, 3:6],
                min=-torch.pi,
                max=torch.pi
            ),
            constrained_scene[:, :, 6:]
        ], dim=2)
        
        return constrained_scene
