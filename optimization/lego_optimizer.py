import torch
import torch.nn.functional as F
from torch.optim import Adam

class LEGOOptimizer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.lr = config.optimizer.learning_rate
        self.beta1 = config.optimizer.beta1
        self.beta2 = config.optimizer.beta2
        self.overlap_weight = config.optimization.overlap_weight
        self.rel_weight = config.optimization.rel_weight
    
    def add_noise(self, scene):
        """添加噪声到场景"""
        pos_noise = torch.randn_like(scene[:, :3]) * self.config.optimization.pos_noise_level
        ang_noise = torch.randn_like(scene[:, 6:7]) * self.config.optimization.ang_noise_level
        
        noisy_scene = scene.clone()
        noisy_scene[:, :3] += pos_noise
        noisy_scene[:, 6:7] += ang_noise
        
        return noisy_scene
    
    def apply_constraints(self, scene):
        """应用场景约束"""
        # 位置约束
        scene = scene.clone()
        scene[:, :3] = torch.clamp(scene[:, :3], -self.config.optimization.position_range, self.config.optimization.position_range)
        
        # 确保y坐标（高度）为正
        scene[:, 1] = torch.abs(scene[:, 1])
        
        # 尺寸约束
        scene[:, 3:6] = torch.clamp(scene[:, 3:6], self.config.optimization.size_min, self.config.optimization.size_max)
        
        # 角度约束（保持在[0, 2π]范围内）
        scene[:, 6] = torch.remainder(scene[:, 6], 2 * torch.pi)
        
        return scene
    
    def compute_overlap_loss(self, scene):
        """计算重叠损失"""
        positions = scene[:, :3]
        sizes = scene[:, 3:6]
        
        num_objects = positions.shape[0]
        overlap_loss = torch.tensor(0.0, device=scene.device)
        
        for i in range(num_objects):
            for j in range(i + 1, num_objects):
                # 计算两个物体的边界框
                min1 = positions[i] - sizes[i] / 2
                max1 = positions[i] + sizes[i] / 2
                min2 = positions[j] - sizes[j] / 2
                max2 = positions[j] + sizes[j] / 2
                
                # 计算重叠体积
                intersection = torch.max(torch.zeros_like(min1), torch.min(max1, max2) - torch.max(min1, min2))
                overlap_volume = torch.prod(torch.max(intersection, torch.zeros_like(intersection)))
                
                overlap_loss += overlap_volume
        
        return overlap_loss
    
    def compute_distance_constraint(self, scene):
        """计算距离约束损失"""
        positions = scene[:, :3]
        num_objects = positions.shape[0]
        distance_loss = torch.tensor(0.0, device=scene.device)
        
        for i in range(num_objects):
            for j in range(i + 1, num_objects):
                distance = torch.norm(positions[i] - positions[j])
                if distance < self.config.optimization.min_distance:
                    distance_loss += (self.config.optimization.min_distance - distance) ** 2
        
        return distance_loss
    
    def compute_floorplan_constraint(self, scene):
        """计算地板平面约束损失"""
        positions = scene[:, :3]
        sizes = scene[:, 3:6]
        
        # 确保物体在地板上
        height_violation = torch.relu(-(positions[:, 1] - sizes[:, 1] / 2))
        floor_loss = torch.sum(height_violation ** 2)
        
        return floor_loss
    
    def optimize_scene(self, init_scene, num_steps=100):
        """优化场景布局"""
        print(f"初始场景形状: {init_scene.shape}")
        print(f"初始场景范围: [{init_scene.min().item():.4f}, {init_scene.max().item():.4f}]")
        
        # 添加噪声
        noisy_scene = self.add_noise(init_scene)
        print(f"添加噪声后场景范围: [{noisy_scene.min().item():.4f}, {noisy_scene.max().item():.4f}]")
        
        # 将场景编码到潜在空间
        with torch.no_grad():
            mu, log_var = self.model.encode(noisy_scene)
            print(f"编码均值范围: [{mu.min().item():.4f}, {mu.max().item():.4f}]")
            print(f"编码方差范围: [{log_var.min().item():.4f}, {log_var.max().item():.4f}]")
            
            z = self.model.reparameterize(mu, log_var)
            print(f"潜在变量范围: [{z.min().item():.4f}, {z.max().item():.4f}]")
        
        # 优化潜在变量
        z = z.detach().clone().requires_grad_(True)
        optimizer = Adam([z], lr=self.lr, betas=(self.beta1, self.beta2))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        
        best_scene = None
        best_loss = float('inf')
        plateau_count = 0
        prev_loss = float('inf')
        
        for step in range(num_steps):
            optimizer.zero_grad()
            
            # 解码生成场景
            scene = self.model.decode(z)
            if step == 0:
                print(f"解码后场景范围: [{scene.min().item():.4f}, {scene.max().item():.4f}]")
            
            # 应用场景约束
            scene = self.apply_constraints(scene)
            if step == 0:
                print(f"应用约束后场景范围: [{scene.min().item():.4f}, {scene.max().item():.4f}]")
            
            # 计算损失
            loss = self.compute_loss(scene, init_scene)
            if step == 0:
                print(f"初始损失: {loss.item():.4f}")
            
            # 检查损失是否停滞
            if abs(loss.item() - prev_loss) < 1e-6:
                plateau_count += 1
            else:
                plateau_count = 0
            prev_loss = loss.item()
            
            # 如果损失停滞，添加随机扰动
            if plateau_count > 5:
                z = z + torch.randn_like(z) * 0.01
                plateau_count = 0
                print(f"步骤 {step+1}: 添加随机扰动")
            
            # 保存最佳结果
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_scene = scene.detach().clone()
                print(f"步骤 {step+1}: 更新最佳损失为 {best_loss:.4f}")
                
                # 如果损失足够小，提前停止
                if best_loss < 0.01:
                    print(f"达到目标损失，在步骤 {step+1} 停止优化")
                    break
            
            # 反向传播
            loss.backward()
            
            # 检查梯度
            if step == 0:
                print(f"初始梯度范围: [{z.grad.min().item():.4f}, {z.grad.max().item():.4f}]")
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_([z], max_norm=1.0)
            
            optimizer.step()
            scheduler.step(loss)
            
            if (step + 1) % 10 == 0:
                print(f"步骤 {step+1}/{num_steps}, 损失: {loss.item():.4f}, 学习率: {optimizer.param_groups[0]['lr']:.6f}")
                print(f"当前梯度范围: [{z.grad.min().item():.4f}, {z.grad.max().item():.4f}]")
            
        return best_scene if best_scene is not None else scene.detach()
    
    def compute_loss(self, scene, target):
        """计算优化损失"""
        # 重建损失（使用L1损失代替MSE）
        recon_loss = F.l1_loss(scene, target)
        
        # 重叠损失
        overlap_loss = self.compute_overlap_loss(scene) * self.overlap_weight
        
        # 相对属性损失
        with torch.no_grad():
            target_rel_attrs, _ = self.model.rel_predictor(target)
        rel_attrs, _ = self.model.rel_predictor(scene)
        rel_loss = F.l1_loss(rel_attrs, target_rel_attrs) * self.rel_weight
        
        # 距离约束损失
        dist_loss = self.compute_distance_constraint(scene)
        
        # 地板平面约束损失
        if self.config.optimization.denoise_within_floorplan:
            floorplan_loss = self.compute_floorplan_constraint(scene)
        else:
            floorplan_loss = torch.tensor(0.0, device=scene.device)
        
        # 总损失（添加权重）
        total_loss = (
            recon_loss * 1.0 +
            overlap_loss * self.overlap_weight +
            rel_loss * self.rel_weight +
            dist_loss * 5.0 +
            floorplan_loss * 10.0
        )
        
        return total_loss 