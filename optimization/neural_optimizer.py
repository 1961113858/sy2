import torch
import torch.nn.functional as F
from torch.optim import Adam

class NeuralLayoutOptimizer:
    def __init__(self, model, config):
        """
        神经网络布局优化器
        Args:
            model: LEGO模型
            config: 配置对象
        """
        self.model = model
        self.config = config
        self.device = config.train.device
        
        # 优化参数
        self.lr = config.optimizer.learning_rate
        self.num_steps = config.optimization.num_steps
        self.noise_std_pos = config.optimization.pos_noise_level
        self.noise_std_angle = config.optimization.ang_noise_level
        
        # 损失权重
        self.overlap_weight = config.optimization.overlap_weight
        self.rel_weight = config.optimization.rel_weight
        
    def add_noise(self, scene):
        """添加噪声到场景"""
        noisy_scene = scene.clone()
        
        # 添加位置噪声
        pos_noise = torch.randn_like(noisy_scene[:, :, :3]) * self.noise_std_pos
        if self.config.optimization.denoise_weigh_by_class:
            # 根据物体体积加权噪声
            volumes = torch.prod(noisy_scene[:, :, 6:9], dim=2, keepdim=True)
            pos_noise = pos_noise * (1.0 / (volumes + 1e-6))
            
        noisy_scene[:, :, :3] = noisy_scene[:, :, :3] + pos_noise
        
        # 添加旋转噪声
        ang_noise = torch.randn_like(noisy_scene[:, :, 3:6]) * self.noise_std_angle
        noisy_scene[:, :, 3:6] = noisy_scene[:, :, 3:6] + ang_noise
        
        return noisy_scene
        
    def optimize_layout(self, init_scene):
        """
        优化场景布局
        Args:
            init_scene: 初始场景 [B, N, D]
        Returns:
            optimized_scene: 优化后的场景 [B, N, D]
        """
        print("开始优化布局...")
        print(f"初始场景形状: {init_scene.shape}")
        
        # 添加噪声
        noisy_scene = self.add_noise(init_scene)
        
        # 将场景编码到潜在空间
        with torch.no_grad():
            mu, log_var = self.model.encode(noisy_scene)
            z = self.model.reparameterize(mu, log_var)
            print(f"潜在向量形状: {z.shape}")
            
        # 优化潜在变量
        z = z.detach().requires_grad_()
        optimizer = Adam([z], lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=20, verbose=True
        )
        
        best_scene = None
        best_loss = float('inf')
        plateau_count = 0
        prev_loss = float('inf')
        
        for step in range(self.num_steps):
            optimizer.zero_grad()
            
            # 解码生成场景
            output = self.model(z)
            scene = output['abs_attrs']
            
            # 应用场景约束
            scene = self.apply_constraints(scene)
            
            # 计算损失
            loss_dict = self.compute_losses(scene, init_scene)
            total_loss = sum(loss_dict.values())
            
            # 检查损失是否停滞
            if abs(total_loss.item() - prev_loss) < 1e-6:
                plateau_count += 1
            else:
                plateau_count = 0
            prev_loss = total_loss.item()
            
            # 如果损失停滞，添加随机扰动
            if plateau_count > 5:
                z = z + torch.randn_like(z) * 0.01
                plateau_count = 0
                print(f"步骤 {step+1}: 添加随机扰动")
            
            # 保存最佳结果
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_scene = scene.detach().clone()
                print(f"步骤 {step+1}: 更新最佳损失为 {best_loss:.4f}")
                
                # 如果损失足够小，提前停止
                if best_loss < 0.01:
                    print(f"达到目标损失，在步骤 {step+1} 停止优化")
                    break
            
            # 反向传播
            total_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_([z], max_norm=1.0)
            
            optimizer.step()
            scheduler.step(total_loss)
            
            if (step + 1) % 10 == 0:
                print(f"步骤 {step+1}/{self.num_steps}")
                for name, value in loss_dict.items():
                    print(f"{name}: {value.item():.4f}")
                print(f"学习率: {optimizer.param_groups[0]['lr']:.6f}")
        
        return best_scene if best_scene is not None else scene.detach()
        
    def apply_constraints(self, scene):
        """应用场景约束"""
        constrained_scene = scene.clone()
        
        # 确保所有家具都在地面上
        constrained_scene = torch.cat([
            constrained_scene[:, :, :2],  # x, y保持不变
            torch.abs(constrained_scene[:, :, 2:3]),  # z坐标必须为正
            constrained_scene[:, :, 3:]   # 其他属性保持不变
        ], dim=2)
        
        # 确保尺寸合理
        constrained_scene = torch.cat([
            constrained_scene[:, :, :6],  # 位置和旋转保持不变
            torch.clamp(
                constrained_scene[:, :, 6:9],
                min=self.config.optimization.size_min,
                max=self.config.optimization.size_max
            ),  # 限制尺寸范围
            constrained_scene[:, :, 9:]   # 类别保持不变
        ], dim=2)
        
        # 确保位置在合理范围内
        if self.config.optimization.denoise_within_floorplan:
            constrained_scene = torch.cat([
                torch.clamp(
                    constrained_scene[:, :, :2],
                    min=-self.config.optimization.position_range,
                    max=self.config.optimization.position_range
                ),  # 限制x-y平面的范围
                constrained_scene[:, :, 2:]  # 其他属性保持不变
            ], dim=2)
        
        # 确保旋转角度在合理范围内
        constrained_scene = torch.cat([
            constrained_scene[:, :, :3],  # 位置保持不变
            torch.remainder(
                constrained_scene[:, :, 3:6],
                2 * torch.pi
            ),  # 将旋转角度限制在[0, 2π]范围内
            constrained_scene[:, :, 6:]   # 其他属性保持不变
        ], dim=2)
        
        return constrained_scene
        
    def compute_losses(self, scene, target):
        """计算优化损失"""
        losses = {}
        
        # 重建损失
        losses['recon'] = F.mse_loss(scene, target)
        
        # 重叠损失
        losses['overlap'] = self.model.compute_overlap_loss(scene) * self.overlap_weight
        
        # 相对属性损失
        with torch.no_grad():
            target_rel = self.model.rel_predictor(target)
        pred_rel = self.model.rel_predictor(scene)
        losses['rel'] = F.mse_loss(pred_rel, target_rel) * self.rel_weight
        
        # 距离约束损失
        losses['distance'] = self.compute_distance_loss(scene)
        
        # 地板平面约束损失
        if self.config.optimization.denoise_within_floorplan:
            losses['floorplan'] = self.compute_floorplan_loss(scene)
        
        return losses
        
    def compute_distance_loss(self, scene):
        """计算距离约束损失"""
        positions = scene[:, :, :2]  # 只考虑x-y平面
        batch_size, num_objects = positions.size(0), positions.size(1)
        
        min_distance = self.config.optimization.min_distance
        distance_loss = torch.tensor(0.0, device=scene.device)
        
        for i in range(num_objects):
            for j in range(i + 1, num_objects):
                dist = torch.norm(positions[:, i] - positions[:, j], dim=1)
                distance_loss = distance_loss + torch.relu(min_distance - dist).mean()
        
        return distance_loss
        
    def compute_floorplan_loss(self, scene):
        """计算地板平面约束损失"""
        positions = scene[:, :, :2]  # 只考虑x-y平面
        range_limit = self.config.optimization.position_range
        
        # 计算超出范围的惩罚
        out_of_range = torch.relu(torch.abs(positions) - range_limit)
        floorplan_loss = out_of_range.mean() * 10.0  # 权重为10.0
        
        return floorplan_loss 