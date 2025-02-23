import os
import torch
import hydra
import numpy as np
import json
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from omegaconf import DictConfig
from models.lego import LEGO
from optimization.neural_optimizer import NeuralLayoutOptimizer
from utils.visualization import visualize_scene
from utils.data import load_test_scene
from pathlib import Path

class SceneOptimizer:
    def __init__(self, config):
        """
        场景优化器
        Args:
            config: 配置对象
        """
        self.config = config
        
        # 初始化LEGO模型
        self.model = LEGO(config)
        
        # 暂时跳过检查点加载
        print("警告：暂时跳过模型检查点加载")
        
        # 设置设备
        self.device = config.train.device
        self.model = self.model.to(self.device)
        
        # 创建神经网络优化器
        self.optimizer = NeuralLayoutOptimizer(
            model=self.model,
            learning_rate=config.optimizer.learning_rate,
            num_steps=config.optimizer.num_steps,
            noise_std_pos=config.noise.position,
            noise_std_angle=config.noise.angle
        )
        
        # 创建输出目录
        self.output_dir = Path(config.output.dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"输出目录: {self.output_dir}")
        
    def optimize_scene(self, scene):
        """优化单个场景"""
        print("\n开始优化场景...")
        
        # 添加批次维度
        if scene.dim() == 2:
            scene = scene.unsqueeze(0)
            
        scene = scene.to(self.device)
        
        # 使用神经网络优化器
        optimized_scene = self.optimizer.optimize_layout(scene)
        
        return optimized_scene
        
    def run_optimization(self):
        """运行场景优化"""
        # 加载测试场景
        test_scene = load_test_scene(self.config)
        print(f"加载测试场景, 形状: {test_scene.shape}")
        
        # 优化场景
        optimized_scene = self.optimize_scene(test_scene)
        
        # 可视化结果
        print("\n保存可视化结果...")
        visualize_scene(
            test_scene.cpu().numpy(),
            optimized_scene.cpu().numpy(),
            save_path=self.output_dir / "optimization_result.png"
        )
        
        print(f"优化完成! 结果保存在: {self.output_dir}")
        return optimized_scene

@hydra.main(config_path="configs", config_name="optimize")
def main(config: DictConfig):
    # 创建优化器
    optimizer = SceneOptimizer(config)
    
    # 运行优化
    optimizer.run_optimization()

if __name__ == "__main__":
    main()

def evaluate_results(scenes, optimized_scenes, config):
    """评估优化结果"""
    metrics = {}
    
    if config.output.metrics.reconstruction_error:
        recon_error = torch.mean(torch.abs(scenes - optimized_scenes))
        metrics['reconstruction_error'] = recon_error.item()
        
    if config.output.metrics.overlap_ratio:
        overlap_ratios = []
        for scene in optimized_scenes:
            overlap_ratio = compute_overlap_ratio(scene)
            overlap_ratios.append(overlap_ratio)
        metrics['overlap_ratio'] = np.mean(overlap_ratios)
        
    if config.output.metrics.scene_validity:
        valid_count = 0
        for scene in optimized_scenes:
            if check_scene_validity(scene):
                valid_count += 1
        metrics['scene_validity'] = valid_count / len(optimized_scenes)
    
    return metrics

def visualize_results(scenes, optimized_scenes, metrics, save_dir):
    """可视化优化结果"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存场景对比图
    for i, (scene, opt_scene) in enumerate(zip(scenes, optimized_scenes)):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        visualize_scene(scene, ax=ax1, title='原始场景')
        visualize_scene(opt_scene, ax=ax2, title='优化后场景')
        
        plt.savefig(os.path.join(save_dir, f'scene_comparison_{i+1}.png'))
        plt.close()
    
    # 保存评估指标
    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # 绘制指标图表
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics_items = list(metrics.items())
    x = range(len(metrics_items))
    values = [v for k, v in metrics_items]
    
    ax.bar(x, values)
    ax.set_xticks(x)
    ax.set_xticklabels([k for k, v in metrics_items], rotation=45)
    ax.set_title('优化结果评估指标')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics.png'))
    plt.close()

def compute_overlap_ratio(scene):
    """计算场景中物体的重叠率"""
    # 提取物体的位置和尺寸信息
    positions = scene[:, :3]  # 假设前3个维度是位置
    sizes = scene[:, 3:6]     # 假设接下来3个维度是尺寸
    
    num_objects = positions.shape[0]
    overlap_count = 0
    total_pairs = num_objects * (num_objects - 1) // 2
    
    # 计算每对物体之间的重叠
    for i in range(num_objects):
        for j in range(i + 1, num_objects):
            # 计算两个物体的边界框
            min1 = positions[i] - sizes[i] / 2
            max1 = positions[i] + sizes[i] / 2
            min2 = positions[j] - sizes[j] / 2
            max2 = positions[j] + sizes[j] / 2
            
            # 检查是否重叠
            overlap = torch.all(max1 > min2) and torch.all(max2 > min1)
            if overlap:
                overlap_count += 1
    
    return overlap_count / total_pairs if total_pairs > 0 else 0.0

def check_scene_validity(scene):
    """检查场景是否有效"""
    # 提取场景信息
    positions = scene[:, :3]
    sizes = scene[:, 3:6]
    
    # 检查位置是否在合理范围内
    position_valid = torch.all((positions >= -5.0) & (positions <= 5.0))
    
    # 检查尺寸是否在合理范围内
    size_valid = torch.all((sizes >= 0.1) & (sizes <= 5.0))
    
    # 检查物体是否在地面上（假设y轴是高度）
    height_valid = torch.all(positions[:, 1] >= 0.0)
    
    # 计算重叠率
    overlap_ratio = compute_overlap_ratio(scene)
    overlap_valid = overlap_ratio < 0.3  # 允许30%的重叠
    
    return position_valid and size_valid and height_valid and overlap_valid

def visualize_scene(scene, ax=None, title=None):
    """可视化场景"""
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
    
    # 提取场景信息
    positions = scene[:, :3].detach().cpu().numpy()
    sizes = scene[:, 3:6].detach().cpu().numpy()
    
    # 绘制每个物体
    for pos, size in zip(positions, sizes):
        # 创建立方体
        x, y, z = pos
        dx, dy, dz = size
        
        # 绘制立方体
        xx = [x-dx/2, x+dx/2]
        yy = [y-dy/2, y+dy/2]
        zz = [z-dz/2, z+dz/2]
        
        xx, yy = np.meshgrid(xx, yy)
        ax.plot_surface(xx, yy, np.full_like(xx, zz[0]))
        ax.plot_surface(xx, yy, np.full_like(xx, zz[1]))
        
        yy, zz = np.meshgrid(yy[:,0], zz)
        ax.plot_surface(np.full_like(yy, xx[0,0]), yy, zz)
        ax.plot_surface(np.full_like(yy, xx[0,1]), yy, zz)
        
        xx, zz = np.meshgrid(xx[0,:], zz[:,0])
        ax.plot_surface(xx, np.full_like(xx, yy[0,0]), zz)
        ax.plot_surface(xx, np.full_like(xx, yy[0,1]), zz)
    
    # 设置视角和标题
    ax.view_init(elev=30, azim=45)
    if title:
        ax.set_title(title)
    
    # 设置轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # 设置轴范围
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_zlim([0, 5])
    
    return ax 