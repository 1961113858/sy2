import os
import hydra
import torch
from pathlib import Path
from omegaconf import DictConfig
from models.lego import LEGO
from optimization.lego_optimizer import LEGOOptimizer
from utils.data import SceneDataset
from utils.visualization import visualize_scene

class SceneOptimizer:
    def __init__(self, config):
        """
        场景优化器
        Args:
            config: 配置对象
        """
        self.config = config
        self.device = config.train.device
        
        # 获取项目根目录
        project_dir = Path("D:/cxcy2/LEGO-Net-main/sy2/Scene-Synthesis-Workspace/Scene-Synthesis")
        
        # 加载LEGO模型
        self.model = LEGO(config)
        checkpoint_path = project_dir / "checkpoints" / "lego_final.ckpt"
        print(f"加载检查点: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        
        # 尝试不同的键名
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint  # 假设直接是状态字典
            
        # 移除"model."前缀
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        
        # 加载状态字典
        try:
            self.model.load_state_dict(state_dict)
            print("成功加载模型权重")
        except Exception as e:
            print(f"加载模型权重时出错: {e}")
            print("可用的键:", state_dict.keys())
            raise
            
        self.model.to(self.device)
        self.model.eval()
        
        # 创建优化器
        self.optimizer = LEGOOptimizer(self.model, config)
        
        # 创建输出目录
        output_dir = project_dir / config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir
        
    def optimize_scene(self, scene):
        """优化单个场景"""
        # 添加批次维度
        if scene.dim() == 2:
            scene = scene.unsqueeze(0)
            
        scene = scene.to(self.device)
        optimized_scene = self.optimizer.optimize_scene(scene)
        return optimized_scene
        
    def run_optimization(self):
        """运行场景优化"""
        # 加载测试数据
        test_dataset = SceneDataset(self.config, split='test')
        
        # 创建输出目录
        output_dir = self.output_dir / 'optimized_scenes'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 优化每个场景
        for i, scene in enumerate(test_dataset):
            print(f"\n优化场景 {i+1}/{len(test_dataset)}")
            
            # 优化场景
            optimized_scene = self.optimize_scene(scene['abs_attrs'])
            
            # 可视化结果
            output_path = output_dir / f'scene_{i:04d}.png'
            visualize_scene(
                scene['abs_attrs'].cpu().numpy(),
                optimized_scene.squeeze(0).cpu().numpy(),
                output_path
            )
            
            if i >= self.config.num_test_samples - 1:
                break

@hydra.main(config_path="configs", config_name="optimize")
def main(config: DictConfig):
    optimizer = SceneOptimizer(config)
    optimizer.run_optimization()

if __name__ == "__main__":
    main()
