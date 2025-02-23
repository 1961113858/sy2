import os
import torch
import hydra
from tqdm import tqdm
from models.lego import LEGO
from data.dataset import RoomDataset
from utils.visualization import visualize_scene, visualize_scene_3d
from utils.metrics import compute_metrics
from omegaconf import DictConfig

class Tester:
    def __init__(self, config):
        """
        测试器
        Args:
            config: 配置对象
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 加载模型
        checkpoint_path = os.path.join(
            config.train.checkpoint_dir,
            "lego_final.ckpt"
        )
        print(f"加载检查点: {checkpoint_path}")
        
        # 创建模型
        self.model = LEGO(config)
        
        # 加载检查点
        checkpoint = torch.load(checkpoint_path)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        # 移除"model."前缀
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        
        # 尝试加载状态字典，忽略不匹配的键
        try:
            self.model.load_state_dict(state_dict, strict=False)
            print("成功加载模型权重（忽略不匹配的键）")
        except Exception as e:
            print(f"警告: 加载模型权重时出现问题: {e}")
            
        self.model.to(self.device).eval()
        
        # 创建输出目录
        self.output_dir = os.path.join(config.test.output_dir, "test_results")
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"输出目录: {self.output_dir}")
        
    def test(self):
        """运行测试"""
        # 加载测试数据
        test_dataset = RoomDataset(
            data_root=self.config.dataset.root,
            room_type=self.config.dataset.room_types[0],
            split="test",
            max_parts=self.config.vae.max_parts
        )
        
        print(f"测试数据集大小: {len(test_dataset)}")
        
        # 存储所有指标
        all_metrics = []
        
        # 测试每个场景
        for i, scene in enumerate(tqdm(test_dataset)):
            # 将张量数据移到设备
            device_scene = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in scene.items()
            }
            
            # 前向传播
            with torch.no_grad():
                pred = self.model(device_scene["abs_attrs"])
                abs_recon = pred["abs_attrs"]
            
            # 计算指标
            metrics = {
                "abs_recon": compute_metrics(
                    scene["abs_attrs"].cpu().numpy(),
                    abs_recon.cpu().numpy()
                )
            }
            all_metrics.append(metrics)
            
            # 可视化重建结果
            if i < 10:  # 只可视化前10个场景
                # 2D可视化
                save_path = os.path.join(
                    self.output_dir,
                    f"scene_{i:04d}.png"
                )
                visualize_scene(
                    abs_recon.squeeze(0).cpu().numpy(),
                    abs_recon.squeeze(0).cpu().numpy(),
                    save_path=save_path
                )
                print(f"保存2D可视化结果: {save_path}")
                
                # 3D可视化
                save_path_3d = os.path.join(
                    self.output_dir,
                    f"scene_{i:04d}_3d.png"
                )
                visualize_scene_3d(
                    abs_recon.squeeze(0).cpu().numpy(),
                    save_path=save_path_3d
                )
                print(f"保存3D可视化结果: {save_path_3d}")
            
            # 每100个样本打印一次当前指标
            if (i + 1) % 100 == 0:
                print(f"\n处理进度: {i+1}/{len(test_dataset)}")
                current_metrics = {}
                for metric_type in ["abs_recon"]:
                    for key in all_metrics[0][metric_type].keys():
                        values = [m[metric_type][key] for m in all_metrics]
                        current_metrics[f"{metric_type}_{key}"] = sum(values) / len(values)
                print("当前指标:")
                for key, value in current_metrics.items():
                    print(f"{key}: {value:.4f}")
            
        # 计算平均指标
        avg_metrics = {}
        for metric_type in ["abs_recon"]:
            for key in all_metrics[0][metric_type].keys():
                values = [m[metric_type][key] for m in all_metrics]
                avg_metrics[f"{metric_type}_{key}"] = sum(values) / len(values)
            
        # 保存指标
        self.save_metrics(avg_metrics)
        
        return avg_metrics
        
    def save_metrics(self, metrics):
        """保存评估指标"""
        save_path = os.path.join(
            self.output_dir,
            "metrics.txt"
        )
        
        with open(save_path, "w") as f:
            for key, value in metrics.items():
                f.write(f"{key}: {value:.4f}\n")
                
    def generate_samples(self, num_samples=10):
        """生成新的场景样本"""
        print(f"\n生成 {num_samples} 个新场景...")
        
        with torch.no_grad():
            # 从标准正态分布采样
            z = torch.randn(
                num_samples,
                self.model.latent_dim
            ).to(self.device)
            
            # 生成场景
            pred = self.model.decode(z)
            abs_attrs = pred
            
            # 可视化生成的样本
            for i in range(num_samples):
                # 2D可视化
                save_path = os.path.join(
                    self.output_dir,
                    f"sample_{i:04d}.png"
                )
                visualize_scene(
                    abs_attrs[i].cpu().numpy(),
                    save_path=save_path
                )
                print(f"保存2D生成样本: {save_path}")
                
                # 3D可视化
                save_path_3d = os.path.join(
                    self.output_dir,
                    f"sample_{i:04d}_3d.png"
                )
                visualize_scene_3d(
                    abs_attrs[i].cpu().numpy(),
                    save_path=save_path_3d
                )
                print(f"保存3D生成样本: {save_path_3d}")
            
            return abs_attrs

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(config: DictConfig):
    print("配置信息:")
    print(f"数据路径: {config.dataset.root}")
    print(f"房间类型: {config.dataset.room_types}")
    print(f"最大物体数: {config.vae.max_parts}")
    
    # 创建测试器
    tester = Tester(config)
    
    # 运行测试
    metrics = tester.test()
    print("\n最终测试指标:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    # 生成新样本
    print("\n开始生成新场景样本...")
    samples = tester.generate_samples(num_samples=5)
    print("生成完成！")

if __name__ == "__main__":
    main()
