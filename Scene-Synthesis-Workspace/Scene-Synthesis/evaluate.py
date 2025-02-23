import os
import torch
import hydra
import numpy as np
from tqdm import tqdm
from models.vae import VAE
from data.dataset import RoomDataset
from utils.metrics import compute_metrics, compute_coverage
from utils.visualization import visualize_scene

class Evaluator:
    def __init__(self, config):
        """
        评估器
        Args:
            config: 配置对象
        """
        self.config = config
        self.device = config.train.device
        
        # 加载模型
        self.model = VAE.load_from_checkpoint(
            config.test.checkpoint
        ).to(self.device).eval()
        
        # 创建输出目录
        os.makedirs(config.test.output_dir, exist_ok=True)
        
    def evaluate(self):
        """运行评估"""
        # 加载测试数据
        test_dataset = RoomDataset(
            data_root=self.config.dataset.root,
            room_type=self.config.dataset.room_types[0],
            split="test",
            max_parts=self.config.dataset.max_parts
        )
        
        # 评估重建质量
        recon_metrics = self.evaluate_reconstruction(test_dataset)
        
        # 评估生成质量
        gen_metrics = self.evaluate_generation(test_dataset)
        
        # 合并所有指标
        metrics = {**recon_metrics, **gen_metrics}
        
        # 保存指标
        self.save_metrics(metrics)
        
        return metrics
        
    def evaluate_reconstruction(self, dataset):
        """评估重建质量"""
        all_metrics = []
        
        for scene in tqdm(dataset, desc="Evaluating reconstruction"):
            # 将场景移到设备
            scene = {k: v.to(self.device) for k, v in scene.items()}
            
            # 重建
            with torch.no_grad():
                pred = self.model(scene["abs_attrs"])
                recon = pred["recon"]
            
            # 计算指标
            metrics = compute_metrics(
                scene["abs_attrs"].cpu().numpy(),
                recon.cpu().numpy()
            )
            all_metrics.append(metrics)
            
        # 计算平均指标
        avg_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            avg_metrics[f"recon_{key}"] = np.mean(values)
            
        return avg_metrics
        
    def evaluate_generation(self, dataset, num_samples=100):
        """评估生成质量"""
        # 生成样本
        with torch.no_grad():
            z = torch.randn(
                num_samples,
                self.model.latent_dim
            ).to(self.device)
            samples = self.model.decode(z)
            
        # 计算覆盖率
        coverage = compute_coverage(
            dataset[0]["abs_attrs"].numpy(),
            samples.cpu().numpy()
        )
        
        # 计算多样性
        diversity = self.compute_diversity(samples)
        
        return {
            "gen_coverage": coverage,
            "gen_diversity": diversity
        }
        
    def compute_diversity(self, samples):
        """计算生成样本的多样性"""
        # 计算所有样本对之间的距离
        samples = samples.cpu().numpy()
        N = len(samples)
        
        distances = []
        for i in range(N):
            for j in range(i+1, N):
                dist = np.linalg.norm(
                    samples[i] - samples[j]
                )
                distances.append(dist)
                
        return np.mean(distances)
        
    def save_metrics(self, metrics):
        """保存评估指标"""
        save_path = os.path.join(
            self.config.test.output_dir,
            "evaluation.txt"
        )
        
        with open(save_path, "w") as f:
            for key, value in metrics.items():
                f.write(f"{key}: {value:.4f}\n")

@hydra.main(config_path="configs", config_name="config")
def main(config):
    evaluator = Evaluator(config)
    metrics = evaluator.evaluate()
    print("Evaluation metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

if __name__ == "__main__":
    main()
