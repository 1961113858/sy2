import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

def load_test_data(config):
    """加载测试数据"""
    # 这里应该实现实际的数据加载逻辑
    # 现在我们创建一些随机数据用于测试
    num_samples = config.num_test_samples
    input_dim = config.model.input_dim
    
    test_data = []
    for _ in range(num_samples):
        # 创建随机场景
        num_objects = np.random.randint(3, 8)  # 每个场景3-7个物体
        scene = torch.zeros(num_objects, input_dim)
        
        # 为每个物体生成随机属性
        for i in range(num_objects):
            # 位置 (x, y, z)
            scene[i, 0:3] = torch.rand(3) * 8 - 4  # 范围[-4, 4]
            scene[i, 1] = torch.abs(scene[i, 1])   # y坐标（高度）必须为正
            
            # 尺寸 (width, height, depth)
            scene[i, 3:6] = torch.rand(3) * 2 + 0.5  # 范围[0.5, 2.5]
            
            # 旋转角度
            scene[i, 6] = torch.rand(1) * 2 * np.pi  # 范围[0, 2π]
            
            # 物体类型（one-hot编码）
            obj_type = np.random.randint(0, config.model.num_object_types)
            scene[i, 7 + obj_type] = 1
        
        test_data.append(scene)
    
    return test_data

class SceneDataset(Dataset):
    def __init__(self, scenes):
        self.scenes = scenes
    
    def __len__(self):
        return len(self.scenes)
    
    def __getitem__(self, idx):
        return self.scenes[idx] 