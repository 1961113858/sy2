import torch
import numpy as np

def load_test_scene(config):
    """
    加载测试场景
    Args:
        config: 配置对象
    Returns:
        scene: 测试场景张量
    """
    # 创建一个简单的测试场景
    num_objects = 5  # 家具数量
    
    # 创建场景张量 [N, D]
    # D = 10: 位置(3) + 旋转(3) + 尺寸(3) + 类别(1)
    scene = torch.zeros(num_objects, 10)
    
    # 随机初始化位置 (-4 到 4)
    scene[:, :3] = torch.rand(num_objects, 3) * 8 - 4
    scene[:, 1] = torch.abs(scene[:, 1])  # y坐标（高度）必须为正
    
    # 随机初始化旋转 (0 到 2π)
    scene[:, 3:6] = torch.rand(num_objects, 3) * 2 * np.pi
    
    # 随机初始化尺寸 (0.5 到 2.5)
    scene[:, 6:9] = torch.rand(num_objects, 3) * 2 + 0.5
    
    # 随机分配类别 (0 到 4)
    scene[:, 9] = torch.randint(0, 5, (num_objects,))
    
    return scene 