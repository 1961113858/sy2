import os
import numpy as np
import torch
from torch.utils.data import Dataset

class SceneDataset(Dataset):
    def __init__(self, config, split='train'):
        """
        场景数据集加载器
        Args:
            config: 配置对象
            split: 数据集划分 ('train', 'val', 'test')
        """
        self.config = config
        self.split = split
        
        # 加载数据
        data_path = os.path.join(
            config.data.root,
            'Bedroom_train_val.npy'  # 直接使用合并后的数据文件
        )
        
        # 加载npy数据
        all_data = np.load(data_path, allow_pickle=True)
        
        # 计算数据集划分
        total_samples = len(all_data)
        train_size = int(total_samples * 0.8)
        val_size = int(total_samples * 0.1)
        
        # 根据split选择相应的数据
        if split == 'train':
            self.data = all_data[:train_size]
        elif split == 'val':
            self.data = all_data[train_size:train_size+val_size]
        else:  # test
            self.data = all_data[train_size+val_size:]
            
        print(f"加载了 {len(self.data)} 个{split}样本")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        scene = self.data[idx]
        
        # 提取有效物体的索引
        valid_abs_index = np.where(scene[:, -1])[0]
        
        # 如果物体数量超过最大限制，随机采样
        if len(valid_abs_index) > self.config.vae.max_parts:
            valid_abs_index = np.random.choice(
                valid_abs_index,
                self.config.vae.max_parts,
                replace=False
            )
            valid_abs_index.sort()
        
        # 提取有效物体的属性
        positions = scene[valid_abs_index, 3:6]    # [x, y, z]
        rotations = scene[valid_abs_index, 0:3]    # [rx, ry, rz]
        sizes = scene[valid_abs_index, 6:9]        # [sx, sy, sz]
        categories = scene[valid_abs_index, 9:10]  # [category]
        
        # 合并绝对属性
        abs_attrs = np.concatenate([
            positions,
            rotations,
            sizes,
            categories
        ], axis=1)
        
        # 计算相对属性
        num_objects = len(valid_abs_index)
        rel_attrs = np.zeros((num_objects, num_objects, 10))
        
        for i in range(num_objects):
            for j in range(num_objects):
                if i != j:
                    # 相对位置
                    rel_attrs[i, j, :3] = positions[i] - positions[j]
                    # 相对旋转
                    rel_attrs[i, j, 3:6] = rotations[i] - rotations[j]
                    # 相对尺寸
                    rel_attrs[i, j, 6:9] = sizes[i] / (sizes[j] + 1e-6)
                    # 类别关系（相同为1，不同为0）
                    rel_attrs[i, j, 9] = float(categories[i] == categories[j])
        
        # 填充到固定大小
        if num_objects < self.config.vae.max_parts:
            pad_size = self.config.vae.max_parts - num_objects
            
            # 填充绝对属性
            abs_attrs_padded = np.zeros((self.config.vae.max_parts, 10))
            abs_attrs_padded[:num_objects] = abs_attrs
            abs_attrs = abs_attrs_padded
            
            # 填充相对属性
            rel_attrs_padded = np.zeros((self.config.vae.max_parts, self.config.vae.max_parts, 10))
            rel_attrs_padded[:num_objects, :num_objects] = rel_attrs
            rel_attrs = rel_attrs_padded
        
        return {
            'abs_attrs': torch.FloatTensor(abs_attrs),
            'rel_attrs': torch.FloatTensor(rel_attrs),
            'num_objects': num_objects
        } 