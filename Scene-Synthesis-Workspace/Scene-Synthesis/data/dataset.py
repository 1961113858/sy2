import os
import torch
import numpy as np
import h5py
from torch.utils.data import Dataset

class RoomDataset(Dataset):
    def __init__(self, data_root, room_type, split="train", max_parts=80):
        """
        房间场景数据集
        Args:
            data_root: 数据根目录
            room_type: 房间类型 ('bedroom' or 'living')
            split: 数据集划分 ('train', 'val', 'test')
            max_parts: 最大物体数量
        """
        self.data_root = data_root
        self.room_type = room_type
        self.split = split
        self.max_parts = max_parts
        
        # 加载数据
        self.data = self._load_data()
        
        # 加载或生成数据集划分索引
        self.indices = self._load_or_generate_split_indices()
        
    def __len__(self):
        return len(self.indices)
        
    def __getitem__(self, idx):
        # 获取实际索引
        real_idx = self.indices[idx]
        
        # 获取场景数据
        scene_data = {
            key: torch.from_numpy(value[real_idx]).float()
            for key, value in self.data.items()
        }
        
        # 处理数据
        processed_data = self._process_data(scene_data)
        
        # 添加批次维度
        for key in processed_data:
            if isinstance(processed_data[key], torch.Tensor):
                processed_data[key] = processed_data[key].unsqueeze(0)
        
        return processed_data
        
    def _load_data(self):
        """加载h5数据"""
        data_path = os.path.join(
            self.data_root,
            self.room_type,
            f"{self.room_type}.h5"
        )
        print(f"加载数据文件: {data_path}")
        
        # 读取h5文件
        with h5py.File(data_path, "r") as f:
            data = {
                key: f[key][:]
                for key in f.keys()
            }
            
        return data
        
    def _load_or_generate_split_indices(self):
        """加载或生成数据集划分索引"""
        indices_path = os.path.join(
            self.data_root,
            f"{self.split}_{self.room_type}_indices.npy"
        )
        
        if os.path.exists(indices_path):
            print(f"加载已有索引文件: {indices_path}")
            return np.load(indices_path)
        else:
            print(f"生成新的索引文件: {indices_path}")
            # 获取数据总量
            total_samples = len(self.data["positions"])
            
            # 计算划分
            train_size = int(total_samples * 0.8)
            val_size = int(total_samples * 0.1)
            test_size = total_samples - train_size - val_size
            
            # 生成随机索引
            all_indices = np.random.permutation(total_samples)
            
            if self.split == "train":
                indices = all_indices[:train_size]
            elif self.split == "val":
                indices = all_indices[train_size:train_size+val_size]
            else:  # test
                indices = all_indices[train_size+val_size:]
            
            # 保存索引
            os.makedirs(os.path.dirname(indices_path), exist_ok=True)
            np.save(indices_path, indices)
            
            return indices
        
    def _process_data(self, scene_data):
        """处理场景数据"""
        num_objects = len(scene_data["positions"])
        
        # 如果物体数量超过最大限制,随机采样
        if num_objects > self.max_parts:
            indices = np.random.choice(
                num_objects,
                self.max_parts,
                replace=False
            )
            indices = sorted(indices)  # 保持顺序
            
            # 更新所有属性
            for key in ["positions", "rotations", "sizes", "categories"]:
                scene_data[key] = scene_data[key][indices]
                
            # 更新相对属性
            for key in ["rel_positions", "rel_rotations", "rel_sizes", "rel_categories"]:
                scene_data[key] = scene_data[key][indices][:, indices]
                
            num_objects = self.max_parts
            
        # 如果物体数量少于最大限制,填充
        elif num_objects < self.max_parts:
            pad_size = self.max_parts - num_objects
            
            # 填充绝对属性
            scene_data["positions"] = torch.cat([
                scene_data["positions"],
                torch.zeros(pad_size, 3)
            ])
            scene_data["rotations"] = torch.cat([
                scene_data["rotations"],
                torch.zeros(pad_size, 3)
            ])
            scene_data["sizes"] = torch.cat([
                scene_data["sizes"],
                torch.ones(pad_size, 3)
            ])
            scene_data["categories"] = torch.cat([
                scene_data["categories"],
                torch.zeros(pad_size, dtype=torch.int64)
            ])
            
            # 填充相对属性
            for key in ["rel_positions", "rel_rotations", "rel_sizes", "rel_categories"]:
                shape = scene_data[key].shape
                pad_shape = (self.max_parts, self.max_parts) + shape[2:]
                padded = torch.zeros(pad_shape)
                padded[:num_objects, :num_objects] = scene_data[key]
                scene_data[key] = padded
                
        # 合并绝对属性
        abs_attrs = torch.cat([
            scene_data["positions"],    # [N, 3]
            scene_data["rotations"],    # [N, 3]
            scene_data["sizes"],        # [N, 3]
            scene_data["categories"].unsqueeze(-1)  # [N, 1]
        ], dim=-1)
        
        # 合并相对属性
        rel_attrs = torch.cat([
            scene_data["rel_positions"],    # [N, N, 3]
            scene_data["rel_rotations"],    # [N, N, 3]
            scene_data["rel_sizes"],        # [N, N, 3]
            scene_data["rel_categories"].unsqueeze(-1)  # [N, N, 1]
        ], dim=-1)
        
        return {
            "abs_attrs": abs_attrs,      # [N, 10]
            "rel_attrs": rel_attrs,      # [N, N, 10]
            "num_objects": num_objects   # 实际物体数量
        }

if __name__ == "__main__":
    # 测试数据集
    dataset = RoomDataset(
        data_root="data/processed",
        room_type="bedroom",
        split="train",
        max_parts=80
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # 测试数据加载
    sample = dataset[0]
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")
        else:
            print(f"{key}: {value}")
