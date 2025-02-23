import os
import numpy as np
import json
import h5py
from tqdm import tqdm
from collections import defaultdict

class DataProcessor:
    def __init__(self, raw_data_path, processed_data_path):
        """
        数据处理器
        Args:
            raw_data_path: 原始数据路径
            processed_data_path: 处理后数据保存路径
        """
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path
        os.makedirs(processed_data_path, exist_ok=True)
        
        # 加载类别映射
        self.category_map = self._load_category_map()
        
    def process_all(self):
        """处理所有数据"""
        # 处理每个房间类型
        for room_type in ["bedroom", "living"]:
            print(f"Processing {room_type} data...")
            self._process_room_type(room_type)
            
    def _process_room_type(self, room_type):
        """处理指定类型的房间数据"""
        # 创建输出目录
        output_dir = os.path.join(self.processed_data_path, room_type)
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取所有房间文件
        room_files = self._get_room_files(room_type)
        
        # 收集所有数据
        all_data = defaultdict(list)
        
        # 处理每个房间
        for file_path in tqdm(room_files):
            data = self._process_single_room(file_path)
            if data is not None:
                for key, value in data.items():
                    all_data[key].append(value)
                    
        # 合并数据
        merged_data = {
            key: np.stack(value) for key, value in all_data.items()
        }
        
        # 保存为h5文件
        output_path = os.path.join(output_dir, f"{room_type}.h5")
        self._save_to_h5(merged_data, output_path)
        
    def _get_room_files(self, room_type):
        """获取指定类型的所有房间文件"""
        room_dir = os.path.join(self.raw_data_path, room_type)
        files = []
        for root, _, filenames in os.walk(room_dir):
            for filename in filenames:
                if filename.endswith(".json"):
                    files.append(os.path.join(root, filename))
        return files
        
    def _process_single_room(self, file_path):
        """处理单个房间数据"""
        try:
            # 加载JSON数据
            with open(file_path, "r") as f:
                data = json.load(f)
                
            # 提取物体属性
            positions = []  # 位置
            rotations = []  # 旋转
            sizes = []     # 尺寸
            categories = [] # 类别
            
            for obj in data["furniture"]:
                # 位置和旋转
                pos = np.array(obj["position"], dtype=np.float32)
                rot = np.array(obj["rotation"], dtype=np.float32)
                
                # 尺寸
                size = np.array(obj["size"], dtype=np.float32)
                
                # 类别ID
                category = self.category_map.get(obj["category"], -1)
                if category == -1:
                    continue
                    
                # 添加属性
                positions.append(pos)
                rotations.append(rot)
                sizes.append(size)
                categories.append(category)
                
            # 检查是否有有效物体
            if not positions:
                return None
                
            # 转换为numpy数组
            positions = np.stack(positions)
            rotations = np.stack(rotations)
            sizes = np.stack(sizes)
            categories = np.array(categories, dtype=np.int64)
            
            # 计算相对属性
            rel_positions = positions[:, None] - positions[None, :]
            rel_rotations = rotations[:, None] - rotations[None, :]
            rel_sizes = sizes[:, None] / (sizes[None, :] + 1e-6)
            
            # 类别关系
            rel_categories = (categories[:, None] == categories[None, :]).astype(np.float32)
            
            return {
                "positions": positions,
                "rotations": rotations,
                "sizes": sizes,
                "categories": categories,
                "rel_positions": rel_positions,
                "rel_rotations": rel_rotations,
                "rel_sizes": rel_sizes,
                "rel_categories": rel_categories
            }
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None
            
    def _load_category_map(self):
        """加载类别映射"""
        # TODO: 从文件加载类别映射
        # 这里使用示例映射
        return {
            "bed": 0,
            "chair": 1,
            "table": 2,
            "sofa": 3,
            "cabinet": 4,
            "desk": 5,
            "lamp": 6,
            "shelf": 7
        }
        
    def _save_to_h5(self, data, output_path):
        """保存数据到h5文件"""
        with h5py.File(output_path, "w") as f:
            for key, value in data.items():
                f.create_dataset(key, data=value)
                
    def create_data_splits(self, train_ratio=0.8, val_ratio=0.1):
        """创建数据集划分"""
        for room_type in ["bedroom", "living"]:
            # 加载数据
            data_path = os.path.join(
                self.processed_data_path,
                room_type,
                f"{room_type}.h5"
            )
            
            with h5py.File(data_path, "r") as f:
                num_samples = f["positions"].shape[0]
                
            # 生成索引
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            
            # 计算划分点
            train_idx = int(num_samples * train_ratio)
            val_idx = int(num_samples * (train_ratio + val_ratio))
            
            # 划分数据集
            splits = {
                "train": indices[:train_idx],
                "val": indices[train_idx:val_idx],
                "test": indices[val_idx:]
            }
            
            # 保存划分结果
            for split_name, split_indices in splits.items():
                split_path = os.path.join(
                    self.processed_data_path,
                    f"{split_name}_{room_type}_indices.npy"
                )
                np.save(split_path, split_indices)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_path", type=str, required=True)
    parser.add_argument("--processed_data_path", type=str, required=True)
    args = parser.parse_args()
    
    processor = DataProcessor(args.raw_data_path, args.processed_data_path)
    processor.process_all()
    processor.create_data_splits()
