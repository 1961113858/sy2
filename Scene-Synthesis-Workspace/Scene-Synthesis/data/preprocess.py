import os
import json
import numpy as np
from tqdm import tqdm
import h5py

class RoomPreprocessor:
    def __init__(self, raw_data_path, output_path):
        """
        房间场景数据预处理器
        Args:
            raw_data_path: 原始数据路径
            output_path: 输出路径
        """
        self.raw_data_path = raw_data_path
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)
        
    def process_all(self):
        """处理所有房间类型"""
        # 处理每种房间类型
        for room_type in ["bedroom", "living"]:
            print(f"Processing {room_type}...")
            self.process_room_type(room_type)
            
    def process_room_type(self, room_type):
        """处理指定类型的房间"""
        # 创建输出目录
        room_output_path = os.path.join(self.output_path, room_type)
        os.makedirs(room_output_path, exist_ok=True)
        
        # 获取所有房间文件
        room_files = self._get_room_files(room_type)
        
        # 处理每个房间
        for file_path in tqdm(room_files):
            room_data = self._process_single_room(file_path)
            if room_data is not None:
                self._save_processed_data(room_data, room_output_path)
                
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
            abs_attrs = []  # 绝对属性
            rel_attrs = []  # 相对属性
            
            for obj in data["furniture"]:
                # 绝对属性: 位置、旋转、尺寸
                pos = np.array(obj["position"])
                rot = np.array(obj["rotation"])
                size = np.array(obj["size"])
                category = obj["category"]
                
                abs_attr = np.concatenate([
                    pos,  # 位置 [3]
                    rot,  # 旋转 [3]
                    size  # 尺寸 [3]
                ])
                abs_attrs.append(abs_attr)
                
                # 计算相对属性
                rel_attr = self._compute_relative_attrs(obj, data["furniture"])
                rel_attrs.append(rel_attr)
                
            # 转换为numpy数组
            abs_attrs = np.stack(abs_attrs)
            rel_attrs = np.stack(rel_attrs)
            
            return {
                "file_id": os.path.splitext(os.path.basename(file_path))[0],
                "abs_attrs": abs_attrs,
                "rel_attrs": rel_attrs
            }
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None
            
    def _compute_relative_attrs(self, obj, all_objects):
        """计算物体间的相对属性"""
        rel_attrs = []
        
        for other in all_objects:
            if other == obj:
                continue
                
            # 相对位置
            rel_pos = np.array(other["position"]) - np.array(obj["position"])
            
            # 相对旋转
            rel_rot = np.array(other["rotation"]) - np.array(obj["rotation"])
            
            # 相对尺寸
            rel_size = np.array(other["size"]) / (np.array(obj["size"]) + 1e-6)
            
            # 类别关系
            rel_cat = float(other["category"] == obj["category"])
            
            # 合并相对属性
            rel_attr = np.concatenate([
                rel_pos,   # 相对位置 [3]
                rel_rot,   # 相对旋转 [3]
                rel_size,  # 相对尺寸 [3]
                [rel_cat]  # 类别关系 [1]
            ])
            rel_attrs.append(rel_attr)
            
        return np.stack(rel_attrs)
        
    def _save_processed_data(self, data, output_path):
        """保存处理后的数据"""
        file_id = data["file_id"]
        
        # 保存绝对属性
        abs_path = os.path.join(output_path, f"{file_id}_abs.npy")
        np.save(abs_path, data["abs_attrs"])
        
        # 保存相对属性
        rel_path = os.path.join(output_path, f"{file_id}_rel.npy")
        np.save(rel_path, data["rel_attrs"])
        
    def create_dataset_splits(self, train_ratio=0.8, val_ratio=0.1):
        """创建数据集划分"""
        for room_type in ["bedroom", "living"]:
            # 获取所有处理后的文件
            room_dir = os.path.join(self.output_path, room_type)
            file_ids = []
            for filename in os.listdir(room_dir):
                if filename.endswith("_abs.npy"):
                    file_id = filename[:-8]  # 移除"_abs.npy"
                    file_ids.append(file_id)
                    
            # 随机打乱
            np.random.shuffle(file_ids)
            
            # 计算划分点
            n = len(file_ids)
            train_idx = int(n * train_ratio)
            val_idx = int(n * (train_ratio + val_ratio))
            
            # 划分数据集
            train_ids = file_ids[:train_idx]
            val_ids = file_ids[train_idx:val_idx]
            test_ids = file_ids[val_idx:]
            
            # 保存划分结果
            splits = {
                "train": train_ids,
                "val": val_ids,
                "test": test_ids
            }
            
            for split, ids in splits.items():
                split_file = os.path.join(
                    self.output_path,
                    f"{split}_{room_type}.txt"
                )
                with open(split_file, "w") as f:
                    for file_id in ids:
                        f.write(f"{file_id}\n")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    
    preprocessor = RoomPreprocessor(args.raw_data_path, args.output_path)
    preprocessor.process_all()
    preprocessor.create_dataset_splits()
