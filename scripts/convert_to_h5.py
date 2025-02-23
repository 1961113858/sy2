import os
import h5py
import numpy as np
import pickle
import argparse
from pathlib import Path

def pad_data(data, max_objects, feature_dim):
    """填充数据到指定大小"""
    num_objects = len(data)
    if num_objects > max_objects:
        return data[:max_objects]
    elif num_objects < max_objects:
        pad_shape = (max_objects - num_objects, feature_dim)
        if feature_dim == 1:
            pad_data = np.zeros(max_objects - num_objects, dtype=data.dtype)
        else:
            pad_data = np.zeros(pad_shape, dtype=data.dtype)
        return np.concatenate([data, pad_data], axis=0)
    return data

def pad_rel_data(data, max_objects):
    """填充相对数据到指定大小"""
    num_objects = len(data)
    if num_objects > max_objects:
        return data[:max_objects, :max_objects]
    elif num_objects < max_objects:
        if len(data.shape) == 2:  # 对于2D数据（如rel_categories）
            pad_shape = (max_objects, max_objects)
        else:  # 对于3D数据（如rel_positions）
            pad_shape = (max_objects, max_objects, data.shape[-1])
        padded = np.zeros(pad_shape, dtype=data.dtype)
        padded[:num_objects, :num_objects] = data
        return padded
    return data

def convert_to_h5(args):
    """将npy和pkl数据转换为h5格式"""
    print("开始数据转换...")
    
    # 创建输出目录
    output_dir = Path(f"data/processed/{args.type}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {output_dir}")
    
    # 读取训练和验证集索引
    train_indices = []
    val_indices = []
    
    train_file = Path(f"data/train_{args.type}.txt")
    val_file = Path(f"data/val_{args.type}.txt")
    
    print(f"读取训练集索引文件: {train_file}")
    with open(train_file, 'r') as f:
        train_indices = [int(line.strip()) for line in f.readlines()]
    print(f"找到 {len(train_indices)} 个训练样本")
    
    print(f"读取验证集索引文件: {val_file}")
    with open(val_file, 'r') as f:
        val_indices = [int(line.strip()) for line in f.readlines()]
    print(f"找到 {len(val_indices)} 个验证样本")
    
    # 创建H5文件
    h5_path = output_dir / f"{args.type}.h5"
    print(f"创建H5文件: {h5_path}")
    
    # 设置最大物体数量
    MAX_OBJECTS = 80
    
    # 读取所有数据
    all_positions = []
    all_rotations = []
    all_sizes = []
    all_categories = []
    all_rel_positions = []
    all_rel_rotations = []
    all_rel_sizes = []
    all_rel_categories = []
    
    # 处理所有场景
    all_indices = train_indices + val_indices
    print("\n处理所有场景...")
    
    for i, idx in enumerate(all_indices):
        if i % 100 == 0:
            print(f"处理场景: {i}/{len(all_indices)}")
            
        # 读取abs数据
        abs_path = Path(f"data/{args.type}/{idx}_abs.npy")
        if not abs_path.exists():
            print(f"警告: 找不到文件 {abs_path}")
            continue
        abs_data = np.load(abs_path, allow_pickle=True)
        
        # 读取rel数据
        rel_path = Path(f"data/{args.type}/{idx}_rel.pkl")
        if not rel_path.exists():
            print(f"警告: 找不到文件 {rel_path}")
            continue
        with open(rel_path, 'rb') as rel_f:
            rel_data = pickle.load(rel_f)
        
        # 提取绝对属性
        positions = []
        rotations = []
        sizes = []
        categories = []
        for item in abs_data:
            positions.append(item['position'])
            rotations.append(item['rotation'])
            sizes.append(item['scale'])
            categories.append(0)  # 暂时用0作为类别
        
        # 转换为numpy数组
        positions = np.array(positions, dtype=np.float32)
        rotations = np.array(rotations, dtype=np.float32)
        sizes = np.array(sizes, dtype=np.float32)
        categories = np.array(categories, dtype=np.int64)
        
        # 填充到固定大小
        positions = pad_data(positions, MAX_OBJECTS, 3)
        rotations = pad_data(rotations, MAX_OBJECTS, 3)
        sizes = pad_data(sizes, MAX_OBJECTS, 3)
        categories = pad_data(categories, MAX_OBJECTS, 1)
        
        # 提取相对属性
        num_objects = min(len(abs_data), MAX_OBJECTS)
        rel_positions = np.zeros((num_objects, num_objects, 3), dtype=np.float32)
        rel_rotations = np.zeros((num_objects, num_objects, 3), dtype=np.float32)
        rel_sizes = np.ones((num_objects, num_objects, 3), dtype=np.float32)
        rel_categories = np.zeros((num_objects, num_objects), dtype=np.int64)  # 注意这里是2D的
        
        # 填充相对属性
        for i in range(num_objects):
            for j in range(num_objects):
                if i != j:
                    rel_positions[i, j] = positions[i] - positions[j]
                    rel_rotations[i, j] = rotations[i] - rotations[j]
        
        # 填充相对属性到固定大小
        rel_positions = pad_rel_data(rel_positions, MAX_OBJECTS)
        rel_rotations = pad_rel_data(rel_rotations, MAX_OBJECTS)
        rel_sizes = pad_rel_data(rel_sizes, MAX_OBJECTS)
        rel_categories = pad_rel_data(rel_categories, MAX_OBJECTS)  # 现在pad_rel_data可以处理2D数据
        
        # 保存数据
        all_positions.append(positions)
        all_rotations.append(rotations)
        all_sizes.append(sizes)
        all_categories.append(categories)
        all_rel_positions.append(rel_positions)
        all_rel_rotations.append(rel_rotations)
        all_rel_sizes.append(rel_sizes)
        all_rel_categories.append(rel_categories)
    
    # 转换为numpy数组
    all_positions = np.stack(all_positions, axis=0)
    all_rotations = np.stack(all_rotations, axis=0)
    all_sizes = np.stack(all_sizes, axis=0)
    all_categories = np.stack(all_categories, axis=0)
    all_rel_positions = np.stack(all_rel_positions, axis=0)
    all_rel_rotations = np.stack(all_rel_rotations, axis=0)
    all_rel_sizes = np.stack(all_rel_sizes, axis=0)
    all_rel_categories = np.stack(all_rel_categories, axis=0)
    
    print("\n数据形状:")
    print(f"positions: {all_positions.shape}")
    print(f"rotations: {all_rotations.shape}")
    print(f"sizes: {all_sizes.shape}")
    print(f"categories: {all_categories.shape}")
    print(f"rel_positions: {all_rel_positions.shape}")
    print(f"rel_rotations: {all_rel_rotations.shape}")
    print(f"rel_sizes: {all_rel_sizes.shape}")
    print(f"rel_categories: {all_rel_categories.shape}")
    
    # 保存到H5文件
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('positions', data=all_positions)
        f.create_dataset('rotations', data=all_rotations)
        f.create_dataset('sizes', data=all_sizes)
        f.create_dataset('categories', data=all_categories)
        f.create_dataset('rel_positions', data=all_rel_positions)
        f.create_dataset('rel_rotations', data=all_rel_rotations)
        f.create_dataset('rel_sizes', data=all_rel_sizes)
        f.create_dataset('rel_categories', data=all_rel_categories)
    
    # 保存数据集划分索引
    train_indices = np.array(list(range(len(train_indices))))
    val_indices = np.array(list(range(len(train_indices), len(all_indices))))
    
    np.save(output_dir.parent / f"train_{args.type}_indices.npy", train_indices)
    np.save(output_dir.parent / f"val_{args.type}_indices.npy", val_indices)
    
    print(f"\n数据转换完成！")
    print(f"H5文件保存在: {h5_path}")
    print(f"训练集索引保存在: {output_dir.parent}/train_{args.type}_indices.npy")
    print(f"验证集索引保存在: {output_dir.parent}/val_{args.type}_indices.npy")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, required=True, choices=["bedroom", "living"])
    args = parser.parse_args()
    
    convert_to_h5(args) 