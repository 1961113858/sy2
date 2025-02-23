import os
import numpy as np
import pickle
import argparse

def generate_dataset(args):
    """生成最终的abs和rel数据集"""
    # 读取合并后的数据
    if args.type == "bedroom":
        data_path = "D:/cxcy2/LEGO-Net-main/sy2/outputs/Bedroom_train_val.npy"
    else:
        data_path = "D:/cxcy2/LEGO-Net-main/sy2/outputs/Livingroom_train_val.npy"
    all_data = np.load(data_path, allow_pickle=True)
    
    # 创建输出目录
    output_dir = f"data/{args.type}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理每个场景
    for i, scene in enumerate(all_data):
        # 绝对坐标数据
        abs_data = process_abs_data(scene)
        np.save(f"{output_dir}/{i}_abs.npy", abs_data)
        
        # 相对坐标数据
        rel_data = process_rel_data(scene)
        with open(f"{output_dir}/{i}_rel.pkl", 'wb') as f:
            pickle.dump(rel_data, f)
            
    # 生成训练集和验证集索引文件
    generate_split_files(args.type, len(all_data))
    
def process_abs_data(scene):
    """处理绝对坐标数据"""
    # 获取有效物体的索引
    valid_abs_index = np.where(scene[:, -1])[0]
    
    abs_data = []
    for idx in valid_abs_index:
        # 从scene数组中提取数据
        # scene格式: [rotation(3), position(3), size(3), indicator(1)]
        abs_data.append({
            "rotation": scene[idx, 0:3].tolist(),
            "position": scene[idx, 3:6].tolist(),
            "scale": scene[idx, 6:9].tolist()
        })
    return abs_data

def process_rel_data(scene):
    """处理相对坐标数据"""
    # 获取有效物体的索引
    valid_abs_index = np.where(scene[:, -1])[0]
    
    rel_data = []
    # 计算相对位置等信息
    for i, idx in enumerate(valid_abs_index):
        rel_pos = []
        rel_rot = []
        # 计算与其他物体的相对位置和旋转
        for other_idx in valid_abs_index:
            if idx != other_idx:
                pos_diff = scene[idx, 3:6] - scene[other_idx, 3:6]
                rot_diff = scene[idx, 0:3] - scene[other_idx, 0:3]
                rel_pos.append(pos_diff.tolist())
                rel_rot.append(rot_diff.tolist())
        
        rel_data.append({
            "relative_positions": rel_pos,
            "relative_rotations": rel_rot
        })
    
    return rel_data

def generate_split_files(room_type, total_size):
    """生成数据集划分文件"""
    train_size = min(4000, total_size)
    indices = np.arange(total_size)
    
    # 保存训练集索引
    with open(f"data/train_{room_type}.txt", 'w') as f:
        for idx in indices[:train_size]:
            f.write(f"{idx}\n")
            
    # 保存验证集索引
    with open(f"data/val_{room_type}.txt", 'w') as f:
        for idx in indices[train_size:]:
            f.write(f"{idx}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, required=True, choices=["bedroom", "living"])
    args = parser.parse_args()
    
    generate_dataset(args) 