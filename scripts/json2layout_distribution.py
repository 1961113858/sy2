import os
import json
import pickle
import argparse
import numpy as np
from collections import Counter
from loguru import logger

def analyze_layout_distribution(args):
    """分析布局分布并生成类别映射"""
    # 读取model_info.json
    model_info_path = "D:/cxcy2/LEGO-Net-main/sy2/data/3d-front/model_info.json"
    with open(model_info_path, 'r', encoding='utf-8') as f:
        model_info = json.load(f)
    
    # 转换model_info为字典格式
    model_info_dict = {item['model_id']: item for item in model_info}
    
    # 设置房间类型
    if args.type == 'bedroom':
        room_types = ['Bedroom', 'MasterBedroom', 'SecondBedroom']
    elif args.type == 'living':
        room_types = ['LivingDiningRoom', 'LivingRoom']
    
    # 读取3D-FRONT数据
    json_path = "D:/cxcy2/LEGO-Net-main/sy2/data/3d-front/3D-FRONT"
    
    # 统计类别分布
    category_counter = Counter()
    
    # 遍历所有JSON文件
    json_files = [f for f in os.listdir(json_path) if f.endswith('.json')]
    print(f"找到 {len(json_files)} 个JSON文件")
    
    for filename in json_files:
        with open(os.path.join(json_path, filename), 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                # 处理家具信息
                furniture_dict = {}
                if 'furniture' in data:
                    for item in data['furniture']:
                        if 'valid' in item and item['valid']:
                            furniture_dict[item['uid']] = item['jid']
                
                # 处理房间信息
                if 'scene' in data and 'room' in data['scene']:
                    for room in data['scene']['room']:
                        if room['type'] in room_types:
                            # 处理房间中的家具
                            for child in room.get('children', []):
                                ref = child.get('ref')
                                if ref in furniture_dict:
                                    model_id = furniture_dict[ref]
                                    if model_id in model_info_dict:
                                        category = model_info_dict[model_id].get('category')
                                        if category:
                                            category_counter[category] += 1
            
            except json.JSONDecodeError:
                print(f"警告: 无法解析文件 {filename}")
                continue
    
    print(f"\n找到 {len(category_counter)} 个不同的家具类别")
    if category_counter:
        print("前10个最常见的类别:")
        for cat, count in category_counter.most_common(10):
            print(f"  {cat}: {count}")
    
    # 选择top-20类别
    top_categories = [cat for cat, count in category_counter.most_common(20)]
    cat2id = {cat: idx for idx, cat in enumerate(top_categories)}
    
    # 创建assets目录
    os.makedirs("assets", exist_ok=True)
    
    # 保存类别映射
    with open(f"assets/cat2id_{args.type}.pkl", 'wb') as f:
        pickle.dump(cat2id, f)
    
    print(f"为{args.type}生成了 {len(cat2id)} 个类别的映射")
    return cat2id

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, required=True, choices=["bedroom", "living"])
    args = parser.parse_args()
    
    analyze_layout_distribution(args) 