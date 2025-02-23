import os
import json
import numpy as np
import pickle
from tqdm import tqdm

def preprocess_3dfront(args):
    """预处理3D-FRONT数据集"""
    # 1. 加载模型信息
    print("加载模型信息...")
    model_info_dict = modelInfo2dict(args.model_info_path)
    print(f"找到 {len(model_info_dict)} 个模型")
    
    # 2. 设置房间类型
    if args.type == 'bedroom':
        room_types = ['Bedroom', 'MasterBedroom', 'SecondBedroom']
    elif args.type == 'living':
        room_types = ['LivingDiningRoom', 'LivingRoom']
    
    # 3. 初始化房间数据字典
    layout_room_dict = {k: [] for k in room_types}
    
    # 4. 加载类别映射
    with open(f'assets/cat2id_{args.type}.pkl', 'rb') as f:
        cat2id_dict = pickle.load(f)
    
    # 5. 处理每个场景文件
    files = os.listdir(args.json_path)
    print(f"开始处理 {len(files)} 个场景文件...")
    
    for n_m, m in enumerate(tqdm(files)):
        if not m.endswith('.json'):
            continue
            
        with open(os.path.join(args.json_path, m), 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 收集家具信息
        model_jid = []
        model_uid = []
        model_bbox = []
        
        for ff in data.get('furniture', []):
            if 'valid' in ff and ff['valid']:
                model_uid.append(ff['uid'])
                model_jid.append(ff['jid'])
                model_bbox.append(ff['bbox'])
        
        # 处理每个房间
        if 'scene' in data and 'room' in data['scene']:
            for room in data['scene']['room']:
                if room['type'] not in room_types:
                    continue
                
                # 处理房间中的家具
                furniture_list = []
                for child in room.get('children', []):
                    ref = child.get('ref')
                    if ref not in model_uid:
                        continue
                        
                    idx = model_uid.index(ref)
                    model_id = model_jid[idx]
                    
                    if model_id not in model_info_dict:
                        continue
                        
                    category = model_info_dict[model_id].get('category')
                    if category not in cat2id_dict:
                        continue
                        
                    furniture_list.append({
                        'model_id': model_id,
                        'category': category,
                        'category_id': cat2id_dict[category],
                        'position': child.get('pos', [0, 0, 0]),
                        'rotation': child.get('rot', [0, 0, 0]),
                        'scale': child.get('scale', [1, 1, 1])
                    })
                
                # 应用过滤条件
                if room['type'].endswith('bedroom'):
                    if not any(item['category'] == 'bed' for item in furniture_list):
                        continue
                elif room['type'].endswith('livingroom'):
                    if len(furniture_list) < 6:
                        continue
                
                if furniture_list:
                    layout_room_dict[room['type']].append({
                        'room_type': room['type'],
                        'furniture': furniture_list,
                        'layout': room.get('layout', {}),
                        'size': room.get('size', [0, 0, 0])
                    })
    
    # 6. 保存处理后的数据
    os.makedirs("outputs", exist_ok=True)
    for room_type, rooms in layout_room_dict.items():
        if rooms:
            output_file = os.path.join("outputs", f"{room_type}.npy")
            np.save(output_file, rooms)
            print(f"保存 {room_type} 数据: {len(rooms)} 个场景")

def modelInfo2dict(model_info_path):
    """将模型信息转换为字典格式"""
    model_info_dict = {}
    with open(model_info_path, 'r') as f:
        info = json.load(f)
    for v in info:
        model_info_dict[v['model_id']] = v
    return model_info_dict

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, required=True, choices=["bedroom", "living"])
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--future_path", type=str, required=True)
    parser.add_argument("--model_info_path", type=str, required=True)
    args = parser.parse_args()
    
    preprocess_3dfront(args) 