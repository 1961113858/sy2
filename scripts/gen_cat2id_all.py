import os
import json
import pickle

def generate_cat2id_all():
    """生成所有类别的映射"""
    # 读取model_info.json
    model_info_path = "D:/cxcy2/LEGO-Net-main/sy2/data/3d-front/model_info.json"
    with open(model_info_path, 'r', encoding='utf-8') as f:
        model_info = json.load(f)
    
    # 收集所有类别
    categories = set()
    for model in model_info:
        if "category" in model:
            categories.add(model["category"])
    
    # 生成类别到ID的映射
    cat2id = {cat: idx for idx, cat in enumerate(sorted(categories))}
    
    # 创建assets目录
    os.makedirs("assets", exist_ok=True)
    
    # 保存映射
    with open("assets/cat2id_all.pkl", 'wb') as f:
        pickle.dump(cat2id, f)
    
    print(f"生成了 {len(cat2id)} 个类别的映射")
    return cat2id

if __name__ == "__main__":
    generate_cat2id_all() 