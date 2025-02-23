import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.room_layout import RoomLayout
from utils.furniture_loader import FurnitureLoader
from utils.furniture_renderer import FurnitureRenderer
import numpy as np

# 3D-FRONT数据路径
MODEL_INFO_PATH = "D:/Sync2Gen-main_2/ToHongXin/3D-FUTURE-model.json"
MODEL_PATH = "D:/Sync2Gen-main_2/ToHongXin/3D-FUTURE-model"

# 示例家具模型ID（这些ID需要根据实际的3D-FRONT数据集调整）
SAMPLE_FURNITURE = {
    'bed': {
        'model_id': '0a72f3a8-2e89-4ca9-8e9e-3f91f0cc0c32',
        'size': [2.0, 1.5, 0.5]
    },
    'nightstand': {
        'model_id': '0a8d6f94-75d4-4c8c-9637-7a2e6d3c4c16',
        'size': [0.5, 0.5, 0.6]
    },
    'wardrobe': {
        'model_id': '0a9f96f4-7c7a-4b5c-9d0c-8b6c5b6f0f0f',
        'size': [1.2, 0.6, 2.0]
    }
}

def create_sample_bedroom_3d():
    """创建使用3D-FRONT模型的示例卧室"""
    # 创建家具加载器和渲染器
    furniture_loader = FurnitureLoader(MODEL_INFO_PATH, MODEL_PATH)
    renderer = FurnitureRenderer(furniture_loader)
    
    # 创建房间布局
    room = RoomLayout()
    
    # 添加墙体 - 4x4米的房间
    room.add_wall([0, 0], [4, 0])  # 下墙
    room.add_wall([4, 0], [4, 4])  # 右墙
    room.add_wall([4, 4], [0, 4])  # 上墙
    room.add_wall([0, 4], [0, 0])  # 左墙
    
    # 准备家具列表
    furniture_list = [
        {
            'model_id': SAMPLE_FURNITURE['bed']['model_id'],
            'position': [2.0, 1.5, 0.25],
            'rotation': [0, 0, 0],
            'scale': SAMPLE_FURNITURE['bed']['size']
        },
        {
            'model_id': SAMPLE_FURNITURE['nightstand']['model_id'],
            'position': [3.0, 1.5, 0.3],
            'rotation': [0, 0, 0],
            'scale': SAMPLE_FURNITURE['nightstand']['size']
        },
        {
            'model_id': SAMPLE_FURNITURE['wardrobe']['model_id'],
            'position': [0.7, 3.5, 1.0],
            'rotation': [0, 0, 0],
            'scale': SAMPLE_FURNITURE['wardrobe']['size']
        }
    ]
    
    return room, renderer, furniture_list

def main():
    # 创建输出目录
    os.makedirs("results", exist_ok=True)
    
    try:
        # 创建3D-FRONT示例卧室
        bedroom, renderer, furniture_list = create_sample_bedroom_3d()
        
        # 生成2D户型图
        bedroom.plot_2d(save_path="results/bedroom_2d.png")
        print("2D户型图已保存到 results/bedroom_2d.png")
        
        # 生成3D户型图
        bedroom.plot_3d(save_path="results/bedroom_3d.png")
        print("3D户型图已保存到 results/bedroom_3d.png")
        
        # 渲染3D-FRONT模型
        renderer.render_room(
            furniture_list,
            save_path="results/bedroom_3dfront.png"
        )
        print("3D-FRONT渲染结果已保存到 results/bedroom_3dfront.png")
        
    except Exception as e:
        print(f"发生错误: {e}")
        
if __name__ == "__main__":
    main() 