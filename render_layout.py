import os
import numpy as np
import matplotlib.pyplot as plt
from utils.room_layout import RoomLayout

# 家具类型映射
FURNITURE_TYPES = {
    0: "床",
    1: "床头柜",
    2: "衣柜",
    3: "书桌",
    4: "椅子"
}

def create_room_layout(scene_data, room_size=(4, 4)):
    """
    从场景数据创建房间布局
    Args:
        scene_data: 场景数据 [N, 10] (位置3 + 旋转3 + 尺寸3 + 类别1)
        room_size: 房间尺寸 (宽, 长)
    """
    # 创建房间布局实例
    room = RoomLayout()
    
    # 添加墙体
    width, length = room_size
    room.add_wall([0, 0], [width, 0])  # 下墙
    room.add_wall([width, 0], [width, length])  # 右墙
    room.add_wall([width, length], [0, length])  # 上墙
    room.add_wall([0, length], [0, 0])  # 左墙
    
    # 添加家具
    valid_mask = np.any(scene_data != 0, axis=1)
    valid_furniture = scene_data[valid_mask]
    
    for furniture in valid_furniture:
        position = furniture[3:6]    # 位置
        rotation = furniture[0:3]    # 旋转
        size = furniture[6:9]        # 尺寸
        category = int(furniture[9]) # 类别
        
        # 确保位置在房间范围内
        position = np.clip(position, [0, 0, 0], [width, length, 3])
        
        room.add_furniture(position, rotation, size, category)
    
    return room

def main():
    # 创建输出目录
    os.makedirs("results", exist_ok=True)
    
    try:
        # 加载卧室数据
        scenes = np.load("outputs/Bedroom_train_val.npy", allow_pickle=True)
        print(f"加载了 {len(scenes)} 个场景")
        
        # 选择前5个场景进行渲染
        for i in range(min(5, len(scenes))):
            print(f"\n处理场景 {i+1}")
            scene_data = scenes[i]
            
            # 创建房间布局
            room = create_room_layout(scene_data)
            
            # 生成2D户型图
            room.plot_2d(save_path=f"results/bedroom_{i+1}_2d.png")
            print(f"2D户型图已保存到 results/bedroom_{i+1}_2d.png")
            
            # 生成3D户型图
            room.plot_3d(save_path=f"results/bedroom_{i+1}_3d.png")
            print(f"3D户型图已保存到 results/bedroom_{i+1}_3d.png")
            
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 