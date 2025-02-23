import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 添加项目根目录到路径
import json
import numpy as np
from examples.room_layout_demo import Scene3DFront
from utils.furniture_loader import FurnitureLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def create_test_scene():
    """创建测试场景"""
    # 创建一个简单的卧室场景
    furniture_list = [
        {
            'position': np.array([2.5, 2.0, 0.3]),  # 床的位置
            'size': np.array([2.0, 1.6, 0.6]),      # 床的尺寸
            'rotation': 0,                           # 床的旋转角度
            'category': 'bed'
        },
        {
            'position': np.array([0.8, 0.8, 0.4]),  # 床头柜的位置
            'size': np.array([0.5, 0.5, 0.8]),      # 床头柜的尺寸
            'rotation': 0,
            'category': 'nightstand'
        },
        {
            'position': np.array([4.0, 1.0, 1.0]),  # 衣柜的位置
            'size': np.array([0.8, 0.6, 2.0]),      # 衣柜的尺寸
            'rotation': 0,
            'category': 'wardrobe'
        },
        {
            'position': np.array([1.5, 3.0, 0.3]),  # 书桌的位置
            'size': np.array([1.2, 0.6, 0.75]),     # 书桌的尺寸
            'rotation': 0,
            'category': 'desk'
        }
    ]
    return furniture_list

def render_test_scene(furniture_list, output_dir="results/test"):
    """渲染测试场景"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建3D场景图
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制房间边界
    room_size = [5, 4, 2.4]  # 房间尺寸 [长, 宽, 高]
    
    # 绘制房间框架
    for i in [0, room_size[0]]:
        for j in [0, room_size[1]]:
            ax.plot([i, i], [j, j], [0, room_size[2]], 'k-', linewidth=1)
    for i in [0, room_size[0]]:
        for k in [0, room_size[2]]:
            ax.plot([i, i], [0, room_size[1]], [k, k], 'k-', linewidth=1)
    for j in [0, room_size[1]]:
        for k in [0, room_size[2]]:
            ax.plot([0, room_size[0]], [j, j], [k, k], 'k-', linewidth=1)
    
    # 绘制家具
    colors = ['purple', 'red', 'orange', 'green', 'blue']
    for idx, furniture in enumerate(furniture_list):
        pos = furniture['position']
        size = furniture['size']
        color = colors[idx % len(colors)]
        
        # 创建立方体的六个面
        x = [pos[0] - size[0]/2, pos[0] + size[0]/2]
        y = [pos[1] - size[1]/2, pos[1] + size[1]/2]
        z = [pos[2], pos[2] + size[2]]
        
        # 创建每个面的顶点
        xx, yy = np.meshgrid([x[0], x[1]], [y[0], y[1]])
        
        # 底面
        ax.plot_surface(xx, yy, np.full_like(xx, z[0]), color=color, alpha=0.7)
        # 顶面
        ax.plot_surface(xx, yy, np.full_like(xx, z[1]), color=color, alpha=0.7)
        
        # 四个侧面
        for i in range(2):
            for j in range(2):
                if i == 0:
                    ax.plot_surface(
                        np.full_like(np.array([[z[0]], [z[1]]]), x[j]),
                        np.array([[y[0], y[1]], [y[0], y[1]]]),
                        np.array([[z[0], z[0]], [z[1], z[1]]]),
                        color=color, alpha=0.7
                    )
                if j == 0:
                    ax.plot_surface(
                        np.array([[x[0], x[1]], [x[0], x[1]]]),
                        np.full_like(np.array([[z[0]], [z[1]]]), y[i]),
                        np.array([[z[0], z[0]], [z[1], z[1]]]),
                        color=color, alpha=0.7
                    )
    
    # 设置图形属性
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.set_title('Room Layout - 3D View')
    
    # 设置坐标轴范围
    ax.set_xlim([0, room_size[0]])
    ax.set_ylim([0, room_size[1]])
    ax.set_zlim([0, room_size[2]])
    
    # 设置网格
    ax.grid(True)
    
    # 设置视角
    ax.view_init(elev=20, azim=45)
    
    # 保存图形
    plt.savefig(os.path.join(output_dir, 'test_scene_3d.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"测试场景渲染完成，结果保存在: {output_dir}")

def main():
    """主函数"""
    try:
        # 创建测试场景
        furniture_list = create_test_scene()
        
        # 渲染场景
        render_test_scene(furniture_list)
        
    except Exception as e:
        print(f"测试时发生错误: {e}")
        import traceback
        traceback.print_exc()

def test_model_categories():
    # 获取当前文件的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 3D-FRONT数据路径
    BASE_PATH = os.path.normpath("D:/cxcy2/LEGO-Net-main/data/3d-front")
    MODEL_INFO_PATH = os.path.join(BASE_PATH, "3D-FUTURE-model", "model_info.json")
    MODEL_PATH = os.path.join(BASE_PATH, "3D-FUTURE-model")
    
    print(f"当前目录: {current_dir}")
    print(f"模型信息文件: {MODEL_INFO_PATH}")
    print(f"模型目录: {MODEL_PATH}")
    
    # 检查文件是否存在
    if not os.path.exists(MODEL_INFO_PATH):
        print(f"错误: 找不到模型信息文件: {MODEL_INFO_PATH}")
        return
        
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 找不到模型目录: {MODEL_PATH}")
        return
    
    # 创建加载器
    loader = FurnitureLoader(MODEL_INFO_PATH, MODEL_PATH)
    
    # 测试一些模型ID
    test_ids = [
        '161c86f3-1eaa-40e3-b412-900bc71d85ee',
        'db84c83f-6de8-4a36-8d88-af157842becc',
        'a3017175-01da-4bbb-a3f4-aa896e3fa604'
    ]
    
    for model_id in test_ids:
        info = loader.get_model_info(model_id)
        if info:
            print(f"\n模型 {model_id}:")
            print(f"类别: {info.get('category', 'unknown')}")
            print(f"超类别: {info.get('super-category', 'unknown')}")
            print(f"风格: {info.get('style', 'unknown')}")
            print(f"材质: {info.get('material', 'unknown')}")

def test_model_rendering():
    """测试模型渲染"""
    # 设置路径
    BASE_PATH = os.path.normpath("D:/cxcy2/LEGO-Net-main/data/3d-front")
    MODEL_INFO_PATH = os.path.join(BASE_PATH, "3D-FUTURE-model", "model_info.json")
    MODEL_PATH = os.path.join(BASE_PATH, "3D-FUTURE-model")
    
    # 创建输出目录
    output_dir = "results/model_test"
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建加载器
    loader = FurnitureLoader(MODEL_INFO_PATH, MODEL_PATH)
    
    # 尝试渲染每个超类别的一个模型
    for super_cat, data in loader.available_models.items():
        if data['_all']:
            model_id = data['_all'][0]
            model_info = loader.get_model_info(model_id)
            if not model_info:
                continue
                
            print(f"\n尝试渲染模型: {model_id}")
            try:
                mesh = loader.get_model_mesh(model_id)
                if mesh:
                    # 使用固定的文件名
                    save_path = os.path.join(output_dir, "image.png")
                    
                    # 渲染并保存
                    if loader.render_model(mesh, save_path):
                        print(f"成功渲染: image.png")
                    else:
                        print(f"渲染失败: image.png")
            except Exception as e:
                print(f"渲染失败: {e}")
                import traceback
                traceback.print_exc()

def test_scene_rendering():
    """测试场景渲染"""
    # 设置路径
    BASE_PATH = os.path.normpath("D:/cxcy2/LEGO-Net-main/data/3d-front")
    MODEL_INFO_PATH = os.path.join(BASE_PATH, "3D-FUTURE-model", "model_info.json")
    MODEL_PATH = os.path.join(BASE_PATH, "3D-FUTURE-model")
    SCENE_PATH = os.path.join(BASE_PATH, "3D-FRONT", "0398b508-cc10-4567-a231-b127c905105a.json")
    
    # 创建输出目录
    output_dir = "results/scene_test"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 创建场景生成器
        print("创建场景生成器...")
        scene_generator = Scene3DFront(MODEL_INFO_PATH, MODEL_PATH)
        
        # 加载场景
        print("\n加载场景...")
        scene_generator.load_scene(SCENE_PATH)
        
        # 添加家具到场景
        print("\n添加家具...")
        furniture_list = [
            {
                'jid': '0a42986e-556c-4afa-9973-86f93de5fa76',  # 柜子
                'pos': [2.0, 0.0, 1.0],  # x, y, z 位置
                'rot': [0, 0, 0]  # 旋转角度
            },
            {
                'jid': 'another_furniture_id',  # 其他家具
                'pos': [3.0, 0.0, 2.0],
                'rot': [0, 90, 0]
            }
            # 可以添加更多家具
        ]
        
        for furniture in furniture_list:
            scene_generator.add_furniture(
                furniture['jid'],
                furniture['pos'],
                furniture['rot']
            )
        
        # 渲染场景
        print("\n渲染场景...")
        scene_generator.render_scene(output_dir)
        
        print(f"\n完成! 结果保存在: {output_dir}/image.png")
        
    except Exception as e:
        print(f"渲染场景失败: {e}")
        import traceback
        traceback.print_exc()

def check_model_info():
    """检查模型信息文件"""
    BASE_PATH = "D:/cxcy2/LEGO-Net-main/data/3d-front"
    MODEL_INFO_PATH = f"{BASE_PATH}/model_info.json"
    
    with open(MODEL_INFO_PATH, 'r', encoding='utf-8') as f:
        model_info = json.load(f)
        
    print(f"模型信息总数: {len(model_info)}")
    print("\n示例模型信息:")
    for item in list(model_info)[:3]:
        print(json.dumps(item, indent=2))
        
    # 检查特定ID
    test_id = "161c86f3-1eaa-40e3-b412-900bc71d85ee"
    if test_id in [item.get('model_id') for item in model_info]:
        print(f"\n找到模型 {test_id}")
    else:
        print(f"\n未找到模型 {test_id}")

if __name__ == "__main__":
    main() 