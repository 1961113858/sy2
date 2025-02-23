import sys
import os
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.room_layout import RoomLayout
from utils.furniture_loader import FurnitureLoader
import numpy as np
import pyrender
import matplotlib.pyplot as plt

class Scene3DFront:
    def __init__(self, model_info_path, model_path):
        """
        初始化3D-FRONT场景生成器
        Args:
            model_info_path: 3D-FUTURE模型信息JSON文件路径
            model_path: 3D-FUTURE模型文件目录
        """
        self.furniture_loader = FurnitureLoader(model_info_path, model_path)
        self.room = RoomLayout()
        
    def load_scene(self, scene_json_path):
        """加载3D-FRONT场景"""
        print("开始加载3D-FRONT场景...")
        
        with open(scene_json_path, 'r', encoding='utf-8') as f:
            scene_data = json.load(f)
            
        print("\n场景数据结构:")
        print(f"类型: {type(scene_data)}")
        if isinstance(scene_data, dict):
            print("键:", list(scene_data.keys()))
        
        # 直接处理家具数据
        if 'furniture' in scene_data:
            print("找到家具数据")
            room_data = {
                'furniture': scene_data['furniture'],
                'meshes': scene_data.get('mesh', [])
            }
            self._process_room(room_data)
            return True
            
        print("未找到家具数据")
        return False
        
    def _process_room(self, room_data):
        """处理房间数据"""
        print("处理房间数据...")
        
        # 处理房间边界
        if 'meshes' in room_data:
            self._process_room_mesh(room_data['meshes'])
            
        # 处理家具
        if 'furniture' in room_data:
            for furniture in room_data['furniture']:
                self._process_furniture(furniture)
                
    def _process_room_mesh(self, meshes):
        """处理房间网格数据"""
        print("处理房间网格...")
        
        try:
            for mesh in meshes:
                print(f"\n处理网格: {mesh.get('type', 'unknown')}")
                print(f"网格数据: {mesh}")  # 打印网格数据以便调试
                
                # 检查xyz数据格式
                if 'xyz' not in mesh:
                    print("警告: 网格缺少xyz数据")
                    continue
                    
                xyz_data = mesh['xyz']
                if not isinstance(xyz_data, (list, np.ndarray)):
                    print(f"警告: 无效的xyz数据格式: {type(xyz_data)}")
                    continue
                
                if mesh['type'] == 'Wall':
                    # 确保vertices是二维数组
                    vertices = np.array(xyz_data).reshape(-1, 3)  # 转换为Nx3数组
                    print(f"墙体顶点: {vertices.shape}")
                    
                    # 添加墙体轮廓
                    for i in range(len(vertices)-1):
                        self.room.add_wall(vertices[i][:2], vertices[i+1][:2])
                    # 闭合墙体
                    self.room.add_wall(vertices[-1][:2], vertices[0][:2])
                    
                elif mesh['type'] in ['Door', 'Window']:
                    # 处理门和窗户
                    try:
                        # 尝试获取起点和终点
                        if isinstance(xyz_data, list) and len(xyz_data) >= 2:
                            start = np.array(xyz_data[0][:2])
                            end = np.array(xyz_data[1][:2])
                            
                            if mesh['type'] == 'Door':
                                self.room.add_door(start, end)
                            else:
                                self.room.add_window(start, end)
                        else:
                            print(f"警告: {mesh['type']} 数据格式不正确")
                    except Exception as e:
                        print(f"处理{mesh['type']}时出错: {e}")
                    
        except Exception as e:
            print(f"处理网格时出错: {e}")
            import traceback
            traceback.print_exc()
            
    def _process_furniture(self, furniture):
        """处理家具数据"""
        try:
            model_id = furniture['jid']  # 3D-FUTURE 模型ID
            model_info = self.furniture_loader.get_model_info(model_id)
            
            if model_info:
                # 获取家具信息
                position = furniture.get('pos', [0, 0, 0])  # 默认位置
                rotation = furniture.get('rot', [0, 0, 0])[1] if furniture.get('rot') else 0  # 默认旋转
                size = model_info.get('size', [1, 1, 1])  # 默认尺寸
                category = model_info.get('category', 'unknown')
                
                # 确保数据是numpy数组
                position = np.array(position) if position else np.array([0, 0, 0])
                size = np.array(size) if size else np.array([1, 1, 1])
                
                print(f"添加家具: {category} (ID: {model_id})")
                print(f"位置: {position}")
                print(f"旋转: {rotation}")
                print(f"尺寸: {size}")
                
                self.room.add_furniture(
                    position=position,
                    size=size,
                    rotation=rotation,
                    furniture_type=category,
                    model_id=model_id  # 使用原始模型ID
                )
            else:
                print(f"未找到家具模型信息且无法使用默认模型: {model_id}")
                
        except Exception as e:
            print(f"处理家具时出错: {e}")
            import traceback
            traceback.print_exc()
            
    def render_scene(self, output_dir):
        """渲染场景"""
        try:
            # 生成3D场景图
            print("生成3D场景图...")
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # 绘制房间边界
            room_size = [5, 4, 2]  # 房间尺寸 [长, 宽, 高]
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
            colors = ['purple', 'red', 'orange', 'green', 'blue']  # 不同家具使用不同颜色
            for idx, furniture in enumerate(self.room.furniture):
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
                        zz = np.array([z[0], z[1]])
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
            
            # 保存3D图形
            plt.savefig(os.path.join(output_dir, 'scene_3d.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"渲染完成，结果保存在: {output_dir}")
            
        except Exception as e:
            print(f"渲染时发生错误: {e}")
            import traceback
            traceback.print_exc()

def list_scene_files(base_path):
    """列出所有场景文件"""
    scene_dir = os.path.join(base_path, "3D-FRONT")
    scene_files = []
    for file in os.listdir(scene_dir):
        if file.endswith('.json'):
            scene_files.append(os.path.join(scene_dir, file))
    return scene_files

def main():
    # 3D-FRONT数据路径
    BASE_PATH = "D:/cxcy2/LEGO-Net-main/data/3d-front"
    MODEL_INFO_PATH = f"{BASE_PATH}/3D-FUTURE-model/model_info.json"
    MODEL_PATH = f"{BASE_PATH}/3D-FUTURE-model"
    
    try:
        print(f"正在加载模型信息: {MODEL_INFO_PATH}")
        print(f"模型路径: {MODEL_PATH}")
        
        # 检查必要文件和目录
        if not os.path.exists(MODEL_INFO_PATH):
            print(f"错误: 模型信息文件不存在: {MODEL_INFO_PATH}")
            return
            
        if not os.path.exists(MODEL_PATH):
            print(f"错误: 模型目录不存在: {MODEL_PATH}")
            return
            
        # 获取所有场景文件
        scene_files = list_scene_files(BASE_PATH)
        if not scene_files:
            print(f"错误: 未找到场景文件在 {BASE_PATH}/3D-FRONT/")
            return
            
        print(f"\n找到 {len(scene_files)} 个场景文件:")
        for i, scene_file in enumerate(scene_files):
            print(f"{i+1}. {os.path.basename(scene_file)}")
            
        # 选择第一个场景文件进行处理
        SCENE_PATH = scene_files[0]
        print(f"\n使用场景文件: {SCENE_PATH}")
        
        # 创建输出目录
        output_dir = "results/3dfront_scenes"
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建场景生成器
        scene_generator = Scene3DFront(MODEL_INFO_PATH, MODEL_PATH)
        
        # 加载场景
        if scene_generator.load_scene(SCENE_PATH):
            # 渲染场景
            scene_generator.render_scene(output_dir)
            print(f"\n场景生成完成！结果保存在 {output_dir} 目录")
        else:
            print("场景加载失败")
            
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 