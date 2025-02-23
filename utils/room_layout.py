import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, PathPatch
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d

class RoomLayout:
    def __init__(self):
        self.walls = []  # 墙体列表
        self.doors = []  # 门列表
        self.windows = []  # 窗户列表
        self.furniture = []  # 家具列表
        
    def add_wall(self, start, end):
        """添加墙体"""
        self.walls.append({
            'start': np.array(start),
            'end': np.array(end)
        })
        
    def add_door(self, start, end):
        """添加门"""
        self.doors.append({
            'start': np.array(start),
            'end': np.array(end)
        })
        
    def add_window(self, start, end):
        """添加窗户"""
        self.windows.append({
            'start': np.array(start),
            'end': np.array(end)
        })
        
    def add_furniture(self, position, size, rotation, furniture_type, model_id=None):
        """添加家具"""
        self.furniture.append({
            'type': furniture_type,
            'position': np.array(position),
            'size': np.array(size),
            'rotation': rotation,
            'model_id': model_id
        })
        
    def plot_2d(self, save_path=None):
        """绘制2D户型图"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 绘制墙体
        for wall in self.walls:
            ax.plot([wall['start'][0], wall['end'][0]], 
                    [wall['start'][1], wall['end'][1]], 
                    'black', linewidth=2)
        
        # 绘制门
        for door in self.doors:
            ax.plot([door['start'][0], door['end'][0]], 
                    [door['start'][1], door['end'][1]], 
                    'brown', linewidth=3)
        
        # 绘制窗户
        for window in self.windows:
            ax.plot([window['start'][0], window['end'][0]], 
                    [window['start'][1], window['end'][1]], 
                    'blue', linewidth=2, linestyle='--')
        
        # 绘制家具
        colors = {
            'bed': 'red',
            'nightstand': 'green',
            'wardrobe': 'purple',
            'desk': 'orange',
            'chair': 'blue',
            'sofa': 'brown',
            'table': 'yellow',
            'cabinet': 'gray'
        }
        
        for furniture in self.furniture:
            try:
                # 获取家具信息
                size = furniture['size']
                width, depth = size[0], size[2]
                pos = furniture['position']
                x, z = pos[0], pos[2]
                
                # 创建矩形
                rect = plt.Rectangle(
                    (x - width/2, z - depth/2),
                    width, depth,
                    angle=np.degrees(furniture['rotation'][1]),
                    facecolor=colors.get(furniture['type'], 'gray'),
                    alpha=0.5
                )
                ax.add_patch(rect)
                
                # 添加标签
                ax.text(x, z, furniture['type'], 
                       horizontalalignment='center',
                       verticalalignment='center')
                   
            except Exception as e:
                print(f"绘制家具时出错: {e}")
                continue
        
        # 设置图形属性
        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Z (meters)')
        ax.set_title('Room Layout - Top View')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
            
    def plot_3d(self, save_path=None):
        """绘制3D户型图"""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        wall_height = 2.4  # 墙高2.4米
        
        # 绘制墙体
        for wall in self.walls:
            ax.plot([wall['start'][0], wall['end'][0]], 
                   [wall['start'][1], wall['end'][1]], 
                   [0, 0], 'black', linewidth=2)
            ax.plot([wall['start'][0], wall['end'][0]], 
                   [wall['start'][1], wall['end'][1]], 
                   [wall_height, wall_height], 'black', linewidth=2)
            ax.plot([wall['start'][0], wall['start'][0]], 
                   [wall['start'][1], wall['start'][1]], 
                   [0, wall_height], 'black', linewidth=2)
            ax.plot([wall['end'][0], wall['end'][0]], 
                   [wall['end'][1], wall['end'][1]], 
                   [0, wall_height], 'black', linewidth=2)
            
        # 绘制家具
        colors = {
            'bed': 'red',
            'nightstand': 'green',
            'wardrobe': 'purple',
            'desk': 'orange'
        }
        
        furniture_heights = {
            'bed': 0.5,
            'nightstand': 0.6,
            'wardrobe': 2.0,
            'desk': 0.75
        }
        
        for furniture in self.furniture:
            width, depth = furniture['size']
            height = furniture_heights.get(furniture['type'], 1.0)
            x, y = furniture['position']
            
            # 创建3D立方体的顶点
            vertices = np.array([
                [x-width/2, y-depth/2, 0],
                [x+width/2, y-depth/2, 0],
                [x+width/2, y+depth/2, 0],
                [x-width/2, y+depth/2, 0],
                [x-width/2, y-depth/2, height],
                [x+width/2, y-depth/2, height],
                [x+width/2, y+depth/2, height],
                [x-width/2, y+depth/2, height]
            ])
            
            # 定义立方体的面
            faces = [
                [vertices[0], vertices[1], vertices[2], vertices[3]],
                [vertices[4], vertices[5], vertices[6], vertices[7]],
                [vertices[0], vertices[1], vertices[5], vertices[4]],
                [vertices[2], vertices[3], vertices[7], vertices[6]],
                [vertices[1], vertices[2], vertices[6], vertices[5]],
                [vertices[3], vertices[0], vertices[4], vertices[7]]
            ]
            
            # 绘制立方体的每个面
            for face in faces:
                poly = art3d.Poly3DCollection([face])
                color = colors.get(furniture['type'], 'gray')
                poly.set_facecolor(color)
                poly.set_alpha(0.5)
                ax.add_collection3d(poly)
        
        # 设置图形属性
        ax.set_box_aspect([1, 1, 0.5])
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_zlabel('Z (meters)')
        ax.set_title('Room Layout - 3D View')
        
        # 设置视角
        ax.view_init(elev=30, azim=45)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
            
    def get_furniture_list(self):
        """获取家具列表"""
        furniture_list = []
        for furniture in self.furniture:
            if 'model_id' in furniture:  # 确保有模型ID
                furniture_list.append({
                    'model_id': furniture['model_id'],
                    'position': furniture['position'],
                    'rotation': furniture['rotation'],
                    'size': furniture['size'],
                    'type': furniture['type']
                })
        return furniture_list
            