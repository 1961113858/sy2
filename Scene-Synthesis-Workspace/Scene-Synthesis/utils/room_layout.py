import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import torch

class RoomLayout:
    def __init__(self):
        self.walls = []  # 存储墙体信息
        self.furniture = []  # 存储家具信息
        
    def add_wall(self, start_point, end_point, height=2.8):
        """添加墙体"""
        self.walls.append({
            'start': np.array(start_point),
            'end': np.array(end_point),
            'height': height
        })
        
    def add_furniture(self, position, rotation, size, category):
        """添加家具"""
        self.furniture.append({
            'position': np.array(position),
            'rotation': np.array(rotation),
            'size': np.array(size),
            'category': category
        })
        
    def plot_2d(self, save_path=None):
        """绘制2D户型图"""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # 绘制墙体
        for wall in self.walls:
            ax.plot([wall['start'][0], wall['end'][0]], 
                   [wall['start'][1], wall['end'][1]], 
                   'k-', linewidth=2)
            
        # 绘制家具
        for furn in self.furniture:
            pos = furn['position'][:2]  # 只取x,y坐标
            size = furn['size'][:2]
            rot = furn['rotation'][2]  # 只取z轴旋转
            
            # 创建矩形顶点
            corners = np.array([
                [-size[0]/2, -size[1]/2],
                [size[0]/2, -size[1]/2],
                [size[0]/2, size[1]/2],
                [-size[0]/2, size[1]/2]
            ])
            
            # 应用旋转
            c, s = np.cos(rot), np.sin(rot)
            R = np.array([[c, -s], [s, c]])
            corners = corners @ R.T
            
            # 应用平移
            corners = corners + pos
            
            # 绘制家具
            polygon = plt.Polygon(corners, facecolor='lightblue', 
                                edgecolor='blue', alpha=0.5)
            ax.add_patch(polygon)
            
        # 设置图形属性
        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('房间平面图')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
            
    def plot_3d(self, save_path=None):
        """绘制3D户型图"""
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制墙体
        for wall in self.walls:
            start = wall['start']
            end = wall['end']
            height = wall['height']
            
            # 创建墙体顶点
            vertices = np.array([
                [start[0], start[1], 0],
                [end[0], end[1], 0],
                [end[0], end[1], height],
                [start[0], start[1], height]
            ])
            
            # 创建面
            faces = [[vertices[i] for i in [0,1,2,3]]]
            
            # 绘制墙体
            wall_poly = Poly3DCollection(faces, alpha=0.25)
            wall_poly.set_facecolor('lightgray')
            ax.add_collection3d(wall_poly)
            
        # 绘制家具
        for furn in self.furniture:
            pos = furn['position']
            size = furn['size']
            rot = furn['rotation'][2]  # 只使用z轴旋转
            
            # 创建立方体顶点
            vertices = np.array([
                [-size[0]/2, -size[1]/2, 0],
                [size[0]/2, -size[1]/2, 0],
                [size[0]/2, size[1]/2, 0],
                [-size[0]/2, size[1]/2, 0],
                [-size[0]/2, -size[1]/2, size[2]],
                [size[0]/2, -size[1]/2, size[2]],
                [size[0]/2, size[1]/2, size[2]],
                [-size[0]/2, size[1]/2, size[2]]
            ])
            
            # 应用旋转
            c, s = np.cos(rot), np.sin(rot)
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            vertices = vertices @ R.T
            
            # 应用平移
            vertices = vertices + pos
            
            # 定义面
            faces = [
                [vertices[i] for i in [0,1,2,3]],  # 底面
                [vertices[i] for i in [4,5,6,7]],  # 顶面
                [vertices[i] for i in [0,1,5,4]],  # 前面
                [vertices[i] for i in [2,3,7,6]],  # 后面
                [vertices[i] for i in [0,3,7,4]],  # 左面
                [vertices[i] for i in [1,2,6,5]]   # 右面
            ]
            
            # 绘制家具
            furniture_poly = Poly3DCollection(faces, alpha=0.5)
            furniture_poly.set_facecolor('lightblue')
            ax.add_collection3d(furniture_poly)
            
        # 设置图形属性
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('房间3D视图')
        
        # 设置视角
        ax.view_init(elev=30, azim=45)
        
        # 设置坐标轴范围
        x_min = min(wall['start'][0] for wall in self.walls)
        x_max = max(wall['end'][0] for wall in self.walls)
        y_min = min(wall['start'][1] for wall in self.walls)
        y_max = max(wall['end'][1] for wall in self.walls)
        z_max = max(wall['height'] for wall in self.walls)
        
        ax.set_xlim([x_min-1, x_max+1])
        ax.set_ylim([y_min-1, y_max+1])
        ax.set_zlim([0, z_max+1])
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show() 