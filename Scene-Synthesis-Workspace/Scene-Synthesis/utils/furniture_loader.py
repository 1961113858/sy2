import json
import os
import numpy as np
import trimesh
from pathlib import Path

class FurnitureLoader:
    def __init__(self, model_info_path, model_path):
        """初始化家具加载器
        
        Args:
            model_info_path: 模型信息JSON文件的路径
            model_path: 3D模型文件所在目录
        """
        self.model_path = Path(model_path)
        self.model_info = self._load_model_info(model_info_path)
        self.category_map = self._create_category_map()
        
    def _load_model_info(self, model_info_path):
        """加载模型信息
        
        Args:
            model_info_path: JSON文件路径
            
        Returns:
            dict: 模型ID到模型信息的映射
        """
        try:
            with open(model_info_path, 'r', encoding='utf-8') as f:
                model_info = json.load(f)
            return {model['model_id']: model for model in model_info}
        except Exception as e:
            print(f"加载模型信息时出错: {e}")
            return {}
            
    def _create_category_map(self):
        """创建类别到ID的映射
        
        Returns:
            dict: 类别名称到类别ID的映射
        """
        categories = set()
        for model in self.model_info.values():
            categories.add(model['category'])
        return {cat: idx for idx, cat in enumerate(sorted(categories))}
        
    def get_model_mesh(self, model_id):
        """获取指定模型ID的3D网格
        
        Args:
            model_id: 模型ID
            
        Returns:
            trimesh.Trimesh: 加载的3D网格，如果加载失败则返回None
        """
        if model_id not in self.model_info:
            print(f"未找到模型ID: {model_id}")
            return None
            
        model_info = self.model_info[model_id]
        obj_path = self.model_path / f"{model_id}.obj"
        
        try:
            mesh = trimesh.load(obj_path)
            return mesh
        except Exception as e:
            print(f"加载模型 {model_id} 时出错: {e}")
            return None
            
    def get_category_id(self, model_id):
        """获取模型的类别ID
        
        Args:
            model_id: 模型ID
            
        Returns:
            int: 类别ID，如果模型不存在则返回-1
        """
        if model_id not in self.model_info:
            return -1
        category = self.model_info[model_id]['category']
        return self.category_map.get(category, -1)
        
    def get_model_info(self, model_id):
        """获取模型的详细信息
        
        Args:
            model_id: 模型ID
            
        Returns:
            dict: 模型信息，如果模型不存在则返回None
        """
        return self.model_info.get(model_id)

class FurnitureRenderer:
    def __init__(self, furniture_loader):
        """
        家具渲染器
        Args:
            furniture_loader: FurnitureLoader实例
        """
        self.loader = furniture_loader
        
    def render_furniture(self, model_id, position, rotation, scale, save_path=None):
        """
        渲染单个家具
        Args:
            model_id: 模型ID
            position: [x, y, z]位置
            rotation: [rx, ry, rz]旋转（弧度）
            scale: [sx, sy, sz]缩放
            save_path: 保存路径
        """
        # 加载模型
        mesh = self.loader.get_model_mesh(model_id)
        if mesh is None:
            return
            
        # 应用变换
        mesh = mesh.copy()
        
        # 缩放
        mesh.apply_scale(scale)
        
        # 旋转
        rotation_matrix = trimesh.transformations.euler_matrix(
            rotation[0], rotation[1], rotation[2], 'xyz'
        )
        mesh.apply_transform(rotation_matrix)
        
        # 平移
        translation_matrix = trimesh.transformations.translation_matrix(position)
        mesh.apply_transform(translation_matrix)
        
        # 渲染
        scene = trimesh.Scene([mesh])
        
        if save_path:
            # 保存渲染结果
            png = scene.save_image(resolution=[1920, 1080])
            with open(save_path, 'wb') as f:
                f.write(png)
        else:
            # 显示渲染结果
            scene.show()
            
    def render_room(self, furniture_list, save_path=None):
        """
        渲染整个房间
        Args:
            furniture_list: 家具列表，每个元素包含：
                {
                    'model_id': 模型ID,
                    'position': [x, y, z],
                    'rotation': [rx, ry, rz],
                    'scale': [sx, sy, sz]
                }
            save_path: 保存路径
        """
        meshes = []
        
        for furniture in furniture_list:
            # 加载模型
            mesh = self.loader.get_model_mesh(furniture['model_id'])
            if mesh is None:
                continue
                
            # 应用变换
            mesh = mesh.copy()
            
            # 缩放
            mesh.apply_scale(furniture['scale'])
            
            # 旋转
            rotation_matrix = trimesh.transformations.euler_matrix(
                furniture['rotation'][0],
                furniture['rotation'][1],
                furniture['rotation'][2],
                'xyz'
            )
            mesh.apply_transform(rotation_matrix)
            
            # 平移
            translation_matrix = trimesh.transformations.translation_matrix(
                furniture['position']
            )
            mesh.apply_transform(translation_matrix)
            
            meshes.append(mesh)
        
        # 创建场景
        scene = trimesh.Scene(meshes)
        
        if save_path:
            # 保存渲染结果
            png = scene.save_image(resolution=[1920, 1080])
            with open(save_path, 'wb') as f:
                f.write(png)
        else:
            # 显示渲染结果
            scene.show() 