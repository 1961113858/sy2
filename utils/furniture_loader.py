import json
import os
import numpy as np
import trimesh
import pyrender
from pathlib import Path
import sys
sys.path.append("D:/cxcy2/LEGO-Net-main/data/3d-front")  # 添加3d-front目录到路径
from categories import _CATEGORIES_3D, _SUPER_CATEGORIES_3D

class FurnitureLoader:
    def __init__(self, model_info_path, model_path):
        """
        初始化家具加载器
        Args:
            model_info_path: 3D-FUTURE模型信息JSON文件路径
            model_path: 3D-FUTURE模型文件目录
        """
        self.model_path = Path(model_path)
        self.model_info = self._load_model_info(model_info_path)
        
        # 创建类别映射
        self.category_map = self._create_category_map()
        self.super_category_map = {cat['category']: cat['id'] for cat in _SUPER_CATEGORIES_3D}
        self.sub_category_map = {cat['category']: cat['super-category'] for cat in _CATEGORIES_3D}
        
        # 按类别组织模型
        self.models_by_category = self._organize_models_by_category()
        
        # 扫描可用模型
        self.available_models = self._scan_model_directory()
        print(f"\n找到 {len(self.available_models)} 个可用模型")
        
        # 添加默认模型映射
        self.default_models = {
            'bed': '0a72f3a8-2e89-4ca9-8e9e-3f91f0cc0c32',  # 一个床的模型ID
            'nightstand': '0a8d6f94-75d4-4c8c-9637-7a2e6d3c4c16',  # 床头柜
            'wardrobe': '0a9f96f4-7c7a-4b5c-9d0c-8b6c5b6f0f0f',  # 衣柜
            'desk': '0b29d749-3b90-4574-8a68-7d3e02d60a03',  # 书桌
            'chair': '0b1d7c45-5bf4-4c63-9066-0b0f6f04f510',  # 椅子
            'sofa': '0b2d7e45-6bf4-4c63-9066-0b0f6f04f520',  # 沙发
            'table': '0b3d7f45-7bf4-4c63-9066-0b0f6f04f530',  # 桌子
            'cabinet': '0b4d8045-8bf4-4c63-9066-0b0f6f04f540',  # 柜子
            'unknown': '0b5d8145-9bf4-4c63-9066-0b0f6f04f550'  # 默认模型
        }
        
    def _load_model_info(self, model_info_path):
        """加载模型信息"""
        try:
            print(f"正在加载模型信息文件: {model_info_path}")
            with open(model_info_path, 'r', encoding='utf-8') as f:
                model_info_list = json.load(f)
                print(f"成功加载 {len(model_info_list)} 个模型信息")
                
                # 将列表转换为字典，使用model_id作为键
                model_info = {item['model_id']: item for item in model_info_list}
                print(f"\n成功处理 {len(model_info)} 个模型信息")
                
                # 统计超类别
                super_categories = {}
                for info in model_info.values():
                    super_cat = info.get('super-category', 'unknown')
                    if super_cat not in super_categories:
                        super_categories[super_cat] = 0
                    super_categories[super_cat] += 1
                
                print("\n超类别统计:")
                for super_cat, count in super_categories.items():
                    print(f"{super_cat}: {count}个模型")
                
                # 打印几个示例
                print("\n示例模型:")
                for model_id in list(model_info.keys())[:3]:
                    info = model_info[model_id]
                    print(f"- ID: {model_id}")
                    print(f"  超类别: {info.get('super-category', 'unknown')}")
                    print(f"  类别: {info.get('category', 'unknown')}")
                    print(f"  风格: {info.get('style', 'unknown')}")
                    print(f"  材质: {info.get('material', 'unknown')}")
                
                return model_info
                
        except Exception as e:
            print(f"加载模型信息时出错: {e}")
            import traceback
            traceback.print_exc()
            return {}
            
    def _create_category_map(self):
        """创建类别到ID的映射"""
        categories = set()
        for model in self.model_info.values():
            categories.add(model['category'])
        return {cat: idx for idx, cat in enumerate(sorted(categories))}
        
    def _organize_models_by_category(self):
        """将模型按超类别和类别组织"""
        models_by_category = {}
        for model_id, info in self.model_info.items():
            super_cat = info.get('super-category', 'unknown')
            category = info.get('category', 'unknown')
            
            # 按超类别组织
            if super_cat not in models_by_category:
                models_by_category[super_cat] = {'_all': [], '_sub': {}}
            models_by_category[super_cat]['_all'].append(model_id)
            
            # 按具体类别组织
            if category not in models_by_category[super_cat]['_sub']:
                models_by_category[super_cat]['_sub'][category] = []
            models_by_category[super_cat]['_sub'][category].append(model_id)
        
        # 打印类别统计
        print("\n模型类别统计:")
        for super_cat, data in models_by_category.items():
            print(f"\n{super_cat}: {len(data['_all'])}个模型")
            for category, models in data['_sub'].items():
                print(f"  - {category}: {len(models)}个")
        
        return models_by_category

    def get_model_by_category(self, category):
        """获取指定类别的随机一个模型ID"""
        if category in self.models_by_category and self.models_by_category[category]:
            models = self.models_by_category[category]
            return models[0]  # 返回第一个模型，也可以随机选择
        return None

    def _scan_model_directory(self):
        """扫描模型目录，找出所有可用的模型"""
        available_models = {}
        try:
            print(f"\n扫描模型目录: {self.model_path}")
            
            # 检查目录是否存在
            model_dir = self.model_path / "3D-FUTURE-model"
            if not model_dir.exists():
                print(f"错误: 模型目录不存在: {model_dir}")
                return {}
            
            # 列出所有可能的模型文件
            model_files = list(model_dir.glob("**/normalized_model.obj")) + list(model_dir.glob("**/raw_model.obj"))
            print(f"找到 {len(model_files)} 个.obj文件")
            
            # 检查每个模型文件
            for obj_file in model_files:
                try:
                    # 从路径中提取模型ID
                    model_id = obj_file.parent.name
                    if model_id in self.model_info:
                        model_info = self.model_info[model_id]
                        super_cat = model_info.get('super-category', 'unknown')
                        category = model_info.get('category', 'unknown')
                        
                        # 按超类别组织
                        if super_cat not in available_models:
                            available_models[super_cat] = {'_all': [], '_sub': {}}
                        if model_id not in available_models[super_cat]['_all']:  # 避免重复
                            available_models[super_cat]['_all'].append(model_id)
                        
                        # 按具体类别组织
                        if category not in available_models[super_cat]['_sub']:
                            available_models[super_cat]['_sub'][category] = []
                        if model_id not in available_models[super_cat]['_sub'][category]:  # 避免重复
                            available_models[super_cat]['_sub'][category].append(model_id)
                        
                        print(f"找到模型: {model_id} ({category})")
                except Exception as e:
                    print(f"处理模型文件时出错 {obj_file}: {e}")
            
            # 打印可用模型统计
            print("\n可用模型统计:")
            for super_cat, data in available_models.items():
                print(f"\n{super_cat}: {len(data['_all'])}个模型")
                for category, models in data['_sub'].items():
                    print(f"  - {category}: {len(models)}个")
            
            return available_models
            
        except Exception as e:
            print(f"扫描模型目录时出错: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def get_default_model(self, category):
        """获取指定类别的默认模型ID"""
        if category in self.available_models and self.available_models[category]:
            # 返回该类别的第一个可用模型
            return self.available_models[category][0]
        return None

    def get_model_info(self, model_id):
        """获取模型信息，如果找不到则使用同类别的替代模型"""
        if model_id not in self.model_info:
            print(f"警告: 找不到模型 {model_id}，尝试使用替代模型")
            # 返回同类别的第一个可用模型
            for info in self.model_info.values():
                if info.get('category') == 'armchair':  # 或其他合适的类别
                    return info
        return self.model_info.get(model_id)
        
    def _guess_furniture_type(self, model_id):
        """使用3D-FRONT类别系统猜测家具类型"""
        try:
            # 检查模型信息
            if model_id in self.model_info:
                model_info = self.model_info[model_id]
                category = model_info.get('category', '')
                
                # 查找超类别
                if category in self.sub_category_map:
                    super_cat = self.sub_category_map[category]
                    print(f"找到家具类别: {category} (属于 {super_cat})")
                    return category
                
            # 如果没有直接匹配，尝试模糊匹配
            for cat in _CATEGORIES_3D:
                category = cat['category'].lower()
                if any(keyword in model_id.lower() for keyword in category.split()):
                    print(f"通过关键词匹配到类别: {cat['category']}")
                    return cat['category']
            
            print(f"无法识别家具类型: {model_id}")
            return 'unknown'
            
        except Exception as e:
            print(f"猜测家具类型时出错: {e}")
            return 'unknown'
        
    def load_model(self, model_id):
        """加载3D模型"""
        model_path = f"{self.model_path}/{model_id}/raw.obj"
        return trimesh.load(model_path)
        
    def get_model_mesh(self, model_id):
        """获取模型的3D网格"""
        try:
            # 构建模型文件路径 - 注意目录结构
            model_path = self.model_path / "3D-FUTURE-model" / model_id / "normalized_model.obj"
            if not model_path.exists():
                # 尝试 raw_model.obj
                model_path = self.model_path / "3D-FUTURE-model" / model_id / "raw_model.obj"
                if not model_path.exists():
                    print(f"模型文件不存在: {model_path}")
                    return None
            
            print(f"加载模型文件: {model_path}")
            # 加载OBJ文件
            mesh = trimesh.load(str(model_path))
            print(f"成功加载模型: {model_id}")
            return mesh
        
        except Exception as e:
            print(f"加载模型失败 {model_id}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def render_model(self, mesh, save_path):
        """渲染3D模型"""
        try:
            # 创建场景
            scene = pyrender.Scene()
            
            # 创建相机
            camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
            
            # 添加网格到场景
            mesh = pyrender.Mesh.from_trimesh(mesh)
            scene.add(mesh)
            
            # 添加相机到场景
            camera_pose = np.array([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 2.0],
                [0.0, 0.0, 0.0, 1.0]
            ])
            scene.add(camera, pose=camera_pose)
            
            # 添加光源
            light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
            scene.add(light, pose=camera_pose)
            
            # 渲染
            r = pyrender.OffscreenRenderer(400, 400)
            color, depth = r.render(scene)
            
            # 保存图像
            import matplotlib.pyplot as plt
            plt.imsave(save_path, color)
            print(f"渲染完成: {save_path}")
            
            return True
        
        except Exception as e:
            print(f"渲染失败: {e}")
            return False

    def get_category_id(self, model_id):
        """获取模型的类别ID"""
        if model_id not in self.model_info:
            return -1
        category = self.model_info[model_id]['category']
        return self.category_map.get(category, -1)
        
    def render_scene(self, furniture_list, save_path):
        """渲染3D场景"""
        # 创建场景
        scene = pyrender.Scene()
        
        # 添加相机
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        scene.add(camera, pose=self._get_camera_pose())
        
        # 添加光源
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
        scene.add(light, pose=self._get_light_pose())
        
        # 添加家具
        for furniture in furniture_list:
            model = self.load_model(furniture['model_id'])
            mesh = pyrender.Mesh.from_trimesh(model)
            
            # 应用变换
            pose = self._get_furniture_pose(
                furniture['position'],
                furniture['rotation']
            )
            scene.add(mesh, pose=pose)
            
        # 渲染场景
        r = pyrender.OffscreenRenderer(1024, 768)
        color, depth = r.render(scene)
        
        # 保存渲染结果
        import imageio
        imageio.imwrite(save_path, color)
        
    def _get_camera_pose(self):
        """获取相机位姿"""
        return np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, np.cos(-np.pi/6), -np.sin(-np.pi/6), 2.0],
            [0.0, np.sin(-np.pi/6), np.cos(-np.pi/6), 2.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        
    def _get_light_pose(self):
        """获取光源位姿"""
        return np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 5.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        
    def _get_furniture_pose(self, position, rotation):
        """获取家具位姿"""
        pose = np.eye(4)
        pose[:3, 3] = position
        pose[:3, :3] = self._rotation_matrix(rotation)
        return pose
        
    def _rotation_matrix(self, rotation):
        """计算旋转矩阵"""
        return np.array([
            [np.cos(rotation), -np.sin(rotation), 0],
            [np.sin(rotation), np.cos(rotation), 0],
            [0, 0, 1]
        ]) 