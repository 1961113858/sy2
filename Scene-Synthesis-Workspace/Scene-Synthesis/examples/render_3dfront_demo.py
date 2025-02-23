import os
import sys
import numpy as np
import trimesh
import pyrender

from utils.furniture_loader import FurnitureLoader
from utils.furniture_renderer import FurnitureRenderer

# 定义3D-FRONT数据路径
MODEL_INFO_PATH = '../data/3d-front/model_info.json'
MODEL_PATH = '../data/3d-front/models'

def create_sample_scene():
    # 创建家具加载器和渲染器
    furniture_loader = FurnitureLoader(MODEL_INFO_PATH, MODEL_PATH)
    renderer = FurnitureRenderer()
    
    # 定义示例家具及其位置
    furniture_list = [
        {
            'model_id': 'bed_0001',  # 示例ID，需要替换为实际的3D-FRONT模型ID
            'position': np.array([0.0, 0.0, 0.0]),
            'rotation': np.array([0.0, 0.0, 0.0]),
            'scale': np.array([1.0, 1.0, 1.0])
        },
        {
            'model_id': 'nightstand_0001',  # 示例ID
            'position': np.array([1.5, 0.0, 0.0]),
            'rotation': np.array([0.0, 0.0, 0.0]),
            'scale': np.array([1.0, 1.0, 1.0])
        }
    ]
    
    # 创建场景
    scene = pyrender.Scene()
    
    # 添加家具到场景
    for furniture in furniture_list:
        mesh = furniture_loader.get_model_mesh(furniture['model_id'])
        if mesh is not None:
            # 转换trimesh网格为pyrender网格
            mesh = pyrender.Mesh.from_trimesh(mesh)
            # 创建节点
            node = pyrender.Node(
                mesh=mesh,
                translation=furniture['position'],
                rotation=furniture['rotation'],
                scale=furniture['scale']
            )
            scene.add_node(node)
    
    # 添加光源
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    scene.add(light)
    
    return scene

def main():
    # 创建输出目录
    os.makedirs('results', exist_ok=True)
    
    # 创建并渲染场景
    scene = create_sample_scene()
    
    # 设置相机
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    camera_pose = np.array([
        [1.0, 0.0, 0.0, 2.0],
        [0.0, 1.0, 0.0, 2.0],
        [0.0, 0.0, 1.0, 2.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    scene.add(camera, pose=camera_pose)
    
    # 渲染场景
    r = pyrender.OffscreenRenderer(400, 400)
    color, depth = r.render(scene)
    
    # 保存渲染结果
    from PIL import Image
    image = Image.fromarray(color)
    image.save('results/3dfront_render.png')
    
    print("渲染完成！结果已保存到 results/3dfront_render.png")

if __name__ == '__main__':
    main() 