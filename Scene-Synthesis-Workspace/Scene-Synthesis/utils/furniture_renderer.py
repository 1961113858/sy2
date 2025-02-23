import numpy as np
import pyrender
import trimesh

class FurnitureRenderer:
    def __init__(self):
        """初始化渲染器"""
        self.scene = pyrender.Scene()
        self._setup_lighting()
        self._setup_camera()
        
    def _setup_lighting(self):
        """设置场景光照"""
        # 添加环境光
        self.scene.ambient_light = np.array([0.2, 0.2, 0.2, 1.0])
        
        # 添加定向光源
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
        light_pose = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 2.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        self.scene.add(light, pose=light_pose)
        
    def _setup_camera(self):
        """设置相机"""
        self.camera = pyrender.PerspectiveCamera(
            yfov=np.pi / 3.0,
            aspectRatio=1.0
        )
        camera_pose = np.array([
            [1.0, 0.0, 0.0, 2.0],
            [0.0, 1.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 2.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        self.scene.add(self.camera, pose=camera_pose)
        
    def render_furniture(self, mesh, position, rotation, scale, output_path=None):
        """渲染单个家具
        
        Args:
            mesh: trimesh.Trimesh对象
            position: 位置坐标 [x, y, z]
            rotation: 旋转角度 [rx, ry, rz]
            scale: 缩放系数 [sx, sy, sz]
            output_path: 输出图像路径
            
        Returns:
            np.ndarray: 渲染的RGB图像
        """
        # 转换trimesh网格为pyrender网格
        mesh = pyrender.Mesh.from_trimesh(mesh)
        
        # 创建节点并添加到场景
        node = pyrender.Node(
            mesh=mesh,
            translation=position,
            rotation=rotation,
            scale=scale
        )
        self.scene.add_node(node)
        
        # 渲染
        r = pyrender.OffscreenRenderer(400, 400)
        color, depth = r.render(self.scene)
        
        # 移除节点
        self.scene.remove_node(node)
        
        # 保存结果
        if output_path:
            from PIL import Image
            image = Image.fromarray(color)
            image.save(output_path)
        
        return color
        
    def render_room(self, furniture_list, output_path=None):
        """渲染包含多个家具的房间
        
        Args:
            furniture_list: 家具列表，每个元素包含mesh和变换信息
            output_path: 输出图像路径
            
        Returns:
            np.ndarray: 渲染的RGB图像
        """
        nodes = []
        
        # 添加所有家具到场景
        for furniture in furniture_list:
            mesh = pyrender.Mesh.from_trimesh(furniture['mesh'])
            node = pyrender.Node(
                mesh=mesh,
                translation=furniture['position'],
                rotation=furniture['rotation'],
                scale=furniture['scale']
            )
            self.scene.add_node(node)
            nodes.append(node)
        
        # 渲染
        r = pyrender.OffscreenRenderer(800, 600)
        color, depth = r.render(self.scene)
        
        # 移除所有节点
        for node in nodes:
            self.scene.remove_node(node)
        
        # 保存结果
        if output_path:
            from PIL import Image
            image = Image.fromarray(color)
            image.save(output_path)
        
        return color 