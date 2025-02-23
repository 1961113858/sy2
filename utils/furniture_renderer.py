import numpy as np
import pyrender
import trimesh

class FurnitureRenderer:
    def __init__(self, furniture_loader):
        """初始化渲染器
        
        Args:
            furniture_loader: FurnitureLoader实例
        """
        self.loader = furniture_loader
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
        
    def render_room(self, furniture_list, save_path=None):
        """渲染包含多个家具的房间
        
        Args:
            furniture_list: 家具列表，每个元素包含：
                {
                    'model_id': 模型ID,
                    'position': [x, y, z],
                    'rotation': [rx, ry, rz],
                    'scale': [sx, sy, sz]
                }
            save_path: 保存路径
            
        Returns:
            np.ndarray: 渲染的RGB图像
        """
        nodes = []
        
        # 添加所有家具到场景
        for furniture in furniture_list:
            # 加载模型
            mesh = self.loader.get_model_mesh(furniture['model_id'])
            if mesh is None:
                continue
                
            # 转换为pyrender网格
            mesh = pyrender.Mesh.from_trimesh(mesh)
            
            # 创建变换矩阵
            transform = np.eye(4)
            
            # 缩放
            transform[:3, :3] *= np.array(furniture['scale'])
            
            # 旋转
            rx, ry, rz = furniture['rotation']
            Rx = np.array([
                [1, 0, 0],
                [0, np.cos(rx), -np.sin(rx)],
                [0, np.sin(rx), np.cos(rx)]
            ])
            Ry = np.array([
                [np.cos(ry), 0, np.sin(ry)],
                [0, 1, 0],
                [-np.sin(ry), 0, np.cos(ry)]
            ])
            Rz = np.array([
                [np.cos(rz), -np.sin(rz), 0],
                [np.sin(rz), np.cos(rz), 0],
                [0, 0, 1]
            ])
            transform[:3, :3] = Rz @ Ry @ Rx @ transform[:3, :3]
            
            # 平移
            transform[:3, 3] = furniture['position']
            
            # 创建节点
            node = pyrender.Node(mesh=mesh, matrix=transform)
            self.scene.add_node(node)
            nodes.append(node)
        
        # 渲染
        r = pyrender.OffscreenRenderer(800, 600)
        color, depth = r.render(self.scene)
        
        # 移除所有节点
        for node in nodes:
            self.scene.remove_node(node)
        
        # 保存结果
        if save_path:
            from PIL import Image
            image = Image.fromarray(color)
            image.save(save_path)
        
        return color 