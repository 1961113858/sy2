import numpy as np
import matplotlib.pyplot as plt

# 添加家具类型映射
FURNITURE_TYPES = {
    0: "床",
    1: "床头柜",
    2: "衣柜",
    3: "书桌",
    4: "椅子"
}

# 添加颜色映射
FURNITURE_COLORS = {
    0: ('#FFB6C1', '#FF69B4'),  # 床：粉色
    1: ('#ADD8E6', '#4169E1'),  # 床头柜：浅蓝色
    2: ('#98FB98', '#228B22'),  # 衣柜：绿色
    3: ('#DEB887', '#8B4513'),  # 书桌：棕色
    4: ('#F0E68C', '#DAA520')   # 椅子：金色
}

def visualize_scene(input_scene, output_scene=None, save_path=None):
    """
    可视化场景布局
    Args:
        input_scene: 输入场景 [N, D] 或 [B, N, D]
        output_scene: 输出场景 [N, D] 或 [B, N, D] (可选)
        save_path: 保存路径 (可选)
    """
    # 确保输入是2D数组
    if input_scene.ndim == 3:
        input_scene = input_scene[0]  # 只取第一个batch
    if output_scene is not None and output_scene.ndim == 3:
        output_scene = output_scene[0]
    
    # 创建图形
    fig = plt.figure(figsize=(15, 6))
    
    # 绘制输入场景
    ax1 = fig.add_subplot(121)
    plot_scene(ax1, input_scene, title="输入场景")
    
    # 绘制输出场景(如果有)
    if output_scene is not None:
        ax2 = fig.add_subplot(122)
        plot_scene(ax2, output_scene, title="优化后的场景")
    
    # 添加图例
    legend_elements = []
    for ftype, name in FURNITURE_TYPES.items():
        color = FURNITURE_COLORS[ftype][0]
        edge_color = FURNITURE_COLORS[ftype][1]
        patch = plt.Rectangle((0, 0), 1, 1, facecolor=color, 
                            edgecolor=edge_color, alpha=0.5, label=name)
        legend_elements.append(patch)
    
    # 将图例放在图形下方
    fig.legend(handles=legend_elements, loc='lower center', 
              ncol=5, bbox_to_anchor=(0.5, -0.1))
    
    # 调整布局以适应图例
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    
    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        
def plot_scene(ax, scene, title=None):
    """
    绘制单个场景的2D俯视图
    Args:
        ax: matplotlib轴对象
        scene: 场景数据 [N, D]
        title: 标题
    """
    # 确保scene是numpy数组
    scene = np.asarray(scene)
    
    # 只处理非零行
    mask = np.any(scene != 0, axis=1)
    scene = scene[mask]
    
    if len(scene) == 0:
        print("警告: 场景中没有有效物体")
        return
    
    # 分离位置、旋转、尺寸和类别
    positions = scene[:, :3]    # [x, y, z]
    rotations = scene[:, 3:6]   # [rx, ry, rz]
    sizes = scene[:, 6:9]       # [sx, sy, sz]
    categories = scene[:, 9].astype(int)  # 类别
    
    # 绘制每个物体的2D投影
    for i in range(len(positions)):
        plot_furniture_2d(
            ax,
            positions[i],
            rotations[i],
            sizes[i],
            categories[i]
        )
    
    # 设置轴标签和标题
    ax.set_xlabel('X (米)')
    ax.set_ylabel('Y (米)')
    if title:
        ax.set_title(title)
    
    # 设置相等的横纵比例
    ax.set_aspect('equal')
    
    # 设置显示范围
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    
    # 添加网格
    ax.grid(True, linestyle='--', alpha=0.3)
    
def plot_furniture_2d(ax, position, rotation, size, category):
    """
    绘制单个家具的2D俯视图
    Args:
        ax: matplotlib轴对象
        position: 位置 [3]
        rotation: 旋转 [3]
        size: 尺寸 [3]
        category: 家具类别
    """
    # 提取2D信息
    x, y = position[:2]
    rz = rotation[2]  # 只使用z轴旋转
    sx, sy = size[:2]  # 只使用x-y平面的尺寸
    
    # 获取家具颜色
    face_color = FURNITURE_COLORS[category % len(FURNITURE_COLORS)][0]
    edge_color = FURNITURE_COLORS[category % len(FURNITURE_COLORS)][1]
    
    # 创建矩形顶点
    corners = np.array([
        [-sx/2, -sy/2],
        [sx/2, -sy/2],
        [sx/2, sy/2],
        [-sx/2, sy/2]
    ])
    
    # 应用旋转
    c, s = np.cos(rz), np.sin(rz)
    R = np.array([[c, -s], [s, c]])
    corners = corners @ R.T
    
    # 应用平移
    corners = corners + np.array([x, y])
    
    # 绘制填充矩形
    polygon = plt.Polygon(corners, facecolor=face_color, 
                         edgecolor=edge_color, alpha=0.5)
    ax.add_patch(polygon)
    
    # 获取家具名称
    furniture_name = FURNITURE_TYPES.get(category % len(FURNITURE_TYPES), str(category))
    
    # 添加家具名称标注
    ax.annotate(furniture_name, (x, y), xytext=(0, 0), 
                textcoords='offset points', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='gray', alpha=0.7),
                fontsize=8) 