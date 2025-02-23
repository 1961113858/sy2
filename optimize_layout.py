import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam

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

class LayoutOptimizer:
    def __init__(self, config=None):
        self.config = config or self._default_config()
        
    def _default_config(self):
        return {
            'learning_rate': 0.01,
            'num_steps': 200,
            'min_distance': 0.5,      # 物体之间的最小距离
            'room_size': 8.0,         # 房间大小
            'overlap_weight': 10.0,    # 重叠损失权重
            'boundary_weight': 5.0,    # 边界约束权重
            'distance_weight': 3.0,    # 距离约束权重
            'center_weight': 1.0       # 中心约束权重
        }
    
    def optimize_layout(self, furniture_data):
        """
        优化家具布局
        Args:
            furniture_data: 包含位置和尺寸的张量 [N, 6] (x, y, z, width, height, depth)
        """
        # 提取位置和尺寸
        positions = furniture_data[:, :3].clone()
        sizes = furniture_data[:, 3:6].clone()
        
        # 将位置转换为可优化的参数
        positions.requires_grad_(True)
        optimizer = Adam([positions], lr=self.config['learning_rate'])
        
        # 优化循环
        best_positions = None
        best_loss = float('inf')
        
        print("开始优化布局...")
        for step in range(self.config['num_steps']):
            optimizer.zero_grad()
            
            # 计算各种损失
            overlap_loss = self._compute_overlap_loss(positions, sizes)
            boundary_loss = self._compute_boundary_loss(positions, sizes)
            distance_loss = self._compute_distance_loss(positions)
            center_loss = self._compute_center_loss(positions)
            
            # 总损失
            total_loss = (
                self.config['overlap_weight'] * overlap_loss +
                self.config['boundary_weight'] * boundary_loss +
                self.config['distance_weight'] * distance_loss +
                self.config['center_weight'] * center_loss
            )
            
            # 反向传播
            total_loss.backward()
            
            # 更新位置
            optimizer.step()
            
            # 保存最佳结果
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_positions = positions.detach().clone()
            
            # 打印进度
            if (step + 1) % 20 == 0:
                print(f"步骤 {step+1}/{self.config['num_steps']}, "
                      f"损失: {total_loss.item():.4f} "
                      f"(重叠: {overlap_loss.item():.4f}, "
                      f"边界: {boundary_loss.item():.4f}, "
                      f"距离: {distance_loss.item():.4f}, "
                      f"中心: {center_loss.item():.4f})")
        
        # 返回优化后的布局
        optimized_furniture = furniture_data.clone()
        optimized_furniture[:, :3] = best_positions
        
        return optimized_furniture
    
    def _compute_overlap_loss(self, positions, sizes):
        """计算重叠损失"""
        num_objects = positions.shape[0]
        overlap_loss = torch.tensor(0.0, requires_grad=True)
        
        for i in range(num_objects):
            for j in range(i + 1, num_objects):
                # 计算两个物体的边界框
                min1 = positions[i] - sizes[i] / 2
                max1 = positions[i] + sizes[i] / 2
                min2 = positions[j] - sizes[j] / 2
                max2 = positions[j] + sizes[j] / 2
                
                # 计算重叠
                overlap = torch.clamp(
                    torch.min(max1, max2) - torch.max(min1, min2),
                    min=0
                )
                overlap_volume = torch.prod(overlap)
                overlap_loss = overlap_loss + overlap_volume
        
        return overlap_loss
    
    def _compute_boundary_loss(self, positions, sizes):
        """计算边界约束损失"""
        # 确保物体在房间范围内
        half_sizes = sizes / 2
        min_coords = positions - half_sizes
        max_coords = positions + half_sizes
        
        room_min = torch.tensor([-self.config['room_size']/2] * 3)
        room_max = torch.tensor([self.config['room_size']/2] * 3)
        
        # 计算超出边界的距离
        out_of_bounds = torch.sum(
            torch.clamp(room_min - min_coords, min=0) +
            torch.clamp(max_coords - room_max, min=0)
        )
        
        return out_of_bounds
    
    def _compute_distance_loss(self, positions):
        """计算距离约束损失"""
        num_objects = positions.shape[0]
        distance_loss = torch.tensor(0.0, requires_grad=True)
        
        for i in range(num_objects):
            for j in range(i + 1, num_objects):
                # 计算物体间距离
                distance = torch.norm(positions[i] - positions[j])
                # 如果距离小于最小距离，添加惩罚
                if distance < self.config['min_distance']:
                    distance_loss = distance_loss + (self.config['min_distance'] - distance) ** 2
        
        return distance_loss
    
    def _compute_center_loss(self, positions):
        """计算中心约束损失（避免物体都挤在一起）"""
        center = torch.mean(positions, dim=0)
        return torch.sum((center) ** 2)  # 鼓励物体分散在房间中心周围
    
    def visualize_layout(self, furniture_data, furniture_types=None, title=None):
        """可视化布局"""
        positions = furniture_data[:, :3].detach().numpy()
        sizes = furniture_data[:, 3:6].detach().numpy()
        
        # 创建3D图
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制每个物体
        for i, (pos, size) in enumerate(zip(positions, sizes)):
            # 获取家具类型和颜色
            furniture_type = i % len(FURNITURE_TYPES) if furniture_types is None else furniture_types[i]
            face_color = FURNITURE_COLORS[furniture_type][0]
            edge_color = FURNITURE_COLORS[furniture_type][1]
            
            # 创建立方体
            x, y, z = pos
            dx, dy, dz = size
            
            # 绘制立方体
            xx = [x-dx/2, x+dx/2]
            yy = [y-dy/2, y+dy/2]
            zz = [z-dz/2, z+dz/2]
            
            xx, yy = np.meshgrid(xx, yy)
            ax.plot_surface(xx, yy, np.full_like(xx, zz[0]), color=face_color, alpha=0.7)
            ax.plot_surface(xx, yy, np.full_like(xx, zz[1]), color=face_color, alpha=0.7)
            
            yy, zz = np.meshgrid(yy[:,0], zz)
            ax.plot_surface(np.full_like(yy, xx[0,0]), yy, zz, color=face_color, alpha=0.7)
            ax.plot_surface(np.full_like(yy, xx[0,1]), yy, zz, color=face_color, alpha=0.7)
            
            xx, zz = np.meshgrid(xx[0,:], zz[:,0])
            ax.plot_surface(xx, np.full_like(xx, yy[0,0]), zz, color=face_color, alpha=0.7)
            ax.plot_surface(xx, np.full_like(xx, yy[0,1]), zz, color=face_color, alpha=0.7)
            
            # 添加家具标签
            furniture_name = FURNITURE_TYPES[furniture_type]
            ax.text(x, y, z + dz/2, furniture_name, 
                   horizontalalignment='center', verticalalignment='bottom')
        
        # 设置视角和标签
        ax.view_init(elev=30, azim=45)
        ax.set_xlabel('X轴 (米)')
        ax.set_ylabel('Y轴 (米)')
        ax.set_zlabel('Z轴 (米)')
        
        # 设置显示范围
        room_size = self.config['room_size']
        ax.set_xlim([-room_size/2, room_size/2])
        ax.set_ylim([-room_size/2, room_size/2])
        ax.set_zlim([0, room_size/2])
        
        if title:
            plt.title(title)
            
        # 添加图例
        legend_elements = []
        for ftype, name in FURNITURE_TYPES.items():
            color = FURNITURE_COLORS[ftype][0]
            legend_elements.append(plt.Rectangle((0, 0), 1, 1, fc=color, label=name))
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        return fig

def optimize_test_layout():
    """测试布局优化"""
    # 创建一些测试数据
    num_furniture = 5
    furniture_data = torch.zeros(num_furniture, 6)
    
    # 随机初始化位置和尺寸
    furniture_data[:, :3] = torch.randn(num_furniture, 3) * 2  # 位置
    furniture_data[:, 3:6] = torch.rand(num_furniture, 3) + 1  # 尺寸
    
    # 创建家具类型列表
    furniture_types = list(range(num_furniture))
    
    # 创建优化器
    optimizer = LayoutOptimizer()
    
    # 可视化原始布局
    print("正在可视化原始布局...")
    fig = optimizer.visualize_layout(furniture_data, furniture_types, "原始布局")
    plt.savefig('original_layout.png')
    plt.close()
    
    # 优化布局
    print("\n开始优化布局...")
    optimized_furniture = optimizer.optimize_layout(furniture_data)
    
    # 可视化优化后的布局
    print("\n正在可视化优化后的布局...")
    fig = optimizer.visualize_layout(optimized_furniture, furniture_types, "优化后的布局")
    plt.savefig('optimized_layout.png')
    plt.close()
    
    print("\n优化完成！结果已保存为 original_layout.png 和 optimized_layout.png")

if __name__ == "__main__":
    optimize_test_layout() 