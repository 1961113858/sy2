import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, PathPatch
from matplotlib.path import Path
import matplotlib.patches as patches

# 家具类型映射及其标准尺寸和样式
FURNITURE_STYLES = {
    0: {  # 床
        'name': '床',
        'color': '#FFE4E1',  # 浅粉色
        'edge_color': '#FFB6C1',
        'pattern': 'bed',
        'min_size': [1.8, 2.0]  # 最小宽度和长度
    },
    1: {  # 床头柜
        'name': '床头柜',
        'color': '#E6E6FA',  # 淡紫色
        'edge_color': '#9370DB',
        'pattern': 'nightstand',
        'min_size': [0.4, 0.4]
    },
    2: {  # 衣柜
        'name': '衣柜',
        'color': '#F0F8FF',  # 淡蓝色
        'edge_color': '#87CEEB',
        'pattern': 'wardrobe',
        'min_size': [0.6, 1.2]
    },
    3: {  # 书桌
        'name': '书桌',
        'color': '#F5F5DC',  # 米色
        'edge_color': '#DEB887',
        'pattern': 'desk',
        'min_size': [0.6, 1.2]
    },
    4: {  # 椅子
        'name': '椅子',
        'color': '#FFF0F5',  # 淡粉色
        'edge_color': '#DDA0DD',
        'pattern': 'chair',
        'min_size': [0.4, 0.4]
    }
}

def draw_bed(ax, x, y, width, height, angle, style):
    """绘制床的图案"""
    # 创建床的主体
    bed = Rectangle((x-width/2, y-height/2), width, height,
                   angle=np.degrees(angle),
                   facecolor=style['color'],
                   edgecolor=style['edge_color'],
                   linewidth=2,
                   alpha=0.7)
    ax.add_patch(bed)
    
    # 添加床头板
    head_width = width * 0.2
    head = Rectangle((x-width/2, y+height/2-head_width), width, head_width,
                    angle=np.degrees(angle),
                    facecolor=style['edge_color'],
                    edgecolor=style['edge_color'],
                    alpha=0.5)
    ax.add_patch(head)
    
    # 添加床垫纹理
    for i in range(3):
        line_y = y - height/3 + i*height/3
        line = plt.Line2D([x-width/3, x+width/3], [line_y, line_y],
                         color=style['edge_color'],
                         alpha=0.3,
                         transform=ax.transData)
        ax.add_line(line)

def draw_wardrobe(ax, x, y, width, height, angle, style):
    """绘制衣柜的图案"""
    # 主体
    wardrobe = Rectangle((x-width/2, y-height/2), width, height,
                        angle=np.degrees(angle),
                        facecolor=style['color'],
                        edgecolor=style['edge_color'],
                        linewidth=2,
                        alpha=0.7)
    ax.add_patch(wardrobe)
    
    # 添加门线
    door_line = plt.Line2D([x, x], [y-height/2, y+height/2],
                          color=style['edge_color'],
                          linewidth=1,
                          alpha=0.5)
    ax.add_line(door_line)

def draw_desk(ax, x, y, width, height, angle, style):
    """绘制书桌的图案"""
    desk = Rectangle((x-width/2, y-height/2), width, height,
                    angle=np.degrees(angle),
                    facecolor=style['color'],
                    edgecolor=style['edge_color'],
                    linewidth=2,
                    alpha=0.7)
    ax.add_patch(desk)

def render_floorplan(scene_data, room_size=(4, 4), output_path=None):
    """渲染精美的户型图"""
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # 设置房间
    width, length = room_size
    
    # 绘制墙体（加厚）
    wall_thickness = 0.2
    walls = [
        Rectangle((0, 0), width, wall_thickness, facecolor='gray', alpha=0.7),  # 下墙
        Rectangle((0, 0), wall_thickness, length, facecolor='gray', alpha=0.7),  # 左墙
        Rectangle((width-wall_thickness, 0), wall_thickness, length, facecolor='gray', alpha=0.7),  # 右墙
        Rectangle((0, length-wall_thickness), width, wall_thickness, facecolor='gray', alpha=0.7)  # 上墙
    ]
    for wall in walls:
        ax.add_patch(wall)
    
    # 添加门（在下墙中间）
    door_width = 0.8
    door_x = width/2 - door_width/2
    door = patches.Arc((door_x, 0), door_width, door_width,
                      theta1=0, theta2=90,
                      color='gray', linewidth=2)
    ax.add_patch(door)
    
    # 绘制家具
    valid_mask = np.any(scene_data != 0, axis=1)
    valid_furniture = scene_data[valid_mask]
    
    for furniture in valid_furniture:
        pos = furniture[3:6]    # 位置
        rot = furniture[0:3]    # 旋转
        size = furniture[6:9]   # 尺寸
        category = int(furniture[9])  # 类别
        
        if category in FURNITURE_STYLES:
            style = FURNITURE_STYLES[category]
            
            # 根据家具类型调用不同的绘制函数
            if category == 0:  # 床
                draw_bed(ax, pos[0], pos[1], size[0], size[1], rot[2], style)
            elif category == 2:  # 衣柜
                draw_wardrobe(ax, pos[0], pos[1], size[0], size[1], rot[2], style)
            elif category == 3:  # 书桌
                draw_desk(ax, pos[0], pos[1], size[0], size[1], rot[2], style)
            else:  # 其他家具使用简单矩形
                rect = Rectangle((pos[0]-size[0]/2, pos[1]-size[1]/2),
                               size[0], size[1],
                               angle=np.degrees(rot[2]),
                               facecolor=style['color'],
                               edgecolor=style['edge_color'],
                               linewidth=2,
                               alpha=0.7)
                ax.add_patch(rect)
    
    # 添加网格和刻度
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_aspect('equal')
    
    # 设置坐标轴范围
    ax.set_xlim(-0.5, width+0.5)
    ax.set_ylim(-0.5, length+0.5)
    
    # 添加标题和图例
    plt.title('户型图 (单位: 米)', pad=20, fontsize=14)
    
    # 添加图例
    legend_elements = [patches.Patch(facecolor=style['color'],
                                   edgecolor=style['edge_color'],
                                   alpha=0.7,
                                   label=style['name'])
                      for style in FURNITURE_STYLES.values()]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # 保存或显示
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def main():
    # 创建输出目录
    os.makedirs("results", exist_ok=True)
    
    try:
        # 加载优化后的场景数据
        scene_data = np.load("optimized_layout.npy")
        
        # 渲染户型图
        render_floorplan(scene_data, output_path="results/floorplan_realistic.png")
        print("户型图已保存到 results/floorplan_realistic.png")
        
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 