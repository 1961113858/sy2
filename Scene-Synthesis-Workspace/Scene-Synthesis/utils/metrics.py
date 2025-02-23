import torch
import numpy as np
from scipy.spatial.distance import cdist

def compute_metrics(input_scene, output_scene):
    """
    计算评估指标
    Args:
        input_scene: 输入场景 [B, N, D] 或 [N, D]
        output_scene: 输出场景 [B, N, D] 或 [N, D]
    Returns:
        metrics: 评估指标字典
    """
    # 确保输入是2D数组
    if input_scene.ndim == 3:
        input_scene = input_scene.reshape(-1, input_scene.shape[-1])
    if output_scene.ndim == 3:
        output_scene = output_scene.reshape(-1, output_scene.shape[-1])
    
    # 计算Chamfer距离
    chamfer_dist = compute_chamfer_distance(
        input_scene[:, :3],  # 只使用位置信息
        output_scene[:, :3]
    )
    
    # 计算旋转误差
    rotation_error = compute_rotation_error(
        input_scene[:, 3:6],
        output_scene[:, 3:6]
    )
    
    # 计算尺寸误差
    scale_error = compute_scale_error(
        input_scene[:, 6:9],
        output_scene[:, 6:9]
    )
    
    return {
        "chamfer_distance": chamfer_dist,
        "rotation_error": rotation_error,
        "scale_error": scale_error
    }
    
def compute_chamfer_distance(x, y):
    """
    计算Chamfer距离
    Args:
        x: 第一组点 [N, 3]
        y: 第二组点 [N, 3]
    Returns:
        distance: Chamfer距离
    """
    # 计算点对距离矩阵
    D = cdist(x, y)
    
    # 计算最近点距离
    min_d_xy = np.min(D, axis=1)
    min_d_yx = np.min(D, axis=0)
    
    # 计算Chamfer距离
    chamfer_dist = np.mean(min_d_xy) + np.mean(min_d_yx)
    return chamfer_dist
    
def compute_rotation_error(r1, r2):
    """
    计算旋转误差
    Args:
        r1: 第一组旋转 [N, 3]
        r2: 第二组旋转 [N, 3]
    Returns:
        error: 平均旋转误差
    """
    return np.mean(np.abs(r1 - r2))
    
def compute_scale_error(s1, s2):
    """
    计算尺寸误差
    Args:
        s1: 第一组尺寸 [N, 3]
        s2: 第二组尺寸 [N, 3]
    Returns:
        error: 平均尺寸误差
    """
    return np.mean(np.abs(s1 - s2))

def compute_coverage(input_scene, output_scene, threshold=0.1):
    """
    计算场景覆盖率
    Args:
        input_scene: 输入场景 [N, 9]
        output_scene: 输出场景 [M, 9]
        threshold: 距离阈值
    Returns:
        coverage: 场景覆盖率
    """
    # 提取位置
    input_pos = input_scene[:, :3]
    output_pos = output_scene[:, :3]
    
    # 计算距离矩阵
    D = cdist(input_pos, output_pos)
    
    # 计算覆盖点数量
    covered = np.min(D, axis=1) < threshold
    coverage = np.mean(covered)
    
    return coverage
