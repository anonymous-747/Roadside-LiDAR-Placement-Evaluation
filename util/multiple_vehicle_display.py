import numpy as np
import open3d as o3d
import sys
import math
#from pointcloud_visualization import filter_points_in_quadrilateral,calculate_box,project_points_to_ground

'''全局及雷达参数 (Global and LiDAR Parameters)
这些参数对于场景中的所有车辆都是通用的'''
LiDAR_height = 8
gamma=0.8
car_width=1.8
car_length=4.5
car_height=1.5
LiDAR_height=6


def _sign(p1, p2, p3):
    """
    计算点 p3 相对于向量 p1->p2 的位置。
    用于判断点在向量的哪一侧。
    使用叉积的 z 分量。
    """
    return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

def _is_inside_triangle(points_xy, v1, v2, v3):
    """
    Vectorized check if points are inside a triangle defined by v1, v2, v3.
    This implementation correctly handles the vectorization.

    参数:
    points_xy (np.ndarray): 形状为 (N, 2) 的点坐标数组。
    v1, v2, v3 (tuple or np.ndarray): 三角形的三个顶点坐标, e.g., (x, y)。

    返回:
    np.ndarray: 形状为 (N,) 的布尔数组，True 表示点在三角形内。
    """
    # 计算每个点相对于三角形三条边的叉积符号
    # 公式: sign = (x2-x1)*(y - y1) - (y2-y1)*(x - x1)
    
    # 边 v1 -> v2
    sign1 = (v2[0] - v1[0]) * (points_xy[:, 1] - v1[1]) - (v2[1] - v1[1]) * (points_xy[:, 0] - v1[0])
    
    # 边 v2 -> v3
    sign2 = (v3[0] - v2[0]) * (points_xy[:, 1] - v2[1]) - (v3[1] - v2[1]) * (points_xy[:, 0] - v2[0])
    
    # 边 v3 -> v1
    sign3 = (v1[0] - v3[0]) * (points_xy[:, 1] - v3[1]) - (v1[1] - v3[1]) * (points_xy[:, 0] - v3[0])

    # 如果一个点在三角形内部，那么它相对于三条边（按统一顺序）的符号应该全部相同。
    # (>= 和 <= 是为了包含位于边上的点)
    all_pos = (sign1 >= 0) & (sign2 >= 0) & (sign3 >= 0)
    all_neg = (sign1 <= 0) & (sign2 <= 0) & (sign3 <= 0)
    
    # 返回一个 (N,) 形状的布尔数组
    return all_pos | all_neg

def filter_points_in_quadrilateral(points, vertices):
    """
    从点云中滤出所有在指定四边形内的点。

    参数:
    points (np.ndarray): 输入的点云数据, 形状为 (N, D)，D>=2。
    vertices (np.ndarray): 四边形的4个顶点, 形状为 (4, 2)。
                           顶点必须按顺序（顺时针或逆时针）提供。

    返回:
    np.ndarray: 在四边形内部的点的数组。
    """
    if points.shape[1] < 2:
        raise ValueError("输入点云的维度必须至少为2 (包含 x 和 y)。")

    if vertices.shape != (4, 2):
        raise ValueError("必须提供4个二维顶点，形状为 (4, 2)。")

    # 我们只关心点的 x 和 y 坐标
    points_xy = points[:, :2]
    
    # 将四边形 (v1, v2, v3, v4) 分割成两个三角形: (v1, v2, v3) 和 (v1, v3, v4)
    v1, v2, v3, v4 = vertices
    
    # 判断点是否在第一个三角形内
    mask_triangle1 = _is_inside_triangle(points_xy, v1, v2, v3)
    
    # 判断点是否在第二个三角形内
    mask_triangle2 = _is_inside_triangle(points_xy, v1, v3, v4)

    mask_triangle3 = _is_inside_triangle(points_xy, v2, v3, v4)
    
    # 只要点在任意一个三角形内，就认为它在四边形内
    final_mask = mask_triangle1 | mask_triangle2 | mask_triangle3
    
    return final_mask


def calculate_box( x , y , theta, car_width=car_width, car_length=car_length, car_height=car_height, LiDAR_height=LiDAR_height,gamma=gamma):

    """根据给定的中心点坐标和旋转角度计算长方形的边界。
    参数
    x (float): 车辆中心点坐标。
    y (float): 车辆中心点坐标。
    theta (float): 车辆行驶方向和车辆中心点与雷达中心点连线夹角。
    返回:
    shadow1, shadow2,shadow3,linesets,car_positions,shadow_positions

    shadow1: shadow of the top of the car
    shadow2: shadow of the front of the car
    shadow3: shadow of the side of the car
    linesets: linesets for visualization
    car_positions: 4 vertex of the car,formed as
            1(closest points to the LiDAR)
          -   -
        -       2
      3       -
        -   -   
          4
    shadow_positions: 4 vertex of the shadow, formed same as that of car_positions
    """

    cos_car_rot = np.cos(theta)
    sin_car_rot = np.sin(theta)

    half_car_w = car_width / 2.0
    half_car_l = car_length / 2.0

    # Create the 4 corners in the box's local coordinate system (unrotated)
    # The box's "length" (x_length) is along its local x-axis
    # The box's "width" (y_length) is along its local y-axis
    local_car_corners = np.array([
        [ half_car_l,  half_car_w], # Front-right
        [ half_car_l, -half_car_w],  # Front-left
        [-half_car_l,  half_car_w], # Back-right
        [-half_car_l, -half_car_w], # Back-left
    ])

    rotation_car_matrix = np.array([
        [cos_car_rot, -sin_car_rot],
        [sin_car_rot,  cos_car_rot]
    ])

    # Rotate the local corners by multiplying with the rotation matrix
    # (N, 2) @ (2, 2) -> (N, 2)
    rotated_car_corners = local_car_corners @ rotation_car_matrix.T

    # Translate the rotated corners to the final box center

    world_car_corners = rotated_car_corners + np.array([x, y])
    
    car_pos1, car_pos2, car_pos3, car_pos4 = world_car_corners

    d1=np.sqrt(car_pos1[0]*car_pos1[0]+car_pos1[1]*car_pos1[1])
    d2=np.sqrt(car_pos2[0]*car_pos2[0]+car_pos2[1]*car_pos2[1])    
    d3=np.sqrt(car_pos3[0]*car_pos3[0]+car_pos3[1]*car_pos3[1])
    d4=np.sqrt(car_pos4[0]*car_pos4[0]+car_pos4[1]*car_pos4[1])
    flag = 0 # d1 min 1,d2 min 2 ,represents the closest points to the      
    if(d1 <= d2 and d1 <= d3 and d1 <= d4):
        flag = 1
    if(d2 <= d3 and d2 <= d4 and d2 <= d1):
        flag = 2
    if(d3 <= d4 and d3 <= d1 and d3 <= d2):
        flag = 3
    if(d4 <= d1 and d4 <= d2 and d4 <= d3):
        flag = 4
    car_positions =sort_car_position(world_car_corners,flag)
    car_pos1,car_pos2,car_pos3,car_pos4=car_positions
    def calaculate_shadow_position(car_pos,fixed_car_height,Lidar_height):
        x,y=car_pos
        d=np.sqrt(x*x+y*y)
        projected_point_distance = d*Lidar_height/(Lidar_height-fixed_car_height)
        ratio=projected_point_distance/d
        shadow_x=x*ratio
        shadow_y=y*ratio
        return [shadow_x,shadow_y]
    fixed_car_height=car_height*gamma
    shadow_pos1= calaculate_shadow_position(car_pos1,fixed_car_height,LiDAR_height)
    shadow_pos2= calaculate_shadow_position(car_pos2,fixed_car_height,LiDAR_height)
    shadow_pos3= calaculate_shadow_position(car_pos3,fixed_car_height,LiDAR_height)
    shadow_pos4= calaculate_shadow_position(car_pos4,fixed_car_height,LiDAR_height)

    shadow_positions=[shadow_pos1,shadow_pos2,shadow_pos3,shadow_pos4]
    linesets=draw_shadow(car_pos1, car_pos2, car_pos3, car_pos4,shadow_pos1,shadow_pos2,shadow_pos3,shadow_pos4)

    shadow1= np.array([shadow_pos1, shadow_pos2, shadow_pos3,shadow_pos4])#shadow1234 top of the car
    shadow2= np.array([car_pos1, shadow_pos1, car_pos2,shadow_pos2])#front of the car
    shadow3= np.array([car_pos1, shadow_pos1, car_pos3,shadow_pos3])#side of the car

    return shadow1, shadow2,shadow3,linesets,car_positions,shadow_positions


def sort_car_position(car_positions,flag):
    '''
    car1,car2,car3,car4=car_positions
    input 2 - 1
          -   -
          -   -
          4 - 3
    output
            1
          -   -
        -       2
      3       -
        -   -   
          4
    '''
    car1,car2,car3,car4=car_positions
    if flag==1:
        car_positions=car1,car2,car3,car4
    if flag==2:
        car_positions=car2,car1,car4,car3
    if flag==3:
        car_positions=car3,car4,car1,car2
    if flag==4:
        car_positions=car4,car3,car2,car1
    return car_positions


def project_points_to_ground(points_to_process, lidar_height_param):
    """
    将输入点云投影到 z=0 的地平面上。
    不符合投影条件的点将保持其原始坐标。

    参数:
    points_to_process (numpy.ndarray): 输入的点云数据，形状为 (N, 3) 或 (N, 4)。
    lidar_height_param (float): 激光雷达的高度。

    返回:
    numpy.ndarray: 处理后的完整点云坐标，形状与输入点云的 (N, 3) 部分相同。
                   其中符合条件的点被投影，不符合条件的点保持原样。
    """
    # ------------------ 关键修改 1: 初始化为原始数据的副本 ------------------
    # 这样，不被投影的点就会保留其原始坐标。
    # 我们只处理 x, y, z，所以取前3列。
    projected_coords = points_to_process[:, :3].copy()
    
    # 从副本中提取 x, y, z
    x, y, z = projected_coords[:, 0], projected_coords[:, 1], projected_coords[:, 2]

    # 为了避免除以零的错误，我们只处理 z < lidar_height_param 的点
    valid_mask = z < lidar_height_param
    
    # 如果没有有效点，直接返回原始点云的副本
    if not np.any(valid_mask):
        print("警告: 没有在激光雷达高度以下的点，所有点都保持原样。")
        return projected_coords

    # 提取有效点进行计算
    x_valid, y_valid, z_valid = x[valid_mask], y[valid_mask], z[valid_mask]
    
    # 计算水平距离
    d = np.sqrt(x_valid**2 + y_valid**2)
    
    # 根据相似三角形原理计算投影后的水平距离
    # 使用 lidar_height_param - z_valid 来确保分母不为零或负数
    zero_height_distance = d * lidar_height_param / (lidar_height_param - z_valid)
    
    # 计算比例因子 ratio = new_distance / old_distance
    # 使用 np.divide 来安全地处理 d=0 的情况 (点在LiDAR正下方)
    ratio = np.divide(zero_height_distance, d, out=np.ones_like(d), where=d!=0)
    
    # ---- 更新被投影点的坐标 ----
    # 只对 `valid_mask` 为 True 的行进行操作
    projected_coords[valid_mask, 0] = x_valid * ratio
    projected_coords[valid_mask, 1] = y_valid * ratio
    projected_coords[valid_mask, 2] = 0 # 将这些点的 z 坐标设置为 0

    # -------------------- 关键修改 2: 返回完整的数组 --------------------
    # 不再使用 [valid_mask] 进行过滤
    return projected_coords


def create_bounding_box(center_x, center_y, length, width, angle_rad, color=[0, 0, 1], add_arrow=False):
    """
    创建表示旋转长方体的 Open3D LineSet 对象。
    Creates an Open3D LineSet object representing a rotated rectangular solid.
    """
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])
    
    half_l, half_w = length / 2, width / 2
    box_corners = np.array([[-half_l, -half_w], [half_l, -half_w], [half_l, half_w], [-half_l, half_w]])
    box_lines = [[0, 1], [1, 2], [2, 3], [3, 0]]
    
    points = box_corners
    lines = box_lines
    
    if add_arrow:
        arrow_length = length * 0.4
        arrow_points = np.array([
            [0, 0],
            [half_l, 0],
            [half_l - arrow_length, half_w * 0.3],
            [half_l - arrow_length, -half_w * 0.3]
        ])
        arrow_lines = [[4, 5], [5, 6], [5, 7]]
        points = np.vstack([points, arrow_points])
        lines.extend(arrow_lines)

    rotated_points = np.dot(points, R.T) + [center_x, center_y]
    points_3d = np.hstack([rotated_points, np.zeros((len(rotated_points), 1))])
    
    colors = [color for _ in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points_3d),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    return line_set

def draw_line_between_points(p1, p2):
    """
    Draws a line in Open3D between two 2D points.

    Args:
        p1 (tuple): The (x, y) coordinates of the first point.
        p2 (tuple): The (x, y) coordinates of the second point.
    """
    # 1. Define the 3D points (vertices) of the line
    # We add a z-coordinate of 0 to the 2D points.
    points = np.array([
        [p1[0], p1[1], 0],
        [p2[0], p2[1], 0]
    ])

    # 2. Define the connections between the points
    # This connects the first point (index 0) to the second point (index 1).
    lines = np.array([
        [0, 1]
    ])
    # 3. Create the LineSet object
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines)
    )
    colors = np.array([[1, 0, 0]]) # Red
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def draw_shadow(c1,c2,c3,c4,s1,s2,s3,s4):
    """
    Draws mulitple lines in Open3D between two 2D points.

    Args:
        cn (tuple): The (x, y) coordinates of the car_position_n.
        sn (tuple): The (x, y) coordinates of the shadow_position_n.
    """
    line_set1=draw_line_between_points(c1, c2)
    line_set2=draw_line_between_points(c1, s3)
    line_set3=draw_line_between_points(c2, s2)
    line_set4=draw_line_between_points(c3, s3)
    line_set5=draw_line_between_points(s3, s4)
    line_set6=draw_line_between_points(s2, s4)
    line_sets=line_set1+line_set2+line_set3+line_set4+line_set5+line_set6
    return line_sets

