import numpy as np
import open3d as o3d
import sys # 导入sys模块以处理错误
import math
from .multiple_vehicle_display import calculate_box, project_points_to_ground,filter_points_in_quadrilateral
from .math_toolbox import calculate_integral_fixed,calculate_integral
gamma=0.8
car_width=1.8
car_length=4.5
car_height=1.5
LiDAR_height=6
IOU_threshold=0.5
n_correlation_value=3


# 车辆中心点坐标和方向角度（单位：弧度）
car_x=-15
car_y=50.01
car_theta=1.57 # 车辆中心点坐标和方向角度（单位：弧度）

def calculate_distribution_and_IOU(front_distribution,side_distribution,IOU_threshold,n_correlation=n_correlation_value):
    front_num=len(front_distribution)
    side_num=len(side_distribution)
    front_num=int(math.ceil(front_num/n_correlation))
    side_num=int(math.ceil(side_num/n_correlation))
    flag_front=0
    flag_side=0
    Confidence_score=front_num+side_num
    if front_num > 0:
        flag_front=1
        max_front=max(front_distribution)
        min_front=min(front_distribution)
    if side_num > 0:
        flag_side=1
        max_side=max(side_distribution)
        min_side=min(side_distribution)

    if flag_front==1 and flag_side == 1:
        value,error=calculate_integral_fixed(front_num, side_num, max_front, max_side,IOU_threshold)
        return value,Confidence_score
        # calculate IOU

    if flag_front==1 and flag_side == 0:
        R=max_front-min_front
        mid=(max_front+min_front)/2
        value,error=calculate_integral(R,front_num,mid,IOU=IOU_threshold)
        return value,Confidence_score
        
    if flag_front==0 and flag_side == 1:
        R=max_side-min_side
        mid=(max_side+min_side)/2
        value,error=calculate_integral(R,side_num,mid,IOU=IOU_threshold)
        return value,Confidence_score

    if flag_front==0 and flag_side == 0:
        return 0,0

def project_points_to_line(points,pos1,pos2):
    distribution=[]
    #height_distribution=[]
    for point in points:
        x1,y1=point[0],point[1]
        height=point[2]
        x2,y2=pos1
        x3,y3=pos2
        value=abs(x2*y1-x1*y2)/(abs(x2*y1-x1*y2)+abs(x1*y3-x3*y1))
        distribution.append(value)
        #height_distribution.append(height)
    return distribution

def project_top_points_to_line(points,pos1,pos2,pos3,pos4):
    distribution=[]
    # points_number=points.shape[0]
    for p in points:
        lp1 = np.asarray(pos1)
        lp2 = np.asarray(pos2)
        lp3 = np.asarray(pos3)
        lp4 = np.asarray(pos4)
        point=p[:2]
        # 1. Handle the edge case where the line is just a point
        line_vec = lp2 - lp1
        line_mag = np.linalg.norm(line_vec)
        line_vec2 = lp4 - lp3
        line_mag2 = np.linalg.norm(line_vec2)
        # 2. Calculate the numerator of the distance formula
        # This is the magnitude of the cross product of two vectors:
        #   - Vector from line_p1 to line_p2
        #   - Vector from line_p1 to the point
        # In 2D, np.cross returns the scalar magnitude of the z-component.
        numerator = np.abs(np.cross(line_vec, lp1 - point))
        numerator2 = np.abs(np.cross(line_vec2, lp3 - point))

        # 3. Divide by the length of the line segment to get the perpendicular distance
        distance = numerator / line_mag    
        distance2 = numerator2 / line_mag2 
        value= distance/(distance+distance2)
        distribution.append(value)
    return distribution

def points_information(filtered_points,shadow1,shadow2,shadow3,IOU_threshold,n_correlation=n_correlation_value):
# remain to add 
    filtered_mask_top=filter_points_in_quadrilateral(filtered_points, shadow1)
    filtered_mask_front=filter_points_in_quadrilateral(filtered_points, shadow2)
    filtered_mask_side=filter_points_in_quadrilateral(filtered_points, shadow3)
    points_top=filtered_points[filtered_mask_top]
    points_front=filtered_points[filtered_mask_front]
    points_side=filtered_points[filtered_mask_side]

    shadow_pos1,shadow_pos2,shadow_pos3,shadow_pos4=shadow1
    front_distribution1=project_points_to_line(points_front,shadow_pos1,shadow_pos2)
    side_distribution1=project_points_to_line(points_side,shadow_pos1,shadow_pos3)
    front_distribution2=project_top_points_to_line(points_top,shadow_pos1,shadow_pos3,shadow_pos2,shadow_pos4)
    side_distribution2=project_top_points_to_line(points_top,shadow_pos1,shadow_pos2,shadow_pos3,shadow_pos4)
    front_distribution =np.concatenate((front_distribution1, front_distribution2))
    side_distribution = np.concatenate((side_distribution1, side_distribution2))
    # height_distribution = np.concatenate((height_distribution1, height_distribution2))        
    information_score,Confidence_score=calculate_distribution_and_IOU(front_distribution,side_distribution,IOU_threshold=IOU_threshold,n_correlation=n_correlation)
    if math.isnan(information_score):
        information_score = 1

    return information_score,Confidence_score


def calculate_score(points, car_x, car_y, car_theta,LiDAR_height=LiDAR_height,car_height=car_height,
                    car_length=car_length, car_width=car_width,
                    gamma=gamma,IOU_threshold=IOU_threshold,n_correlation=n_correlation_value):
    """
    计算点云中在旋转矩形内的点的索引。

    参数:
    points (np.ndarray): 点云数据，形状为 (N, 4)。
    car_x (float): 车辆中心点的 x 坐标。
    car_y (float): 车辆中心点的 y 坐标。
    car_length (float): 车辆的长度。
    car_width (float): 车辆的宽度。
    angle_radians (float): 车辆相对于世界坐标系的旋转角度（弧度）。

    返回:
    np.ndarray: 该点对应角度的评分
    """

    all_points = points
    projected_all_points=project_points_to_ground(all_points,lidar_height_param=LiDAR_height)
    # 2. 定义筛选区域参数 (这部分与您的代码相同)
    # (可选) 给投影后的点云上色，以便区分。这里我们设置为红色 [R, G, B]。
    shadow1,shadow2,shadow3,linesets,car_positions,shadow_positions=calculate_box(car_x, car_y, car_theta,car_height=car_height,car_width=car_width,car_length=car_length,LiDAR_height=LiDAR_height,gamma=gamma)


    filtered_mask1=filter_points_in_quadrilateral(projected_all_points, shadow1)
    filtered_mask2=filter_points_in_quadrilateral(projected_all_points, shadow2)
    filtered_mask3=filter_points_in_quadrilateral(projected_all_points, shadow3)
    mask_sum=filtered_mask1 | filtered_mask2 | filtered_mask3

    filtered_points=projected_all_points[mask_sum]
    


    total_point_num=filtered_points.shape[0]
    #return total_point_num
    information_score,confidence_score=points_information(filtered_points,shadow1,shadow2,shadow3,IOU_threshold=IOU_threshold,n_correlation=n_correlation)

    
    return information_score,confidence_score


