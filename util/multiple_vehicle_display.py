import numpy as np
import open3d as o3d
import sys
import math
# from pointcloud_visualization import filter_points_in_quadrilateral, calculate_box, project_points_to_ground

"""
Global and LiDAR parameters.
These parameters are shared by all vehicles in the scene.
"""

gamma = 0.8          # Scaling factor for the "effective" car height when computing shadow
car_width = 1.8      # Width of the vehicle (meters)
car_length = 4.5     # Length of the vehicle (meters)
car_height = 1.5     # Height of the vehicle (meters)
LiDAR_height = 6     # LiDAR height (meters) used in calculations


def _sign(p1, p2, p3):
    """
    Compute the signed area (cross product z-component) of the triangle formed by points p1, p2, p3.

    This effectively tells which side of the directed edge p1 -> p2 the point p3 lies on.

    Args:
        p1, p2, p3: 2D points in (x, y) form.

    Returns:
        float: The signed value of the cross product. A positive or negative
               sign indicates different sides; zero means collinear.
    """
    return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])


def _is_inside_triangle(points_xy, v1, v2, v3):
    """
    Vectorized check whether multiple 2D points lie inside a triangle.

    The triangle is defined by vertices v1, v2, v3 (in 2D), and the function
    evaluates all points in 'points_xy' at once.

    Args:
        points_xy (np.ndarray): Array of shape (N, 2) containing point coordinates (x, y).
        v1, v2, v3 (tuple or np.ndarray): The three vertices of the triangle, each as (x, y).

    Returns:
        np.ndarray: Boolean array of shape (N,), where True means the point is inside
                    or on the boundary of the triangle.
    """
    # Edge v1 -> v2
    sign1 = (v2[0] - v1[0]) * (points_xy[:, 1] - v1[1]) - (v2[1] - v1[1]) * (points_xy[:, 0] - v1[0])

    # Edge v2 -> v3
    sign2 = (v3[0] - v2[0]) * (points_xy[:, 1] - v2[1]) - (v3[1] - v2[1]) * (points_xy[:, 0] - v2[0])

    # Edge v3 -> v1
    sign3 = (v1[0] - v3[0]) * (points_xy[:, 1] - v3[1]) - (v1[1] - v3[1]) * (points_xy[:, 0] - v3[0])

    # For a point inside the triangle, the signs with respect to all three edges
    # should be either all non-negative or all non-positive.
    all_pos = (sign1 >= 0) & (sign2 >= 0) & (sign3 >= 0)
    all_neg = (sign1 <= 0) & (sign2 <= 0) & (sign3 <= 0)

    return all_pos | all_neg


def filter_points_in_quadrilateral(points, vertices):
    """
    Filter all points in a point cloud that lie inside a given quadrilateral.

    The quadrilateral is split into triangles, and a point is considered inside the
    quadrilateral if it is inside any of these triangles.

    Args:
        points (np.ndarray):
            Input point cloud of shape (N, D), where D >= 2.
            Only the first two dimensions (x, y) are used for the test.
        vertices (np.ndarray):
            Array of 4 vertices of the quadrilateral with shape (4, 2).
            The vertices must be provided in order (clockwise or counterclockwise).

    Returns:
        np.ndarray:
            A boolean mask of shape (N,). True indicates the point lies inside the quadrilateral.
    """
    if points.shape[1] < 2:
        raise ValueError("Input point cloud must have at least 2 dimensions (contain x and y).")

    if vertices.shape != (4, 2):
        raise ValueError("You must provide exactly 4 vertices with shape (4, 2).")

    # Only x, y coordinates are used
    points_xy = points[:, :2]

    # Quadrilateral vertices (v1, v2, v3, v4)
    v1, v2, v3, v4 = vertices

    # Check point-in-triangle for three triangle combinations that cover the quadrilateral
    mask_triangle1 = _is_inside_triangle(points_xy, v1, v2, v3)
    mask_triangle2 = _is_inside_triangle(points_xy, v1, v3, v4)
    mask_triangle3 = _is_inside_triangle(points_xy, v2, v3, v4)

    # A point inside any triangle is considered inside the quadrilateral
    final_mask = mask_triangle1 | mask_triangle2 | mask_triangle3

    return final_mask


def calculate_box(
    x,
    y,
    theta,
    car_width=car_width,
    car_length=car_length,
    car_height=car_height,
    LiDAR_height=LiDAR_height,
    gamma=gamma
):
    """
    Compute the footprint and "shadow" of a vehicle based on its pose and LiDAR geometry.

    The function:
      1. Builds a rectangle representing the car in the xy-plane.
      2. Rotates and translates it to the given center (x, y) with heading theta.
      3. Sorts the four corners so that:
         - car_pos1 is the closest vertex to the LiDAR.
         - The order is consistent as:
               1 (closest to LiDAR)
             -   -
           -       2
         3       -
           -   -
               4
      4. Projects the car corners down to a ground-plane "shadow" based on LiDAR height.

    Args:
        x (float): Car center x-coordinate.
        y (float): Car center y-coordinate.
        theta (float): Car heading angle (radians). This is both the vehicle heading and the
                       angle between the car center and the LiDAR center.
        car_width (float): Width of the car.
        car_length (float): Length of the car.
        car_height (float): Height of the car.
        LiDAR_height (float): Height of the LiDAR.
        gamma (float): Factor to scale car_height when computing "effective" height for shadow.

    Returns:
        shadow1 (np.ndarray): 4x2 array, projected positions (x, y) of all 4 car corners
                              (shadow of the top of the car).
        shadow2 (np.ndarray): 4x2 array, positions of [car_pos1, shadow_pos1, car_pos2, shadow_pos2],
                              representing the front face and its shadow.
        shadow3 (np.ndarray): 4x2 array, positions of [car_pos1, shadow_pos1, car_pos3, shadow_pos3],
                              representing the side face and its shadow.
        linesets (o3d.geometry.LineSet): Open3D line objects for visualizing car–shadow edges.
        car_positions (list): Sorted list [car_pos1, car_pos2, car_pos3, car_pos4].
        shadow_positions (list): List [shadow_pos1, shadow_pos2, shadow_pos3, shadow_pos4] of
                                 projected corner positions on the ground.
    """

    # Precompute sine/cosine of heading angle
    cos_car_rot = np.cos(theta)
    sin_car_rot = np.sin(theta)

    half_car_w = car_width / 2.0
    half_car_l = car_length / 2.0

    # Define the 4 corners in the local (unrotated) car frame
    # Length is along local x-axis, width is along local y-axis
    local_car_corners = np.array([
        [ half_car_l,  half_car_w],   # Front-right
        [ half_car_l, -half_car_w],   # Front-left
        [-half_car_l,  half_car_w],   # Back-right
        [-half_car_l, -half_car_w],   # Back-left
    ])

    # Rotation matrix for car orientation
    rotation_car_matrix = np.array([
        [cos_car_rot, -sin_car_rot],
        [sin_car_rot,  cos_car_rot]
    ])

    # Rotate the local corners
    rotated_car_corners = local_car_corners @ rotation_car_matrix.T

    # Translate rotated corners to world coordinates
    world_car_corners = rotated_car_corners + np.array([x, y])

    car_pos1, car_pos2, car_pos3, car_pos4 = world_car_corners

    # Compute distance from LiDAR (assumed at origin) to each corner
    d1 = np.sqrt(car_pos1[0] ** 2 + car_pos1[1] ** 2)
    d2 = np.sqrt(car_pos2[0] ** 2 + car_pos2[1] ** 2)
    d3 = np.sqrt(car_pos3[0] ** 2 + car_pos3[1] ** 2)
    d4 = np.sqrt(car_pos4[0] ** 2 + car_pos4[1] ** 2)

    # Determine which corner is closest to the LiDAR
    flag = 0  # 1,2,3,4 indicates which corner is closest
    if d1 <= d2 and d1 <= d3 and d1 <= d4:
        flag = 1
    if d2 <= d3 and d2 <= d4 and d2 <= d1:
        flag = 2
    if d3 <= d4 and d3 <= d1 and d3 <= d2:
        flag = 3
    if d4 <= d1 and d4 <= d2 and d4 <= d3:
        flag = 4

    # Sort corners into a consistent order based on the closest corner
    car_positions = sort_car_position(world_car_corners, flag)
    car_pos1, car_pos2, car_pos3, car_pos4 = car_positions

    def calaculate_shadow_position(car_pos, fixed_car_height, Lidar_height):
        """
        Compute the ground-plane shadow of a car corner given LiDAR and car heights.

        Args:
            car_pos (array-like): 2D position (x, y) of a car corner.
            fixed_car_height (float): Effective height of the corner to project.
            Lidar_height (float): Height of the LiDAR.

        Returns:
            [shadow_x, shadow_y]: 2D ground-point of the projected shadow.
        """
        x, y = car_pos
        d = np.sqrt(x * x + y * y)

        # Distance from LiDAR to ground projection using similar triangles
        projected_point_distance = d * Lidar_height / (Lidar_height - fixed_car_height)
        ratio = projected_point_distance / d

        shadow_x = x * ratio
        shadow_y = y * ratio
        return [shadow_x, shadow_y]

    # Use gamma * car_height as the effective height to project
    fixed_car_height = car_height * gamma

    shadow_pos1 = calaculate_shadow_position(car_pos1, fixed_car_height, LiDAR_height)
    shadow_pos2 = calaculate_shadow_position(car_pos2, fixed_car_height, LiDAR_height)
    shadow_pos3 = calaculate_shadow_position(car_pos3, fixed_car_height, LiDAR_height)
    shadow_pos4 = calaculate_shadow_position(car_pos4, fixed_car_height, LiDAR_height)

    shadow_positions = [shadow_pos1, shadow_pos2, shadow_pos3, shadow_pos4]

    # Linesets for visualization between car corners and their shadows
    linesets = draw_shadow(
        car_pos1, car_pos2, car_pos3, car_pos4,
        shadow_pos1, shadow_pos2, shadow_pos3, shadow_pos4
    )

    # shadow1: full "top" shadow polygon
    shadow1 = np.array([shadow_pos1, shadow_pos2, shadow_pos3, shadow_pos4])

    # shadow2: front edge and its shadow
    shadow2 = np.array([car_pos1, shadow_pos1, car_pos2, shadow_pos2])

    # shadow3: side edge and its shadow
    shadow3 = np.array([car_pos1, shadow_pos1, car_pos3, shadow_pos3])

    return shadow1, shadow2, shadow3, linesets, car_positions, shadow_positions


def sort_car_position(car_positions, flag):
    """
    Reorder car corner positions into a canonical order.

    Input corner order (example):
        2 - 1
        -   -
        -   -
        4 - 3

    Output order (desired):
            1
          -   -
        -       2
      3       -
        -   -
            4

    Args:
        car_positions: Iterable of 4 points (car1, car2, car3, car4).
        flag (int): Index (1–4) of the corner that is closest to the LiDAR.

    Returns:
        tuple: (car_pos1, car_pos2, car_pos3, car_pos4) in the normalized order.
    """
    car1, car2, car3, car4 = car_positions

    if flag == 1:
        car_positions = car1, car2, car3, car4
    if flag == 2:
        car_positions = car2, car1, car4, car3
    if flag == 3:
        car_positions = car3, car4, car1, car2
    if flag == 4:
        car_positions = car4, car3, car2, car1

    return car_positions


def project_points_to_ground(points_to_process, lidar_height_param):
    """
    Project 3D points onto the ground plane z = 0 based on LiDAR height.

    Points with z >= lidar_height_param are not projected (their coordinates are kept).
    For points with z < lidar_height_param, we use similar triangles to compute
    where the ray from the LiDAR through the point intersects the ground plane.

    Args:
        points_to_process (np.ndarray):
            Input point cloud of shape (N, 3) or (N, 4). Only the first 3 columns (x, y, z) are used.
        lidar_height_param (float):
            Height of the LiDAR above the ground.

    Returns:
        np.ndarray:
            Array of shape (N, 3) containing the processed coordinates.
            Projected points have z=0; others remain unchanged.
    """
    # Make a copy so un-projected points retain their original coordinates
    projected_coords = points_to_process[:, :3].copy()

    x, y, z = projected_coords[:, 0], projected_coords[:, 1], projected_coords[:, 2]

    # Only project points below the LiDAR height
    valid_mask = z < lidar_height_param

    # If no valid points, return as is
    if not np.any(valid_mask):
        print("Warning: No points are below the LiDAR height; all points remain unchanged.")
        return projected_coords

    x_valid, y_valid, z_valid = x[valid_mask], y[valid_mask], z[valid_mask]

    # Horizontal distance from LiDAR
    d = np.sqrt(x_valid ** 2 + y_valid ** 2)

    # Distance to ground intersection along the ray using similar triangles:
    #   zero_height_distance = d * lidar_height / (lidar_height - z)
    zero_height_distance = d * lidar_height_param / (lidar_height_param - z_valid)

    # Ratio between new horizontal distance and old distance
    # Use safe division to handle d=0
    ratio = np.divide(zero_height_distance, d, out=np.ones_like(d), where=d != 0)

    # Update only the valid points
    projected_coords[valid_mask, 0] = x_valid * ratio
    projected_coords[valid_mask, 1] = y_valid * ratio
    projected_coords[valid_mask, 2] = 0  # Ground plane

    return projected_coords


def create_bounding_box(center_x, center_y, length, width, angle_rad, color=[0, 0, 1], add_arrow=False):
    """
    Create an Open3D LineSet representing a rotated rectangle (optionally with a heading arrow).

    The rectangle is assumed to lie on the ground plane (z = 0).

    Args:
        center_x (float): Center x-coordinate of the box.
        center_y (float): Center y-coordinate of the box.
        length (float): Length of the box (along local x-axis).
        width (float): Width of the box (along local y-axis).
        angle_rad (float): Rotation angle in radians (about z-axis).
        color (list[float]): RGB color of the lines, values in [0, 1].
        add_arrow (bool): If True, add an arrow to indicate heading.

    Returns:
        o3d.geometry.LineSet: LineSet object representing the box (and arrow if enabled).
    """
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])

    half_l, half_w = length / 2, width / 2

    # Base rectangle corners in local frame
    box_corners = np.array([
        [-half_l, -half_w],
        [ half_l, -half_w],
        [ half_l,  half_w],
        [-half_l,  half_w]
    ])
    box_lines = [[0, 1], [1, 2], [2, 3], [3, 0]]

    points = box_corners
    lines = box_lines

    if add_arrow:
        # Arrow points in local frame, starting at the box center and pointing along +x
        arrow_length = length * 0.4
        arrow_points = np.array([
            [0, 0],                            # 4: center
            [half_l, 0],                       # 5: arrow tip
            [half_l - arrow_length, half_w * 0.3],   # 6: upper arrow wing
            [half_l - arrow_length, -half_w * 0.3],  # 7: lower arrow wing
        ])
        arrow_lines = [[4, 5], [5, 6], [5, 7]]

        points = np.vstack([points, arrow_points])
        lines.extend(arrow_lines)

    # Rotate and translate to world frame
    rotated_points = np.dot(points, R.T) + [center_x, center_y]

    # Add z=0 for all points to make them 3D
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
    Draw a single line in Open3D between two 2D points (with z fixed to 0).

    Args:
        p1 (tuple or list): (x, y) coordinates of the first point.
        p2 (tuple or list): (x, y) coordinates of the second point.

    Returns:
        o3d.geometry.LineSet: LineSet object representing the line.
    """
    # Convert 2D points to 3D by adding z = 0
    points = np.array([
        [p1[0], p1[1], 0],
        [p2[0], p2[1], 0]
    ])

    # Single segment: connect point 0 to point 1
    lines = np.array([[0, 1]])

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines)
    )

    # Color the line red
    colors = np.array([[1, 0, 0]])
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set


def draw_shadow(c1, c2, c3, c4, s1, s2, s3, s4):
    """
    Draw multiple line segments between car corners and their projected shadow points.

    Args:
        c1, c2, c3, c4: 2D positions (x, y) of the car's four corners.
        s1, s2, s3, s4: 2D positions (x, y) of the corresponding shadow points.

    Returns:
        o3d.geometry.LineSet: Combined LineSet of all connecting edges.
    """
    # Lines between car and shadow points and between shadow points
    line_set1 = draw_line_between_points(c1, c2)
    line_set2 = draw_line_between_points(c1, s3)
    line_set3 = draw_line_between_points(c2, s2)
    line_set4 = draw_line_between_points(c3, s3)
    line_set5 = draw_line_between_points(s3, s4)
    line_set6 = draw_line_between_points(s2, s4)

    # LineSets are additive in Open3D, which merges their geometries
    line_sets = line_set1 + line_set2 + line_set3 + line_set4 + line_set5 + line_set6
    return line_sets
