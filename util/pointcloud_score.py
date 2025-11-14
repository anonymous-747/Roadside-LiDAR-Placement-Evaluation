import numpy as np
import open3d as o3d
import sys  # Import sys in case we want to handle errors or debug
import math

from .multiple_vehicle_display import (
    calculate_box,
    project_points_to_ground,
    filter_points_in_quadrilateral,
)
from .math_toolbox import calculate_integral_fixed, calculate_integral

# ------------------------------
# Global configuration
# ------------------------------
gamma = 0.8
car_width = 1.8
car_length = 4.5
car_height = 1.5
LiDAR_height = 6
IOU_threshold = 0.5
n_correlation_value = 3

# Example vehicle pose (center position and yaw angle in radians)
car_x = -15
car_y = 50.01
car_theta = 1.57  # Vehicle heading angle (radians)


def calculate_distribution_and_IOU(
    front_distribution,
    side_distribution,
    IOU_threshold,
    n_correlation=n_correlation_value,
):
    """
    Compute the information score based on point distributions on the front and side.

    The idea:
      1. Downsample the number of points (front/side) by a factor n_correlation,
         which roughly represents how many points are considered to be
         correlated / grouped.
      2. Depending on whether we have both front and side data or only one of them,
         call different integral functions from math_toolbox to evaluate the
         probability / score that the observed distributions satisfy an IOU-based condition.
      3. Return the computed information score plus a simple confidence score,
         which is the (downsampled) number of front and side points.

    Args:
        front_distribution (array-like): Scalar values in [0, 1] describing
            how points project along front-related lines.
        side_distribution (array-like): Scalar values in [0, 1] describing
            how points project along side-related lines.
        IOU_threshold (float): IOU threshold used inside the integrals.
        n_correlation (int): Grouping factor for points. The effective number
            of "independent" points is ceil(num_points / n_correlation).

    Returns:
        (information_score, confidence_score):
            information_score (float): Result from the integrals (probability-like).
            confidence_score (int): Effective count of independent front + side samples.
    """
    front_num = len(front_distribution)
    side_num = len(side_distribution)

    # Approximate number of independent samples (ceil to avoid zero)
    front_num = int(math.ceil(front_num / n_correlation))
    side_num = int(math.ceil(side_num / n_correlation))

    flag_front = 0
    flag_side = 0
    confidence_score = front_num + side_num

    if front_num > 0:
        flag_front = 1
        max_front = max(front_distribution)
        min_front = min(front_distribution)

    if side_num > 0:
        flag_side = 1
        max_side = max(side_distribution)
        min_side = min(side_distribution)

    # Case 1: We have both front and side distributions
    if flag_front == 1 and flag_side == 1:
        value, error = calculate_integral_fixed(
            front_num, side_num, max_front, max_side, IOU_threshold
        )
        return value, confidence_score

    # Case 2: Only front distribution is available
    if flag_front == 1 and flag_side == 0:
        R = max_front - min_front
        mid = (max_front + min_front) / 2.0
        value, error = calculate_integral(R, front_num, mid, IOU=IOU_threshold)
        return value, confidence_score

    # Case 3: Only side distribution is available
    if flag_front == 0 and flag_side == 1:
        R = max_side - min_side
        mid = (max_side + min_side) / 2.0
        value, error = calculate_integral(R, side_num, mid, IOU=IOU_threshold)
        return value, confidence_score

    # Case 4: No points at all
    if flag_front == 0 and flag_side == 0:
        return 0, 0


def project_points_to_line(points, pos1, pos2):
    """
    Project points to a normalized 1D distribution based on two endpoints.

    The function uses an area-based ratio involving the two line endpoints (pos1, pos2)
    and each point. The exact geometric meaning is not a standard distance, but
    the output is a scalar in (0, 1) that can be treated as a position-like
    feature along a virtual axis.

    Args:
        points (np.ndarray): Array of shape (N, 3) or (N, 4). Only x, y are used.
        pos1 (tuple/list): First 2D point (x2, y2) defining the line.
        pos2 (tuple/list): Second 2D point (x3, y3) defining the line.

    Returns:
        list[float]: A list of scalar values derived from each point w.r.t. the line.
    """
    distribution = []
    # height_distribution = []  # If needed, we could also track heights

    for point in points:
        x1, y1 = point[0], point[1]
        height = point[2]

        x2, y2 = pos1
        x3, y3 = pos2

        # Using a ratio of area-like terms as a pseudo "projection" in [0, 1]
        numerator = abs(x2 * y1 - x1 * y2)
        denominator = numerator + abs(x1 * y3 - x3 * y1)
        value = numerator / denominator if denominator != 0 else 0.0

        distribution.append(value)
        # height_distribution.append(height)

    return distribution


def project_top_points_to_line(points, pos1, pos2, pos3, pos4):
    """
    Project points to a normalized 1D distribution using distances to two segments.

    This function:
      1. Computes the perpendicular distance from each point to segment (pos1-pos2).
      2. Computes the distance to another segment (pos3-pos4).
      3. Returns distance1 / (distance1 + distance2), which lies in (0, 1).

    This gives a relative measure of "closeness" of each point to one segment
    versus the other.

    Args:
        points (np.ndarray): Array of shape (N, 3) or (N, 4). Only x, y are used.
        pos1, pos2, pos3, pos4 (tuple/list): 2D coordinates of segment endpoints.

    Returns:
        list[float]: A list of scalar values in [0, 1], one per point.
    """
    distribution = []

    for p in points:
        lp1 = np.asarray(pos1)
        lp2 = np.asarray(pos2)
        lp3 = np.asarray(pos3)
        lp4 = np.asarray(pos4)

        point = p[:2]

        # Vector for the first segment
        line_vec = lp2 - lp1
        line_mag = np.linalg.norm(line_vec)

        # Vector for the second segment
        line_vec2 = lp4 - lp3
        line_mag2 = np.linalg.norm(line_vec2)

        # Cross product magnitude gives twice the area of the triangle, which is
        # the numerator for distance to a line: distance = |(p2-p1) x (p1-p)| / |p2-p1|
        numerator = np.abs(np.cross(line_vec, lp1 - point))
        numerator2 = np.abs(np.cross(line_vec2, lp3 - point))

        # Perpendicular distances to each segment
        distance = numerator / line_mag if line_mag != 0 else 0.0
        distance2 = numerator2 / line_mag2 if line_mag2 != 0 else 0.0

        denom = distance + distance2
        value = distance / denom if denom != 0 else 0.0
        distribution.append(value)

    return distribution


def points_information(
    filtered_points,
    shadow1,
    shadow2,
    shadow3,
    IOU_threshold,
    n_correlation=n_correlation_value,
):
    """
    Compute point-based information score for a single vehicle given shadow regions.

    Steps:
      1. Use shadow polygons (shadow1/2/3) to divide points into:
         - top points (shadow1),
         - front points (shadow2),
         - side points (shadow3).
      2. Project these subsets to different line-based distributions (front/side).
      3. Concatenate the distributions and call calculate_distribution_and_IOU
         to get an information score and confidence.
      4. If the resulting score is NaN, set it to 1 (fallback).

    Args:
        filtered_points (np.ndarray): Points already filtered to be near the vehicle.
        shadow1 (np.ndarray): 4x2 array for the top shadow polygon.
        shadow2 (np.ndarray): 4x2 array for the front shadow polygon.
        shadow3 (np.ndarray): 4x2 array for the side shadow polygon.
        IOU_threshold (float): IOU threshold used for integral-based scoring.
        n_correlation (int): Correlation factor for distribution grouping.

    Returns:
        (information_score, confidence_score)
    """
    # Mask points inside each shadow region
    filtered_mask_top = filter_points_in_quadrilateral(filtered_points, shadow1)
    filtered_mask_front = filter_points_in_quadrilateral(filtered_points, shadow2)
    filtered_mask_side = filter_points_in_quadrilateral(filtered_points, shadow3)

    points_top = filtered_points[filtered_mask_top]
    points_front = filtered_points[filtered_mask_front]
    points_side = filtered_points[filtered_mask_side]

    # Unpack the 4 corner positions of shadow1 (top shadow)
    shadow_pos1, shadow_pos2, shadow_pos3, shadow_pos4 = shadow1

    # Projections from front and side views
    front_distribution1 = project_points_to_line(points_front, shadow_pos1, shadow_pos2)
    side_distribution1 = project_points_to_line(points_side, shadow_pos1, shadow_pos3)

    # Projections from top view to front and side directions
    front_distribution2 = project_top_points_to_line(
        points_top, shadow_pos1, shadow_pos3, shadow_pos2, shadow_pos4
    )
    side_distribution2 = project_top_points_to_line(
        points_top, shadow_pos1, shadow_pos2, shadow_pos3, shadow_pos4
    )

    # Merge distributions
    front_distribution = np.concatenate((front_distribution1, front_distribution2))
    side_distribution = np.concatenate((side_distribution1, side_distribution2))
    # height_distribution = np.concatenate((height_distribution1, height_distribution2))

    information_score, confidence_score = calculate_distribution_and_IOU(
        front_distribution,
        side_distribution,
        IOU_threshold=IOU_threshold,
        n_correlation=n_correlation,
    )

    # Guard against NaN (e.g., numerical issues in integral)
    if math.isnan(information_score):
        information_score = 1

    return information_score, confidence_score


def calculate_score(
    points,
    car_x,
    car_y,
    car_theta,
    LiDAR_height=LiDAR_height,
    car_height=car_height,
    car_length=car_length,
    car_width=car_width,
    gamma=gamma,
    IOU_threshold=IOU_threshold,
    n_correlation=n_correlation_value,
):
    """
    Compute information and confidence scores for a vehicle given a point cloud.

    The function:
      1. Projects all LiDAR points down to the ground plane.
      2. Builds the vehicle's "shadow" polygons (top, front, side) using its pose
         and dimensions (via calculate_box).
      3. Filters points that fall inside any of these polygons.
      4. Passes the filtered points and polygons to points_information to compute
         an information score and a confidence score.

    Args:
        points (np.ndarray): Raw point cloud, shape (N, 3) or (N, 4).
        car_x (float): Vehicle center x coordinate (world frame).
        car_y (float): Vehicle center y coordinate (world frame).
        car_theta (float): Vehicle yaw angle in radians.
        LiDAR_height (float): Height of the LiDAR above ground.
        car_height (float): Vehicle height.
        car_length (float): Vehicle length.
        car_width (float): Vehicle width.
        gamma (float): Scaling factor for effective height when building shadow.
        IOU_threshold (float): IOU threshold used in the scoring integrals.
        n_correlation (int): Grouping factor for point correlation.

    Returns:
        (information_score, confidence_score):
            information_score (float): Final information score for this vehicle.
            confidence_score (int): Effective independent point count near the vehicle.
    """
    # 1. Project all points onto the ground plane
    all_points = points
    projected_all_points = project_points_to_ground(
        all_points, lidar_height_param=LiDAR_height
    )

    # 2. Construct shadow polygons and linesets for the vehicle based on its pose
    shadow1, shadow2, shadow3, linesets, car_positions, shadow_positions = calculate_box(
        car_x,
        car_y,
        car_theta,
        car_height=car_height,
        car_width=car_width,
        car_length=car_length,
        LiDAR_height=LiDAR_height,
        gamma=gamma,
    )

    # 3. Filter points that lie inside any of the shadow polygons
    filtered_mask1 = filter_points_in_quadrilateral(projected_all_points, shadow1)
    filtered_mask2 = filter_points_in_quadrilateral(projected_all_points, shadow2)
    filtered_mask3 = filter_points_in_quadrilateral(projected_all_points, shadow3)
    mask_sum = filtered_mask1 | filtered_mask2 | filtered_mask3

    filtered_points = projected_all_points[mask_sum]

    # total_point_num = filtered_points.shape[0]  # If needed for debugging or stats

    # 4. Compute point-based information score from these filtered points
    information_score, confidence_score = points_information(
        filtered_points,
        shadow1,
        shadow2,
        shadow3,
        IOU_threshold=IOU_threshold,
        n_correlation=n_correlation,
    )

    return information_score, confidence_score
