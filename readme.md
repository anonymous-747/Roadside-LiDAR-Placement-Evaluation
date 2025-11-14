# LiDAR Point Cloud Analysis for Vehicle Detection Performance Estimation
This project is a collection of Python scripts designed to analyze LiDAR point clouds and estimate the performance of 3D object detectors (like Average Precision, AP) without requiring ground truth labels from the detector itself. It introduces metrics like Estimated AP (E-AP) and Expected Ground Vehicle Score (EGVS) to evaluate detection quality based on the physical properties of the point cloud and expected vehicle locations.

The primary use case is for LiDAR placement optimization, allowing users to assess the quality of sensor data from a specific viewpoint before training and deploying a full detection model.

## Core Concepts
* Estimated Average Precision (E-AP): This is a novel metric calculated by this toolset. Instead of matching detector outputs to ground truth boxes, it analyzes the point cloud data within potential vehicle locations. It generates an "information score" (how well the points resemble a vehicle) and a "confidence score" for each potential object. These scores are then used to compute a precision-recall curve and ultimately the E-AP.

* Expected Ground Vehicle Score (EGVS): A metric that evaluates a LiDAR position's overall effectiveness. It voxelizes the space, calculates the historical probability of a vehicle appearing in each voxel, and weights this by the number of LiDAR points that fall into that voxel. A higher EGVS suggests a better sensor position for observing vehicles.

* Shadow Analysis: The scripts heavily rely on projecting the "shadow" a vehicle would cast on the ground from the LiDAR's perspective. The distribution and density of points within this shadow region are key inputs for calculating the information and confidence scores.

## File Descriptions
Here is a breakdown of what each script does:

- EGVS.py 
    - Calculates the Expected Ground Vehicle Score (EGVS).

    - It creates a heatmap of vehicle probabilities from label files and then counts the number of point cloud points that fall within each voxel to generate the final score.

- calculate_estimated_AP.py

    - The main script for computing the Estimated Average Precision (E-AP).

    - It uses a voxel map of vehicle locations and calls pointcloud_score.py to evaluate each potential vehicle.

    - It then calculates a precision-recall curve based on the returned scores to find the E-AP.

- pointcloud_score.py

    - A core utility that calculates an "information score" and a "confidence score" for a single potential vehicle.

    - It projects the vehicle's shadow and analyzes the distribution of points on the vehicle's front, side, and top surfaces to determine if the object is likely a well-observed vehicle.

- math_toolbox.py

    - A helper script containing complex mathematical functions, primarily for numerical integration (dblquad).

    - These functions are accelerated with Numba (@jit) and are used by pointcloud_score.py to calculate probabilities based on point distribution models.

- multiple_vehicle_display.py & draw_occluded_pointcloud.py

    - Visualization scripts that use Open3D.

    - They can render a point cloud, overlay vehicle bounding boxes, and draw the calculated shadow projections on the ground. This is crucial for debugging and understanding the geometric analysis.

- car_differenet_range.py

    - An analysis script that investigates the relationship between the number of points on a vehicle and the "confidence score" calculated by pointcloud_score.py.

    - It processes data from various scenes (highway, crossroad, curve), performs a linear regression, and plots the results to validate the scoring metric.

- parametre_sen_test.py

    - A script for conducting sensitivity analysis on the model's parameters.

    - It systematically varies key parameters (e.g., gamma, car_height, n_correlation) and re-calculates the E-AP to understand how each parameter affects the outcome. This is used for tuning the estimation model.

- draw_figure.py

    -A plotting script that uses Matplotlib to generate scatter plots.

    - It compares the E-AP and EGVS metrics calculated by these scripts against ground truth AP values from established detectors (like PointPillar and PVRCNN) to validate the accuracy of the estimation methods.

- demo_estimated_AP.py

    - A simple example script demonstrating how to use the core function calculate_estimated_AP on a sample point cloud and label directory.

## How to Run
Dependencies: Ensure you have the required Python libraries installed:
```bash
conda create -n RLiDAR python=3.8
conda activate RLiDAR
pip install numpy open3d matplotlib scipy numba
```

Demo: To run a simple demonstration of the E-AP calculation, execute the demo script. You will need a demo folder containing a vehicle_num subdirectory with label files and a pure_points subdirectory with a .pcd file.
```python
python demo_estimated_AP.py
```

Visualization: To visualize the point clouds with vehicle boxes and shadows, run draw_occluded_pointcloud.py or multiple_vehicle_display.py after configuring the file paths inside the script.

```python
python draw_occluded_pointcloud.py
```
