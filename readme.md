# Evaluating Roadside LiDAR Placement with a Probability-based Surrogate Metric

This repository provides the official Python implementation for the paper: **"Evaluating Roadside LiDAR Placement with a Probability-based Surrogate Metric"**.

The core of this work is **E-AP**, a novel surrogate metric that provides a direct, computationally efficient estimation of object detection Average Precision (AP) for a given roadside LiDAR configuration. This metric allows for the rapid evaluation of different sensor placements without the need to collect unique datasets or train new detection models for every potential setup.


## How E-AP Works

The E-AP calculation pipeline, as implemented in this repository, follows the methodology described in the paper:

1. **Vehicle Heatmap Generation (`voxelize_map.py`)**

   * The process begins by analyzing a `labels_directory` which contains ground-truth vehicle poses from a simulation or dataset.

   * It aggregates all vehicle poses (`x`, `y`, `theta`) into a 2D-plus-orientation grid, creating a vehicle distribution heatmap. This map, `V(x,y,θ)`, represents the number of times a vehicle appeared at each specific pose.

2. **Per-Pose Scoring (BOPE) (`pointcloud_score.py`)**

   * For each populated cell in the heatmap (representing a common vehicle pose), the script calculates two key values:

     * **Occlusion Shadow Calculation:** It first computes the 2D ground-plane occlusion shadow that a vehicle at this pose would cast, relative to the LiDAR's position (`multiple_vehicle_display.py`).

     * **Point Cloud Filtering:** It takes an "empty-scene" point cloud (a scan of the environment *without* any vehicles) and filters it, keeping only the points that fall inside the calculated shadow.

     * **Bayesian Estimation:** Using these "shadowed" points, it applies Bayesian inference (via `math_toolbox.py`) to estimate the probability of a successful detection. This probabilistic accuracy is the **Bayesian Occlusion-based Perception Estimation (BOPE)**, referred to in the code as `information_score`.

     * A `confidence_score` (based on the number of points in the shadow) is also calculated.

3. **Final E-AP Calculation (`calculate_estimated_AP.py`)**

   * As described in Algorithm 1 of the paper, all scored poses are sorted in descending order by their `confidence_score`.

   * The script iterates this sorted list, using the `information_score` (BOPE) and the vehicle count (`number`) from each pose to probabilistically build a Precision-Recall curve.

   * The final **E-AP** score is the Area Under this P-R curve (calculated as AP@R40), which serves as the final surrogate metric for the LiDAR placement.

## Repository Structure

* `demo_estimated_AP.py`: The main script to run a demo calculation. This is the best place to start.

* `calculate_estimated_AP.py`: The main orchestrator that implements Algorithm 1, tying together the heatmap and the per-pose scores.

* `voxelize_map.py`: Reads label files and generates the vehicle distribution heatmap.

* `pointcloud_score.py`: Calculates the BOPE (`information_score`) and `confidence_score` for a single vehicle pose.

* `multiple_vehicle_display.py`: A geometry helper library for calculating 2D vehicle bounding boxes and their occlusion shadows.

* `math_toolbox.py`: The core mathematical functions (using `scipy` and `numba`) that perform the numerical integration for the Bayesian estimation.

* `demo/`: A folder containing example data.

  * `demo/vehicle_num/`: An example `labels_directory`.

  * `demo/pure_points/`: An example "empty-scene" point cloud.

## How to Run Demo

### 1. Prerequisites

You will need the following Python libraries. You can install them via pip:

```
conda create -n eval_LiDAR python=3.10
conda activate eval_LiDAR
pip install numpy open3d scipy numba matplotlib
```

### 2. Extract the data

```
unzip ./demo/vehicle_num.zip
```
### 3. Run the Script


```
python demo_estimated_AP.py
```

## How to Prepare Your Data


To evaluate your own scene, you need two things:

1. **Empty-Scene Point Cloud:** A single `.pcd` file of your scene with **no vehicles** or dynamic objects. This is used to find points within the occlusion shadows.

   * Place this in a folder (e.g., `demo/pure_points/`).

2. **Vehicle Labels Directory:** A folder containing `.txt` files that log the poses of all vehicles in your dataset. Each line in each file must represent a single vehicle in the following format:


```
x y z dx dy dz dtheta
```

* The script primarily uses `x` (col 0), `y` (col 1), and `dtheta` (col 6) to build the heatmap.
