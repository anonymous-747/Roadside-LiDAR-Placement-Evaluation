import numpy as np
import matplotlib.pyplot as plt
import os
import glob


pie=np.pi
def create_voxel_heatmap(region, label_folder):
    """
    Generates and displays a heatmap from label files within a specified region.

    Args:
        region (list): A list containing [x_start, y_start, x_end, y_end].
        label_folder (str): The path to the folder containing the label files.
    """
    x_start, y_start, x_end, y_end = region
    
    # Calculate the dimensions of the voxel grid
    # We use ceiling to ensure we cover the entire region
    x_voxels = int(np.ceil(x_end - x_start))
    y_voxels = int(np.ceil(y_end - y_start))

    # Initialize a 2D numpy array (our voxel grid) with zeros
    voxel_grid = np.zeros((y_voxels, x_voxels,8))
    voxel_grid_sum = np.zeros((y_voxels, x_voxels))
    # Check if the label folder exists
    if not os.path.isdir(label_folder):
        print(f"Error: The folder '{label_folder}' does not exist.")
        return

    # Process each label file in the specified folder
    # We look for files ending with .txt
    for label_file in glob.glob(os.path.join(label_folder, '*.txt')):
        try:
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 2:
                        continue # Skip malformed lines
                        
                    # Extract x and y coordinates
                    x, y,theta = float(parts[0]), float(parts[1]), float(parts[6])
                    number=-99
                    if( 3* pie/4 - pie/8) <= theta < (3* pie/4 + pie/8):
                        number = 7
                    if( 2* pie/4 - pie/8) <= theta < (2* pie/4 + pie/8):
                        number = 6
                    if( 1* pie/4 - pie/8) <= theta < (1* pie/4 + pie/8):
                        number = 5
                    if( 0 - pie/8) <= theta < (0 + pie/8):
                        number = 4
                    if( -1* pie/4 - pie/8) <= theta < (-1* pie/4 + pie/8):
                        number = 3
                    if( -2* pie/4 - pie/8) <= theta < (-2* pie/4 + pie/8):
                        number = 2
                    if( -3* pie/4 - pie/8) <= theta < (-3* pie/4 + pie/8):
                        number = 1
                    if theta >= pie -pie/8 or theta < -pie+pie/8 :
                        number = 0
                    # Check if the point is within our defined region
                    if x_start <= x < x_end and y_start <= y < y_end:
                        # Determine which voxel the point belongs to
                        voxel_x = int(x - x_start)
                        voxel_y = int(y - y_start)
                        
                        # Increment the count for that voxel
                        # Note: We index with (y, x) because of numpy's (row, col) convention
                        if 0 <= voxel_y < y_voxels and 0 <= voxel_x < x_voxels:
                            voxel_grid[voxel_y, voxel_x,number] += 1
                            voxel_grid_sum[voxel_y, voxel_x] += 1
        except Exception as e:
            print(f"Could not process file {label_file}: {e}")
    populated_voxels = []
    grid_height, grid_width,a = voxel_grid.shape
    
    # Iterate over each voxel in the grid
    for y_index in range(grid_height):
        for x_index in range(grid_width):
            for number in range(8):
            # Get the count of points in the current voxel
                count = voxel_grid[y_index, x_index,number]
                
                # If the voxel contains one or more points, record it
                if count > 0:
                    # Calculate the real-world coordinates of the voxel's bottom-left corner
                    real_x = x_start + x_index+0.5
                    real_y = y_start + y_index+0.5
                    theta=  pie* number / 4 - pie
                    # Append the [x, y, count] to our list
                    populated_voxels.append([real_x, real_y,theta, int(count)])

    # --- Visualization ---
    if np.sum(voxel_grid) == 0:
        print("No data points found within the specified region. Cannot generate heatmap.")
        return
    return populated_voxels

