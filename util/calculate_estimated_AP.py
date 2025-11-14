from .voxelize_map import create_voxel_heatmap
from .pointcloud_score import calculate_score
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

gamma=0.8 # Blind Spot Correction Factor
car_width=2
car_length=5
car_height=1.5
LiDAR_height=6
IOU_threshold=0.75
n_correlation_value=3
def calculate_ap_at_r(precision, recall, num_samples=40):
    """
    Calculates the Average Precision (AP) using a specified number of recall sampling points.

    This method approximates the area under the precision-recall curve by
    averaging the precision values at a set of evenly spaced recall levels.

    Args:
        precision (np.array or list): An array of precision values from the PR curve.
        recall (np.array or list): An array of recall values from the PR curve.
        num_samples (int): The number of recall points to sample (e.g., 40 for AP|R40).

    Returns:
        float: The approximated Average Precision.
    """
    # Ensure precision and recall are numpy arrays for vectorized operations
    precision = np.array(precision)
    recall = np.array(recall)
    
    # Create an array of `num_samples` evenly spaced recall points from 0 to 1.
    recall_levels = np.linspace(0.0, 1.0, num_samples)
    interpolated_precision = []

    for r_level in recall_levels:
        # For a given recall level `r_level`, find all precision values
        # where the corresponding recall is greater than or equal to `r_level`.
        possible_precisions = precision[recall >= r_level]
        
        # If there are such points, the interpolated precision is the maximum of these.
        # Otherwise, it's 0.
        p_max = possible_precisions.max() if len(possible_precisions) > 0 else 0.0
        interpolated_precision.append(p_max)
        
    # The AP is the mean of these interpolated precision values.
    approximated_ap = np.mean(interpolated_precision)
    
    return approximated_ap


def calculate_estimated_AP(points,labels_directory,region_of_interest,LiDAR_height=LiDAR_height,car_length=car_length, car_width=car_width, car_height=car_height,
                                            gamma=gamma,IOU_threshold=IOU_threshold,n_correlation=n_correlation_value):
    
    total_true_positives=0
    predictions=[]
    voxels=create_voxel_heatmap(region_of_interest, labels_directory)
    for item in voxels:
        x,y,theta,number=item
        try:
            # --- Attempt to calculate the score ---
            information_score, confidence_score = calculate_score(points, x, y, theta, LiDAR_height=LiDAR_height,
                                                                    car_length=car_length, car_width=car_width, car_height=car_height,
                                                                    gamma=gamma,IOU_threshold=IOU_threshold,n_correlation=n_correlation)
            #print(f"Voxel at (x, y) = ({x}, {y},{theta}) has information score: {information_score}, confidence score: {confidence_score}")
            # This code runs only if the above line succeeds
            total_true_positives += number
            predict_labels = [number, information_score, confidence_score]
            predictions.append(predict_labels)

        except Exception as e:
            # --- This block runs if an error occurs in the 'try' block ---
            print(f"An error occurred for voxel at (x, y) = ({x}, {y})")
            print(f"  > Error message: {e}")
            # Continue to the next item in the loop, skipping the problematic one
            continue
       
    predictions.sort(key=lambda x: x[2], reverse=True)
    true_positives = 0
    false_positives = 0
    precision=[]
    recall=[]
    for i, (car_number,information_score,confidence_score)  in enumerate(predictions):
        true_positives +=car_number*information_score

        false_positives += car_number*(1-information_score)
        current_precision=true_positives/(true_positives+false_positives)
        current_recall=true_positives/total_true_positives
        precision.append(current_precision)
        recall.append(current_recall)
    recall, precision = zip(*sorted(zip(recall, precision)))
    recall = list(recall)
    precision = list(precision)
    APR40=calculate_ap_at_r(precision=precision,recall=recall)
    print(APR40)
    # Plotting the Precision-Recall Curve
    return APR40




