from util.calculate_estimated_AP import calculate_estimated_AP
import numpy as np
import open3d as o3d

region_of_interest = [-20, -200, 20, 200]
LiDAR_height=6
car_width=2
car_length=5
car_height=1.5
LiDAR_height=6
IOU_threshold=0.5
gamma=0.8
n_correlation=3

def main():
    labels_directory = './demo/vehicle_num'
    pcd_filepath="./demo/pure_points/00000.pcd"
    pcd = o3d.io.read_point_cloud(pcd_filepath)
    points = np.asarray(pcd.points)
    value5=calculate_estimated_AP(points,labels_directory,region_of_interest = region_of_interest,LiDAR_height=LiDAR_height,car_length=car_length, car_width=car_width, car_height=car_height,
                                            gamma=gamma,IOU_threshold=IOU_threshold,n_correlation=n_correlation)
    print("Estimated AP is ",value5)


if __name__ == "__main__":
    main()