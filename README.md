# pcd-6dof-pose

Point cloud based zero-shot 6-DoF pose estimation package for ROS.

This package consumes point clouds filtered by an external detector and estimates the 6-DoF pose of a target object (e.g., drogue) using clustering and PCA / weighted PCA.

## Overview

- Intended to be used together with an existing perception stack (RGB-D camera, 2D detector, RGBD fusion, etc.).
- Assumes an upstream pipeline that provides a point cloud containing only the target object.
- Estimates a stable 6-DoF pose from this point cloud and (optionally) publishes visualization data for RViz.

## Dependencies

- ROS Noetic
- catkin workspace
- Runtime dependencies (see `package.xml` for full list):
	- `rospy`, `sensor_msgs`, `geometry_msgs`, `std_msgs`, `vision_msgs`, `message_filters`
	- `detection_msgs` (for `BoundingBoxes`)
	- Python 3 packages: `numpy`, `scipy`, `opencv-python`, `open3d`

## Installation

```bash
cd ~/catkin_ws/src
git clone https://github.com/<your-id>/pcd-6dof-pose.git
cd ~/catkin_ws
rosdep install --from-paths src --ignore-src -r -y
catkin_make
source devel/setup.bash
```

## How to Run (integration scenario)

This package is **not** intended to run in isolation. It assumes that:

- An RGB-D camera node is running and publishing depth / point cloud data (e.g., `sensor_msgs/PointCloud2` on a `/camera/.../points` topic).
- A 2D object detector (e.g., YOLO-based) is running and providing detections (e.g., `detection_msgs/BoundingBoxes` on `/yolo_world/detections`).
- Some external component fuses detections with the depth data and outputs a point cloud that mainly contains the target object (by default this package expects a filtered `sensor_msgs/PointCloud2` on `/detection/lidar_detector/yolo_objects_pointcloud`, which can be remapped).

In your own system, once such a filtered point cloud is available, you can integrate this package by launching its nodes (e.g., via `pcd_6dof.launch`) and wiring the input/output topics to match your existing stack.

## Launch

- Main launch file:

```bash
roslaunch pcd_6dof_pose pcd_6dof.launch
```

## Repository Structure

- `src/`
	- `pose_estimator_node.py` – main point cloud based 6-DoF pose estimator
	- `compare_pca.py` – PCA vs weighted PCA orientation comparison
	- `compare_clustering.py` – clustering algorithms (DBSCAN, Agglomerative, HDBSCAN) comparison
- `launch/`
	- `pcd_6dof.launch` – launch file for the pose estimation pipeline
- `cluster/` – saved clustered point clouds (`.npy`)
- `pca_orientation/`, `wpca_orientation/` – saved orientation matrices for offline analysis

## Notes

- This package is designed to be plugged into an existing perception stack (camera + YOLO-World + RGBD drogue detector).
- Frame names such as `camera_depth_optical_frame` and `world` should match your TF tree.
- DBSCAN thresholds (`eps`, `min_samples`) and z filtering thresholds are tuned for a specific setup and may require adjustment for other sensors/environments.
