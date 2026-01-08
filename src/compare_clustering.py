#!/usr/bin/env python3
import rospy
import tf2_ros
import numpy as np
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
import hdbscan
from sensor_msgs.msg import PointCloud2, CameraInfo
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker
from std_msgs.msg import Header
from sensor_msgs import point_cloud2 as pc2
from detection_msgs.msg import BoundingBoxes  # replace if custom

import struct


class CompareClustering:
    def __init__(self):
        rospy.init_node("compare_clustering_node")

        self.intrinsics = None
        self.br = tf2_ros.TransformBroadcaster()

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        rospy.Subscriber("/camera/color/camera_info", CameraInfo, self.camera_info_callback)
        rospy.Subscriber("/yolo_world/detections", BoundingBoxes, self.detections_callback)
        rospy.Subscriber("/detection/lidar_detector/yolo_objects_pointcloud", PointCloud2, self.pointcloud_callback)

        self.pose_pub = rospy.Publisher("/6dof_pose", PoseStamped, queue_size=10)
        self.marker_pub = rospy.Publisher("/6dof_pose_marker", Marker, queue_size=1)
        self.filtered_cloud_pub = rospy.Publisher("/compare_cluster", PointCloud2, queue_size=1)

        self.latest_bounding_boxes = []
        self.latest_pointcloud = None

    def camera_info_callback(self, msg: CameraInfo):
        self.intrinsics = {
            'fx': msg.K[0], 'fy': msg.K[4], 'cx': msg.K[2], 'cy': msg.K[5]
        }

    def detections_callback(self, msg: BoundingBoxes):
        self.latest_bounding_boxes = msg.bounding_boxes

    def pointcloud_callback(self, msg: PointCloud2):
        self.latest_pointcloud = msg
        if not self.latest_bounding_boxes:
            return

        pc_points = list(pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z")))
        if len(pc_points) < 3:
            return

        points_np = np.array(pc_points)
        z_min = points_np[:, 2].min()
        z_thresh = z_min + 0.15 # cropping threshold (0.08)
        filtered_points = points_np[points_np[:, 2] <= z_thresh]

        if len(filtered_points) < 30:
            rospy.logwarn("Filtered points too few")
            return

        # clustering algorithms to compare
        algorithms = {
            "DBSCAN": DBSCAN(eps=0.02, min_samples=20),
            "Agglomerative": AgglomerativeClustering(n_clusters=None, distance_threshold=0.5),
            "HDBSCAN": hdbscan.HDBSCAN(min_samples=20),
        }

        for name, algo in algorithms.items():
            try:
                labels = algo.fit_predict(filtered_points)
            except Exception as e:
                rospy.logwarn(f"{name} failed: {e}")
                continue

            if len(set(labels)) <= 1:
                rospy.logwarn(f"{name}: No valid clusters")
                continue

            cloud_msg = self.create_colored_cloud(filtered_points, labels, frame_id=msg.header.frame_id)
            self.filtered_cloud_pub.publish(cloud_msg)
            rospy.loginfo(f"Published clustered result from {name}")
            rospy.sleep(1.0)  # pause for RViz comparison

    def create_colored_cloud(self, points: np.ndarray, labels: np.ndarray, frame_id: str) -> PointCloud2:
        colors = self.label_to_color(labels)
        cloud_data = []

        for pt, rgb in zip(points, colors):
            r, g, b = rgb
            rgb_int = struct.unpack('I', struct.pack('BBBB', b, g, r, 0))[0]
            cloud_data.append([pt[0], pt[1], pt[2], rgb_int])

        fields = [
            pc2.PointField('x', 0, pc2.PointField.FLOAT32, 1),
            pc2.PointField('y', 4, pc2.PointField.FLOAT32, 1),
            pc2.PointField('z', 8, pc2.PointField.FLOAT32, 1),
            pc2.PointField('rgb', 12, pc2.PointField.UINT32, 1),
        ]

        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = frame_id

        return pc2.create_cloud(header, fields, cloud_data)

    def label_to_color(self, labels: np.ndarray):
        """Generate RGB color per cluster label."""
        unique = np.unique(labels)
        colormap = {label: (np.random.randint(50, 256),
                            np.random.randint(50, 256),
                            np.random.randint(50, 256)) for label in unique if label != -1}
        colormap[-1] = (100, 100, 100)  # noise
        return [colormap[label] for label in labels]


if __name__ == "__main__":
    try:
        CompareClustering()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
