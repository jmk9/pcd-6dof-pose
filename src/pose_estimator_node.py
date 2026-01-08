#!/usr/bin/env python3

import rospy
import numpy as np
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import PoseStamped, TransformStamped
from detection_msgs.msg import BoundingBoxes
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker
import sensor_msgs.point_cloud2 as pc2
from scipy.spatial.transform import Rotation
from sklearn.cluster import DBSCAN
import std_msgs.msg
import tf2_ros
import tf2_geometry_msgs

class YOLOPoseEstimator:
    def __init__(self):
        rospy.init_node("yolo_pose_estimator_node")

        self.intrinsics = None
        self.br = tf2_ros.TransformBroadcaster()

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        rospy.Subscriber("/camera/color/camera_info", CameraInfo, self.camera_info_callback)
        rospy.Subscriber("/yolo_world/detections", BoundingBoxes, self.detections_callback)
        rospy.Subscriber("/detection/lidar_detector/yolo_objects_pointcloud", PointCloud2, self.pointcloud_callback)

        self.pose_pub = rospy.Publisher("/6dof_pose", PoseStamped, queue_size=10)
        self.marker_pub = rospy.Publisher("/6dof_pose_marker", Marker, queue_size=1)
        self.filtered_cloud_pub = rospy.Publisher("/filtered_cluster_pointcloud", PointCloud2, queue_size=1)

        self.latest_bounding_boxes = []
        self.latest_pointcloud = None

    def camera_info_callback(self, msg):
        self.intrinsics = {
            'fx': msg.K[0], 'fy': msg.K[4], 'cx': msg.K[2], 'cy': msg.K[5]
        }

    def detections_callback(self, msg):
        self.latest_bounding_boxes = msg.bounding_boxes

    def pointcloud_callback(self, msg):
        self.latest_pointcloud = msg
        if not self.latest_bounding_boxes:
            return

        pc_points = list(pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z")))
        if len(pc_points) < 30:
            return

        points_np = np.array(pc_points)
        z_min = points_np[:, 2].min()
        z_thresh = z_min + 0.08
        filtered_points = points_np[points_np[:, 2] <= z_thresh]

        if len(filtered_points) < 30:
            rospy.logwarn("Filtered points too few")
            return

        db = DBSCAN(eps=0.05, min_samples=20).fit(filtered_points)
        labels = db.labels_
        unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)

        if len(counts) == 0:
            rospy.logwarn("No valid clusters found")
            return

        best_cluster = unique_labels[np.argmax(counts)]
        clustered_points = filtered_points[labels == best_cluster]

        # PointCloud for cluster visualization
        filtered_cloud_msg = self.create_cloud_from_xyz(clustered_points, "camera_depth_optical_frame")
        self.filtered_cloud_pub.publish(filtered_cloud_msg)

        # 6D pose estimation
        centroid = np.mean(clustered_points, axis=0)
        cov = np.cov(clustered_points.T)
        eigvals, eigvecs = np.linalg.eigh(cov)

        sorted_indices = np.argsort(eigvals)[::-1]
        x_axis = eigvecs[:, sorted_indices[0]]
        y_axis = eigvecs[:, sorted_indices[1]]
        z_axis = eigvecs[:, sorted_indices[2]]

        camera_forward = np.array([0.0, 0.0, 1.0])
        if np.dot(z_axis, camera_forward) < 0:
            z_axis = -z_axis

        x_axis = np.cross(y_axis, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)
        z_axis /= np.linalg.norm(z_axis)

        R = np.column_stack((x_axis, y_axis, z_axis))

        theta = -np.pi / 2
        Ry = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])
        R = R @ Ry
        quat = Rotation.from_matrix(R).as_quat()

        # --- updated section start ---
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()  # ⚠ ensure timestamp is available for TF lookup
        pose.header.frame_id = "camera_depth_optical_frame"
        pose.pose.position.x = centroid[0]
        pose.pose.position.y = centroid[1]
        pose.pose.position.z = centroid[2]
        pose.pose.orientation.x = quat[0]
        pose.pose.orientation.y = quat[1]
        pose.pose.orientation.z = quat[2]
        pose.pose.orientation.w = quat[3]

        try:
            transform = self.tf_buffer.lookup_transform("world",  # ⬅ target frame
                                                        pose.header.frame_id,  # ⬅ source frame
                                                        pose.header.stamp,
                                                        rospy.Duration(1.0))
            pose_world = tf2_geometry_msgs.do_transform_pose(pose, transform)
            self.pose_pub.publish(pose_world)  # ⬅ publish with world as reference frame

            # t = TransformStamped()
            # t.header.stamp = rospy.Time.now()
            # t.header.frame_id = "camera_depth_optical_frame"
            # t.child_frame_id = "6dof_pose"
            # t.transform.translation.x = pose_world.pose.position.x
            # t.transform.translation.y = pose_world.pose.position.y
            # t.transform.translation.z = pose_world.pose.position.z
            # t.transform.rotation = pose_world.pose.orientation
            # self.br.sendTransform(t)

        except Exception as e:
            rospy.logwarn(f"[TF Transform 실패] {e}")
            return
        # --- updated section end ---

    def create_cloud_from_xyz(self, points: np.ndarray, frame_id: str) -> PointCloud2:
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = frame_id

        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1)
        ]
        return pc2.create_cloud(header, fields, points.tolist())

if __name__ == '__main__':
    try:
        YOLOPoseEstimator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
