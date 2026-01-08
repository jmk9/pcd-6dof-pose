#!/usr/bin/env python3
import rospy
import numpy as np
import std_msgs.msg
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation
from sklearn.cluster import DBSCAN

class PCAPoseComparer:
    def __init__(self):
        rospy.init_node("compare_pca_wpca_node")

        rospy.Subscriber("/detection/lidar_detector/yolo_objects_pointcloud", PointCloud2, self.pointcloud_callback)

        self.pca_pub = rospy.Publisher("/pca_pose", PoseStamped, queue_size=1)
        self.wpca_pub = rospy.Publisher("/wpca_pose", PoseStamped, queue_size=1)
        self.filtered_cloud_pub = rospy.Publisher("/filtered_cluster_pointcloud", PointCloud2, queue_size=1)

        self.clustered_points = None
        self.save_count = 0
        rospy.Timer(rospy.Duration(1.0), self.timer_callback, oneshot=False)  # start timer with 1-second period
        self.start_time = rospy.Time.now()

        self.R_pca = None
        self.R_wpca = None


    def timer_callback(self, event):
        if self.clustered_points is None:
            return

        elapsed = (rospy.Time.now() - self.start_time).to_sec()
        if elapsed < 5.0:
            return  # wait for 5 seconds

        if self.save_count >= 10:
            return  # stop after saving 10 times

        filename = f"/root/drogue_ws/src/pcd_6dof_pose/cluster/clustered_{self.save_count:02d}.npy"
        np.save(filename, self.clustered_points)
        rospy.loginfo(f"[💾] Saved clustered points to {filename}")
        self.save_count += 1

    def timer_callback(self, event):
        if self.clustered_points is None or self.R_pca is None or self.R_wpca is None:
            return

        elapsed = (rospy.Time.now() - self.start_time).to_sec()
        if elapsed < 5.0:
            return

        if self.save_count >= 10:
            return

        i = self.save_count
        np.save(f"/root/drogue_ws/src/pcd_6dof_pose/cluster/clustered_{i:02d}.npy", self.clustered_points)
        np.save(f"/root/drogue_ws/src/pcd_6dof_pose/pca_orientation/pca_orientation_{i:02d}.npy", self.R_pca)
        np.save(f"/root/drogue_ws/src/pcd_6dof_pose/wpca_orientation/wpca_orientation_{i:02d}.npy", self.R_wpca)

        rospy.loginfo(f"[💾] Saved clustered and orientations to /tmp/*_{i:02d}.npy")
        self.save_count += 1


    def pointcloud_callback(self, msg: PointCloud2):
        points = list(pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z")))
        if len(points) < 30:
            return

        np_points = np.array(points)

        # --- filtering based on z-min ---
        z_min = np_points[:, 2].min()
        tau = 0.06
        z_thresh = z_min + tau
        filtered = np_points[np_points[:, 2] <= z_thresh]

        if len(filtered) < 30:
            rospy.logwarn("Filtered points too few")
            return

        # --- DBSCAN clustering ---
        db = DBSCAN(eps=0.05, min_samples=10).fit(filtered)
        labels = db.labels_

        unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
        if len(counts) == 0:
            rospy.logwarn("No clusters found")
            return

        best_cluster = unique_labels[np.argmax(counts)]
        clustered = filtered[labels == best_cluster]

        if len(clustered) < 10:
            rospy.logwarn("Selected cluster too few")
            return

        # --- publish ---
        self.filtered_cloud_pub.publish(self.create_cloud(clustered, msg.header.frame_id))

        self.clustered_points = clustered.copy()


        # --- PCA ---
        centroid = np.mean(filtered, axis=0)
        cov = np.cov(filtered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        pca_orientation = self.extract_orientation(eigvecs, eigvals)

        pose_pca = self.make_pose(centroid, pca_orientation, msg.header.frame_id)
        self.pca_pub.publish(pose_pca)

        self.R_pca = pca_orientation

        # --- WPCA ---
        # distance-based weights from the centroid
        dists = np.linalg.norm(filtered - centroid, axis=1)
        w = 1 / (dists + 1e-6)
        w /= np.sum(w)

        mean_w = np.average(filtered, axis=0, weights=w)
        centered = filtered - mean_w
        cov_w = (w[:, None] * centered).T @ centered

        eigvals_w, eigvecs_w = np.linalg.eigh(cov_w)
        wpca_orientation = self.extract_orientation(eigvecs_w, eigvals_w)

        pose_wpca = self.make_pose(mean_w, wpca_orientation, msg.header.frame_id)
        self.wpca_pub.publish(pose_wpca)

        self.R_wpca = wpca_orientation

    def extract_orientation(self, eigvecs: np.ndarray, eigvals: np.ndarray) -> np.ndarray:
        idx = np.argsort(eigvals)[::-1]
        x = eigvecs[:, idx[0]]
        y = eigvecs[:, idx[1]]
        z = eigvecs[:, idx[2]]

        camera_forward = np.array([0, 0, 1])
        if np.dot(z, camera_forward) < 0:
            z = -z
        x = np.cross(y, z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        y /= np.linalg.norm(y)
        z /= np.linalg.norm(z)

        R = np.column_stack((x, y, z))
        return R

    def make_pose(self, center: np.ndarray, R: np.ndarray, frame_id: str) -> PoseStamped:
        theta_x = np.pi / 2
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(theta_x), -np.sin(theta_x)],
            [0, np.sin(theta_x),  np.cos(theta_x)]
        ])
        R = Rx @ R  # multiply X-axis rotation in front of original R (local -> rotated)

        quat = Rotation.from_matrix(R).as_quat()
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = frame_id
        pose.pose.position.x = center[0]
        pose.pose.position.y = center[1]
        pose.pose.position.z = center[2]
        pose.pose.orientation.x = quat[0]
        pose.pose.orientation.y = quat[1]
        pose.pose.orientation.z = quat[2]
        pose.pose.orientation.w = quat[3]
        return pose

    def create_cloud(self, points: np.ndarray, frame_id: str) -> PointCloud2:
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = frame_id

        fields = [
            PointField("x", 0, PointField.FLOAT32, 1),
            PointField("y", 4, PointField.FLOAT32, 1),
            PointField("z", 8, PointField.FLOAT32, 1)
        ]
        return pc2.create_cloud(header, fields, points.tolist())

if __name__ == "__main__":
    try:
        PCAPoseComparer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
