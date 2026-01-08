#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker

def publish_static_pose():
    rospy.init_node('static_6dof_pose_publisher', anonymous=True)
    pose_pub = rospy.Publisher('/6dof_pose', PoseStamped, queue_size=10)
    marker_pub = rospy.Publisher('/6dof_pose_marker', Marker, queue_size=1)
    rate = rospy.Rate(10)  # 10 Hz

    pose_msg = PoseStamped()
    pose_msg.header.frame_id = "camera_link"
    pose_msg.pose.position.x = 0.03924533233428793
    pose_msg.pose.position.y = -0.028416701200830178
    pose_msg.pose.position.z = 0.2526014346131663
    pose_msg.pose.orientation.x = -0.397281107064794
    pose_msg.pose.orientation.y = 0.6008930745301194
    pose_msg.pose.orientation.z = -0.419922097848802
    pose_msg.pose.orientation.w = -0.552051326136776

    while not rospy.is_shutdown():
        # Update timestamp
        pose_msg.header.stamp = rospy.Time.now()

        # Publish pose
        pose_pub.publish(pose_msg)

        # Publish marker for visualization
        marker = Marker()
        marker.header = pose_msg.header
        marker.ns = "static_pose_arrow"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.pose = pose_msg.pose
        marker.scale.x = 0.1  # arrow length
        marker.scale.y = 0.02
        marker.scale.z = 0.02
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 1.0
        marker.lifetime = rospy.Duration(0.2)
        marker_pub.publish(marker)

        rate.sleep()

if __name__ == '__main__':
    try:
        publish_static_pose()
    except rospy.ROSInterruptException:
        pass
