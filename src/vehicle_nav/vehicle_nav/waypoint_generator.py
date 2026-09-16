import random
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped


class WaypointGenerator(Node):
    def __init__(self):
        super().__init__('waypoint_generator')

        # Keep the last message "pinned" so late-joining subscribers still get it
        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.publisher_ = self.create_publisher(Path, 'waypoints', qos)

        num_waypoints = 4
        x_range = (2.0, 8.0)
        y_range = (-3.0, 3.0)

        path_msg = Path()
        path_msg.header.frame_id = 'world'
        path_msg.header.stamp = self.get_clock().now().to_msg()

        self.get_logger().info(f'Generating {num_waypoints} random waypoints:')
        for i in range(num_waypoints):
            x = random.uniform(*x_range)
            y = random.uniform(*y_range)

            pose = PoseStamped()
            pose.header.frame_id = 'world'
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            path_msg.poses.append(pose)

            self.get_logger().info(f'  Waypoint {i+1}: x={x:.2f}, y={y:.2f}')

        # Small delay so the publisher has time to register on the network
        # before we send the message
        time.sleep(1.0)
        self.publisher_.publish(path_msg)
        self.get_logger().info('Waypoints published.')


def main(args=None):
    rclpy.init(args=args)
    node = WaypointGenerator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
