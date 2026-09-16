import random
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped


# Must match navigator.py's obstacle definitions
OBSTACLES = [
    {'cx': 3.0, 'cy': 0.0, 'hx': 0.5, 'hy': 1.5},
    {'cx': 6.0, 'cy': -2.0, 'hx': 0.5, 'hy': 1.5},
]
EXCLUSION_MARGIN = 1.2   # keep waypoints at least this far from any obstacle edge


def too_close_to_obstacle(x, y):
    for obs in OBSTACLES:
        hx = obs['hx'] + EXCLUSION_MARGIN
        hy = obs['hy'] + EXCLUSION_MARGIN
        if abs(x - obs['cx']) <= hx and abs(y - obs['cy']) <= hy:
            return True
    return False


class WaypointGenerator(Node):
    def __init__(self):
        super().__init__('waypoint_generator')

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

        self.get_logger().info(f'Generating {num_waypoints} random waypoints (avoiding obstacles):')
        for i in range(num_waypoints):
            attempts = 0
            while True:
                x = random.uniform(*x_range)
                y = random.uniform(*y_range)
                attempts += 1
                if not too_close_to_obstacle(x, y):
                    break
                if attempts > 200:
                    self.get_logger().warn(
                        f'  Waypoint {i+1}: could not find a clear spot after 200 tries, using last attempt anyway.')
                    break

            pose = PoseStamped()
            pose.header.frame_id = 'world'
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            path_msg.poses.append(pose)

            self.get_logger().info(f'  Waypoint {i+1}: x={x:.2f}, y={y:.2f} (took {attempts} attempt(s))')

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
