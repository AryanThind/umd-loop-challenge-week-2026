import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, qos_profile_sensor_data
from geometry_msgs.msg import Twist, PoseArray
from nav_msgs.msg import Path


OBSTACLES = [
    (3.0, 0.0, 2.8),
    (6.0, -2.0, 2.8),
]

WAYPOINT_TOLERANCE = 0.4
FORWARD_SPEED = 0.6
TURN_SPEED = 1.0
OBSTACLE_LOOKAHEAD = 3.5
DETECT_CONE_DEG = 55
CLEAR_CONE_DEG = 85

STUCK_CHECK_TICKS = 15     # ~1.5s at 10Hz
STUCK_DISTANCE_THRESH = 0.08
RECOVERY_TICKS = 20        # ~2s of recovery maneuver


def yaw_from_quaternion(q):
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


class Navigator(Node):
    def __init__(self):
        super().__init__('navigator')

        waypoints_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(Path, 'waypoints', self.waypoints_callback, waypoints_qos)
        self.create_subscription(PoseArray, '/model/vehicle/pose', self.pose_callback, qos_profile_sensor_data)
        self.cmd_pub = self.create_publisher(Twist, '/model/vehicle/cmd_vel', 10)

        self.waypoints = []
        self.current_index = 0
        self.current_x = None
        self.current_y = None
        self.current_yaw = None
        self.finished = False

        self.avoiding = False
        self.avoid_direction = 0.0

        # Stuck detection
        self.check_pos = None
        self.stuck_counter = 0
        self.recovering = False
        self.recovery_counter = 0
        self.recovery_direction = 1.0

        self._tick_count = 0
        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info('Navigator started, waiting for waypoints and pose...')

    def waypoints_callback(self, msg):
        if not self.waypoints:
            self.waypoints = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
            self.get_logger().info(f'Received {len(self.waypoints)} waypoints.')

    def pose_callback(self, msg):
        if len(msg.poses) > 0:
            pose = msg.poses[0]
            self.current_x = pose.position.x
            self.current_y = pose.position.y
            self.current_yaw = yaw_from_quaternion(pose.orientation)

    def closest_obstacle_angle(self):
        best = None
        for (ox, oy, radius) in OBSTACLES:
            dx = ox - self.current_x
            dy = oy - self.current_y
            dist = math.hypot(dx, dy)
            if dist > OBSTACLE_LOOKAHEAD or dist > radius:
                continue
            angle_to_obstacle = math.atan2(dy, dx)
            signed_offset = normalize_angle(angle_to_obstacle - self.current_yaw)
            angle_diff = abs(signed_offset)
            if best is None or dist < best[2]:
                best = (angle_diff, signed_offset, dist)
        return best

    def check_if_stuck(self):
        if self.check_pos is None:
            self.check_pos = (self.current_x, self.current_y)
            self.stuck_counter = 0
            return False

        self.stuck_counter += 1
        if self.stuck_counter >= STUCK_CHECK_TICKS:
            moved = math.hypot(self.current_x - self.check_pos[0], self.current_y - self.check_pos[1])
            self.check_pos = (self.current_x, self.current_y)
            self.stuck_counter = 0
            if moved < STUCK_DISTANCE_THRESH:
                return True
        return False

    def control_loop(self):
        if self.finished:
            return
        if self.current_x is None or not self.waypoints:
            return

        cmd = Twist()

        # --- Recovery mode: back up and turn, ignore everything else ---
        if self.recovering:
            self.recovery_counter += 1
            cmd.linear.x = -0.3
            cmd.angular.z = TURN_SPEED * self.recovery_direction
            self.cmd_pub.publish(cmd)
            if self.recovery_counter >= RECOVERY_TICKS:
                self.recovering = False
                self.avoiding = False
                self.get_logger().info('Recovery maneuver complete, resuming navigation.')
            return

        goal_x, goal_y = self.waypoints[self.current_index]
        dx = goal_x - self.current_x
        dy = goal_y - self.current_y
        distance = math.hypot(dx, dy)

        if distance < WAYPOINT_TOLERANCE:
            self.get_logger().info(f'Reached waypoint {self.current_index + 1}/{len(self.waypoints)}.')
            self.current_index += 1
            self.avoiding = False
            self.check_pos = None
            if self.current_index >= len(self.waypoints):
                self.finished = True
                self.cmd_pub.publish(Twist())
                self.get_logger().info('All waypoints reached. Done.')
            return

        # Check for stuck condition (jammed against something despite commands)
        if self.check_if_stuck():
            self.recovering = True
            self.recovery_counter = 0
            self.recovery_direction = -self.avoid_direction if self.avoid_direction != 0 else 1.0
            self.get_logger().info('Stuck detected! Backing up and turning to recover.')
            return

        obstacle_info = self.closest_obstacle_angle()

        if self.avoiding:
            still_blocked = obstacle_info is not None and obstacle_info[0] < math.radians(CLEAR_CONE_DEG)
            if still_blocked:
                cmd.linear.x = 0.0
                cmd.angular.z = TURN_SPEED * self.avoid_direction
                self.cmd_pub.publish(cmd)
                return
            else:
                self.avoiding = False
                self.get_logger().info('Clear of obstacle, resuming toward waypoint.')

        if obstacle_info is not None and obstacle_info[0] < math.radians(DETECT_CONE_DEG):
            angle_diff, signed_offset, dist = obstacle_info
            self.avoiding = True
            self.avoid_direction = -1.0 if signed_offset > 0 else 1.0
            self.get_logger().info(
                f'Obstacle detected at {dist:.2f}m, angle_diff={math.degrees(angle_diff):.0f} deg - turning.')
            cmd.linear.x = 0.0
            cmd.angular.z = TURN_SPEED * self.avoid_direction
            self.cmd_pub.publish(cmd)
            return

        goal_heading = math.atan2(dy, dx)
        angle_error = normalize_angle(goal_heading - self.current_yaw)
        cmd.angular.z = max(-TURN_SPEED, min(TURN_SPEED, 2.0 * angle_error))
        cmd.linear.x = FORWARD_SPEED if abs(angle_error) < math.radians(30) else 0.15
        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = Navigator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
