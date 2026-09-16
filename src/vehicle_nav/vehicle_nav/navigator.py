import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, qos_profile_sensor_data
from geometry_msgs.msg import Twist, PoseArray
from nav_msgs.msg import Path


OBSTACLES = [
    {'cx': 3.0, 'cy': 0.0, 'hx': 0.5, 'hy': 1.5},
    {'cx': 7.0, 'cy': -3.0, 'hx': 0.5, 'hy': 1.5},
    {'cx': 5.0, 'cy': 3.0, 'hx': 1.5, 'hy': 0.5},
    {'cx': 9.0, 'cy': 0.0, 'hx': 0.6, 'hy': 0.6},
]
MARGIN = 0.6

WAYPOINT_TOLERANCE = 0.4
FORWARD_SPEED = 1.0
TURN_SPEED = 1.4

DETECT_RANGE = 1.3         # start turning when this close to an obstacle's edge
DETECT_CONE_DEG = 50
CLEAR_CONE_DEG = 55
CLEAR_SUSTAIN_TICKS = 5    # ~0.8s of being clear before we stop turning
CLEAR_DRIVE_DISTANCE = 1.0

STUCK_CHECK_TICKS = 25
STUCK_DISTANCE_THRESH = 0.12
RECOVERY_TICKS = 20


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


def nearest_point_and_normal(px, py, obs):
    hx, hy = obs['hx'] + MARGIN, obs['hy'] + MARGIN
    nx = max(obs['cx'] - hx, min(px, obs['cx'] + hx))
    ny = max(obs['cy'] - hy, min(py, obs['cy'] + hy))
    dx, dy = px - nx, py - ny
    dist = math.hypot(dx, dy)
    if dist > 1e-6:
        return (dx / dist, dy / dist), dist
    fx, fy = px - obs['cx'], py - obs['cy']
    fdist = math.hypot(fx, fy) or 1.0
    return (fx / fdist, fy / fdist), 0.0


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

        self.state = 'SEEKING'   # SEEKING, TURNING, CLEARING
        self.turn_direction = 1.0
        self.clear_counter = 0
        self.clear_drive_start = None

        self.check_pos = None
        self.stuck_counter = 0
        self.recovering = False
        self.recovery_counter = 0

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

    def nearest_obstacle_ahead(self):
        # Returns (angle_diff, signed_offset, dist, normal) for the closest
        # relevant obstacle, using its real rectangular shape
        best = None
        for obs in OBSTACLES:
            (nx_dir, ny_dir), dist = nearest_point_and_normal(self.current_x, self.current_y, obs)
            if dist > DETECT_RANGE:
                continue
            # direction FROM vehicle TOWARD the obstacle is opposite the normal
            angle_to_obstacle = math.atan2(-ny_dir, -nx_dir)
            signed_offset = normalize_angle(angle_to_obstacle - self.current_yaw)
            angle_diff = abs(signed_offset)
            if best is None or dist < best[2]:
                best = (angle_diff, signed_offset, dist, (nx_dir, ny_dir))
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

        if self.recovering:
            self.recovery_counter += 1
            cmd.linear.x = -0.3
            cmd.angular.z = TURN_SPEED * self.turn_direction
            self.cmd_pub.publish(cmd)
            if self.recovery_counter >= RECOVERY_TICKS:
                self.recovering = False
                self.state = 'SEEKING'
                self.check_pos = None
                self.get_logger().info('Recovery complete, resuming navigation.')
            return

        goal_x, goal_y = self.waypoints[self.current_index]
        dx = goal_x - self.current_x
        dy = goal_y - self.current_y
        distance = math.hypot(dx, dy)

        if distance < WAYPOINT_TOLERANCE:
            self.get_logger().info(f'Reached waypoint {self.current_index + 1}/{len(self.waypoints)}.')
            self.current_index += 1
            self.state = 'SEEKING'
            self.check_pos = None
            if self.current_index >= len(self.waypoints):
                self.finished = True
                self.cmd_pub.publish(Twist())
                self.get_logger().info('All waypoints reached. Done.')
            return

        obstacle = self.nearest_obstacle_ahead()

        # ---------------- TURNING ----------------
        if self.state == 'TURNING':
            still_blocked = obstacle is not None and obstacle[0] < math.radians(CLEAR_CONE_DEG)
            if still_blocked:
                self.clear_counter = 0
                cmd.angular.z = TURN_SPEED * self.turn_direction
                cmd.linear.x = 0.0
                self.cmd_pub.publish(cmd)
                return
            else:
                self.clear_counter += 1
                cmd.angular.z = TURN_SPEED * self.turn_direction * 0.4
                cmd.linear.x = 0.0
                self.cmd_pub.publish(cmd)
                if self.clear_counter >= CLEAR_SUSTAIN_TICKS:
                    self.state = 'CLEARING'
                    self.clear_drive_start = (self.current_x, self.current_y)
                    self.get_logger().info('Clear of obstacle - driving forward to get past it.')
                return

        # ---------------- CLEARING ----------------
        if self.state == 'CLEARING':
            traveled = math.hypot(self.current_x - self.clear_drive_start[0],
                                   self.current_y - self.clear_drive_start[1])
            reblocked = obstacle is not None and obstacle[0] < math.radians(DETECT_CONE_DEG)
            if reblocked:
                self.state = 'TURNING'
                self.turn_direction = -1.0 if obstacle[1] > 0 else 1.0
                self.get_logger().info('Obstacle reappeared while clearing - turning again.')
                return
            if traveled >= CLEAR_DRIVE_DISTANCE:
                self.state = 'SEEKING'
                self.check_pos = None
                self.get_logger().info('Finished clearing obstacle, resuming toward waypoint.')
            else:
                cmd.linear.x = FORWARD_SPEED
                cmd.angular.z = 0.0
                self.cmd_pub.publish(cmd)
                if self.check_if_stuck():
                    self.recovering = True
                    self.recovery_counter = 0
                    self.get_logger().info('Stuck while clearing! Backing up.')
                return

        # ---------------- SEEKING ----------------
        if obstacle is not None and obstacle[0] < math.radians(DETECT_CONE_DEG):
            angle_diff, signed_offset, dist, normal = obstacle
            self.state = 'TURNING'
            self.turn_direction = -1.0 if signed_offset > 0 else 1.0
            self.clear_counter = 0
            self.get_logger().info(f'Obstacle detected at {dist:.2f}m - turning.')
            return

        goal_heading = math.atan2(dy, dx)
        angle_error = normalize_angle(goal_heading - self.current_yaw)
        aligned = abs(angle_error) < math.radians(25)

        cmd.angular.z = max(-TURN_SPEED, min(TURN_SPEED, 2.0 * angle_error))
        cmd.linear.x = FORWARD_SPEED if aligned else 0.1
        self.cmd_pub.publish(cmd)

        if aligned:
            if self.check_if_stuck():
                self.state = 'TURNING'
                self.turn_direction = 1.0 if (self._tick_count % 2 == 0) else -1.0
                self.get_logger().info('Stuck while seeking! Forcing a turn.')
        else:
            self.check_pos = None

        self._tick_count += 1
        if self._tick_count % 10 == 0:
            self.get_logger().info(
                f'[STATUS] state={self.state} goal={self.current_index + 1}/{len(self.waypoints)} '
                f'dist_to_goal={distance:.2f}m pos=({self.current_x:.2f},{self.current_y:.2f})')


def main(args=None):
    rclpy.init(args=args)
    node = Navigator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
