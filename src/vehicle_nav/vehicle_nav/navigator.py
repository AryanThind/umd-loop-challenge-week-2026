import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, qos_profile_sensor_data
from geometry_msgs.msg import Twist, PoseArray
from nav_msgs.msg import Path


# Obstacles as rectangles, matching vehicle_world.sdf box sizes (1 x 3),
# so we can reason about them as actual shapes, not just circles.
# cx, cy = center, hx, hy = half-width/half-height
OBSTACLES = [
    {'cx': 3.0, 'cy': 0.0, 'hx': 0.5, 'hy': 1.5},
    {'cx': 6.0, 'cy': -2.0, 'hx': 0.5, 'hy': 1.5},
]

MARGIN = 0.8            # safety inflation around each obstacle (vehicle size + buffer)
STANDOFF = 1.0           # desired distance to keep from an obstacle while wall-following
ENTER_DIST = 0.35        # how close (to the inflated boundary) triggers wall-following
EXIT_CLEAR_DIST = 1.3    # must be at least this far from the obstacle to exit wall-following

WAYPOINT_TOLERANCE = 0.4
FORWARD_SPEED = 0.6
FOLLOW_SPEED = 0.35
TURN_SPEED = 1.0

STUCK_CHECK_TICKS = 15
STUCK_DISTANCE_THRESH = 0.08
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
        normal = (dx / dist, dy / dist)
    else:
        # We're inside the inflated box (shouldn't normally happen); push
        # away from the obstacle's center as a fallback
        fx, fy = px - obs['cx'], py - obs['cy']
        fdist = math.hypot(fx, fy) or 1.0
        normal = (fx / fdist, fy / fdist)
    return nx, ny, normal, dist


def point_in_inflated_rect(px, py, obs):
    hx, hy = obs['hx'] + MARGIN, obs['hy'] + MARGIN
    return abs(px - obs['cx']) <= hx and abs(py - obs['cy']) <= hy


def path_blocked_by(obs, x1, y1, x2, y2, samples=12):
    for i in range(samples + 1):
        t = i / samples
        px = x1 + (x2 - x1) * t
        py = y1 + (y2 - y1) * t
        if point_in_inflated_rect(px, py, obs):
            return True
    return False


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

        self.state = 'GO_TO_GOAL'   # or 'WALL_FOLLOW'
        self.wall_obstacle = None
        self.wall_side = 0.0        # +1 = counter-clockwise, -1 = clockwise

        self.check_pos = None
        self.stuck_counter = 0
        self.recovering = False
        self.recovery_counter = 0
        self.recovery_direction = 1.0

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
            cmd.angular.z = TURN_SPEED * self.recovery_direction
            self.cmd_pub.publish(cmd)
            if self.recovery_counter >= RECOVERY_TICKS:
                self.recovering = False
                self.state = 'GO_TO_GOAL'
                self.wall_obstacle = None
                self.get_logger().info('Recovery maneuver complete, resuming navigation.')
            return

        goal_x, goal_y = self.waypoints[self.current_index]
        dx = goal_x - self.current_x
        dy = goal_y - self.current_y
        distance = math.hypot(dx, dy)

        if distance < WAYPOINT_TOLERANCE:
            self.get_logger().info(f'Reached waypoint {self.current_index + 1}/{len(self.waypoints)}.')
            self.current_index += 1
            self.state = 'GO_TO_GOAL'
            self.wall_obstacle = None
            self.check_pos = None
            if self.current_index >= len(self.waypoints):
                self.finished = True
                self.cmd_pub.publish(Twist())
                self.get_logger().info('All waypoints reached. Done.')
            return

        if self.check_if_stuck():
            self.recovering = True
            self.recovery_counter = 0
            self.recovery_direction = -self.wall_side if self.wall_side != 0 else 1.0
            self.get_logger().info('Stuck detected! Backing up and turning to recover.')
            return

        if self.state == 'WALL_FOLLOW':
            obs = self.wall_obstacle
            nx, ny, normal, dist_to_boundary = nearest_point_and_normal(self.current_x, self.current_y, obs)
            path_clear = not path_blocked_by(obs, self.current_x, self.current_y, goal_x, goal_y)

            if path_clear and dist_to_boundary > EXIT_CLEAR_DIST:
                self.state = 'GO_TO_GOAL'
                self.wall_obstacle = None
                self.get_logger().info('Path to waypoint is clear, leaving wall-follow mode.')
            else:
                side = self.wall_side
                tangent = (-normal[1] * side, normal[0] * side)
                standoff_error = dist_to_boundary - STANDOFF
                # blend: mostly move along the wall, nudge toward/away to hold standoff distance
                desired_x = tangent[0] + normal[0] * (-standoff_error) * 0.6
                desired_y = tangent[1] + normal[1] * (-standoff_error) * 0.6
                desired_heading = math.atan2(desired_y, desired_x)
                angle_error = normalize_angle(desired_heading - self.current_yaw)
                cmd.angular.z = max(-TURN_SPEED, min(TURN_SPEED, 2.0 * angle_error))
                cmd.linear.x = FOLLOW_SPEED if abs(angle_error) < math.radians(60) else 0.1
                self.cmd_pub.publish(cmd)
                return

        # --- GO_TO_GOAL state ---
        react_obstacle = None
        for obs in OBSTACLES:
            nx, ny, normal, dist_to_boundary = nearest_point_and_normal(self.current_x, self.current_y, obs)
            blocked = path_blocked_by(obs, self.current_x, self.current_y, goal_x, goal_y)
            if dist_to_boundary < ENTER_DIST or (blocked and dist_to_boundary < 3.0):
                react_obstacle = obs
                react_normal = normal
                break

        if react_obstacle is not None:
            # Decide which side to go around: whichever side points more
            # toward the goal direction
            goal_heading = math.atan2(dy, dx)
            tangent_ccw = (-react_normal[1], react_normal[0])
            heading_ccw = math.atan2(tangent_ccw[1], tangent_ccw[0])
            diff_ccw = abs(normalize_angle(heading_ccw - goal_heading))
            side = 1.0 if diff_ccw < math.pi / 2 else -1.0

            self.state = 'WALL_FOLLOW'
            self.wall_obstacle = react_obstacle
            self.wall_side = side
            self.get_logger().info(
                f'Obstacle encountered, entering wall-follow (side={"CCW" if side > 0 else "CW"}).')
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
