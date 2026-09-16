import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory('vehicle_nav')
    world_path = os.path.join(pkg_share, 'worlds', 'vehicle_world.sdf')

    gazebo = ExecuteProcess(
        cmd=['gz', 'sim', '--render-engine', 'ogre', '-r', world_path],
        output='screen'
    )

    cmd_vel_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['/model/vehicle/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist'],
        output='screen'
    )

    pose_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['/model/vehicle/pose@geometry_msgs/msg/PoseArray@gz.msgs.Pose_V'],
        output='screen'
    )

    waypoint_generator = Node(
        package='vehicle_nav',
        executable='waypoint_generator',
        output='screen'
    )

    navigator = Node(
        package='vehicle_nav',
        executable='navigator',
        output='screen'
    )

    return LaunchDescription([
        gazebo,
        TimerAction(period=2.0, actions=[cmd_vel_bridge, pose_bridge]),
        TimerAction(period=4.0, actions=[waypoint_generator]),
        TimerAction(period=5.0, actions=[navigator]),
    ])
