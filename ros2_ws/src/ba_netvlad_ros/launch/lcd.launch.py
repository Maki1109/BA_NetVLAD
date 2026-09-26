"""Launch the BA-NetVLAD live loop-closure node.

Usage:
    ros2 launch ba_netvlad_ros lcd.launch.py
    ros2 launch ba_netvlad_ros lcd.launch.py params_file:=/path/to/my_params.yaml

BA_NETVLAD_REPO_ROOT must be set (in the environment, or overridden below) to
the path of the ba_netvlad Python repo so the node can `import ba_netvlad`
without duplicating pipeline code inside this ROS package.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_params = os.path.join(
        get_package_share_directory('ba_netvlad_ros'), 'config', 'params.yaml')

    params_file_arg = DeclareLaunchArgument(
        'params_file', default_value=default_params,
        description='Path to the parameters YAML for ba_netvlad_lcd_node.')

    repo_root_arg = DeclareLaunchArgument(
        'ba_netvlad_repo_root',
        default_value=os.environ.get('BA_NETVLAD_REPO_ROOT', ''),
        description='Path to the ba_netvlad Python repo (contains the '
                    'ba_netvlad/ package); required so the node can import it.')

    return LaunchDescription([
        params_file_arg,
        repo_root_arg,
        SetEnvironmentVariable('BA_NETVLAD_REPO_ROOT', LaunchConfiguration('ba_netvlad_repo_root')),
        Node(
            package='ba_netvlad_ros',
            executable='lcd_node',
            name='ba_netvlad_lcd_node',
            output='screen',
            emulate_tty=True,
            parameters=[LaunchConfiguration('params_file')],
        ),
    ])
