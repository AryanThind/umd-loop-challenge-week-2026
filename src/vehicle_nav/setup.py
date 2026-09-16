import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'vehicle_nav'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.sdf')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='aryan-thind',
    maintainer_email='aryan-thind@todo.todo',
    description='Autonomous vehicle waypoint navigation with obstacle avoidance in Gazebo',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'waypoint_generator = vehicle_nav.waypoint_generator:main',
            'navigator = vehicle_nav.navigator:main',
        ],
    },
)
