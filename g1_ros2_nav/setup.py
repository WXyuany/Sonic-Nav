from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'g1_ros2_nav'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Developer',
    maintainer_email='user@example.com',
    description='ROS2 navigation bridge for GR00T WBC G1 humanoid robot.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'g1_bridge = g1_ros2_nav.g1_bridge:main',
            'cmd_vel_bridge = g1_ros2_nav.cmd_vel_bridge:main',
        ],
    },
)
