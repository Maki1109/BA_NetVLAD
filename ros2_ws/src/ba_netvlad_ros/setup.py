from setuptools import find_packages, setup
from glob import glob

package_name = 'ba_netvlad_ros'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Luong Nguyen Viet Sang',
    maintainer_email='23020762@vnu.edu.vn',
    description='Live BA-NetVLAD loop-closure detection node for TurtleBot4.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lcd_node = ba_netvlad_ros.lcd_node:main',
        ],
    },
)
