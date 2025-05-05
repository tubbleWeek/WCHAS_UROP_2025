from setuptools import find_packages, setup

package_name = 'project_tools'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='william',
    maintainer_email='you@example.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'initial_pose_setter = project_tools.initial_pose_set:main',
            'gazebo_model_tracker = project_tools.gazebo_model_subscriber:main',
            'map_transformation_node = project_tools.map_to_odom_tf:main'
        ],
    },
)
