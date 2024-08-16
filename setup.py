from setuptools import find_packages, setup

package_name = 'aruco_ps'

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
    maintainer='root',
    maintainer_email='anon@ubicoders.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "global_aruco_node = aruco_ps.psnode_global_aruco:main",
            "body_aruco_node = aruco_ps.psnode_body_aruco:main",    
        ],
    },
)
