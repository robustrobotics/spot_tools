import pathlib

from setuptools import find_packages, setup

package_name = "spot_tools_ros"


def get_share_info(top_level, pattern, dest=None):
    curr_path = pathlib.Path(__file__).absolute().parent
    dest = pathlib.Path("share") / package_name if dest is None else pathlib.Path(dest)
    files = [x.relative_to(curr_path) for x in (curr_path / top_level).rglob(pattern)]
    parent_map = {}
    for x in files:
        key = str(dest / x.parent)
        parent_map[key] = parent_map.get(key, []) + [str(x)]
    return [(k, v) for k, v in parent_map.items()]


launch_files = get_share_info("launch", "*.launch.yaml")
config_files = get_share_info("config", "*.yaml")
config_files_csv = get_share_info("config", "*.csv")
rviz_files = get_share_info("rviz", "*.rviz")

data_files = (
    [
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ]
    + launch_files
    + config_files
    + config_files_csv
    + rviz_files
)


setup(
    name=package_name,
    version="0.0.0",
    package_dir={"": "src"},
    packages=find_packages("src"),
    data_files=data_files,
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="aaron",
    maintainer_email="aaronray@mit.edu",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "spot_executor_node = spot_tools_ros.spot_executor_ros:main",
            "spot_sensor_node = spot_tools_ros.spot_sensors:main",
        ],
    },
)
