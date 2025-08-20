from setuptools import find_packages, setup

setup(
    name="spot_tools",
    version="0.0.1",
    url="",
    author="",
    author_email="",
    description="Tools for Spot",
    package_dir={"": "src"},
    packages=find_packages("src"),
    package_data={"": ["*.yaml", "*.jpg"]},
    install_requires=[
        "numpy",
        "matplotlib",
        "pyyaml",
        "pytest",
        "pre-commit",
        "bosdyn-client",
        "scikit-image",
        "opencv-python-headless",
        "shapely",
        "onnxruntime",
        "transforms3d",
    ],
)
