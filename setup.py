from setuptools import setup, find_packages

setup(
    name="pose_init",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "opencv-python",
    ],
)
