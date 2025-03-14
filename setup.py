from setuptools import find_packages, setup

setup(
    name="pytorch_slam",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.20.0",
        "opencv-python>=4.5.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "tqdm>=4.60.0",
    ],
    author="Houssem Boulahbal",
    author_email="houssem.boulahbal@gmail.com",
    description="A PyTorch-based library for SLAM",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/pytorch_slam",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.8",
)
