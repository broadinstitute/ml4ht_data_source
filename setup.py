from setuptools import find_packages, setup

setup(
    name="ml4ht",
    version="0.0.5dev8",
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=["numpy", "pandas", "torch"],
)
