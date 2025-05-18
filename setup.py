from setuptools import setup, find_packages

setup(
    name="ramp_recovery_system",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
    ],
    python_requires=">=3.8",
    author="Your Name",
    description="A system for recovering ramp geometry from GPS data",
) 