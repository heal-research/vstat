from skbuild import setup  # This line replaces 'from setuptools import setup'
import platform

setup(
    name="vstat",
    version="1.0.0",
    description="vectorized statistics library",
    author='Bogdan Burlacu',
    packages = ['vstat'],
    python_requires=">=3.8",
    cmake_args=[f'-DCPM_USE_LOCAL_PACKAGES=1', '-Dvstat_BUILD_PYTHON=1']
)
