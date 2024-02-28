import os
from setuptools import setup, find_packages
cwd = os.getcwd()

setup(
    name='openbtmixing',
    version='1.0',
    packages=find_packages(where='openbtmixing'),
    package_dir={'': 'openbtmixing'},
    package_data={'': ['src/*.cpp', 'src/*.h']},
)
