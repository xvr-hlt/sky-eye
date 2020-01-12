from io import open

from setuptools import find_packages, setup

version = '0.0.1'

REQUIRES = [
    'numpy',
    'scipy',
    'albumentations',
    'click',
    'wandb',
    'shapely',
    'ttach',
    'pandas',
    'pytorch_toolbelt',
    'fire==0.2.1',
    'pillow==6.2.*',
    'imantics==0.1.11'
]

setup(
    name='xv',
    version=version,
    description='',
    author='Xavier Holt',
    author_email='holt.xavier@gmail.com',
    maintainer='Xavier Holt',
    maintainer_email='holt.xavier@gmail.com',
    url='https://github.com/xvr-hlt/sky-eye',
    install_requires=REQUIRES,
    packages=find_packages(),
    entry_points = {
        'console_scripts': ['sky-eye=xv:main'],
    }
)
