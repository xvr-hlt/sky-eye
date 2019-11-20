from io import open

from setuptools import find_packages, setup

version = '0.0.1'

with open('README.md', 'r', encoding='utf-8') as f:
    readme = f.read()

REQUIRES = [
    'numpy',
    'scipy',
    'albumentations',
    'click',
    'wandb',
    'shapely',
    'segmentation-models-pytorch',
    'ttach'
    
    #'git+https://github.com/qubvel/segmentation_models.pytorch',
    #'git+https://github.com/qubvel/ttach'
]

setup(
    name='xv',
    version=version,
    description='',
    long_description=readme,
    author='Xavier Holt',
    author_email='holt.xavier@gmail.com',
    maintainer='Xavier Holt',
    maintainer_email='holt.xavier@gmail.com',
    url='https://github.com/xvr-hlt/sky-eye',
    install_requires=REQUIRES,
    packages=find_packages(),
)

