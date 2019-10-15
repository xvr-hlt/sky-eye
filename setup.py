from io import open

from setuptools import find_packages, setup

version = '0.0.1'

with open('README.md', 'r', encoding='utf-8') as f:
    readme = f.read()

REQUIRES = [
    'numpy==1.17.2',
    'scipy==1.3.1',
    'torch==1.2.0',
    'albumentations==0.3.3',
    'click==7.0',
    'wandb==0.8.12'
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

