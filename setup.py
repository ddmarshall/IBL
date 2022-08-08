# -*- coding: utf-8 -*-

# Based on: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE.md') as f:
    license = f.read()

setup(
    name='pyBL',
    version='0.0.4-develop',
    description='Integral boundary layer method implementations for use with inviscid flow solvers.',
    long_description=readme,
    author='David D. Marshall',
    author_email='ddmarshall@gmail.com',
    url='https://github.com/ddmarshall/IBL',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

