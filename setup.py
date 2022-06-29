# -*- coding: utf-8 -*-

# Based on: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE.md') as f:
    license = f.read()

setup(
    name='pyBL',
    version='0.1.0',
    description='Integral boundary layer method implementations for use with inviscid flow solvers.',
    long_description=readme,
    author='Kenneth Reitz',
    author_email='me@kennethreitz.com',
    url='https://github.com/ddmarshall/IBL',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

