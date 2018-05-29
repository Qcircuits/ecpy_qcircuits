#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

import os.path
import sys

sys.path.insert(0, os.path.abspath('.'))
from exopy_qcircuits.version import __version__

PROJECT_NAME = 'exopy_qcircuits'

setup(
    name=PROJECT_NAME,
    description='Template for Exopy extension packages',
    version=__version__,
    long_description=open('README.md').read(),
    author='see AUTHORS',
    author_email='',
    url='',  # URL of the git repository
    download_url='',  # URL of the zip or tar.gz master branch
    keywords='experiment automation GUI',
    license='BSD',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Physics',
        'Programming Language :: Python :: 3.6',
        ],
    zip_safe=False,
    packages=find_packages(exclude=['tests', 'tests.*']),
    package_data={'': ['*.enaml']},
    requires=['exopy'],
    install_requires=['exopy'],
    entry_points={
        'exopy_package_extension':
        'exopy_qcircuits = %s:list_manifests' % PROJECT_NAME}
)
