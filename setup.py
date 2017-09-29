#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst
import sys

# import ah_bootstrap
from setuptools import setup



setup(name='pschitt',
      version=1.0,
      description="DESCRIPTION",
      # these should be minimum list of what is needed to run (note
      # don't need to list the sub-dependencies like numpy, since
      # astropy already depends on it)
      install_requires=[
          # 'astropy',
          # 'numpy',
          # 'scipy>=0.19',
          # 'matplotlib>=2.0',
          # 'numba'
      ],
      packages=['pschitt'],
      package_data={'pschitt': ['data/*.dat']},
      tests_require=['pytest'],
      author='Thomas Vuillaume',
      author_email='thomas.vuillaume@lapp.in2p3.fr',
      license='BSD3',
      url='https://github.com/vuillaut/pschitt',
      long_description='',
      classifiers=[
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD License',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering :: Astronomy',
          'Development Status :: 3 - Alpha',
      ]
      )
