#!/usr/bin/env python3

from setuptools import setup

setup(name='Willump',
      version='0.1',
      description='Willump Is a Low-Latency Useful ML Platform',
      author='Peter Kraft',
      author_email='kraftp@cs.stanford.edu',
      url='https://github.com/stanford-futuredata/Willump',
      packages=['willump', 'willump.evaluation', 'willump.graph'],
      test_suite='tests',
     )