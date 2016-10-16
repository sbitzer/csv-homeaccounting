# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 18:06:36 2016

@author: bitzer
"""

from setuptools import setup

setup(
    name='csv-homeaccounting',
    version='0.1.0',
    author='Sebastian Bitzer',
    author_email='official@sbitzer.eu',
    packages=['homeaccounting'],
    description=('Keeps track of finances at home based on csv-files that can '
                 'typically be downloaded from online banking websites.'),
    classifiers=[
                'Development Status :: 3 - Alpha',
                'Operating System :: OS Independent',
                'Intended Audience :: End Users/Desktop',
                'License :: OSI Approved :: BSD License',
                'Programming Language :: Python :: 3',
                'Topic :: Office/Business :: Financial :: Accounting',
                 ]
)