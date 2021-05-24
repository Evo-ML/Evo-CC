import os
from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'An Open-Source Classification-Based Nature-Inspired Optimization Clustering Framework in Python'
LONG_DESCRIPTION = 'The framework is an open source and cross-platform framework implemented in Python which uses the classification technique with the evolutionary clustering approach provided by the EvoCluster framework. The goal of this framework is to provide a user-friendly and customizable implementation of theclassification-based evolutionary clusteringwhich can be utilized by experienced and non-experienced users for different applications. The framework can also be used by researchers who can benefit from the frameworkfor their research studies.'

# Setting up
setup(
       # the name must match the folder name 'EvoCC'
        name="EvoCC", 
        version=VERSION,
        author="Dang Trung Anh, Raneem Qaddoura",
        author_email="dangtrunganh@gmail.com, raneem.qaddoura@gmail.com",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages('EvoCC'),
        # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
        install_requires=[], 
        url='https://github.com/housecricket/evoCC',
        keywords=['python', 'first package'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)