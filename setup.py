
import os
from setuptools import setup, find_packages


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name="LabNanofisica",
    version="0.1",
    author="Federico Barabas",
    author_email="fede.barabas@gmail.com",
    description=("Script collection for single-molecule and imaging "
                 "experiments in general."),
    license="BSD",
    keywords="single-molecule imaging",
    url="https://github.com/fedebarabas/LabNanofisica",
    packages=find_packages(),
#    long_description=read('README'),
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python :: 3.4",
    ],
)
