import os

from setuptools import setup, find_packages

with open(os.sep.join(
        [os.path.dirname(os.path.abspath(__file__)), "paulicirq", "_version.py"]
)) as vfile:
    import re

    text = vfile.read()
    version_pattern = re.compile(
        r"""__version__ = (["'])(?P<version>[\w.\-]+)(\1)"""
    )
    __version__ = re.match(version_pattern, text).group("version")

description = "Toolkit for quantum computing based on Cirq."
long_description = open("README.md").read()

install_requires = [
    "cirq==0.5.0",
    "ddt",
    "openfermion>=0.10.0",
    "mpmath"
    # "openfermioncirq"
]

setup(

    name="paulicirq",
    version=__version__,

    author="psrit",
    author_email="xiaojx13@outlook.com",
    url="https://github.com/Psrit/paulicirq",

    long_description=long_description,
    long_description_content_type='text/markdown',

    classifiers=[
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Operating System :: Unix",
        "Programming Language :: Python",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],

    packages=find_packages(exclude=("tests*",)),

    install_requires=install_requires

)
