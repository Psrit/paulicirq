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


def _load_requirements():
    lines = open("requirements.txt").readlines()
    terms = []
    for line in lines:
        m = re.match(r"(?P<term>[^\s#]*)\s*(?P<comment>#.*)?", line)
        _term = m.groupdict()["term"].strip()
        if _term:
            terms.append(_term)
    return terms


install_requires = _load_requirements()

setup(

    name="paulicirq",
    version=__version__,

    author="xiaojx",
    author_email="xiaojx13@outlook.com",
    url="https://github.com/Psrit/paulicirq",

    long_description=long_description,
    long_description_content_type='text/markdown',

    classifiers=[
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Operating System :: Unix",
        "Programming Language :: Python :: 3.8",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],

    packages=find_packages(exclude=("tests*",)),

    install_requires=install_requires

)
