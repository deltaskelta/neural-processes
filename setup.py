#!/usr/bin/env python

from distutils.core import setup

setup(
    name="neural processes",
    version="1.0",
    description="Neural Processes Implementation",
    author="Emilien Dupont, Jeff Willette",
    url="https://github.com/deltaskelta/neural-processes",
    package_data={"neural_processes": ["py.typed"]},
    packages=["neural_processes"],
)
