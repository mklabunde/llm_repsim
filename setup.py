# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import find_packages, setup

with open("README.md") as f:
    readme = f.read()

# with open("LICENSE") as f:
#     license = f.read()

setup(
    name="llmcomp",
    version="anonymous",
    description="anonymous",
    long_description=readme,
    author="anonymous",
    author_email="anonymous",
    url="anonymous",
    # license=license,
    packages=find_packages(exclude=("tests", "docs")),
)
