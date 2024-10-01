from setuptools import setup, find_packages
import os

_CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))

with open('requirements.txt') as f:
    required = f.read().splitlines()

try:
    README = open(os.path.join(_CURRENT_DIR, "README.md"), encoding="utf-8").read()
except IOError:
    README = ""

setup(
    name="cvit",
    version="0.0.1",
    url=" ",
    author="Sifan Wang, Hanwen Wang and Shyam Sankaran",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=required,
    license="Apache 2.0",
    description=" ",
    long_description=open(os.path.join(_CURRENT_DIR, "README.md")).read(),
    long_description_content_type="text/markdown",
)