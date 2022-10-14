from setuptools import setup, find_packages

# To use a consistent encoding
from codecs import open
from os import path
import numpy

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

dev_requires = [
    "flake8==5.0.4",
    "black==22.3.0",
    "isort==5.10.1",
    "pre-commit>=2.17.0,<=2.19.0",
]

setup(
    name="gss",
    version="0.3.0",
    description="Guided Source Separation (based on pb_chime5)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/desh2608/gss",
    author="Desh Raj",
    author_email="r.desh26@gmail.com",
    keywords="speech enhancement gss",  # Optional
    packages=find_packages(exclude=["contrib", "docs", "tests"]),  # Required
    install_requires=[
        "cached_property",
        "numpy",
        "lhotse @ git+http://github.com/lhotse-speech/lhotse",
    ],
    extras_require={
        "dev": dev_requires,
    },
    include_dirs=[numpy.get_include()],
)
