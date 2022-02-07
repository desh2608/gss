from setuptools import setup, find_packages

# To use a consistent encoding
from codecs import open
from os import path
import numpy

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="gss",
    version="0.1.0",
    description="Guided Source Separation (based on pb_chime5)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/desh2608/gss",
    author="Desh Raj",
    author_email="r.desh26@gmail.com",
    keywords="speech enhancement gss",  # Optional
    packages=find_packages(exclude=["contrib", "docs", "tests"]),  # Required
    install_requires=[
        "nara_wpe>=0.0.6",
        "Cython",
        "numpy==1.20.3",
        "scikit-learn==0.19.2",  # Don't upgrade scikit-learn (sklearn.mixture.gaussian_mixture is removed)
        "lhotse @ git+http://github.com/lhotse-speech/lhotse",
        "plz @ git+http://github.com/pzelasko/plz",
    ],  # Optional
    extras_require={  # Optional
        "dev": ["check-manifest"],
        "test": ["coverage"],
    },
    include_dirs=[numpy.get_include()],
)
