#!/usr/bin/env python3
"""
Use this script like:
$ gss enhance --help
$ gss utils --help
"""

# Note: we import all the CLI modes here so they get auto-registered
#       in the main CLI entry-point. Then, setuptools is told to
#       invoke the "cli()" method from this script.
from gss.bin.modes import *
