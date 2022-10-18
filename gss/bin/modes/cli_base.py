import logging

import click


@click.group()
def cli():
    """
    The shell entry point to `gss`, a tool and a library for GSS-based front-end enhancement.
    """
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
    )
