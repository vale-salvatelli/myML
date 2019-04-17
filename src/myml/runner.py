#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This is a skeleton file for a runner
"""

import argparse
import sys
import logging

import pandas as pd

from myml import __version__

__author__ = "vale-salvatelli"
__copyright__ = "vale-salvatelli"
__license__ = "mit"

_logger = logging.getLogger(__name__)


def load_dataset(path):
    """Example function

    Args:
      path (str): string containing the path

    Returns:
      df: pandas.DataFrame
    """
    return pd.read_csv(path)


def parse_args(args):
    """Parse command line parameters

    Args:
      args ([str]): command line parameters as list of strings

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(
        description="MyML Package")
    parser.add_argument(
        '--version',
        action='version',
        version='myML {ver}'.format(ver=__version__))
    parser.add_argument(
        '-v',
        '--verbose',
        dest="loglevel",
        help="set loglevel to INFO",
        action='store_const',
        const=logging.INFO)
    parser.add_argument(
        '-vv',
        '--very-verbose',
        dest="loglevel",
        help="set loglevel to DEBUG",
        action='store_const',
        const=logging.DEBUG)
    parser.add_argument(
        dest="path",
        help="Path to the dataset that you want to process",
        type=str,
        metavar="STR")
    return parser.parse_args(args)


def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(level=loglevel, stream=sys.stdout,
                        format=logformat, datefmt="%Y-%m-%d %H:%M:%S")


def main(args):
    """Main entry point allowing external calls

    Args:
      args ([str]): command line parameter list
    """
    args = parse_args(args)
    setup_logging(args.loglevel)
    _logger.info("Computation starts")
    _logger.debug("Loading file...")
    print("The {}-th Fibonacci number is {}".format(args.n, fib(args.n)))
    _logger.info("Computation ends")


def run():
    """Entry point for console_scripts
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
