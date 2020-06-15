import sys


def error(message):
    """Print an error message."""
    print("[ERROR] %s" % (message,))


def error_exit(message):
    """Print an error message and exit"""
    error(message)
    sys.exit(-1)