"""`python -m ampzip` runs the simplifier CLI; `python -m ampzip.webservice` runs the web service."""
import sys

from .simplifier.cli import main

if __name__ == "__main__":
    sys.exit(main())
