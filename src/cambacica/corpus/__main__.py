"""Package entry point for python -m cambacica.corpus."""

import sys
from cambacica.corpus.cli import main

if __name__ == "__main__":
    import os
    sys.stdout.flush()
    sys.stderr.flush()
    ret = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(ret)
