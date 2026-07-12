import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin2"))

from operaxn.main import main

if __name__ == "__main__":
    sys.exit(main())
