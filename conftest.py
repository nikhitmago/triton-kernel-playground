import os
import sys

# Make the repo-root modules (vector_add.py, fused_softmax.py) importable
# from tests living under tests/.
sys.path.insert(0, os.path.dirname(__file__))
