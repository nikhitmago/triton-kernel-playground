import os
import sys

# Make the kernel modules under src/ (vector_add.py, fused_softmax.py)
# importable from tests living under tests/.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
