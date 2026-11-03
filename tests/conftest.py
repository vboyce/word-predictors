import sys
from pathlib import Path

import torch

# Lets tests import the tiny_lm helper module.
sys.path.insert(0, str(Path(__file__).parent))

# Leave most of the machine's cores free while tests run.
torch.set_num_threads(2)
