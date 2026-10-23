"""Kelly Auction Game Simulation Package.

Reproduces the learning dynamics and simulation results from:
"Learning in Proportional Allocation Auction Games" (Ben Mazziane et al., 2026).
"""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.makedirs("/tmp/matplotlib", exist_ok=True)

__version__ = "0.1.0"

