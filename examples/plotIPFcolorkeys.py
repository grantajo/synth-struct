# synth-struct/examples/plotIPFcolorkeys.py

"""
This example plots all of the IPF colorkeys for future reference.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from orix import plot
from orix.quaternion import symmetry

from synth_struct.plotting import ipfcolorkeys

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
outdir = project_root / "output/IPFcolorkeys/"
outdir.mkdir(exist_ok=True)

print("=" * 17, "IPF Color Keys", "=" * 17)

ipfcolorkeys.plot_all_colorkeys(outdir)

print(f"Saved IPF color keys to: \n{outdir}")
print("-" * 50)
