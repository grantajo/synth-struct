# synth-struct/examples/plotIPFcolorkeys.py

"""
This example plots all of the IPF colorkeys for future reference.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from orix import plot
from orix.quaternion import symmetry

from synth_struct.plotting import ipfcolorkeys

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
outdir = project_root / "output/IPFcolorkeys/"
outdir.mkdir(exist_ok=True)

ipfcolorkeys.plotIPFcolorkeys(outdir)


