# synth_struct/src/synth_struct/plotting/ipfcolorkeys.py

"""
This file saves all of the IPF Color Keys from orix to a folder in the output directory
"""

import sys
import os

import matplotlib.pyplot as plt
from orix import plot
from orix.quaternion import symmetry

sys.path.insert(0, "../src")
outdir = "../output/IPFColorKeys"
os.makedirs(outdir, exist_ok=True)

# List of all the point groups
pg_laue = [
    symmetry.Ci,  #
    symmetry.C2h,
    symmetry.D2h,
    symmetry.S6,
    symmetry.D3d,
    symmetry.C4h,
    symmetry.C6h,
    symmetry.D6h,
    symmetry.Th,
    symmetry.Oh,
]

# Plot all of the IPF color maps
for pg in pg_laue:
    ipfkey = plot.IPFColorKeyTSL(pg)
    fig = ipfkey.plot(return_figure=True)

    name = pg.name.replace("/", "_")
    fig.savefig(f"{outdir}/{name}.png", dpi=150)
    plt.close(fig)
