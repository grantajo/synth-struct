import sys
sys.path.insert(0, '../src')
import os

import numpy as np
import matplotlib.pyplot as plt

from orix import plot
from orix.quaternion import symmetry

outdir = '../output/IPFColorKeys'
os.makedirs(outdir, exist_ok=True)

"""
This file saves all of the IPF Color Keys from orix to a folder in the output directory
"""

# Create
pg_laue = [
    symmetry.Ci, # 
    symmetry.C2h,
    symmetry.D2h,
    symmetry.S6,
    symmetry.D3d,
    symmetry.C4h,
    symmetry.C6h,
    symmetry.D6h,
    symmetry.Th,
    symmetry.Oh
]

for pg in pg_laue:
    ipfkey = plot.IPFColorKeyTSL(pg)
    fig = ipfkey.plot(return_figure=True)
    
    name = pg.name.replace("/", "_")
    fig.savefig(f'{outdir}/{name}.png', dpi=150)
    plt.close(fig)
    
    
    

