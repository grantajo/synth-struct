import sys
sys.path.insert(0, '../src')

from texture import Texture
from euler_quat_converter import euler_to_quat, quat_to_euler
import numpy as np

# Create some test orientations
orientations = Texture.random_orientations(5, seed=54)

print("Original Euler angles:")
for gid, angles in orientations.items():
    print(f"  Grain {gid}: {np.degrees(angles)}")

# Convert to quaternions
quats = euler_to_quat(orientations)

# Convert back to Euler angles
recovered = quat_to_euler(quats)

print("\nRecovered Euler angles:")
for gid, angles in recovered.items():
    print(f"  Grain {gid}: {np.degrees(angles)}")

# Check if they match
print("\nDifference (should be near zero):")
for gid in orientations.keys():
    diff = np.abs(orientations[gid] - recovered[gid])
    print(f"  Grain {gid}: {diff}")
