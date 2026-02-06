# synth_struct/src/synth_struct/_cpp_extensions/__init__.py
"""
C++ accelerated functions for synth_struct.
"""

try:
    from .aniso_voronoi_eigen import aniso_voronoi_assignment

    EIGEN_AVAILABLE = True
except ImportError as e:
    EIGEN_AVAILABLE = False
    print(f"Warning: C++ Eigen extension not available: {e}")
    print("Falling back to NumPy implementation.")
    aniso_voronoi_assignment = None

__all__ = ["aniso_voronoi_assignment", "EIGEN_AVAILABLE"]
