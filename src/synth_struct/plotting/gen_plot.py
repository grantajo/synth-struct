# synth-struct/src/synth_struct/plotting/gen_plot.py

"""
This is a series of plotting functions to help with plotting.

This is for general plotting, such as Grain IDs, etc.
"""

import numpy as np
import matplotlib.pyplot as plt

from .plot_utils import shuffle_display_grain_ids, create_grain_boundary_overlay


class Plotter:
    """
    Static methods for plotting microstructures with various options.
    """

    @staticmethod
    def plot_grain_ids(
        ax,
        micro,
        shuffle=False,
        seed=None,
        cmap="nipy_spectral",
        show_boundaries=False,
        boundary_color="black",
        boundary_width=0.5,
        slice_direction="z",
        slice_index=None,
        colorbar=True,
        title=None,
        **kwargs,
    ):
        """
        Plot grain IDs from a microstructure.

        Args:
        - ax: matplotlib axis
        - micro: Microstructure class object
        - shuffle: bool - SHuffle grain IDs for better color contrast (useful for lath microstructure)
        - cmap: str - Colormap name
        - show_boundaries: bool - Overlay grain boundaries
        - boundary_color: str - Color for brain boundaries
        - boundary_width: float - Line width for grain boundaries
        - slice_direction: str - For 3D: 'x', 'y', or 'z' (normal to slice)
        - slice_index: int or None - For 3D: index of slice. If None, uses middle
        - colorbar: bool - Show colorbar
        - title: str or None - Plot title
        - **kwargs: Additional arguments passed to imshow

        Returns:
        - Image artist from imshow

        Example:
            fig, ax = plt.subplots()
            Plotter.plot_grain_ids(ax, micro, shuffle=True, seed=42)
        """

        if shuffle:
            grain_ids = shuffle_display_grain_ids(
                micro.grain_ids, micro.num_grains, seed=seed
            )

        else:
            grain_ids = micro.grain_ids

        # For 3D, take middle slice if not specified
        if grain_ids.ndim == 3:
            slice_direction = slice_direction.lower()
            if slice_direction == "z":
                # XY slice
                if slice_index is None:
                    slice_index = grain_ids.shape[2] // 2
                grain_ids_2d = grain_ids[:, :, slice_index]
                xlabel, ylabel = "X", "Y"
                if title is None:
                    title = f"Grain IDs - XY slice (z={slice_index})"

            elif slice_direction == "y":
                # XZ slice
                if slice_index is None:
                    slice_index = grain_ids.shape[1] // 2
                grain_ids_2d = grain_ids[:, slice_index, :]
                xlabel, ylabel = "X", "Z"
                if title is None:
                    title = f"Grain IDs - XZ slice (y={slice_index})"

            elif slice_direction == "x":
                if slice_index is None:
                    slice_index = grain_ids.shape[0] // 2
                grain_ids_2d = grain_ids[slice_index, :, :]
                xlabel, ylabel = "Y", "Z"
                if title is None:
                    title = f"Grain IDs - YZ slice (x={slice_index})"

            else:
                raise ValueError(
                    f"slice_direction must be 'x', 'y', or 'z', got '{slice_direction}'"
                )

            grain_ids = grain_ids_2d

        else:
            # 2D case
            xlabel, ylabel = "X", "Y"
            if title is None:
                title = "Grain IDs"

        im = ax.imshow(grain_ids, cmap=cmap, origin="lower", **kwargs)

        if show_boundaries:
            boundaries = create_grain_boundary_overlay(grain_ids)
            ax.contour(boundaries, colors=boundary_color, linewidths=boundary_width)

        if colorbar:
            plt.colorbar(im, ax=ax, label="Grain ID")

        if title:
            ax.set_title(title)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        return im

    @staticmethod
    def plot_3d_slices(
        fig,
        micro,
        slice_indices=None,
        shuffle=False,
        seed=None,
        cmap="nipy_spectral",
        show_boundaries=False,
        suptitle=None,
        **kwargs,
    ):
        """
        Plot three orthogonal slices from a 3D microstructure.

        Args:
        - fig: matplotlib figure
        - micro: Microstructure object (must be 3D)
        - slice_indices: dict or None - {'x': idx, 'y': idx, 'z': idx}
                         If None, uses middle slices
        - shuffle: bool = Shuffle grain IDs for better color contrast
        - seed: int or None - Random seed for shuffling
        - cmap: str - Colormap name
        - show_boundaries: bool - Overlay grain boundaries
        - suptitle: str or None - Figure suptitle
        - **kwargs: Additional arguments passed to imshow

        Returns:
        - List of three image artists [xy_im, xz_im, yz_im]

        Example:
            fig = plt.figure(figsize=(15,5))
            Plotter.plot_3d_slices(fig, micro, shuffle=True)
        """

        if micro.grain_ids.ndim != 3:
            raise ValueError("Microstructure must be 3D")

        if shuffle:
            grain_ids = shuffle_display_grain_ids(
                micro.grain_ids, micro.num_grains, seed=seed
            )

        else:
            grain_ids = micro.grain_ids

        if slice_indices is None:
            slice_indices = {
                "x": micro.dimensions[0] // 2,
                "y": micro.dimensions[1] // 2,
                "z": micro.dimensions[2] // 2,
            }

        axes = fig.subplots(1, 3)

        # XY slice
        z_idx = slice_indices["z"]
        xy_slice = grain_ids[:, :, z_idx]
        im1 = axes[0].imshow(xy_slice, cmap=cmap, origin="lower", **kwargs)
        axes[0].set_title(f"XY Slice (z={z_idx})")
        axes[0].set_xlabel("X")
        axes[0].set_ylabel("Y")
        if show_boundaries:
            boundaries = create_grain_boundary_overlay(xy_slice)
            axes[0].contour(boundaries, colors="black", linewidths=0.5)

        # XZ slice (constant y)
        y_idx = slice_indices["y"]
        xz_slice = grain_ids[:, y_idx, :]
        im2 = axes[1].imshow(xz_slice, cmap=cmap, origin="lower", **kwargs)
        axes[1].set_title(f"XZ Slice (y={y_idx})")
        axes[1].set_xlabel("X")
        axes[1].set_ylabel("Z")
        if show_boundaries:
            boundaries = create_grain_boundary_overlay(xz_slice)
            axes[1].contour(boundaries, colors="black", linewidths=0.5)

        # YZ slice (constant x)
        x_idx = slice_indices["x"]
        yz_slice = grain_ids[x_idx, :, :]
        im3 = axes[2].imshow(yz_slice, cmap=cmap, origin="lower", **kwargs)
        axes[2].set_title(f"YZ Slice (x={x_idx})")
        axes[2].set_xlabel("Y")
        axes[2].set_ylabel("Z")
        if show_boundaries:
            boundaries = create_grain_boundary_overlay(yz_slice)
            axes[2].contour(boundaries, colors="black", linewidths=0.5)

        if suptitle:
            fig.suptitle(suptitle, fontsize=14, y=1.02)

        plt.tight_layout()

        return [im1, im2, im3]

    @staticmethod
    def plot_grain_size_dist(
        ax, micro, bins=50, log_scale=False, title="Grain Size Distribution", **kwargs
    ):
        """
        Plot histogram of grain sizes (in voxels).

        Args:
        - ax: matplotlib axis
        - microstructure: Microstructure object
        - bins: int - Number of histogram bins
        - log_scale: bool - Use log scale for y-axis
        - title: str - Plot title
        - **kwargs: Additional arguments passed to hist

        Returns:
        - Histogram artists

        Example:
            fig, ax = plt.subplots()
            Plotter.plot_grain_size_dist(ax, micro)
        """

        # Count voxels per grain
        unique_ids, counts = np.unique(
            micro.grain_ids[micro.grain_ids > 0], return_counts=True
        )

        n, bins_edges, patches = ax.hist(counts, bins=bins, edgecolor="black", **kwargs)

        if log_scale:
            ax.set_yscale("log")

        ax.set_xlabel("Grain Size (voxels)")
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.grid(alpha=0.3)

        mean_size = np.mean(counts)
        median_size = np.median(counts)
        ax.axvline(
            mean_size, color="red", linestyle="--", label=f"Mean: {mean_size:.1f}"
        )
        ax.axvline(
            median_size,
            color="blue",
            linestyle="--",
            label=f"Median: {median_size:.1f}",
        )
        ax.legend()

        return n, bins_edges, patches
