"""
Visualization Module - Animated Plots for Optimizer Comparison

This module creates animated visualizations showing how different optimizers
navigate loss landscapes. The animations help build intuition about:

1. WHY SGD bounces in valleys (large gradient in one direction)
2. WHY Adam normalizes and goes straight (divides by variance)
3. WHY Newton solves quadratics instantly (uses curvature)
4. WHY Momentum overshoots (accumulated velocity)

ANIMATION TYPES:
================
1. Contour Plot Animation - Shows optimizers racing on a 2D contour map
2. 3D Surface Animation - Shows paths on a 3D surface (rotatable)
3. State Evolution - Shows Adam's m and v changing over time
4. Step Size Comparison - Shows effective step size for each optimizer

DARK THEME:
===========
All visualizations use a dark theme for:
- Better visibility of colored optimizer paths
- Easier on the eyes
- Professional appearance

SAVING OPTIONS:
===============
- GIF: Universal, works everywhere
- MP4: Higher quality, requires ffmpeg
- PNG: Static snapshot of final state
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Optional, Tuple, Dict, Any
import io
import warnings

# Suppress matplotlib warnings about animation
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# Import from our modules (will work once package is installed)
try:
    from .landscapes import Landscape
    from .optimizers import Optimizer, OptimizerState
except ImportError:
    # For standalone testing
    pass


# Dark theme configuration
DARK_THEME = {
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor": "#16213e",
    "axes.edgecolor": "#e94560",
    "axes.labelcolor": "#eee",
    "text.color": "#eee",
    "xtick.color": "#aaa",
    "ytick.color": "#aaa",
    "grid.color": "#333",
    "legend.facecolor": "#1a1a2e",
    "legend.edgecolor": "#444",
}


def apply_dark_theme():
    """Apply dark theme to matplotlib."""
    for key, value in DARK_THEME.items():
        plt.rcParams[key] = value


def create_contour_figure(
    landscape: "Landscape",
    resolution: int = 100,
    levels: int = 50,
    figsize: Tuple[int, int] = (12, 8),
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a contour plot of the landscape.

    This is the "map" on which optimizers will navigate.

    Args:
        landscape: The loss landscape to visualize
        resolution: Grid resolution
        levels: Number of contour levels
        figsize: Figure size

    Returns:
        (fig, ax) tuple
    """
    apply_dark_theme()

    fig, ax = plt.subplots(figsize=figsize)

    # Generate meshgrid
    X, Y, Z = landscape.get_meshgrid(resolution)

    # Create filled contour plot
    # Use log scale for better visibility of valleys
    Z_plot = np.log10(Z - Z.min() + 1)

    contour = ax.contourf(X, Y, Z_plot, levels=levels, cmap="viridis", alpha=0.8)
    ax.contour(
        X, Y, Z_plot, levels=levels // 2, colors="white", alpha=0.3, linewidths=0.5
    )

    # Add colorbar
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label("log₁₀(Loss + 1)", color="#eee")

    # Mark the optimum
    opt_x, opt_y = landscape.optimal_point
    if np.isfinite(opt_x) and np.isfinite(opt_y):
        ax.scatter(
            [opt_x],
            [opt_y],
            c="yellow",
            s=200,
            marker="*",
            zorder=100,
            label="Optimum",
            edgecolors="black",
        )

    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_title(landscape.name, fontsize=14, fontweight="bold")

    return fig, ax


def animate_optimization(
    landscape: "Landscape",
    optimizers: List["Optimizer"],
    start_point: Tuple[float, float] = (-8.0, 3.0),
    steps: int = 100,
    interval: int = 50,  # ms between frames
    figsize: Tuple[int, int] = (14, 8),
    show_gradient: bool = False,
    save_path: Optional[str] = None,
    fps: int = 20,
) -> animation.FuncAnimation:
    """
    Create an animated visualization of optimizers racing on a landscape.

    This is the MAIN visualization function!

    Args:
        landscape: The loss surface to optimize on
        optimizers: List of optimizers to compare
        start_point: Starting (x, y) position
        steps: Number of optimization steps
        interval: Milliseconds between animation frames
        figsize: Figure size
        show_gradient: Show gradient arrows at each point
        save_path: If provided, save animation to this path (.gif or .mp4)
        fps: Frames per second for saved animation

    Returns:
        The animation object

    Example:
        anim = animate_optimization(
            Valley(condition=50),
            [SGD(lr=0.03), Adam(lr=0.5)],
            start_point=(-8, 3),
            steps=100,
            save_path="optimizer_race.gif"
        )
    """
    apply_dark_theme()

    # Run all optimizers
    histories = []
    for opt in optimizers:
        history = opt.optimize(landscape, start_point[0], start_point[1], steps)
        histories.append(history)

    # Create figure with side panels
    fig = plt.figure(figsize=figsize)

    # Main contour plot
    ax_main = fig.add_axes([0.05, 0.1, 0.6, 0.8])

    # Loss curve panel
    ax_loss = fig.add_axes([0.7, 0.55, 0.25, 0.35])
    ax_loss.set_facecolor("#16213e")

    # Info panel
    ax_info = fig.add_axes([0.7, 0.1, 0.25, 0.35])
    ax_info.set_facecolor("#16213e")
    ax_info.axis("off")

    # Draw landscape contours
    X, Y, Z = landscape.get_meshgrid(100)
    Z_plot = np.log10(Z - Z.min() + 1)
    ax_main.contourf(X, Y, Z_plot, levels=50, cmap="viridis", alpha=0.8)
    ax_main.contour(X, Y, Z_plot, levels=25, colors="white", alpha=0.3, linewidths=0.5)

    # Mark optimum
    opt_x, opt_y = landscape.optimal_point
    if np.isfinite(opt_x) and np.isfinite(opt_y):
        ax_main.scatter(
            [opt_x],
            [opt_y],
            c="yellow",
            s=200,
            marker="*",
            zorder=100,
            edgecolors="black",
            linewidths=1,
        )

    # Mark start
    ax_main.scatter(
        [start_point[0]],
        [start_point[1]],
        c="white",
        s=150,
        marker="o",
        zorder=100,
        edgecolors="black",
        linewidths=2,
    )

    ax_main.set_xlabel("x", fontsize=12)
    ax_main.set_ylabel("y", fontsize=12)
    ax_main.set_title(
        f"{landscape.name} - Optimizer Race", fontsize=14, fontweight="bold"
    )

    # Initialize optimizer paths (lines and dots)
    lines = []
    dots = []
    for i, opt in enumerate(optimizers):
        (line,) = ax_main.plot(
            [], [], "-", color=opt.color, linewidth=2, alpha=0.8, label=opt.name
        )
        (dot,) = ax_main.plot(
            [],
            [],
            "o",
            color=opt.color,
            markersize=10,
            markeredgecolor="white",
            markeredgewidth=2,
        )
        lines.append(line)
        dots.append(dot)

    ax_main.legend(loc="upper right", fontsize=9)

    # Initialize loss curves
    loss_lines = []
    for opt in optimizers:
        (loss_line,) = ax_loss.plot(
            [], [], "-", color=opt.color, linewidth=2, alpha=0.8
        )
        loss_lines.append(loss_line)

    ax_loss.set_xlabel("Step", fontsize=10)
    ax_loss.set_ylabel("Loss (log scale)", fontsize=10)
    ax_loss.set_title("Loss vs Step", fontsize=11)
    ax_loss.set_yscale("log")
    ax_loss.grid(True, alpha=0.3)

    # Set loss axis limits
    all_losses = [h.loss for history in histories for h in history]
    ax_loss.set_xlim(0, steps)
    ax_loss.set_ylim(min(all_losses) * 0.5, max(all_losses) * 2)

    # Text for info panel
    info_text = ax_info.text(
        0.5,
        0.5,
        "",
        transform=ax_info.transAxes,
        fontsize=10,
        verticalalignment="center",
        horizontalalignment="center",
        family="monospace",
    )

    step_text = ax_main.text(
        0.02,
        0.98,
        "",
        transform=ax_main.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="#1a1a2e", alpha=0.8),
    )

    def init():
        """Initialize animation."""
        for line, dot in zip(lines, dots):
            line.set_data([], [])
            dot.set_data([], [])
        for loss_line in loss_lines:
            loss_line.set_data([], [])
        info_text.set_text("")
        step_text.set_text("")
        return lines + dots + loss_lines + [info_text, step_text]

    def animate(frame):
        """Update animation for frame."""
        # Update each optimizer's path
        info_lines = []

        for i, (history, line, dot, loss_line, opt) in enumerate(
            zip(histories, lines, dots, loss_lines, optimizers)
        ):
            # Get path up to current frame
            path_x = [h.x for h in history[: frame + 1]]
            path_y = [h.y for h in history[: frame + 1]]
            losses = [h.loss for h in history[: frame + 1]]

            # Update path line
            line.set_data(path_x, path_y)

            # Update current position dot
            if frame < len(history):
                dot.set_data([history[frame].x], [history[frame].y])

            # Update loss curve
            loss_line.set_data(range(len(losses)), losses)

            # Build info text
            if frame < len(history):
                state = history[frame]
                info_lines.append(f"{opt.name}:")
                info_lines.append(f"  pos: ({state.x:.3f}, {state.y:.3f})")
                info_lines.append(f"  loss: {state.loss:.4f}")
                info_lines.append(
                    f"  |grad|: {np.sqrt(state.grad_x**2 + state.grad_y**2):.4f}"
                )

                # Show Adam's internal state
                if "v_x_hat" in state.extra:
                    info_lines.append(
                        f"  step_scale: {state.extra['step_scale_x']:.4f}"
                    )
                info_lines.append("")

        info_text.set_text("\n".join(info_lines))
        step_text.set_text(f"Step: {frame}/{steps}")

        return lines + dots + loss_lines + [info_text, step_text]

    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=steps + 1, interval=interval, blit=True
    )

    # Save if path provided
    if save_path:
        print(f"Saving animation to {save_path}...")
        if save_path.endswith(".gif"):
            anim.save(save_path, writer="pillow", fps=fps)
        elif save_path.endswith(".mp4"):
            anim.save(save_path, writer="ffmpeg", fps=fps)
        else:
            anim.save(save_path, fps=fps)
        print(f"Saved!")

    return anim


def plot_optimization_paths(
    landscape: "Landscape",
    optimizers: List["Optimizer"],
    start_point: Tuple[float, float] = (-8.0, 3.0),
    steps: int = 100,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create a static plot showing all optimizer paths.

    Useful for quick comparisons without animation.
    """
    apply_dark_theme()

    # Run all optimizers
    for opt in optimizers:
        opt.optimize(landscape, start_point[0], start_point[1], steps)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Draw landscape
    X, Y, Z = landscape.get_meshgrid(100)
    Z_plot = np.log10(Z - Z.min() + 1)
    ax.contourf(X, Y, Z_plot, levels=50, cmap="viridis", alpha=0.8)
    ax.contour(X, Y, Z_plot, levels=25, colors="white", alpha=0.3, linewidths=0.5)

    # Mark optimum
    opt_x, opt_y = landscape.optimal_point
    if np.isfinite(opt_x) and np.isfinite(opt_y):
        ax.scatter(
            [opt_x],
            [opt_y],
            c="yellow",
            s=200,
            marker="*",
            zorder=100,
            edgecolors="black",
        )

    # Mark start
    ax.scatter(
        [start_point[0]],
        [start_point[1]],
        c="white",
        s=150,
        marker="o",
        zorder=100,
        edgecolors="black",
        linewidths=2,
    )

    # Plot each optimizer's path
    for opt in optimizers:
        path = opt.get_path()
        ax.plot(
            path[:, 0],
            path[:, 1],
            "-o",
            color=opt.color,
            linewidth=2,
            markersize=3,
            alpha=0.8,
            label=opt.name,
        )

        # Mark final position
        ax.scatter(
            [path[-1, 0]],
            [path[-1, 1]],
            c=opt.color,
            s=100,
            marker="s",
            edgecolors="white",
            linewidths=2,
            zorder=99,
        )

    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_title(
        f"{landscape.name} - Optimization Paths", fontsize=14, fontweight="bold"
    )
    ax.legend(loc="upper right")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
        print(f"Saved to {save_path}")

    return fig


def plot_adam_state_evolution(
    landscape: "Landscape",
    start_point: Tuple[float, float] = (-8.0, 3.0),
    steps: int = 100,
    lr: float = 0.5,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize how Adam's internal state (m, v) evolves over time.

    This helps understand:
    - How the first moment (m) smooths the gradient
    - How the second moment (v) estimates variance
    - How bias correction affects early steps
    - How the effective step size changes

    This is directly related to your question:
    "What is variance in Adam?"
    """
    apply_dark_theme()

    # Import here to avoid circular imports
    from .optimizers import Adam

    adam = Adam(lr=lr)
    history = adam.optimize(landscape, start_point[0], start_point[1], steps)

    # Extract state history
    t_vals = [h.step for h in history]

    # Raw moments
    m_x = [h.extra.get("m_x", 0) for h in history]
    m_y = [h.extra.get("m_y", 0) for h in history]
    v_x = [h.extra.get("v_x", 0) for h in history]
    v_y = [h.extra.get("v_y", 0) for h in history]

    # Bias-corrected moments
    m_x_hat = [h.extra.get("m_x_hat", 0) for h in history]
    m_y_hat = [h.extra.get("m_y_hat", 0) for h in history]
    v_x_hat = [h.extra.get("v_x_hat", 0) for h in history]
    v_y_hat = [h.extra.get("v_y_hat", 0) for h in history]

    # Step scales
    scale_x = [h.extra.get("step_scale_x", 1) for h in history]
    scale_y = [h.extra.get("step_scale_y", 1) for h in history]

    # Raw gradients
    grad_x = [h.grad_x for h in history]
    grad_y = [h.grad_y for h in history]

    # Losses
    losses = [h.loss for h in history]

    # Create figure
    fig, axes = plt.subplots(3, 2, figsize=figsize)

    # Row 1: First moment (m) - Smoothed gradient
    axes[0, 0].plot(t_vals, grad_x, "r-", alpha=0.5, label="Raw grad_x")
    axes[0, 0].plot(t_vals, m_x, "r-", linewidth=2, label="m_x (smoothed)")
    axes[0, 0].plot(t_vals, m_x_hat, "r--", linewidth=2, label="m̂_x (corrected)")
    axes[0, 0].set_title("First Moment (x) - Gradient Smoothing", fontsize=11)
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(t_vals, grad_y, "b-", alpha=0.5, label="Raw grad_y")
    axes[0, 1].plot(t_vals, m_y, "b-", linewidth=2, label="m_y (smoothed)")
    axes[0, 1].plot(t_vals, m_y_hat, "b--", linewidth=2, label="m̂_y (corrected)")
    axes[0, 1].set_title("First Moment (y) - Gradient Smoothing", fontsize=11)
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    # Row 2: Second moment (v) - Variance estimate
    axes[1, 0].plot(t_vals, [g**2 for g in grad_x], "r-", alpha=0.5, label="grad_x²")
    axes[1, 0].plot(t_vals, v_x, "r-", linewidth=2, label="v_x (smoothed)")
    axes[1, 0].plot(t_vals, v_x_hat, "r--", linewidth=2, label="v̂_x (corrected)")
    axes[1, 0].set_title("Second Moment (x) - Variance Estimate", fontsize=11)
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale("log")

    axes[1, 1].plot(t_vals, [g**2 for g in grad_y], "b-", alpha=0.5, label="grad_y²")
    axes[1, 1].plot(t_vals, v_y, "b-", linewidth=2, label="v_y (smoothed)")
    axes[1, 1].plot(t_vals, v_y_hat, "b--", linewidth=2, label="v̂_y (corrected)")
    axes[1, 1].set_title("Second Moment (y) - Variance Estimate", fontsize=11)
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale("log")

    # Row 3: Effective step size and loss
    axes[2, 0].plot(t_vals, scale_x, "r-", linewidth=2, label="sqrt(v̂_x) + ε")
    axes[2, 0].plot(t_vals, scale_y, "b-", linewidth=2, label="sqrt(v̂_y) + ε")
    axes[2, 0].set_title("Step Scale (what we divide by)", fontsize=11)
    axes[2, 0].set_xlabel("Step")
    axes[2, 0].legend(fontsize=8)
    axes[2, 0].grid(True, alpha=0.3)

    axes[2, 1].plot(t_vals, losses, "g-", linewidth=2)
    axes[2, 1].set_title("Loss over Time", fontsize=11)
    axes[2, 1].set_xlabel("Step")
    axes[2, 1].set_ylabel("Loss")
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].set_yscale("log")

    fig.suptitle(
        f"Adam Internal State Evolution on {landscape.name}",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(
            save_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight"
        )
        print(f"Saved to {save_path}")

    return fig


def plot_step_size_comparison(
    landscape: "Landscape",
    optimizers: List["Optimizer"],
    start_point: Tuple[float, float] = (-8.0, 3.0),
    steps: int = 100,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Compare effective step sizes across optimizers.

    This shows:
    - SGD: Step size = |lr * gradient| (varies wildly)
    - Adam: Step size is normalized (relatively constant)
    - Newton: Can take huge steps (goes to minimum fast)
    """
    apply_dark_theme()

    # Run all optimizers
    for opt in optimizers:
        opt.optimize(landscape, start_point[0], start_point[1], steps)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot step sizes
    for opt in optimizers:
        history = opt.history

        # Compute step sizes
        step_sizes = []
        for i in range(1, len(history)):
            dx = history[i].x - history[i - 1].x
            dy = history[i].y - history[i - 1].y
            step_size = np.sqrt(dx**2 + dy**2)
            step_sizes.append(step_size)

        ax1.plot(step_sizes, color=opt.color, linewidth=2, label=opt.name, alpha=0.8)

    ax1.set_xlabel("Step")
    ax1.set_ylabel("Step Size (|Δw|)")
    ax1.set_title("Step Size per Iteration")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")

    # Plot cumulative distance
    for opt in optimizers:
        history = opt.history

        cumulative = [0]
        for i in range(1, len(history)):
            dx = history[i].x - history[i - 1].x
            dy = history[i].y - history[i - 1].y
            cumulative.append(cumulative[-1] + np.sqrt(dx**2 + dy**2))

        ax2.plot(cumulative, color=opt.color, linewidth=2, label=opt.name, alpha=0.8)

    ax2.set_xlabel("Step")
    ax2.set_ylabel("Cumulative Distance Traveled")
    ax2.set_title("Total Path Length")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        f"Step Size Analysis on {landscape.name}", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
        print(f"Saved to {save_path}")

    return fig


def plot_3d_surface(
    landscape: "Landscape",
    optimizers: Optional[List["Optimizer"]] = None,
    start_point: Tuple[float, float] = (-8.0, 3.0),
    steps: int = 100,
    resolution: int = 50,
    figsize: Tuple[int, int] = (12, 8),
    elev: float = 30,
    azim: float = -60,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create a 3D surface plot with optimizer paths.
    """
    apply_dark_theme()

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # Generate surface
    X, Y, Z = landscape.get_meshgrid(resolution)

    # Clip extreme values for better visualization
    Z_clipped = np.clip(Z, Z.min(), np.percentile(Z, 95))

    # Plot surface
    ax.plot_surface(
        X, Y, Z_clipped, cmap="viridis", alpha=0.6, edgecolor="none", antialiased=True
    )

    # Plot optimizer paths if provided
    if optimizers:
        for opt in optimizers:
            opt.optimize(landscape, start_point[0], start_point[1], steps)
            path = opt.get_path()
            losses = opt.get_losses()

            ax.plot(
                path[:, 0],
                path[:, 1],
                losses,
                color=opt.color,
                linewidth=3,
                label=opt.name,
            )
            ax.scatter(
                path[-1, 0], path[-1, 1], losses[-1], color=opt.color, s=100, marker="o"
            )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("Loss")
    ax.set_title(f"{landscape.name} - 3D View", fontsize=14, fontweight="bold")
    ax.view_init(elev=elev, azim=azim)

    if optimizers:
        ax.legend()

    if save_path:
        plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
        print(f"Saved to {save_path}")

    return fig


def create_learning_report(
    landscape: "Landscape",
    optimizers: List["Optimizer"],
    start_point: Tuple[float, float] = (-8.0, 3.0),
    steps: int = 100,
    save_dir: str = ".",
) -> None:
    """
    Generate a complete visual report comparing optimizers.

    Creates:
    - optimizer_paths.png: Static comparison of all paths
    - adam_state.png: Adam's internal state evolution
    - step_sizes.png: Step size analysis
    - surface_3d.png: 3D surface view
    - optimizer_race.gif: Animated race
    """
    import os

    print("=" * 60)
    print(f"Generating Learning Report for {landscape.name}")
    print("=" * 60)

    # 1. Static paths
    print("\n1. Generating static paths plot...")
    plot_optimization_paths(
        landscape,
        optimizers,
        start_point,
        steps,
        save_path=os.path.join(save_dir, "optimizer_paths.png"),
    )
    plt.close()

    # 2. Adam state evolution
    print("\n2. Generating Adam state evolution...")
    plot_adam_state_evolution(
        landscape,
        start_point,
        steps,
        save_path=os.path.join(save_dir, "adam_state.png"),
    )
    plt.close()

    # 3. Step size comparison
    print("\n3. Generating step size comparison...")
    plot_step_size_comparison(
        landscape,
        optimizers,
        start_point,
        steps,
        save_path=os.path.join(save_dir, "step_sizes.png"),
    )
    plt.close()

    # 4. 3D surface
    print("\n4. Generating 3D surface...")
    plot_3d_surface(
        landscape,
        optimizers,
        start_point,
        steps,
        save_path=os.path.join(save_dir, "surface_3d.png"),
    )
    plt.close()

    # 5. Animation
    print("\n5. Generating animation (this may take a moment)...")
    animate_optimization(
        landscape,
        optimizers,
        start_point,
        steps,
        save_path=os.path.join(save_dir, "optimizer_race.gif"),
        fps=15,
    )
    plt.close()

    print("\n" + "=" * 60)
    print("Report complete! Files saved:")
    print(f"  - {save_dir}/optimizer_paths.png")
    print(f"  - {save_dir}/adam_state.png")
    print(f"  - {save_dir}/step_sizes.png")
    print(f"  - {save_dir}/surface_3d.png")
    print(f"  - {save_dir}/optimizer_race.gif")
    print("=" * 60)


if __name__ == "__main__":
    # Quick demo
    from .landscapes import Valley
    from .optimizers import SGD, Adam, Momentum, Newton

    valley = Valley(condition=50)
    optimizers = [
        SGD(lr=0.03),
        Momentum(lr=0.01),
        Adam(lr=0.5),
        Newton(lr=1.0, damping=0.1),
    ]

    # Generate static plot
    fig = plot_optimization_paths(valley, optimizers, start_point=(-8, 3), steps=100)
    plt.show()
