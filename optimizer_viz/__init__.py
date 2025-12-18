"""
Optimizer Visualization - A Learning Tool for Understanding Gradient Descent

This package provides interactive visualizations to understand how different
optimization algorithms (SGD, Momentum, Adam, Muon/Newton) navigate loss landscapes.

Key Modules:
- landscapes: Different loss surfaces (Valley, Saddle, Rosenbrock, etc.)
- optimizers: Custom implementations FROM SCRATCH (not PyTorch black boxes)
- visualizer: Animated plots and GIF/video generation
- distributed_sim: Simulate gradient averaging (prep for NCCL/DDP concepts)
- interactive: CLI interface for experimenting

Example Usage:
    from optimizer_viz import landscapes, optimizers, visualizer

    # Create a landscape
    valley = landscapes.Valley()

    # Create optimizers
    sgd = optimizers.SGD(lr=0.01)
    adam = optimizers.Adam(lr=0.1)

    # Visualize them racing
    visualizer.animate_comparison(
        landscape=valley,
        optimizers=[sgd, adam],
        start_point=(-8.0, 3.0),
        steps=100
    )

Author: Learning through visualization
"""

__version__ = "0.1.0"

from . import landscapes
from . import optimizers

# visualizer is optional (requires matplotlib)
try:
    from . import visualizer
    __all__ = ["landscapes", "optimizers", "visualizer"]
except ImportError:
    __all__ = ["landscapes", "optimizers"]
