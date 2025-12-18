"""
Loss Landscapes for Optimization Visualization

This module defines different loss surfaces (functions) that optimizers navigate.
Each landscape is designed to teach a specific concept about optimization.

MATH BACKGROUND (from your notes):
-----------------------------------
The optimization problem is: minimize f(x, y)

The gradient at any point tells us the direction of steepest ASCENT:
    grad f = [df/dx, df/dy]

So we move in the NEGATIVE gradient direction to descend.

The Hessian matrix tells us about CURVATURE:
    H = [[d²f/dx², d²f/dxdy],
         [d²f/dydx, d²f/dy²]]

- If H is "ill-conditioned" (eigenvalues very different), the landscape is like
  a narrow valley. SGD bounces off the walls.
- Newton's method uses H^(-1) to "flatten" the curvature.

LANDSCAPES AND WHAT THEY TEACH:
-------------------------------
1. Valley (Ill-Conditioned Quadratic):
   - f(x,y) = x² + 50y²
   - VERY steep in y-direction, flat in x-direction
   - SGD bounces, Adam normalizes, Newton solves instantly

2. Saddle Point:
   - f(x,y) = x² - y²
   - Gradient is tiny at the center (plateau!)
   - SGD gets stuck, Adam's variance helps escape

3. Rosenbrock (Banana Valley):
   - f(x,y) = (1-x)² + 100(y-x²)²
   - The classic "hard" optimization problem
   - Narrow, curved valley - tests all optimizers

4. Beale Function:
   - More complex multi-valley surface
   - Tests momentum overshooting

5. Rastrigin Function:
   - Many local minima (like a waffle)
   - For future: shows why global optimization is hard
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np


@dataclass
class Point:
    """A point in 2D space with its loss value."""

    x: float
    y: float
    loss: Optional[float] = None

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y])


class Landscape(ABC):
    """
    Base class for all loss landscapes.

    Each landscape must implement:
    - evaluate(x, y): Get the loss value at a point
    - gradient(x, y): Get the gradient (analytically, not numerically!)
    - hessian(x, y): Get the Hessian matrix (for Newton method)

    The gradients are computed ANALYTICALLY (like you did in backprop)
    instead of using autograd - this is more educational!
    """

    name: str = "Base Landscape"
    description: str = ""
    optimal_point: Tuple[float, float] = (0.0, 0.0)
    optimal_value: float = 0.0

    # Default viewing bounds
    x_range: Tuple[float, float] = (-10.0, 10.0)
    y_range: Tuple[float, float] = (-10.0, 10.0)

    @abstractmethod
    def evaluate(self, x: float, y: float) -> float:
        """
        Compute the loss at point (x, y).

        This is like the "Forward Pass" in your backprop notes.
        """
        pass

    @abstractmethod
    def gradient(self, x: float, y: float) -> Tuple[float, float]:
        """
        Compute the gradient (df/dx, df/dy) at point (x, y).

        This is the "Backward Pass" - computing derivatives!

        MATH: For f(x,y), the gradient is:
            grad f = [∂f/∂x, ∂f/∂y]

        Returns:
            Tuple of (df/dx, df/dy)
        """
        pass

    @abstractmethod
    def hessian(self, x: float, y: float) -> np.ndarray:
        """
        Compute the Hessian matrix at point (x, y).

        This is the CURVATURE - used by Newton's method!

        MATH: The Hessian is:
            H = [[∂²f/∂x², ∂²f/∂x∂y],
                 [∂²f/∂y∂x, ∂²f/∂y²]]

        Returns:
            2x2 numpy array representing the Hessian
        """
        pass

    def evaluate_batch(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Evaluate the function on a meshgrid for visualization."""
        return np.vectorize(self.evaluate)(x, y)

    def gradient_batch(
        self, x: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute gradients on a meshgrid for quiver plots."""
        grad_x = np.zeros_like(x)
        grad_y = np.zeros_like(y)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                gx, gy = self.gradient(x[i, j], y[i, j])
                grad_x[i, j] = gx
                grad_y[i, j] = gy
        return grad_x, grad_y

    def get_meshgrid(
        self, resolution: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a meshgrid for visualization.

        Returns:
            Tuple of (X, Y, Z) arrays for contour/surface plots
        """
        x = np.linspace(self.x_range[0], self.x_range[1], resolution)
        y = np.linspace(self.y_range[0], self.y_range[1], resolution)
        X, Y = np.meshgrid(x, y)
        Z = self.evaluate_batch(X, Y)
        return X, Y, Z

    def condition_number(self, x: float, y: float) -> float:
        """
        Compute the condition number at a point.

        The condition number tells us how "ill-conditioned" the problem is:
        - Condition number = 1: Perfect (like a circular valley)
        - Condition number >> 1: Bad (like a narrow valley)

        MATH: condition_number = |λ_max| / |λ_min|
        where λ are eigenvalues of the Hessian.
        """
        H = self.hessian(x, y)
        eigenvalues = np.linalg.eigvalsh(H)
        # Avoid division by zero
        min_ev = np.abs(eigenvalues).min()
        if min_ev < 1e-10:
            return float("inf")
        return np.abs(eigenvalues).max() / min_ev


class Valley(Landscape):
    """
    The "Taco Shell" Valley - An Ill-Conditioned Quadratic.

    f(x, y) = x² + condition * y²

    WHAT THIS TEACHES:
    ------------------
    When `condition` is large (e.g., 50), the function is:
    - VERY steep in the y-direction (high gradient)
    - Flat in the x-direction (low gradient)

    This creates a narrow "taco shell" shape.

    OPTIMIZER BEHAVIOR:
    - SGD: Bounces off the steep walls (y-direction), wastes energy
    - Momentum: Builds up speed, overshoots, oscillates
    - Adam: Normalizes! The y-gradient is large but so is its variance,
            so Adam divides them and takes normal steps
    - Newton: Uses the Hessian to "squash" the valley into a circle,
              then walks straight to the minimum

    GRADIENT:
    ∂f/∂x = 2x
    ∂f/∂y = 2 * condition * y

    HESSIAN:
    H = [[2, 0],
         [0, 2 * condition]]

    Notice the Hessian eigenvalues are 2 and 2*condition.
    The condition number is exactly `condition`!
    """

    def __init__(self, condition: float = 50.0):
        """
        Args:
            condition: The conditioning factor. Higher = more ill-conditioned.
                       Default 50 means y-direction is 50x steeper than x.
        """
        self.condition = condition
        self.name = f"Valley (condition={condition})"
        self.description = (
            f"f(x,y) = x² + {condition}y² - Tests SGD bouncing vs Adam normalizing"
        )
        self.optimal_point = (0.0, 0.0)
        self.optimal_value = 0.0
        self.x_range = (-10.0, 10.0)
        self.y_range = (-5.0, 5.0)  # Narrower in y since it's the steep direction

    def evaluate(self, x: float, y: float) -> float:
        """f(x, y) = x² + condition * y²"""
        return x**2 + self.condition * y**2

    def gradient(self, x: float, y: float) -> Tuple[float, float]:
        """
        Gradient of f(x, y) = x² + condition * y²

        ∂f/∂x = 2x
        ∂f/∂y = 2 * condition * y

        Notice: If y = 1 and condition = 50, the y-gradient is 100!
        But if x = 1, the x-gradient is only 2.
        This 50x difference causes SGD to bounce.
        """
        df_dx = 2 * x
        df_dy = 2 * self.condition * y
        return (df_dx, df_dy)

    def hessian(self, x: float, y: float) -> np.ndarray:
        """
        Hessian of f(x, y) = x² + condition * y²

        H = [[∂²f/∂x², ∂²f/∂x∂y],
             [∂²f/∂y∂x, ∂²f/∂y²]]
           = [[2, 0],
              [0, 2 * condition]]

        This is a DIAGONAL matrix! The eigenvalues are on the diagonal.
        - λ₁ = 2 (curvature in x-direction)
        - λ₂ = 2 * condition (curvature in y-direction)

        Newton's method inverts this:
        H⁻¹ = [[1/2, 0],
               [0, 1/(2*condition)]]

        This "shrinks" the y-step and "expands" the x-step, equalizing them!
        """
        return np.array([[2.0, 0.0], [0.0, 2.0 * self.condition]])


class SaddlePoint(Landscape):
    """
    A Saddle Point Surface.

    f(x, y) = x² - y²

    WHAT THIS TEACHES:
    ------------------
    At the origin (0, 0), the gradient is ZERO, but it's NOT a minimum!
    - It curves UP in the x-direction (local minimum in x)
    - It curves DOWN in the y-direction (local maximum in y)

    This is the "Plateau Problem" from your Gemini conversation:
    "In high dimensions, there are no local minima, only saddle points."

    OPTIMIZER BEHAVIOR:
    - SGD: If it lands near the center, gradient is tiny, it barely moves
    - Adam: Even though gradient is tiny, Adam looks at the VARIANCE.
            If variance is also tiny but consistent, Adam normalizes and moves!
    - Newton: The Hessian has negative eigenvalue, Newton would go UPHILL in y!
              (This is why Newton needs modifications for non-convex functions)

    GRADIENT:
    ∂f/∂x = 2x
    ∂f/∂y = -2y

    At origin: grad = (0, 0) - THE PLATEAU!

    HESSIAN:
    H = [[2, 0],
         [0, -2]]

    Eigenvalues: +2 and -2 (opposite signs = saddle point)
    """

    def __init__(self):
        self.name = "Saddle Point"
        self.description = "f(x,y) = x² - y² - Tests escaping plateaus"
        self.optimal_point = (0.0, float("inf"))  # No minimum, goes to -infinity in y
        self.optimal_value = float("-inf")
        self.x_range = (-5.0, 5.0)
        self.y_range = (-5.0, 5.0)

    def evaluate(self, x: float, y: float) -> float:
        """f(x, y) = x² - y²"""
        return x**2 - y**2

    def gradient(self, x: float, y: float) -> Tuple[float, float]:
        """
        ∂f/∂x = 2x
        ∂f/∂y = -2y

        KEY INSIGHT: At origin, gradient is (0, 0).
        This is the "saddle point plateau" where SGD gets stuck!
        """
        return (2 * x, -2 * y)

    def hessian(self, x: float, y: float) -> np.ndarray:
        """
        H = [[2, 0],
             [0, -2]]

        The NEGATIVE eigenvalue (-2) indicates this is a saddle point.
        Newton's method would fail here because H is not positive definite!
        """
        return np.array([[2.0, 0.0], [0.0, -2.0]])


class Rosenbrock(Landscape):
    """
    The Rosenbrock "Banana Valley" Function.

    f(x, y) = (a - x)² + b * (y - x²)²

    Classic parameters: a = 1, b = 100

    WHAT THIS TEACHES:
    ------------------
    This is THE classic optimization test function. It has:
    - A global minimum at (a, a²) = (1, 1)
    - A narrow, curved valley that's hard to navigate
    - The valley floor is relatively flat (slow progress)
    - The valley walls are steep (bouncing for SGD)

    Named after Howard Rosenbrock (1960).

    OPTIMIZER BEHAVIOR:
    - All optimizers struggle here!
    - The curve of the valley defeats simple momentum
    - Adam helps but still slow
    - Newton would help if we could compute Hessian accurately

    GRADIENT:
    ∂f/∂x = -2(a - x) - 4bx(y - x²)
          = 2(x - a) + 4bx(x² - y)
    ∂f/∂y = 2b(y - x²)

    HESSIAN:
    ∂²f/∂x² = 2 + 8bx² + 4b(x² - y) = 2 + 12bx² - 4by
    ∂²f/∂x∂y = -4bx
    ∂²f/∂y² = 2b
    """

    def __init__(self, a: float = 1.0, b: float = 100.0):
        self.a = a
        self.b = b
        self.name = f"Rosenbrock (a={a}, b={b})"
        self.description = (
            f"f(x,y) = ({a}-x)² + {b}(y-x²)² - The classic 'banana valley'"
        )
        self.optimal_point = (a, a**2)
        self.optimal_value = 0.0
        self.x_range = (-2.0, 2.0)
        self.y_range = (-1.0, 3.0)

    def evaluate(self, x: float, y: float) -> float:
        """f(x, y) = (a - x)² + b * (y - x²)²"""
        return (self.a - x) ** 2 + self.b * (y - x**2) ** 2

    def gradient(self, x: float, y: float) -> Tuple[float, float]:
        """
        ∂f/∂x = -2(a - x) - 4bx(y - x²)
        ∂f/∂y = 2b(y - x²)
        """
        df_dx = -2 * (self.a - x) + 4 * self.b * x * (x**2 - y)
        df_dy = 2 * self.b * (y - x**2)
        return (df_dx, df_dy)

    def hessian(self, x: float, y: float) -> np.ndarray:
        """
        H = [[2 + 12bx² - 4by,  -4bx],
             [-4bx,              2b ]]
        """
        h_xx = 2 + 12 * self.b * x**2 - 4 * self.b * y
        h_xy = -4 * self.b * x
        h_yy = 2 * self.b
        return np.array([[h_xx, h_xy], [h_xy, h_yy]])


class Beale(Landscape):
    """
    The Beale Function - A More Complex Multi-Valley Surface.

    f(x, y) = (1.5 - x + xy)² + (2.25 - x + xy²)² + (2.625 - x + xy³)²

    WHAT THIS TEACHES:
    ------------------
    - Multiple interacting terms create complex terrain
    - Tests momentum overshooting on sharp turns
    - Global minimum at (3, 0.5)

    This is harder than Rosenbrock because the valley has sharper curves.
    """

    def __init__(self):
        self.name = "Beale Function"
        self.description = "Complex multi-term surface - Tests momentum overshooting"
        self.optimal_point = (3.0, 0.5)
        self.optimal_value = 0.0
        self.x_range = (-4.5, 4.5)
        self.y_range = (-4.5, 4.5)

    def evaluate(self, x: float, y: float) -> float:
        """f(x,y) = (1.5 - x + xy)² + (2.25 - x + xy²)² + (2.625 - x + xy³)²"""
        term1 = (1.5 - x + x * y) ** 2
        term2 = (2.25 - x + x * y**2) ** 2
        term3 = (2.625 - x + x * y**3) ** 2
        return term1 + term2 + term3

    def gradient(self, x: float, y: float) -> Tuple[float, float]:
        """
        Using chain rule on each term:

        Let g₁ = 1.5 - x + xy, g₂ = 2.25 - x + xy², g₃ = 2.625 - x + xy³

        ∂f/∂x = 2g₁(∂g₁/∂x) + 2g₂(∂g₂/∂x) + 2g₃(∂g₃/∂x)
        ∂f/∂y = 2g₁(∂g₁/∂y) + 2g₂(∂g₂/∂y) + 2g₃(∂g₃/∂y)

        where:
        ∂g₁/∂x = -1 + y,  ∂g₁/∂y = x
        ∂g₂/∂x = -1 + y², ∂g₂/∂y = 2xy
        ∂g₃/∂x = -1 + y³, ∂g₃/∂y = 3xy²
        """
        g1 = 1.5 - x + x * y
        g2 = 2.25 - x + x * y**2
        g3 = 2.625 - x + x * y**3

        dg1_dx = -1 + y
        dg1_dy = x
        dg2_dx = -1 + y**2
        dg2_dy = 2 * x * y
        dg3_dx = -1 + y**3
        dg3_dy = 3 * x * y**2

        df_dx = 2 * g1 * dg1_dx + 2 * g2 * dg2_dx + 2 * g3 * dg3_dx
        df_dy = 2 * g1 * dg1_dy + 2 * g2 * dg2_dy + 2 * g3 * dg3_dy

        return (df_dx, df_dy)

    def hessian(self, x: float, y: float) -> np.ndarray:
        """
        Compute Hessian numerically for this complex function.
        (The analytical form is very messy)
        """
        eps = 1e-5

        # Finite difference approximation
        grad_x_plus = self.gradient(x + eps, y)
        grad_x_minus = self.gradient(x - eps, y)
        grad_y_plus = self.gradient(x, y + eps)
        grad_y_minus = self.gradient(x, y - eps)

        h_xx = (grad_x_plus[0] - grad_x_minus[0]) / (2 * eps)
        h_xy = (grad_y_plus[0] - grad_y_minus[0]) / (2 * eps)
        h_yy = (grad_y_plus[1] - grad_y_minus[1]) / (2 * eps)

        return np.array([[h_xx, h_xy], [h_xy, h_yy]])


class Rastrigin(Landscape):
    """
    The Rastrigin Function - Many Local Minima.

    f(x, y) = 20 + (x² - 10cos(2πx)) + (y² - 10cos(2πy))

    WHAT THIS TEACHES:
    ------------------
    - Has MANY local minima (like a waffle or egg carton)
    - Global minimum at (0, 0)
    - Standard gradient descent gets trapped in local minima!

    This shows WHY we need:
    - Random restarts
    - Simulated annealing
    - Genetic algorithms
    - Or larger learning rates to "jump over" local minima

    GRADIENT:
    ∂f/∂x = 2x + 20π·sin(2πx)
    ∂f/∂y = 2y + 20π·sin(2πy)
    """

    def __init__(self, A: float = 10.0):
        self.A = A
        self.name = "Rastrigin Function"
        self.description = (
            "f(x,y) = 20 + x² + y² - 10(cos(2πx) + cos(2πy)) - Many local minima!"
        )
        self.optimal_point = (0.0, 0.0)
        self.optimal_value = 0.0
        self.x_range = (-5.12, 5.12)
        self.y_range = (-5.12, 5.12)

    def evaluate(self, x: float, y: float) -> float:
        """f(x,y) = 2A + (x² - A·cos(2πx)) + (y² - A·cos(2πy))"""
        return (
            2 * self.A
            + (x**2 - self.A * np.cos(2 * np.pi * x))
            + (y**2 - self.A * np.cos(2 * np.pi * y))
        )

    def gradient(self, x: float, y: float) -> Tuple[float, float]:
        """
        ∂f/∂x = 2x + 2πA·sin(2πx)
        ∂f/∂y = 2y + 2πA·sin(2πy)
        """
        df_dx = 2 * x + 2 * np.pi * self.A * np.sin(2 * np.pi * x)
        df_dy = 2 * y + 2 * np.pi * self.A * np.sin(2 * np.pi * y)
        return (df_dx, df_dy)

    def hessian(self, x: float, y: float) -> np.ndarray:
        """
        ∂²f/∂x² = 2 + 4π²A·cos(2πx)
        ∂²f/∂y² = 2 + 4π²A·cos(2πy)
        ∂²f/∂x∂y = 0 (no cross terms)
        """
        h_xx = 2 + 4 * np.pi**2 * self.A * np.cos(2 * np.pi * x)
        h_yy = 2 + 4 * np.pi**2 * self.A * np.cos(2 * np.pi * y)
        return np.array([[h_xx, 0.0], [0.0, h_yy]])


class Matyas(Landscape):
    """
    The Matyas Function - A Convex Quadratic (Plate-Shaped)

    f(x, y) = 0.26(x^2 + y^2) - 0.48xy

    WHAT THIS TEACHES:
    ------------------
    - It's a convex function (one global minimum at 0,0)
    - BUT it is "ill-conditioned" (condition number = 25)
    - The contour lines are ellipses, not circles

    This is perfect for comparing Gradient Descent vs Conjugate Gradient:
    - GD will zigzag in the narrow valley
    - CG will converge in exactly 2 steps (for a 2D quadratic)!

    CONSTANTS:
    Hessian H = [[0.52, -0.48],
                 [-0.48, 0.52]]

    Eigenvalues:
    λ_min = 0.04 (Strong convexity constant)
    λ_max = 1.00 (Lipschitz constant)
    Condition Number κ = 25
    """

    def __init__(self):
        self.name = "Matyas Function"
        self.description = "f(x,y) = 0.26(x^2 + y^2) - 0.48xy - Quadratic with κ=25"
        self.optimal_point = (0.0, 0.0)
        self.optimal_value = 0.0
        self.x_range = (-10.0, 10.0)
        self.y_range = (-10.0, 10.0)

    def evaluate(self, x: float, y: float) -> float:
        return 0.26 * (x**2 + y**2) - 0.48 * x * y

    def gradient(self, x: float, y: float) -> Tuple[float, float]:
        df_dx = 0.52 * x - 0.48 * y
        df_dy = 0.52 * y - 0.48 * x
        return (df_dx, df_dy)

    def hessian(self, x: float, y: float) -> np.ndarray:
        return np.array([[0.52, -0.48], [-0.48, 0.52]])


class Himmelblau(Landscape):
    """
    The Himmelblau Function - Multi-Modal (4 Minima)

    f(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2

    WHAT THIS TEACHES:
    ------------------
    - It is NON-CONVEX (has ridges and valleys)
    - Has 4 identical global minima (f=0)
    - Has 1 local maximum and 4 saddle points

    This is perfect for:
    - Testing if an optimizer finds the *nearest* minimum
    - Testing behavior near saddle points
    - Testing Quasi-Newton (BFGS) on non-convex terrain

    Global Minima:
    (3, 2), (-2.805, 3.131), (-3.779, -3.283), (3.584, -1.848)
    """

    def __init__(self):
        self.name = "Himmelblau Function"
        self.description = "Has 4 distinct global minima - Tests multi-modal search"
        self.optimal_point = (3.0, 2.0)  # One of the 4
        self.optimal_value = 0.0
        self.x_range = (-5.0, 5.0)
        self.y_range = (-5.0, 5.0)

    def evaluate(self, x: float, y: float) -> float:
        try:
            return (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2
        except OverflowError:
            return float("inf")

    def gradient(self, x: float, y: float) -> Tuple[float, float]:
        try:
            df_dx = 2 * (x**2 + y - 11) * (2 * x) + 2 * (x + y**2 - 7)
            df_dy = 2 * (x**2 + y - 11) + 2 * (x + y**2 - 7) * (2 * y)
            return (df_dx, df_dy)
        except OverflowError:
            return (0.0, 0.0)  # Stop optimization if we hit infinity

    def hessian(self, x: float, y: float) -> np.ndarray:
        term1 = x**2 + y - 11
        term2 = x + y**2 - 7

        h_xx = 4 * term1 + 2 * (2 * x) * (2 * x) + 2
        h_xy = 2 * (2 * x) + 2 * (2 * y)
        h_yy = 2 + 4 * term2 + 2 * (2 * y) * (2 * y)

        return np.array([[h_xx, h_xy], [h_xy, h_yy]])


# Convenience dictionary of all landscapes
LANDSCAPES = {
    "valley": Valley,
    "saddle": SaddlePoint,
    "rosenbrock": Rosenbrock,
    "beale": Beale,
    "rastrigin": Rastrigin,
    "matyas": Matyas,
    "himmelblau": Himmelblau,
}


def get_landscape(name: str, **kwargs) -> Landscape:
    """
    Factory function to get a landscape by name.

    Args:
        name: One of 'valley', 'saddle', 'rosenbrock', 'beale', 'rastrigin'
        **kwargs: Parameters for the landscape (e.g., condition=50 for Valley)

    Returns:
        Landscape instance
    """
    if name.lower() not in LANDSCAPES:
        raise ValueError(
            f"Unknown landscape: {name}. Choose from: {list(LANDSCAPES.keys())}"
        )
    return LANDSCAPES[name.lower()](**kwargs)


if __name__ == "__main__":
    # Quick test
    print("Testing landscapes...")

    valley = Valley(condition=50)
    print(f"\n{valley.name}")
    print(f"  f(1, 1) = {valley.evaluate(1, 1)}")
    print(f"  gradient at (1, 1) = {valley.gradient(1, 1)}")
    print(f"  Hessian at (1, 1) =\n{valley.hessian(1, 1)}")
    print(f"  Condition number = {valley.condition_number(1, 1)}")

    saddle = SaddlePoint()
    print(f"\n{saddle.name}")
    print(f"  f(0.1, 0.1) = {saddle.evaluate(0.1, 0.1)}")
    print(f"  gradient at (0, 0) = {saddle.gradient(0, 0)} (THE PLATEAU!)")

    rosenbrock = Rosenbrock()
    print(f"\n{rosenbrock.name}")
    print(
        f"  Optimal at {rosenbrock.optimal_point}, value = {rosenbrock.optimal_value}"
    )
    print(f"  f(0, 0) = {rosenbrock.evaluate(0, 0)}")
