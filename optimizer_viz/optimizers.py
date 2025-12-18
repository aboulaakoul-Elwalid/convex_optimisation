"""
Optimizers Implemented FROM SCRATCH

This is the educational core of the project. Each optimizer is implemented
using only NumPy (no PyTorch autograd!) so you can see exactly what's happening.

CONNECTING TO YOUR MATH NOTES:
==============================

From your Gemini conversation, you learned:
1. The backward pass computes gradients (∂E/∂w)
2. The weight update is: w_new = w_old - lr * gradient
3. Adam normalizes the gradient by its variance

From your PDF (Chapter 4 - Méthodes de Descente):
1. "Pas Fixe" (Fixed Step) = SGD with constant learning rate
2. "Stratégie de Newton" = Using H⁻¹ @ gradient (what we call Newton/Muon)
3. The "Conjugate Gradient" idea = Similar to Momentum

This file implements these ideas so you can SEE them work!

OPTIMIZER STATE TRACKING:
=========================
Each optimizer tracks its full history:
- path: List of (x, y, loss) at each step
- gradients: Gradient at each step
- states: Internal state (m, v for Adam) at each step

This allows us to visualize not just WHERE they go, but WHY.

THE KEY INSIGHT - WHY ADAM WORKS:
=================================
From your conversation about "variance":

SGD:    step = lr * gradient
        (If gradient is 0.00001, step is 0.00001 - TINY!)

Adam:   step = lr * m_hat / (sqrt(v_hat) + eps)
        (If gradient is 0.00001 consistently, v_hat is also tiny,
         so the division AMPLIFIES the step!)

Adam effectively asks: "Is this gradient small because we're on a plateau,
or because we're near the minimum?" It uses variance to tell the difference.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from .landscapes import Landscape


@dataclass
class OptimizerState:
    """Snapshot of optimizer state at a single step."""

    step: int
    x: float
    y: float
    loss: float
    grad_x: float
    grad_y: float
    # Additional state (e.g., momentum, variance for Adam)
    extra: Dict[str, Any] = field(default_factory=dict)


class Optimizer(ABC):
    """
    Base class for all optimizers.

    USAGE:
    ------
    optimizer = SGD(lr=0.01)
    history = optimizer.optimize(landscape, start_x=-8, start_y=3, steps=100)

    The history contains the full path and state at each step.
    """

    name: str = "Base Optimizer"
    color: str = "blue"  # For visualization

    def __init__(self, lr: float = 0.01):
        self.lr = lr
        self.history: List[OptimizerState] = []

    @abstractmethod
    def step(
        self, x: float, y: float, grad_x: float, grad_y: float
    ) -> Tuple[float, float]:
        """
        Perform one optimization step.

        This is the CORE of each optimizer - the update rule!

        Args:
            x, y: Current position
            grad_x, grad_y: Gradient at current position

        Returns:
            (new_x, new_y): Updated position
        """
        pass

    def reset(self):
        """Reset optimizer state for a new run."""
        self.history = []

    def optimize(
        self,
        landscape: Landscape,
        start_x: float,
        start_y: float,
        steps: int = 100,
        verbose: bool = False,
    ) -> List[OptimizerState]:
        """
        Run optimization on a landscape.

        This is like running a training loop, but in 2D so we can visualize!

        Args:
            landscape: The loss surface to optimize on
            start_x, start_y: Starting position
            steps: Number of optimization steps
            verbose: Print progress

        Returns:
            List of OptimizerState objects (the full history)
        """
        self.reset()

        x, y = start_x, start_y

        for step_num in range(steps):
            # Forward pass: compute loss
            loss = landscape.evaluate(x, y)

            # Backward pass: compute gradient
            grad_x, grad_y = landscape.gradient(x, y)

            # Record state BEFORE update
            state = OptimizerState(
                step=step_num,
                x=x,
                y=y,
                loss=loss,
                grad_x=grad_x,
                grad_y=grad_y,
                extra=self._get_extra_state(),
            )
            self.history.append(state)

            if verbose and step_num % 10 == 0:
                print(
                    f"Step {step_num}: pos=({x:.4f}, {y:.4f}), loss={loss:.6f}, "
                    f"grad=({grad_x:.4f}, {grad_y:.4f})"
                )

            # Optimization step: update position
            x, y = self.step(x, y, grad_x, grad_y)

        # Record final state
        final_loss = landscape.evaluate(x, y)
        final_grad = landscape.gradient(x, y)
        self.history.append(
            OptimizerState(
                step=steps,
                x=x,
                y=y,
                loss=final_loss,
                grad_x=final_grad[0],
                grad_y=final_grad[1],
                extra=self._get_extra_state(),
            )
        )

        return self.history

    def _get_extra_state(self) -> Dict[str, Any]:
        """Override in subclasses to track additional state (e.g., momentum)."""
        return {}

    def get_path(self) -> np.ndarray:
        """Get the optimization path as a numpy array of shape (steps, 2)."""
        return np.array([[s.x, s.y] for s in self.history])

    def get_losses(self) -> np.ndarray:
        """Get the loss at each step."""
        return np.array([s.loss for s in self.history])


class SGD(Optimizer):
    """
    Stochastic Gradient Descent - The Simplest Optimizer

    UPDATE RULE:
    ============
    w_new = w_old - lr * gradient

    That's it! Just subtract the gradient scaled by learning rate.

    FROM YOUR NOTES:
    ================
    This is "Gradient à Pas Fixe" from your PDF (Section 4.1.2).
    The "pas" (step) is fixed = learning rate.

    MATH:
    =====
    x_new = x - lr * ∂f/∂x
    y_new = y - lr * ∂f/∂y

    PROBLEMS:
    =========
    1. On ill-conditioned landscapes (Valley): Bounces between walls
       - If gradient in y is 100x larger, step in y is 100x larger
       - This causes zig-zagging, not progress

    2. On plateaus (Saddle): Tiny gradient = tiny step = stuck
       - No mechanism to "sprint across" flat regions

    WHY WE NEED BETTER OPTIMIZERS:
    ==============================
    SGD treats all gradients equally. But some gradients are "noise" (steep walls)
    and some are "signal" (the actual direction to the minimum).
    """

    def __init__(self, lr: float = 0.01):
        super().__init__(lr)
        self.name = f"SGD (lr={lr})"
        self.color = "red"

    def step(
        self, x: float, y: float, grad_x: float, grad_y: float
    ) -> Tuple[float, float]:
        """
        SGD Update:
            x_new = x - lr * grad_x
            y_new = y - lr * grad_y

        Simple subtraction! No memory of past gradients.
        """
        new_x = x - self.lr * grad_x
        new_y = y - self.lr * grad_y
        return new_x, new_y


class Momentum(Optimizer):
    """
    SGD with Momentum - Adding "Inertia" to Optimization

    UPDATE RULE:
    ============
    v = beta * v_prev + gradient        # Accumulate velocity
    w_new = w_old - lr * v              # Use velocity for update

    FROM YOUR NOTES:
    ================
    This is related to "Gradient Conjugué" from your PDF.
    Instead of just following the current gradient, we remember past gradients.

    MATH:
    =====
    v_x = beta * v_x + grad_x           # Velocity in x
    v_y = beta * v_y + grad_y           # Velocity in y
    x_new = x - lr * v_x
    y_new = y - lr * v_y

    INTUITION:
    ==========
    Think of a ball rolling down a hill:
    - It builds up speed (momentum) as it goes
    - It can roll through small bumps without stopping
    - It can overshoot and oscillate around the minimum

    ADVANTAGES:
    ===========
    1. Smooths out zig-zagging (bounces average out)
    2. Accelerates through flat regions (momentum builds up)
    3. Can escape shallow local minima

    DISADVANTAGES:
    ==============
    1. Can overshoot the minimum (too much momentum)
    2. Needs careful tuning of beta
    3. Still doesn't adapt to curvature like Adam
    """

    def __init__(self, lr: float = 0.01, beta: float = 0.9):
        super().__init__(lr)
        self.beta = beta
        self.name = f"Momentum (lr={lr}, β={beta})"
        self.color = "blue"

        # Velocity state
        self.v_x = 0.0
        self.v_y = 0.0

    def reset(self):
        super().reset()
        self.v_x = 0.0
        self.v_y = 0.0

    def step(
        self, x: float, y: float, grad_x: float, grad_y: float
    ) -> Tuple[float, float]:
        """
        Momentum Update:
            v = beta * v_prev + gradient
            w_new = w - lr * v

        The velocity accumulates past gradients with exponential decay.
        """
        # Update velocity (accumulate momentum)
        self.v_x = self.beta * self.v_x + grad_x
        self.v_y = self.beta * self.v_y + grad_y

        # Update position using velocity
        new_x = x - self.lr * self.v_x
        new_y = y - self.lr * self.v_y

        return new_x, new_y

    def _get_extra_state(self) -> Dict[str, Any]:
        return {"v_x": self.v_x, "v_y": self.v_y}


class Adam(Optimizer):
    """
    Adam - Adaptive Moment Estimation

    THE OPTIMIZER YOU'VE BEEN LEARNING ABOUT!

    UPDATE RULE:
    ============
    m = beta1 * m + (1 - beta1) * gradient          # First moment (mean)
    v = beta2 * v + (1 - beta2) * gradient²         # Second moment (variance)
    m_hat = m / (1 - beta1^t)                       # Bias correction
    v_hat = v / (1 - beta2^t)                       # Bias correction
    w_new = w - lr * m_hat / (sqrt(v_hat) + eps)    # Normalized update

    FROM YOUR GEMINI CONVERSATION:
    ==============================
    "Adam looks at that tiny gradient of 0.00001 and asks a second question:
     'Has the gradient been consistently tiny for a while?' (Variance)"

    If gradient is tiny AND variance is tiny, Adam AMPLIFIES the step!
    This is how it "sprints across plateaus."

    THE KEY INSIGHT:
    ================
    The division by sqrt(v_hat) is the magic:

    - Large gradient, large variance → Normal step (they cancel)
    - Small gradient, small variance → LARGE step (division amplifies!)
    - Large gradient, small variance → This means big, consistent signal!

    BIAS CORRECTION:
    ================
    At the start (t=1), m and v are initialized to 0.
    This makes early estimates biased toward 0.

    The correction m_hat = m / (1 - beta1^t) fixes this:
    - At t=1: m_hat = m / (1 - 0.9) = m / 0.1 = 10 * m (big boost!)
    - At t=100: m_hat = m / (1 - 0.9^100) ≈ m (no correction needed)

    WHY BETA VALUES:
    ================
    - beta1 = 0.9: Smooth the gradient over ~10 steps
    - beta2 = 0.999: Smooth the variance over ~1000 steps

    We smooth variance more because we want a stable estimate of the
    "typical gradient magnitude" for this parameter.
    """

    def __init__(
        self,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.name = f"Adam (lr={lr})"
        self.color = "green"

        # First moment (mean of gradients)
        self.m_x = 0.0
        self.m_y = 0.0

        # Second moment (mean of squared gradients = variance proxy)
        self.v_x = 0.0
        self.v_y = 0.0

        # Timestep (for bias correction)
        self.t = 0

    def reset(self):
        super().reset()
        self.m_x = 0.0
        self.m_y = 0.0
        self.v_x = 0.0
        self.v_y = 0.0
        self.t = 0

    def step(
        self, x: float, y: float, grad_x: float, grad_y: float
    ) -> Tuple[float, float]:
        """
        Adam Update - The Full Algorithm!

        This is the exact math from the Adam paper (Kingma & Ba, 2014).
        """
        self.t += 1

        # ============================================
        # STEP 1: Update first moment (gradient mean)
        # ============================================
        # m = beta1 * m + (1 - beta1) * gradient
        # This is an exponential moving average of the gradient
        self.m_x = self.beta1 * self.m_x + (1 - self.beta1) * grad_x
        self.m_y = self.beta1 * self.m_y + (1 - self.beta1) * grad_y

        # ================================================
        # STEP 2: Update second moment (gradient variance)
        # ================================================
        # v = beta2 * v + (1 - beta2) * gradient²
        # This is an EMA of the SQUARED gradient (variance proxy)
        self.v_x = self.beta2 * self.v_x + (1 - self.beta2) * grad_x**2
        self.v_y = self.beta2 * self.v_y + (1 - self.beta2) * grad_y**2

        # ========================
        # STEP 3: Bias correction
        # ========================
        # At early timesteps, m and v are biased toward 0
        # because they were initialized to 0.
        # This correction "inflates" them to the correct scale.
        m_x_hat = self.m_x / (1 - self.beta1**self.t)
        m_y_hat = self.m_y / (1 - self.beta1**self.t)
        v_x_hat = self.v_x / (1 - self.beta2**self.t)
        v_y_hat = self.v_y / (1 - self.beta2**self.t)

        # =====================
        # STEP 4: Update weights
        # =====================
        # w_new = w - lr * m_hat / (sqrt(v_hat) + eps)
        #
        # THE KEY: Division by sqrt(v_hat) normalizes the step!
        # - If variance is high (gradient is noisy), take smaller steps
        # - If variance is low (gradient is consistent), take larger steps
        new_x = x - self.lr * m_x_hat / (np.sqrt(v_x_hat) + self.eps)
        new_y = y - self.lr * m_y_hat / (np.sqrt(v_y_hat) + self.eps)

        return new_x, new_y

    def _get_extra_state(self) -> Dict[str, Any]:
        # Compute bias-corrected values for visualization
        if self.t > 0:
            m_x_hat = self.m_x / (1 - self.beta1**self.t)
            m_y_hat = self.m_y / (1 - self.beta1**self.t)
            v_x_hat = self.v_x / (1 - self.beta2**self.t)
            v_y_hat = self.v_y / (1 - self.beta2**self.t)
        else:
            m_x_hat = m_y_hat = v_x_hat = v_y_hat = 0.0

        return {
            "m_x": self.m_x,
            "m_y": self.m_y,
            "v_x": self.v_x,
            "v_y": self.v_y,
            "m_x_hat": m_x_hat,
            "m_y_hat": m_y_hat,
            "v_x_hat": v_x_hat,
            "v_y_hat": v_y_hat,
            "t": self.t,
            # The effective step size (what we divide by)
            "step_scale_x": np.sqrt(v_x_hat) + self.eps if self.t > 0 else 1.0,
            "step_scale_y": np.sqrt(v_y_hat) + self.eps if self.t > 0 else 1.0,
        }


class Newton(Optimizer):
    """
    Newton's Method - Using Curvature Information

    UPDATE RULE:
    ============
    w_new = w - H⁻¹ @ gradient

    Where H is the Hessian matrix (second derivatives = curvature).

    FROM YOUR PDF (Section 4.1.1.1 - Stratégie de Newton):
    ======================================================
    "La stratégie de Newton consiste à approximer f par son développement
     de Taylor à l'ordre 2 et à minimiser cette approximation."

    Instead of just following the gradient (first derivative),
    Newton uses curvature (second derivative) to find a better direction.

    INTUITION:
    ==========
    Imagine you're in a narrow valley:
    - The gradient points at a steep wall (not toward the minimum!)
    - The Hessian knows the valley is narrow in one direction
    - H⁻¹ "rotates" the gradient to point down the valley floor

    THE MATH:
    =========
    For quadratic functions, Newton converges in ONE step!

    f(x,y) = x² + 50y²
    grad = [2x, 100y]
    H = [[2, 0], [0, 100]]
    H⁻¹ = [[0.5, 0], [0, 0.01]]

    H⁻¹ @ grad = [0.5 * 2x, 0.01 * 100y] = [x, y]

    So: w_new = w - [x, y] = [0, 0]  (The minimum!)

    PROBLEMS:
    =========
    1. Computing H⁻¹ is O(n³) for n parameters - impossible for neural nets!
    2. H might not be positive definite (saddle points) → wrong direction
    3. Far from minimum, quadratic approximation is bad

    This is why we use APPROXIMATIONS like Muon.
    """

    def __init__(self, lr: float = 1.0, damping: float = 0.0):
        """
        Args:
            lr: Learning rate (usually 1.0 for pure Newton)
            damping: Add to diagonal for stability (Levenberg-Marquardt)
        """
        super().__init__(lr)
        self.damping = damping
        self.name = f"Newton (lr={lr})"
        self.color = "purple"
        self.landscape: Optional[Landscape] = None

    def set_landscape(self, landscape: Landscape):
        """Newton needs access to the Hessian, so we store the landscape."""
        self.landscape = landscape

    def step(
        self, x: float, y: float, grad_x: float, grad_y: float
    ) -> Tuple[float, float]:
        """
        Newton Update:
            w_new = w - lr * H⁻¹ @ gradient

        For 2D, H is just a 2x2 matrix, so inversion is cheap.
        """
        if self.landscape is None:
            raise ValueError("Must call set_landscape() before optimizing!")

        # Get the Hessian at current point
        H = self.landscape.hessian(x, y)

        # Add damping for stability (like Levenberg-Marquardt)
        H_damped = H + self.damping * np.eye(2)

        # Compute the gradient vector
        grad = np.array([grad_x, grad_y])

        try:
            # Compute H⁻¹ @ gradient
            # This is the "Newton direction"
            H_inv = np.linalg.inv(H_damped)
            newton_step = H_inv @ grad
        except np.linalg.LinAlgError:
            # If Hessian is singular, fall back to gradient descent
            newton_step = grad

        # Update
        new_x = x - self.lr * newton_step[0]
        new_y = y - self.lr * newton_step[1]

        return new_x, new_y

    def optimize(
        self,
        landscape: Landscape,
        start_x: float,
        start_y: float,
        steps: int = 100,
        verbose: bool = False,
    ) -> List[OptimizerState]:
        """Override to set landscape automatically."""
        self.set_landscape(landscape)
        return super().optimize(landscape, start_x, start_y, steps, verbose)

    def _get_extra_state(self) -> Dict[str, Any]:
        return {"method": "exact_hessian"}


class Muon(Optimizer):
    """
    Muon - Momentum Orthogonalized Update

    A MODERN APPROXIMATION TO NEWTON'S METHOD!

    FROM YOUR GEMINI CONVERSATION:
    ==============================
    "Muon uses a trick called Newton-Schulz Iteration to approximate
     the whitening effect without storing the matrix."

    THE IDEA:
    =========
    We want to "whiten" the gradient - remove correlations and normalize scale.
    This is similar to what the Hessian inverse does, but cheaper!

    Newton-Schulz iteration finds the matrix square root inverse:
        X_new = 0.5 * X @ (3*I - X @ X)

    After a few iterations, X approaches A^(-1/2).

    For optimization, we want to orthogonalize the gradient matrix.
    This means making the singular values all equal to 1.

    SIMPLIFIED VERSION (What we implement):
    =======================================
    For 2D, we implement a simplified version:
    1. Compute gradient covariance over recent steps
    2. Use Newton-Schulz to approximate the whitening matrix
    3. Apply it to the gradient

    REAL MUON (for neural nets):
    ============================
    The real Muon (Keller Jordan, Modded-NanoGPT) works on weight matrices:
    1. Accumulate gradients with momentum
    2. Use Newton-Schulz to orthogonalize the gradient matrix
    3. The result has orthogonal columns (like in SVD with all σ = 1)

    This is "Muon" = "Momentum + Orthogonalization"
    """

    def __init__(
        self,
        lr: float = 0.02,
        momentum: float = 0.95,
        ns_steps: int = 5,  # Newton-Schulz iterations
        use_simplified: bool = True,  # Use simplified 2D version
    ):
        super().__init__(lr)
        self.momentum = momentum
        self.ns_steps = ns_steps
        self.use_simplified = use_simplified
        self.name = f"Muon (lr={lr})"
        self.color = "orange"

        # Momentum buffer
        self.m_x = 0.0
        self.m_y = 0.0

        # Gradient history for covariance estimation
        self.grad_history: List[np.ndarray] = []
        self.history_size = 10

    def reset(self):
        super().reset()
        self.m_x = 0.0
        self.m_y = 0.0
        self.grad_history = []

    def _newton_schulz_2x2(self, M: np.ndarray, steps: int = 5) -> np.ndarray:
        """
        Newton-Schulz iteration to approximate M^(-1/2).

        Algorithm:
            X_0 = M / ||M||  (normalize)
            X_{k+1} = 0.5 * X_k @ (3*I - X_k @ X_k)

        After convergence: X @ X ≈ M^(-1)
        So X ≈ M^(-1/2)
        """
        # Normalize
        norm = np.linalg.norm(M, "fro")
        if norm < 1e-8:
            return np.eye(2)
        X = M / norm

        I = np.eye(2)
        for _ in range(steps):
            X = 0.5 * X @ (3 * I - X @ X)

        # Scale back
        return X / np.sqrt(norm)

    def step(
        self, x: float, y: float, grad_x: float, grad_y: float
    ) -> Tuple[float, float]:
        """
        Muon Update - Momentum + Orthogonalization

        For 2D, we implement a simplified version that captures the essence.
        """
        grad = np.array([grad_x, grad_y])

        # Update momentum (like regular momentum)
        self.m_x = self.momentum * self.m_x + grad_x
        self.m_y = self.momentum * self.m_y + grad_y
        m = np.array([self.m_x, self.m_y])

        # Store gradient for covariance estimation
        self.grad_history.append(grad)
        if len(self.grad_history) > self.history_size:
            self.grad_history.pop(0)

        if self.use_simplified and len(self.grad_history) >= 2:
            # Estimate gradient covariance
            grads = np.array(self.grad_history)  # (n, 2)
            cov = grads.T @ grads / len(grads)  # (2, 2)

            # Add small regularization
            cov += 1e-4 * np.eye(2)

            # Newton-Schulz to get whitening matrix
            W = self._newton_schulz_2x2(cov, self.ns_steps)

            # Orthogonalize the momentum
            m_ortho = W @ m
        else:
            # Not enough history, use regular momentum
            m_ortho = m

        # Normalize the orthogonalized update
        norm = np.linalg.norm(m_ortho)
        if norm > 1e-8:
            m_ortho = m_ortho / norm * np.linalg.norm(m)

        # Update
        new_x = x - self.lr * m_ortho[0]
        new_y = y - self.lr * m_ortho[1]

        return new_x, new_y

    def _get_extra_state(self) -> Dict[str, Any]:
        return {
            "m_x": self.m_x,
            "m_y": self.m_y,
            "grad_history_size": len(self.grad_history),
        }


class RMSProp(Optimizer):
    """
    RMSProp - Root Mean Square Propagation

    THE PREDECESSOR TO ADAM!

    UPDATE RULE:
    ============
    v = beta * v + (1 - beta) * gradient²    # Running average of squared gradients
    w_new = w - lr * gradient / (sqrt(v) + eps)

    HISTORY:
    ========
    Proposed by Geoffrey Hinton in his Coursera lectures (unpublished!).
    It's like Adam but without the first moment (mean of gradients).

    WHY IT WORKS:
    =============
    Same as Adam's variance normalization:
    - Divides by sqrt of recent squared gradients
    - Parameters with large gradients get smaller updates
    - Parameters with small gradients get larger updates

    COMPARISON TO ADAM:
    ===================
    RMSProp: step = lr * gradient / sqrt(v)
    Adam:    step = lr * m / sqrt(v)

    Adam replaces the raw gradient with a smoothed version (m).
    This gives Adam better momentum behavior.
    """

    def __init__(self, lr: float = 0.01, beta: float = 0.9, eps: float = 1e-8):
        super().__init__(lr)
        self.beta = beta
        self.eps = eps
        self.name = f"RMSProp (lr={lr})"
        self.color = "cyan"

        self.v_x = 0.0
        self.v_y = 0.0

    def reset(self):
        super().reset()
        self.v_x = 0.0
        self.v_y = 0.0

    def step(
        self, x: float, y: float, grad_x: float, grad_y: float
    ) -> Tuple[float, float]:
        """
        RMSProp Update:
            v = beta * v + (1 - beta) * gradient²
            w_new = w - lr * gradient / (sqrt(v) + eps)
        """
        # Update running average of squared gradients
        self.v_x = self.beta * self.v_x + (1 - self.beta) * grad_x**2
        self.v_y = self.beta * self.v_y + (1 - self.beta) * grad_y**2

        # Normalized update
        new_x = x - self.lr * grad_x / (np.sqrt(self.v_x) + self.eps)
        new_y = y - self.lr * grad_y / (np.sqrt(self.v_y) + self.eps)

        return new_x, new_y

    def _get_extra_state(self) -> Dict[str, Any]:
        return {"v_x": self.v_x, "v_y": self.v_y}


class ConjugateGradientPR(Optimizer):
    """
    Conjugate Gradient (Polak-Ribière) - The "Smart" Momentum.
    """

    def __init__(self, lr: float = 0.05):
        super().__init__(lr)
        self.name = "Conjugate Gradient (PR)"
        self.color = "lime"
        self.prev_g = None
        self.prev_d = None
        self.landscape: Optional[Landscape] = None

    def set_landscape(self, landscape: Landscape):
        self.landscape = landscape

    def optimize(
        self,
        landscape: Landscape,
        start_x: float,
        start_y: float,
        steps: int = 100,
        verbose: bool = False,
    ):
        self.set_landscape(landscape)
        return super().optimize(landscape, start_x, start_y, steps, verbose)

    def line_search(self, x, y, d_x, d_y):
        """Golden Section Line Search"""
        if self.landscape is None:
            return self.lr  # Fallback

        # Bracket the minimum
        a = 0.0
        b = 1.0  # Initial bracket size (guess)

        # Grow bracket until we find a value higher than previous
        # (This is a simplification, we assume min is within [0, 1] or close)
        # For efficiency in this demo, we'll use a fixed number of iterations of Golden Section

        gr = (np.sqrt(5) - 1) / 2
        c = b - (b - a) * gr
        d = a + (b - a) * gr

        fc = self.landscape.evaluate(x + c * d_x, y + c * d_y)
        fd = self.landscape.evaluate(x + d * d_x, y + d * d_y)

        for _ in range(10):  # 10 iterations is enough for demo
            if fc < fd:
                b = d
                d = c
                fd = fc
                c = b - (b - a) * gr
                fc = self.landscape.evaluate(x + c * d_x, y + c * d_y)
            else:
                a = c
                c = d
                fc = fd
                d = a + (b - a) * gr
                fd = self.landscape.evaluate(x + d * d_x, y + d * d_y)

        return (b + a) / 2

    def reset(self):
        super().reset()
        self.prev_g = None
        self.prev_d = None

    def step(
        self, x: float, y: float, grad_x: float, grad_y: float
    ) -> Tuple[float, float]:
        g = np.array([grad_x, grad_y])

        if self.prev_g is None or self.prev_d is None:
            d = -g
            beta = 0.0
        else:
            # Polak-Ribiere
            y_k = g - self.prev_g
            num = np.dot(g, y_k)
            den = np.dot(self.prev_g, self.prev_g) + 1e-10
            beta = max(0.0, float(num / den))

            d = -g + beta * self.prev_d

            # Restart if not descent
            if np.dot(d, g) >= 0:
                d = -g
                beta = 0.0

        # Perform Line Search
        alpha = self.lr

        if self.landscape:
            # "Cheat" for demo stability if we have Hessian (Quadratic)
            try:
                H = self.landscape.hessian(x, y)
                denom = np.dot(d, H @ d)
                if denom > 1e-10:
                    # Exact line search for quadratic
                    alpha = -np.dot(g, d) / denom
                else:
                    alpha = self.line_search(x, y, d[0], d[1])
            except:
                alpha = self.line_search(x, y, d[0], d[1])
        else:
            alpha = self.lr

        # Apply step
        new_pos = np.array([x, y]) + alpha * d

        self.prev_g = g
        self.prev_d = d

        return new_pos[0], new_pos[1]

    def _get_extra_state(self) -> Dict[str, Any]:
        return {"beta": 0.0 if self.prev_g is None else "calc"}


class BFGS(Optimizer):
    """
    BFGS (Broyden–Fletcher–Goldfarb–Shanno) - Quasi-Newton.
    """

    def __init__(self, lr: float = 1.0):
        super().__init__(lr)
        self.name = "BFGS"
        self.color = "magenta"
        self.H_inv = np.eye(2)
        self.prev_pos = None
        self.prev_grad = None
        self.landscape: Optional[Landscape] = None

    def set_landscape(self, landscape: Landscape):
        self.landscape = landscape

    def optimize(
        self,
        landscape: Landscape,
        start_x: float,
        start_y: float,
        steps: int = 100,
        verbose: bool = False,
    ):
        self.set_landscape(landscape)
        return super().optimize(landscape, start_x, start_y, steps, verbose)

    def reset(self):
        super().reset()
        self.H_inv = np.eye(2)
        self.prev_pos = None
        self.prev_grad = None

    def step(
        self, x: float, y: float, grad_x: float, grad_y: float
    ) -> Tuple[float, float]:
        current_pos = np.array([x, y])
        current_grad = np.array([grad_x, grad_y])

        if self.prev_pos is not None and self.prev_grad is not None:
            s = current_pos - self.prev_pos
            y_vec = current_grad - self.prev_grad

            rho_denom = np.dot(y_vec, s)
            if abs(rho_denom) > 1e-10:
                rho = 1.0 / rho_denom
                I = np.eye(2)
                V = I - rho * np.outer(s, y_vec)
                self.H_inv = V @ self.H_inv @ V.T + rho * np.outer(s, s)

        p = -self.H_inv @ current_grad

        # Check descent
        if np.dot(p, current_grad) > 0:
            p = -current_grad
            self.H_inv = np.eye(2)  # Reset Hessian if bad direction

        # Line Search (Wolfe is better, but backtracking is simpler)
        # We'll use a simple backtracking line search
        alpha = 1.0
        c = 0.5
        tau = 0.5

        if self.landscape:
            # Simple Armijo backtracking
            f_curr = self.landscape.evaluate(x, y)
            m = np.dot(current_grad, p)
            # Ensure m is negative (descent)

            for _ in range(10):
                new_x = x + alpha * p[0]
                new_y = y + alpha * p[1]
                # Check bounds/overflow
                if abs(new_x) > 1e4 or abs(new_y) > 1e4:
                    f_new = float("inf")
                else:
                    f_new = self.landscape.evaluate(new_x, new_y)

                if f_new <= f_curr + alpha * 1e-4 * m:
                    break
                alpha *= tau

        new_pos = current_pos + alpha * p

        self.prev_pos = current_pos
        self.prev_grad = current_grad

        return new_pos[0], new_pos[1]

    def _get_extra_state(self) -> Dict[str, Any]:
        return {"H_inv_approx": "updated" if self.prev_pos is not None else "identity"}


# Convenience dictionary of all optimizers
OPTIMIZERS = {
    "sgd": SGD,
    "momentum": Momentum,
    "adam": Adam,
    "rmsprop": RMSProp,
    "newton": Newton,
    "muon": Muon,
    "cg_pr": ConjugateGradientPR,
    "bfgs": BFGS,
}


def get_optimizer(name: str, **kwargs) -> Optimizer:
    """
    Factory function to get an optimizer by name.

    Args:
        name: One of 'sgd', 'momentum', 'adam', 'rmsprop', 'newton', 'muon', 'cg_pr', 'bfgs'
        **kwargs: Parameters for the optimizer

    Returns:
        Optimizer instance
    """
    if name.lower() not in OPTIMIZERS:
        raise ValueError(
            f"Unknown optimizer: {name}. Choose from: {list(OPTIMIZERS.keys())}"
        )
    return OPTIMIZERS[name.lower()](**kwargs)


def create_comparison_set(
    include: Optional[List[str]] = None, lr_scale: float = 1.0
) -> List[Optimizer]:
    """
    Create a set of optimizers with default settings for comparison.

    The learning rates are tuned for typical landscapes.

    Args:
        include: List of optimizer names to include (None = all)
        lr_scale: Scale all learning rates by this factor

    Returns:
        List of configured optimizers
    """
    defaults = [
        SGD(lr=0.03 * lr_scale),
        Momentum(lr=0.01 * lr_scale, beta=0.9),
        RMSProp(lr=0.1 * lr_scale),
        Adam(lr=0.5 * lr_scale),
        Newton(lr=1.0, damping=0.1),
        Muon(lr=0.02 * lr_scale),
        ConjugateGradientPR(lr=0.1 * lr_scale),
        BFGS(lr=0.5 * lr_scale),
    ]

    if include is None:
        return defaults

    return [
        opt
        for opt in defaults
        if any(name.lower() in opt.name.lower() for name in include)
    ]


if __name__ == "__main__":
    # Quick test
    from .landscapes import Valley

    print("Testing optimizers on Valley landscape...")
    valley = Valley(condition=50)

    for OptimizerClass in [SGD, Momentum, Adam, Newton]:
        if OptimizerClass == Newton:
            opt = OptimizerClass(lr=1.0)
        elif OptimizerClass == Adam:
            opt = OptimizerClass(lr=0.5)
        elif OptimizerClass == Momentum:
            opt = OptimizerClass(lr=0.01)
        else:
            opt = OptimizerClass(lr=0.03)

        history = opt.optimize(valley, start_x=-8, start_y=3, steps=50)

        print(f"\n{opt.name}:")
        print(
            f"  Start: ({history[0].x:.4f}, {history[0].y:.4f}), loss={history[0].loss:.4f}"
        )
        print(
            f"  End:   ({history[-1].x:.4f}, {history[-1].y:.4f}), loss={history[-1].loss:.4f}"
        )
