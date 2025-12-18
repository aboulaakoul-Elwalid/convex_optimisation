from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import math
from typing import List, Dict, Any, Optional, Tuple, Callable


# Import your actual math libraries
from . import landscapes
from . import optimizers

app = FastAPI()

# Allow CORS so the browser (running on port 8080) can talk to this API (port 8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SimulationRequest(BaseModel):
    algo: str
    landscape: str
    start_x: float
    start_y: float
    lr: float
    steps: int = 100
    validate_pd: bool = False
    force_run: bool = False
    matrix_a: Optional[List[List[float]]] = None
    vector_b: Optional[List[float]] = None


class StepData(BaseModel):
    iter: int
    x: float
    y: float
    z: float
    grad_norm: float


class ValidationInfo(BaseModel):
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    skipped: bool = False


class SimulationResponse(BaseModel):
    path: List[StepData]
    converged: bool
    validation: ValidationInfo = Field(default_factory=ValidationInfo)


SAFE_FLOAT_LIMIT = 1e12


def validate_spd_matrix(matrix: List[List[float]], lr: float):
    """Validate SPD and learning-rate bound for quadratic forms."""
    warnings: List[str] = []
    errors: List[str] = []

    mat = np.array(matrix, dtype=float)
    if mat.shape != (2, 2):
        errors.append("Matrix A must be 2x2 for the quadratic validation.")
        return False, None, warnings, errors

    if not np.allclose(mat, mat.T, atol=1e-8):
        errors.append("Matrix A must be symmetric to be SPD.")
        return False, None, warnings, errors

    eigenvalues = np.linalg.eigvalsh(mat)
    lambda_min = float(np.min(eigenvalues))
    lambda_max = float(np.max(eigenvalues))

    if lambda_min <= 0:
        errors.append(
            f"Matrix is not positive definite (λ_min={lambda_min:.4e}). Unique minimum not guaranteed."
        )
        return False, None, warnings, errors

    step_limit = None
    if lambda_max > 0:
        step_limit = (2 * lambda_min) / (lambda_max**2)
        if lr >= step_limit:
            warnings.append(
                f"Learning rate {lr:.4f} exceeds theoretical stability bound ({step_limit:.4f})."
            )

    return True, step_limit, warnings, errors


def safe_float(
    val: Any, limit: float = SAFE_FLOAT_LIMIT, fallback: float = 0.0
) -> float:
    """Clamp values to a safe finite float for JSON serialization."""
    try:
        if val is None:
            return fallback
        val = float(val)
    except (TypeError, ValueError, OverflowError):
        return fallback

    if not math.isfinite(val):
        return math.copysign(limit, val if isinstance(val, (int, float)) else 1.0)

    if abs(val) > limit:
        return math.copysign(limit, val)

    return val


@app.post("/simulate", response_model=SimulationResponse)
async def simulate(req: SimulationRequest):
    try:
        # 1. Select Landscape
        landscape_cls = getattr(landscapes, req.landscape, None)
        if not landscape_cls:
            # Fallback for case-insensitive lookup
            ls_map = {
                "matyas": landscapes.Matyas,
                "himmelblau": landscapes.Himmelblau,
                "rosenbrock": landscapes.Rosenbrock,
            }
            landscape_cls = ls_map.get(req.landscape.lower())

        if not landscape_cls:
            raise HTTPException(
                status_code=400, detail=f"Unknown landscape: {req.landscape}"
            )

        landscape = landscape_cls()

        # 2. Select Optimizer
        # Map frontend names to Python class names
        algo_map = {
            "GD": optimizers.SGD,  # SGD is the fixed step gradient descent
            "Momentum": optimizers.Momentum,
            "CG_PR": optimizers.ConjugateGradientPR,
            "BFGS": optimizers.BFGS,
        }

        opt_cls = algo_map.get(req.algo)
        if not opt_cls:
            raise HTTPException(
                status_code=400, detail=f"Unknown optimizer: {req.algo}"
            )

        optimizer = opt_cls(lr=req.lr)

        validation_info = ValidationInfo()

        if req.validate_pd and req.matrix_a:
            is_spd, step_limit, warnings, errors = validate_spd_matrix(
                req.matrix_a, req.lr
            )
            validation_info.warnings.extend(warnings)
            validation_info.errors.extend(errors)

            if not is_spd and not req.force_run:
                validation_info.skipped = True
                return SimulationResponse(
                    path=[], converged=False, validation=validation_info
                )

            if step_limit is not None and req.lr >= step_limit and not req.force_run:
                validation_info.errors.append(
                    f"Learning rate {req.lr:.4f} exceeds theoretical bound ({step_limit:.4f})."
                )
                validation_info.skipped = True
                return SimulationResponse(
                    path=[], converged=False, validation=validation_info
                )

        max_iter = req.steps

        # 3. Run Simulation (Using YOUR Python Code)
        # The .optimize() method returns a list of OptimizerState objects
        history = optimizer.optimize(
            landscape=landscape,
            start_x=req.start_x,
            start_y=req.start_y,
            steps=max_iter,
        )

        # 4. Format Response with Safety Checks
        response_path = []
        last_grad_norm = float("inf")
        last_loss = float("inf")
        capped_any = False

        for state in history:
            try:
                grad_n = math.hypot(state.grad_x, state.grad_y)
            except (OverflowError, ValueError):
                grad_n = float("inf")

            safe_x = safe_float(state.x)
            safe_y = safe_float(state.y)
            safe_loss = safe_float(state.loss)
            safe_grad = safe_float(grad_n)

            if abs(safe_x) >= SAFE_FLOAT_LIMIT or abs(safe_y) >= SAFE_FLOAT_LIMIT:
                capped_any = True
            if abs(safe_loss) >= SAFE_FLOAT_LIMIT or abs(safe_grad) >= SAFE_FLOAT_LIMIT:
                capped_any = True

            response_path.append(
                StepData(
                    iter=state.step,
                    x=safe_x,
                    y=safe_y,
                    z=safe_loss,
                    grad_norm=safe_grad,
                )
            )

            last_grad_norm = safe_grad
            last_loss = safe_loss

        if not response_path:
            converged = False
        else:
            diverged = (
                not math.isfinite(last_grad_norm)
                or last_grad_norm >= 1e6
                or not math.isfinite(last_loss)
                or last_loss >= 1e6
                or capped_any
                or (response_path[-1].iter >= max_iter and last_grad_norm > 1e-3)
            )
            converged = not diverged and last_grad_norm < 1e-4 and abs(last_loss) < 1e6

        return SimulationResponse(
            path=response_path, converged=converged, validation=validation_info
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    print("Starting Optimization Backend...")
    print("This server runs YOUR Python code to power the web visualization.")
    uvicorn.run(app, host="0.0.0.0", port=8000)
