import json
import time
import numpy as np
from typing import Dict, Any, List

from .landscapes import Matyas, Himmelblau, Rosenbrock, Valley
from .optimizers import SGD, Momentum, ConjugateGradientPR, BFGS, Optimizer


def run_benchmark(landscape_name: str, steps: int = 200) -> Dict[str, Any]:
    """
    Run a benchmark comparing GD, CG-PR, and BFGS on a given landscape.
    """

    # 1. Setup Landscape
    if landscape_name.lower() == "matyas":
        landscape = Matyas()
        start_point = (4.0, 4.0)
    elif landscape_name.lower() == "himmelblau":
        landscape = Himmelblau()
        start_point = (0.0, 0.0)  # Saddle point start
    elif landscape_name.lower() == "rosenbrock":
        landscape = Rosenbrock()
        start_point = (-1.5, 1.0)
    elif landscape_name.lower() == "valley":
        landscape = Valley(condition=20)
        start_point = (-2.0, 1.0)
    else:
        raise ValueError(f"Unknown landscape: {landscape_name}")

    print(f"--- Benchmarking on {landscape.name} ---")
    print(f"Start Point: {start_point}")
    print(f"Optimal Value: {landscape.optimal_value}")

    # 2. Setup Optimizers
    # Tuned learning rates for fair comparison
    optimizers = [
        SGD(lr=0.05),
        Momentum(lr=0.01, beta=0.9),
        ConjugateGradientPR(lr=0.1),  # CG usually takes larger steps
        BFGS(lr=1.0),  # Quasi-Newton uses lr=1 typically
    ]

    results = {}

    for opt in optimizers:
        start_time = time.time()

        # Run optimization
        history = opt.optimize(
            landscape, start_x=start_point[0], start_y=start_point[1], steps=steps
        )

        end_time = time.time()
        duration = end_time - start_time

        # Analyze results
        final_state = history[-1]
        final_loss = final_state.loss
        final_grad_norm = np.linalg.norm([final_state.grad_x, final_state.grad_y])

        # Check convergence (simple threshold)
        converged_step = steps
        for i, state in enumerate(history):
            grad_norm = np.linalg.norm([state.grad_x, state.grad_y])
            if grad_norm < 1e-4:
                converged_step = i
                break

        print(
            f"{opt.name:25} | Steps: {converged_step:3} | Loss: {final_loss:.2e} | Time: {duration * 1000:.2f}ms"
        )

        results[opt.name] = {
            "steps_to_converge": converged_step,
            "final_loss": final_loss,
            "final_grad_norm": final_grad_norm,
            "time_ms": duration * 1000,
            "path": [
                [s.x, s.y] for s in history[: converged_step + 1]
            ],  # Only save path up to convergence
        }

    return results


if __name__ == "__main__":
    # Run the two key experiments
    print("\n=== EXPERIMENT 1: QUADRATIC CONVERGENCE (Matyas) ===")
    matyas_results = run_benchmark("matyas")

    print("\n=== EXPERIMENT 2: NON-CONVEX OPTIMIZATION (Himmelblau) ===")
    himmelblau_results = run_benchmark("himmelblau")

    # Save to JSON for the frontend or report
    full_report = {"matyas": matyas_results, "himmelblau": himmelblau_results}

    with open("benchmark_results.json", "w") as f:
        json.dump(full_report, f, indent=2)

    print("\nResults saved to benchmark_results.json")
