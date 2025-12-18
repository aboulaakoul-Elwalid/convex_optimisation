# Mathematical Convergence Analysis & Optimization Visualizer

A comprehensive toolkit for analyzing and visualizing mathematical optimization algorithms (GD, CG-PR, BFGS) on complex loss landscapes.

## Quick Start

This project uses a **Python Backend** to run the exact math from your codebase and a **WebGL Frontend** to visualize the results.

### 1. Start the Backend API (Terminal 1)

```bash
# Install dependencies if needed
pip install fastapi uvicorn numpy

# Run the server
python3 server.py
```
*You should see: "Uvicorn running on http://0.0.0.0:8000"*

### 2. Start the Frontend (Terminal 2)

```bash
cd web
python3 -m http.server 8080
```

### 3. Open the App

Navigate to: **http://localhost:8080/convergence_analysis.html**

---

## Features

### Single Source of Truth Architecture
- **Backend (`server.py`):** Imports your exact `optimizers.py` classes. When you click "Run", the Python code executes the optimization loop.
- **Frontend:** Visualizes the trajectory data returned by Python.
- **Benefit:** The visualization is 100% faithful to the Python implementation.

### Interactive 3D Analysis
- **Presentation-Ready Theme:** Clean white background
- **Real-Time Benchmarking:** Watch algorithms race side-by-side
- **Visual Clamping:** Solves visibility issues on steep functions (Himmelblau)
- **Click-to-Set Start:** Click anywhere on the 3D surface to set the starting point
- **Manual Start Position:** Enter exact X/Y coordinates for precise control

### SPD Matrix Validation
- **Positive Definiteness Check:** Validates that matrix A is symmetric positive definite
- **Step-Size Bounds:** Computes theoretical stability bound based on eigenvalues
- **Warnings/Errors Display:** Shows validation results in the Math Monitor panel
- **Force Run Option:** Proceed with simulation despite validation failures

### Mathematical Landscapes
- **Matyas Function:** Quadratic, convex. f(x,y) = 0.26(x² + y²) - 0.48xy
- **Himmelblau Function:** Non-convex, 4 global minima

---

## Algorithms Implemented (Python)

All algorithms are implemented in `optimizers.py` using pure NumPy:

| Algorithm | Type | Implementation Detail |
|-----------|------|-----------------------|
| **Gradient Descent** | First-Order | Fixed learning rate step |
| **Conjugate Gradient (PR)** | First-Order | Polak-Ribière update + Golden Section Line Search |
| **BFGS** | Quasi-Newton | Rank-2 Hessian update + Armijo Backtracking Line Search |

---

## API Endpoints

### POST /simulate

Run an optimization simulation.

**Request Body:**
```json
{
  "algo": "GD" | "CG_PR" | "BFGS",
  "landscape": "Matyas" | "Himmelblau",
  "start_x": 8.0,
  "start_y": 8.0,
  "lr": 0.05,
  "steps": 100,
  "validate_pd": false,
  "force_run": false,
  "matrix_a": [[2, 0], [0, 2]],
  "vector_b": [0, 0]
}
```

**Response:**
```json
{
  "path": [{"iter": 0, "x": 8.0, "y": 8.0, "z": 12.48, "grad_norm": 2.56}, ...],
  "converged": true,
  "validation": {
    "warnings": [],
    "errors": [],
    "skipped": false
  }
}
```

---

## Project Structure

```
optimizer_viz/
├── server.py              # FastAPI Backend
├── optimizers.py          # Core Math Implementation
├── landscapes.py          # Landscape Definitions
├── benchmarks.py          # Headless Benchmark Script
├── web/
│   ├── convergence_analysis.html  # Main WebGL Frontend
│   ├── index.html
│   └── ...
└── README.md
```

## Requirements

- Python 3.8+
- FastAPI, Uvicorn, NumPy
- Modern Web Browser (Chrome, Firefox, Edge)

## Usage Tips

1. **Comparing Algorithms:** Select two different algorithms to watch them race side-by-side
2. **Click-to-Set:** Click the "Click on Surface to Set Start" button, then click anywhere on the 3D landscape
3. **SPD Validation:** Expand "Advanced: SPD Validation" to test convergence bounds for quadratic forms
4. **Himmelblau Tips:** This function has 4 global minima - try different start positions to see which one the optimizer finds
