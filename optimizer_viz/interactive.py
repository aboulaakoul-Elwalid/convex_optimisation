"""
Interactive CLI Interface for Optimizer Visualization

This provides a beautiful command-line interface for experimenting
with optimizers and generating visualizations.

USAGE:
======
python -m optimizer_viz

Or after installing:
optimizer-viz
"""

import sys
import os
from typing import List, Optional, Tuple
from pathlib import Path

# Try rich for beautiful CLI, fall back to basic print
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.prompt import Prompt, IntPrompt, FloatPrompt, Confirm
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich import print as rprint

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    rprint = print

# Import our modules
from . import landscapes as land
from . import optimizers as opt
from . import visualizer as viz


def create_console():
    """Create rich console if available."""
    if RICH_AVAILABLE:
        return Console()
    return None


def print_header(console):
    """Print the welcome header."""
    header = """
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║     ██████╗ ██████╗ ████████╗██╗███╗   ███╗██╗███████╗███████╗   ║
║    ██╔═══██╗██╔══██╗╚══██╔══╝██║████╗ ████║██║╚══███╔╝██╔════╝   ║
║    ██║   ██║██████╔╝   ██║   ██║██╔████╔██║██║  ███╔╝ █████╗     ║
║    ██║   ██║██╔═══╝    ██║   ██║██║╚██╔╝██║██║ ███╔╝  ██╔══╝     ║
║    ╚██████╔╝██║        ██║   ██║██║ ╚═╝ ██║██║███████╗███████╗   ║
║     ╚═════╝ ╚═╝        ╚═╝   ╚═╝╚═╝     ╚═╝╚═╝╚══════╝╚══════╝   ║
║                                                                   ║
║              V I S U A L I Z A T I O N   T O O L                 ║
║                                                                   ║
║        Learn SGD, Momentum, Adam, and Newton/Muon by             ║
║        watching them navigate loss landscapes!                    ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
"""
    if RICH_AVAILABLE and console:
        console.print(Panel(header, style="bold cyan", border_style="cyan"))
    else:
        print(header)


def print_menu(console):
    """Print the main menu."""
    if RICH_AVAILABLE and console:
        table = Table(title="Main Menu", show_header=False, border_style="cyan")
        table.add_column("Option", style="bold yellow")
        table.add_column("Description", style="white")

        table.add_row("1", "🏁 Race Optimizers - Animated comparison on a landscape")
        table.add_row("2", "🔍 Explore Single Optimizer - Step-by-step with state")
        table.add_row("3", "📊 Adam State Evolution - Visualize m, v, and step scaling")
        table.add_row("4", "📐 Step Size Analysis - Compare effective step sizes")
        table.add_row("5", "🗻 3D Surface View - See the landscape in 3D")
        table.add_row("6", "📝 Generate Full Report - All visualizations at once")
        table.add_row("7", "🎓 Quick Tutorial - Learn the concepts")
        table.add_row("8", "⚙️  Settings - Configure defaults")
        table.add_row("q", "👋 Quit")

        console.print(table)
    else:
        print("""
    Main Menu:
    ----------
    1. Race Optimizers - Animated comparison on a landscape
    2. Explore Single Optimizer - Step-by-step with state
    3. Adam State Evolution - Visualize m, v, and step scaling
    4. Step Size Analysis - Compare effective step sizes
    5. 3D Surface View - See the landscape in 3D
    6. Generate Full Report - All visualizations at once
    7. Quick Tutorial - Learn the concepts
    8. Settings - Configure defaults
    q. Quit
""")


def select_landscape(console) -> land.Landscape:
    """Interactively select a landscape."""
    if RICH_AVAILABLE and console:
        table = Table(title="Available Landscapes", border_style="green")
        table.add_column("#", style="yellow")
        table.add_column("Name", style="bold")
        table.add_column("Description")
        table.add_column("What it teaches")

        table.add_row(
            "1", "Valley", "f(x,y) = x² + 50y²", "SGD bouncing, Adam normalizing"
        )
        table.add_row(
            "2", "Saddle Point", "f(x,y) = x² - y²", "Plateau problem, escaping saddles"
        )
        table.add_row(
            "3", "Rosenbrock", "Banana valley", "Classic optimization challenge"
        )
        table.add_row("4", "Beale", "Multi-valley surface", "Momentum overshooting")
        table.add_row(
            "5", "Rastrigin", "Many local minima", "Global vs local optimization"
        )

        console.print(table)
        choice = Prompt.ask(
            "Select landscape", choices=["1", "2", "3", "4", "5"], default="1"
        )
    else:
        print("\nLandscapes:")
        print("  1. Valley (x² + 50y²)")
        print("  2. Saddle Point (x² - y²)")
        print("  3. Rosenbrock (banana valley)")
        print("  4. Beale (multi-valley)")
        print("  5. Rastrigin (many minima)")
        choice = input("Select [1-5, default=1]: ") or "1"

    landscapes = {
        "1": land.Valley(condition=50),
        "2": land.SaddlePoint(),
        "3": land.Rosenbrock(),
        "4": land.Beale(),
        "5": land.Rastrigin(),
    }

    return landscapes.get(choice, land.Valley())


def select_optimizers(console) -> List[opt.Optimizer]:
    """Interactively select optimizers to compare."""
    if RICH_AVAILABLE and console:
        table = Table(title="Available Optimizers", border_style="blue")
        table.add_column("#", style="yellow")
        table.add_column("Name", style="bold")
        table.add_column("Key Idea")

        table.add_row("1", "SGD", "Simple gradient descent - the baseline")
        table.add_row("2", "Momentum", "SGD + velocity - smooths oscillations")
        table.add_row("3", "RMSProp", "Adaptive step size - divides by variance")
        table.add_row("4", "Adam", "Momentum + RMSProp - the gold standard")
        table.add_row("5", "Newton", "Uses Hessian - instant for quadratics")
        table.add_row("6", "Muon", "Momentum + Orthogonalization")
        table.add_row("a", "All of the above")

        console.print(table)
        choices = Prompt.ask("Select optimizers (comma-separated)", default="1,4,5")
    else:
        print("\nOptimizers:")
        print("  1. SGD")
        print("  2. Momentum")
        print("  3. RMSProp")
        print("  4. Adam")
        print("  5. Newton")
        print("  6. Muon")
        print("  a. All")
        choices = input("Select [comma-separated, default=1,4,5]: ") or "1,4,5"

    if choices.lower() == "a":
        choices = "1,2,3,4,5,6"

    optimizers = []
    optimizer_map = {
        "1": opt.SGD(lr=0.03),
        "2": opt.Momentum(lr=0.01, beta=0.9),
        "3": opt.RMSProp(lr=0.1),
        "4": opt.Adam(lr=0.5),
        "5": opt.Newton(lr=1.0, damping=0.1),
        "6": opt.Muon(lr=0.02),
    }

    for c in choices.split(","):
        c = c.strip()
        if c in optimizer_map:
            optimizers.append(optimizer_map[c])

    return optimizers if optimizers else [opt.SGD(lr=0.03), opt.Adam(lr=0.5)]


def get_start_point(console) -> Tuple[float, float]:
    """Get starting point from user."""
    if RICH_AVAILABLE and console:
        x = FloatPrompt.ask("Start X", default=-8.0)
        y = FloatPrompt.ask("Start Y", default=3.0)
    else:
        x = float(input("Start X [default=-8.0]: ") or -8.0)
        y = float(input("Start Y [default=3.0]: ") or 3.0)
    return (x, y)


def get_steps(console) -> int:
    """Get number of steps from user."""
    if RICH_AVAILABLE and console:
        return IntPrompt.ask("Number of steps", default=100)
    else:
        return int(input("Number of steps [default=100]: ") or 100)


def race_optimizers(console):
    """Option 1: Race optimizers with animation."""
    if RICH_AVAILABLE and console:
        console.print("\n[bold cyan]🏁 OPTIMIZER RACE[/bold cyan]")
        console.print("Watch optimizers navigate the loss landscape!\n")
    else:
        print("\n=== OPTIMIZER RACE ===\n")

    landscape = select_landscape(console)
    optimizers = select_optimizers(console)
    start_point = get_start_point(console)
    steps = get_steps(console)

    if RICH_AVAILABLE and console:
        save = Confirm.ask("Save animation as GIF?", default=True)
        save_path = (
            Prompt.ask("Save path", default="optimizer_race.gif") if save else None
        )
    else:
        save = input("Save as GIF? [Y/n]: ").lower() != "n"
        save_path = (
            input("Save path [optimizer_race.gif]: ") or "optimizer_race.gif"
            if save
            else None
        )

    print("\nGenerating animation...")
    import matplotlib

    matplotlib.use("TkAgg")  # Use interactive backend
    import matplotlib.pyplot as plt

    anim = viz.animate_optimization(
        landscape, optimizers, start_point, steps, save_path=save_path
    )

    plt.show()


def explore_single_optimizer(console):
    """Option 2: Step through a single optimizer."""
    if RICH_AVAILABLE and console:
        console.print("\n[bold cyan]🔍 SINGLE OPTIMIZER EXPLORATION[/bold cyan]")
    else:
        print("\n=== SINGLE OPTIMIZER EXPLORATION ===\n")

    landscape = select_landscape(console)

    # Select single optimizer
    if RICH_AVAILABLE and console:
        choice = Prompt.ask(
            "Select optimizer",
            choices=["sgd", "momentum", "adam", "newton", "muon"],
            default="adam",
        )
    else:
        choice = (
            input("Optimizer [sgd/momentum/adam/newton/muon, default=adam]: ") or "adam"
        )

    optimizer = opt.get_optimizer(choice)
    start_point = get_start_point(console)
    steps = get_steps(console)

    # Run optimization with verbose output
    print(f"\nOptimizing with {optimizer.name}...")
    history = optimizer.optimize(
        landscape, start_point[0], start_point[1], steps, verbose=True
    )

    # Show final results
    final = history[-1]
    if RICH_AVAILABLE and console:
        console.print(f"\n[green]✓ Optimization complete![/green]")
        console.print(f"  Final position: ({final.x:.6f}, {final.y:.6f})")
        console.print(f"  Final loss: {final.loss:.8f}")
        console.print(f"  Optimal: {landscape.optimal_point}")
    else:
        print(f"\nOptimization complete!")
        print(f"  Final position: ({final.x:.6f}, {final.y:.6f})")
        print(f"  Final loss: {final.loss:.8f}")

    # Show plot
    import matplotlib.pyplot as plt

    viz.plot_optimization_paths(landscape, [optimizer], start_point, steps)
    plt.show()


def adam_state_evolution(console):
    """Option 3: Visualize Adam's internal state."""
    if RICH_AVAILABLE and console:
        console.print("\n[bold cyan]📊 ADAM STATE EVOLUTION[/bold cyan]")
        console.print("See how m (momentum) and v (variance) change over time.\n")
        console.print(
            "[yellow]This answers your question: 'What is variance in Adam?'[/yellow]\n"
        )
    else:
        print("\n=== ADAM STATE EVOLUTION ===\n")

    landscape = select_landscape(console)
    start_point = get_start_point(console)
    steps = get_steps(console)

    if RICH_AVAILABLE and console:
        lr = FloatPrompt.ask("Adam learning rate", default=0.5)
    else:
        lr = float(input("Adam learning rate [default=0.5]: ") or 0.5)

    print("\nGenerating state evolution plot...")
    import matplotlib.pyplot as plt

    viz.plot_adam_state_evolution(
        landscape, start_point, steps, lr, save_path="adam_state.png"
    )
    plt.show()


def step_size_analysis(console):
    """Option 4: Compare step sizes."""
    if RICH_AVAILABLE and console:
        console.print("\n[bold cyan]📐 STEP SIZE ANALYSIS[/bold cyan]")
    else:
        print("\n=== STEP SIZE ANALYSIS ===\n")

    landscape = select_landscape(console)
    optimizers = select_optimizers(console)
    start_point = get_start_point(console)
    steps = get_steps(console)

    print("\nGenerating step size analysis...")
    import matplotlib.pyplot as plt

    viz.plot_step_size_comparison(
        landscape, optimizers, start_point, steps, save_path="step_sizes.png"
    )
    plt.show()


def view_3d_surface(console):
    """Option 5: 3D surface view."""
    if RICH_AVAILABLE and console:
        console.print("\n[bold cyan]🗻 3D SURFACE VIEW[/bold cyan]")
    else:
        print("\n=== 3D SURFACE VIEW ===\n")

    landscape = select_landscape(console)

    if RICH_AVAILABLE and console:
        add_paths = Confirm.ask("Add optimizer paths?", default=True)
    else:
        add_paths = input("Add optimizer paths? [Y/n]: ").lower() != "n"

    optimizers = select_optimizers(console) if add_paths else None
    start_point = get_start_point(console) if add_paths else (-8, 3)
    steps = get_steps(console) if add_paths else 100

    print("\nGenerating 3D surface...")
    import matplotlib.pyplot as plt

    viz.plot_3d_surface(
        landscape, optimizers, start_point, steps, save_path="surface_3d.png"
    )
    plt.show()


def generate_report(console):
    """Option 6: Generate full report."""
    if RICH_AVAILABLE and console:
        console.print("\n[bold cyan]📝 GENERATE FULL REPORT[/bold cyan]")
        console.print("This will create all visualizations in one go.\n")
    else:
        print("\n=== GENERATE FULL REPORT ===\n")

    landscape = select_landscape(console)
    optimizers = select_optimizers(console)
    start_point = get_start_point(console)
    steps = get_steps(console)

    if RICH_AVAILABLE and console:
        save_dir = Prompt.ask("Save directory", default="./optimizer_viz_report")
    else:
        save_dir = (
            input("Save directory [./optimizer_viz_report]: ")
            or "./optimizer_viz_report"
        )

    Path(save_dir).mkdir(exist_ok=True)

    viz.create_learning_report(landscape, optimizers, start_point, steps, save_dir)

    if RICH_AVAILABLE and console:
        console.print(f"\n[green]✓ Report generated in {save_dir}/[/green]")


def print_tutorial(console):
    """Option 7: Quick tutorial."""
    tutorial = """
╔══════════════════════════════════════════════════════════════════════════╗
║                         OPTIMIZER TUTORIAL                                ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  WHY DO WE NEED DIFFERENT OPTIMIZERS?                                    ║
║  ════════════════════════════════════                                    ║
║                                                                          ║
║  The problem: Gradient Descent (SGD) follows the steepest slope.         ║
║  But in high dimensions, this is often NOT the best direction!           ║
║                                                                          ║
║  Imagine a narrow valley (like a taco shell):                            ║
║  - The walls are steep (large gradient)                                  ║
║  - The floor is flat (small gradient)                                    ║
║  - SGD bounces between walls instead of walking down the floor!          ║
║                                                                          ║
║  OPTIMIZER COMPARISON:                                                   ║
║  ════════════════════                                                    ║
║                                                                          ║
║  SGD:        step = lr * gradient                                        ║
║              Problem: Step size varies wildly with gradient magnitude    ║
║                                                                          ║
║  Momentum:   v = 0.9 * v + gradient                                      ║
║              step = lr * v                                               ║
║              Better: Smooths oscillations, builds speed on flat regions  ║
║                                                                          ║
║  Adam:       m = 0.9 * m + gradient           (smoothed gradient)        ║
║              v = 0.999 * v + gradient²        (variance estimate)        ║
║              step = lr * m / sqrt(v)          (NORMALIZED step!)         ║
║              Best: Adapts to the "loudness" of each gradient!            ║
║                                                                          ║
║  Newton:     step = H⁻¹ @ gradient            (uses curvature)           ║
║              Perfect: Solves quadratics in ONE step!                     ║
║              Problem: Computing H⁻¹ is expensive for neural nets         ║
║                                                                          ║
║  THE KEY INSIGHT (from your learning):                                   ║
║  ═════════════════════════════════════                                   ║
║                                                                          ║
║  Adam divides by sqrt(v) - the "variance" of recent gradients.           ║
║                                                                          ║
║  - If gradient is LARGE but variance is LARGE → normal step              ║
║    (This is just noise from steep walls)                                 ║
║                                                                          ║
║  - If gradient is SMALL but variance is SMALL → LARGE step!              ║
║    (This is a consistent signal - sprint across the plateau!)            ║
║                                                                          ║
║  CONNECTION TO DISTRIBUTED TRAINING (NCCL):                              ║
║  ═══════════════════════════════════════════                             ║
║                                                                          ║
║  In DDP, each GPU computes gradients on different data.                  ║
║  NCCL's all_reduce AVERAGES these gradients.                             ║
║                                                                          ║
║  The gradient matrix you see here is exactly what gets transported!      ║
║  When NCCL is slow, GPUs wait for the gradient average to arrive.        ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
"""
    if RICH_AVAILABLE and console:
        console.print(Panel(tutorial, title="Tutorial", border_style="cyan"))
    else:
        print(tutorial)

    input("\nPress Enter to continue...")


def main():
    """Main entry point."""
    console = create_console()

    while True:
        print_header(console)
        print_menu(console)

        if RICH_AVAILABLE and console:
            choice = Prompt.ask("Select option", default="1")
        else:
            choice = input("\nSelect option [1-8, q]: ") or "1"

        choice = choice.lower().strip()

        if choice == "q" or choice == "quit":
            if RICH_AVAILABLE and console:
                console.print("\n[cyan]Thanks for learning! 👋[/cyan]")
            else:
                print("\nThanks for learning!")
            break
        elif choice == "1":
            race_optimizers(console)
        elif choice == "2":
            explore_single_optimizer(console)
        elif choice == "3":
            adam_state_evolution(console)
        elif choice == "4":
            step_size_analysis(console)
        elif choice == "5":
            view_3d_surface(console)
        elif choice == "6":
            generate_report(console)
        elif choice == "7":
            print_tutorial(console)
        elif choice == "8":
            if RICH_AVAILABLE and console:
                console.print("[yellow]Settings coming soon![/yellow]")
            else:
                print("Settings coming soon!")
        else:
            if RICH_AVAILABLE and console:
                console.print(f"[red]Unknown option: {choice}[/red]")
            else:
                print(f"Unknown option: {choice}")

        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()
