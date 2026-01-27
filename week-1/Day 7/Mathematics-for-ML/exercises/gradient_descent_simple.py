"""
Exercise: Basic Gradient Descent Implementation

This exercise implements gradient descent to minimize a simple quadratic function.
f(x,y) = x² + y²
"""

import numpy as np
import matplotlib.pyplot as plt
from optimization.gradient_descent import (
    gradient_descent,
    quadratic_function,
    quadratic_gradient,
    visualize_convergence,
    visualize_2d_path
)

def main():
    # Test function: f(x,y) = x² + y²
    # Starting point: [5, 5]
    start_point = np.array([5.0, 5.0])

    # Different learning rates to compare
    learning_rates = [0.01, 0.1, 0.5, 0.9]
    num_iterations = 50

    print("Testing gradient descent on f(x,y) = x² + y²")
    print(f"Starting point: {start_point}")
    print(f"Expected minimum: [0, 0]")
    print()

    all_paths = {}
    all_values = {}

    for lr in learning_rates:
        print(f"Learning rate: {lr}")

        # Run gradient descent
        path, values = gradient_descent(
            objective_fn=quadratic_function,
            gradient_fn=quadratic_gradient,
            start_point=start_point,
            learning_rate=lr,
            num_iterations=num_iterations
        )

        all_paths[lr] = path
        all_values[lr] = values

        # Final point and value
        final_point = path[-1]
        final_value = values[-1]

        print(f"  Final point: [{final_point[0]:.4f}, {final_point[1]:.4f}]")
        print(f"  Final value: {final_value:.6f}")
        print(f"  Distance from origin: {np.linalg.norm(final_point):.6f}")
        print()

    # Create visualizations
    plt.figure(figsize=(15, 10))

    # Subplot 1: Convergence curves for different learning rates
    plt.subplot(2, 2, 1)
    for lr in learning_rates:
        plt.plot(all_values[lr], label=f'LR={lr}', marker='o', markersize=3)
    plt.title('Convergence Curves: Function Value vs Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Function Value')
    plt.legend()
    plt.grid(True)

    # Subplot 2: Path for learning rate 0.1 (most stable)
    plt.subplot(2, 2, 2)
    if 0.1 in all_paths:
        path_01 = all_paths[0.1]
        x_coords = [point[0] for point in path_01]
        y_coords = [point[1] for point in path_01]

        plt.plot(x_coords, y_coords, 'ro-', markersize=4, linewidth=2, label='Optimization Path')
        plt.plot(x_coords[0], y_coords[0], 'go', markersize=8, label='Start')
        plt.plot(x_coords[-1], y_coords[-1], 'r*', markersize=12, label='End')
        plt.title(f'Optimization Path (LR=0.1)')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)

    # Subplot 3: Contour plot with path for LR=0.1
    plt.subplot(2, 2, 3)
    if 0.1 in all_paths:
        x = np.linspace(-6, 6, 100)
        y = np.linspace(-6, 6, 100)
        X, Y = np.meshgrid(x, y)
        Z = X**2 + Y**2

        contour = plt.contour(X, Y, Z, levels=20)
        plt.clabel(contour, inline=True, fontsize=8)

        path_01 = all_paths[0.1]
        x_coords = [point[0] for point in path_01]
        y_coords = [point[1] for point in path_01]

        plt.plot(x_coords, y_coords, 'ro-', markersize=4, linewidth=2, label='Path')
        plt.plot(x_coords[0], y_coords[0], 'go', markersize=8, label='Start [5,5]')
        plt.plot(x_coords[-1], y_coords[-1], 'r*', markersize=12, label='End')
        plt.title('Contour Plot with Optimization Path (LR=0.1)')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)

    # Subplot 4: Comparison of final distances from origin
    plt.subplot(2, 2, 4)
    final_distances = [np.linalg.norm(all_paths[lr][-1]) for lr in learning_rates]
    plt.bar([str(lr) for lr in learning_rates], final_distances)
    plt.title('Final Distance from Origin by Learning Rate')
    plt.xlabel('Learning Rate')
    plt.ylabel('Distance from Origin')
    plt.grid(True, axis='y')

    plt.tight_layout()
    plt.savefig('outputs/plots/quadratic_optimization_analysis.png')
    plt.show()

if __name__ == "__main__":
    main()