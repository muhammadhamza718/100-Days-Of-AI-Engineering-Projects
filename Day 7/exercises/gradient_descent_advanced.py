"""
Exercise: Advanced Gradient Descent Implementation

This exercise implements gradient descent to minimize the Rosenbrock function.
f(x,y) = (1-x)² + 100(y-x²)²
The global minimum is at (1,1) where f(1,1) = 0.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from optimization.gradient_descent import (
    gradient_descent,
    rosenbrock_function,
    rosenbrock_gradient
)

def visualize_rosenbrock_3d(path: list, save_path: str = None):
    """
    Create a 3D visualization of the Rosenbrock function and optimization path.

    Args:
        path: List of parameter vectors representing the optimization path
        save_path: Path to save the plot (optional)
    """
    # Extract x and y coordinates from the path
    x_coords = [point[0] for point in path]
    y_coords = [point[1] for point in path]
    z_coords = [rosenbrock_function(point) for point in path]

    # Create a grid for the surface
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-1, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = rosenbrock_function(np.array([X[i, j], Y[i, j]]))

    fig = plt.figure(figsize=(14, 6))

    # 3D surface plot
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)
    ax1.plot(x_coords, y_coords, z_coords, 'ro-', markersize=5, linewidth=2, label='Optimization Path')
    ax1.scatter([x_coords[0]], [y_coords[0]], [z_coords[0]], color='green', s=100, label='Start')
    ax1.scatter([x_coords[-1]], [y_coords[-1]], [z_coords[-1]], color='red', s=150, marker='*', label='End')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('f(x,y)')
    ax1.set_title('3D Trajectory on Rosenbrock Function')
    ax1.legend()

    # 2D contour with path
    ax2 = fig.add_subplot(1, 2, 2)
    contour = ax2.contour(X, Y, Z, levels=20)
    ax2.clabel(contour, inline=True, fontsize=8)
    ax2.plot(x_coords, y_coords, 'ro-', markersize=5, linewidth=2, label='Optimization Path')
    ax2.plot(x_coords[0], y_coords[0], 'go', markersize=10, label='Start')
    ax2.plot(x_coords[-1], y_coords[-1], 'r*', markersize=15, label='End')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Optimization Path on Contour Plot')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def main():
    # Test function: Rosenbrock function
    # f(x,y) = (1-x)² + 100(y-x²)²
    # Global minimum at (1,1) where f(1,1) = 0
    start_point = np.array([-1.0, 1.0])  # Common starting point for Rosenbrock

    # Different learning rates to compare
    learning_rates = [0.001, 0.005, 0.01, 0.05]
    num_iterations = 1000

    print("Testing gradient descent on Rosenbrock function:")
    print("f(x,y) = (1-x)² + 100(y-x²)²")
    print(f"Starting point: {start_point}")
    print(f"Expected minimum: [1, 1] with value 0")
    print()

    all_paths = {}
    all_values = {}

    for lr in learning_rates:
        print(f"Learning rate: {lr}")

        # Run gradient descent
        path, values = gradient_descent(
            objective_fn=rosenbrock_function,
            gradient_fn=rosenbrock_gradient,
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
        print(f"  Distance from minimum [1,1]: {np.linalg.norm(final_point - np.array([1, 1])):.6f}")
        print()

    # Create visualizations for the most successful learning rate
    best_lr = 0.005  # Usually works well for Rosenbrock
    if best_lr in all_paths:
        # Save the 3D trajectory
        visualize_rosenbrock_3d(all_paths[best_lr], 'outputs/plots/rosenbrock_3d_trajectory.png')

        # Plot convergence curve
        plt.figure(figsize=(10, 6))
        plt.plot(all_values[best_lr], marker='o')
        plt.title(f'Convergence Curve for Rosenbrock Function (LR={best_lr})')
        plt.xlabel('Iteration')
        plt.ylabel('Function Value')
        plt.grid(True)
        plt.yscale('log')  # Log scale due to wide range of values
        plt.savefig('outputs/plots/rosenbrock_convergence.png')
        plt.show()

        # Plot the path in 2D
        path_best = all_paths[best_lr]
        x_coords = [point[0] for point in path_best]
        y_coords = [point[1] for point in path_best]

        plt.figure(figsize=(10, 8))

        # Contour plot
        x = np.linspace(-1.5, 1.5, 100)
        y = np.linspace(-0.5, 1.5, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = rosenbrock_function(np.array([X[i, j], Y[i, j]]))

        contour = plt.contour(X, Y, Z, levels=20)
        plt.clabel(contour, inline=True, fontsize=8)
        plt.plot(x_coords, y_coords, 'ro-', markersize=4, linewidth=2, label='Optimization Path')
        plt.plot(x_coords[0], y_coords[0], 'go', markersize=8, label='Start [-1,1]')
        plt.plot(x_coords[-1], y_coords[-1], 'r*', markersize=12, label='End')
        plt.plot(1, 1, 'k*', markersize=15, label='Global Min [1,1]')
        plt.title(f'Rosenbrock Function Optimization Path (LR={best_lr})')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)
        plt.savefig('outputs/plots/rosenbrock_optimization_path.png')
        plt.show()

    # Compare all learning rates
    plt.figure(figsize=(12, 8))

    # Subplot 1: Convergence comparison
    plt.subplot(2, 2, 1)
    for lr in learning_rates:
        plt.plot(all_values[lr][:200], label=f'LR={lr}', linewidth=2)  # First 200 iterations
    plt.title('Convergence Comparison (First 200 Iterations)')
    plt.xlabel('Iteration')
    plt.ylabel('Function Value')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')

    # Subplot 2: Final values comparison
    plt.subplot(2, 2, 2)
    final_values = [all_values[lr][-1] for lr in learning_rates]
    plt.bar([str(lr) for lr in learning_rates], final_values)
    plt.title('Final Function Values by Learning Rate')
    plt.xlabel('Learning Rate')
    plt.ylabel('Final Function Value')
    plt.yscale('log')
    plt.grid(True, axis='y')

    # Subplot 3: Distance from optimum
    plt.subplot(2, 2, 3)
    final_distances = [np.linalg.norm(all_paths[lr][-1] - np.array([1, 1])) for lr in learning_rates]
    plt.bar([str(lr) for lr in learning_rates], final_distances)
    plt.title('Distance from Optimum [1,1] by Learning Rate')
    plt.xlabel('Learning Rate')
    plt.ylabel('Distance from [1,1]')
    plt.grid(True, axis='y')

    # Subplot 4: Path lengths
    plt.subplot(2, 2, 4)
    path_lengths = []
    for lr in learning_rates:
        path = all_paths[lr]
        length = 0
        for i in range(1, len(path)):
            length += np.linalg.norm(path[i] - path[i-1])
        path_lengths.append(length)

    plt.bar([str(lr) for lr in learning_rates], path_lengths)
    plt.title('Total Path Length by Learning Rate')
    plt.xlabel('Learning Rate')
    plt.ylabel('Total Path Length')
    plt.grid(True, axis='y')

    plt.tight_layout()
    plt.savefig('outputs/plots/rosenbrock_learning_rate_comparison.png')
    plt.show()


if __name__ == "__main__":
    main()