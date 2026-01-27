---
id: 001
title: "Create Project Constitution"
stage: "constitution"
date_iso: "2026-01-21"
surface: "agent"
model: "claude-sonnet-4-5-20250929"
feature: "none"
branch: "main"
user: "user"
command: "/sp.constitution"
labels: ["constitution", "setup", "mathematics", "ml"]
files_yaml:
  - ".specify/memory/constitution.md"
tests_yaml: []
links:
  spec: null
  ticket: null
  adr: null
  pr: null
---

# Create Project Constitution

## Prompt Text

```
/sp.constitution

# Day 7: Mathematics for Machine Learning

## 🎯 Goal
Implement mathematical foundations for ML as defined in @[Day 7/Mathematics_for_ML.md]: linear algebra, calculus, probability, and gradient descent from scratch.

## 📂 Project Setup
- **Work Dir:** `Day 7\Mathematics-for-ML\`
- **Init:** `uv init` and `uv add numpy matplotlib scipy pytest`
- **Structure:**
  - `linear_algebra/` - vectors, matrices, eigenvalues
  - `calculus/` - derivatives, gradients
  - `probability/` - distributions, statistics
  - `optimization/` - gradient descent (CORE)
  - `exercises/` - GD implementations

## ⚖️ Non-Negotiables
1. **From Scratch:** Implement ALL algorithms manually (NO scipy.optimize, NO np.linalg.eig for exercises)
2. **No Notebooks:** Python scripts only ([.py](cci:7://file:///f:/Courses/Hamza/100-Days-Of-AI-Engineering-Projects/Day%201-2/src/__init__.py:0:0-0:0) files)
3. **Use NumPy:** Only for arrays/basic math, NOT for the algorithms you're implementing
4. **Testing:** pytest with 80%+ coverage

## 📜 Mandatory Deliverables
1. **Gradient Descent Algorithm:**
   - Function signature: `gradient_descent(objective_fn, gradient_fn, start_point, learning_rate, num_iterations)`
   - Test on f(x,y) = x² + y², starting at [5, 5]
   - Compare learning rates: [0.01, 0.1, 0.5, 0.9]

2. **Exercises:**
   - `exercises/gradient_descent_simple.py` - Basic quadratic minimization
   - `exercises/gradient_descent_advanced.py` - Rosenbrock function

3. **Visualizations (save to outputs/plots/):**
   - Convergence curves (function value vs iterations)
   - Optimization paths on contour plots
   - 3D trajectories

4. **Analysis:** `optimization_analysis.md` with learning rate effects and convergence patterns

## ✅ Success Criteria
- GD converges to [0,0] for quadratic function
- All implementations have docstrings with math formulas
- Tests pass with 80%+ coverage
- At least 5 plots generated
```

## Response Text

Created a comprehensive project constitution for the Mathematics for Machine Learning project that emphasizes mathematical rigor, educational clarity, from-scratch implementation, proper use of NumPy for foundation only, comprehensive testing, and visualization capabilities. The constitution establishes clear principles and governance for the project.

## Outcome

Successfully created `.specify/memory/constitution.md` with 6 core principles tailored to the mathematical ML project requirements, including technology stack constraints, performance standards, and development workflow requirements.

## Evaluation

The constitution addresses all key requirements from the project specification:
- Emphasizes mathematical correctness and educational value
- Enforces from-scratch implementation while allowing NumPy for basic operations
- Mandates comprehensive testing with 80%+ coverage
- Requires visualization capabilities
- Establishes clear governance and compliance review processes