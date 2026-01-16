---
id: 4
title: NumPy Mastery Task Breakdown
stage: tasks
date: 2026-01-15
surface: agent
model: claude-sonnet-4-5-20250929
feature: numpy-mastery
branch: 001-numpy-mastery
user: hamza
command: /sp.tasks
labels: ["numpy", "tasks", "breakdown", "implementation"]
links:
  spec: specs/numpy-mastery/spec.md
  plan: specs/numpy-mastery/plan.md
  tasks: specs/numpy-mastery/tasks.md
  adr: null
  pr: null
files:
 - specs/numpy-mastery/tasks.md
tests:
 - 30 granular tasks created across 5 phases
 - All tasks are independently testable
 - Constitutional compliance verified
 - Estimated 3-4 hour completion time
---

## Prompt

sp.tasks
# Task Breakdown: NumPy Mastery Module

Based on the **Implementation Plan**, **Specifications**, and **Constitution**, please break down the project into granular, actionable tasks.

**Task Categories:**

1.  **Environment & Setup**
    *   Initialize project with `uv`.
    *   Create virtual environment and install `numpy`, `matplotlib`, `pillow`.

2.  **Core Concepts (Notebook/Script)**
    *   Create `basics.ipynb`.
    *   Implement Array Operations examples.
    *   Implement Vectorization demo with `%timeit`.
    *   Implement Broadcasting examples.
    *   Implement Basic Algebra (Dot product, Matrix mult).

3.  **Exercise: From-Scratch Implementation**
    *   Create `exercises.py`.
    *   Implement `custom_dot(a, b)` using loops.
    *   Implement `custom_matmul(A, B)` using loops.
    *   Implement `custom_transpose(A)` using loops.
    *   Add comparison logic to verify against NumPy.

4.  **Mini-Project: Image Filters**
    *   Create `image_filters.ipynb`.
    *   Implement image loading with `PIL`/`ImageIO`.
    *   Implement Grayscale conversion (Vectorized).
    *   Implement Inversion filter (Vectorized).
    *   Display Side-by-Side comparison (Original vs Processed).

**Output Constraint:**
Ensure each task is small enough to be completed in a single code editing step.

## Response snapshot

Created comprehensive task breakdown with 30 granular, actionable tasks organized into 5 phases:

**Phase 1: Environment & Setup (4 tasks)**
- Task 1.1: Create project directory structure
- Task 1.2: Initialize virtual environment with uv
- Task 1.3: Install core dependencies
- Task 1.4: Create environment verification script

**Phase 2: Core Concepts Notebook (6 tasks)**
- Task 2.1: Create main Jupyter notebook structure
- Task 2.2: Add array creation and operations examples
- Task 2.3: Implement vectorization performance comparison
- Task 2.4: Add broadcasting examples with explanation
- Task 2.5: Add linear algebra operations examples
- Task 2.6: Add random sampling and statistics examples

**Phase 3: From-Scratch Exercises (7 tasks)**
- Task 3.1: Create exercises.py file structure
- Task 3.2: Implement custom_dot() function
- Task 3.3: Implement custom_matmul() function
- Task 3.4: Implement custom_transpose() function
- Task 3.5: Create validation functions
- Task 3.6: Create performance comparison function
- Task 3.7: Add main execution block

**Phase 4: Image Processing Mini-Project (9 tasks)**
- Task 4.1: Create image_filters.py file structure
- Task 4.2: Implement image loading method
- Task 4.3: Implement grayscale conversion method
- Task 4.4: Implement color inversion method
- Task 4.5: Create test image generation function
- Task 4.6: Create display comparison method
- Task 4.7: Create save image method
- Task 4.8: Create demonstration function
- Task 4.9: Add main execution block

**Phase 5: Validation & Testing (4 tasks)**
- Task 5.1: Create unit tests for matrix operations
- Task 5.2: Create unit tests for image filters
- Task 5.3: Create performance validation script
- Task 5.4: Create complete test runner

**Key Features:**
- Each task is designed for single code editing step completion
- Clear acceptance criteria for every task
- Dependencies mapped between tasks
- Estimated 3-4 hour total completion time
- Full constitutional compliance verified
- Independent testability for all components

## Outcome

- ✅ Impact: Created 30 granular, testable tasks ready for implementation
- 🧪 Tests: Each task has clear acceptance criteria and testing approach
- 📁 Files: specs/numpy-mastery/tasks.md (comprehensive 800+ line task breakdown)
- 🔁 Next prompts: Ready for implementation - tasks can be executed individually
- 🧠 Reflection: Tasks balance completeness with granularity - each can be completed in 5-20 minutes

## Evaluation notes (flywheel)

- Failure modes observed: N/A (all tasks clearly defined)
- Graders run and results (PASS/FAIL): All tasks pass acceptance criteria definition
- Prompt variant (if applicable): Standard task breakdown with educational focus
- Next experiment: Implementation can begin - recommend starting with Phase 1 (Environment)

**Implementation Ready**: All tasks are actionable, testable, and follow constitutional requirements.

---

**Phase 1 (Environment) is blocking for Phases 2-4. Phase 5 depends on Phases 2-4.**
**Recommended order**: 1.1 → 1.2 → 1.3 → 1.4 → [Parallel: 2.x, 3.x, 4.x] → 5.x