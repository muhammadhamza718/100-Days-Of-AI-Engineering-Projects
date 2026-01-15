---
id: 2
title: NumPy Mastery Specification
stage: spec
date: 2026-01-15
surface: agent
model: claude-sonnet-4-5-20250929
feature: numpy-mastery
branch: 001-numpy-mastery
user: hamza
command: /sp.specify
labels: ["numpy", "learning-module", "spec-driven"]
links:
  spec: specs/numpy-mastery/spec.md
  ticket: null
  adr: null
  pr: null
files:
 - specs/numpy-mastery/spec.md
 - specs/numpy-mastery/checklists/requirements.md
tests:
 - Specification quality validation completed
 - All checklist items passed
 - No clarification markers remaining
---

## Prompt

sp.specify
<command-message>sp.specify</command-message>
<command-name>/sp.specify</command-name>
<command-args>
# Project Specifications: Day 1-2 NumPy Mastery

# Overview
We are building a learning module for NumPy that includes educational code examples, a "from-scratch" implementation exercise, and a practical image processing mini-project.

# Functional Requirements

## 1. Core NumPy Concepts
Implement clear, executable examples for:
- **Array Operations**: Creation, indexing, slicing, and shapes.
- **Vectorization**: Demonstrate performance gains by rewriting a "Sum of Squares" loop using `np.sum()`.
- **Broadcasting**: Show arithmetic between arrays of different shapes (e.g., (2,3) + (3,)).
- **Linear Algebra**: Implement Dot Product (`dot`), Matrix Multiplication (`@`), Inverse, and Eigenvalues.
- **Statistics**: Random sampling (uniform, normal, randint) and stats (mean, std, max, argmax).

## 2. Exercise: Matrix Operations from Scratch
Create a Python script that implements the following *without* NumPy's linear algebra functions:
- `custom_matmul(A, B)`: Matrix multiplication using loops.
- `custom_dot(a, b)`: Dot product using loops.
- `custom_transpose(A)`: Matrix transpose using loops.
*Constraint:* Comparison with NumPy equivalents using `timeit` is required.

## 3. Mini-Project: Image Filters
Create a Jupyter Notebook or Script that:
- Loads an image using `PIL` or `ImageIO` into a NumPy array.
- Implements **Grayscale Conversion**: $Y = 0.2989R + 0.5870G + 0.1140B$.
- Implements **Color Inversion**: $255 - pixel\_value$.
- (Optional) Implements a Box Blur filter.

# Technical Stack & Environment
- **Language**: Python 3.x
- **Core Library**: NumPy
- **Visualization**: Matplotlib
- **Utilities**: PIL (Python Imaging Library) or ImageIO
- **Package Manager**: uv
- **Environment**: Jupyter Notebook or VS Code

make this specs in this directory not outside of the Day 1-2 folder</command-args>

## Response snapshot

Created comprehensive specification for NumPy Mastery Learning Module with:

**User Scenarios** (3 prioritized stories):
- P1: Learn Core NumPy Array Operations (foundational)
- P2: Implement Matrix Operations from Scratch (deep understanding)
- P3: Build Image Processing Mini-Project (applied learning)

**Requirements**:
- 13 functional requirements covering all learning objectives
- 3 key entities (NumPy Array, Image Array, Performance Metric)
- 4 edge cases identified (dimension mismatches, channel ordering, etc.)

**Success Criteria**:
- 8 measurable outcomes including 10-100x performance improvements
- 4-6 hour completion time target
- Compatibility with Jupyter Notebook and VS Code

**Validation**:
- All checklist items passed ✅
- No [NEEDS CLARIFICATION] markers
- Technology-agnostic success criteria
- Clear scope boundaries

## Outcome

- ✅ Impact: Created specification ready for architectural planning phase
- 🧪 Tests: Specification quality validation completed - all items pass
- 📁 Files: specs/numpy-mastery/spec.md, specs/numpy-mastery/checklists/requirements.md
- 🔁 Next prompts: Ready for /sp.plan to begin architectural design
- 🧠 Reflection: Specification maintains appropriate abstraction level while being specific enough to guide implementation

## Evaluation notes (flywheel)

- Failure modes observed: N/A (all requirements clear)
- Graders run and results (PASS/FAIL): All checklist items PASS
- Prompt variant (if applicable): Standard specification creation
- Next experiment: Proceed to /sp.plan for architectural planning