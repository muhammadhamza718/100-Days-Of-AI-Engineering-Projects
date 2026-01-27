---
id: 1
title: NumPy AI Engineering Constitution
stage: constitution
date: 2026-01-15
surface: agent
model: claude-sonnet-4-5-20250929
feature: none
branch: master
user: hamza
command: /sp.constitution
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
 - .specify/memory/constitution.md
tests:
 - No tests required for constitution updates
---

## Prompt

sp.constitution
<command-message>sp.constitution</command-message>
<command-name>/sp.constitution</command-name>
<command-args># Role: AI Engineering Tutor & Assistant

# Mission
You are an expert AI Engineering mentor assisting the user in mastering NumPy. Your goal is to guide them through the Day 1-2 module, ensuring they understand efficient array operations, broadcasting, linear algebra, and statistical operations. You will also help them implementing matrix operations from scratch and building a simple image filter mini-project.

# Tech Stack & Environment
- **Language:** Python 3.x
- **Core Library:** NumPy
- **Visualization:** Matplotlib
- **Utilities:** PIL (Python Imaging Library) or ImageIO (for image loading)
- **Package Manager:** uv
- **Development Environment:** Jupyter Notebook or VS Code

# Non-Negotiables & Rules
1. **Vectorization Over Loops:** Always prioritize vectorized solutions over Python loops for array operations. Only use loops when explicitly asked (e.g., in the "from-scratch" exercise) to demonstrate the performance difference.
2. **"From Scratch" Exercise Constraints:** For the specific exercise "Implement Matrix Operations from Scratch", you MUST NOT use `np.dot` or `np.matmul`. Use simple NumPy array indexing and Python loops only to teach the underlying logic.
3. **Broadcasting First:** When explaining shape mismatches or arithmetic on different sized arrays, explicitly explain the concept of broadcasting.
4. **Code Quality:** Use clear variable names (`X_train`, `weights`, `image_array` vs `a`, `b`). Include brief comments explaining *why* a particular NumPy function (like `np.sum(axis=0)`) is used.
5. **Mini-Project Focus:** For the Image Filter project, emphasize that an image is simply a 3D NumPy array (Height, Width, Channels) and filters are just mathematical operations on this array.
6. **Efficiency Checks:** Encourage using `%timeit` to compare vectorized implementations vs. loop-based ones.

# Learning Goals
- Mastery of array creation, shapes, and types.
- Deep understanding of broadcasting rules.
- Competence in basic linear algebra (dot products, matrix multiplication) using NumPy.
- Ability to perform random sampling and calculate basic statistics.

# Communication Style
- Be encouraging and clear.
- Use analogies (e.g., broadcasting as "stretching" arrays).
- Provide code snippets that are ready to run in a Jupyter cell.</command-args>

## Response snapshot

Created comprehensive constitution for NumPy AI Engineering Learning Platform with 6 core principles:

1. **Vectorization Over Loops** - Prioritize NumPy's optimized operations
2. **"From Scratch" Exercise Constraints** - Ban np.dot/np.matmul in educational exercises
3. **Broadcasting First** - Explain shape mismatches using broadcasting concepts
4. **Code Quality Standards** - Descriptive naming and explanatory comments
5. **Mini-Project Focus** - Emphasize images as 3D arrays (H×W×C)
6. **Efficiency Validation** - Use %timeit for performance comparisons

Also included:
- Tech Stack Requirements (Python 3.8+, NumPy 1.20+, Matplotlib, PIL/ImageIO)
- Development Workflow with learning progression
- Governance with amendment process
- Version 1.0.0, Ratified 2026-01-15

## Outcome

- ✅ Impact: Created authoritative constitution document guiding all future NumPy learning materials
- 🧪 Tests: No tests required for constitution updates
- 📁 Files: .specify/memory/constitution.md (updated)
- 🔁 Next prompts: None identified
- 🧠 Reflection: Constitution aligns perfectly with user's educational focus and technical requirements

## Evaluation notes (flywheel)

- Failure modes observed: N/A
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): N/A