---
id: 008
title: Enforce Directory Structure
stage: tasks
date: 2026-01-18
surface: agent
model: claude-sonnet-4-5-20250929
feature: pandas-mastery
branch: 001-pandas-mastery
user: user
command: /sp.tasks
labels: [tasks, directory-structure, organization]
links:
  spec: F:/Courses/Hamza/100-Days-Of-AI-Engineering-Projects/Day 3-4/specs/001-pandas-mastery/spec.md
  tasks: F:/Courses/Hamza/100-Days-Of-AI-Engineering-Projects/Day 3-4/specs/001-pandas-mastery/tasks.md
  ticket: null
  adr: null
  pr: null
files:
  - specs/001-pandas-mastery/tasks.md
tests:
  - none
---

## Prompt

```text
@tasks.md
Please update the tasks.md file to enforce a strict directory structure.
1. All project files (notebooks, data, utils, venv) MUST be created inside a single folder named `Pandas-Mastery-Project/`.
2. Do not create any files directly in the root of `Day 3-4/`.
3. Update every single task description to include the prefix `Pandas-Mastery-Project/` before the file paths (e.g., `Pandas-Mastery-Project/notebooks/pandas_fundamentals.ipynb`).
4. Update the "Path Conventions" section to explicitly state this rule.
```

## Response snapshot

Updated tasks.md file to enforce strict directory structure. Modified all task descriptions to include `Pandas-Mastery-Project/` prefix for all file paths. Updated Path Conventions section to explicitly state the directory rule: all project files must be created inside `Pandas-Mastery-Project/` directory and no files should be created directly in the root of `Day 3-4/`. Updated all notebook, data, utils, and documentation paths to follow the new structure.

## Outcome

- ✅ Impact: Directory structure enforced across all tasks
- 🧪 Tests: none
- 📁 Files: Updated tasks.md with proper directory prefixes
- 🔁 Next prompts: Ready for implementation with correct structure
- 🧠 Reflection: All tasks now properly scoped to Pandas-Mastery-Project directory

## Evaluation notes (flywheel)

- Failure modes observed: none
- Graders run and results (PASS/FAIL): PASS
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): N/A