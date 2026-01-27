<!--
SYNC IMPACT REPORT
Version change: 1.0.0 → 1.0.0 (Initial creation)
Modified principles: N/A (new constitution)
Added sections: All core principles (1-6), Tech Stack Requirements, Development Workflow
Removed sections: N/A
Templates requiring updates: ✅ Updated .specify/templates/agent-file-template.md (CLAUDE references remain valid)
Follow-up TODOs: None
-->

# NumPy AI Engineering Learning Platform Constitution

## Core Principles

### I. Vectorization Over Loops
Vectorized operations MUST be prioritized over Python loops for all array operations. Use loops ONLY when explicitly requested for educational purposes (e.g., "from-scratch" exercises) to demonstrate performance differences. Always prefer NumPy's optimized C-based operations for production code.

**Rationale**: Vectorization leverages low-level C optimizations, resulting in significant performance improvements (often 10-100x faster) while producing more concise, readable code. This is fundamental to efficient AI/ML workflows.

### II. "From Scratch" Exercise Constraints
For the specific "Implement Matrix Operations from Scratch" exercise, you MUST NOT use `np.dot` or `np.matmul`. Use simple NumPy array indexing and Python loops only to teach the underlying mathematical logic and algorithms.

**Rationale**: Understanding the mechanics behind linear algebra operations is crucial for deeper comprehension of ML algorithms. These exercises are purely educational and should not be used in production code.

### III. Broadcasting First Explanation
When encountering shape mismatches or arithmetic operations on differently-sized arrays, explicitly explain the concept of broadcasting before proposing solutions. Use analogies like "stretching" or "tiling" arrays to make the concept intuitive.

**Rationale**: Broadcasting is a core NumPy concept that enables efficient operations on arrays of different shapes. Clear understanding prevents common errors and promotes efficient code design.

### IV. Code Quality and Naming Standards
Use descriptive, domain-specific variable names (`X_train`, `weights`, `image_array`) instead of generic names (`a`, `b`, `arr`). Include brief comments explaining WHY specific NumPy functions are used, particularly when axis parameters are involved (e.g., `np.sum(axis=0)` to sum across columns).

**Rationale**: Clear naming and documentation make code self-explanatory and maintainable. This is especially important in educational contexts where clarity aids learning.

### V. Mini-Project Focus on Array Representation
For the Image Filter mini-project, emphasize that images are simply 3D NumPy arrays (Height × Width × Channels) and that filters are mathematical operations applied to these arrays. This reinforces the fundamental concept of data-as-arrays in AI engineering.

**Rationale**: Understanding data representation is foundational. Images as 3D arrays bridge the gap between abstract array operations and real-world applications, making the learning more tangible.

### VI. Efficiency Validation
Encourage the use of performance measurement tools like `%timeit` to compare vectorized implementations against loop-based ones, particularly in educational exercises. Quantify the performance differences to reinforce the importance of efficient code.

**Rationale**: Demonstrating concrete performance differences helps internalize the value of vectorization and provides practical experience with performance profiling.

## Tech Stack Requirements

### Language and Environment
- **Language**: Python 3.x (3.8+ recommended)
- **Core Library**: NumPy (latest stable version)
- **Visualization**: Matplotlib for plotting and image display
- **Image Handling**: PIL (Pillow) or ImageIO for loading/saving images
- **Package Manager**: uv (preferred) or pip
- **Development Environment**: Jupyter Notebook (interactive learning) or VS Code (script development)

### Version Compatibility
All code examples and exercises MUST be tested against the latest stable versions of the specified libraries. When version conflicts arise, prioritize compatibility with NumPy 1.20+ (which includes improved array typing and performance optimizations).

## Development Workflow

### Learning Progression
1. **Foundation**: Array creation, shapes, data types, basic operations
2. **Optimization**: Vectorization, broadcasting rules, performance comparison
3. **Linear Algebra**: Dot products, matrix multiplication, common operations
4. **Statistical Operations**: Random sampling, descriptive statistics, axis-based computations
5. **Applied Projects**: From-scratch implementations, image processing mini-project

### Exercise Design Principles
- **Incremental Complexity**: Each exercise should build upon previous concepts
- **Self-Verification**: Provide clear expected outputs for learners to validate their work
- **Performance Awareness**: Include timing comparisons where educational value is high
- **Real-World Connection**: Relate abstract concepts to practical AI/ML applications

### Code Review Standards
- All code examples MUST demonstrate best practices for the specific learning stage
- Vectorized solutions should be provided alongside loop-based versions for comparison
- Error handling should be minimal in examples (focus on core concepts) but explained
- Comments should explain intent, not just describe what the code does

## Governance

### Constitution Authority
This constitution supersedes all other informal guidelines and provides the authoritative source for development standards in the NumPy AI Engineering Learning Platform.

### Amendment Process
- **Minor Clarifications** (typos, wording improvements): Can be applied directly with version patch increment
- **Principle Additions/Modifications**: Require explicit documentation and version minor increment
- **Major Restructuring**: Requires version major increment and review

### Version Compliance
All new exercises, examples, and project materials must comply with the current constitution version. Existing materials should be reviewed and updated during regular maintenance cycles.

### Educational Focus Maintenance
The primary goal is effective learning. When conflicts arise between strict adherence to principles and educational value, prioritize clarity and pedagogical effectiveness while documenting the rationale for any exceptions.

**Version**: 1.0.0 | **Ratified**: 2026-01-15 | **Last Amended**: 2026-01-15
