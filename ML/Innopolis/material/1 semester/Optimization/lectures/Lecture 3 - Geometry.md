#### 1. Polyhedron Geometry
- **Polyhedra** are subsets of   $\mathbb{R}^n$  that can be written as the intersection of a finite number of half-spaces.
  - Formally:   $P = \{ x \in \mathbb{R}^n | A^T x \geq b \}$  , where  $A$  is a matrix and  $b$  is a vector.
  - A polyhedron is always **convex** since it is the intersection of convex sets.
  - A bounded polyhedron is called a **polytope**.
#### 2. Examples of Polyhedra
- **Half-space**: The set of points satisfying a linear inequality.
- **Hyperplane**: The set of points satisfying a linear equality.
- **System of linear equalities and inequalities**:   $A^T x \leq b$ ,  $A^T x = b$  .
- **Polyhedron in standard form**:  $\{ x | Ax = b, x \geq 0 \}$.
#### 3. Fundamental Theory
- Linear optimization problems are solved using the **simplex method**, which is based on geometric principles.
- **Fundamental theorem**: If a linear optimization problem has a finite optimal cost and the polyhedron has a **vertex**, then one of the vertices is an optimal solution.
#### 4. Basic Feasible Solution (BFS)
- **Active constraints** are those for which  $a^T x^* = b$  at a point  $x^*$. All equality constraints are active at any point of the polyhedron.
- A **basic feasible solution** is a point where   $n$   linearly independent constraints are active.
- If more than   $n$   constraints are active, the solution is **degenerate**.
#### 5. Vertices and Extreme Points
- An **extreme point** of a polyhedron  $P$  is a point that cannot be expressed as a convex combination of other points of   $P$  . Formally, there are no two distinct points  $y, z \in P$  and a scalar  $0 \leq \lambda \leq 1$  such that  $x = \lambda y + (1 - \lambda) z$  .
- A **vertex** of a polyhedron  P  is a point that can be separated from  P  by a hyperplane, i.e., there exists a vector   $c$   such that   $c^T x < c^T y$   for all   $y \in P, y \neq x$  .
- **Theorem**: A point is a vertex if and only if it is an extreme point, and it is also a basic feasible solution.
#### 6. Vertex Calculation/Enumeration
- **Geometric form**: For a polyhedron  $P = \{ x | A x \geq b \}$, list all square submatrices of $A$ with $n$ rows and compute  $x^* = A^{-1} b$. If  $x^*$   satisfies all constraints, it is a vertex.
- **Standard form**: For a polyhedron  $P = \{ x | Ax = b, x \geq 0 \}$, choose   m   basic variables such that the matrix   $A_B$   is non-singular. Solve   $A_B x_B = b$, and if $x_B \geq 0$, the solution is a vertex.
#### 7. Degeneracy
- A **degenerate** basic feasible solution occurs when some of the basic variables  $x_B$ are zero. Degeneracy happens when the number of active constraints exceeds the number of independent variables.
#### 8. Examples
- Various examples illustrate how to choose active constraints and find basic feasible solutions by selecting linearly independent constraints.
#### 9. Adjacent Basic Feasible Solutions
- Two basic feasible solutions are **adjacent** if they share   $n - 1$   common active constraints.