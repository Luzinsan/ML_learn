# General Form
$$\min_{x \in \mathcal{R}^n}(\text{or }\max)f(x)\text{ s.t. }x \in \mathcal{D}$$
where:
- $f$ is linear: $f(x):\sum^n_{i=1}c_i x_i = c^T x$, where $c \in \mathcal{R}^n$ 
- $\mathcal{D}$ is a polyhedron
$$x \in \mathcal{D} \equiv \left\{
\begin{array}{ccc}
a_i^T x \ge b_i & \text{pour }i \in \mathcal{\iota} \\
a^T_i x = b_i & \text{pour }i \in \mathcal{\varepsilon}
\end{array}
\right\}$$
where $a_i \in \mathcal{R}^n\space et\space b_i \in \mathcal{R}\space\space \forall i \in \iota, \varepsilon$ 
> Remark: $a_i^T x \le b_i \iff (-a_i)^T x \ge -(b_i)$

# Geometric form
> A linear program under $\color{Red}\text{geometric form}$ is a problem of the form:
### Geometric form Formulation
> $$\begin{array}{ccc}\min_{x \in \mathcal{R}^n}c^Tx\\s.t.\space Ax \ge b,\end{array}$$
> where $A \in \mathcal{R}^{m \times n}\space and\space b\in \mathcal{R}^m$
- We have: $$Ax \ge b \equiv a^T_i x \ge b_i \space for\space 1 \le i \le m,$$
where $a_i \in \mathcal{R}^n$ is the vector equal to the $i_{th}$ row of the matrix $A$ 
## From general form to geometric form
We have that
- $\max_x c^T x = -\min_x (-c^T x)$ 
- The constraint $a^T x = b$ is equivalent to the joint constraints $a^T x \ge b$ and $a^T x \le b$ 
- **Conclusion**. Any linear program can be written in geometric form
# Standard form
- A linear program in $\color{Red}\text{standard form}$ is a problem of the form:
### Standard form Formulation
$$\begin{array}{ccc} \min_{x\in \mathcal{R}^n} c^T x\\
  s.t.\space Ax = b, \\
  x \ge 0,\end{array}$$
  where $A \in \mathcal{R}^{m \times n}$ and $b \in \mathcal{R}^m$ 
### Interpretation of the standard form:
  > Let $a^i$ be the $i_{th}$ column of $A$. We are looking for quantities $x_i \ge 0$ for which $\sum_i x_i a^i = b$ and s.t. $c^T x$ is minimum. The problem is that of synthesizing the target vector $b$ by a choice of positive quantities $x_i$ which minimize the total cost $c^T x$
## From geometric to standard form
- **Eliminating inequality constraints**
  For each inequality of the type $\sum_j a_{ij} x_j \ge b_j$, we introduce a $\color{red}\text{slack variable}$ $s_i$. The inequality $\sum_j a_{ij} x_j \ge b_i$ is replaced by the constraints $\sum_j a_{ij} x_j - s_i = b_i$ and $s_i \ge 0$
- **Elimination of free variables**:
  A free variable $x_i$ is replaced by $x_i = x^{+}_i - x^{-}_i$, where $x^{+}_{i}$ and $x^{-}_i$ are the $\color{red}\text{new variables}$ for which we impose $x^{+}_i \ge 0$ and $x^{-}_i \ge 0$ .
---
# **Forms of Linear Programming (LP)**
## **General form**
 $$\min_{x \in \mathbb{R}^n} c^T x \quad \text{subject to} \quad a_i^T x \geq b_i \quad \forall i \in I, \quad a_j^T x = b_j \quad \forall j \in E.$$
 The general form includes both **inequalities** $a_i^T x \geq b_i$ and **equalities** $a_j^T x = b_j$, giving flexibility in constraints.

## **Geometric form**
$$\min_{x \in \mathbb{R}^n} c^T x \quad \text{subject to} \quad Ax \geq b.$$
This form uses only *inequalities*, which simplifies the representation.
## **Standard form**:
 $$\min_{x \in \mathbb{R}^n} c^T x \quad \text{subject to} \quad Ax = b, \quad x \geq 0.$$
Here, all constraints are *equalities*, and the variables must be non-negative.
# **Transformation Between Forms**
## **From general to geometric form**
Equalities $a_j^T x = b_j$ are rewritten as two inequalities: $a_j^T x \geq b_j$ and $a_j^T x \leq b_j$.
## **From geometric to standard form**: 
For each inequality $a_i^T x \geq b_i$, introduce a **slack variable** $s_i \geq 0$ so that $a_i^T x - s_i = b_i$.
Free variables (without non-negativity constraints) are split into two non-negative variables: $x_i = x_i^+ - x_i^-$, where $x_i^+, x_i^- \geq 0$.
# **Example: Transforming to Standard Form**
Consider the problem:   $$\min_{x_1, x_2} 2x_1 + 4x_2 \quad \text{subject to} \quad x_1 + x_2 \geq 3, \quad 3x_1 + 2x_2 = 14, \quad x_1 \geq 0.$$
Use a slack variable for the inequality: $x_1 + x_2 - s_1 = 3$, $s_1 \geq 0$.
If $x_2$ is free, represent it as $x_2 = x_2^+ - x_2^-$, where $x_2^+, x_2^- \geq 0$.
   
The transformed problem becomes:   $$\min_{x_1, x_2^+, x_2^-, s_1} 2x_1 + 4x_2^+ - 4x_2^- \quad \text{subject to} \quad x_1 + x_2^+ - x_2^- - s_1 = 3, \quad 3x_1 + 2x_2^+ - 2x_2^- = 14, \quad x_1, x_2^+, x_2^-, s_1 \geq 0.$$
# **Handling Piecewise Linear and Absolute Value Functions**
**Piecewise linear functions** (e.g.,  $f(x) = \max(c_1^T x + d_1, c_2^T x + d_2)$ ) can be minimized via linear programming by introducing a variable $t$ to represent the maximum of multiple linear expressions.
**Absolute value**: For  $f(x) = |c^T x - b|$ , rewrite it as:
 $$\min t \quad \text{subject to} \quad t \geq c^T x - b, \quad t \geq b - c^T x.$$
# **Chebyshev Center of a Polyhedron**
The Chebyshev center is the center of the largest ball that can be inscribed in a polyhedron $P = \{x \in \mathbb{R}^n \mid a_i^T x \geq b_i\}$
The problem is formulated as finding a point $c \in P$ that maximizes the minimum distance from $c$ to the hyperplanes:
$$\max_{c \in P, t \in \mathbb{R}} t \quad \text{subject to} \quad a_i^T c \geq b_i + t \quad \forall i.$$