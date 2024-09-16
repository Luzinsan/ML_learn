Opening Remark and Credit
About more than 264 years ago:
> “Nothing takes place in the world whose meaning is not that of some maximum or minimum.” 
> Leonhard Euler (1707-1783)

Help to make the **best** decision.
$$\left\{
\begin{array}{ccc}
Decision & \text{vector of variable }\mathcal{x} \\
Best & \text{objective function }\mathcal{f(x)} \\
Constraint & \text{feasible set }\mathcal{X}
\end{array}
\right\}
\to \color{RubineRed}Optimization$$
$$\min_{x\in\mathcal{X}}f(x)$$
- $\color{RubineRed}Many$ applications in practice
- $\color{RubineRed}Efficient$ methods in practice
- Modelling and resolution of $\color{RubineRed}\text{large-scale}$ problems
---
# Notation and formulation
Minimization of function $f$: $\mathcal{R}^n \to \mathcal{R}$ over a feasible set $\mathcal{X}$ writes as:
$$\begin{array}{ccc}
\min_{x} & f(x) \\
s.t & x \in \mathcal{X}
\end{array}$$
- The feasible set is usually defined with functional constraints:
$$\mathcal{X} = \{x \in \mathcal{R}^n | h_i(x)=0 \space \forall i\in \varepsilon, \space h_j(x) \ge 0\space \forall j \in \iota \}  $$
- The constraints $h_i$, $h_j$ are functions $h_i$, $h_j$: $\mathcal{R}^n\in\mathcal{R}$:
	- $h_i(x)=0$ is an *equality constraint*
	- $h_j(x)\ge0$ is an *inequality constraint*
- $\varepsilon$ and $\iota$ are set of *indices*

The general optimization problem can be therefore written as follows:
$$\begin{array}{ccc}
\min_{x} & f(x) \\
s.t & h_i(x)=0\space\forall i\in \varepsilon \\
\space & h_j(x) \ge 0 \space \forall j \in \iota
\end{array}$$

> **Particular case: all the functions are linear**
> If all the functions $f(\dot), h_i(\dot), h_j(\dot)$ are linear, the general problem can be simplified as:
> $$\begin{array}{ccc}
\min_{x} & c^T x \\
s.t & Ax \ge b \\
\end{array}$$
> - The constraints $Ax \ge n$ defines a feasible set that is a polyhedron (a polytope if bounded and non-empty)
> - The objective function $c^T x$ forms a translating hyperplane in space
> - In the most case: an optimal solution will be one of the vertices of the polyhedron
> **Why the linear programming?**
> - Many problems can be modelled as linear programs
> - There is a very efficient algorithm (*the simplex algorithm*) for solving these problems
> - The problems have a very rich structure (properties of optimality, duality)
> **Why studying non-linear programming?**
> - Some problems are impossible to model linearly
> - The simplex algorithm for linear programming is inapplicable to non-linear problems

## Applications
- Planning, management and scheduling
  Production, schedules, crew composition, etc.
- Design and conception
  Sizing and optimization of structures and networks
- Economy and finance
  Portfolio selection, balance calculation
- Location and transport
  Relocation of depots, integrated circuits, tours
- Data analysis, machine learning
  Recommendation systems, image analysis, automatic document classification, clustering...
# Faces of optimization
1. Modeling - translating the problem into mathematical language (more delicate than it looks)
   Formulation of an optimization problem
2. Solving - development and implementation of efficient resolution algorithms in theory and practice
Close relationship:
- Formulating models/programs that can be solved
- Developing methods applicable to realistic models/programs
# Taxonomy and Terminology
## Optimization

 - Optimization problems often have thousands of variables and constraints. They rarely have an analytical solution. 
 - We’re looking for optimization algorithms that are fast, easy to implement, require little computation time and memory, are not sensitive to rounding errors, are guaranteed to converge, and allow post-optimal analysis (ideally!). 
 - Modeling is a crucial aspect of optimization.
 - **A model is just a model.** We live in a world of satisfying approximation. We don’t always seek to optimize exactly, but often to optimize satisfactorily. 
 - Each model has its own resolution method. The more precise the model class, the more efficient the method used. 
 - In general, compromise between model quality, complexity and resolution method (exact or approximate resolution - heuristics)
## Taxonomy (hierarchy/types of optimization problems)
$$\min_{x={(x_1,x_2,...,x_n)^T\in\mathcal{R}^n}}f(x_1,x_2,...,x_n)$$
in the feasible set, that is $(x_1, x_2,...,x_n)^T\in D$ 
- Variables: continuous, discrete, binary, etc.
- Constraints: (none), linear, convex, integer, Boolean, etc.
- Objective: linear, quadratic, non-linear, non-differentiable, polynomial, convex, etc.
Models: Multi-criteria optimization, stochastic models, temporal models
Change category: sometimes possible via reformulation
## Feasibility
Problems in *finite* dimensions
$$\left\{
\begin{array}{ccc}
Decision & \text{vector of variable }\mathcal{x} \\
Best & \text{objective function }\mathcal{f(x)} \\
Constraint & \text{feasible set }\mathcal{D}
\end{array}
\right\}
\to \color{RubineRed}Optimization$$
- Any point $x$ belonging to the feasible set is called *feasible solutions*
- When $D \ne \emptyset$, the problem is said to be possible or feasible
- When $D = \emptyset$, the problem is said to be impossible or infeasible
## Optimal value
$$\min_{x\in\mathcal{R}^n}{f(x)} \space s.t.\space x \in D$$
- The ==optimal value== of the problem, denoted $f^*$, is the *infimum* of objective function values for feasible solution, i.e.
$$f^* = inf\{f(x)|x \in \mathcal{D}\}$$
- When $f^*$ is finite, the problem is said to be *bounded*
- When $f^* = -\infty$ , the problem is said to be *unbounded*
- When the problem is impossible, we conventionally set $f^*=+\infty$ (worst possible value for a minimum)
## Optimal solution
$$\min_{x\in\mathcal{R}^n}{f(x)} \space s.t.\space x \in D \space\space\space\space\space\space et \space \space \space \space \space \space 
f^* = inf\{f(x)|x \in \mathcal{D}\}$$
- An ==optimal solution==, denoted $x^*$, is a feasible solution that possesses the optimal value, i.e.
$$x^*\text{ is an optimal solution } \iff x^* \in \mathcal{D} \space and\space f(x^*) = f^*$$
- This last property can be equivalently reformulated as
$$x^* \in \mathcal{D} \space and\space f(x^*) \le f(y)\text{ for all feasible solution } y \in \mathcal{D},\space or$$
$$x^* \in \mathcal{D}\text{ and none feasible solution } y \in \mathcal{D}\text{ satisfies } f(y) < f(x^*)$$

- A problem that has (at least) one optimal solution is said to be ==solvable==, otherwise it is called ==unsolvable==
- An impossible or unbounded problem is never solvable, but there are also possible, bounded and unsolvable, for instance: $$min{\frac{1}{x}}\space s.t.\space x > 0 \space gives\space f^*=0\text{ but is unsolvable}$$![[Reciprocal function.png]]
## Problem types
$$\min_{x \in \mathcal{R}^n}{f(x)}\text{ such that x }\in \mathcal{D}\text{   and  }f^* = \inf{\{f(x) | x \in \mathcal{D}\}}$$
- Without loss of generality, we can consider only the ==minimization==
- If ==maximization==, we have equivalence between optimal solutions:
$$x^*\text{ optimal for }\max_{x \in \mathcal{D} \subseteq \mathcal{R}^n} f(x)\text{    }\iff x^*\text{ optimal for }\min_{x\in\mathcal{D}\subseteq\mathcal{R}^n}-f(x)$$
and for *optimal values* (with a double minus sign):
$$\sup{\{f(x) | x \in \mathcal{D} \subseteq \mathcal{R}^n\} = -\inf\{-f(x)|x \in \mathcal{D} \subseteq \mathcal{R}^n\}}$$
- The problem of finding an admissible point (without an objective function) is a special case an optimization problem, and can be formally expressed using a constant (or zero) objective function.
$$\min_{x \in \mathcal{R}^n} 0 \text{ s.t. } x \in \mathcal{D}$$
## Description of the feasible set - linear problems
$$\min_{x\in\mathcal{R}^n}f(x)\text{ s.t. }h_i(x)=0\text{ pour }i\in\varepsilon\text{ and }h_i(x) \ge 0\text{ pour }i\in\iota$$
is called a linear optimization problem when all functions are ==linear or affine==:
- Objective function: $f(x)=c_1 x_1 + c_2 x_2 + ... + c_n x_n$ (constant term is useless)
- Constraints: $h_i(x) = a_{i1} x_1 + a_{i2} x_2 + ... + a_{in} x_n - b_i$
  hence the equivalent formulation
  $$\min_{x \in \mathcal{R}^n}c^T x\space\space\space\text{   s.t.    }\space\space\space a^T_i x = b_i \text{ for } i \in \varepsilon\text{ and } a^T_i x\ge b_i \text{ for }i\in \iota$$
  with columns vector $c = (c_1\space c_2 \space ... \space c_n)^T$ and $a_i = (a_{i1} \space a_{i2} \space ... \space a_{in})^T$ defined  $\forall i \in \varepsilon \cup \iota$


