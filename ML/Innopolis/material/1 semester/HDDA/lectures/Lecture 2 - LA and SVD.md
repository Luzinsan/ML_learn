### **Outline of Lecture 2**

1. **Matrix Norms**
2. **Four Fundamental Subspaces**
3. **Unitary and Orthogonal Matrices, and Projectors**
4. **Singular Value Decomposition (SVD)**
5. **Rank, Range, and Null Space**
6. **Low-Rank Approximation**
---
### **1. Matrix Norms**
Matrix norms are used to measure the "size" or "length" of a matrix. They generalize vector norms to matrices and help in understanding how a matrix transforms vectors.
#### **Definition of a Matrix Norm**:
A matrix norm  $||A||$  assigns a non-negative number to a matrix  $A \in \mathbb{R}^{m \times n}$  such that:
1.  $||A|| = 0$  if and only if $A = 0$ (the matrix has all zero entries).
2.  $||cA|| = |c| \cdot ||A||$  for any scalar $c$ and matrix $A$.
3. **Triangle inequality**:  $||A + B|| \leq ||A|| + ||B||$ for any matrices $A$ and $B$.
These properties are similar to vector norms.
#### **Induced Matrix Norms**:
Matrix norms can be **induced** from vector norms. This means the matrix norm depends on how much a matrix can *stretch* a vector. For a vector $x$, the induced matrix norm is given by:
$||A|| = \max_{x \neq 0} \frac{||Ax||}{||x||}$
This measures how much the matrix $A$ amplifies a vector $x$.
##### **Example**:
Consider a matrix $A$:
$A = \begin{pmatrix} 2 & -1 \\ -1 & 2 \end{pmatrix}$
Let  $x = \begin{pmatrix} 1 \\ 1 \end{pmatrix}$. The result of multiplying $A$ by $x$ is:
$Ax = \begin{pmatrix} 1 \\ 1 \end{pmatrix}$
In this case, the matrix does not stretch the vector, so the norm of the vector stays the same.
However, for  $x = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$, we get:
$Ax = \begin{pmatrix} 2 \\ -1 \end{pmatrix}$
The matrix amplifies the vector's norm by a factor of $3$.
#### **Special Cases of Induced Norms**:
- **1-norm**: Maximum sum of absolute column entries.
- **Infinity-norm**: Maximum sum of absolute row entries.
- **Spectral norm**: Related to the largest eigenvalue of $A^T A$.
---
### **2. Four Fundamental Subspaces**
Every matrix $A \in \mathbb{R}^{m \times n}$  is associated with four fundamental subspaces that describe how it acts on vectors.
#### 1. **Range (Column Space)**:
- The **range** or **column space** of a matrix $A$ is the set of all vectors that can be expressed as $Ax$ for some vector $x$. This describes how the matrix transforms vectors into a new space.
- **Example**: If $A$ is a $3 \times 2$ matrix, the range is a subspace of  $\mathbb{R}^3$ .
#### 2. **Null Space (Kernel)**:
- The **null space** of a matrix $A$ is the set of all vectors $x$ such that $Ax = 0$. It tells us which vectors are "killed" by the matrix transformation.
- **Example**: For a matrix  $A = \begin{pmatrix} 1 & 2 \\ 3 & 6 \end{pmatrix}$, the null space consists of all vectors proportional to  $\begin{pmatrix} -2 \\ 1 \end{pmatrix}$  because multiplying $A$ by this vector gives the zero vector.
#### 3. **Row Space**:
- The **row space** is the span of the rows of the matrix. It tells us how the matrix acts in the space of its rows.
#### 4. **Left Null Space**:
- The **left null space** is the set of vectors $y$ such that  $A^T y = 0$. This space is orthogonal to the row space.
Each of these spaces plays a role in solving systems of linear equations, matrix decompositions, and understanding the structure of a matrix.
---
### **3. Unitary and Orthogonal Matrices**
#### **Orthogonal Matrices**:
- A matrix  $Q \in \mathbb{R}^{n \times n}$  is **orthogonal** if its columns are orthonormal, meaning  $Q^T Q = I$  (identity matrix).
- Orthogonal matrices preserve the length of vectors:  $||Qx|| = ||x||$ .
- **Geometric Interpretation**: Orthogonal matrices represent rotations or reflections in space.
#### **Unitary Matrices**:
- A matrix  $U \in \mathbb{C}^{n \times n}$  is **unitary** if  $U^H U = I$ , where  $U^H$  is the conjugate transpose of $U$. Unitary matrices generalize orthogonal matrices to complex numbers.
---
### **Projectors (Projection Matrices)**

A **projector** is a matrix that projects a vector onto a subspace. 
#### **Definition:**
A matrix $P \in \mathbb{R}^{n \times n}$  is a **projector** (or projection matrix) if it satisfies:
$P^2 = P$
This means applying the projection twice is the same as applying it once, i.e., projecting a vector that is already in the subspace does nothing further.
##### **Example**:
Consider a $2D$ vector space where we project onto the $x$-axis. The projection matrix is:
$P = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}$
For any vector  $v = (x, y)^T$ , we have:
$Pv = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix} \begin{pmatrix} x \\ y \end{pmatrix} = \begin{pmatrix} x \\ 0 \end{pmatrix}$
This projects the vector onto the x-axis.
### **Types of Projections**

#### **1. Orthogonal Projection**:
An **orthogonal projector** projects vectors onto a subspace in such a way that the difference between the vector and its projection is orthogonal to the subspace.
##### **Properties**:
-  $P^T = P$, meaning that the matrix is **symmetric**.
- **Orthogonality Condition**: If $P$ projects onto subspace $S$, then for any vector $v \in S$,  $v - Pv$  is orthogonal to $S$.
##### **Example**:
Consider the matrix that projects onto the line  $y = x$:
$P = \frac{1}{2} \begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}$
This is an orthogonal projector, as it satisfies $P^T = P$.

### **5. Rank, Range, and Null Space**
Understanding the **rank**, **range**, and **null space** of a matrix helps in determining how "powerful" the matrix is in transforming vectors and solving linear systems.
#### **Rank**:
- The **rank** of a matrix $A$ is the dimension of its range (column space), i.e., the maximum number of linearly independent columns (or rows) in the matrix.
- If a matrix has full rank, it means that all its columns (or rows) are linearly independent, and the matrix can fully span the space it operates in.
  - **Example**: Consider the matrix $A = \begin{pmatrix} 1 & 2 \\ 3 & 6 \end{pmatrix}$. The second column is a multiple of the first one, so the rank of $A$ is 1 (not full rank), which means that $A$ does not have independent directions and compresses vectors into a lower-dimensional space.
- **Rank-Nullity Theorem**: The rank of a matrix plus the dimension of its null space equals the number of columns:
  $\text{Rank}(A) + \text{Nullity}(A) = n$
  This helps to understand how much information the matrix preserves versus how much it "loses" (null space).

---
### **6. Low-Rank Approximation**

#### **Why Low-Rank Approximation?**:
In many real-world applications, such as data compression, machine learning, and signal processing, we often deal with large matrices that contain redundant or noisy data. Low-rank approximation helps us simplify the matrix by capturing its most important features and discarding unnecessary information.

---
---
---
[[Singular Value Decomposition (SVD)]]