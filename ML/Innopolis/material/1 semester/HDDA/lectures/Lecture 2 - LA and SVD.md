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
# Singular Value Decomposition (SVD)

The **Singular Value Decomposition (SVD)** is a fundamental matrix factorization technique. It decomposes any real or complex matrix $A$ into three other matrices, providing deep insight into the geometry of linear transformations.
#### **Definition**:
For a matrix  $A \in \mathbb{R}^{m \times n}$, the SVD is a factorization of  $A$  into three matrices:
$A = U \Sigma V^T$
where:
-  $U \in \mathbb{R}^{m \times m}$  is an orthogonal matrix (the left singular vectors),
-  $\Sigma \in \mathbb{R}^{m \times n}$  is a diagonal matrix containing the singular values,
-  $V \in \mathbb{R}^{n \times n}$  is an orthogonal matrix (the right singular vectors).
---
### **Intuition Behind SVD**
SVD can be interpreted geometrically as a transformation:
1. **Right singular vectors** (columns of  $V$) define a set of orthonormal axes in the input space. A matrix whose columns are the **right singular vectors**, which give another set of orthonormal directions.
2. The **singular values** (diagonal elements of $\Sigma$) represent how much the matrix stretches or compresses vectors along those axes.
3. **Left singular vectors** (columns of $U$) define a set of orthonormal axes in the output space. A matrix whose columns are the **left singular vectors** of $A$, representing the directions in which the matrix acts.
SVD breaks down any matrix $A$ into a combination of rotations (or reflections) and scaling.
---

### **Key Theorem: Existence and Uniqueness of SVD**
**Theorem**: Every matrix  $A \in \mathbb{R}^{m \times n}$  has a singular value decomposition. The singular values  $\sigma_1, \sigma_2, ..., \sigma_r$  are unique, and the left and right singular vectors are unique up to signs when the singular values are distinct.

#### **Proof Outline**:
1. **Step 1: Eigenvalue Decomposition**. For  A^T A , compute the eigenvalues and eigenvectors. These eigenvectors form the matrix  V , and the square roots of the eigenvalues form the singular values (the diagonal entries of  \Sigma ).
2. **Step 2: Construction of  U **. The matrix  U  is constructed from the normalized columns of  A V \Sigma^{-1} .

---

### **Properties of SVD**

1. **Rank of the Matrix**: The rank of  A  is the number of non-zero singular values in  \Sigma .
2. **Norms**:
   - The **Frobenius norm** of  A  is:
     
     ||A||_F = \sqrt{\sum_{i=1}^{r} \sigma_i^2}
     
   - The **spectral norm** of  A  is  ||A||_2 = \sigma_1  (the largest singular value).

3. **Low-Rank Approximation**: The best rank- k  approximation to a matrix  A  is given by the truncated SVD, where only the top  k  singular values and their corresponding singular vectors are retained.

---

### **Example**:

Consider the matrix:

A = \begin{pmatrix} 3 & 1 \\ 1 & 3 \end{pmatrix}

To compute the SVD:
1. Compute  A^T A :
   
   A^T A = \begin{pmatrix} 10 & 6 \\ 6 & 10 \end{pmatrix}
   
   Find the eigenvalues and eigenvectors of  A^T A , which give the right singular vectors  V  and the singular values (square roots of the eigenvalues).
   
2. Construct the diagonal matrix  \Sigma  using the singular values, and find the left singular vectors  U .

---

### **Geometric Interpretation**

SVD reveals how the matrix  A  transforms a unit sphere in the input space:
- The matrix  V  rotates or reflects the sphere.
- The diagonal matrix  \Sigma  stretches or compresses the axes of the sphere by the singular values.
- The matrix  U  rotates the stretched shape to fit the output space.

---

### **Applications of SVD**

1. **Dimensionality Reduction**: By retaining only the largest singular values, we can approximate a matrix with lower rank while preserving the most important information

Let’s continue breaking down **Singular Value Decomposition (SVD)** based on your lecture.

---

### **Singular Value Decomposition (SVD)**

SVD is a matrix factorization method that expresses any matrix  A \in \mathbb{R}^{m \times n}  as a product of three matrices: an orthogonal matrix  U , a diagonal matrix  \Sigma , and another orthogonal matrix  V^T . This decomposition provides crucial insights into the structure of matrices and is useful for many applications, from dimensionality reduction to solving linear systems.

The SVD is defined as:

A = U \Sigma V^T

where:
-  U \in \mathbb{R}^{m \times m}  is an orthogonal matrix whose columns are the **left singular vectors** of  A .
-  \Sigma \in \mathbb{R}^{m \times n}  is a diagonal matrix whose diagonal elements are the **singular values** of  A . These singular values are non-negative and typically arranged in descending order.
-  V^T \in \mathbb{R}^{n \times n}  is the transpose of an orthogonal matrix whose columns are the **right singular vectors** of  A .

---

### **Steps for Computing SVD**

1. **Find Eigenvalues and Eigenvectors of  A^T A **:
   - Compute  A^T A . The **right singular vectors** of  A  are the eigenvectors of  A^T A .
   - The **singular values** are the square roots of the eigenvalues of  A^T A .

2. **Find Left Singular Vectors  U **:
   - The **left singular vectors** are obtained by multiplying  A  by the right singular vectors, then normalizing:
   
   u_i = \frac{Av_i}{\sigma_i}
   
   where  v_i  is the  i -th right singular vector and  \sigma_i  is the corresponding singular value.

3. **Construct Diagonal Matrix  \Sigma **:
   - Place the singular values  \sigma_1, \sigma_2, \dots, \sigma_r  on the diagonal of  \Sigma , where  r  is the rank of  A .

---

### **Geometric Interpretation of SVD**

SVD provides a geometric interpretation of matrix transformations. The matrix  A  can be viewed as transforming a vector in three stages:
1. **Rotation** (or reflection) by  V^T ,
2. **Scaling** along principal axes (as defined by the singular values) by  \Sigma ,
3. **Rotation** (or reflection) by  U .

This means that any matrix can be broken down into these three simple transformations, which is a powerful concept for analyzing the behavior of matrices.

---

### **Example of SVD Calculation**

Let’s compute the SVD of a simple matrix:

A = \begin{pmatrix} 3 & 1 \\ 1 & 3 \end{pmatrix}


#### **Step 1: Compute  A^T A **

A^T A = \begin{pmatrix} 3 & 1 \\ 1 & 3 \end{pmatrix}^T \begin{pmatrix} 3 & 1 \\ 1 & 3 \end{pmatrix} = \begin{pmatrix} 10 & 6 \\ 6 & 10 \end{pmatrix}

The eigenvalues of  A^T A  are  \lambda_1 = 16  and  \lambda_2 = 4 , which means the singular values are  \sigma_1 = \sqrt{16} = 4  and  \sigma_2 = \sqrt{4} = 2 .

#### **Step 2: Find Right Singular Vectors (Columns of  V )**
Solve for the eigenvectors of  A^T A . The eigenvector corresponding to  \lambda_1 = 16  is  v_1 = \begin{pmatrix} 1 \\ 1 \end{pmatrix} , and the eigenvector corresponding to  \lambda_2 = 4  is  v_2 = \begin{pmatrix} 1 \\ -1 \end{pmatrix} .

#### **Step 3: Compute Left Singular Vectors (Columns of  U )**
Using the formula  u_i = \frac{Av_i}{\sigma_i} , compute:
-  u_1 = \frac{A v_1}{4} = \begin{pmatrix} 1 \\ 1 \end{pmatrix} ,
-  u_2 = \frac{A v_2}{2} = \begin{pmatrix} 1 \\ -1 \end{pmatrix} .

Thus, the SVD of  A  is:

A = U \Sigma V^T = \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix} \begin{pmatrix} 4 & 0 \\ 0 & 2 \end{pmatrix} \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}^T


---

### **Applications of SVD**

SVD is a versatile tool with many applications in various fields of mathematics, computer science, and engineering:

1. **Dimensionality Reduction**: SVD is used in principal component analysis (PCA) to reduce the dimensionality of datasets by keeping only the most significant singular values and discarding the smaller ones.
   
   - **Example**: In image compression, the SVD can be used to approximate an image matrix by keeping only the largest singular values, significantly reducing the storage required while retaining most of the image’s information.

2. **Solving Linear Systems**: For systems of equations  A x = b , if  A  is not invertible, the SVD provides the **Moore-Penrose pseudoinverse**  A^+ = V \Sigma^+ U^T , which gives the best solution in a least-squares sense.

3. **Noise Reduction**: In signal processing, noise is often associated with smaller singular values. By setting these small values to zero, SVD helps in denoising data.

4. **Matrix Approximation**: The truncated SVD provides the best low-rank approximation to a matrix. Given a matrix  A , the matrix  A_k  formed by keeping only the top  k  singular values minimizes the Frobenius norm  ||A - A_k||_F .

---

### **Low-Rank Approximation via SVD**

A powerful application of SVD is the ability to approximate matrices by matrices of lower rank. This is especially useful when dealing with large datasets, where storing the full matrix is impractical.

#### **Best Low-Rank Approximation Theorem**:
Given a matrix  A  and its SVD  A = U \Sigma V^T , the best rank- k  approximation to  A  (in terms of minimizing the Frobenius norm) is given by:

A_k = U_k \Sigma_k V_k^T

where  U_k  contains the first  k  columns of  U ,  \Sigma_k  contains the first  k \times k  block of  \Sigma , and  V_k^T  contains the first  k  rows of  V^T .

This low-rank approximation captures the most important features of the matrix while discarding the less significant information associated with the smaller singular values.

##### **Example**:
Consider an image represented as a matrix  A \in \mathbb{R}^{500 \times 500} . The SVD of  A  allows us to approximate the image using only the largest singular values, say the top 50, leading to a significant reduction in the storage required for the image.

---

### **Conclusion**

The **Singular Value Decomposition (SVD)** is an indispensable tool in linear algebra, offering insights into the structure of matrices and providing powerful methods for data analysis, dimensionality reduction, and solving linear systems. Its geometric interpretation, combined with its theoretical and practical applications, makes SVD a cornerstone in both theoretical and applied mathematics.
