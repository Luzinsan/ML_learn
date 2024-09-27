# Hyperparameters
- In ML, are a parameters whose values are used to control the learning process
- They are set not learned like mode parameters
Examples: $\alpha$ in [[Gradient Descent, Logistic Regression, etc#Gradient Descent|GD]] 

## Validation & Test Sets
- we can tune hyperparameters using the validation set
- In general, we split out data into: 
  ![[Pasted image 20240916073105.png|400]]
- When we have hyperparameters to tune, we split as follows:
  ![[Pasted image 20240916073209.png|400]]
- As we train models as follows:
  ![[Pasted image 20240916073317.png|400]]

- Solves the "*Model Selection*" problem: selecting the optimal hyperparameters for a given ML model
- But there is another problem:
	- Once we have chosen a model, how we estimate its true error?
	- => We are using an independent "true" set to estimate the true error
	- But the true error is a model's error when tested in the *entire population*
# K-fold Cross-Validation
$$E = \frac{1}{K}\sum^K_{i=1}E_i$$
For k=3 it looks like:
![[Pasted image 20240916073800.png|400]]
# Regularization
Goal of ML:
$$f: \mathbf{x} \to y$$
> So that we can use the learned function to make a predictions on $\color{red}{unseen}$ data
> What can go wrong?: *placeholder*
> Why does it happen?: because when we are learning the function, we measure the success as : $\color{blue}{\text{Success = Goodness of Fit}}$ 
> It leads to **regularization**

- Improves the "*test error*" by compromising on the "*train error*"
- To achieve this, the most common approach is to change the way we measure success of learning as (where usually added a $\lambda \in [0, \infty)$): $$\color{blue}{\text{Success = Googness of Fit}} + \color{red}{\lambda}\color{green}{\text{Simplicity of the Model}}$$
- Mathematically: $$\frac{1}{n}\sum^n_{i=1}(y_i - \mathbf{w}^T\mathbf{x_i})^2 + \lambda \sum^p_{j=1}w_j^2$$
## $\mathcal{L_2}$ Regularization - Ridge Regression
$$\mathcal{L_2} = \frac{1}{n}\sum^n_{i=1}(y_i - \mathbf{w}^T\mathbf{x}_i)^2 + \lambda \sum^p_{j=1}w_j^2$$
where $\lambda \ge 0$ - is a *tuning parameter*

- If $\lambda$ is at a “good” value, regularization helps to avoid [[Introduction to Machine Learning#**Overfitting**|overfitting]]
- Choosing $\lambda$ may be hard: cross-validation is often used
- If there are irrelevant features in the input (i.e. features that do not affect the output), $\mathcal{L2}$ will give them small, but non-zero weights
- Ideally, irrelevant input should have weights exactly equal to 0

## $\mathcal{L_1}$ Regularization - Ridge Lasso
$$\mathcal{L_1} = \frac{1}{n}\sum^n_{i=1}(y_i - {\color{blue}{\mathbf{w}^T\mathbf{x}_i}})^2 + \lambda \sum^p_{j=1}|w_j|$$
where $\lambda \ge 0$ - is a *tuning parameter*

- Similar to $\mathcal{L_2}$
- $\mathcal{L_1}$ *shrinks* the coefficient estimates towards zero
- However, $\mathcal{L_1}$ penalty has the effect of forcing some of the coefficient estimates to be exactly equal to **zero** when the tuning parameter $\lambda$ is sufficiently large
- Hence, much like best subset selection, $\mathcal{L_1}$ performs variable selection
- As a result, models are generally much easier to interpret
- We say that the $\mathcal{L_1}$ yields *sparse models*

# Optimization problem
![[Pasted image 20240916080435.png|400]]
- Mathematical problem in which we want to MAXIMIZE or MINIMIZE a given function: => solution x=4
- But what if we are told that x must be odd?: => solution x=3
**Constrained Optimization Problems**
Optimization problems where a function $f(x)$ has to be *maximized* or *minimized* subject to (s.t.) some constraint(s) $\phi(x)$
- Such optimization problems are solved using **Lagrange Multiplier Method**
- In particular, we take our objective function and the constraint(s) and do the following:
	- We make a new objective function
	- This new objective function contains both the original objective and additional term(s)
	- The additional term(s) represent(s) our constraint(s)
$$\begin{array}{ccc}argmin_{w}f(w) &\to& argmin_wf(w)-\alpha(g(w)-a)\\
s.t.\space g(w)<a & \to & s.t. \space a>0
\end{array}$$
$\color{red}{\alpha}$ - are called **Lagrange Multipliers** 
Thus, in the context of optimization problem:$$\begin{array}{ccc}
\text{Ridge Regression} & \text{Lasso Regression} \\
\min_{\mathbf{w}}\{\frac{1}{n}\sum^n_{i=1}(y_i - \mathbf{w}^T\mathbf{x}_i)^2 \} & \min_{\mathbf{w}}\{\frac{1}{n}\sum^n_{i=1}(y_i - \mathbf{w}^T\mathbf{x}_i)^2 \} \\
s.t.\space \sum^p_{j=1}w^2_j \le s & s.t.\space \sum^p_{j=1}|w_j| \le s
\end{array}$$
## Constraints in a Two Dimensional Space
![[Pasted image 20240916081851.png|400]]

$$\begin{array}{ccc}
\text{Ridge Regression} & \text{Lasso Regression} \\
w^2_1 + w^2_2 \le s & |w_1| + |w_2| \le s 

\end{array}$$

# High Dimensional Data
High Dimensional Data is:
- Difficult to visualize
- Difficult to analyze
- Difficult to understand – to get insight from the data (correlation and predictions)

> We have a dataset of n data points, where each point is p-dimensional $$X = \{(x_i) | x_i \in \mathbb{R}^p\}^n_{i=1}$$
> The number of parameters in a ML model usually depends on the parameter $p$
> Thus if $p$ is *very large*, it can make parameter estimation challenging
> It can also make visualization of the data *very hard*

**Solution**
To solve the problem, we usually transform every $p$-dimensional point $\mathbf{x}_i$ to a new $d$-dimensional point $\mathbf{x}_i'$
•Such that $d < p$
$$X' = \{(x_i') | x_i' \in \mathbb{R}^d\}^n_{i=1}$$
- This process is called $\color{red}{\text{projection}}$
   ![[Pasted image 20240916082631.png|300]]![[Pasted image 20240916082739.png|300]]![[Pasted image 20240916082756.png|300]]

![[Pasted image 20240916082832.png|400]]

- When projecting data to a lower dimensional space, we would like to retain as much of the structural information about our data as possible
- And this is where variance can help us
- We can compute variance in each one-dimensional space as:$$\begin{array}{ccc}
  \sigma^2=\frac{1}{n}\sum^n_{i=1}(x_i'-\mu_{x'})^2 &&& \mu_{x'}=\frac{1}{n}\sum^n_{i=1}x_i'
  \end{array}$$
# Principal Component Analysis (PCA)
- One of the most widely used technique for projecting data into lower dimensions
- When projecting data from $p$-dimensional to a $d$-dimensional space, PCA defines d vectors, each represented as $\mathbf{w}_j$ where $j = 1, ⋯ , d$
- Each vector is $p$-dimensional – that is, $\mathbf{w} \in \mathbb{R}^p$
- The $i_{th}$ projected point is represented as $x_i' = [x_{i1}',x_{i2}',...,x_{id}']^T$ $$x_{id}' = \mathbf{w}^T_d\mathbf{x}_i$$
- PCA uses $\color{red}{\text{variance}}$ in the projected space as its criteria to choose $\color{red}\mathbf{w}_d$
- In particular, $\mathbf{w}_1$ will be the vector that will keep the variance in $\color{red}{x_{i1}'}$ as high as possible
- $\color{red}{\mathbf{w}_2}$ will be chooses to maximize the variance, too, but with an additional constraint
- $\color{red}{\mathbf{w}_2}$ must be orthogonal to $\color{red}{\mathbf{w}_1}$
**In general**:$$\mathbf{w}^T_i\mathbf{w}_j,\space\forall \ne j$$
- In addition the previous constraint, the PCA also demands that:$$\mathbf{w}^T_d\mathbf{w}_d=1$$
- Which means that each vector should have a length of 1
- Finally, PCA requires that each original dimension has zero mean: $$\mu_{\mathbf{x}}=\frac{1}{n}\sum^n_{i=1}x_i = 0$$
#### Summary
- PCA reduces dimensionality by projecting data from a high dimensional space to a lower dimensional space
- It does so by using a set of vectors
- Vectors will be chosen such that they maximize the variance in the project space
- Furthermore, the vectors
	- Should be orthogonal
	- Have unit length
- Finally, data should have zero mean