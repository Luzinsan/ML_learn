### Lecture 3: Linear Regression

#### 1. **Supervised Learning**

The lecture begins with an overview of **supervised learning**, where the goal is to learn a function that maps input features $x$ to outputs $y$ based on a training set. Supervised learning problems can be divided into two categories:

- **Regression**: Predicts continuous variables (e.g., predicting house prices based on features like area).
- **Classification**: Predicts discrete labels (e.g., identifying whether a given object is a cat or a dog).

For example, the dataset might contain living areas of houses and their corresponding prices, and the goal would be to predict the price of a house given its living area. 

#### Training Set
- **Input**  $x^{(i)}$ : Features such as the area of a house.
- **Output**  $y^{(i)}$ : The target or predicted value, such as the price of the house.

The learning algorithm generates a **hypothesis** $h(x)$, which maps inputs to predicted outputs. In regression problems, the function $h(x)$ aims to predict continuous values.

---

#### 2. **Linear Regression with One Variable**
##### **Main Formulation**
In **one-dimensional linear regression**, we aim to find a straight line that best fits the training data. Given a set of points  $(x^{(i)}, y^{(i)})$ , the goal is to approximate these points using a line:
$y = \theta_1 x + \theta_0$
Here:
-  $\theta_1$  is the slope of the line,
-  $\theta_0$  is the intercept.

In **matrix form**, this can be expressed as:
$A \theta = b$
where:
-  $A$  is the matrix of input features,
-  $\theta = [\theta_1, \theta_0]^T$  are the unknown parameters,
-  $b$  is the vector of outputs.
##### **Least-Squares Objective**
The **least-squares problem** is defined as minimizing the squared difference between the predicted values and the actual data:
$\min_{\theta} \sum_{i=1}^{m} \left( y^{(i)} - (\theta_1 x^{(i)} + \theta_0) \right)^2$

This objective function can also be expressed as minimizing the $\ell_2$ -norm of the error:
$\min_{\theta} || A\theta - b ||_2^2$
The solution to this problem gives us the best-fitting line that minimizes the total squared error.
##### **Direct Methods**
Direct methods for solving the least-squares problem involve computing the **normal equation**:
$A^T A \theta = A^T b$
The solution is then given by:
$\theta = (A^T A)^{-1} A^T b$
This method works when  $A^T A$  is invertible, but for large datasets, computing the inverse can be computationally expensive.
##### **Iterative Methods**
For large-scale problems, **iterative methods** such as **gradient descent** are preferred. Gradient descent updates the parameters  \theta  iteratively by moving in the direction of the steepest descent:
$\theta := \theta - \alpha \nabla_{\theta} J(\theta)$
where  \alpha  is the learning rate, and  $\nabla_{\theta} J(\theta)$  is the gradient of the cost function with respect to the parameters.

---
#### 3. **Linear Regression with Multiple Variables**
##### **Main Formulation**
In the case of **multiple variables**, the hypothesis function is extended to:
$h(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \dots + \theta_n x_n$
In matrix form, this is written as:
$h(x) = X \theta$
where:
-  $X$  is the matrix of input features,
-  $\theta$  is the vector of parameters,
-  $h(x)$  is the predicted output.
The goal is still to minimize the squared difference between the predicted and actual values. The objective function remains the same, and the solution can be computed using the normal equation or iterative methods such as gradient descent.
##### **Example**: Cement Heat Generation Data
In this example, the lecture introduces a dataset that describes heat generation for various cement mixtures with four basic ingredients. The goal is to determine the relationship between the proportions of the ingredients and the heat generated, which is a classic multiple regression problem.

---
#### 4. **Probabilistic Interpretation**

The lecture then provides a **probabilistic interpretation** of linear regression. The model assumes that the target variables  $y^{(i)}$  are related to the input variables  $x^{(i)}$  through a linear relationship plus some noise  \epsilon :
$y^{(i)} = \theta^T x^{(i)} + \epsilon^{(i)}$
The noise  \epsilon  is assumed to be normally distributed with mean zero and variance  \sigma^2 , i.e.,
$\epsilon^{(i)} \sim \mathcal{N}(0, \sigma^2)$
This leads to the assumption that the observed  $y^{(i)}$  is drawn from a normal distribution with mean  $\theta^T x^{(i)}$  and variance  $\sigma^2$:
$y^{(i)} \sim \mathcal{N}(\theta^T x^{(i)}, \sigma^2)$

##### **Maximum Likelihood Estimation (MLE)**
Using this probabilistic model, the parameters  $\theta$  can be estimated by maximizing the **likelihood** of the observed data. The likelihood function is given by:

$L(\theta) = \prod_{i=1}^{m} p(y^{(i)} | x^{(i)}; \theta)$

where  $p(y^{(i)} | x^{(i)}; \theta)$  is the probability of observing  $y^{(i)}$  given  $x^{(i)}$.
##### **Log-Likelihood**
For convenience, we maximize the **log-likelihood** instead of the likelihood:
$\log L(\theta) = -\frac{1}{2\sigma^2} \sum_{i=1}^{m} (y^{(i)} - \theta^T x^{(i)})^2 + \text{constant}$

Maximizing the log-likelihood is equivalent to minimizing the least-squares cost function, establishing a direct connection between least-squares regression and MLE.

---
#### 5. **Summary**

- **Supervised learning** focuses on learning a function to map inputs to outputs.
- **Linear regression** can be solved using direct methods (normal equation) or iterative methods (gradient descent).
- **Multiple-variable linear regression** extends the concept to higher dimensions.
- The **probabilistic interpretation** of linear regression justifies least-squares as a maximum likelihood estimator under Gaussian noise assumptions.

This detailed explanation covers all key points in the lecture, following the original content closely.

---
# log-likelihood
Let's dive deep into the **log-likelihood** in the context of linear regression, explaining every step clearly and thoroughly.

### 1. **Understanding Likelihood**

In linear regression, we assume that the observed data $y^{(i)}$ is generated by a **linear model** with some random noise:
$y^{(i)} = \theta^T x^{(i)} + \epsilon^{(i)}$
where:
-  $\theta^T x^{(i)}$  is the predicted value,
-  $\epsilon^{(i)}$  is the noise, which is assumed to be normally distributed with mean zero and variance  $\sigma^2$:
  $\epsilon^{(i)} \sim \mathcal{N}(0, \sigma^2)$
Given this, the observed  $y^{(i)}$  is also normally distributed:
$y^{(i)} \sim \mathcal{N}(\theta^T x^{(i)}, \sigma^2)$
This means that each observation  $y^{(i)}$  comes from a **normal distribution** with:
- **mean**  $\theta^T x^{(i)}$ ,
- **variance**  $\sigma^2$ .

### 2. **Probability Density Function (PDF)** of Normal Distribution

The **probability density function** for a normal distribution with mean  $\mu$  and variance  $\sigma^2$  is:
$p(y^{(i)} | x^{(i)}; \theta) = \frac{1}{\sqrt{2\pi \sigma^2}} \exp\left( -\frac{(y^{(i)} - \theta^T x^{(i)})^2}{2\sigma^2} \right)$
This gives us the probability of observing  $y^{(i)}$  given  $x^{(i)}$  and parameters  $\theta$  and  $\sigma^2$ .
### 3. **Likelihood Function**
The **likelihood** function represents the joint probability of observing the entire dataset  $\{y^{(i)}, x^{(i)}\}$  given the parameters  $\theta$  and  $\sigma^2$ . Assuming that the observations are **independent** (which is a key assumption in regression), the total likelihood is the product of the individual probabilities for each observation:
$L(\theta, \sigma^2) = \prod_{i=1}^{m} p(y^{(i)} | x^{(i)}; \theta, \sigma^2)$
Substituting the normal distribution's PDF for each  $y^{(i)}$ :
$L(\theta, \sigma^2) = \prod_{i=1}^{m} \frac{1}{\sqrt{2\pi \sigma^2}} \exp\left( -\frac{(y^{(i)} - \theta^T x^{(i)})^2}{2\sigma^2} \right)$
### 4. **Log-Likelihood Function**
To simplify the likelihood calculation, we often take the **logarithm** of the likelihood function, turning the product into a sum. This is the **log-likelihood function**:
$\log L(\theta, \sigma^2) = \sum_{i=1}^{m} \log p(y^{(i)} | x^{(i)}; \theta, \sigma^2)$
Substituting the PDF of the normal distribution:
$\log L(\theta, \sigma^2) = \sum_{i=1}^{m} \left( \log \frac{1}{\sqrt{2\pi \sigma^2}} - \frac{(y^{(i)} - \theta^T x^{(i)})^2}{2\sigma^2} \right)$
This simplifies to:
$\log L(\theta, \sigma^2) = -\frac{m}{2} \log(2\pi \sigma^2) - \frac{1}{2\sigma^2} \sum_{i=1}^{m} (y^{(i)} - \theta^T x^{(i)})^2$
### 5. **Interpretation of Log-Likelihood**
The log-likelihood function tells us how likely the observed data is, given the model parameters  $\theta$  and  $\sigma^2$. Our goal is to **maximize the log-likelihood** in order to find the best parameters  $\theta$  and  $\sigma^2$ . In practice, maximizing the log-likelihood is equivalent to finding the parameters that make the observed data most probable under the model.
### 6. **Maximizing the Log-Likelihood (MLE)**
To estimate the parameters  $\theta$  and  $\sigma^2$ , we apply the **maximum likelihood estimation (MLE)** technique, which involves maximizing the log-likelihood function.
#### Step 1: **Maximizing with Respect to  $\theta$ **

We first focus on maximizing the log-likelihood with respect to  \theta . The terms that depend on  \theta  are:
$-\frac{1}{2\sigma^2} \sum_{i=1}^{m} (y^{(i)} - \theta^T x^{(i)})^2$
Maximizing this is equivalent to **minimizing** the sum of squared errors:
$\min_{\theta} \sum_{i=1}^{m} (y^{(i)} - \theta^T x^{(i)})^2$
This is exactly the **least-squares regression** problem, so the MLE for  $\theta$  is the same as the solution for least-squares linear regression:
$\theta = (X^T X)^{-1} X^T y$
#### Step 2: **Maximizing with Respect to  $\sigma^2$ **

Next, we maximize the log-likelihood with respect to  $\sigma^2$ . The relevant term in the log-likelihood is:
$-\frac{m}{2} \log(2\pi \sigma^2) - \frac{1}{2\sigma^2} \sum_{i=1}^{m} (y^{(i)} - \theta^T x^{(i)})^2$
To find the MLE for  $\sigma^2$ , we take the derivative of the log-likelihood with respect to  $\sigma^2$  and set it equal to zero:
$\frac{\partial}{\partial \sigma^2} \log L(\theta, \sigma^2) = 0$
Solving this gives:
$\sigma^2 = \frac{1}{m} \sum_{i=1}^{m} (y^{(i)} - \theta^T x^{(i)})^2$
This is the **variance of the residuals** (the errors between the observed and predicted values).
### 7. **Summary of Log-Likelihood**
1. The **likelihood function** tells us the probability of the observed data given the parameters  $\theta$  and  $\sigma^2$ .
2. The **log-likelihood** is the logarithm of the likelihood function, making it easier to work with because it turns products into sums.
3. **Maximizing the log-likelihood** with respect to  \theta  is equivalent to solving the least-squares problem.
4. The **MLE** for  $\sigma^2$  is the variance of the residuals.
The log-likelihood provides a solid statistical foundation for linear regression, connecting least-squares estimation with maximum likelihood estimation under the assumption of normally distributed errors. This approach is widely used in regression analysis, making it a key concept in both theoretical and applied statistics.
Let me know if you need further clarification!