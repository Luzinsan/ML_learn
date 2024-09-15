> Just equation of a straight line
$$y=ax+b$$
$a$ - *slope*. 
![[linear regression - diff slopes.png|200]]: *different slope* and the same intercept
$b$ - *intercept*
![[linear regression - diff intercepts.png|200]]: *different intercept* and the same slope

$$y=w_1 x + w_0$$
$w_0$ (intercept) - an average value of the target when the rest features don't impact (not available) (the default level of target)

Why **Simple Linear Regression**:
- *Regression* - because the target that are trying to estimate is numerical
- *Linear* - because the estimating function represent function of the straight line
- *Simple* - because it consists a single feature

### General Form
$$y=w_0 + w_1x+w_2x_2+...+w_px_p$$
- the response variable is quantitative
- the relationship between response and predictors is assumed to be linear in the inputs
- thus we are restricting ourselves to a hypothesis space of linear functions

#### Advantages
- Easy for inferencing, because after training we can look at the parameters and find out what the features impact more to our target
- Serves as a good jumping point for more powerful and complex approaches

### Train Linear Regression Model
We have estimated function (on some iteration) and actual points:
$e_i = y_i - f(x_i)$ ![[calc. error.png|200]]
$$\mathcal{L}(w_0, w_1)=\frac{1}{n}\sum^n_{i=1}(y_i-(w_0+w_1 x_i))^2$$
- this loss function - MSE - mean squared error
Then we calculate the differences between predicted and actual labels on entire dataset and use this results in forming the loss (or cost) function (that tell us how bad or good working our predictor), that we need to minimize. 
> So, the aim of this process is to find the value of parameters that minimize this loss function. 
$$\underset{w_0, w_1}{argmin}\space\mathcal{L}(w_0, w_1)$$

### Derivative
> A mathematical tool that can tell us whether a function increases or decreases as we slightly increase its input
- derivative of a function of a single variable: $$f'(x) = \lim_{dx \to 0}\frac{f(x+dx)-f(x)}{dx}$$
- ![[derivative sample.png|150]]
  More simpler explanation: we would like to know will the function's value increase or decrease if we increase the value of abscissa, thus we can use derivative of the function at that interested point.
**Minimum and maximum**
- Can be found by calculating the equation are equated to zero
![[min-max.png|250]]
#### Convex vs. Non-convex
- In **convex** we have only one minimum which is global minimum.
  ![[convex.png|150]]
- In **non-convex** we have multiple minimum points which are local minimums and one global minimum:
   ![[non-convex.png|200]]

> **MSE** is convex: ![[MSE-3D.png|200]]
> at the *unique minimum* of our loss function, it's "partial" derivative with respect to $w_0$ and $w_1$ will be zero
> We can find this solution using *direct solution* that is called as **Least Square Solution**

#### Least Square Solution
*Steps of algorithm:*
1. Compute partial derivatives of the loss (objective) function with respect to weights ($w_0$ and $w_1$ in this case)
2. Set them to $0$
3. Solve for $w_0$ and $w_1$
Solution step-by-step:
Put: $\mathcal{L}(w_0, w_1) = \frac{1}{n}\sum^n_{i=1}(y_i - (w_0 + w_1 x_i))^2$
1. Expand: $\mathcal{L}(w_0, w_1) = \frac{1}{n}\sum^n_{i=1}(y_i^2 - 2y_iw_0 - 2y_iw_1x_i + w_0^2 +2 w_0 w_1x_i + w_1^2 x_i^2)$ 
2. Partial derivative (of the loss function) with respect to $w_0$: $\frac{\mathcal{L}(w_0, w_1)}{dw_0} = 2w_0 -\frac{2}{n}\sum^{n}_{i=1}y_i + \frac{2w_1}{n}\sum^{n}_{i=1}x_i$    
	1. $2w_0 -\frac{2}{n}\sum^{n}_{i=1}y_i + \frac{2w_1}{n}\sum^{n}_{i=1}x_i = 0$
	2. $w_0 =\frac{1}{n}\sum^{n}_{i=1}y_i + \frac{w_1}{n}\sum^{n}_{i=1}x_i$
	3. $\color{red}{w_0 = \overline{y} + w_1 \overline{x}}$ 
3. Partial derivative with respect to $w_1$: $\frac{\mathcal{L}(w_0, w_1)}{dw_1} = -\frac{2}{n}\sum^{n}_{i=1}y_ix_i +\frac{2w_0}{n}\sum^{n}_{i=1}x_i+\frac{2w_1}{n}\sum^{n}_{i=1}x_i^2$
	1. $-\frac{2}{n}\sum^{n}_{i=1}y_ix_i +\frac{2w_0}{n}\sum^{n}_{i=1}x_i+\frac{2w_1}{n}\sum^{n}_{i=1}x_i^2 = 0$
	2. $-\overline{yx} +w_0\overline{x}+w_1\overline{x^2} = 0$ 
	3. $w_1 = \frac{\overline{yx} -w_0\overline{x}}{\overline{x^2}}$ | $\color{red}{w_0 = \overline{y} + w_1 \overline{x}}$
	4. $w_1 = \frac{\overline{xy} -(\overline{y} + w_1 \overline{x})\overline{x}}{\overline{x^2}}$
	5. $w_1 \overline{x^2} = \overline{xy} -\overline{x}\space\overline{y} - w_1 \overline{x}^2$
	6. $w_1 (\overline{x^2} - \overline{x}^2) = \overline{xy} -\overline{x}\space\overline{y}$
	7. $\color{red}{w_1 = \frac{\overline{xy} -\overline{x}\space\overline{y}}{\overline{x^2} - \overline{x}^2}}$

# Expending Linear Regression
## Polynomial Regression
![[polynomial function with 2 and 4 degree.png|500]]
- Using the same framework, to fit a family of more complex models through a transformation of *predictors*
- Then the model will be still *linear in parameters*, but *polynomial in predictors*: $$y=w_0 + w_1x+w_2x^2+...+w_dx^d$$
- Needs to keep in mind an *overfitting* and use validation techniques, fit the hyperparameters
> Hyperparameters are parameters in our learning problem that don't learn from the data but rather we decide the value ourself. In this case: degree of the polynomial

We introduce new linear parameters, but it doesn't impact on the view of error (e.g. MSE), because in partial derivative we calculate with respect to the certain parameters rather than features (they don't participate in out optimization problem).

## Loss functions
### MSE - mean squared error
- **Good**: when there could be outliers in the data - *mse good deals with outliers*
- **Bad**: because its derivative is the same everywhere - *gradient is constant*
### MAE - mean absolute error
- **Good**: gradient is large for large loss and decreases as loss approaches 0 - *gradient is smooth*
- **Bad**: when there are outliers in the data - *mse bad deals with outliers*
![[MSE vs. MAE.png|200]]
### Huber loss
- Goods of both MSE and MAE
- But has extra *hyperparameters* that needs tuning
---
- Why is predicting the number of defects in a software a regression problem?
- What do the words "simple", "linear", and "regression" represent in Simple Linear Regression?
- Why do we want to minimize MSE with respect to the parameters of our regression model?
- What is the role of derivative in this minimization problem?
- What are convex functions and why are we delighted that MSE is convex function?
- What is linear regression problem?
- Why is linear regression inadequate for non-linear problems?
- What is a polynomial function and how can they help us in non-linear regression problems?