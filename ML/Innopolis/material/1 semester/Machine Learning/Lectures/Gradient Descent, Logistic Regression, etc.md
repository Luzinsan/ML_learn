# Gradient Descent
$$\mathcal{L}(w_0, w_1) = \frac{1}{n}\sum^n_{i=1}(y_i - (w_0 + w_1 x_i))^2 = \frac{1}{n}\sum^n_{i=1}(y_i - \hat{y_i})^2$$
- Learning the machine learning model by *iteratively* reducing the loss 
- "learn by making mistakes"
> Gradient is a $\color{red}{vector}$: has both $\color{blue}{magnitude}$ and $\color{blue}{direction}$
> It always points to the direction of the steepest increase in the loss function
> And our objective: **reduce the loss**
> Thus we can reduce the loss by taking $\color{blue}{\text{a step}}$ in the direction of $\color{red}{\text{negative gradient}}$
> 
1. Start with initial assumption of our parameter values
2. Computer errors (loss function)
3. Used our wrong predictions we update the parameters of model and get a new model, continue to 2nd step
4. It continues until convergence (reached either to target minimum point or closer to it).
#### In action
- Pick a starting value of $w_i$ and compute the starting loss at this point ![[Pasted image 20240915115154.png|200]] 
- Compute the $\color{blue}{gradient}$ of the loss curve at the starting point (and get a vector)
- Gradient of the curve at any point is equal to the $\color{blue}{derivative}$ of the curve at that point (*a slight modification for easier math*):$$\begin{array}{ccc}
  \mathcal{L}(w_0, w_1) = \frac{1}{\color{red}{2} n}\sum^n_{i=1}(y_1 - (w_0 + w_1 x_i))^2 \\
  \mathcal{L}(w_0, w_1) = \frac{1}{2n}\sum^n_{i=1}(y_1^2 - 2y_i(w_0 + w_1 x_i) + (w_0 + w_1 x_i)^2) \\
  \mathcal{L}(w_0, w_1) = \frac{1}{2n}\sum^n_{i=1}(y_1^2 - 2y_iw_0 - 2y_iw_1 x_i + w_0^2 + 2w_0w_1 x_i + w_1^2x_i^2) \\ 
  \frac{\mathcal{dL}}{\mathcal{d}w_0} = \frac{1}{2n}\sum^n_{i=1}(-2y_i + 2w_0 + 2w_1 x_i) = \frac{1}{n}\sum^n_{i=1}(-y_i + (w_0 + w_1 x_i))  = \frac{1}{n}\sum^n_{i=1}({\color{red}{\hat{y_i}}} - y_i) \\
  \frac{\mathcal{dL}}{\mathcal{d}w_1} = \frac{1}{2n}\sum^n_{i=1}(-2y_ix_i + 2w_0x_i + 2w_1 x_i^2) = \frac{1}{n}\sum^n_{i=1}(-y_i + (w_0 + w_1 x_i)){\color{red}{x_i}}  = \frac{1}{n}\sum^n_{i=1}({\color{red}{\hat{y_i}}} - y_i){\color{red}{x_i}}
  \end{array}$$
- Reduce the loss by taking $\color{blue}{\text{a step}}$ in the direction of negative gradient (*iteratively*): ![[Pasted image 20240915120445.png|200]]
- We should move (*update*) in direction with some $\color{blue}{fraction}$ (known as the $\color{red}{\text{learning rate}}$) of the gradient's magnitude: $$\begin{array}{ccc}
  w_0  = w_0 - {\color{red}{\alpha}} \frac{\mathcal{dL}}{\mathcal{dw_0}} = w_0 - {\color{red}{\alpha}}\frac{1}{n}\sum^n_{i=1}({\color{blue}{\hat{y_i}}} - y_i)\\
  w_1  = w_1 - {\color{red}{\alpha}} \frac{\mathcal{dL}}{\mathcal{dw_1}} = w_1 - {\color{red}{\alpha}}\frac{1}{n}\sum^n_{i=1}({\color{blue}{\hat{y_i}}} - y_i){\color{yellow}{x_i}}
  \end{array}$$
- The gradient descent the repeats this process, edging ever closer to the minimum: ![[Pasted image 20240915120710.png|200]]
## Multiple predictors
$$y=w_0 + w_1x_1 + w_2x_2 + ... + w_dx_d$$
Rewrite weights update process (where the upper index is the number of *feature* and the $0_{th}$ feature is a vector of zero values): $$\begin{array}{ccc}
  w_0  = w_0 - \alpha\frac{\mathcal{dL}}{\mathcal{dw_0}} = w_0 - \alpha\frac{1}{n}\sum^n_{i=1}(\hat{y_i} - y_i)x_i^0\\
  w_1  = w_1 - \alpha\frac{\mathcal{dL}}{\mathcal{dw_1}} = w_1 - \alpha\frac{1}{n}\sum^n_{i=1}(\hat{y_i} - y_i)x_i^1 \\
  \text{or general view:} \\
  w_j  = w_j - \alpha\frac{\mathcal{dL}}{\mathcal{dw_j}} = w_j - \alpha\frac{1}{n}\sum^n_{i=1}(\hat{y_i} - y_i)x_i^j
  \end{array}$$
# Classification
> Response is qualitative, discrete or categorical
## Binary classification
> Classification task that has two class labels

**Approach #1**: instead of predicting the class of a data point, what if we computed the $\color{pink}{probability}$ of the data point belonging to a certain class. Thus we have *p(Approved = Yes | creditscore)*. And we'll "threshold" the result to decide the class.![[Pasted image 20240915151244.png|200]]
*Advantage* of this approach: $y$ becomes continuous, thus we can solve this as a **regression** problem.
*Disadvantage*: this model is unlimited and we can get a negative prediction, but this isn't possible probability as well as a probability that greater than 1 (because the rules of probability: $0\le p \le 1$)

**Approach #2**: using the [[Функции активации#Sigmoid Function|sigmoid]] (logistic) function. **Logistic regression** follows from using this function: $$\sigma(z) = p(x)=\frac{1}{1+e^{-z}}$$
- We take a linear regression model: $z = w_0 + w_1x_1$ 
- The result of that prediction we apply on sigmoid function![[Pasted image 20240915151308.png|300]]
- Thus, we must model p(x) ($0 \le p(x) \le 1$) using a function that gives outputs between $0$ and $1$ $\forall x$
- For multiple case: $$\begin{array}{ccc}p(\mathcal{x})=\frac{1}{1+e^{-z}}\\z=w_0+w_1x_1+w_2x_2+...+w_dx_d\end{array}$$
### Logistic regression loss function
Let's rewrite linear regression model in vector notation: $y=w_0+w_1x_1+w_2x_2+...+w_dx_d$ => $\color{red}{y=\mathbf{w}^T\mathbf{x}}$*, if:*
$$\begin{array}{ccc}

\mathbf{w} = \left[\begin{array}{ccc}w_0\\w_1\\\vdots\\w_d\end{array}\right] & 
\mathbf{x} = \left[\begin{array}{ccc}1\\x_1\\\vdots\\x_d\end{array}\right]
\end{array}$$
If predicted label noted as: $\hat{y^i} = \frac{1}{1+e^{-\mathbf{w}^T\mathbf{x}^i}}$: $$\mathcal{L}(\mathbf{w}) = \frac{1}{n}\sum^n_{i=1}({\color{red}{y^i} }- \hat{y^i})^2$$
- Due to $\color{red}{\text{non-linearity of the sigmoid function in hypothesis of Logistic Resression}}$ => **MSE** is $\color{blue}{\text{not convex}}$ anymore 
---
Put we have two probability distributions for each training example: one is the *true distribution* and the other is the distribution that the model *predicts* for the training example.
In order to see how well our model is doing, we can measure the **difference between these two distributions**, and this is where cross entropy comes into play.
#### Cross Entropy
[[Softmax Regression#Loss Function - Cross-entropy loss|link]] 
- is a measure of the difference between two probability distributions (let's $t$ & $p$), which is defined as (where n - amount of classes): $$H(t,p) = -\sum^n_{i=1} t(i) * \log(p(i))$$
Example: ![[Pasted image 20240915162756.png|200]]
Cross-Entropy is a $\color{red}{\text{convex function}}$
##### Binary Cross-Entropy Loss
> Cross Entropy in a binary classification problem
> Let's $p$ - predicted probability ($1 \le p \le 1$) and $y$ - true label $\in \{0,1\}$
$$\begin{array}{ccc}\text{Binary Cross-Entropy Loss} = -(y*\log(p) + (1-y)*\log(1-p))\\\mathcal{L}(\mathbf{w})=-\frac{1}{n}\sum^n_{i=1}\left(y_i*\log(p(x_i)) + (1-y_i)*\log(1-p(x_i)))\right)\end{array}$$
- We'll define out objective function as: $argmin_{w_0,w_1}\mathcal{L}(\mathbf{w})$  
**Gradient descent**:
$$w_j = w_j - \alpha \frac{\mathcal{dL}}{\mathcal{dw_i}}$$
- Solved a gradient we get: $$w_j = w_j - \alpha \frac{1}{n}\sum^n_{i=1}(p(x_i) - y_i)x_{ij}$$ 
### Summary of Logistic Regression
1. Initialize the model parameters with random values.
2. Feed the training data through the model, and use the model's predictions to calculate the cross-entropy loss between the predicted probabilities and the true labels.
3. Calculate the gradient of the loss function with respect to the model parameters.
4. Update the model parameters in the opposite direction of the gradient, using a small learning rate. This will 'step' the model in the direction that decreases the loss.
5. Repeat steps 2-4 until the model converges, or until a maximum number of iterations is reached.
6. During training, the model will 'learn' the optimal parameters that minimize the cross-entropy loss on the training data. Once training is complete, the model can be used to make predictions on new, unseen examples.

## Types of Gradient Descent
- Batch GD - compute the derivatives and update the parameters on the **entire** dataset. Ok if the dataset is quite small. Updates are much more smooth/stable because we use entire data and update on it.
- Stochastic GD - compute the loss value for only one example, update parameters and repeat this process for each following example on updated model. Updates are much more fast, but more noisy because we update upon on each example.
- Mini-batch GD - the middle approach between other two types of GD -  compute the loss on the particular part of dataset and update parameters on this part of dataset, then choose another part and repeat this process

# Metrics
**General requirements for ML Models:**
- Better than random guessing
- Better than majority guessing
## Confusion matrix
![[Pasted image 20240915165950.png|400]]

## Accuracy
$$\begin{array}{ccc}Acc = \frac{\text{number of properly classified examples}}{\text{number of all samples}}\\or\\Acc=\frac{\text{TN + TP}}{\text{TN + TP + FP + FN}}\end{array}$$
- Sometimes, accuracy isn't a good metric to use to measure performance of machine learning models, e.g. because class-imbalance, where: ![[Pasted image 20240915165845.png]]
## Precision
- All the positive predictions that we are made are actually true - it shows the quality of positive predictions
$$Precision=\frac{\text{TP}}{\text{TP + FP}}$$
## Recall
 - how many actual positives have you missed
$$Recall=\frac{\text{TP}}{\text{TP + FN}}$$
## F1-score
$$F1=2\times\frac{\text{Precision * Recall}}{\text{Precision + Recall}}$$
## ROC (Receiver Operator Characteristic) AUC (Are Under the Curve)


# Extra: Relationship b/w MLE and Log-Loss

[[Softmax Regression#Loss Function - Cross-entropy loss]]
![[Pasted image 20240915190716.png]]

---
# Reflection
- Why is predicting whether a software is malware or not-a-malware a classification problem?
- What do the words “logistic”, and “regression” represent in Logistic Regression?
- Why do we want to minimise the cross entropy loss with respect to the parameters of our logistic regression model?
- What is thresholding and why is it important?
- What is ROC curve and how can we use it for thresholding?