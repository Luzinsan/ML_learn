Needs to understand:
- What is machine learning?
- The available machine learning methods at our proposal
- Characteristics of those methods
- Circumstances under which a method would be most effective
- Their theoretical underpinnings

**Machine learning** 
- is computer programs (algorithms) that perform a task (solve a problem), which are trained through experience  extracted some insights/features/knowledge by the data and improve their performance.
- is a function estimation $\hat{f}$ where we assume a function can maps an observations ($x$) to our targets ($y$) (all from out data)
- Examples: object detention, instance segmentation, spam detection, disease prediction, whether forecasting, text summarizing, etc. 

$$D = \{(x_i, y_i)\}^N_{i=1}$$
**Data** - is what we refer to as *experience*
$x \in \mathcal{R}^d$ - observations (*features*) about the world
$y$ - output (*labels*) that we want to predict from the observation
$N$ - amount of examples that will be shown to an algorithm

**Performance** - needs to be defined according to the certain task and quantified by the metrics (objectives), that can tell our algorithm how good or bad in tends to performing our task

**Goal of learning**: learning of inferring a "functional" relationship between predictors ($x_i$) and target ($y_i$).
$$D = \{(x_i, y_i)\}^N_{i=1}$$where $x \in \mathcal{R}^d$, $y = f(x)$
**Assumption**: there is a function $f$ that can map $x$ to $y$
$\hat{f} \approx f$ - Goal of the learning

==Estimating== **$f$** 
**Assumption**: $f$ is finite function of certain number of parameters
We represent $\hat{f}$ (estimated function) like a set of parameters (recalls as weights): $y = f(x; w)$ 
The aim is: to estimate this parameters from the data

"*Supervised*" Learning Algorithm: $\hat{f}: X \to Y$
$$D = \{(x_i, y_i) \in X \times Y: 1\le i \le N\}$$
- where $Y$ - supervisions 
- That type of algorithms are trying to perform a task such as predicting a *labels* from the observations and comparing predictions with true label to estimate its performance

Based of the type of the label:
- **Classification** - is the type of supervised learning when we have *categorical* labels as a target
- **Regression** is a type of supervised learning when we have a *numerical* labels as a target

There are 2 phase:
- **Training** phase: $T_r = \{x_i, y_i\}^N_{i=1}$ is train data that we use during train time to *estimate* $\hat{f}(x)$
- **Assessing** (evaluation) phase: to assess the quality of estimate, we can compute (with learned model):
	- $MSE_{T_r} = Ave_{i \in T_r}[y_i - \hat{f}(x_i)]^2$ 
	- But this is not a reliable approach: 
		- because we always have just a peace of data that never cover general population
		- and assessing the train data doesn't show how model to predict on the other data/unseen set and maybe the model just memorize the data (with the noisy factors) rather than to *identify general patterns* from the data  
		- Data is inherently **noisy**
	- Use $T_e = \{x_i, y_i\}^M_{i=1}$
	  $MSE_{T_e} = Ave_{i \in T_e}[y_i - \hat{f}(x_i)]^2$ 

2 type of wrong results:
![[underfitting-overfitting plot.png|300]]![[Overfitting&Underfitting cases.png|340]]
	- **Underfitting**: $y=\frac{w_0+w_1 x}{f}$ - happens when we have quite *simple model* - train and test error will be large
	![[Underfitting.png|200]]
	- **Overfitting**: $y=\frac{w_0 + w_1 x + w_2 x^2 + w_3 x^8}{f}$ - happens when the model (often complex, but not always) is performing well on the train data, but is performing bad on the test data, moreover the differences of errors  become greater with time to getting more *complex model* (huge gap between train and test error)
	  ![[Overfitting.png|200]]
		- Most complex model
		- Smallest training error
		- But *largest test error*
	- We are making a compromise between underfitting and overfitting. In theory of machine learning this decision known as **Bias Variance Tradeoff**
	- Note: if the problem is simple even weak model can be appropriate and complex model can overfit on this data. Overfitting and underfitting don't *just* depend upon the complexity of the **model** and *also* depend upon of the complexity the **data**

**Bias Variance Tradeoff**
> https://medium.com/snu-ai/the-bias-variance-trade-off-a-mathematical-view-14ff9dfe5a3c

$E(y_0 - \hat{f}(x_0))^2 = Var(\hat{f}(x_0)) + [Bias(\hat{f}(x_0))]^2 + Var(\varepsilon)$
- It's about a making a tradeoff (compromise) between a simple model (underfitting) and complex model (overfitting)
- *Simple model* is calls as biased model. Underfitting is calls as **bias** ($Bias(\hat{f})$). It means no matter how much data we show our model, it always will be biased, so we need to increase the complex of model.
- *Complex model* has a large **variance** ($Var(\hat{f})$) with its predictions. Variance represents the variation that we can expect from the model if we train that on different dataset about the same problem. If we wand to reach a tradeoff, we need to reduce this variance, that memorize unimportant noisy.
> Typically, as the flexibility or complexity of !𝑓 increases, its variance increases, and its bias decreases. So choosing the flexibility based on average test error amounts to a bias-variance trade-off.
- Randomness of noise $Var(\varepsilon)$ (**irreducible error**) is smt that is beyond over control.
- **Summary**:
	- Underfitting: bias is very large and variance very small
	- Overfitting: bias is very small but variance very large

---
- What exactly does it mean for machine to learn something?
- Why would you use machine learning in an application such spam filtering?
- What problems are supervised learning problems?
- Why is estimating the price of a house, given its features, a regression problem but spam filtering a classification problem?
- How is supervised machine learning from data similar to function estimation?
- What are underfitting and overfitting, and how can we detect them?
- Will a complex model always overfit and a simple model always underfit?