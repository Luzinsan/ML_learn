# Supervised vs. Unsupervised learning
> Mathematically. Let: $$\mathcal{D} \subset \mathbb{R}^n$$ where $\mathcal{D}$ is an open bounded set of dimension $n$.
> Further, let $$\mathcal{D}' \subset \mathcal{D}$$

- The goal of **classification** is to build a *classifier* labeling all data in $\mathcal{D}$ given data from $\mathcal{D}'$.  
	- More precisely: consider a set of data points $x^{(i)} \in \mathbb{R}^n$ and labels $y^{(i)}$ for each point where $j=1,...,o.$, where $o$ is the number of points in $\mathcal{D}$
	- For the training, wee have $m \le 0$  points/samples defining $\mathcal{D}'$.

- **Supervised learning** provides labels for the training stage:
	- **Input**: data set $\{x^{(i)}\}^o_{i=1}$ and labels $\color{red}{\{y^(i)\}^m_{i=1}}$
	- **Output**: labels $\{y^{(i)}\}^o_{i=1}$ with $y^{(i)} \in \{-1,+1\}, \forall i$ 
		- all the labels are (usually) known in order to build the classifier on $\mathcal{D}'$
		- The classifier is then applied to $\mathcal{D}$.
- For **unsupervised learning**:
	- **Input**: data set $\{x^{(i)}\}^o_{i=1}$ with $x^{i} \in \mathbb{R}^n, \forall i$
	- **Output**: labels $\{y^{(i)}\}^o_{i=1}$ with $y^{(i)} \in \{-1,+1\}, \forall i$
		- The mathematical framing of unsupervised learning is focused on producing labels $y^{(i)} \in \{-1,+1\}, \forall$ the data.
		- Generally, the data $x^{(i)}$ used for training the classifier is from $\mathcal{D}'$.
		- The classifier is then more broadly applied, i.e. it generalizes, to the open bounded domain $\mathcal{D}$
> In both cases: if the data used to build a classifier only samples a small portion of the larger domain, then it is often the case that the classifier will not generalize well.

[[Supervised Learning]]
[[Unsupervised Learning]]
