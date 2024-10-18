> - Often the membership of a cluster can replace the role of a label in the training dataset (for [[Supervised Learning]]) 
> - Clusters reveal a lot of information about the *underlying structure of the data*. 
> - In practice, we don't always have access to this additional label. Instead, one uses clustering algorithms to find natural clustering of the data.

**Broad definition of clustering**
- Clustering is an any partition of the dataset $\mathcal{D}$ into $K$ subsets $\mathcal{C}_1, ..., \mathcal{C}_K$ (with $K$ "given" or to predict), where: $$\cup^K_{k=1}\mathcal{C}_k = \mathcal{D}, \text{ and } \cap^K_{k=1}\mathcal{C}_k=\emptyset$$
- But not all partitions are a natural clustering of the dataset. The goal is to define **what a "good" clustering is**
> **A "good" cluster** is a subset of points which are closer to the *mean* of their own cluster than to the mean of other clusters.
> - Let $\mathcal{C}_k$ be one of the clusters for a dataset $\mathcal{D}$. Let $m_k = |\mathcal{C}_k|$ denote the cluster size. The *mean* of the cluster $\mathcal{C}_k$ is $$\mu_k=\frac{1}{m_k}\sum_{x^{(i)}\in \mathcal{C}_k}x^{(i)}$$ and the *variance* within the cluster $\mathcal{C}_k$ is $$\sigma^2_k = \frac{1}{m_k}\sum_{m_k}||x^{(i)} - \mu_k||^2_2$$

The most popular algorithm to solve this problem: [[k-means clustering]]

