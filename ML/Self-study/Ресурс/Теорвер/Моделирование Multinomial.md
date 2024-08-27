```python
%matplotlib inline
import random
import torch
from torch.distributions.multinomial import Multinomial
from d2l import torch as d2l
```

- Моделирование ситуаций (100 объектов, вероятность выпадения 50/50)
```python
fair_probs = torch.tensor([0.5, 0.5])
Multinomial(100, fair_probs).sample()
```
> По мере роста числа повторений вероятности гарантированно сходятся с истинным базовым вероятностям - закон больших чисел
> **Центральная предельная теорема** - по мере увеличения размера выборки $n$, ошибка должна уменьшатся со скоростью $\frac{1}{\sqrt{n}}$ 
```python
counts = Multinomial(1, fair_probs).sample((10000,))
cum_counts = counts.cumsum(dim=0)
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)
estimates = estimates.numpy()

d2l.set_figsize((4.5, 3.5))
d2l.plt.plot(estimates[:, 0], label=("P(coin=heads)"))
d2l.plt.plot(estimates[:, 1], label=("P(coin=tails)"))
d2l.plt.axhline(y=0.5, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Samples')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend();
```

