> [[Linear Regression]]
## SyntheticRegressionData
```python
class SyntheticRegressionData(d2l.DataModule):  #@save
    """Synthetic data for linear regression."""
    def __init__(self, w, b, noise=0.01, num_train=1000, num_val=1000,
                 batch_size=32):
        super().__init__()
        self.save_hyperparameters()
        n = num_train + num_val
        self.X = torch.randn(n, len(w))
        noise = torch.randn(n, 1) * noise
        self.y = torch.matmul(self.X, w.reshape((-1, 1))) + b + noise
        
    # Модификация 3.3.3
    def get_dataloader(self, train):
	    i = slice(0, self.num_train) if train else slice(self.num_train, None)
	    return self.get_tensorloader((self.X, self.y), train, i)
```
> Использование:
```python
data = SyntheticRegressionData(w=torch.tensor([2, -3.4]), b=4.2)
X, y = next(iter(data.train_dataloader()))
print('X shape:', X.shape, '\ny shape:', y.shape)
# torch.utils.data.DataLoader предоставляет встроенные методы. Например:
len(data.train_dataloader()) # ==> 32 - кол-во minibatch'ей
```
## LinearRegressionScratch
```python
class LinearRegressionScratch(d2l.Module):  #@save
    """The linear regression model implemented from scratch."""
    def __init__(self, num_inputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.w = torch.normal(0, sigma, (num_inputs, 1), requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)
    
    def forward(self, X):
	    return torch.matmul(X, self.w) + self.b

	def loss(self, y_hat, y):
	    l = (y_hat - y) ** 2 / 2
	    return l.mean()
	
	def configure_optimizers(self):
	    return SGD([self.w, self.b], self.lr)
```
- `forward()` - [[Linear Regression#Model]]
- `loss()` - [[Linear Regression#Функция потерь - Loss Function]]
### WeightDecayScratch
```python
class WeightDecayScratch(d2l.LinearRegressionScratch):
    def __init__(self, num_inputs, lambd, lr, sigma=0.01):
        super().__init__(num_inputs, lr, sigma)
        self.save_hyperparameters()

    def loss(self, y_hat, y):
        return (super().loss(y_hat, y) +
                self.lambd * l2_penalty(self.w))
```


# Обучение
## Linear Regression
1. Инициализируются параметры: $(\mathbf{w},b)$
2. Цикл до точки остановы:
	1. Вычисляется градиент: $\mathbf{g} \leftarrow \partial_{(\mathbf{w},b)}\frac{1}{|\mathcal{B}|}\sum_{i \in \mathcal{B}}l(\mathbf{x}^{(i)},y^{(i)}, \mathbf{w}, b)$ 
	2. Обновляются параметры: $(\mathbf{w}, b) \leftarrow (\mathbf{w},b) - \eta \mathbf{g}$

```python
model = LinearRegressionScratch(2, lr=0.03)
data = d2l.SyntheticRegressionData(w=torch.tensor([2, -3.4]), b=4.2)
trainer = d2l.Trainer(max_epochs=3)
trainer.fit(model, data)
```

### Проверка весов
```python
with torch.no_grad():
    print(f'error in estimating w: {data.w - model.w.reshape(data.w.shape)}')
    print(f'error in estimating b: {data.b - model.b}')
```
## WeightDecay for Linear Regression
```python
data = Data(num_train=20, num_val=100, num_inputs=200, batch_size=5)
trainer = d2l.Trainer(max_epochs=10)

def train_scratch(lambd):
    model = WeightDecayScratch(num_inputs=200, lambd=lambd, lr=0.01)
    model.board.yscale='log'
    trainer.fit(model, data)
    print('L2 norm of w:', float(l2_penalty(model.w)))
```