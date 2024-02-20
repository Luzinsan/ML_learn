3 основных класса, состовляющие API для любых моделей:
1. `Module` - содержит модели, функции потерь и методы оптимизации - базовый класс всех моделей, от которого будем наследоваться
2. `DataModule` - загрузчики данных для обучения и валидации (проверки)
3. `Trainer` - класс, объединяющий два предыдущих, позволяет обучать модели на различных аппаратных платформах - обучает параметры модели (определяемой в экземпляре класса `Module`) на данных (указанных в экземпляре `DataModule`)

# Утилиты
- Декоратор `add_to_class`, позволяющий расширять уже определённые классы:
```python
def add_to_class(Class):  #@save
    """Register functions as methods in created class."""
    def wrapper(obj):
        setattr(Class, obj.__name__, obj)
    return wrapper
```
> Использование: добавление метода `do()` к классу `A` (при этом из метода доступны все атрибуты и методы класса `A`):
```python
@add_to_class(A)
def do(self):
    print('Class attribute "b" is', self.b)

a.do()
# Class attribute "b" is 1
```
- Служебный класс `HyperParameters`, который сохраняет все аргументы метода `__init__()` как атрибуты класса:
```python
class HyperParameters:  #@save
    """The base class of hyperparameters."""
    def save_hyperparameters(self, ignore=[]):
        raise NotImplemented
```
> Использование: наследование от `d2l.HyperParameters` и вызов метода `save_hyperparameters` (с указанием аргументов, которые не будут сохранены как атрибуты) позволяет неявно добавить атрибуты в класс:
```python
# Call the fully implemented HyperParameters class saved in d2l
class B(d2l.HyperParameters):
    def __init__(self, a, b, c):
        self.save_hyperparameters(ignore=['c'])
        print('self.a =', self.a, 'self.b =', self.b)
        print('There is no self.c =', not hasattr(self, 'c'))

b = B(a=1, b=2, c=3)
# self.a = 1 self.b = 2
# There is no self.c = True
```
- Интерактивное отображение хода эксперимента во время его проведения с помощью класса `ProgressBoard`. Метод `draw()` отображает точку `(x,y)` на рисунке с подписью `label`, указываемой в легенде. Параметр `every_n` сглаживает линию, показывая на рисунке $\frac{1}{n}$ точек => чем больше `n`, тем более кривая более ломаная:
```python
class ProgressBoard(d2l.HyperParameters):  #@save
    """The board that plots data points in animation."""
    def __init__(self, xlabel=None, ylabel=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 ls=['-', '--', '-.', ':'], colors=['C0', 'C1', 'C2', 'C3'],
                 fig=None, axes=None, figsize=(3.5, 2.5), display=True):
        self.save_hyperparameters()

    def draw(self, x, y, label, every_n=1):
        raise NotImplemented
```
> Использование: инициализация board и отрисовка функций
```python
board = d2l.ProgressBoard('x')
for x in np.arange(0, 10, 0.1):
    board.draw(x, np.sin(x), 'sin', every_n=2)
    board.draw(x, np.cos(x), 'cos', every_n=10)
```
---
# Module
```python
class Module(nn.Module, d2l.HyperParameters):  #@save
    """The base class of models."""
    def __init__(self, plot_train_per_epoch=2, plot_valid_per_epoch=1):
        super().__init__()
        self.save_hyperparameters()
        self.board = ProgressBoard()

    def loss(self, y_hat, y):
        raise NotImplementedError

    def forward(self, X):
        assert hasattr(self, 'net'), 'Neural network is defined'
        return self.net(X)

    def plot(self, key, value, train):
        """Plot a point in animation."""
        assert hasattr(self, 'trainer'), 'Trainer is not inited'
        self.board.xlabel = 'epoch'
        if train:
            x = self.trainer.train_batch_idx / \
                self.trainer.num_train_batches
            n = self.trainer.num_train_batches / \
                self.plot_train_per_epoch
        else:
            x = self.trainer.epoch + 1
            n = self.trainer.num_val_batches / \
                self.plot_valid_per_epoch
        self.board.draw(x, value.to(d2l.cpu()).detach().numpy(),
                        ('train_' if train else 'val_') + key,
                        every_n=int(n))

    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=True)
        return l

    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=False)

    def configure_optimizers(self):
        raise NotImplementedError
```
- `__init__` - конструктор, который сохраняет параметры обучения и инициализирует ProgressBoard
- `training_step()` - метод, принимающий minibatch, который отрисовывает значение loss функции в текущий момент $x_i$ и возвращает это значение.
- `configure_optimizers()` - метод, возвращающий метод оптимизации, или их список, который используется для обновления параметров обучения.
- `forward()` - метод для вычисления вывода (?)
	- В классе nn.Module есть магический метод `__call__()`, который вызывает метод `forward()`
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
## SGD
```python
class SGD(d2l.HyperParameters):  #@save
    """Minibatch stochastic gradient descent."""
    def __init__(self, params, lr):
        self.save_hyperparameters()

    def step(self):
        for param in self.params:
            param -= self.lr * param.grad

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()
```
- `step()` - [[Linear Regression#Minibatch Stochastic Gradient Descent]]
# DataModule
```python
class DataModule(d2l.HyperParameters):  #@save
    """The base class of data."""
    def __init__(self, root='../data', num_workers=4):
        self.save_hyperparameters()

    def get_dataloader(self, train):
        raise NotImplementedError

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)

	# Модификация 3.3.3
	def get_tensorloader(self, tensors, train, indices=slice(0, None)):
	    tensors = tuple(a[indices] for a in tensors)
	    dataset = torch.utils.data.TensorDataset(*tensors)
	    return torch.utils.data.DataLoader(dataset, self.batch_size,
	                                       shuffle=train)
```
- `__init__()` - конструктор, отвечающий за подговку данных: загрузку, предварительную обработку, если это необходимо
- `train_dataloader()` - возвращает загрузчик данных для обучающего набора данных - генератор, возвращающий minibatch на каждый его вызов (далее этот minibatch поступает в метод training_step класса Module)
- `val_dataloader()` - возвращает загрузчик набора валидационных (проверочных) данных.
> Data Loader - удобный способ абстрагировать процесс загрузки и управления данными. Их можно компоновать
> Т.о. Один и тот же алгоритм может обрабатывать множество различных типов и источников данных без необходимости модификации.
> Могут использоваться для описания всего пайплайна обработки данных. 
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
## Weight Decay
> Синтетический набор данных для иллюстрации weight decay
```python
class Data(d2l.DataModule):
    def __init__(self, num_train, num_val, num_inputs, batch_size):
        self.save_hyperparameters()
        n = num_train + num_val
        self.X = torch.randn(n, num_inputs)
        noise = torch.randn(n, 1) * 0.01
        w, b = torch.ones((num_inputs, 1)) * 0.01, 0.05
        self.y = torch.matmul(self.X, w) + b + noise

    def get_dataloader(self, train):
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader([self.X, self.y], train, i)
```

```python
def l2_penalty(w):
    return (w ** 2).sum() / 2
```
# Trainer
```python
class Trainer(d2l.HyperParameters):  #@save
    """The base class for training models with data."""
    def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
        self.save_hyperparameters()
        assert num_gpus == 0, 'No GPU support yet'

    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (len(self.val_dataloader)
                                if self.val_dataloader is not None else 0)

    def prepare_model(self, model):
        model.trainer = self
        model.board.xlim = [0, self.max_epochs]
        self.model = model

    def fit(self, model, data):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = model.configure_optimizers()
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        for self.epoch in range(self.max_epochs):
            self.fit_epoch()
	
	# 3.4.4 Модификация
    def fit_epoch(self):
	    self.model.train()
	    for batch in self.train_dataloader:
	        loss = self.model.training_step(self.prepare_batch(batch))
	        self.optim.zero_grad()
	        with torch.no_grad():
	            loss.backward()
	            if self.gradient_clip_val > 0:  # To be discussed later
	                self.clip_gradients(self.gradient_clip_val, self.model)
	            self.optim.step()
	        self.train_batch_idx += 1
	    if self.val_dataloader is None:
	        return
	    self.model.eval()
	    for batch in self.val_dataloader:
	        with torch.no_grad():
	            self.model.validation_step(self.prepare_batch(batch))
	        self.val_batch_idx += 1

	# 3.4.4 Модификация
	def prepare_batch(self, batch):
	    return batch
```
- `fit()` - метод, принимающий model (экз. класса `Module`) и data (экз. класса `DataModule`). Метод итерирует по всему набору данных столько раз, сколько указано в `max_epochs`. 
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