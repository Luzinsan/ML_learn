> [[Многослойный Перцептрон]]

### Гиперпараметры
- Количество слоёв
- Количество нейронов в слое - ширина слоя - выбирается в виде значения, кратного степени двойки (эффективно в вычислительном смысле из-за способа выделения и адресации памяти)
# Реализация
## From scratch
```python
class MLPScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, num_hiddens, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens) * sigma)
        self.b1 = nn.Parameter(torch.zeros(num_hiddens))
        self.W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs) * sigma)
        self.b2 = nn.Parameter(torch.zeros(num_outputs))
```
- `nn.Parameter` - для автоматической инициализации атрибута класса в качестве параметра, который будет отслеживаться с помощью `autograd`

### Функция активации
```python
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)
```
### Forward
```python
@d2l.add_to_class(MLPScratch)
def forward(self, X):
    X = X.reshape((-1, self.num_inputs))
    H = relu(torch.matmul(X, self.W1) + self.b1)
    return torch.matmul(H, self.W2) + self.b2
```
### Обучение
```python
model = MLPScratch(num_inputs=784, num_outputs=10, num_hiddens=256, lr=0.1)
data = d2l.FashionMNIST(batch_size=256)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)
```
## With API
### Модель
```python
class MLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(nn.Flatten(), nn.LazyLinear(num_hiddens),
                                 nn.ReLU(), nn.LazyLinear(num_outputs))
```
- По сути то же самое, что и [[Softmax Regression]], но добавляется скрытый слой
- Последовательность действий (архитектура): входной слой схлопывается до 1D => полносвязный линейный скрытый слой (с нейронами = кол-во скрытых нейронов) => функция активации ReLU => полносвязный выходной слой (с нейронами = кол-во классов)
### Обучение
```python
model = MLP(num_outputs=10, num_hiddens=256, lr=0.1)
trainer.fit(model, data)
```