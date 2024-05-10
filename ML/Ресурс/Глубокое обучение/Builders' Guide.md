```python
net = nn.Sequential(nn.LazyLinear(8),
                    nn.ReLU(),
                    nn.LazyLinear(1))
```
- Можем получить доступ к параметрам конкретного слоя:
```python
net[2].state_dict()
# OrderedDict([('weight',
#               tensor([[-0.1649,  0.0605,  0.1694, -0.2524,  0.3526, -0.3414, -0.2322,  0.0822]])),
#              ('bias', tensor([0.0709]))])
```
- Каждый параметр - это экземпляр класса Parameter
```python
type(net[2].bias), net[2].bias.data
# (torch.nn.parameter.Parameter, tensor([0.0709]))
```
- Вывод всех параметров всех слоёв:
```python
[(name, param.shape) for name, param in net.named_parameters()]
#[('0.weight', torch.Size([8, 4])),
# ('0.bias', torch.Size([8])),
# ('2.weight', torch.Size([1, 8])),
# ('2.bias', torch.Size([1]))]
```
###  Связанные параметры
- Если возникает необходимость иметь несколько слоёв с одинаковыми параметрами (весами, для этого их выделяют в отдельных экземпляр и далее ссылаются несколько раз. (Если изменится один тензор, следом изменится и другой) Пример:
```python
# We need to give the shared layer a name so that we can refer to its
# parameters
shared = nn.LazyLinear(8)
net = nn.Sequential(nn.LazyLinear(8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.LazyLinear(1))

net(X)
# Check whether the parameters are the same
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# Make sure that they are actually the same object rather than just having the
# same value
print(net[2].weight.data[0] == net[4].weight.data[0])
# tensor([True, True, True, True, True, True, True, True])
# tensor([True, True, True, True, True, True, True, True])
```
# Инициализация параметров
> [[Numerical Stability#Инициализация параметров]]

![[Numerical Stability#Инициализация по-умолчанию]]
```python
def init_normal(module):
    if type(module) == nn.Linear:
        nn.init.normal_(module.weight, mean=0, std=0.01)
        nn.init.zeros_(module.bias)

net.apply(init_normal)
net[0].weight.data[0], net[0].bias.data[0]
# (tensor([-0.0129, -0.0007, -0.0033,  0.0276]), tensor(0.))
```
- метод `apply()` возволяет применить функцию инициализации (`init_normal` - инициализация весов ли слоя значениями из гауссова распределения с $E=0$ и $\sigma=0.01$)
- Использование `nn.init.zeros_` и `nn.init.constant_` позволяет инициализировать скалярным значением
```python
def init_xavier(module):
    if type(module) == nn.Linear:
        nn.init.xavier_uniform_(module.weight)

def init_42(module):
    if type(module) == nn.Linear:
        nn.init.constant_(module.weight, 42)

net[0].apply(init_xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data)
```
- на каждый слой можно применить разную инициализацию
- `nn.init.xavier_uniform_` для [[Numerical Stability#Xavier init - Инициализация Ксавье|Инициализации Ксавье]]
## Пользовательская инициализация
```python
def my_init(module):
    if type(module) == nn.Linear:
        print("Init", *[(name, param.shape)
                        for name, param in module.named_parameters()][0])
        nn.init.uniform_(module.weight, -10, 10)
        module.weight.data *= module.weight.data.abs() >= 5

net.apply(my_init)
net[0].weight[:2]
```
- Здесь инициализация соответствует правилу: $$\begin{aligned}
    w \sim \begin{cases}
        U(5, 10) & \textrm{ with probability } \frac{1}{4} \\
            0    & \textrm{ with probability } \frac{1}{2} \\
        U(-10, -5) & \textrm{ with probability } \frac{1}{4}
    \end{cases}
\end{aligned}$$
> Метод для пробного прогона входных данных и инициализации параметров заданным методом:
```python
@d2l.add_to_class(d2l.Module)  #@save
def apply_init(self, inputs, init=None):
    self.forward(*inputs)
    if init is not None:
        self.net.apply(init)
```
# Кастомные слои
> Рано или поздно вам понадобится слой, которого еще нет в структуре глубокого обучения. В этих случаях необходимо создать собственный слой
> 
```python
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))

    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)
```
# File I/O (функции save/load)
- Для записи в файл тензора:
```python
x = torch.arange(4)
torch.save(x, 'x-file')
```
- Для чтения с файла:
```python
x2 = torch.load('x-file')
```
- Запись массива и словаря тензоров:
```python
y = torch.zeros(4)
torch.save([x, y],'x-files')
x2, y2 = torch.load('x-files')
(x2, y2)

mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
mydict2
```
## Сохранение весов модели
```python
net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)
# Y - для последующей сверки весов
```
- Сохранение всех весов (но не архитектуры): `torch.save(net.state_dict(), 'mlp.params')`
- Загружаем сохранённые веса в clone:
```python
clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
clone.eval()
```
- Сверяемся `Y==clone(X)`:
```python
Y_clone = clone(X)
Y_clone == Y
```